#!/usr/bin/env python3
"""
Convert raw M4 sweep logs into grouped flows and per-request duration summaries.

For every sweep subdirectory this script:
  1. Reads client_*.log traces and writes grouped_flows.txt in ns-3 style format.
  2. Parses grouped flows to compute UD and RDMA durations, producing m4_outputv2.txt.

Usage:
    python process.py [SWEEPS_DIR]

If SWEEPS_DIR is omitted the script searches for sweeps_4/, then sweeps_12/, then sweeps/.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
from typing import Dict, Iterable, List, Sequence, Tuple


CLIENT_LOG_PATTERN = "client_*.log"
NUMERIC_FIELDS = {"ts_ns", "id", "clt", "wrkr", "slot", "size", "start_ns", "dur_ns", "wire_bytes"}


def parse_line(line: str) -> Dict[str, int]:
    """Convert a log line into a dictionary of fields."""
    parts = line.strip().split()
    data: Dict[str, int] = {}
    if not parts:
        return data

    event_token = parts[0]
    if "=" in event_token:
        _, event_value = event_token.split("=", 1)
        data["event"] = event_value

    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in NUMERIC_FIELDS:
            try:
                data[key] = int(value)
            except ValueError:
                continue
        else:
            data[key] = value
    return data


def format_event_ns3(event: Dict[str, int]) -> str:
    """Render an event dictionary using the ns-3 grouped trace syntax."""
    role_map = {
        "req_send": "client req_send",
        "req_recv": "server req_recv",
        "resp_send": "server resp_send",
        "resp_recv_ud": "client resp_recv",
        "hand_send": "client hand_send",
        "hand_recv": "server hand_recv",
        "resp_rdma_read": "client rdma_recv",
        "resp_rdma_send": "server rdma_send",
        "rdma_send": "server rdma_send",
        "rdma_recv": "client rdma_recv",
    }
    role_event = role_map.get(event.get("event", ""), event.get("event", "event"))
    size = event.get("size", event.get("wire_bytes", 0))
    return (
        f"[{role_event}] t={event.get('ts_ns', 0)} ns reqId={event.get('id', 0)} "
        f"size={size}B client_node_id={event.get('clt', 0)}"
    )


def collect_events(subdir: pathlib.Path) -> Dict[Tuple[int, int], List[Dict[str, int]]]:
    """Gather all client log events grouped by (client, request id)."""
    flows: Dict[Tuple[int, int], List[Dict[str, int]]] = collections.defaultdict(list)
    for log_path in sorted(subdir.glob(CLIENT_LOG_PATTERN)):
        with log_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                event = parse_line(raw)
                if not event or "clt" not in event or "id" not in event:
                    continue
                key = (event["clt"], event["id"])
                flows[key].append(event)
    return flows


def write_grouped_flows(flows: Dict[Tuple[int, int], List[Dict[str, int]]], out_path: pathlib.Path) -> None:
    """Write grouped flows in ns-3 style format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for (_, req_id), events in sorted(
            flows.items(), key=lambda entry: min(event.get("ts_ns", 0) for event in entry[1])
        ):
            handle.write(f"### reqId={req_id} ###\n")
            for event in sorted(events, key=lambda e: e["ts_ns"]):
                handle.write(format_event_ns3(event) + "\n")
            handle.write("\n")


def parse_grouped(path: pathlib.Path) -> Dict[int, List[dict]]:
    """Parse grouped_flows.txt into per-request event lists."""
    flows: Dict[int, List[dict]] = collections.defaultdict(list)
    current_req: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("### reqId="):
                try:
                    current_req = int(line.split("=", 1)[1].split()[0])
                except ValueError:
                    current_req = None
                continue
            if current_req is None:
                continue
            match = EVENT_RE.match(line)
            if not match:
                continue
            side, event_name, ts, req_id, client = match.groups()
            flows[current_req].append(
                {
                    "side": side,
                    "event": event_name,
                    "t": int(ts),
                    "client": int(client),
                    "raw": line,
                }
            )
    return flows


EVENT_RE = re.compile(r"\[(\w+)\s+([\w_]+)\]\s+t=(\d+)\s+ns\s+reqId=(\d+).*client_node_id=(\d+)")


def compute_durations(flows: Dict[int, List[dict]]) -> List[dict]:
    """Compute UD (hand_send - req_send) and RDMA (rdma_recv - hand_send) durations."""
    outputs: List[dict] = []
    sorted_items = sorted(flows.items(), key=lambda kv: min(event["t"] for event in kv[1]))

    for req_id, events in sorted_items:
        events_sorted = sorted(events, key=lambda event: event["t"])
        client = events_sorted[0]["client"] if events_sorted else 0

        req_send = next((e["t"] for e in events_sorted if e["event"] == "req_send"), None)
        hand_send = next((e["t"] for e in events_sorted if e["event"] == "hand_send"), None)
        rdma_recv = next((e["t"] for e in events_sorted if e["event"] == "rdma_recv"), None)

        if req_send is not None and hand_send is not None:
            outputs.append(
                {
                    "type": "ud",
                    "client": client,
                    "id": req_id,
                    "dur": hand_send - req_send,
                    "ts": req_send,
                }
            )

        if hand_send is not None and rdma_recv is not None:
            outputs.append(
                {
                    "type": "rdma",
                    "client": client,
                    "id": req_id,
                    "dur": rdma_recv - hand_send,
                    "ts": hand_send,
                }
            )

    outputs.sort(key=lambda entry: entry["ts"])
    return outputs


def write_m4_output(outputs: Sequence[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for entry in outputs:
            handle.write(
                f"[{entry['type']}] client={entry['client']} id={entry['id']} dur_ns={entry['dur']}\n"
            )


def detect_default_sweeps_dir(base: pathlib.Path) -> pathlib.Path:
    candidates = [base / "sweeps_4", base / "sweeps_12", base / "sweeps"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def process_directory(sweeps_dir: pathlib.Path) -> None:
    if not sweeps_dir.exists() or not sweeps_dir.is_dir():
        raise SystemExit(f"[error] sweeps path not found: {sweeps_dir}")

    grouped_count = 0
    output_count = 0

    for subdir in sorted(sweeps_dir.iterdir()):
        if not subdir.is_dir():
            continue

        flows = collect_events(subdir)
        if not flows:
            continue

        grouped_path = subdir / "grouped_flows.txt"
        write_grouped_flows(flows, grouped_path)
        grouped_count += 1

        try:
            parsed = parse_grouped(grouped_path)
            durations = compute_durations(parsed)
            if durations:
                write_m4_output(durations, subdir / "m4_outputv2.txt")
                output_count += 1
        except Exception as exc:
            print(f"[warn] failed to compute durations for {grouped_path}: {exc}")
            continue

    print(f"[done] Wrote grouped_flows.txt in {grouped_count} subdirectories under {sweeps_dir}")
    print(f"[done] Wrote m4_outputv2.txt in {output_count} subdirectories under {sweeps_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped flows and M4 duration summaries for sweeps."
    )
    parser.add_argument(
        "sweeps_dir",
        nargs="?",
        help="Path to sweeps directory (default: auto-detect sweeps_4/, sweeps_12/, or sweeps/ next to this script).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent

    sweeps_dir = (
        pathlib.Path(args.sweeps_dir).expanduser().resolve()
        if args.sweeps_dir
        else detect_default_sweeps_dir(script_dir)
    )

    process_directory(sweeps_dir)


if __name__ == "__main__":
    main()
