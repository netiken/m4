#!/usr/bin/env python3
"""
Generate ns-3 style outputs from UNISON sweep results.

For each sweep directory this script:
  1. Normalises client/server traces from stdout.txt into grouped_flows.txt
  2. Computes UD and RDMA durations from the grouped flows and writes ns3_output.txt

Usage:
    python process.py [SWEEPS_DIR]

If SWEEPS_DIR is omitted we look for sweeps_4/ then sweeps/ relative to this script.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

EVENT_RE = re.compile(
    r"\[(\w+)\s+([\w_]+)\]\s+t=(\d+)\s+ns\s+reqId=(\d+).*client_node_id=(\d+)"
)


def process_stream_into_groups(stream: Iterable[str], groups: "OrderedDict[str, List[str]]") -> None:
    """Group lines from stdout.txt by reqId for easier post-processing."""
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        match = re.search(r"\breqId=(\d+)", line)
        if not match:
            continue
        reqid = match.group(1)
        groups.setdefault(reqid, []).append(line)


def write_groups(groups: "OrderedDict[str, List[str]]", out_path: pathlib.Path) -> None:
    """Write grouped flow traces to grouped_flows.txt."""
    # NOTE: We do NOT scale timestamps here because application completion time
    # depends on concurrency/parallelism, not just sum of individual flows.
    # Only per-flow RDMA durations are scaled in ns3_output.txt.
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for reqid, lines in groups.items():
            handle.write(f"### reqId={reqid} ###\n")
            for line in lines:
                handle.write(line + "\n")
            handle.write("\n")


def parse_grouped(path: pathlib.Path) -> Dict[int, List[dict]]:
    """Parse grouped_flows.txt into event dictionaries keyed by reqId."""
    flows: Dict[int, List[dict]] = collections.defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            match = EVENT_RE.search(raw)
            if not match:
                continue
            side, event, ts, reqid, client = match.groups()
            flows[int(reqid)].append(
                {
                    "side": side,
                    "event": event,
                    "t": int(ts),
                    "client": int(client),
                    "raw": raw.rstrip("\n"),
                }
            )
    return flows


def extract_sequences(events: Sequence[dict], pattern: Sequence[str]) -> List[List[dict]]:
    """Find non-overlapping subsequences that match the provided event order."""
    sequences: List[List[dict]] = []
    n = len(events)
    i = 0
    while i < n:
        matched: List[tuple[int, dict]] = []
        pos = i
        for want in pattern:
            j = pos
            while j < n and events[j]["event"] != want:
                j += 1
            if j == n:
                matched = []
                break
            matched.append((j, events[j]))
            pos = j + 1
        if not matched:
            break
        sequences.append([event for _, event in matched])
        i = matched[-1][0] + 1
    return sequences


def compute_sequences(flows: Dict[int, List[dict]], scenario_dir: Path) -> List[dict]:
    """Compute UD/RDMA durations for each grouped request."""
    outputs: List[dict] = []
    dup_counter = collections.defaultdict(int)
    
    # ⚠️ CRITICAL FIX: DO NOT subtract server overhead!
    # Real testbed logs INCLUDE server processing time in UD duration (~460ms)
    # Simulators must report FULL UD time (network + server) for fair comparison
    SERVER_OVERHEAD_NS = 0  # Don't subtract - report full simulated time
    
    # NO RDMA scaling - report NS3's raw simulation timing
    # (Previous 0.5x scaling was inconsistent across scenarios)
    
    sorted_items = sorted(flows.items(), key=lambda kv: min(event["t"] for event in kv[1]))
    for reqid, events in sorted_items:
        events_sorted = sorted(events, key=lambda event: event["t"])

        client = None
        for event in events_sorted:
            if event["event"] == "req_send":
                client = event["client"] - 1
                break
        if client is None and events_sorted:
            client = events_sorted[0]["client"] - 1
        if client is None:
            client = 0

        ud_sequences = extract_sequences(events_sorted, ["req_send", "req_recv", "resp_send", "resp_recv"])
        # ⚠️ CRITICAL FIX: RDMA duration must match FlowSim's calculation!
        # FlowSim: rdma_dur = rdma_recv - handshake_send (includes handshake transit)
        # NS3 must do the same for fair comparison
        rdma_sequences = extract_sequences(events_sorted, ["hand_send", "rdma_recv"])
        
        for seq in ud_sequences:
            req_send, req_recv, resp_send, resp_recv = (entry["t"] for entry in seq)
            # Subtract scaled server overhead to get network-only time
            ud_duration = (resp_recv - req_send) - SERVER_OVERHEAD_NS
            suffix = "" if dup_counter[reqid] == 0 else f"-{dup_counter[reqid]}"
            dup_counter[reqid] += 1
            outputs.append(
                {
                    "type": "ud",
                    "client": client,
                    "id": f"{reqid}{suffix}",
                    "dur": ud_duration,
                    "ts": seq[0]["t"],
                }
            )

        for seq in rdma_sequences:
            hand_send, rdma_recv = (entry["t"] for entry in seq)
            # ⚠️ Match FlowSim: RDMA duration = rdma_recv - handshake_send
            rdma_duration = rdma_recv - hand_send
            suffix = "" if dup_counter[reqid] == 0 else f"-{dup_counter[reqid]}"
            dup_counter[reqid] += 1
            outputs.append(
                {
                    "type": "rdma",
                    "client": client,
                    "id": f"{reqid}{suffix}",
                    "dur": rdma_duration,
                    "ts": seq[0]["t"],
                }
            )

    outputs.sort(key=lambda entry: entry["ts"])
    return outputs


def write_ns3_output(outputs: List[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for entry in outputs:
            handle.write(
                f"[{entry['type']}] client={entry['client']} id={entry['id']} dur_ns={entry['dur']}\n"
            )


def detect_default_sweeps_dir(base_dir: pathlib.Path) -> pathlib.Path:
    candidates = [base_dir / "sweeps_12", base_dir / "sweeps"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def process_directory(sweeps_dir: pathlib.Path) -> None:
    if not sweeps_dir.exists() or not sweeps_dir.is_dir():
        print(f"[error] sweeps path not found: {sweeps_dir}", file=sys.stderr)
        sys.exit(1)

    grouped_written = 0
    ns3_written = 0

    for subdir in sorted(sweeps_dir.iterdir()):
        if not subdir.is_dir():
            continue

        stdout_file = subdir / "stdout.txt"
        if not stdout_file.exists():
            continue

        grouped_path = subdir / "grouped_flows.txt"
        groups: "OrderedDict[str, List[str]]" = OrderedDict()
        try:
            with stdout_file.open("r", encoding="utf-8") as stream:
                process_stream_into_groups(stream, groups)
        except Exception as exc:
            print(f"[warn] failed to read {stdout_file}: {exc}", file=sys.stderr)
            continue

        write_groups(groups, grouped_path)
        grouped_written += 1

        try:
            flows = parse_grouped(grouped_path)
            outputs = compute_sequences(flows, subdir)
            write_ns3_output(outputs, subdir / "ns3_output.txt")
            ns3_written += 1
        except Exception as exc:
            print(f"[warn] failed to compute ns3 output for {grouped_path}: {exc}", file=sys.stderr)
            continue

    print(f"[done] Wrote grouped_flows.txt to {grouped_written} subdirectories under {sweeps_dir}")
    print(f"[done] Wrote ns3_output.txt to {ns3_written} subdirectories under {sweeps_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grouped_flows.txt and ns3_output.txt for UNISON sweeps.")
    parser.add_argument(
        "sweeps_dir",
        nargs="?",
        help="Path to sweeps directory (default: auto-detect sweeps_4/ or sweeps/ next to this script).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent

    if args.sweeps_dir:
        sweeps_dir = pathlib.Path(args.sweeps_dir).expanduser().resolve()
    else:
        sweeps_dir = detect_default_sweeps_dir(script_dir)

    process_directory(sweeps_dir)


if __name__ == "__main__":
    main()
