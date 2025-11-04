#!/usr/bin/env python3
"""
Compute delays between req_send and resp_recv_ud events for matching flow_id values
from one or more log files. Outputs CSV rows with flow_id, request/response
timestamps, and delay.

Usage examples:
  python3 compute_delays.py --input flow_id/c1/log_0.txt --head 10
  python3 compute_delays.py --input flow_id/c1/log_0.txt flow_id_1000/c1/log_0.txt --units ms --output delays.csv
"""

import argparse
import csv
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple


def parse_kv_tokens(line: str) -> Dict[str, str]:
    """Parse space-separated key=value tokens from a log line.

    Returns a dict of tokens. Tokens without '=' are ignored.
    """
    tokens: Dict[str, str] = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        tokens[key] = value
    return tokens


def iter_delays_from_file(
    file_path: Path,
    stop_after: Optional[int] = None,
) -> Iterable[Tuple[int, int, int]]:
    """Yield (flow_id, req_ts_ns, resp_ts_ns) tuples from a single log file.

    stop_after: if provided, stop yielding after this many pairs are produced.
    """
    # For each flow_id, maintain a queue of outstanding req_send timestamps
    outstanding_req_by_flow: Dict[int, Deque[int]] = defaultdict(deque)
    produced = 0

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "event=req_send" in line:
                tokens = parse_kv_tokens(line)
                try:
                    flow_id = int(tokens.get("flow_id", ""))
                    ts_ns = int(tokens.get("ts_ns", ""))
                except ValueError:
                    continue
                outstanding_req_by_flow[flow_id].append(ts_ns)
            elif "event=resp_recv_ud" in line:
                tokens = parse_kv_tokens(line)
                try:
                    flow_id = int(tokens.get("flow_id", ""))
                    resp_ts_ns = int(tokens.get("ts_ns", ""))
                except ValueError:
                    continue
                req_queue = outstanding_req_by_flow.get(flow_id)
                if req_queue and len(req_queue) > 0:
                    req_ts_ns = req_queue.popleft()
                    yield (flow_id, req_ts_ns, resp_ts_ns)
                    produced += 1
                    if stop_after is not None and produced >= stop_after:
                        return
                # If no matching request is found yet, skip. Later lines may include the request.


def unit_divisor(units: str) -> float:
    if units == "ns":
        return 1.0
    if units == "us":
        return 1_000.0
    if units == "ms":
        return 1_000_000.0
    if units == "s":
        return 1_000_000_000.0
    raise ValueError(f"Unsupported units: {units}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute req->resp UD delays per flow_id from logs")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Path(s) to input log file(s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Optional output CSV file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--units",
        choices=["ns", "us", "ms", "s"],
        default="ns",
        help="Units for delay column (default: ns)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="If >0, stop after emitting this many rows across all inputs",
    )

    args = parser.parse_args(argv)

    paths: List[Path] = [Path(p) for p in args.input]
    for p in paths:
        if not p.exists():
            print(f"Input not found: {p}", file=sys.stderr)
            return 2

    total_to_emit: Optional[int] = args.head if args.head and args.head > 0 else None
    remaining = total_to_emit

    out_fp = None
    writer: Optional[csv.writer] = None
    try:
        if args.output:
            out_fp = open(args.output, "w", newline="", encoding="utf-8")
            writer = csv.writer(out_fp)
        else:
            writer = csv.writer(sys.stdout)

        # Header
        delay_col_name = f"delay_{args.units}"
        writer.writerow(["flow_id", "req_ts_ns", "resp_ts_ns", delay_col_name])

        div = unit_divisor(args.units)

        for path in paths:
            to_take_from_this_file: Optional[int] = None
            if remaining is not None:
                to_take_from_this_file = remaining

            for flow_id, req_ts_ns, resp_ts_ns in iter_delays_from_file(path, stop_after=to_take_from_this_file):
                delay_ns = resp_ts_ns - req_ts_ns
                delay_val = delay_ns / div
                writer.writerow([flow_id, req_ts_ns, resp_ts_ns, f"{delay_val:.6f}"])
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return 0

        return 0
    finally:
        if out_fp is not None:
            out_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())


