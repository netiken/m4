#!/usr/bin/env python3
"""
New matching logic (sequential per worker, ignoring slot):
1) Group client events by worker id for req_send and resp_recv_ud; group worker events by worker id for req_recv and resp_send.
2) For each worker, sort each list by ts_ns and pair sequentially by index:
   - client->worker: pair client req_send[i] with worker req_recv[i]
   - worker->client: pair worker resp_send[i] with client resp_recv_ud[i]
3) Write all raw paired lines to debug files: debug/client-worker_<wrkr>.debug
4) Optionally skip K pairs per worker before computing diffs via --skip_first K
5) Plot overlaid CDFs of both series.

Usage:
  python3 plot_latencies.py \
      --client c1/log_0.normalized.txt \
      --workers_glob server/worker_log_*.normalized.txt \
      --xunit ms \
      --out latency_cdfs.png \
      --csv latency_ecdfs.csv \
      --skip_first 100

Defaults assume running from the 16_static directory.
"""

import argparse
import glob
import math
import os
import re
import sys
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CDFs of request/response latencies across client/worker logs")
    parser.add_argument("--client", default="c1/log_0.normalized.txt", help="Path to normalized client log")
    parser.add_argument(
        "--workers_glob",
        default="server/worker_log_*.normalized.txt",
        help="Glob for normalized worker logs",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="If set, run over a scenario directory containing c*/ and server/ subdirs",
    )
    parser.add_argument("--xunit", choices=["ns", "us", "ms", "s"], default="us", help="X-axis unit")
    parser.add_argument("--out", default="latency_cdfs.png", help="Output PNG path")
    parser.add_argument(
        "--from_debug_pairs",
        action="store_true",
        help="If set, read per-worker pairs from debug_pairs/ and plot a CDF per file",
    )
    parser.add_argument(
        "--pairs_glob",
        default="debug_pairs/client-worker_*.debug_pairs",
        help="Glob for debug_pairs input when --from_debug_pairs is used",
    )
    parser.add_argument(
        "--do_pairs_plots",
        action="store_true",
        help="When processing clients (single or scenario), also generate per-worker CDFs from debug_pairs",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV to write ECDF samples: columns (series,x,ecdf). Series in {c2w,w2c}",
    )
    parser.add_argument(
        "--skip_first",
        type=int,
        default=20,
        help="Number of earliest pairs to skip (warm-up) for each series, ordered by pair timestamp",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid on the plot",
    )
    return parser.parse_args()


def parse_kv_line(line: str) -> Dict[str, str]:
    parts = line.strip().split()
    out: Dict[str, str] = {}
    for token in parts:
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def load_client_events_sequential(client_path: str) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, List[Tuple[int, str]]]]:
    """Return per-worker ordered lists of (ts_ns, raw_line) for client events."""
    req_send: Dict[int, List[Tuple[int, str]]] = {}
    resp_recv: Dict[int, List[Tuple[int, str]]] = {}
    with open(client_path, "r") as f:
        for line in f:
            if not line.startswith("event="):
                continue
            kv = parse_kv_line(line)
            ev = kv.get("event")
            if ev not in {"req_send", "resp_recv_ud"}:
                continue
            ts_ns = int(kv["ts_ns"]) if "ts_ns" in kv else None
            wrkr = int(kv["wrkr"]) if "wrkr" in kv else None
            if ts_ns is None or wrkr is None:
                continue
            if ev == "req_send":
                req_send.setdefault(wrkr, []).append((ts_ns, line.rstrip("\n")))
            else:
                resp_recv.setdefault(wrkr, []).append((ts_ns, line.rstrip("\n")))
    # Ensure time order
    for m in (req_send, resp_recv):
        for wrkr in m:
            m[wrkr].sort(key=lambda x: x[0])
    return req_send, resp_recv


def load_worker_events_sequential(workers_glob: str, only_client_id: int = -1) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, List[Tuple[int, str]]]]:
    """Return per-worker ordered lists of (ts_ns, raw_line) for worker events."""
    req_recv: Dict[int, List[Tuple[int, str]]] = {}
    resp_send: Dict[int, List[Tuple[int, str]]] = {}
    paths = sorted(glob.glob(workers_glob))
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("event="):
                    continue
                kv = parse_kv_line(line)
                ev = kv.get("event")
                if ev not in {"req_recv", "resp_send"}:
                    continue
                ts_ns = int(kv["ts_ns"]) if "ts_ns" in kv else None
                wrkr = int(kv["wrkr"]) if "wrkr" in kv else None
                clt = int(kv["clt"]) if "clt" in kv else None
                if ts_ns is None or wrkr is None:
                    continue
                if only_client_id != -1 and clt is not None and clt != only_client_id:
                    continue
                if ev == "req_recv":
                    req_recv.setdefault(wrkr, []).append((ts_ns, line.rstrip("\n")))
                else:
                    resp_send.setdefault(wrkr, []).append((ts_ns, line.rstrip("\n")))
    # Ensure time order
    for m in (req_recv, resp_send):
        for wrkr in m:
            m[wrkr].sort(key=lambda x: x[0])
    return req_recv, resp_send

def build_pairs_per_worker(
    left_by_wrkr: Dict[int, List[Tuple[int, str]]],
    right_by_wrkr: Dict[int, List[Tuple[int, str]]],
) -> Dict[int, List[Tuple[Tuple[int, str], Tuple[int, str]]]]:
    """Index-align sequentially per worker; return pairs of ((ts,line)_left, (ts,line)_right)."""
    pairs: Dict[int, List[Tuple[Tuple[int, str], Tuple[int, str]]]] = {}
    for wrkr, left_list in left_by_wrkr.items():
        right_list = right_by_wrkr.get(wrkr)
        if not right_list:
            continue
        n = min(len(left_list), len(right_list))
        if n <= 0:
            continue
        pairs[wrkr] = [(left_list[i], right_list[i]) for i in range(n)]
    return pairs


def write_debug_files(
    debug_dir: str,
    client_req_send: Dict[int, List[Tuple[int, str]]],
    client_resp_recv: Dict[int, List[Tuple[int, str]]],
    worker_req_recv: Dict[int, List[Tuple[int, str]]],
    worker_resp_send: Dict[int, List[Tuple[int, str]]],
) -> None:
    """Write a time-sorted merge of raw lines from client and worker per worker id."""
    os.makedirs(debug_dir, exist_ok=True)
    all_workers = set(client_req_send.keys()) | set(client_resp_recv.keys()) | set(worker_req_recv.keys()) | set(worker_resp_send.keys())
    for wrkr in sorted(all_workers):
        merged: List[Tuple[int, str]] = []
        for ts, line in client_req_send.get(wrkr, []):
            merged.append((ts, f"CLIENT {line}"))
        for ts, line in client_resp_recv.get(wrkr, []):
            merged.append((ts, f"CLIENT {line}"))
        for ts, line in worker_req_recv.get(wrkr, []):
            merged.append((ts, f"WORKER {line}"))
        for ts, line in worker_resp_send.get(wrkr, []):
            merged.append((ts, f"WORKER {line}"))
        merged.sort(key=lambda x: x[0])
        path = os.path.join(debug_dir, f"client-worker_{wrkr}.debug")
        with open(path, "w") as f:
            for ts, line in merged:
                f.write(f"{line}\n")


def write_debug_pairs(
    debug_pairs_dir: str,
    client_req_send: Dict[int, List[Tuple[int, str]]],
    worker_req_recv: Dict[int, List[Tuple[int, str]]],
) -> None:
    """Write per-worker sequential pairs for req_send (client) and req_recv (worker).

    Output format per line:
      [k] CLIENT: <raw client req_send line>
      [k] WORKER: <raw worker req_recv line>
    """
    os.makedirs(debug_pairs_dir, exist_ok=True)
    all_workers = set(client_req_send.keys()) | set(worker_req_recv.keys())
    for wrkr in sorted(all_workers):
        c_list = sorted(client_req_send.get(wrkr, []), key=lambda x: x[0])
        w_list = sorted(worker_req_recv.get(wrkr, []), key=lambda x: x[0])
        n = min(len(c_list), len(w_list))
        if n == 0:
            continue
        path = os.path.join(debug_pairs_dir, f"client-worker_{wrkr}.debug_pairs")
        with open(path, "w") as f:
            for k in range(n):
                _, cline = c_list[k]
                _, wline = w_list[k]
                f.write(f"[{k}] CLIENT: {cline}\n")
                f.write(f"[{k}] WORKER: {wline}\n")


def load_diffs_from_debug_pairs(pairs_path: str) -> List[int]:
    """Parse a debug_pairs file and return list of deltas (worker.req_recv ts - client.req_send ts).

    Emits a warning including the file and item index when a delta is negative.
    """
    diffs: List[int] = []
    with open(pairs_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # Expect pairs of lines per k
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        client_line = lines[i]
        worker_line = lines[i + 1]
        # Extract item index [k] from the start of the line if present
        k_match = re.match(r"^\[(\d+)\]", client_line)
        k_idx = int(k_match.group(1)) if k_match else (i // 2)
        kv_c = parse_kv_line(client_line)
        kv_w = parse_kv_line(worker_line)
        if "ts_ns" in kv_c and "ts_ns" in kv_w:
            try:
                c_ts = int(kv_c["ts_ns"]) 
                w_ts = int(kv_w["ts_ns"]) 
                delta = w_ts - c_ts
                if delta < 0:
                    print(f"WARNING: {pairs_path} item={k_idx} negative delta {delta} ns")
                diffs.append(delta)
            except Exception:
                continue
    return diffs


def plot_cdf(values_ns: List[int], out_png: str, unit: str, title: str, no_grid: bool = False) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required to plot. Install with: pip install matplotlib")
        print("Error:", e)
        return
    xs_ns, ys = compute_ecdf(values_ns)
    scale, unit_label = get_scale(unit)
    xs = [v / scale for v in xs_ns]
    plt.figure(figsize=(7.5, 4.5))
    if xs:
        plt.plot(xs, ys, drawstyle="steps-post")
    plt.xlabel(f"latency ({unit_label})")
    plt.ylabel("CDF")
    plt.title(title)
    if not no_grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=150)


def compute_ecdf(values: List[int]) -> Tuple[List[int], List[float]]:
    if not values:
        return [], []
    values_sorted = sorted(values)
    n = len(values_sorted)
    y = [(i + 1) / n for i in range(n)]
    return values_sorted, y


def compute_percentile(values_sorted: List[int], p: float) -> float:
    if not values_sorted:
        return float("nan")
    if p <= 0:
        return float(values_sorted[0])
    if p >= 100:
        return float(values_sorted[-1])
    n = len(values_sorted)
    k = (p / 100.0) * (n - 1)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(values_sorted[int(k)])
    return values_sorted[lower] + (values_sorted[upper] - values_sorted[lower]) * (k - lower)


def get_scale(unit: str) -> Tuple[float, str]:
    if unit == "ns":
        return 1.0, "ns"
    if unit == "us":
        return 1e3, "µs"
    if unit == "ms":
        return 1e6, "ms"
    if unit == "s":
        return 1e9, "s"
    return 1e6, "ms"


def main() -> int:
    args = parse_args()

    client_path = os.path.abspath(args.client)
    workers_glob = os.path.abspath(args.workers_glob)

    # Helper to run the standard single-client pipeline
    def run_single_client(client_log_path: str, workers_glob_path: str) -> int:
        nonlocal args
        _saved_client = client_log_path
        _saved_workers = workers_glob_path
        # Reuse the existing logic below by shadowing local variables
        nonlocal client_path, workers_glob
        client_path = os.path.abspath(_saved_client)
        workers_glob = os.path.abspath(_saved_workers)

        if not os.path.isfile(client_path):
            print(f"Client log not found: {client_path}")
            return 2
        worker_paths = sorted(glob.glob(workers_glob))
        if not worker_paths:
            print(f"No worker logs matched: {workers_glob}")
            return 2

        client_req_send, client_resp_recv = load_client_events_sequential(client_path)
        # Detect client id from the client log; default to -1 if absent
        client_id = -1
        try:
            with open(client_path, "r") as _f:
                for _line in _f:
                    if not _line.startswith("event="):
                        continue
                    _kv = parse_kv_line(_line)
                    if "clt" in _kv:
                        client_id = int(_kv["clt"])  # use first seen clt
                        break
        except Exception:
            pass
        worker_req_recv, worker_resp_send = load_worker_events_sequential(workers_glob, only_client_id=client_id)

        # Build per-worker sequential pairs
        pairs_c2w = build_pairs_per_worker(client_req_send, worker_req_recv)
        pairs_w2c = build_pairs_per_worker(worker_resp_send, client_resp_recv)

        # Write debug files with all raw pairs
        # Place debug outputs next to the client log directory
        base_dir = os.path.dirname(os.path.abspath(client_path))
        debug_dir = os.path.join(base_dir, "debug")
        write_debug_files(debug_dir, client_req_send, client_resp_recv, worker_req_recv, worker_resp_send)

        # Also write sequential debug_pairs for req_send/req_recv
        debug_pairs_dir = os.path.join(base_dir, "debug_pairs")
        write_debug_pairs(debug_pairs_dir, client_req_send, worker_req_recv)

        # Print negatives for client->worker with raw lines to aid debugging
        worker_dir = os.path.dirname(workers_glob)
        for wrkr, plist in pairs_c2w.items():
            for idx, ((cts, cline), (wts, wline)) in enumerate(plist):
                delta = wts - cts
                if delta < 0:
                    print(
                        f"NEGATIVE c2w: worker={wrkr}, idx={idx}, delta={delta} ns\n"
                        f"  CLIENT ({client_path}): {cline}\n"
                        f"  WORKER ({os.path.join(worker_dir, f'worker_log_{wrkr}.txt')}): {wline}"
                    )

        # Apply per-worker skip before computing diffs
        skip_k = max(0, int(args.skip_first))
        c2w_ns: List[int] = []
        w2c_ns: List[int] = []
        for wrkr, plist in pairs_c2w.items():
            for (cts, _cline), (wts, _wline) in plist[skip_k:]:
                c2w_ns.append(wts - cts)
        for wrkr, plist in pairs_w2c.items():
            for (wts, _wline), (cts, _cline) in plist[skip_k:]:
                w2c_ns.append(cts - wts)

        def summarize(label: str, values: List[int]) -> None:
            if not values:
                print(f"{label}: no pairs matched")
                return
            xs, _ = compute_ecdf(values)
            p50 = compute_percentile(xs, 50)
            p90 = compute_percentile(xs, 90)
            p99 = compute_percentile(xs, 99)
            print(
                f"{label}: count={len(xs)}, min={xs[0]} ns, p50={p50:.0f} ns, p90={p90:.0f} ns, p99={p99:.0f} ns, max={xs[-1]} ns"
            )

        summarize("client->worker (req_send to req_recv)", c2w_ns)
        summarize("worker->client (resp_send to resp_recv_ud)", w2c_ns)

        # Plot CDFs
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib is required to plot. Install with: pip install matplotlib")
            print("Error:", e)
            return 3

        scale, unit_label = get_scale(args.xunit)
        c2w_xs_ns, c2w_ys = compute_ecdf(c2w_ns)
        w2c_xs_ns, w2c_ys = compute_ecdf(w2c_ns)
        c2w_xs = [v / scale for v in c2w_xs_ns]
        w2c_xs = [v / scale for v in w2c_xs_ns]

        plt.figure(figsize=(8, 5))
        if c2w_xs:
            plt.plot(c2w_xs, c2w_ys, drawstyle="steps-post", label="client->worker")
        if w2c_xs:
            plt.plot(w2c_xs, w2c_ys, drawstyle="steps-post", label="worker->client")
        plt.xlabel(f"latency ({unit_label})")
        plt.ylabel("CDF")
        plt.title("Latency CDFs: client↔worker")
        plt.legend()
        if not args.no_grid:
            plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Suffix output with skip count if provided
        base_out = args.out
        if args.skip_first > 0:
            root, ext = os.path.splitext(base_out)
            base_out = f"{root}_skip{args.skip_first}{ext or '.png'}"
        out_png = os.path.abspath(base_out if os.path.isabs(base_out) else os.path.join(base_dir, base_out))
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")

        # Optional: per-worker pairs plots when requested
        if args.do_pairs_plots:
            pairs_files = sorted(glob.glob(os.path.join(debug_pairs_dir, "client-worker_*.debug_pairs")))
            all_diffs: List[int] = []
            for pf in pairs_files:
                diffs = load_diffs_from_debug_pairs(pf)
                if args.skip_first > 0:
                    diffs = diffs[args.skip_first :]
                all_diffs.extend(diffs)
                base = os.path.splitext(os.path.basename(pf))[0]
                out_png_pf = os.path.join(base_dir, f"{base}_cdf.png")
                plot_cdf(diffs, out_png_pf, args.xunit, f"CDF of req_send->req_recv for {base}", no_grid=args.no_grid)
                print(f"Saved plot -> {out_png_pf}")
            overall_out = os.path.join(base_dir, "client-worker_all_cdf.png")
            plot_cdf(all_diffs, overall_out, args.xunit, "CDF of req_send->req_recv (all workers)", no_grid=args.no_grid)
            print(f"Saved plot -> {overall_out}")

        return 0

    # Scenario mode: discover clients and workers automatically and run for each client
    if args.scenario:
        scenario_dir = os.path.abspath(args.scenario)
        if not os.path.isdir(scenario_dir):
            print(f"Scenario directory not found: {scenario_dir}")
            return 2
        # Determine workers glob automatically (prefer normalized, fallback to raw)
        worker_glob_candidates = [
            os.path.join(scenario_dir, "server", "worker_log_*.normalized.txt"),
            os.path.join(scenario_dir, "server", "worker_log_*.txt"),
        ]
        chosen_workers_glob = None
        for cand in worker_glob_candidates:
            if glob.glob(cand):
                chosen_workers_glob = cand
                break
        if not chosen_workers_glob:
            print(f"No worker logs found under {os.path.join(scenario_dir, 'server')}")
            return 2

        # Find client logs under c*/
        client_logs: List[str] = []
        for cdir in sorted(glob.glob(os.path.join(scenario_dir, "c*"))):
            if not os.path.isdir(cdir):
                continue
            # Prefer normalized, fallback to raw
            for pattern in ("log_*.normalized.txt", "log_*.txt"):
                files = sorted(glob.glob(os.path.join(cdir, pattern)))
                if files:
                    client_logs.extend(files)
                    break
        if not client_logs:
            print(f"No client logs found under {scenario_dir}")
            return 2

        rc = 0
        for cl in client_logs:
            # Set default output to the client directory
            args.out = os.path.join(os.path.dirname(cl), "latency_cdfs_seq_raw_filtered.png")
            _rc = run_single_client(cl, chosen_workers_glob)
            rc = rc or _rc
        return rc

    if args.from_debug_pairs:
        # Plot per-worker CDFs directly from debug_pairs files
        pairs_files = sorted(glob.glob(os.path.abspath(args.pairs_glob)))
        if not pairs_files:
            print(f"No debug_pairs files matched: {args.pairs_glob}")
            return 2
        all_diffs: List[int] = []
        for pf in pairs_files:
            diffs = load_diffs_from_debug_pairs(pf)
            if args.skip_first > 0:
                diffs = diffs[args.skip_first :]
            # Include negatives on plots; warnings are printed during load
            all_diffs.extend(diffs)
            base = os.path.splitext(os.path.basename(pf))[0]
            out_png = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{base}_cdf.png")
            title = f"CDF of req_send->req_recv for {base}"
            plot_cdf(diffs, out_png, args.xunit, title, no_grid=args.no_grid)
            print(f"Saved plot -> {out_png}")
        # Overall combined plot
        overall_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "client-worker_all_cdf.png")
        plot_cdf(all_diffs, overall_out, args.xunit, "CDF of req_send->req_recv (all workers)", no_grid=args.no_grid)
        print(f"Saved plot -> {overall_out}")
        return 0

    if not os.path.isfile(client_path):
        print(f"Client log not found: {client_path}")
        return 2
    worker_paths = sorted(glob.glob(workers_glob))
    if not worker_paths:
        print(f"No worker logs matched: {workers_glob}")
        return 2

    client_req_send, client_resp_recv = load_client_events_sequential(client_path)
    # Detect client id from the client log; default to -1 if absent
    client_id = -1
    try:
        with open(client_path, "r") as _f:
            for _line in _f:
                if not _line.startswith("event="):
                    continue
                _kv = parse_kv_line(_line)
                if "clt" in _kv:
                    client_id = int(_kv["clt"])  # use first seen clt
                    break
    except Exception:
        pass
    worker_req_recv, worker_resp_send = load_worker_events_sequential(workers_glob, only_client_id=client_id)

    # Build per-worker sequential pairs
    pairs_c2w = build_pairs_per_worker(client_req_send, worker_req_recv)
    pairs_w2c = build_pairs_per_worker(worker_resp_send, client_resp_recv)

    # Write debug files with all raw pairs
    # Place debug outputs next to the client log directory
    base_dir = os.path.dirname(os.path.abspath(client_path))
    debug_dir = os.path.join(base_dir, "debug")
    write_debug_files(debug_dir, client_req_send, client_resp_recv, worker_req_recv, worker_resp_send)

    # Also write sequential debug_pairs for req_send/req_recv
    debug_pairs_dir = os.path.join(base_dir, "debug_pairs")
    write_debug_pairs(debug_pairs_dir, client_req_send, worker_req_recv)

    # Print negatives for client->worker with raw lines to aid debugging
    worker_dir = os.path.dirname(workers_glob)
    for wrkr, plist in pairs_c2w.items():
        for idx, ((cts, cline), (wts, wline)) in enumerate(plist):
            delta = wts - cts
            if delta < 0:
                print(
                    f"NEGATIVE c2w: worker={wrkr}, idx={idx}, delta={delta} ns\n"
                    f"  CLIENT ({client_path}): {cline}\n"
                    f"  WORKER ({os.path.join(worker_dir, f'worker_log_{wrkr}.txt')}): {wline}"
                )

    # Apply per-worker skip before computing diffs
    skip_k = max(0, int(args.skip_first))
    c2w_ns: List[int] = []
    w2c_ns: List[int] = []
    for wrkr, plist in pairs_c2w.items():
        for (cts, _cline), (wts, _wline) in plist[skip_k:]:
            c2w_ns.append(wts - cts)
    for wrkr, plist in pairs_w2c.items():
        for (wts, _wline), (cts, _cline) in plist[skip_k:]:
            w2c_ns.append(cts - wts)

    def summarize(label: str, values: List[int]) -> None:
        if not values:
            print(f"{label}: no pairs matched")
            return
        xs, _ = compute_ecdf(values)
        p50 = compute_percentile(xs, 50)
        p90 = compute_percentile(xs, 90)
        p99 = compute_percentile(xs, 99)
        print(
            f"{label}: count={len(xs)}, min={xs[0]} ns, p50={p50:.0f} ns, p90={p90:.0f} ns, p99={p99:.0f} ns, max={xs[-1]} ns"
        )

    summarize("client->worker (req_send to req_recv)", c2w_ns)
    summarize("worker->client (resp_send to resp_recv_ud)", w2c_ns)

    # Plot CDFs
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required to plot. Install with: pip install matplotlib")
        print("Error:", e)
        return 3

    scale, unit_label = get_scale(args.xunit)
    c2w_xs_ns, c2w_ys = compute_ecdf(c2w_ns)
    w2c_xs_ns, w2c_ys = compute_ecdf(w2c_ns)
    c2w_xs = [v / scale for v in c2w_xs_ns]
    w2c_xs = [v / scale for v in w2c_xs_ns]

    plt.figure(figsize=(8, 5))
    if c2w_xs:
        plt.plot(c2w_xs, c2w_ys, drawstyle="steps-post", label="client->worker")
    if w2c_xs:
        plt.plot(w2c_xs, w2c_ys, drawstyle="steps-post", label="worker->client")
    plt.xlabel(f"latency ({unit_label})")
    plt.ylabel("CDF")
    plt.title("Latency CDFs: client↔worker")
    plt.legend()
    if not args.no_grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Suffix output with skip count if provided
    base_out = args.out
    if args.skip_first > 0:
        root, ext = os.path.splitext(base_out)
        base_out = f"{root}_skip{args.skip_first}{ext or '.png'}"
    out_png = os.path.abspath(base_out)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot -> {out_png}")

    # Optional CSV
    if args.csv:
        out_csv = os.path.abspath(args.csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w") as f:
            f.write("series,x,ecdf\n")
            for x, y in zip(c2w_xs, c2w_ys):
                f.write(f"c2w,{x},{y}\n")
            for x, y in zip(w2c_xs, w2c_ys):
                f.write(f"w2c,{x},{y}\n")
        print(f"Wrote ECDF CSV -> {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


