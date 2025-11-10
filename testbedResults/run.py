#!/usr/bin/env python3
"""
Unified runner for M4 network simulator backends.
Merges run_sweep.py and process.py for both NS3 and FlowSim.

Usage:
    python run.py ns3 [--jobs N]
    python run.py flowsim [--jobs N]
    python run.py all [--jobs N]
"""

import argparse
import collections
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

# Shared sweep configuration
WINDOW_SIZES: List[int] = [1, 2, 4]
RDMA_TITLES_BASE: Dict[int, str] = {
    102408: "100",
    256008: "250",
    307208: "300",
    409608: "400",
    512008: "500",
    665608: "650",
    768008: "750",
    921608: "900",
    1024008: "1000",
}
RDMA_SIZES: List[int] = list(RDMA_TITLES_BASE.keys())

# Paths
ROOT_DIR = pathlib.Path(__file__).resolve().parent
BACKENDS_DIR = ROOT_DIR / "backends"


class NS3Backend:
    """NS3 (UNISON) backend"""
    
    def __init__(self):
        self.name = "ns3"
        self.backend_dir = BACKENDS_DIR / "UNISON"
        self.results_dir = ROOT_DIR / "eval_test" / "ns3"
        self.binary_path = self.backend_dir / "build" / "scratch" / "ns3.39-twelve-optimized"
            
    def run_sweep(self, jobs: int) -> bool:
        """Run NS3 sweep - from backends/UNISON/run_sweep.py"""
        if not self.binary_path.exists():
            print(f"❌ NS3 binary not found: {self.binary_path}", file=sys.stderr)
            print("Build NS3 first: cd backends/UNISON && ./ns3 build", file=sys.stderr)
            return False
        
        # Clean and create output directory
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[ns3] Running sweep with {jobs} parallel jobs...")
        
        # Generate all sweep tasks
        tasks = []
        for window in WINDOW_SIZES:
            for rdma in RDMA_SIZES:
                title = RDMA_TITLES_BASE[rdma]
                run_tag = f"{title}_{window}"
                run_dir = self.results_dir / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((run_tag, window, rdma, run_dir))
        
        def _run_task(task):
            run_tag, window, rdma, run_dir = task
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"
            
            cmd = [
                str(self.binary_path.resolve()),
                f"--maxWindows={window}",
                f"--dataBytes={rdma}"
            ]
            
            with stdout_path.open("w") as out, stderr_path.open("w") as err:
                proc = subprocess.run(cmd, cwd=str(self.backend_dir), stdout=out, stderr=err, text=True)
            
            if proc.returncode != 0:
                print(f"  -> non-zero exit {proc.returncode}; check {stderr_path}")
            return run_tag
        
        # Run tasks in parallel
        if jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(_run_task, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        run_tag = future.result()
                        print(f"✓ Completed {run_tag}")
                    except Exception as e:
                        print(f"✗ Failed: {e}", file=sys.stderr)
                        return False
        else:
            for task in tasks:
                run_tag = _run_task(task)
                print(f"✓ Completed {run_tag}")
        
        return True
            
    def process_results(self) -> bool:
        """Process NS3 results - from backends/UNISON/process.py"""
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}", file=sys.stderr)
            return False
        
        print(f"[ns3] Processing results...")
        
        EVENT_RE = re.compile(
            r"\[(\w+)\s+([\w_]+)\]\s+t=(\d+)\s+ns\s+reqId=(\d+).*client_node_id=(\d+)"
        )
        
        def extract_sequences(events, pattern):
            """Find non-overlapping subsequences that match the provided event order."""
            sequences = []
            n = len(events)
            i = 0
            while i < n:
                matched = []
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
        
        for subdir in sorted(self.results_dir.iterdir()):
            if not subdir.is_dir():
                continue
                
            stdout_file = subdir / "stdout.txt"
            if not stdout_file.exists():
                continue
            
            # Step 1: Group flows from stdout.txt
            groups = OrderedDict()
            with stdout_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.search(r"\breqId=(\d+)", line)
                    if not match:
                        continue
                    reqid = match.group(1)
                    groups.setdefault(reqid, []).append(line)
            
            # Write grouped_flows.txt
            grouped_file = subdir / "grouped_flows.txt"
            with grouped_file.open("w") as f:
                for reqid, lines in groups.items():
                    f.write(f"### reqId={reqid} ###\n")
                    for line in lines:
                        f.write(f"{line}\n")
                    f.write("\n")
            
            # Step 2: Parse events
            flows = collections.defaultdict(list)
            for reqid, lines in groups.items():
                for line in lines:
                    match = EVENT_RE.match(line)
                    if match:
                        side, event, ts, req_id, client = match.groups()
                        flows[reqid].append({
                            "side": side,
                            "event": event,
                            "t": int(ts),
                            "client": int(client),
                        })
            
            # Step 3: Compute UD and RDMA durations (EXACT logic from process.py)
            outputs = []
            dup_counter = collections.defaultdict(int)
            
            sorted_items = sorted(flows.items(), key=lambda kv: min(event["t"] for event in kv[1]))
            for reqid, events in sorted_items:
                events_sorted = sorted(events, key=lambda event: event["t"])
                
                # Get client (1-indexed in logs, convert to 0-indexed)
                client = None
                for event in events_sorted:
                    if event["event"] == "req_send":
                        client = event["client"] - 1
                        break
                if client is None and events_sorted:
                    client = events_sorted[0]["client"] - 1
                if client is None:
                    client = 0
                
                # Extract UD sequences (handles multiple UD phases)
                ud_sequences = extract_sequences(events_sorted, ["req_send", "req_recv", "resp_send", "resp_recv"])
                for seq in ud_sequences:
                    req_send, req_recv, resp_send, resp_recv = (entry["t"] for entry in seq)
                    ud_duration = resp_recv - req_send
                    suffix = "" if dup_counter[reqid] == 0 else f"-{dup_counter[reqid]}"
                    dup_counter[reqid] += 1
                    outputs.append({
                        "type": "ud",
                        "client": client,
                        "id": f"{reqid}{suffix}",
                        "dur": ud_duration,
                        "ts": seq[0]["t"],
                    })
                
                # Extract RDMA sequences (handles multiple RDMA phases)
                rdma_sequences = extract_sequences(events_sorted, ["hand_send", "rdma_recv"])
                for seq in rdma_sequences:
                    hand_send, rdma_recv = (entry["t"] for entry in seq)
                    rdma_duration = rdma_recv - hand_send
                    suffix = "" if dup_counter[reqid] == 0 else f"-{dup_counter[reqid]}"
                    dup_counter[reqid] += 1
                    outputs.append({
                        "type": "rdma",
                        "client": client,
                        "id": f"{reqid}{suffix}",
                        "dur": rdma_duration,
                        "ts": seq[0]["t"],
                    })
            
            # Sort by timestamp and write
            outputs.sort(key=lambda entry: entry["ts"])
            output_file = subdir / "ns3_output.txt"
            with output_file.open("w") as f:
                for entry in outputs:
                    f.write(f"[{entry['type']}] client={entry['client']} id={entry['id']} dur_ns={entry['dur']}\n")
        
        return True


class FlowSimBackend:
    """FlowSim backend"""
    
    def __init__(self):
        self.name = "flowsim"
        self.backend_dir = BACKENDS_DIR / "flowsim"
        self.results_dir = ROOT_DIR / "eval_test" / "flowsim"
        self.binary_path = self.backend_dir / "main"
            
    def run_sweep(self, jobs: int) -> bool:
        """Run FlowSim sweep - from backends/flowsim/run_sweep.py"""
        if not self.binary_path.exists():
            print(f"❌ FlowSim binary not found: {self.binary_path}", file=sys.stderr)
            print("Build FlowSim first: cd backends/flowsim && make", file=sys.stderr)
            return False
        
        # Clean and create output directory
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[flowsim] Running sweep with {jobs} parallel jobs...")
        
        # Generate all sweep tasks
        tasks = []
        for window in WINDOW_SIZES:
            for rdma in RDMA_SIZES:
                title = RDMA_TITLES_BASE[rdma]
                run_tag = f"{title}_{window}"
                run_dir = self.results_dir / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((run_tag, window, rdma, run_dir))
        
        def _run_task(task):
            run_tag, window, rdma, run_dir = task
            
            # Run inside a temp directory (like original run_sweep.py)
            with tempfile.TemporaryDirectory(prefix="flowsim_run_") as tmp_str:
                tmp_dir = pathlib.Path(tmp_str)
                
                stdout_path = run_dir / "stdout.txt"
                stderr_path = run_dir / "stderr.txt"
                
                cmd = [str(self.binary_path.resolve()), str(window), str(rdma), "12"]
                
                with stdout_path.open("w") as out, stderr_path.open("w") as err:
                    proc = subprocess.run(cmd, cwd=str(tmp_dir), stdout=out, stderr=err, text=True)
                
                if proc.returncode != 0:
                    print(f"  -> non-zero exit {proc.returncode}; check {stderr_path}")
                
                # Collect outputs from temp directory
                for name in ["flows.txt", "server.log", "flowsim_output.txt"]:
                    src = tmp_dir / name
                    if src.exists():
                        shutil.copy2(src, run_dir / name)
                for log in tmp_dir.glob("client_*.log"):
                    shutil.copy2(log, run_dir / log.name)
            
            return run_tag
        
        # Run tasks in parallel
        if jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(_run_task, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        run_tag = future.result()
                        print(f"✓ Completed {run_tag}")
                    except Exception as e:
                        print(f"✗ Failed: {e}", file=sys.stderr)
                        return False
        else:
            for task in tasks:
                run_tag = _run_task(task)
                print(f"✓ Completed {run_tag}")
        
        return True
    
    def process_results(self) -> bool:
        """Process FlowSim results - from backends/flowsim/process.py"""
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}", file=sys.stderr)
            return False
        
        print(f"[flowsim] Processing results...")
        
        CLIENT_LOG_NAMES = [f"client_{i}.log" for i in range(0, 12)]
        
        def parse_line(line):
            """Parse a line into a dict"""
            parts = line.strip().split()
            event_type = parts[0].split("=")[1]
            data = {}
            for part in parts[1:]:
                if '=' in part:
                    k, v = part.split('=', 1)
                    if k in ['ts_ns', 'id', 'clt', 'wrkr', 'slot', 'size', 'start_ns', 'dur_ns']:
                        v = int(v)
                    data[k] = v
            data['event'] = event_type
            return data
        
        def format_event_ns3(e):
            """Format an event in ns3-style log line"""
            role_map = {
                "req_send": "client req_send",
                "req_recv": "server req_recv",
                "resp_send": "server resp_send",
                "resp_recv_ud": "client resp_recv",
                "hand_send": "client hand_send",
                "hand_recv": "server hand_recv",
                "resp_rdma_read": "client rdma_recv",
                "resp_rdma_send": "server rdma_send",
            }
            role_event = role_map.get(e['event'], e['event'])
            size_str = f"{e['size']}B" if 'size' in e else f"{e.get('wire_bytes',0)}B"
            return f"[{role_event}] t={e['ts_ns']} ns reqId={e['id']} size={size_str} client_node_id={e['clt']}"
        
        for subdir in sorted(self.results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            # Read client logs and parse events
            all_events = []
            for log_name in CLIENT_LOG_NAMES:
                log_file = subdir / log_name
                if not log_file.exists():
                    continue
                with log_file.open() as f:
                    for line in f:
                        if line.strip():
                            all_events.append(parse_line(line))
            
            # Group events by client+id
            flows = {}
            for e in all_events:
                flow_key = (e['clt'], e['id'])
                if flow_key not in flows:
                    flows[flow_key] = []
                flows[flow_key].append(e)
            
            # Write grouped_flows.txt (EXACT logic from original process.py)
            # Note: flowsim_output.txt already generated by C++ binary during sweep
            grouped_file = subdir / "grouped_flows.txt"
            with grouped_file.open("w") as out:
                for (clt, reqId), events in sorted(flows.items()):
                    out.write(f"### reqId={reqId} ###\n")
                    for e in sorted(events, key=lambda x: x['ts_ns']):
                        out.write(format_event_ns3(e) + "\n")
                    out.write("\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified runner for M4 network simulator backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "backend",
        choices=["ns3", "flowsim", "all"],
        help="Backend to run"
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=32,
        help="Number of parallel jobs for NS3 (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Select backends
    if args.backend == "all":
        backends = [NS3Backend(), FlowSimBackend()]
    elif args.backend == "ns3":
        backends = [NS3Backend()]
    else:
        backends = [FlowSimBackend()]
    
    # Run each backend
    all_success = True
    for backend in backends:
        print(f"\n{'='*80}")
        print(f"[{backend.name}] Running sweep...")
        print(f"{'='*80}\n")
        
        if not backend.run_sweep(args.jobs):
            print(f"\n❌ [{backend.name}] Sweep FAILED\n", file=sys.stderr)
            all_success = False
            continue
        
        print(f"\n{'='*80}")
        print(f"[{backend.name}] Processing results...")
        print(f"{'='*80}\n")
        
        if not backend.process_results():
            print(f"\n❌ [{backend.name}] Processing FAILED\n", file=sys.stderr)
            all_success = False
            continue
        
        print(f"\n✅ [{backend.name}] Complete! Results in {backend.results_dir}\n")
    
    if all_success:
        print("\n" + "="*80)
        print("🎉 ALL DONE!")
        print("="*80)
        print(f"\nRun analysis: python analyze.py")
        print("="*80 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
