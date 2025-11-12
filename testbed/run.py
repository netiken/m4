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
import threading
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

# Quick test configuration (4 scenarios: small/large RDMA × single/multi window)
QUICK_WINDOW_SIZES: List[int] = [1, 4]
QUICK_RDMA_SIZES: List[int] = [256008, 1024008]  # 250KB and 1000KB

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
            
    def run_sweep(self, jobs: int, quick: bool = False) -> bool:
        """Run NS3 sweep - from backends/UNISON/run_sweep.py"""
        if not self.binary_path.exists():
            print(f"❌ NS3 binary not found: {self.binary_path}", file=sys.stderr)
            print("Build NS3 first: cd backends/UNISON && ./ns3 build", file=sys.stderr)
            return False
        
        # Clean and create output directory
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Select sweep parameters based on quick mode
        window_sizes = QUICK_WINDOW_SIZES if quick else WINDOW_SIZES
        rdma_sizes = QUICK_RDMA_SIZES if quick else RDMA_SIZES
        mode_str = "quick test (4 scenarios)" if quick else f"full sweep ({len(window_sizes) * len(rdma_sizes)} scenarios)"
        
        print(f"[ns3] Running {mode_str} with {jobs} parallel jobs...")
        
        # Generate all sweep tasks
        tasks = []
        for window in window_sizes:
            for rdma in rdma_sizes:
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
            
            # Compute UD and RDMA durations
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


class M4Backend:
    """M4 (ML-enhanced) backend"""
    
    def __init__(self, gpu_ids: list = None, model_dir: str = "models_v12"):
        self.name = "m4"
        self.backend_dir = BACKENDS_DIR / "m4"
        self.results_dir = ROOT_DIR / "eval_test" / "m4"
        self.binary_path = self.backend_dir / "build" / "main"
        self._file_lock = threading.Lock()  # Lock to prevent concurrent file access
        # Support multiple GPUs - assign tasks round-robin to available GPUs
        self.gpu_ids = gpu_ids if gpu_ids else [0]  # Default to GPU 0
        self._gpu_counter = 0  # Counter for round-robin GPU assignment
        self.model_dir = model_dir  # Model directory for ML inference
        self.model_dir_path = (self.backend_dir / self.model_dir).resolve()
            
    def run_sweep(self, jobs: int, quick: bool = False) -> bool:
        """Run M4 sweep - from backends/m4/run_sweep.py"""
        if not self.binary_path.exists():
            print(f"❌ M4 binary not found: {self.binary_path}", file=sys.stderr)
            print("Build M4 first: ./build.sh m4", file=sys.stderr)
            return False
        
        if not self.model_dir_path.exists():
            available = sorted(
                p.name for p in self.backend_dir.glob("models_v*") if p.is_dir()
            )
            print(f"❌ Model directory not found: {self.model_dir_path}", file=sys.stderr)
            if available:
                print(f"   Available model directories: {', '.join(available)}", file=sys.stderr)
            else:
                print("   No models_v* directories found under backends/m4/", file=sys.stderr)
            return False
        
        # Clean and create output directory
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit jobs to number of available GPUs (each job needs 1 GPU)
        max_jobs = len(self.gpu_ids)
        actual_jobs = min(jobs, max_jobs)
        if jobs > max_jobs:
            print(f"[m4] WARNING: Requested {jobs} jobs but only {max_jobs} GPU(s) available. Using {actual_jobs} jobs.")
        
        # Select sweep parameters based on quick mode
        window_sizes = QUICK_WINDOW_SIZES if quick else WINDOW_SIZES
        rdma_sizes = QUICK_RDMA_SIZES if quick else RDMA_SIZES
        mode_str = "quick test (4 scenarios)" if quick else f"full sweep ({len(window_sizes) * len(rdma_sizes)} scenarios)"
        
        print(f"[m4] Running {mode_str} with {actual_jobs} parallel jobs on GPU(s): {self.gpu_ids}...")
        
        # Generate all sweep tasks
        tasks = []
        for window in window_sizes:
            for rdma in rdma_sizes:
                title = RDMA_TITLES_BASE[rdma]
                run_tag = f"{title}_{window}"
                run_dir = self.results_dir / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((run_tag, window, rdma, run_dir))
        
        def _run_task(task):
            run_tag, window, rdma, run_dir = task
            
            # Assign GPU in round-robin fashion
            with self._file_lock:
                gpu_id = self.gpu_ids[self._gpu_counter % len(self.gpu_ids)]
                self._gpu_counter += 1
            
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"
            
            cmd = [
                str(self.binary_path.resolve()),
                str(window),
                str(rdma),
                "12",
                str(gpu_id),
                str(self.model_dir_path),
            ]
            
            with stdout_path.open("w") as out, stderr_path.open("w") as err:
                proc = subprocess.run(cmd, cwd=str(run_dir), stdout=out, stderr=err, text=True)
            
            if proc.returncode != 0:
                print(f"  -> [GPU {gpu_id}] non-zero exit {proc.returncode}; check {stderr_path}")
            
            return (run_tag, gpu_id)
        
        # Run tasks in parallel (limited to number of GPUs)
        if actual_jobs > 1:
            with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
                futures = {executor.submit(_run_task, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        run_tag, gpu_id = future.result()
                        print(f"✓ Completed {run_tag} on GPU {gpu_id}")
                    except Exception as e:
                        print(f"✗ Failed: {e}", file=sys.stderr)
                        return False
        else:
            for task in tasks:
                run_tag, gpu_id = _run_task(task)
                print(f"✓ Completed {run_tag} on GPU {gpu_id}")
        
        return True
    
    def process_results(self) -> bool:
        """Process M4 results - from backends/m4/process.py"""
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}", file=sys.stderr)
            return False
        
        print(f"[m4] Processing results...")
        
        CLIENT_LOG_PATTERN = "client_*.log"
        NUMERIC_FIELDS = {"ts_ns", "id", "clt", "wrkr", "slot", "size", "start_ns", "dur_ns", "wire_bytes"}
        
        def parse_line(line):
            """Parse a line into a dict"""
            parts = line.strip().split()
            data = {}
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
        
        def format_event_ns3(event):
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
                "rdma_send": "server rdma_send",
                "rdma_recv": "client rdma_recv",
            }
            role_event = role_map.get(event.get("event", ""), event.get("event", "event"))
            size = event.get("size", event.get("wire_bytes", 0))
            return (
                f"[{role_event}] t={event.get('ts_ns', 0)} ns reqId={event.get('id', 0)} "
                f"size={size}B client_node_id={event.get('clt', 0)}"
            )
        
        EVENT_RE = re.compile(r"\[(\w+)\s+([\w_]+)\]\s+t=(\d+)\s+ns\s+reqId=(\d+).*client_node_id=(\d+)")
        
        for subdir in sorted(self.results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            # Step 1: Collect events from client logs
            flows = collections.defaultdict(list)
            for log_path in sorted(subdir.glob(CLIENT_LOG_PATTERN)):
                with log_path.open("r") as f:
                    for raw in f:
                        if not raw.strip():
                            continue
                        event = parse_line(raw)
                        if not event or "clt" not in event or "id" not in event:
                            continue
                        key = (event["clt"], event["id"])
                        flows[key].append(event)
            
            if not flows:
                continue
            
            # Step 2: Write grouped_flows.txt
            grouped_file = subdir / "grouped_flows.txt"
            with grouped_file.open("w") as out:
                for (_, req_id), events in sorted(flows.items(), key=lambda entry: min(event.get("ts_ns", 0) for event in entry[1])):
                    out.write(f"### reqId={req_id} ###\n")
                    for event in sorted(events, key=lambda e: e["ts_ns"]):
                        out.write(format_event_ns3(event) + "\n")
                    out.write("\n")
            
            # Step 3: Parse grouped flows and compute durations
            parsed_flows = collections.defaultdict(list)
            with grouped_file.open("r") as f:
                current_req = None
                for raw in f:
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
                    parsed_flows[current_req].append({
                        "side": side,
                        "event": event_name,
                        "t": int(ts),
                        "client": int(client),
                    })
            
            # Step 4: Compute UD and RDMA durations
            outputs = []
            sorted_items = sorted(parsed_flows.items(), key=lambda kv: min(event["t"] for event in kv[1]))
            
            for req_id, events in sorted_items:
                events_sorted = sorted(events, key=lambda event: event["t"])
                client = events_sorted[0]["client"] if events_sorted else 0
                
                req_send = next((e["t"] for e in events_sorted if e["event"] == "req_send"), None)
                hand_send = next((e["t"] for e in events_sorted if e["event"] == "hand_send"), None)
                rdma_recv = next((e["t"] for e in events_sorted if e["event"] == "rdma_recv"), None)
                
                if req_send is not None and hand_send is not None:
                    outputs.append({
                        "type": "ud",
                        "client": client,
                        "id": req_id,
                        "dur": hand_send - req_send,
                        "ts": req_send,
                    })
                
                if hand_send is not None and rdma_recv is not None:
                    outputs.append({
                        "type": "rdma",
                        "client": client,
                        "id": req_id,
                        "dur": rdma_recv - hand_send,
                        "ts": hand_send,
                    })
            
            # Sort by timestamp and write m4_output.txt (compatible with analyze.py)
            outputs.sort(key=lambda entry: entry["ts"])
            output_file = subdir / "m4_output.txt"
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
            
    def run_sweep(self, jobs: int, quick: bool = False) -> bool:
        """Run FlowSim sweep - from backends/flowsim/run_sweep.py"""
        if not self.binary_path.exists():
            print(f"❌ FlowSim binary not found: {self.binary_path}", file=sys.stderr)
            print("Build FlowSim first: cd backends/flowsim && make", file=sys.stderr)
            return False
        
        # Clean and create output directory
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Select sweep parameters based on quick mode
        window_sizes = QUICK_WINDOW_SIZES if quick else WINDOW_SIZES
        rdma_sizes = QUICK_RDMA_SIZES if quick else RDMA_SIZES
        mode_str = "quick test (4 scenarios)" if quick else f"full sweep ({len(window_sizes) * len(rdma_sizes)} scenarios)"
        
        print(f"[flowsim] Running {mode_str} with {jobs} parallel jobs...")
        
        # Generate all sweep tasks
        tasks = []
        for window in window_sizes:
            for rdma in rdma_sizes:
                title = RDMA_TITLES_BASE[rdma]
                run_tag = f"{title}_{window}"
                run_dir = self.results_dir / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((run_tag, window, rdma, run_dir))
        
        def _run_task(task):
            run_tag, window, rdma, run_dir = task
            
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"
            
            cmd = [str(self.binary_path.resolve()), str(window), str(rdma), "12"]
            
            with stdout_path.open("w") as out, stderr_path.open("w") as err:
                proc = subprocess.run(cmd, cwd=str(run_dir), stdout=out, stderr=err, text=True)
            
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
            
            # Write grouped_flows.txt
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
        choices=["ns3", "flowsim", "m4", "all"],
        help="Backend to run"
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=32,
        help="Number of parallel jobs (default: 32)"
    )
    parser.add_argument(
        "--gpu", "-g",
        type=str,
        default="0,1,2,3",
        help="GPU ID(s) for M4 backend. Single GPU: '0', Multiple GPUs: '0,1,2,3' (default: '0,1,2,3')"
    )
    parser.add_argument(
        "--model-dir", "-m",
        type=str,
        default="models_v12",
        help="Model directory for M4 ML inference (default: 'models_v12')"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick test with only 4 scenarios (100KB/1000KB × window 1/4) instead of full 27 scenarios"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    try:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
    except ValueError:
        print(f"❌ Invalid GPU ID format: {args.gpu}. Use format like '0' or '0,1,2'", file=sys.stderr)
        return 1
    
    # Select backends
    if args.backend == "all":
        backends = [NS3Backend(), FlowSimBackend(), M4Backend(gpu_ids=gpu_ids, model_dir=args.model_dir)]
    elif args.backend == "ns3":
        backends = [NS3Backend()]
    elif args.backend == "flowsim":
        backends = [FlowSimBackend()]
    else:
        backends = [M4Backend(gpu_ids=gpu_ids, model_dir=args.model_dir)]
    
    # Run each backend
    all_success = True
    for backend in backends:
        print(f"\n{'='*80}")
        print(f"[{backend.name}] Running sweep...")
        print(f"{'='*80}\n")
        
        if not backend.run_sweep(args.jobs, quick=args.quick):
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
