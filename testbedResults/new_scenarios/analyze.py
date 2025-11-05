import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
TRIM_SAMPLES = 100
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

LOCAL_SCALES = {
    "ud": float(os.getenv("LOCAL_UD_SCALE", "1")),
    "rdma": float(os.getenv("LOCAL_RDMA_SCALE", "1")),
}
LOCAL_E2E_SCALE = float(os.getenv("LOCAL_E2E_SCALE", "1"))

# Regex patterns
LINE_RE = re.compile(r"\[(ud|rdma)\] client=(\d+) id=\d+(?:-\d+)? dur_ns=(\d+)")
TS_NS_RE = re.compile(r"ts_ns=(\d+)")


@dataclass
class Source:
    name: str
    base: Path
    filenames: Tuple[str, ...]
    rdma_scale: float = 1.0

    def resolve_file(self, sweep_dir: str, scenario: str) -> Path:
        search_dir = self.base / sweep_dir if sweep_dir else self.base
        for filename in self.filenames:
            candidate = search_dir / scenario / filename
            if candidate.exists():
                return candidate
        return search_dir / scenario / self.filenames[0]


SOURCES = [
    Source("flowsim", REPO_ROOT / "flowsim", ("flowsim_output.txt",)),
    Source("ns3", REPO_ROOT / "UNISON", ("ns3_output.txt",)),
    Source("m4", REPO_ROOT / "m4", ("m4_outputv2.txt", "m4_output.txt", "ns3_output.txt"), rdma_scale=1),
]

TOPOLOGIES = [
    ("sweeps_12", SCRIPT_DIR / "expirements_12", "12"),
]


def trim_series(values: List[int], trim: int = TRIM_SAMPLES) -> List[int]:
    """Trim samples from both ends."""
    if not values or trim <= 0 or len(values) <= trim:
        return values if trim <= 0 else []
    if len(values) <= trim * 2:
        return values[trim:]
    return values[trim:-trim]


def parse_results(file_path: Path) -> Dict[Tuple[int, str], List[int]]:
    """Parse duration results from file."""
    results = defaultdict(list)
    try:
        with file_path.open("r") as f:
            for line in f:
                if match := LINE_RE.match(line.strip()):
                    op_type, client, duration = match.groups()
                    results[(int(client), op_type)].append(int(duration))
    except FileNotFoundError:
        pass
    return dict(sorted(results.items()))


def compute_e2e_duration(scenario_dir: Path, trim: int = TRIM_SAMPLES) -> Optional[Tuple[int, int, int]]:
    """Compute end-to-end duration from event logs."""
    timestamps = []
    
    # Try flows_debug.txt first
    flows_debug = scenario_dir / "flows_debug.txt"
    if flows_debug.exists():
        with flows_debug.open("r") as f:
            for line in f:
                if match := TS_NS_RE.search(line):
                    timestamps.append(int(match.group(1)))
    
    # Try grouped_flows.txt
    if not timestamps:
        grouped_ts_re = re.compile(r"t=(\d+)\s+ns")
        grouped = scenario_dir / "grouped_flows.txt"
        if grouped.exists():
            with grouped.open("r") as f:
                for line in f:
                    if match := grouped_ts_re.search(line):
                        timestamps.append(int(match.group(1)))
    
    # Try client logs
    if not timestamps:
        for log_file in scenario_dir.glob("client*.log"):
            with log_file.open("r") as f:
                for line in f:
                    if match := TS_NS_RE.search(line):
                        timestamps.append(int(match.group(1)))
    
    if not timestamps:
        return None
    
    timestamps.sort()
    if trim > 0 and len(timestamps) > trim * 2:
        timestamps = timestamps[trim:-trim]
    
    start_ns, end_ns = timestamps[0], timestamps[-1]
    return start_ns, end_ns, end_ns - start_ns


def compute_errors(local_vals: List[int], remote_vals: List[int], 
                   op_type: str, rdma_scale: float, local_scale: float) -> np.ndarray:
    """Compute relative errors between local and remote measurements."""
    count = min(len(local_vals), len(remote_vals))
    if count == 0:
        return np.array([])
    
    local = np.array(local_vals[:count], dtype=float) * local_scale
    remote = np.array(remote_vals[:count], dtype=float)
    
    if op_type == "rdma" and rdma_scale != 1.0:
        remote *= rdma_scale
    
    mask = local != 0
    if not np.any(mask):
        return np.array([])
    
    return np.abs(local[mask] - remote[mask]) / local[mask]


def process_scenario(args) -> Dict:
    """Process a single scenario (for parallel execution)."""
    scenario_path, sweep_dir, sources = args
    scenario = scenario_path.name
    
    local_file = scenario_path / "real_world.txt"
    if not local_file.exists():
        return {"scenario": scenario, "skip": True}
    
    local_results = parse_results(local_file)
    if not local_results:
        return {"scenario": scenario, "skip": True}
    
    local_trimmed = {k: trim_series(v) for k, v in local_results.items()}
    
    # Only compute E2E for scenarios ending in "_1"
    compute_e2e = True # scenario.endswith("_1")
    local_e2e = compute_e2e_duration(scenario_path) if compute_e2e else None
    
    result = {
        "scenario": scenario,
        "skip": False,
        "local_e2e": local_e2e,
        "errors": {},
        "local_vals": local_trimmed,
        "remote_vals": {},  # Store remote values for box plots
    }
    
    for source in sources:
        remote_file = source.resolve_file(sweep_dir, scenario)
        if not remote_file.exists():
            continue
        
        remote_results = parse_results(remote_file)
        remote_trimmed = {k: trim_series(v) for k, v in remote_results.items()}
        remote_e2e = compute_e2e_duration(remote_file.parent) if compute_e2e else None
        
        # Store remote values for box plots
        result["remote_vals"][source.name] = remote_trimmed
        
        errors = []
        for key, local_vals in local_trimmed.items():
            if key not in remote_trimmed:
                continue
            errs = compute_errors(
                local_vals, remote_trimmed[key], key[1],
                source.rdma_scale, LOCAL_SCALES.get(key[1], 1.0)
            )
            errors.extend(errs.tolist())
        
        result["errors"][source.name] = {
            "per_request": np.array(errors) if errors else np.array([]),
            "remote_e2e": remote_e2e,
        }
    
    return result


def plot_cdf(data: Dict[str, np.ndarray], output_path: Path, title: str):
    """Plot CDF of errors."""
    plt.figure(figsize=(10, 6))
    
    for name, errors in data.items():
        if len(errors) == 0:
            continue
        sorted_errors = np.sort(errors)
        y = np.linspace(0, 1, len(sorted_errors), endpoint=False)
        plt.step(sorted_errors, y, where="post", label=name, linewidth=2)
    
    plt.xlabel("Relative Error")
    #plt.xscale("log")
    plt.xlim(1e-2, 10)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_e2e_times(scenario_results: List[Dict], output_path: Path, topology_label: str):
    """Plot end-to-end times by scenario."""
    plt.figure(figsize=(10, 6))
    
    # Collect times by first value in scenario name (only for _1 scenarios)
    local_by_first = {}  # first_val -> time_ms
    source_by_first = defaultdict(dict)  # source_name -> {first_val -> time_ms}
    
    for result in scenario_results:
        #if result.get("skip") or not result["scenario"].endswith("_1"):
        #    continue
        
        # Parse first value from scenario name
        parts = re.findall(r"\d+", result["scenario"])
        if not parts:
            continue
        first_val = int(parts[0])
        
        # Local E2E
        if result.get("local_e2e"):
            _, _, duration_ns = result["local_e2e"]
            local_by_first[first_val] = duration_ns * LOCAL_E2E_SCALE / 1e6
        
        # Source E2E
        for source_name, data in result.get("errors", {}).items():
            if data.get("remote_e2e"):
                _, _, duration_ns = data["remote_e2e"]
                source_by_first[source_name][first_val] = duration_ns / 1e6
    
    # Plot local as a line connecting points
    if local_by_first:
        sorted_local = sorted(local_by_first.items())
        xs, ys = zip(*sorted_local)
        plt.plot(xs, ys, marker="D", label="local", linewidth=2, 
                color="tab:red", markersize=8)
    
    # Plot each source as points
    colors = {"flowsim": "tab:blue", "ns3": "tab:orange", "m4": "tab:green"}
    markers = {"flowsim": "o", "ns3": "s", "m4": "^"}
    
    for source_name in ["flowsim", "ns3", "m4"]:
        times = source_by_first.get(source_name, {})
        if times:
            sorted_times = sorted(times.items())
            xs, ys = zip(*sorted_times)
            plt.plot(xs, ys, marker=markers[source_name], label=source_name,
                    linewidth=2, color=colors[source_name], markersize=8)
    
    plt.xlabel("First Scenario Value")
    plt.ylabel("E2E Time (ms)")
    plt.title(f"End-to-End Time (topology {topology_label})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_scenario_box_plots(scenario_results: List[Dict], output_dir: Path, topology_label: str):
    """Generate box plots for each scenario showing duration distributions."""
    for result in scenario_results:
        if result.get("skip"):
            continue
        
        scenario = result["scenario"]
        local_vals = result.get("local_vals", {})
        
        # Get all unique keys from local and remote
        all_keys = set(local_vals.keys())
        for data in result.get("errors", {}).values():
            # Keys are stored in local_vals, we just need to check they exist
            pass
        
        if not all_keys:
            continue
        
        # Group by client and op_type
        clients = sorted({k[0] for k in all_keys})
        op_types = sorted({k[1] for k in all_keys})
        
        # Create subplot grid
        n_clients = len(clients)
        n_ops = len(op_types)
        
        fig, axes = plt.subplots(n_ops, n_clients, figsize=(4 * n_clients, 4 * n_ops))
        if n_ops == 1 and n_clients == 1:
            axes = [[axes]]
        elif n_ops == 1:
            axes = [axes]
        elif n_clients == 1:
            axes = [[ax] for ax in axes]
        
        for op_idx, op_type in enumerate(op_types):
            for client_idx, client in enumerate(clients):
                key = (client, op_type)
                ax = axes[op_idx][client_idx]
                
                if key not in local_vals:
                    ax.set_visible(False)
                    continue
                
                plot_data = []
                labels = []
                colors_list = []
                
                # Add local data
                if key in local_vals and local_vals[key]:
                    plot_data.append(local_vals[key])
                    labels.append("local")
                    colors_list.append("lightcoral")
                
                # Add source data
                source_colors = {"flowsim": "lightblue", "ns3": "lightgreen", "m4": "lightyellow"}
                for source_name in ["flowsim", "ns3", "m4"]:
                    errors_data = result.get("errors", {}).get(source_name)
                    if not errors_data:
                        continue
                    
                    # We need to get the actual remote values, not just errors
                    # Let's store them in the result during processing
                    remote_vals = result.get("remote_vals", {}).get(source_name, {}).get(key)
                    if remote_vals and len(remote_vals) > 0:
                        plot_data.append(remote_vals)
                        labels.append(source_name)
                        colors_list.append(source_colors[source_name])
                
                if plot_data:
                    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
                    for patch, color in zip(bp["boxes"], colors_list):
                        patch.set_facecolor(color)
                    
                    ax.set_title(f"Client {client}, {op_type}")
                    ax.set_ylabel("Duration (ns)")
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_visible(False)
        
        plt.suptitle(f"Duration Distributions - {scenario} (topology {topology_label})", 
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        safe_scenario = scenario.replace("/", "_").replace("\\", "_")
        plt.savefig(output_dir / f"box_{safe_scenario}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved box plot: box_{safe_scenario}.png")


def process_topology(sweep_dir: str, local_base: Path, topology_label: str):
    """Process entire topology in parallel."""
    if not local_base.is_dir():
        print(f"[skip] {local_base} not found")
        return
    
    output_dir = SCRIPT_DIR / "sweepfigures" / sweep_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n===== Processing topology {topology_label} ({sweep_dir}) =====")
    
    # Collect scenarios
    scenarios = [(p, sweep_dir, SOURCES) for p in sorted(local_base.iterdir()) if p.is_dir()]
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_scenario, args): args for args in scenarios}
        for future in as_completed(futures):
            results.append(future.result())
    
    # Aggregate results
    all_errors = defaultdict(list)
    e2e_errors = defaultdict(list)
    
    for result in sorted(results, key=lambda r: r.get("scenario", "")):
        if result.get("skip"):
            continue
        
        scenario = result["scenario"]
        print(f"\n=== {scenario} ===")
        
        # Print E2E for "_1" scenarios
        if True: #scenario.endswith("_1") and result.get("local_e2e"):
            _, _, local_dur = result["local_e2e"]
            local_dur_scaled = local_dur * LOCAL_E2E_SCALE
            print(f"  Local E2E: {local_dur_scaled:.0f} ns ({local_dur_scaled/1e6:.3f} ms)")
            
            for source_name, data in result.get("errors", {}).items():
                if data.get("remote_e2e"):
                    _, _, remote_dur = data["remote_e2e"]
                    print(f"  {source_name} E2E: {remote_dur:.0f} ns ({remote_dur/1e6:.3f} ms)")
                    
                    # Compute E2E error
                    if local_dur_scaled > 0:
                        e2e_err = abs(remote_dur - local_dur_scaled) / local_dur_scaled
                        e2e_errors[source_name].append(e2e_err)
                        print(f"    E2E error: {e2e_err:.4f}")
        
        # Per-request errors
        for source_name, data in result.get("errors", {}).items():
            errors = data["per_request"]
            if len(errors) > 0:
                all_errors[source_name].extend(errors)
                print(f"  {source_name}: median={np.median(errors):.4f}, n={len(errors)}")
    
    # Print summary
    print("\n=== Overall Results ===")
    for name in all_errors:
        errs = np.array(all_errors[name])
        print(f"{name}: median={np.median(errs):.4f}, mean={np.mean(errs):.4f}, n={len(errs)}")
    
    print("\n=== E2E Error Summary ===")
    for name in e2e_errors:
        errs = np.array(e2e_errors[name])
        print(f"{name}: median={np.median(errs):.4f}, mean={np.mean(errs):.4f}, n={len(errs)}")
    
    # Generate plots
    plot_cdf(
        {k: np.array(v) for k, v in all_errors.items()},
        output_dir / "relative_errors.png",
        f"CDF of Relative Errors (topology {topology_label})"
    )
    
    plot_e2e_times(results, output_dir / "e2e_times.png", topology_label)
    
    # Generate per-scenario box plots
    #print("\n=== Generating box plots ===")
    plot_scenario_box_plots(results, output_dir, topology_label)
    
    # Save data
    for name, errs in all_errors.items():
        np.save(output_dir / f"errors_{name}.npy", np.array(errs))
    
    print(f"\nSaved to {output_dir}/")


def main():
    for sweep_dir, local_base, label in TOPOLOGIES:
        if local_base.is_dir():
            process_topology(sweep_dir, local_base, label)


if __name__ == "__main__":
    main()