#!/usr/bin/env python3
"""
M4 Network Simulation Results Analyzer

One script to analyze and plot results from all three network backends:
- M4 (ML-based network simulator)
- FlowSim (Flow-level simulator) 
- NS3/UNISON (Packet-level simulator)

Usage:
    python analyze.py                    # Analyze all scenarios, generate all plots
    python analyze.py --scenario 100_2  # Analyze one specific scenario
    python analyze.py --no-plots        # Just show summary statistics
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Parse simulation output lines: [ud|rdma] client=X id=Y dur_ns=Z
LINE_RE = re.compile(r"\[(ud|rdma)\] client=(\d+) id=\d+(?:-\d+)? dur_ns=(\d+)")

# All 11-client experiment scenarios  
ALL_SCENARIOS = [
    # "100_2", "100_4", 
    "250_1", "250_2", "250_4", 
    "300_1", "300_2", "300_4",
    "400_1", "400_2", "400_4", 
    "500_1", "500_2", "500_4", 
    "650_1", "650_2", "650_4", 
    "750_1", "750_2", "750_4", 
    "900_1", "900_2", "900_4",
    "1000_1", "1000_2", "1000_4"
    # "250_1", "250_2",
    # "300_1", "300_2", 
    # "400_1", "400_2", 
    # "500_1", "500_2", 
    # "650_1", "650_2", 
    # "750_1", "750_2", 
    # "900_1", "900_2", 
    # "1000_1", "1000_2", 
]

# Quick test scenarios (matches run.py --quick)
QUICK_SCENARIOS = [
    "250_1", "250_4",    # 250KB RDMA, window 1 & 4
    "1000_1", "1000_4"   # 1000KB RDMA, window 1 & 4
]

# Plot styling constants (matching notebook style exactly)
OURS_LABEL = "m4"  # Label for M4 in plots (following paper convention)
PLOT_COLORS = {"real_world": "black", "flowsim": "orange", "ns3": "crimson", "m4": "cornflowerblue"}
PLOT_MARKERS = {"real_world": "D", "flowsim": "^", "ns3": "o", "m4": "X"}  # Testbed=diamond, flowSim=triangle, UNISON=circle, FLS=X
PLOT_LABELS = {"real_world": "Testbed", "flowsim": "flowSim", "ns3": "UNISON", "m4": OURS_LABEL}
PERFLOW_COLORS = ["orange", "blueviolet", "cornflowerblue"]  # flowSim, ns3, FLS
PERFLOW_LABELS = ["flowSim", "ns3", OURS_LABEL]
def load_data(file_path: Path, trim: int = 0) -> Tuple[Dict[Tuple[int, str], List[int]], List[Tuple[int, str, int]]]:
    """Load experiment data from a simulation output file.
    
    Returns:
        (data_dict, trimmed_entries): Dictionary of flows and list of trimmed entries with timestamps
    """
    if not file_path.exists():
        return {}, []
    
    ordered_entries: List[Tuple[int, str, int]] = []
    with file_path.open("r") as f:
        for line in f:
            match = LINE_RE.match(line.strip())
            if match:
                op_type, client_str, dur_str = match.groups()
                ordered_entries.append((int(client_str), op_type, int(dur_str)))
    
    # Apply trim: remove first and last 'trim' entries
    trimmed_entries = ordered_entries
    if trim > 0 and len(ordered_entries) > 2 * trim:
        trimmed_entries = ordered_entries[trim:-trim]
    
    results = defaultdict(list)
    for client_id, phase, duration in trimmed_entries:
        results[(client_id, phase)].append(duration)
    
    return dict(results), trimmed_entries


def compute_end2end_times(data: Dict) -> List[float]:
    """Compute end-to-end application times (UD + RDMA phases combined)."""
    # Group flows by client to match UD and RDMA phases
    client_flows = defaultdict(lambda: {"ud": [], "rdma": []})
    
    for key, values in data.items():
        client_id, phase = key
        client_flows[client_id][phase].extend(values)
    
    end2end_times = []
    for client_id, phases in client_flows.items():
        # With corrected flow pairing in load_data, UD and RDMA should now be properly aligned
        min_len = min(len(phases["ud"]), len(phases["rdma"]))
        for i in range(min_len):
            # End-to-end = UD phase + RDMA phase  
            total_ns = phases["ud"][i] + phases["rdma"][i]
            end2end_times.append(total_ns / 1000.0)  # Convert to microseconds
    
    return end2end_times


def compute_relative_errors(real_values: List[float], sim_values: List[float]) -> np.ndarray:
    """
    Compute simple relative errors between real-world and simulated values.
    Formula: |real - sim| / real (same as original analyze.py)
    """
    if not real_values or not sim_values:
        return np.array([])
    
    # Match lengths
    min_len = min(len(real_values), len(sim_values))
    real_arr = np.array(real_values[:min_len])
    sim_arr = np.array(sim_values[:min_len])
    
    # Avoid division by zero
    mask = real_arr != 0
    if not np.any(mask):
        return np.array([])
    
    # Simple relative error: |real - sim| / real
    return np.abs(real_arr[mask] - sim_arr[mask]) / real_arr[mask]


def compute_signed_relative_errors(real_values: List[float], sim_values: List[float]) -> np.ndarray:
    """
    Compute signed relative errors between real-world and simulated values.
    Formula: (sim - real) / real  (preserve sign to see under/over-estimation)
    """
    if not real_values or not sim_values:
        return np.array([])
    
    min_len = min(len(real_values), len(sim_values))
    real_arr = np.array(real_values[:min_len])
    sim_arr = np.array(sim_values[:min_len])
    
    mask = real_arr != 0
    if not np.any(mask):
        return np.array([])
    
    return (sim_arr[mask] - real_arr[mask]) / real_arr[mask]


def flatten_phase_series(data: Dict[Tuple[int, str], List[int]], phase: str) -> List[int]:
    """Aggregate per-client series for a specific phase."""
    series: List[int] = []
    for (client_id, op_type) in sorted(data.keys()):
        if op_type != phase:
            continue
        series.extend(data[(client_id, op_type)])
    return series


def compute_e2e_duration_from_logs(scenario_dir: Path, backend: str = None, trim: int = 500) -> Optional[int]:
    """Compute true end-to-end application completion time from timestamp logs with trimming support."""
    timestamps = []
    
    # For M4 backend, use flows.txt which has ACTUAL timestamps (includes server delay)
    # Client logs have PREDICTED timestamps which we use for per-flow FCT metrics only
    if backend == "m4":
        flows_txt = scenario_dir / "flows.txt"
        if flows_txt.exists():
            try:
                with flows_txt.open("r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            # Format: op_index client_id worker_id slot req_bytes resp_bytes start_ns end_ns fct_ns stage
                            start_ns = int(parts[6])
                            end_ns = int(parts[7])
                            timestamps.append(start_ns)
                            timestamps.append(end_ns)
            except Exception:
                pass
    
    # For real world data, try flows_debug.txt
    if not timestamps and (backend is None or backend == "real_world"):
        flows_debug = scenario_dir / "flows_debug.txt"
        if flows_debug.exists():
            import re
            ts_re = re.compile(r"ts_ns=(\d+)")
            try:
                with flows_debug.open("r") as f:
                    for line in f:
                        if match := ts_re.search(line):
                            timestamps.append(int(match.group(1)))
            except Exception:
                pass
    
    # For all other backends, try grouped_flows.txt with t=X ns format
    if not timestamps:
        grouped_flows = scenario_dir / "grouped_flows.txt"
        if grouped_flows.exists():
            import re
            t_re = re.compile(r"t=(\d+)\s+ns")
            try:
                with grouped_flows.open("r") as f:
                    for line in f:
                        if match := t_re.search(line):
                            timestamps.append(int(match.group(1)))
            except Exception:
                pass
    
    if not timestamps:
        return None
    
    timestamps.sort()
    
    # Apply trim: remove first and last 'trim' timestamps (same as flow trim)
    if trim > 0 and len(timestamps) > 2 * trim:
        timestamps = timestamps[trim:-trim]
    
    if not timestamps:
        return None
    
    return timestamps[-1] - timestamps[0]  # End-to-end duration after trim


def analyze_scenario(scenario: str, base_dir: Path = None) -> Dict:
    """Analyze one scenario and return data for aggregation."""
    
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    # File paths for each backend in eval_test structure
    files = {
        "real_world": base_dir / "eval_test" / "testbed" / scenario / "real_world.txt",
        "flowsim": base_dir / "eval_test" / "flowsim" / scenario / "flowsim_output.txt",
        "ns3": base_dir / "eval_test" / "ns3" / scenario / "ns3_output.txt",
        "m4": base_dir / "eval_test" / "m4" / scenario / "m4_output.txt"
    }
    
    # Directory paths for timestamp computation
    dirs = {
        "real_world": base_dir / "eval_test" / "testbed" / scenario,
        "flowsim": base_dir / "eval_test" / "flowsim" / scenario,
        "ns3": base_dir / "eval_test" / "ns3" / scenario,
        "m4": base_dir / "eval_test" / "m4" / scenario
    }
    
    # Load data from all backends with symmetric trimming
    TRIM_FLOWS = 50
    all_data = {}
    for name, file_path in files.items():
        if file_path.exists():
            data, _ = load_data(file_path, trim=TRIM_FLOWS)
            if data:
                all_data[name] = data
    
    if "real_world" not in all_data:
        return None
    
    # Compute end-to-end times for each backend
    scenario_results = {"scenario": scenario}
    
    for backend, data in all_data.items():
        # Per-flow times (for individual flow analysis)
        end2end_times = compute_end2end_times(data)
        
        # Compute application completion time from TRIMMED data (respects trim)
        app_completion_time = compute_e2e_duration_from_logs(dirs[backend], backend, trim=TRIM_FLOWS)
        
        if end2end_times:
            scenario_results[backend] = {
                "end2end_times": end2end_times,  # Per-flow times (for per-flow error analysis)
                "median": np.median(end2end_times),
                "p90": np.percentile(end2end_times, 90),
                "count": len(end2end_times),
                "app_completion_time": app_completion_time,  # True scenario completion time (ns)
                "app_completion_time_raw": app_completion_time,
            }
    
    # Compute simple relative errors vs real-world (like original analyze.py)
    if "real_world" in scenario_results:
        real_times = scenario_results["real_world"]["end2end_times"]
        real_ud_series = flatten_phase_series(all_data["real_world"], "ud")
        real_rdma_series = flatten_phase_series(all_data["real_world"], "rdma")
        
        for backend in ["m4", "flowsim", "ns3"]:
            if backend in scenario_results:
                sim_times = scenario_results[backend]["end2end_times"]
                
                # Compute simple relative errors: |real - sim| / real
                relative_errors = compute_relative_errors(real_times, sim_times)
                signed_errors = compute_signed_relative_errors(real_times, sim_times)
                ud_signed = compute_signed_relative_errors(
                    real_ud_series,
                    flatten_phase_series(all_data[backend], "ud"),
                )
                rdma_signed = compute_signed_relative_errors(
                    real_rdma_series,
                    flatten_phase_series(all_data[backend], "rdma"),
                )
                
                if len(relative_errors) > 0:
                    scenario_results[backend]["relative_errors"] = relative_errors
                    scenario_results[backend]["median_error"] = np.median(relative_errors)
                else:
                    scenario_results[backend]["relative_errors"] = np.array([])
                    scenario_results[backend]["median_error"] = float('inf')
                
                if len(signed_errors) > 0:
                    scenario_results[backend]["signed_errors"] = signed_errors
                else:
                    scenario_results[backend]["signed_errors"] = np.array([])
                
                scenario_results[backend]["signed_errors_by_phase"] = {
                    "ud": ud_signed if len(ud_signed) > 0 else np.array([]),
                    "rdma": rdma_signed if len(rdma_signed) > 0 else np.array([]),
                }
    
    return scenario_results


def extract_packet_size(scenario: str) -> int:
    """Extract packet size in KB from scenario name (e.g., '100_2' -> 100)."""
    return int(scenario.split('_')[0])


def extract_window_size(scenario: str) -> int:
    """Extract window size from scenario name (e.g., '100_2' -> 2)."""
    return int(scenario.split('_')[1])


def generate_overall_plots_by_window_size(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate overall plot for window size 2 only."""
    
    # Group results by window size
    results_by_window = {"2": []}
    
    for result in all_scenario_results:
        if not result:
            continue
        window_size = str(extract_window_size(result["scenario"]))
        if window_size == "2":
            results_by_window["2"].append(result)
    
    # Create plot for window size 2 only
    for window_size, scenario_results in results_by_window.items():
        if not scenario_results:
            continue
            
        plt.figure(figsize=(7, 4.5))
        
        # Collect data points by backend for this window size
        backend_data = {"real_world": [], "flowsim": [], "ns3": [], "m4": []}
        
        for result in scenario_results:
            packet_size = extract_packet_size(result["scenario"])
            
            for backend in ["real_world", "flowsim", "ns3", "m4"]:
                if backend in result and "app_completion_time" in result[backend]:
                    # Use application completion time (scenario duration), not median per-flow
                    app_time_s = result[backend]["app_completion_time"] / 1e9  # ns -> s
                    backend_data[backend].append((packet_size, app_time_s * 1000))  # s -> ms
        
        # Plot each backend using global styling constants
        for backend in ["real_world", "flowsim", "ns3", "m4"]:
            data_points = backend_data[backend]
            if not data_points:
                continue
                
            xs, ys = zip(*data_points)
            plt.plot(xs, ys, marker=PLOT_MARKERS[backend], label=PLOT_LABELS[backend],
                    color=PLOT_COLORS[backend], markersize=8, linestyle='None')
        
        # Apply notebook styling
        plt.xlabel("Size of Data Packets (KB)", fontsize=15)
        plt.ylabel("Application Completion\nTime (ms)", fontsize=15)
        # plt.title(f"(a) Application completion time vs. size of data packets")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # Save with window size suffix
        filename = f'm4-testbed-overall-window{window_size}.png'
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📁 Saved: {results_dir / filename}")


def generate_overall_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate combined overall plot and separate plots by window size."""
    
    # Generate separate plots by window size (more detailed analysis)
    generate_overall_plots_by_window_size(all_scenario_results, results_dir)
    
    # Also generate combined plot for comparison
    plt.figure(figsize=(10, 6))
    
    # Collect data points by backend (all window sizes combined)
    backend_data = {"real_world": [], "flowsim": [], "ns3": [], "m4": []}
    
    for result in all_scenario_results:
        if not result:
            continue
            
        packet_size = extract_packet_size(result["scenario"])
        
        for backend in ["real_world", "flowsim", "ns3", "m4"]:
            if backend in result and "app_completion_time" in result[backend]:
                # Use application completion time (scenario duration), not median per-flow
                app_time_s = result[backend]["app_completion_time"] / 1e9  # ns -> s
                backend_data[backend].append((packet_size, app_time_s * 1000))  # s -> ms
    
    # Plot each backend using global styling constants
    for backend in ["real_world", "flowsim", "ns3", "m4"]:
        data_points = backend_data[backend]
        if not data_points:
            continue
            
        xs, ys = zip(*data_points)
        plt.plot(xs, ys, marker=PLOT_MARKERS[backend], label=PLOT_LABELS[backend],
                linewidth=2, color=PLOT_COLORS[backend], markersize=8, linestyle='None')
    
    # Apply exact original styling
    plt.xlabel("Size of Data Packets (KB)")
    plt.ylabel("Application Completion\nTime (ms)")
    plt.title("(a) Application completion time vs. size of data packets\n(All Window Sizes Combined)")
    plt.grid(True, linestyle="--", alpha=0.6)  # Exact original grid style
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'm4-testbed-overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📁 Saved: {results_dir / 'm4-testbed-overall.png'}")


def generate_perflow_by_window_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate per-flow CDFs separated by window size."""
    
    # Group errors by window size
    for window_size in [1, 2, 4]:
        plt.figure(figsize=(8, 6))
        
        # Collect relative errors for this window size
        window_errors = {"m4": [], "flowsim": [], "ns3": []}
        
        for result in all_scenario_results:
            if not result:
                continue
            result_window = extract_window_size(result["scenario"])
            if result_window != window_size:
                continue
                
            for backend in ["m4", "flowsim", "ns3"]:
                if backend in result and "relative_errors" in result[backend]:
                    window_errors[backend].extend(result[backend]["relative_errors"])
        
        # Use global styling constants
        backends = ["flowsim", "ns3", "m4"]
        
        # Plot each backend as CDF
        for i, backend in enumerate(backends):
            if backend in window_errors and window_errors[backend]:
                errors = np.array(window_errors[backend])
                finite_errors = errors[np.isfinite(errors)]
                
                if len(finite_errors) == 0:
                    continue
                    
                # Sort and create CDF
                arr = np.sort(finite_errors)
                y = np.linspace(0, 1, len(arr), endpoint=False)
                
                # Convert to percentage
                arr_pct = arr * 100
                y_pct = y * 100
                
                plt.step(arr_pct, y_pct, where="post", label=PERFLOW_LABELS[i], 
                        linewidth=2, color=PERFLOW_COLORS[i])
        
        # Apply styling
        plt.xlabel("Magnitude of relative estimation error\nfor per-flow FCT slowdown (%)", fontsize=15)
        plt.ylabel("CDF (%)", fontsize=15)
        plt.title(f"(b) CDF of per-flow FCT slowdown errors\n(Window Size: {window_size})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=18, loc=4)
        plt.xlim(0.5, 1000)
        plt.xscale('log')
        plt.tight_layout()
        
        filename = f'm4-testbed-perflow-window{window_size}.png'
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📁 Saved: {results_dir / filename}")


def generate_perflow_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate m4-testbed-perflow.png - Per-flow CDF plot using exact original styling."""
    
    plt.figure(figsize=(7, 4.5))
    
    # Collect all relative errors across scenarios (simple metric)
    all_relative_errors = {"m4": [], "flowsim": [], "ns3": []}
    
    for result in all_scenario_results:
        if not result:
            continue
        for backend in ["m4", "flowsim", "ns3"]:
            if backend in result and "relative_errors" in result[backend]:
                all_relative_errors[backend].extend(result[backend]["relative_errors"])
    
    # Use global styling constants
    backends = ["flowsim", "ns3", "m4"]
    
    # Plot each backend as CDF
    for i, backend in enumerate(backends):
        if backend in all_relative_errors and all_relative_errors[backend]:
            errors = np.array(all_relative_errors[backend])
            # Remove infinite values
            finite_errors = errors[np.isfinite(errors)]
            
            if len(finite_errors) == 0:
                continue
                
            # Sort and create CDF
            arr = np.sort(finite_errors)
            y = np.linspace(0, 1, len(arr), endpoint=False)
            
            # Convert to percentage for both axes
            arr_pct = arr * 100  # Convert to percentage
            y_pct = y * 100      # Convert to percentage
            
            # Plot with step function
            plt.step(arr_pct, y_pct, where="post", label=PERFLOW_LABELS[i], 
                    linewidth=2, color=PERFLOW_COLORS[i])
    
    # Apply notebook styling
    plt.xlabel("Magnitude of relative estimation error\nfor per-flow FCT slowdown (%)", fontsize=15)
    plt.ylabel("CDF (%)", fontsize=15)  
    plt.title("(b) CDF of per-flow FCT slowdown errors")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18, loc=4)  # loc=4 is lower right
    plt.tight_layout()
    plt.xlim(0.5, 1000)
    plt.xscale('log')
    plt.savefig(results_dir / 'm4-testbed-perflow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📁 Saved: {results_dir / 'm4-testbed-perflow.png'}")


def generate_perflow_signed_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate CDF plot of signed per-flow relative errors (retaining bias direction)."""
    
    plt.figure(figsize=(8, 6))
    
    all_signed_errors = {"m4": [], "flowsim": [], "ns3": []}
    
    for result in all_scenario_results:
        if not result:
            continue
        for backend in ["m4", "flowsim", "ns3"]:
            signed = result.get(backend, {}).get("signed_errors")
            if signed is not None and len(signed) > 0 and np.all(np.isfinite(signed)):
                all_signed_errors[backend].extend(signed)
    
    backends = ["flowsim", "ns3", "m4"]
    x_min, x_max = None, None
    
    for i, backend in enumerate(backends):
        errors = all_signed_errors.get(backend, [])
        if not errors:
            continue
        
        arr = np.sort(np.array(errors) * 100.0)  # Convert to percentage
        if len(arr) == 0:
            continue
        y = np.linspace(0, 1, len(arr), endpoint=False) * 100.0
        
        x_min = arr[0] if x_min is None else min(x_min, arr[0])
        x_max = arr[-1] if x_max is None else max(x_max, arr[-1])
        
        plt.step(arr, y, where="post", label=PERFLOW_LABELS[i],
                 linewidth=2, color=PERFLOW_COLORS[i])
    
    plt.xlabel("Signed relative estimation error for per-flow FCT slowdown (%)", fontsize=15)
    plt.ylabel("CDF (%)", fontsize=15)
    plt.title("(b) CDF of per-flow slowdown errors (signed)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=18, loc=4)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    
    if x_min is not None and x_max is not None:
        span = x_max - x_min
        if span <= 0:
            span = max(abs(x_min), abs(x_max))
            x_min, x_max = -span, span
        margin = max(5.0, span * 0.1)
        center = 0.5 * (x_min + x_max)
        half_span = 0.5 * span + margin
        x_min = center - half_span
        x_max = center + half_span
        if x_min >= x_max:
            half_span = max(abs(x_min), abs(x_max), 10.0)
            x_min, x_max = -half_span, half_span
        plt.xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'm4-testbed-perflow-signed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📁 Saved: {results_dir / 'm4-testbed-perflow-signed.png'}")


def generate_perflow_signed_by_window_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate signed per-flow CDF plots separated by window size."""
    
    for window_size in [1, 2, 4]:
        plt.figure(figsize=(8, 6))
        
        window_signed = {"m4": [], "flowsim": [], "ns3": []}
        
        for result in all_scenario_results:
            if not result:
                continue
            if extract_window_size(result["scenario"]) != window_size:
                continue
            
            for backend in ["m4", "flowsim", "ns3"]:
                signed = result.get(backend, {}).get("signed_errors")
                if signed is not None and len(signed) > 0 and np.all(np.isfinite(signed)):
                    window_signed[backend].extend(signed)
        
        x_min, x_max = None, None
        backends = ["flowsim", "ns3", "m4"]
        for i, backend in enumerate(backends):
            errors = window_signed.get(backend, [])
            if not errors:
                continue
            
            arr = np.sort(np.array(errors) * 100.0)
            if len(arr) == 0:
                continue
            y = np.linspace(0, 1, len(arr), endpoint=False) * 100.0
            
            x_min = arr[0] if x_min is None else min(x_min, arr[0])
            x_max = arr[-1] if x_max is None else max(x_max, arr[-1])
            
            plt.step(arr, y, where="post", label=PERFLOW_LABELS[i],
                     linewidth=2, color=PERFLOW_COLORS[i])
        
        plt.xlabel("Signed relative estimation error for per-flow FCT slowdown (%)", fontsize=15)
        plt.ylabel("CDF (%)", fontsize=15)
        plt.title(f"(b) CDF of per-flow slowdown errors (signed)\n(Window Size: {window_size})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=18, loc=4)
        plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xlim(-100, 100)
        
        plt.tight_layout()
        filename = f'm4-testbed-perflow-signed-window{window_size}.png'
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📁 Saved: {results_dir / filename}")


def generate_perflow_signed_phase_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate signed per-flow CDF plots separated by flow type (UD vs RDMA)."""
    
    phase_labels = {"ud": "UD flows", "rdma": "RDMA flows"}
    
    for phase in ["ud", "rdma"]:
        plt.figure(figsize=(8, 6))
        
        phase_signed = {"m4": [], "flowsim": [], "ns3": []}
        
        for result in all_scenario_results:
            if not result:
                continue
            for backend in ["m4", "flowsim", "ns3"]:
                signed_map = result.get(backend, {}).get("signed_errors_by_phase", {})
                arr = signed_map.get(phase)
                if arr is not None and len(arr) > 0 and np.all(np.isfinite(arr)):
                    phase_signed[backend].extend(arr)
        
        backends = ["flowsim", "ns3", "m4"]
        for i, backend in enumerate(backends):
            errors = phase_signed.get(backend, [])
            if not errors:
                continue
            
            arr = np.sort(np.array(errors) * 100.0)
            if len(arr) == 0:
                continue
            y = np.linspace(0, 1, len(arr), endpoint=False) * 100.0
            
            plt.step(arr, y, where="post", label=PERFLOW_LABELS[i],
                     linewidth=2, color=PERFLOW_COLORS[i])
        
        plt.xlabel("Signed relative estimation error for per-flow FCT slowdown (%)", fontsize=15)
        plt.ylabel("CDF (%)", fontsize=15)
        plt.title(f"(b) CDF of per-flow slowdown errors (signed)\n{phase_labels[phase]}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=18, loc=4)
        plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xlim(-100, 100)
        plt.tight_layout()
        
        filename = f'm4-testbed-perflow-signed-{phase}.png'
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📁 Saved: {results_dir / filename}")


def generate_plots(all_scenario_results: List[Dict], base_dir: Path = None) -> None:
    """Generate figures: window2 overall plot and per-flow accuracy plot only"""
    
    if base_dir is None:
        base_dir = Path(__file__).parent
        
    results_dir = base_dir / "results" 
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Generating figures...")
    
    # Generate only window2 overall plot and per-flow plot
    generate_overall_plots_by_window_size(all_scenario_results, results_dir)
    generate_perflow_plot(all_scenario_results, results_dir)
    
    # Generate summary statistics
    with open(results_dir / 'accuracy_summary.txt', 'w') as f:
        f.write("M4 Network Simulation Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GENERATED FIGURES:\n")
        f.write("1. m4-testbed-overall-window2.png - Application completion time (window size 2)\n")
        f.write("2. m4-testbed-perflow.png - Per-flow FCT accuracy (CDF of relative errors)\n\n")
        
        # Collect all relative errors for summary
        all_relative_errors_summary = {"m4": [], "flowsim": [], "ns3": []}
        for result in all_scenario_results:
            if not result:
                continue
            for backend in ["m4", "flowsim", "ns3"]:
                if backend in result and "relative_errors" in result[backend]:
                    all_relative_errors_summary[backend].extend(result[backend]["relative_errors"])
        
        # Show scenario coverage
        f.write(f"SCENARIO COVERAGE:\n")
        packet_sizes = set()
        for result in all_scenario_results:
            if result:
                packet_sizes.add(extract_packet_size(result["scenario"]))
        f.write(f"  Packet sizes: {sorted(packet_sizes)} KB\n")
        f.write(f"  Total scenarios: {len(all_scenario_results)}\n\n")
        
        # Relative error statistics
        f.write("RELATIVE ESTIMATION ERRORS (as percentage):\n")
        cdf_labels = {'m4': OURS_LABEL, 'ns3': 'ns3', 'flowsim': 'flowSim'}
        accuracy_ranking = []
        
        for backend in ["m4", "ns3", "flowsim"]:
            if all_relative_errors_summary[backend]:
                errors_pct = np.array(all_relative_errors_summary[backend]) * 100
                median_error = np.median(errors_pct)
                p90_error = np.percentile(errors_pct, 90)
                accuracy_ranking.append((cdf_labels[backend], median_error))
                
                f.write(f"  {cdf_labels[backend]:8}: median: {median_error:.1f}%, "
                       f"p90: {p90_error:.1f}%, "
                       f"samples: {len(errors_pct):6}\n")
        
        # Show accuracy ranking
        accuracy_ranking.sort(key=lambda x: x[1])  # Sort by median error (lower = better)
        f.write(f"\nACCURACY RANKING (best to worst):\n")
        for i, (backend, error) in enumerate(accuracy_ranking, 1):
            f.write(f"  {i}. {backend} (median error: {error:.1f}%)\n")
    
    print(f"  📁 Saved: {results_dir / 'accuracy_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze M4 network simulation results")
    parser.add_argument("--scenario", help="Analyze specific scenario (e.g., '100_2')")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="Analyze only quick test scenarios (100_1, 100_4, 1000_1, 1000_4)")
    args = parser.parse_args()
    
    print("🚀 M4 Network Simulation Results Analyzer")
    print("=" * 50)
    
    if args.scenario:
        if args.scenario not in ALL_SCENARIOS and args.scenario not in QUICK_SCENARIOS:
            print(f"❌ Unknown scenario: {args.scenario}")
            print(f"Available: {ALL_SCENARIOS + QUICK_SCENARIOS}")
            return
        scenarios = [args.scenario]
    elif args.quick:
        scenarios = QUICK_SCENARIOS
        print("⚡ Quick test mode: analyzing 4 scenarios")
    else:
        scenarios = ALL_SCENARIOS
    
    print(f"📊 Analyzing {len(scenarios)} scenario(s)")
    
    # Analyze each scenario and collect results
    all_scenario_results = []
    for scenario in scenarios:
        print(f"Processing {scenario}...", end=" ")
        result = analyze_scenario(scenario)
        if result:
            all_scenario_results.append(result)
            print(f"✓ ({result.get('real_world', {}).get('count', 0)} flows)")
        else:
            print("✗ (no data)")
    
    if not all_scenario_results:
        print("❌ No data found for any scenarios")
        return
    
    print(f"\n📈 Summary across {len(all_scenario_results)} scenarios:")
    
    # Collect scenario-level end-to-end time errors and per-flow errors separately
    backend_stats = {
        "m4": {
            "end2end_errors": [],
            "perflow_errors": [],
            "perflow_signed_phases": {"ud": [], "rdma": []},
            "app_pairs": [],
        },
        "flowsim": {
            "end2end_errors": [],
            "perflow_errors": [],
            "perflow_signed_phases": {"ud": [], "rdma": []},
        }, 
        "ns3": {
            "end2end_errors": [],
            "perflow_errors": [],
            "perflow_signed_phases": {"ud": [], "rdma": []},
        }
    }
    
    for result in all_scenario_results:
        # End-to-end errors: true application completion time comparison
        if "real_world" in result and "app_completion_time" in result["real_world"]:
            real_app_time = result["real_world"]["app_completion_time"]
            
            for backend in backend_stats.keys():
                if backend in result and "app_completion_time" in result[backend]:
                    sim_app_time = result[backend]["app_completion_time"]
                    # End-to-end error: |real_app_time - sim_app_time| / real_app_time
                    if real_app_time and sim_app_time and real_app_time > 0:
                        e2e_error = abs(real_app_time - sim_app_time) / real_app_time
                        backend_stats[backend]["end2end_errors"].append(e2e_error)
                        if backend == "m4":
                            backend_stats[backend]["app_pairs"].append(
                                (
                                    result["scenario"],
                                    real_app_time,
                                    sim_app_time,
                                    result[backend].get("app_completion_time_raw"),
                                )
                            )
        
        # Per-flow errors: individual flow completion time comparison
        for backend in backend_stats.keys():
            if backend in result and "relative_errors" in result[backend]:
                errors = result[backend]["relative_errors"]
                if len(errors) > 0 and np.all(np.isfinite(errors)):
                    backend_stats[backend]["perflow_errors"].extend(errors)
            if backend in result:
                signed_phase = result[backend].get("signed_errors_by_phase", {})
                for phase in ["ud", "rdma"]:
                    arr = signed_phase.get(phase)
                    if arr is not None and len(arr) > 0 and np.all(np.isfinite(arr)):
                        backend_stats[backend]["perflow_signed_phases"][phase].extend(arr)
    
    # Display comprehensive statistics
    print("\n📊 END-TO-END APPLICATION COMPLETION TIME ERRORS:")
    print("Backend   | Median Error | Mean Error   | Std Dev     | Count")
    print("----------|--------------|--------------|-------------|-------")
    
    e2e_accuracy_pairs = []
    for backend in ["m4", "flowsim", "ns3"]:
        errors = backend_stats[backend]["end2end_errors"]
        if errors:
            median_err = np.median(errors)
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            count = len(errors)
            e2e_accuracy_pairs.append((backend, median_err))
            print(f"{backend:9} | {median_err:11.1%} | {mean_err:11.1%} | {std_err:10.1%} | {count:5}")
        else:
            print(f"{backend:9} | {'N/A':11} | {'N/A':11} | {'N/A':10} | {'0':5}")
    
    print("\n📊 PER-FLOW FCT ESTIMATION ERRORS:")  
    print("Backend   | Median Error | Mean Error   | Std Dev     | Count")
    print("----------|--------------|--------------|-------------|-------")
    
    pf_accuracy_pairs = []
    for backend in ["m4", "flowsim", "ns3"]:
        errors = backend_stats[backend]["perflow_errors"] 
        if errors:
            median_err = np.median(errors)
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            count = len(errors)
            pf_accuracy_pairs.append((backend, median_err))
            print(f"{backend:9} | {median_err:11.1%} | {mean_err:11.1%} | {std_err:10.1%} | {count:5}")
        else:
            print(f"{backend:9} | {'N/A':11} | {'N/A':11} | {'N/A':10} | {'0':5}")
    
    # Show overall accuracy ranking
    print("\n🏆 OVERALL ACCURACY RANKING (by median end-to-end error):")
    e2e_accuracy_pairs.sort(key=lambda x: x[1])  # Sort by error (lower = better)
    
    for i, (backend, error) in enumerate(e2e_accuracy_pairs, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {i}. {status} {backend:8}: {error:.1%} median error")

    # Show application completion time comparison for M4 vs real world
    app_pairs = backend_stats["m4"].get("app_pairs", [])
    if app_pairs:
        print("\n📊 Application completion time (seconds):")
        print(f"{'scenario':10} {'real':>10} {'m4':>10}")
        for scenario, real_ns, m4_ns, _ in sorted(app_pairs):
            real_s = real_ns / 1e9 if real_ns else float("nan")
            m4_s = m4_ns / 1e9 if m4_ns else float("nan")
            print(f"{scenario:10} {real_s:10.3f} {m4_s:10.3f}")
    
    # Generate plots if requested
    if not args.no_plots:
        generate_plots(all_scenario_results)
    
    print(f"\n✅ Analysis complete!")
    if not args.no_plots:
        print(f"📁 Check results/ directory for key_analysis.png and accuracy_summary.txt")


if __name__ == "__main__":
    main()
