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

# Plot styling constants (define once, use everywhere)
OURS_LABEL = "FLS"  # Label for M4 in plots (following paper convention)
PLOT_COLORS = {"real_world": "black", "flowsim": "tab:blue", "ns3": "tab:orange", "m4": "tab:green"}
PLOT_MARKERS = {"real_world": "D", "flowsim": "o", "ns3": "s", "m4": "^"}
PLOT_LABELS = {"real_world": "Testbed", "flowsim": "flowSim", "ns3": "UNISON", "m4": OURS_LABEL}
PERFLOW_COLORS = ["orange", "blueviolet", "cornflowerblue"]  # flowSim, ns3, FLS
PERFLOW_LABELS = ["flowSim", "ns3", OURS_LABEL]
def load_data(file_path: Path, trim: int = 0) -> Dict[Tuple[int, str], List[int]]:
    """Load experiment data from a simulation output file."""
    results = defaultdict(list)
    
    if not file_path.exists():
        return dict(results)
    
    with open(file_path, 'r') as f:
        for line in f:
            match = LINE_RE.match(line.strip())
            if match:
                op_type, client_str, dur_str = match.groups()
                results[(int(client_str), op_type)].append(int(dur_str))
    
    # Trim warmup samples
    if trim > 0:
        for key in results:
            series = results[key]
            if len(series) > 2 * trim:
                results[key] = series[trim:-trim]
    
    return dict(results)


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


def compute_e2e_duration_from_logs(scenario_dir: Path, backend: str = None) -> Optional[int]:
    """Compute true end-to-end application completion time from timestamp logs."""
    timestamps = []
    
    # For real world data, try flows_debug.txt
    if backend is None or backend == "real_world":
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
    
    # For all backends, try grouped_flows.txt with t=X ns format
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
    return timestamps[-1] - timestamps[0]  # End-to-end duration


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
    
    # Load data from all backends
    all_data = {}
    for name, file_path in files.items():
        if file_path.exists():
            data = load_data(file_path)
            if data:
                all_data[name] = data
    
    if "real_world" not in all_data:
        return None
    
    # Compute end-to-end times for each backend
    scenario_results = {"scenario": scenario}
    
    for backend, data in all_data.items():
        # Per-flow times (for individual flow analysis)
        end2end_times = compute_end2end_times(data)
        
        # True application completion time (scenario-level)
        app_completion_time = compute_e2e_duration_from_logs(dirs[backend], backend)
        
        if end2end_times:
            scenario_results[backend] = {
                "end2end_times": end2end_times,  # Per-flow times (for per-flow error analysis)
                "median": np.median(end2end_times),
                "p90": np.percentile(end2end_times, 90),
                "count": len(end2end_times),
                "app_completion_time": app_completion_time  # True scenario completion time (ns)
            }
    
    # Compute simple relative errors vs real-world (like original analyze.py)
    if "real_world" in scenario_results:
        real_times = scenario_results["real_world"]["end2end_times"]
        
        for backend in ["m4", "flowsim", "ns3"]:
            if backend in scenario_results:
                sim_times = scenario_results[backend]["end2end_times"]
                
                # Compute simple relative errors: |real - sim| / real
                relative_errors = compute_relative_errors(real_times, sim_times)
                
                if len(relative_errors) > 0:
                    scenario_results[backend]["relative_errors"] = relative_errors
                    scenario_results[backend]["median_error"] = np.median(relative_errors)
                else:
                    scenario_results[backend]["relative_errors"] = np.array([])
                    scenario_results[backend]["median_error"] = float('inf')
    
    return scenario_results


def extract_packet_size(scenario: str) -> int:
    """Extract packet size in KB from scenario name (e.g., '100_2' -> 100)."""
    return int(scenario.split('_')[0])


def extract_window_size(scenario: str) -> int:
    """Extract window size from scenario name (e.g., '100_2' -> 2)."""
    return int(scenario.split('_')[1])


def generate_overall_plots_by_window_size(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate separate overall plots for each window size."""
    
    # Group results by window size
    results_by_window = {"1": [], "2": [], "4": []}
    
    for result in all_scenario_results:
        if not result:
            continue
        window_size = str(extract_window_size(result["scenario"]))
        if window_size in results_by_window:
            results_by_window[window_size].append(result)
    
    # Create separate plot for each window size
    for window_size, scenario_results in results_by_window.items():
        if not scenario_results:
            continue
            
        plt.figure(figsize=(10, 6))
        
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
                    linewidth=2, color=PLOT_COLORS[backend], markersize=8, linestyle='None')
        
        # Apply exact original styling with window size in title
        plt.xlabel("Size of Data Packets (KB)")
        plt.ylabel("Application Completion\nTime (ms)")
        plt.title(f"(a) Application completion time vs. size of data packets\n(Window Size: {window_size})")
        plt.grid(True, linestyle="--", alpha=0.6)  # Exact original grid style
        plt.legend()
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
        plt.tight_layout()
        
        filename = f'm4-testbed-perflow-window{window_size}.png'
        plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📁 Saved: {results_dir / filename}")


def generate_perflow_plot(all_scenario_results: List[Dict], results_dir: Path) -> None:
    """Generate m4-testbed-perflow.png - Per-flow CDF plot using exact original styling."""
    
    plt.figure(figsize=(8, 6))
    
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
    
    # Plot each backend as CDF using exact original style
    for i, backend in enumerate(backends):
        if backend in all_relative_errors and all_relative_errors[backend]:
            errors = np.array(all_relative_errors[backend])
            # Remove infinite values like in original
            finite_errors = errors[np.isfinite(errors)]
            
            if len(finite_errors) == 0:
                continue
                
            # Sort and create CDF exactly like original
            arr = np.sort(finite_errors)
            y = np.linspace(0, 1, len(arr), endpoint=False)
            
            # Convert to percentage for both axes to match original
            arr_pct = arr * 100  # Convert to percentage
            y_pct = y * 100      # Convert to percentage
            
            # Plot with step function exactly like original
            plt.step(arr_pct, y_pct, where="post", label=PERFLOW_LABELS[i], 
                    linewidth=2, color=PERFLOW_COLORS[i])
    
    # Apply exact original styling from make_plots.py
    plt.xlabel("Magnitude of relative estimation error\nfor per-flow FCT slowdown (%)", fontsize=15)
    plt.ylabel("CDF (%)", fontsize=15)  
    plt.title("(b) CDF of per-flow FCT slowdown errors")
    plt.grid(True, linestyle="--", alpha=0.6)  # Exact original grid style
    plt.legend(fontsize=18, loc=4)  # loc=4 is lower right, exact original style
    plt.tight_layout()
    plt.xlim(0, 1000)
    plt.savefig(results_dir / 'm4-testbed-perflow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📁 Saved: {results_dir / 'm4-testbed-perflow.png'}")


def generate_plots(all_scenario_results: List[Dict], base_dir: Path = None) -> None:
    """Generate figures: overall plots (combined + by window size) and per-flow accuracy"""
    
    if base_dir is None:
        base_dir = Path(__file__).parent
        
    results_dir = base_dir / "results" 
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Generating figures with flow separation...")
    
    # Generate overall plots (combined + separated by window size)
    generate_overall_plot(all_scenario_results, results_dir)
    generate_perflow_plot(all_scenario_results, results_dir)
    generate_perflow_by_window_plot(all_scenario_results, results_dir)
    
    # Generate summary statistics
    with open(results_dir / 'accuracy_summary.txt', 'w') as f:
        f.write("M4 Network Simulation Analysis - Matching Your Target Figures\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GENERATED FIGURES:\n")
        f.write("1. m4-testbed-overall.png - Overall results (all window sizes combined)\n")
        f.write("2. m4-testbed-overall-window1.png - Results for window size 1\n")
        f.write("3. m4-testbed-overall-window2.png - Results for window size 2\n") 
        f.write("4. m4-testbed-overall-window4.png - Results for window size 4\n")
        f.write("5. m4-testbed-perflow.png - Per-flow accuracy (CDF of estimation errors)\n\n")
        
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
    args = parser.parse_args()
    
    print("🚀 M4 Network Simulation Results Analyzer")
    print("=" * 50)
    
    if args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"❌ Unknown scenario: {args.scenario}")
            print(f"Available: {ALL_SCENARIOS}")
            return
        scenarios = [args.scenario]
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
        "m4": {"end2end_errors": [], "perflow_errors": []},
        "flowsim": {"end2end_errors": [], "perflow_errors": []}, 
        "ns3": {"end2end_errors": [], "perflow_errors": []}
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
        
        # Per-flow errors: individual flow completion time comparison
        for backend in backend_stats.keys():
            if backend in result and "relative_errors" in result[backend]:
                errors = result[backend]["relative_errors"]
                if len(errors) > 0 and np.all(np.isfinite(errors)):
                    backend_stats[backend]["perflow_errors"].extend(errors)
    
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
    
    # Generate plots if requested
    if not args.no_plots:
        generate_plots(all_scenario_results)
    
    print(f"\n✅ Analysis complete!")
    if not args.no_plots:
        print(f"📁 Check results/ directory for key_analysis.png and accuracy_summary.txt")


if __name__ == "__main__":
    main()
