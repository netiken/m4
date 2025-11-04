import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

line_re = re.compile(r"\[(ud|rdma)\] client=(\d+) id=\d+(?:-\d+)? dur_ns=(\d+)")

def parse_file(file_path):
    """Parse a log file and extract duration data."""
    results = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                m = line_re.match(line.strip())
                if not m:
                    continue
                op_type, client, dur = m.groups()
                key = (int(client), op_type)
                results.setdefault(key, []).append(int(dur))
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None
    
    return dict(sorted(results.items(), key=lambda x: x[0][0]))

def analyze_scenario_data(scenario_dir):
    """Analyze the data within a scenario directory."""
    real_world_file = os.path.join(scenario_dir, "real_world.txt")
    
    if not os.path.exists(real_world_file):
        print(f"❌ real_world.txt not found in {scenario_dir}")
        return None
    
    # Parse the real world data
    real_world_data = parse_file(real_world_file)
    if real_world_data is None:
        return None
    
    # Calculate statistics for each client and operation type
    all_durations = []
    client_stats = {}
    
    for (client, op_type), durations in real_world_data.items():
        durations = np.array(durations)
        all_durations.extend(durations)
        
        if client not in client_stats:
            client_stats[client] = {}
        
        client_stats[client][op_type] = {
            'count': len(durations),
            'mean': np.mean(durations),
            'median': np.median(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'p95': np.percentile(durations, 95),
            'p99': np.percentile(durations, 99)
        }
    
    all_durations = np.array(all_durations)
    
    return {
        'all_durations': all_durations,
        'client_stats': client_stats,
        'total_samples': len(all_durations),
        'num_clients': len(set(k[0] for k in real_world_data.keys())),
        'operation_types': list(set(k[1] for k in real_world_data.keys()))
    }

def create_duration_cdf_figure(scenario, data, output_dir):
    """Create a CDF plot of durations for a scenario."""
    if data is None:
        print(f"⚠️ No valid data for scenario {scenario}")
        return False
    
    plt.figure(figsize=(12, 8))
    
    all_durations = data['all_durations']
    client_stats = data['client_stats']
    
    # Create main CDF plot
    plt.subplot(2, 2, 1)
    sorted_durations = np.sort(all_durations)
    y = np.linspace(0, 1, len(sorted_durations), endpoint=False)
    plt.step(sorted_durations, y, where="post", linewidth=2, label=f"All Operations")
    plt.xlabel("Duration (ns)")
    plt.xscale("log")
    plt.ylabel("CDF")
    plt.title(f"Duration CDF - Scenario {scenario}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Create per-client plot
    plt.subplot(2, 2, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_stats)))
    for i, (client, stats) in enumerate(client_stats.items()):
        if 'ud' in stats and 'rdma' in stats:
            ud_durations = stats['ud']['mean']
            rdma_durations = stats['rdma']['mean']
            plt.scatter(ud_durations, rdma_durations, 
                       color=colors[i], label=f'Client {client}', s=50)
    
    plt.xlabel("UD Mean Duration (ns)")
    plt.ylabel("RDMA Mean Duration (ns)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("UD vs RDMA Durations by Client")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Create operation type comparison
    plt.subplot(2, 2, 3)
    op_types = data['operation_types']
    op_means = []
    op_medians = []
    op_stds = []
    
    for op_type in op_types:
        op_durations = []
        for client_stats in client_stats.values():
            if op_type in client_stats:
                # Get all durations for this op_type across all clients
                # We need to reconstruct this from the original data
                pass
    
    # For now, let's create a simple histogram
    plt.hist(all_durations, bins=50, alpha=0.7, density=True)
    plt.xlabel("Duration (ns)")
    plt.xscale("log")
    plt.ylabel("Density")
    plt.title("Duration Distribution")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Create statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calculate overall statistics
    overall_stats = {
        'Total Samples': data['total_samples'],
        'Number of Clients': data['num_clients'],
        'Mean Duration (ns)': f"{np.mean(all_durations):.2e}",
        'Median Duration (ns)': f"{np.median(all_durations):.2e}",
        'Std Duration (ns)': f"{np.std(all_durations):.2e}",
        'Min Duration (ns)': f"{np.min(all_durations):.2e}",
        'Max Duration (ns)': f"{np.max(all_durations):.2e}",
        'P95 Duration (ns)': f"{np.percentile(all_durations, 95):.2e}",
        'P99 Duration (ns)': f"{np.percentile(all_durations, 99):.2e}"
    }
    
    stats_text = "\n".join([f"{k}: {v}" for k, v in overall_stats.items()])
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f"{scenario}_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created analysis figure for {scenario}: {output_file}")
    print(f"   Total samples: {data['total_samples']}, Clients: {data['num_clients']}")
    print(f"   Mean duration: {np.mean(all_durations):.2e} ns, Median: {np.median(all_durations):.2e} ns")
    
    return True

def main():
    expirements_dir = "expirements"
    output_dir = "sweepfigures"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories in expirements
    scenarios = [d for d in os.listdir(expirements_dir) 
                 if os.path.isdir(os.path.join(expirements_dir, d))]
    
    print(f"Found {len(scenarios)} scenarios to process...")
    
    successful_figures = 0
    
    for scenario in sorted(scenarios):
        scenario_dir = os.path.join(expirements_dir, scenario)
        
        print(f"\n=== Processing Scenario: {scenario} ===")
        
        # Analyze the scenario data
        data = analyze_scenario_data(scenario_dir)
        
        if data is not None:
            if create_duration_cdf_figure(scenario, data, output_dir):
                successful_figures += 1
        else:
            print(f"❌ No valid data found for {scenario}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully created {successful_figures} figures out of {len(scenarios)} scenarios")
    print(f"Figures saved in: {output_dir}/")

if __name__ == "__main__":
    main()
