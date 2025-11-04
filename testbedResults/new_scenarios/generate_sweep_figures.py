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

def calculate_relative_errors(local_file, remote_file):
    """Calculate relative errors between local and remote files."""
    local_results = parse_file(local_file)
    remote_results = parse_file(remote_file)
    
    if local_results is None or remote_results is None:
        return None
    
    if local_results.keys() != remote_results.keys():
        print(f"⚠️ Key mismatch for {local_file}")
        return None

    remotes, locals = [], []
    for key in local_results.keys():
        if key in remote_results:
            # Take the same number of samples from both
            min_len = min(len(remote_results[key]), len(local_results[key]))
            if min_len > 500:  # Skip first 500 samples
                remote_vals = remote_results[key][500:min_len]
                local_vals = local_results[key][500:min_len]
                remotes.extend(remote_vals)
                locals.extend(local_vals)

    if not remotes or not locals:
        return None
        
    remotes = np.array(remotes)
    locals = np.array(locals)
    relative_error = np.abs(locals - remotes) / locals
    return relative_error

def create_error_figure(scenario, errors, output_dir):
    """Create a CDF plot of relative errors for a scenario."""
    if errors is None or len(errors) == 0:
        print(f"⚠️ No valid errors for scenario {scenario}")
        return False
    
    plt.figure(figsize=(10, 6))
    
    # Sort errors and create CDF
    sorted_errors = np.sort(errors)
    y = np.linspace(0, 1, len(sorted_errors), endpoint=False)
    
    # Create step plot
    plt.step(sorted_errors, y, where="post", linewidth=2, label=f"Scenario {scenario}")
    
    plt.xlabel("Relative Error", fontsize=12)
    plt.xscale("log")
    plt.xlim(1e-3, 10)
    plt.ylabel("CDF", fontsize=12)
    plt.title(f"CDF of Relative Errors - Scenario {scenario}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    
    # Add statistics text
    median_error = np.median(errors)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    stats_text = f"Median: {median_error:.4f}\nMean: {mean_error:.4f}\nMax: {max_error:.4f}\nMin: {min_error:.4f}\nSamples: {len(errors)}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f"{scenario}_errors.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created figure for {scenario}: {output_file}")
    print(f"   Median error: {median_error:.6f}, Mean error: {mean_error:.6f}, Samples: {len(errors)}")
    
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
        local_file = os.path.join(scenario_dir, "real_world.txt")
        
        print(f"\n=== Processing Scenario: {scenario} ===")
        
        # Look for comparison files in the same directory
        # Try different possible remote file names
        remote_candidates = [
            "flowsim_output.txt",
            "m4_output.txt", 
            "ns3_output.txt",
            "ns3/ns3_output.txt"
        ]
        
        errors = None
        for remote_candidate in remote_candidates:
            remote_file = os.path.join(scenario_dir, remote_candidate)
            if os.path.exists(remote_file):
                print(f"Found comparison file: {remote_candidate}")
                errors = calculate_relative_errors(local_file, remote_file)
                if errors is not None and len(errors) > 0:
                    break
        
        if errors is not None and len(errors) > 0:
            if create_error_figure(scenario, errors, output_dir):
                successful_figures += 1
        else:
            print(f"❌ No valid comparison data found for {scenario}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully created {successful_figures} figures out of {len(scenarios)} scenarios")
    print(f"Figures saved in: {output_dir}/")

if __name__ == "__main__":
    main()
