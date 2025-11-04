import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add path for util.plot import (following notebook style)
sys.path.append('.')

# Try to import plot_cdf from util.plot like the notebook does
try:
    from util.plot import plot_cdf
    use_util_plot = True
    print("Using util.plot.plot_cdf function (notebook style)")
except ImportError:
    use_util_plot = False
    print("util.plot not available, using matplotlib directly")


def load_separate_remote_errors():
    """Load separate error data for each remote from analyze.py output files"""
    
    # Define the remote data files and their display names
    remote_files = {
        'flowSim': 'relative_errors_flowsim-output.npy',
        'ns3': 'relative_errors_ns3-output.npy', 
        'FLS': 'relative_errors_m4-newmodel.npy'  # Using FLS like in notebook
    }
    
    error_data = {}
    
    for name, filename in remote_files.items():
        try:
            errors = np.load(filename)
            error_data[name] = errors
            print(f"Loaded {len(errors)} samples for {name} from {filename}")
            print(f"  Median: {np.median(errors):.6f}, Mean: {np.mean(errors):.6f}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping {name}")
    
    return error_data

# Load the separate error data for each remote
error_data = load_separate_remote_errors()

if not error_data:
    print("No remote data files found. Run analyze.py first to generate the data.")
    exit(1)

# Prepare data in the format expected by plot_cdf (list of arrays)
plot_data = []
legend_list = []

for name, errors in error_data.items():
    # Remove infinite values
    finite_errors = errors[np.isfinite(errors)]
    print(f"{name}: Using {len(finite_errors)} finite samples (median: {np.median(finite_errors):.6f})")
    plot_data.append(finite_errors)
    legend_list.append(name)

# Create CDF plot following the exact notebook style
if use_util_plot:
    # Use the same format as the notebook
    plot_cdf(
        plot_data,
        f"./figs/relative_errors_cdf.pdf",
        legend_list,
        x_label="Relative Error",
        log_switch=True,
        rotate_xaxis=False,
        xlim_bottom=0.01,
        xlim=10,
        fontsize=15,
        legend_font=18,
        loc=4,
        enable_abs=True,
        group_size=3,
        colors=["orange", "blueviolet", "cornflowerblue"],
        fig_idx=0,
        fig_size=(8, 6),
    )
    print("CDF plot saved to figs/relative_errors_cdf.pdf (using notebook style)")
else:
    # Fallback to matplotlib if util.plot is not available
    plt.figure(figsize=(8, 6))
    
    colors = ["orange", "blueviolet", "cornflowerblue"]
    
    # Plot each remote as a separate line
    for i, (errors, name) in enumerate(zip(plot_data, legend_list)):
        # Sort and create CDF
        arr = np.sort(errors)
        y = np.linspace(0, 1, len(arr), endpoint=False)
        
        # Plot with step function
        plt.step(arr, y, where="post", label=name, linewidth=2, color=colors[i % len(colors)])

    plt.xlabel("Relative Error", fontsize=15)
    plt.xscale("log")
    plt.xlim(0.01, 10)
    plt.ylabel("CDF", fontsize=15)
    plt.title("CDF of Relative Errors by Remote Source")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=18, loc=4)
    plt.tight_layout()

    # Save the plot
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/relative_errors_cdf.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("CDF plot saved to figs/relative_errors_cdf.pdf (matplotlib fallback)")

print("Successfully created CDF plot with notebook formatting style!")