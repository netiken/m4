import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_sweep_figures():
    """Analyze the generated sweep figures and create a summary."""
    sweepfigures_dir = "sweepfigures"
    
    if not os.path.exists(sweepfigures_dir):
        print("❌ sweepfigures directory not found!")
        return
    
    # Get all PNG files
    png_files = [f for f in os.listdir(sweepfigures_dir) if f.endswith('.png')]
    
    print(f"Found {len(png_files)} analysis figures in {sweepfigures_dir}/")
    print("\n=== Figure Summary ===")
    
    # Group by scenario patterns
    scenarios = {}
    for png_file in sorted(png_files):
        scenario_name = png_file.replace('_analysis.png', '')
        
        # Extract the main parameter (first number)
        if '_' in scenario_name:
            main_param = scenario_name.split('_')[0]
            if main_param not in scenarios:
                scenarios[main_param] = []
            scenarios[main_param].append(scenario_name)
        else:
            if 'other' not in scenarios:
                scenarios['other'] = []
            scenarios['other'].append(scenario_name)
    
    # Print organized summary
    for main_param, sub_scenarios in sorted(scenarios.items()):
        print(f"\n{main_param} scenarios ({len(sub_scenarios)}):")
        for scenario in sorted(sub_scenarios):
            print(f"  - {scenario}_analysis.png")
    
    print(f"\n=== File Details ===")
    print(f"Total figures created: {len(png_files)}")
    print(f"Directory: {os.path.abspath(sweepfigures_dir)}")
    
    # Check file sizes
    total_size = 0
    for png_file in png_files:
        file_path = os.path.join(sweepfigures_dir, png_file)
        size = os.path.getsize(file_path)
        total_size += size
    
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print(f"Average size per figure: {total_size / len(png_files) / 1024:.2f} KB")

if __name__ == "__main__":
    analyze_sweep_figures()
