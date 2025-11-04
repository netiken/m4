#!/usr/bin/env python3
"""
Analyze end-to-end completion times from simulation result files.
Excludes first 50 and last 50 flows based on client req_send events.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

def parse_file(filepath):
    """Parse a result file and extract flow information."""
    flows = {}
    req_send_order = []  # Track order of req_send events for files without IDs
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('RDMA MTP:'):
                continue
                
            # Handle different file formats
            if 'event=' in line:
                # Check if this file has explicit flow IDs
                id_match = re.search(r'id=(\d+)', line)
                if id_match:
                    # Format: event=req_send ts_ns=1 id=0 ...
                    match = re.search(r'event=(\w+)\s+ts_ns=(\d+)\s+id=(\d+)', line)
                    if match:
                        event_type = match.group(1)
                        timestamp = int(match.group(2))
                        flow_id = int(match.group(3))
                        
                        if flow_id not in flows:
                            flows[flow_id] = {}
                        
                        if event_type == 'req_send':
                            flows[flow_id]['start_time'] = timestamp
                        elif event_type == 'resp_rdma_read':
                            flows[flow_id]['end_time'] = timestamp
                else:
                    # Format without explicit IDs - use client/worker/slot to match flows
                    match = re.search(r'event=(\w+)\s+ts_ns=(\d+).*?clt=(\d+)\s+wrkr=(\d+)\s+slot=(\d+)', line)
                    if match:
                        event_type = match.group(1)
                        timestamp = int(match.group(2))
                        client = int(match.group(3))
                        worker = int(match.group(4))
                        slot = int(match.group(5))
                        
                        # Create unique flow key from client, worker, slot
                        flow_key = (client, worker, slot)
                        
                        if event_type == 'req_send':
                            # Use order of req_send as flow ID
                            flow_id = len(req_send_order)
                            req_send_order.append((flow_key, timestamp))
                            
                            if flow_id not in flows:
                                flows[flow_id] = {}
                            flows[flow_id]['start_time'] = timestamp
                            flows[flow_id]['flow_key'] = flow_key
                            
                        elif event_type == 'resp_rdma_read':
                            # Find matching req_send by flow_key
                            for flow_id, flow_data in flows.items():
                                if ('flow_key' in flow_data and 
                                    flow_data['flow_key'] == flow_key and 
                                    'end_time' not in flow_data):
                                    flow_data['end_time'] = timestamp
                                    break
            else:
                # Format: [client req_send] t=0 ns reqId=1000001 ...
                if '[client req_send]' in line:
                    match = re.search(r't=(\d+)\s+ns\s+reqId=(\d+)', line)
                    if match:
                        timestamp = int(match.group(1))
                        req_id = int(match.group(2))
                        flow_id = req_id - 1000000  # Convert to 0-based ID
                        
                        if flow_id not in flows:
                            flows[flow_id] = {}
                        flows[flow_id]['start_time'] = timestamp
                        
                elif '[client rdma_recv]' in line:
                    match = re.search(r't=(\d+)\s+ns\s+reqId=(\d+)', line)
                    if match:
                        timestamp = int(match.group(1))
                        req_id = int(match.group(2))
                        flow_id = req_id - 1000000  # Convert to 0-based ID
                        
                        if flow_id not in flows:
                            flows[flow_id] = {}
                        flows[flow_id]['end_time'] = timestamp
    
    return flows

def calculate_completion_times(flows):
    """Calculate completion times for flows that have both start and end times."""
    completion_times = []
    valid_flow_ids = []
    
    for flow_id in sorted(flows.keys()):
        flow = flows[flow_id]
        if 'start_time' in flow and 'end_time' in flow:
            completion_time = flow['end_time'] - flow['start_time']
            completion_times.append(completion_time)
            valid_flow_ids.append(flow_id)
    
    return completion_times, valid_flow_ids

def filter_flows(completion_times, valid_flow_ids, skip_first=50, skip_last=50):
    """Remove first and last N flows."""
    if len(completion_times) <= skip_first + skip_last:
        print(f"Warning: Not enough flows to skip {skip_first} + {skip_last}. Total flows: {len(completion_times)}")
        return completion_times, valid_flow_ids
    
    start_idx = skip_first
    end_idx = len(completion_times) - skip_last
    
    filtered_times = completion_times[start_idx:end_idx]
    filtered_ids = valid_flow_ids[start_idx:end_idx]
    
    return filtered_times, filtered_ids

def calculate_total_experiment_duration(flows):
    """Calculate total experiment duration from first request to last completion."""
    all_start_times = []
    all_end_times = []
    
    for flow_id, flow_data in flows.items():
        if 'start_time' in flow_data:
            all_start_times.append(flow_data['start_time'])
        if 'end_time' in flow_data:
            all_end_times.append(flow_data['end_time'])
    
    if all_start_times and all_end_times:
        first_start = min(all_start_times)
        last_end = max(all_end_times)
        total_duration_ns = last_end - first_start
        return total_duration_ns, first_start, last_end
    
    return None, None, None

def main():
    # File paths
    files = {
        'FlowSim': '/Users/omchabra/Downloads/new_scenarios/big_singleton_2/flowsim_res.txt',
        'M4': '/Users/omchabra/Downloads/new_scenarios/big_singleton_2/m4_res.txt',
        'RealWorld': '/Users/omchabra/Downloads/new_scenarios/big_singleton_2/realworld_res.txt',
        'Unsion': '/Users/omchabra/Downloads/new_scenarios/big_singleton_2/unsion_res.txt'
    }
    
    results = {}
    
    print("Processing files...")
    for name, filepath in files.items():
        print(f"\nProcessing {name}...")
        
        # Parse file
        flows = parse_file(filepath)
        print(f"  Total flows parsed: {len(flows)}")
        
        # Calculate total experiment duration
        total_duration_ns, first_start, last_end = calculate_total_experiment_duration(flows)
        if total_duration_ns:
            total_duration_ms = total_duration_ns / 1_000_000  # Convert to milliseconds
            total_duration_s = total_duration_ns / 1_000_000_000  # Convert to seconds
            print(f"  TOTAL EXPERIMENT DURATION:")
            print(f"    {total_duration_ms:.2f} milliseconds")
            print(f"    {total_duration_s:.3f} seconds")
            print(f"    First request at: {first_start} ns")
            print(f"    Last completion at: {last_end} ns")
        
        # Calculate completion times
        completion_times, valid_flow_ids = calculate_completion_times(flows)
        print(f"  Valid flows (with start and end): {len(completion_times)}")
        
        # Filter out first 50 and last 50 flows
        filtered_times, filtered_ids = filter_flows(completion_times, valid_flow_ids, 50, 50)
        print(f"  Flows after filtering (removed first 50 and last 50): {len(filtered_times)}")
        
        if filtered_times:
            # Convert to microseconds for better readability
            filtered_times_us = [t / 1000 for t in filtered_times]
            
            results[name] = {
                'completion_times_ns': filtered_times,
                'completion_times_us': filtered_times_us,
                'flow_ids': filtered_ids,
                'total_duration_ns': total_duration_ns,
                'total_duration_ms': total_duration_ms if total_duration_ns else None,
                'total_duration_s': total_duration_s if total_duration_ns else None,
                'first_start': first_start,
                'last_end': last_end,
                'stats': {
                    'mean_us': np.mean(filtered_times_us),
                    'median_us': np.median(filtered_times_us),
                    'std_us': np.std(filtered_times_us),
                    'min_us': np.min(filtered_times_us),
                    'max_us': np.max(filtered_times_us),
                    'count': len(filtered_times_us)
                }
            }
            
            print(f"  Individual Flow Statistics (microseconds):")
            print(f"    Mean: {results[name]['stats']['mean_us']:.2f}")
            print(f"    Median: {results[name]['stats']['median_us']:.2f}")
            print(f"    Std Dev: {results[name]['stats']['std_us']:.2f}")
            print(f"    Min: {results[name]['stats']['min_us']:.2f}")
            print(f"    Max: {results[name]['stats']['max_us']:.2f}")
        else:
            print(f"  No valid completion times found for {name}")
            if total_duration_ns:
                results[name] = {
                    'total_duration_ns': total_duration_ns,
                    'total_duration_ms': total_duration_ms,
                    'total_duration_s': total_duration_s,
                    'first_start': first_start,
                    'last_end': last_end
                }
    
    # Create plots
    if results:
        print("\nCreating plots...")
        
        # Plot 1: Box plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        box_data = []
        box_labels = []
        for name, data in results.items():
            if data['completion_times_us']:
                box_data.append(data['completion_times_us'])
                box_labels.append(name)
        
        if box_data:
            ax1.boxplot(box_data, labels=box_labels)
            ax1.set_title('End-to-End Completion Time Comparison')
            ax1.set_ylabel('Completion Time (microseconds)')
            ax1.grid(True, alpha=0.3)
            
            # Histogram comparison
            colors = ['blue', 'red', 'green', 'orange']
            for i, (name, data) in enumerate(results.items()):
                if data['completion_times_us']:
                    ax2.hist(data['completion_times_us'], bins=50, alpha=0.7, 
                            label=name, color=colors[i % len(colors)], density=True)
            
            ax2.set_title('Completion Time Distribution')
            ax2.set_xlabel('Completion Time (microseconds)')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/Users/omchabra/Downloads/new_scenarios/completion_times_comparison.png', dpi=300, bbox_inches='tight')
            print("Plot saved to completion_times_comparison.png")
            
        # Create summary table
        print("\n" + "="*80)
        print("TOTAL EXPERIMENT DURATIONS")
        print("="*80)
        
        duration_data = {}
        for name, data in results.items():
            if 'total_duration_s' in data and data['total_duration_s'] is not None:
                duration_data[name] = {
                    'Duration (seconds)': data['total_duration_s'],
                    'Duration (ms)': data['total_duration_ms'],
                    'Total Flows': data.get('stats', {}).get('count', 'N/A')
                }
        
        if duration_data:
            duration_df = pd.DataFrame(duration_data).T
            print(duration_df.round(3))
        
        print("\n" + "="*80)
        print("INDIVIDUAL FLOW STATISTICS (microseconds)")
        print("="*80)
        
        stats_data = {}
        for name, data in results.items():
            if 'stats' in data:
                stats_data[name] = data['stats']
        
        if stats_data:
            summary_df = pd.DataFrame(stats_data).T
            print(summary_df.round(2))
        
        # Save detailed results
        with open('/Users/omchabra/Downloads/new_scenarios/completion_times_summary.txt', 'w') as f:
            f.write("End-to-End Completion Time Analysis\n")
            f.write("="*50 + "\n\n")
            f.write("Files analyzed:\n")
            for name, filepath in files.items():
                f.write(f"  {name}: {filepath}\n")
            f.write(f"\nFiltering: Removed first 50 and last 50 flows\n\n")
            
            f.write("Summary Statistics (microseconds):\n")
            f.write("-" * 50 + "\n")
            f.write(summary_df.round(2).to_string())
            f.write("\n\n")
            
            for name, data in results.items():
                f.write(f"{name} - Raw completion times (first 20):\n")
                times_sample = data['completion_times_us'][:20]
                f.write(f"  {times_sample}\n\n")
        
        print(f"\nResults saved to:")
        print(f"  - Plot: /Users/omchabra/Downloads/new_scenarios/completion_times_comparison.png")
        print(f"  - Summary: /Users/omchabra/Downloads/new_scenarios/completion_times_summary.txt")
    
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    main()
