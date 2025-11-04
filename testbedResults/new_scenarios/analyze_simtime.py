#!/usr/bin/env python3
"""
Script to analyze simulation time ranges across all experiments.
Extracts the min and max ts_ns from the last 1000 flows in flows_debug.txt files and reports the difference.
"""

import os
import re
import sys
from pathlib import Path
import glob


def extract_timestamp_range_from_flows_debug(exp_dir):
    """
    Extract the min and max timestamp from flows_debug.txt file in an experiment directory.
    Only considers the last 1000 flows in the file.
    Returns (min_timestamp, max_timestamp) in nanoseconds, or (None, None) if not found.
    """
    flows_debug_file = exp_dir / "flows_debug.txt"
    
    if not flows_debug_file.exists():
        return None, None
    
    # First pass: count total flows and collect flow data
    flow_data = []
    current_flow_timestamps = []
    
    try:
        with open(flows_debug_file, 'r') as f:
            for line in f:
                # Check if this is a new flow (starts with flow_id= at beginning of line)
                if line.startswith("flow_id="):
                    # Save previous flow's timestamps if any
                    if current_flow_timestamps:
                        flow_data.append(current_flow_timestamps)
                    current_flow_timestamps = []
                    continue
                
                # Collect timestamps for current flow
                if "ts_ns=" in line:
                    match = re.search(r'ts_ns=(\d+)', line)
                    if match:
                        timestamp = int(match.group(1))
                        current_flow_timestamps.append(timestamp)
            
            # Don't forget the last flow
            if current_flow_timestamps:
                flow_data.append(current_flow_timestamps)
                
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"Error reading {flows_debug_file}: {e}")
        return None, None
    
    # Take only the last 1000 flows
    total_flows = len(flow_data)
    if total_flows == 0:
        return None, None
    
    # Get the last 1000 flows (or all flows if less than 1000)
    last_1000_flows = flow_data[-1000:] if total_flows >= 1000 else flow_data
    
    # Find min and max timestamps from the last 1000 flows
    min_timestamp = None
    max_timestamp = None
    
    for flow_timestamps in last_1000_flows:
        for timestamp in flow_timestamps:
            if min_timestamp is None or timestamp < min_timestamp:
                min_timestamp = timestamp
            if max_timestamp is None or timestamp > max_timestamp:
                max_timestamp = timestamp
    
    return min_timestamp, max_timestamp


def parse_experiment_name(exp_name):
    """
    Parse experiment directory name to extract parameters.
    Expected format: {flows}_{threads}
    """
    try:
        parts = exp_name.split('_')
        if len(parts) == 2:
            flows = int(parts[0])
            threads = int(parts[1])
            return flows, threads
    except ValueError:
        pass
    return None, None


def format_time(ns):
    """Format nanoseconds into human-readable format."""
    if ns is None:
        return "N/A"
    
    # Convert to different units
    us = ns / 1000
    ms = ns / 1_000_000
    s = ns / 1_000_000_000
    
    if s >= 1:
        return f"{s:.3f} s"
    elif ms >= 1:
        return f"{ms:.3f} ms"
    elif us >= 1:
        return f"{us:.3f} μs"
    else:
        return f"{ns} ns"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze simulation time ranges across all experiments")
    parser.add_argument("--experiments-dir", default="expirements_12", 
                       help="Path to experiments directory")
    parser.add_argument("--output", "-o", help="Output file to save results (CSV format)")
    parser.add_argument("--sort-by", choices=['experiment', 'flows', 'threads', 'range'], default='experiment',
                       help="Sort results by specified column")
    parser.add_argument("--simple-csv", help="Output simple CSV with just experiment,formatted_range")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output CSV, no console output")
    
    args = parser.parse_args()
    experiments_dir = Path(args.experiments_dir)
    
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        sys.exit(1)
    
    # Collect all experiment results
    results = []
    
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        # Extract min and max timestamps from flows_debug.txt in this experiment directory
        min_ts, max_ts = extract_timestamp_range_from_flows_debug(exp_dir)
        flows, threads = parse_experiment_name(exp_dir.name)
        
        time_range = None
        if min_ts is not None and max_ts is not None:
            time_range = max_ts - min_ts
        
        if min_ts is None or max_ts is None:
            print(f"Warning: No timestamps found in flows_debug.txt for {exp_dir.name}")
        
        results.append({
            'experiment': exp_dir.name,
            'flows': flows,
            'threads': threads,
            'min_ts_ns': min_ts,
            'max_ts_ns': max_ts,
            'time_range_ns': time_range,
            'time_range_formatted': format_time(time_range)
        })
    
    # Sort results
    if args.sort_by == 'flows':
        results.sort(key=lambda x: (x['flows'] or 0, x['threads'] or 0))
    elif args.sort_by == 'threads':
        results.sort(key=lambda x: (x['threads'] or 0, x['flows'] or 0))
    elif args.sort_by == 'range':
        results.sort(key=lambda x: x['time_range_ns'] or 0)
    else:  # sort by experiment name (default)
        results.sort(key=lambda x: x['experiment'])
    
    # Handle simple CSV output
    if args.simple_csv:
        import csv
        with open(args.simple_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sweep', 'time_in_milliseconds'])
            for result in results:
                # Convert nanoseconds to milliseconds
                time_ms = result['time_range_ns'] / 1_000_000 if result['time_range_ns'] is not None else None
                writer.writerow([result['experiment'], time_ms])
        if not args.quiet:
            print(f"Simple CSV saved to: {args.simple_csv}")
        return
    
    # Print results (unless quiet mode)
    if not args.quiet:
        print("Simulation Time Range Analysis")
        print("=" * 80)
        print(f"{'Experiment':<12} {'Flows':<8} {'Threads':<8} {'Min ts_ns':<15} {'Max ts_ns':<15} {'Range':<12}")
        print("-" * 80)
        
        for result in results:
            flows_str = str(result['flows']) if result['flows'] is not None else "N/A"
            threads_str = str(result['threads']) if result['threads'] is not None else "N/A"
            min_ts_str = str(result['min_ts_ns']) if result['min_ts_ns'] is not None else "N/A"
            max_ts_str = str(result['max_ts_ns']) if result['max_ts_ns'] is not None else "N/A"
            
            print(f"{result['experiment']:<12} {flows_str:<8} {threads_str:<8} {min_ts_str:<15} {max_ts_str:<15} {result['time_range_formatted']:<12}")
    
    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['experiment', 'flows', 'threads', 'min_ts_ns', 'max_ts_ns', 'time_range_ns', 'time_range_formatted']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")
    
    # Summary statistics (skip if quiet mode)
    if args.quiet:
        return
        
    valid_ranges = [r['time_range_ns'] for r in results if r['time_range_ns'] is not None]
    
    if valid_ranges:
        print("\nSummary Statistics:")
        print("-" * 30)
        print(f"Total experiments: {len(results)}")
        print(f"Valid results: {len(valid_ranges)}")
        print(f"Min time range: {format_time(min(valid_ranges))}")
        print(f"Max time range: {format_time(max(valid_ranges))}")
        print(f"Average time range: {format_time(sum(valid_ranges) // len(valid_ranges))}")
        
        # Group by flows and threads for analysis
        print("\nGrouped Analysis:")
        print("-" * 30)
        
        # Group by flows
        flows_groups = {}
        threads_groups = {}
        
        for result in results:
            if result['time_range_ns'] is not None:
                flows = result['flows']
                threads = result['threads']
                
                if flows is not None:
                    if flows not in flows_groups:
                        flows_groups[flows] = []
                    flows_groups[flows].append(result['time_range_ns'])
                
                if threads is not None:
                    if threads not in threads_groups:
                        threads_groups[threads] = []
                    threads_groups[threads].append(result['time_range_ns'])
        
        print("\nBy number of flows:")
        for flows in sorted(flows_groups.keys()):
            ranges = flows_groups[flows]
            avg_range = sum(ranges) // len(ranges)
            print(f"  {flows} flows: avg {format_time(avg_range)} ({len(ranges)} experiments)")
        
        print("\nBy number of threads:")
        for threads in sorted(threads_groups.keys()):
            ranges = threads_groups[threads]
            avg_range = sum(ranges) // len(ranges)
            print(f"  {threads} threads: avg {format_time(avg_range)} ({len(ranges)} experiments)")


if __name__ == "__main__":
    main()
