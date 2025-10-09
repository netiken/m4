#!/usr/bin/env python3
"""
Script to analyze simulation times across all sweeps.
Extracts the final simulation time from client*.log and server.log files.
"""

import os
import re
import sys
from pathlib import Path
import glob


def extract_max_timestamp_from_logs(sweep_dir):
    """
    Extract the maximum timestamp from all client*.log and server.log files in a sweep directory.
    Returns the maximum timestamp in nanoseconds, or None if not found.
    """
    max_timestamp = None
    log_files = []
    
    # Find all client*.log files and server.log
    log_files.extend(glob.glob(str(sweep_dir / "client*.log")))
    server_log = sweep_dir / "server.log"
    if server_log.exists():
        log_files.append(str(server_log))
    
    if not log_files:
        return None
        
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                # Read the file backwards to find the last timestamp quickly
                # For efficiency, we'll read the last few lines instead of the whole file
                f.seek(0, 2)  # Go to end of file
                file_size = f.tell()
                
                # Read last few KB to find the last timestamp
                chunk_size = min(8192, file_size)
                f.seek(max(0, file_size - chunk_size))
                lines = f.readlines()
                
            # Look for the last line with a timestamp in this file
            file_max_timestamp = None
            for line in reversed(lines):
                # Match pattern like "ts_ns=182240593"
                match = re.search(r'ts_ns=(\d+)', line)
                if match:
                    file_max_timestamp = int(match.group(1))
                    break
            
            # Update global max if this file has a larger timestamp
            if file_max_timestamp is not None:
                if max_timestamp is None or file_max_timestamp > max_timestamp:
                    max_timestamp = file_max_timestamp
                    
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    return max_timestamp


def parse_sweep_name(sweep_name):
    """
    Parse sweep directory name to extract parameters.
    Expected format: {flows}_{threads}
    """
    try:
        parts = sweep_name.split('_')
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
    
    parser = argparse.ArgumentParser(description="Analyze simulation times across all sweeps")
    parser.add_argument("--sweeps-dir", default="/data1/lichenni/projects/om/m4/inference/sweeps_12", 
                       help="Path to sweeps directory")
    parser.add_argument("--output", "-o", help="Output file to save results (CSV format)")
    parser.add_argument("--sort-by", choices=['sweep', 'flows', 'threads', 'time'], default='sweep',
                       help="Sort results by specified column")
    parser.add_argument("--simple-csv", help="Output simple CSV with just sweep,formatted_time")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output CSV, no console output")
    
    args = parser.parse_args()
    sweeps_dir = Path(args.sweeps_dir)
    
    if not sweeps_dir.exists():
        print(f"Sweeps directory not found: {sweeps_dir}")
        sys.exit(1)
    
    # Collect all sweep results
    results = []
    
    for sweep_dir in sorted(sweeps_dir.iterdir()):
        if not sweep_dir.is_dir():
            continue
            
        # Extract max timestamp from all log files in this sweep directory
        sim_time_ns = extract_max_timestamp_from_logs(sweep_dir)
        flows, threads = parse_sweep_name(sweep_dir.name)
        
        if sim_time_ns is None:
            print(f"Warning: No timestamps found in log files for {sweep_dir.name}")
        
        results.append({
            'sweep': sweep_dir.name,
            'flows': flows,
            'threads': threads,
            'sim_time_ns': sim_time_ns,
            'sim_time_formatted': format_time(sim_time_ns)
        })
    
    # Sort results
    if args.sort_by == 'flows':
        results.sort(key=lambda x: (x['flows'] or 0, x['threads'] or 0))
    elif args.sort_by == 'threads':
        results.sort(key=lambda x: (x['threads'] or 0, x['flows'] or 0))
    elif args.sort_by == 'time':
        results.sort(key=lambda x: x['sim_time_ns'] or 0)
    else:  # sort by sweep name (default)
        results.sort(key=lambda x: x['sweep'])
    
    # Handle simple CSV output
    if args.simple_csv:
        import csv
        with open(args.simple_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sweep', 'formatted'])
            for result in results:
                writer.writerow([result['sweep'], result['sim_time_formatted']])
        if not args.quiet:
            print(f"Simple CSV saved to: {args.simple_csv}")
        return
    
    # Print results (unless quiet mode)
    if not args.quiet:
        print("Simulation Time Analysis")
        print("=" * 60)
        print(f"{'Sweep':<12} {'Flows':<8} {'Threads':<8} {'Sim Time (ns)':<15} {'Formatted':<12}")
        print("-" * 60)
        
        for result in results:
            flows_str = str(result['flows']) if result['flows'] is not None else "N/A"
            threads_str = str(result['threads']) if result['threads'] is not None else "N/A"
            sim_time_str = str(result['sim_time_ns']) if result['sim_time_ns'] is not None else "N/A"
            
            print(f"{result['sweep']:<12} {flows_str:<8} {threads_str:<8} {sim_time_str:<15} {result['sim_time_formatted']:<12}")
    
    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['sweep', 'flows', 'threads', 'sim_time_ns', 'sim_time_formatted']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")
    
    # Summary statistics (skip if quiet mode)
    if args.quiet:
        return
        
    valid_times = [r['sim_time_ns'] for r in results if r['sim_time_ns'] is not None]
    
    if valid_times:
        print("\nSummary Statistics:")
        print("-" * 30)
        print(f"Total simulations: {len(results)}")
        print(f"Valid results: {len(valid_times)}")
        print(f"Min simulation time: {format_time(min(valid_times))}")
        print(f"Max simulation time: {format_time(max(valid_times))}")
        print(f"Average simulation time: {format_time(sum(valid_times) // len(valid_times))}")
        
        # Group by flows and threads for analysis
        print("\nGrouped Analysis:")
        print("-" * 30)
        
        # Group by flows
        flows_groups = {}
        threads_groups = {}
        
        for result in results:
            if result['sim_time_ns'] is not None:
                flows = result['flows']
                threads = result['threads']
                
                if flows is not None:
                    if flows not in flows_groups:
                        flows_groups[flows] = []
                    flows_groups[flows].append(result['sim_time_ns'])
                
                if threads is not None:
                    if threads not in threads_groups:
                        threads_groups[threads] = []
                    threads_groups[threads].append(result['sim_time_ns'])
        
        print("\nBy number of flows:")
        for flows in sorted(flows_groups.keys()):
            times = flows_groups[flows]
            avg_time = sum(times) // len(times)
            print(f"  {flows} flows: avg {format_time(avg_time)} ({len(times)} simulations)")
        
        print("\nBy number of threads:")
        for threads in sorted(threads_groups.keys()):
            times = threads_groups[threads]
            avg_time = sum(times) // len(times)
            print(f"  {threads} threads: avg {format_time(avg_time)} ({len(times)} simulations)")


if __name__ == "__main__":
    main()
