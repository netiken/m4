#!/usr/bin/env python3
"""
Diagnostic tool to analyze M4's prediction errors and identify root causes.
This will help us understand what specific improvements are needed to reach 20% error.
"""

import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_m4_output(file_path):
    """Parse M4 output file to extract predicted FCTs."""
    results = {'ud': {}, 'rdma': {}}
    with open(file_path) as f:
        for line in f:
            m = re.match(r'\[(ud|rdma)\] client=(\d+) id=(\d+) dur_ns=(\d+)', line)
            if m:
                flow_type, client, flow_id, dur_ns = m.groups()
                results[flow_type][(int(client), int(flow_id))] = int(dur_ns)
    return results

def parse_grouped_flows(file_path):
    """Parse grouped_flows.txt to extract actual FCTs."""
    results = {'ud': {}, 'rdma': {}}
    with open(file_path) as f:
        for line in f:
            m = re.match(r'\[(ud|rdma)\] client=(\d+) id=(\d+) dur_ns=(\d+)', line)
            if m:
                flow_type, client, flow_id, dur_ns = m.groups()
                results[flow_type][(int(client), int(flow_id))] = int(dur_ns)
    return results

def compute_ideal_fct(flow_size, n_links=2):
    """Compute ideal FCT using the NEW formula from main.cpp."""
    MTU = 1000.0
    HEADER_SIZE = 48.0
    BYTES_TO_NS = 0.8
    PROPAGATION_PER_LINK = 1000.0
    SERVER_OVERHEAD_NS = 87000
    
    num_packets = np.ceil(flow_size / MTU)
    total_bytes = flow_size + num_packets * HEADER_SIZE
    transmission_ns = total_bytes * BYTES_TO_NS
    propagation_ns = PROPAGATION_PER_LINK * n_links
    
    return propagation_ns + transmission_ns + SERVER_OVERHEAD_NS

def analyze_scenario(scenario_dir):
    """Analyze a single scenario and return detailed error statistics."""
    m4_file = scenario_dir / "m4_output.txt"
    grouped_file = scenario_dir / "grouped_flows.txt"
    
    if not m4_file.exists() or not grouped_file.exists():
        return None
    
    predicted = parse_m4_output(m4_file)
    actual = parse_grouped_flows(grouped_file)
    
    errors = {'ud': [], 'rdma': []}
    slowdowns_pred = {'ud': [], 'rdma': []}
    slowdowns_actual = {'ud': [], 'rdma': []}
    
    # UD flows: 17B
    for key in predicted['ud']:
        if key in actual['ud']:
            pred_ns = predicted['ud'][key]
            act_ns = actual['ud'][key]
            ideal = compute_ideal_fct(17, 2)
            
            error = abs(pred_ns - act_ns) / act_ns * 100
            errors['ud'].append(error)
            slowdowns_pred['ud'].append(pred_ns / ideal)
            slowdowns_actual['ud'].append(act_ns / ideal)
    
    # RDMA flows: size depends on window
    # Extract window from scenario name
    scenario_name = scenario_dir.name
    window_size = int(scenario_name.split('_')[0])
    rdma_size = 102400 + window_size * 8  # 102400 base + 8*window
    
    for key in predicted['rdma']:
        if key in actual['rdma']:
            pred_ns = predicted['rdma'][key]
            act_ns = actual['rdma'][key]
            ideal = compute_ideal_fct(rdma_size, 2)
            
            error = abs(pred_ns - act_ns) / act_ns * 100
            errors['rdma'].append(error)
            slowdowns_pred['rdma'].append(pred_ns / ideal)
            slowdowns_actual['rdma'].append(act_ns / ideal)
    
    return {
        'errors': errors,
        'slowdowns_pred': slowdowns_pred,
        'slowdowns_actual': slowdowns_actual,
        'window': window_size
    }

def main():
    print("🔬 M4 Error Diagnosis Tool")
    print("=" * 80)
    print()
    
    eval_dir = Path("/data1/lichenni/m4/testbed/eval_test/m4")
    
    # Collect all scenario results
    all_results = []
    for scenario_dir in sorted(eval_dir.iterdir()):
        if scenario_dir.is_dir():
            result = analyze_scenario(scenario_dir)
            if result:
                result['scenario'] = scenario_dir.name
                all_results.append(result)
                print(f"✓ Analyzed {scenario_dir.name}")
    
    print()
    print("=" * 80)
    print("📊 ERROR ANALYSIS BY FLOW TYPE")
    print("=" * 80)
    print()
    
    # Aggregate by flow type
    all_ud_errors = []
    all_rdma_errors = []
    all_ud_sldn_pred = []
    all_ud_sldn_actual = []
    all_rdma_sldn_pred = []
    all_rdma_sldn_actual = []
    
    for result in all_results:
        all_ud_errors.extend(result['errors']['ud'])
        all_rdma_errors.extend(result['errors']['rdma'])
        all_ud_sldn_pred.extend(result['slowdowns_pred']['ud'])
        all_ud_sldn_actual.extend(result['slowdowns_actual']['ud'])
        all_rdma_sldn_pred.extend(result['slowdowns_pred']['rdma'])
        all_rdma_sldn_actual.extend(result['slowdowns_actual']['rdma'])
    
    print("UD FLOWS (17B):")
    print(f"  Count:         {len(all_ud_errors)}")
    print(f"  Median Error:  {np.median(all_ud_errors):.1f}%")
    print(f"  Mean Error:    {np.mean(all_ud_errors):.1f}%")
    print(f"  Std Dev:       {np.std(all_ud_errors):.1f}%")
    print(f"  P90 Error:     {np.percentile(all_ud_errors, 90):.1f}%")
    print(f"  P99 Error:     {np.percentile(all_ud_errors, 99):.1f}%")
    print()
    print(f"  Predicted Slowdown: median={np.median(all_ud_sldn_pred):.2f}, mean={np.mean(all_ud_sldn_pred):.2f}")
    print(f"  Actual Slowdown:    median={np.median(all_ud_sldn_actual):.2f}, mean={np.mean(all_ud_sldn_actual):.2f}")
    print(f"  → Slowdown Bias:    {(np.mean(all_ud_sldn_pred) / np.mean(all_ud_sldn_actual) - 1) * 100:+.1f}%")
    print()
    
    print("RDMA FLOWS (102408B - 102408B + window*8):")
    print(f"  Count:         {len(all_rdma_errors)}")
    print(f"  Median Error:  {np.median(all_rdma_errors):.1f}%")
    print(f"  Mean Error:    {np.mean(all_rdma_errors):.1f}%")
    print(f"  Std Dev:       {np.std(all_rdma_errors):.1f}%")
    print(f"  P90 Error:     {np.percentile(all_rdma_errors, 90):.1f}%")
    print(f"  P99 Error:     {np.percentile(all_rdma_errors, 99):.1f}%")
    print()
    print(f"  Predicted Slowdown: median={np.median(all_rdma_sldn_pred):.2f}, mean={np.mean(all_rdma_sldn_pred):.2f}")
    print(f"  Actual Slowdown:    median={np.median(all_rdma_sldn_actual):.2f}, mean={np.mean(all_rdma_sldn_actual):.2f}")
    print(f"  → Slowdown Bias:    {(np.mean(all_rdma_sldn_pred) / np.mean(all_rdma_sldn_actual) - 1) * 100:+.1f}%")
    print()
    
    # Analyze by window size
    print("=" * 80)
    print("📊 ERROR ANALYSIS BY WINDOW SIZE (LOAD)")
    print("=" * 80)
    print()
    
    by_window = defaultdict(lambda: {'ud': [], 'rdma': []})
    for result in all_results:
        window = result['window']
        by_window[window]['ud'].extend(result['errors']['ud'])
        by_window[window]['rdma'].extend(result['errors']['rdma'])
    
    print("Window | UD Median | UD Mean | RDMA Median | RDMA Mean | Combined")
    print("-------|-----------|---------|-------------|-----------|----------")
    for window in sorted(by_window.keys()):
        ud_median = np.median(by_window[window]['ud'])
        ud_mean = np.mean(by_window[window]['ud'])
        rdma_median = np.median(by_window[window]['rdma'])
        rdma_mean = np.mean(by_window[window]['rdma'])
        combined = np.median(by_window[window]['ud'] + by_window[window]['rdma'])
        print(f"{window:6d} | {ud_median:8.1f}% | {ud_mean:6.1f}% | {rdma_median:10.1f}% | {rdma_mean:8.1f}% | {combined:7.1f}%")
    
    print()
    print("=" * 80)
    print("🎯 KEY INSIGHTS")
    print("=" * 80)
    print()
    
    # Calculate overall bias
    ud_bias = (np.mean(all_ud_sldn_pred) / np.mean(all_ud_sldn_actual) - 1) * 100
    rdma_bias = (np.mean(all_rdma_sldn_pred) / np.mean(all_rdma_sldn_actual) - 1) * 100
    
    print("1. SLOWDOWN PREDICTION BIAS:")
    if abs(ud_bias) > 10:
        print(f"   ⚠️  UD flows: {ud_bias:+.1f}% bias ({'UNDER' if ud_bias < 0 else 'OVER'}-predicting slowdowns)")
    else:
        print(f"   ✓  UD flows: {ud_bias:+.1f}% bias (well calibrated)")
    
    if abs(rdma_bias) > 10:
        print(f"   ⚠️  RDMA flows: {rdma_bias:+.1f}% bias ({'UNDER' if rdma_bias < 0 else 'OVER'}-predicting slowdowns)")
    else:
        print(f"   ✓  RDMA flows: {rdma_bias:+.1f}% bias (well calibrated)")
    print()
    
    print("2. ERROR DISTRIBUTION:")
    if np.median(all_ud_errors) < np.median(all_rdma_errors):
        print(f"   • UD flows are MORE accurate ({np.median(all_ud_errors):.1f}% vs {np.median(all_rdma_errors):.1f}%)")
    else:
        print(f"   • RDMA flows are MORE accurate ({np.median(all_rdma_errors):.1f}% vs {np.median(all_ud_errors):.1f}%)")
    print()
    
    print("3. LOAD DEPENDENCY:")
    low_load = np.median(by_window[100]['ud'] + by_window[100]['rdma'])
    high_load = np.median(by_window[1000]['ud'] + by_window[1000]['rdma'])
    if high_load > low_load * 1.2:
        print(f"   ⚠️  High load (window=1000) has {(high_load/low_load - 1)*100:+.1f}% more error than low load")
        print("   → Model may need better contention modeling")
    else:
        print(f"   ✓  Load scaling is good ({(high_load/low_load - 1)*100:+.1f}% difference)")
    print()
    
    print("=" * 80)
    print("💡 RECOMMENDATIONS TO REACH 20% ERROR")
    print("=" * 80)
    print()
    
    # Generate recommendations
    recommendations = []
    
    if abs(ud_bias) > 15 or abs(rdma_bias) > 15:
        recommendations.append(
            "🔧 CALIBRATION FIX: Significant slowdown bias detected!\n"
            "   → Retrain models with the NEW ideal FCT formula\n"
            "   → This should immediately reduce errors by 10-20pp"
        )
    
    if np.std(all_ud_errors) > 100 or np.std(all_rdma_errors) > 100:
        recommendations.append(
            "📊 VARIANCE REDUCTION: High error variance detected\n"
            "   → Add more training data (especially high-load scenarios)\n"
            "   → Increase model capacity (more hidden units or layers)"
        )
    
    if high_load > low_load * 1.3:
        recommendations.append(
            "🌊 CONTENTION MODELING: Errors increase significantly at high load\n"
            "   → Improve GNN layers (more message passing rounds?)\n"
            "   → Add explicit queue length features"
        )
    
    if len(recommendations) == 0:
        recommendations.append(
            "✨ MODEL IS WELL-CALIBRATED!\n"
            "   → Main path to 20%: Retrain with new ideal FCT formula\n"
            "   → Consider hyperparameter tuning (learning rate, batch size)\n"
            "   → Add data augmentation for rare scenarios"
        )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
        print()
    
    print("=" * 80)
    print("✅ Diagnosis complete!")
    print()

if __name__ == "__main__":
    main()

