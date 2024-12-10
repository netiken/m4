import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_events_and_compute_neighbors(all_events, max_hops=3):
    """
    Optimized processing of flow events to compute neighbor counts up to max_hops.

    Parameters:
    - all_events (list): List of tuples (event_time, flow_change, links, flow_id).
    - max_hops (int): Maximum number of hops to calculate neighbors.

    Returns:
    - neighbor_counts_by_flow (dict): Dictionary where keys are flow IDs and values are lists of neighbor counts per hop.
    """
    link_to_flows = defaultdict(set)  # Map links to active flows
    flow_neighbors = defaultdict(set)  # Map each flow to its neighbors
    neighbor_counts_by_flow = defaultdict(list)  # Store neighbor counts by flow

    for event_time, flow_change, links, flow_id in all_events:
        affected_flows = set()

        if flow_change == 1:  # Flow arrival
            current_flows = {flow_id}
            for link in links:
                current_flows.update(link_to_flows[link])

            # Update neighbors for current flows
            for flow in current_flows:
                flow_neighbors[flow].update(current_flows)
                flow_neighbors[flow].discard(flow)
                affected_flows.add(flow)

            # Update link-to-flows mapping
            for link in links:
                link_to_flows[link].add(flow_id)

        elif flow_change == -1:  # Flow completion
            for link in links:
                link_to_flows[link].discard(flow_id)
            if flow_id in flow_neighbors:
                # Remove completed flow and adjust neighbors
                for neighbor in flow_neighbors[flow_id]:
                    if flow_id in flow_neighbors[neighbor]:
                        flow_neighbors[neighbor].discard(flow_id)
                        affected_flows.add(neighbor)
                flow_neighbors.pop(flow_id, None)

        # Compute neighbor counts for affected flows
        for flow in affected_flows:
            neighbors = flow_neighbors.get(flow, set())
            visited = {flow}
            current_hop = neighbors.copy()
            hop_counts = []

            for _ in range(max_hops):
                if not current_hop:
                    break
                hop_counts.append(len(current_hop))
                next_hop = {
                    n
                    for neighbor in current_hop
                    for n in flow_neighbors.get(neighbor, set())
                    if n not in visited
                }
                visited.update(current_hop)
                current_hop = next_hop

            # If no neighbors or no hops processed, default to [0]
            neighbor_counts_by_flow[flow] = hop_counts if hop_counts else [0]

    return neighbor_counts_by_flow


def plot_per_hop_neighbors(res_total, max_hops=3, scenario_labels=None):
    """
    Plot the per-hop neighbors for all scenarios in separate figures.

    Parameters:
    - res_total (list): List of results for each scenario.
    - max_hops (int): Number of hops to include in the plot.
    - scenario_labels (list): List of labels for each scenario.
    """
    for hop in range(max_hops):
        plt.figure(figsize=(6, 4))
        for scenario_idx, scenario_data in enumerate(res_total):
            all_hop_neighbors = [
                counts[hop] if hop < len(counts) else 0
                for neighbor_counts in scenario_data
                for counts in neighbor_counts.values()
            ]
            sorted_values = np.sort(all_hop_neighbors)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            label = (
                scenario_labels[scenario_idx]
                if scenario_labels
                else f"Scenario-{scenario_idx}"
            )
            plt.plot(sorted_values, cdf * 100, label=label, linewidth=2)

        plt.xlabel(f"# of Neighbors at Hop {hop + 1}", fontsize=15)
        plt.ylabel("CDF (%)", fontsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.title(f"Per-Hop Neighbors (Hop {hop + 1})", fontsize=15)
        plt.show()


# Main Script
res_total = []
dir_input = "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_train"
topo_type = "_topology_flows"
data_list = []
sampled_list = np.random.choice(np.arange(4000), 100, replace=False)

for shard in sampled_list:
    for n_flows in [2000]:
        for n_hosts in [32]:
            for sample in [0]:
                topo_type_cur = topo_type
                spec = f"{shard}/ns3"
                try:
                    fid = np.load(f"{dir_input}/{spec}/fid{topo_type_cur}.npy")
                    if (
                        len(fid) == len(set(fid))
                        and np.all(fid[:-1] <= fid[1:])
                        and len(fid) % n_flows == 0
                    ):
                        data_list.append((spec, (0, n_hosts - 1), topo_type_cur))
                except:
                    continue

print(len(data_list))

res_n_flows_active = []
for spec_idx, (spec, src_dst_pair_target, topo_type) in enumerate(data_list):
    dir_input_tmp = f"{dir_input}/{spec}"
    fat = np.load(f"{dir_input_tmp}/fat.npy")
    fct = fat + np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    link_info = np.load(f"{dir_input_tmp}/flow_to_path.npy", allow_pickle=True)

    arrival_events = [(fat[i], 1, set(link_info[i]), i) for i in range(len(fat))]
    completion_events = [(fct[i], -1, set(link_info[i]), i) for i in range(len(fct))]
    all_events = sorted(arrival_events + completion_events, key=lambda x: x[0])

    neighbor_counts = process_events_and_compute_neighbors(all_events)
    res_n_flows_active.append(neighbor_counts)

res_total.append(res_n_flows_active)

# Scenario Labels
scenario_labels = [f"Scenario {i+1}" for i in range(len(res_total))]

# Plot Results
plot_per_hop_neighbors(res_total, max_hops=3, scenario_labels=scenario_labels)
