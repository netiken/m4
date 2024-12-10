import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_events_and_compute_neighbors(all_events, max_hops=3):
    """
    Optimized processing of flow events to compute neighbor counts up to max_hops.

    Parameters:
    - all_events (list): List of tuples (event_time, flow_change, links, flow_id).
      Each tuple represents an event:
        event_time: The timestamp of the event
        flow_change: +1 for arrival, -1 for completion
        links: A set or iterable of link IDs involved in this event
        flow_id: The ID of the flow arriving or completing

    - max_hops (int): Maximum number of hops to calculate neighbors.

    Returns:
    - neighbor_counts_by_flow (dict): Dictionary where keys are flow IDs and values are lists of neighbor counts per hop.
      For each flow:
        - hop_counts: A list of integers, each representing the number of neighbors at that hop level.
        - If a flow has no neighbors at any hop, it defaults to [0].
    """
    link_to_flows = defaultdict(set)  # Map each link to active flows
    flow_neighbors = defaultdict(set)  # Map each flow to its neighbors
    neighbor_counts_by_flow = defaultdict(
        list
    )  # Store neighbor counts per hop for each flow

    for event_time, flow_change, links, flow_id in all_events:
        affected_flows = set()

        if flow_change == 1:  # Flow arrival
            # For each link involved, add the new flow, then update neighbors for all flows on that link
            for link in links:
                # Add the arriving flow to this link's active flows
                link_to_flows[link].add(flow_id)
                flows_on_link = link_to_flows[link].copy()

                # Update neighbors for flows on this link: each flow on this link becomes neighbors with all others on the same link
                for f in flows_on_link:
                    flow_neighbors[f].update(flows_on_link)
                    flow_neighbors[f].discard(f)  # Remove self to avoid loops

                # All flows on this link are affected by the arrival of the new flow
                affected_flows.update(flows_on_link)

        elif flow_change == -1:  # Flow completion
            # Remove the completed flow from each link
            for link in links:
                link_to_flows[link].discard(flow_id)

            # If the completed flow was known in flow_neighbors, remove it and update affected flows
            if flow_id in flow_neighbors:
                old_neighbors = flow_neighbors[flow_id]
                flow_neighbors.pop(flow_id, None)
                # Remove the completed flow from all its neighbors
                for neighbor in old_neighbors:
                    if flow_id in flow_neighbors[neighbor]:
                        flow_neighbors[neighbor].discard(flow_id)
                        affected_flows.add(neighbor)
                # The completed flow itself is considered affected for this event
                affected_flows.add(flow_id)

        # Now calculate neighbor counts for all affected flows
        for flow in affected_flows:
            neighbors = flow_neighbors.get(flow, set())
            visited = {flow}
            current_hop = neighbors.copy()
            hop_counts = []

            # Perform a BFS-like multi-hop neighbor exploration
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

            # If no neighbors at any hop, default to [0]
            neighbor_counts_by_flow[flow] = hop_counts if hop_counts else [0]

    return neighbor_counts_by_flow


def plot_cdf_for_hops(res_total, max_hops=3, dataset_labels=None):
    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(len(res_total))]

    for hop_idx in range(max_hops):
        plt.figure(figsize=(6, 4))
        for dataset_idx, scenario_data in enumerate(res_total):
            # scenario_data is a list of dictionaries
            # each dictionary: {flow_id: [hop_counts]}
            hop_values = []
            for neighbor_counts_by_flow in scenario_data:
                # neighbor_counts_by_flow is a dict {flow_id: [hop_counts]}
                for hop_counts in neighbor_counts_by_flow.values():
                    # hop_counts is a list of integers
                    # Check if hop_idx < len(hop_counts)
                    if hop_idx < len(hop_counts):
                        hop_values.append(hop_counts[hop_idx])
                    else:
                        hop_values.append(0)

            if len(hop_values) == 0:
                # No data for this hop, skip plotting
                continue

            sorted_values = np.sort(hop_values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            label = dataset_labels[dataset_idx]
            plt.plot(sorted_values, cdf * 100, label=label, linewidth=2)

        plt.xlabel(f"# of Neighbors at Hop {hop_idx + 1}", fontsize=15)
        plt.ylabel("CDF (%)", fontsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.title(f"Per-Hop Neighbors (Hop {hop_idx + 1})", fontsize=15)
        plt.show()


# Main Script
res_total = []
dir_input = "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_train"
topo_type = "_topology_flows"
data_list = []
sampled_list = np.random.choice(np.arange(4000), 2, replace=False)

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
