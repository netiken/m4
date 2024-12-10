import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_events_and_compute_neighbors(all_events, max_hops=3):
    """
    Process flow arrival and completion events, calculating neighbor counts up to max_hops
    for all affected flows at each event, and returning a list of snapshots with these counts.

    Parameters:
    - all_events (list): List of tuples (event_time, flow_change, links, flow_id).
      Each tuple is:
        event_time: The timestamp of the event
        flow_change: +1 for arrival, -1 for completion
        links: A set or iterable of link IDs involved in this event
        flow_id: The ID of the arriving or completing flow
    - max_hops (int): Maximum number of hops to calculate neighbors.

    Returns:
    - res_snapshots (list): A list of dictionaries, one for each event processed.
      Each dictionary maps:
         flow_id -> hop_counts (list of integers)
      If a flow has no neighbors at any hop, hop_counts defaults to [0].
    """
    link_to_flows = defaultdict(set)  # Map each link to active flows
    flow_neighbors = defaultdict(set)  # Map each flow to its neighbors
    res_snapshots = []  # Store a snapshot of neighbor counts for each event

    for event_time, flow_change, links, flow_id in all_events:
        affected_flows = set()

        if flow_change == 1:  # Flow arrival
            # For each link, add the new flow and update neighbors just for flows on that link
            for link in links:
                link_to_flows[link].add(flow_id)
                flows_on_link = link_to_flows[link].copy()

                # Update neighbors for flows on this link
                for f in flows_on_link:
                    flow_neighbors[f].update(flows_on_link)
                    flow_neighbors[f].discard(f)  # Remove self
                affected_flows.update(flows_on_link)

        elif flow_change == -1:  # Flow completion
            # Remove the completed flow from its links
            for link in links:
                link_to_flows[link].discard(flow_id)

            # If we know about this flow, remove it and adjust neighbors
            if flow_id in flow_neighbors:
                old_neighbors = flow_neighbors[flow_id]
                flow_neighbors.pop(flow_id, None)
                for neighbor in old_neighbors:
                    if flow_id in flow_neighbors[neighbor]:
                        flow_neighbors[neighbor].discard(flow_id)
                        affected_flows.add(neighbor)
                affected_flows.add(flow_id)

        # Compute neighbor counts for all affected flows at this event
        event_counts = {}
        for flow in affected_flows:
            neighbors = flow_neighbors.get(flow, set())
            visited = {flow}
            current_hop = neighbors.copy()
            hop_counts = []

            # BFS-like multi-hop neighbor exploration
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

            # Default to [0] if no neighbors found
            hop_counts = hop_counts if hop_counts else [0]
            event_counts[flow] = hop_counts

        # Record the snapshot for this event
        res_snapshots.append(event_counts)

    return res_snapshots


def plot_cdf_for_hops(res_total, max_hops=3, dataset_labels=None):
    """
    Plot the per-hop neighbors for all datasets from the snapshots returned by process_events_and_compute_neighbors.

    Parameters:
    - res_total (list): A list of datasets, where each dataset is a list of "event_counts".
      Each "event_counts" is a list of snapshot dictionaries.
      Each snapshot dictionary: {flow_id: [hop_counts]}.
    - max_hops (int): Number of hop levels to plot.
    - dataset_labels (list): Labels for each dataset.
    """
    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(len(res_total))]

    for hop_idx in range(max_hops):
        plt.figure(figsize=(6, 4))
        for dataset_idx, scenario_data in enumerate(res_total):
            # scenario_data: a list of event_counts
            # event_counts: a list of snapshot dicts
            # snapshot_dict: {flow_id: [hop_counts]}

            hop_values = []
            for event_counts in scenario_data:
                # event_counts is a list of dictionaries
                for snapshot_dict in event_counts:
                    # snapshot_dict: {flow_id: hop_counts}
                    for hop_counts_array in snapshot_dict.values():
                        if hop_idx < len(hop_counts_array):
                            hop_values.append(hop_counts_array[hop_idx])
                        else:
                            hop_values.append(0)

            if len(hop_values) == 0:
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
dataset_labels = ["Train", "Test", "m4-train"]

# Plot Results
plot_cdf_for_hops(res_total, max_hops=3, dataset_labels=dataset_labels)
