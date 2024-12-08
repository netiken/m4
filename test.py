import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict


def process_events(all_events):
    link_to_graph = {}  # Map each link to its graph ID
    graph_flow_count = defaultdict(int)  # Track active flow counts for each graph ID
    next_graph_id = 0  # Unique graph ID counter
    active_flows_over_time = []  # Store results

    for event_time, flow_change, links in all_events:
        # Find all graph IDs associated with the current links
        graph_ids = {link_to_graph[link] for link in links if link in link_to_graph}

        if flow_change == 1:  # Flow arrival
            if not graph_ids:
                # No overlapping graphs, assign a new graph ID
                new_graph_id = next_graph_id
                for link in links:
                    link_to_graph[link] = new_graph_id
                next_graph_id += 1
            else:
                # Merge all graph IDs into one
                new_graph_id = min(graph_ids)
                for graph_id in graph_ids:
                    if graph_id != new_graph_id:
                        # Reassign links to the merged graph ID
                        for link, gid in list(link_to_graph.items()):
                            if gid == graph_id:
                                link_to_graph[link] = new_graph_id
                        # Merge flow counts
                        graph_flow_count[new_graph_id] += graph_flow_count[graph_id]
                        del graph_flow_count[graph_id]

                # Assign the merged graph ID to the current links
                for link in links:
                    link_to_graph[link] = new_graph_id

            graph_flow_count[new_graph_id] += 1
            active_flows_over_time.append(graph_flow_count[new_graph_id])
        else:  # Flow completion
            if not graph_ids:
                print(f"Warning: No graph ID found for links: {links}")
                continue
            graph_id = next(iter(graph_ids))
            graph_flow_count[graph_id] -= 1

            if graph_flow_count[graph_id] == 0:
                # Remove graph if no active flows
                for link in list(link_to_graph):
                    if link_to_graph[link] == graph_id:
                        del link_to_graph[link]
                del graph_flow_count[graph_id]
            else:
                active_flows_over_time.append(graph_flow_count[graph_id])
    return active_flows_over_time


def plot_cdf(data, labels, xlabel, ylabel, title=""):
    plt.figure(figsize=(6, 4))
    for i, values in enumerate(data):
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf * 100, label=labels[i], linewidth=2)

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis="both", which="major", labelsize=15)
    if title:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.show()


lr = 10
UNIT_G = 1e9
MTU = 1000
labels = {
    0: "(0,1KB]",
    1: "(1KB,10KB]",
    2: "(10KB,50KB]",
    3: "(50KB,inf)",
}

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
                    # else:
                    #     print(f"{spec}: {len(fid)}")
                except:
                    continue

print(len(data_list))

res_n_flows_active_spatial = []
for spec_idx, (spec, src_dst_pair_target, topo_type) in enumerate(data_list):
    dir_input_tmp = f"{dir_input}/{spec}"
    fat = np.load(f"{dir_input_tmp}/fat.npy")
    fct = fat + np.load(f"{dir_input_tmp}/fct{topo_type}.npy")

    link_info = np.load(f"{dir_input_tmp}/flow_to_path.npy", allow_pickle=True)

    # Create arrival and completion events
    arrival_events = [(fat[i], 1, set(link_info[i])) for i in range(len(fat))]
    completion_events = [(fct[i], -1, set(link_info[i])) for i in range(len(fct))]

    # Combine and sort events by time
    all_events = sorted(arrival_events + completion_events, key=lambda x: x[0])

    active_flows_over_time = process_events(all_events)

    res_n_flows_active_spatial.append(active_flows_over_time)

plot_cdf(
    res_n_flows_active_spatial,
    ["Train", "Test"],
    "# of Active Flows over Time",
    "CDF (%)",
)
