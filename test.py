import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_events_with_neighbors(all_events):
    link_to_flows = defaultdict(set)
    flow_neighbors = defaultdict(set)
    snapshots = []

    for event_time, flow_change, links, flow_id in all_events:
        if flow_change == 1:
            current_flows = {flow_id}
            for link in links:
                current_flows.update(link_to_flows[link])

            for flow in current_flows:
                flow_neighbors[flow].update(current_flows)
                flow_neighbors[flow].discard(flow)

            for link in links:
                link_to_flows[link].add(flow_id)
        else:
            for link in links:
                link_to_flows[link].discard(flow_id)
            # flow_neighbors.pop(flow_id, None)

        active_flows = {flow for flows in link_to_flows.values() for flow in flows}
        if len(active_flows) > 1:
            snapshots.append(
                {
                    "time": event_time,
                    "active_flows": active_flows,
                    "flow_neighbors": {
                        flow: neighbors.copy()
                        for flow, neighbors in flow_neighbors.items()
                    },
                }
            )

    return snapshots


def compute_neighbor_counts_by_hop(flow_neighbors):
    """
    Compute neighbor counts for each flow by hop.

    Parameters:
    - flow_neighbors (dict): Neighbors of each flow.

    Returns:
    - neighbor_counts (list): List of lists containing neighbor counts by hop.
    """
    neighbor_counts = []

    for flow, neighbors in flow_neighbors.items():
        if not neighbors:  # Skip flows with no neighbors
            neighbor_counts.append([0])
            continue

        visited = {flow}  # Start with the current flow as visited
        current_hop = set(neighbors)
        hop_counts = []

        while current_hop:
            hop_counts.append(len(current_hop))  # Count current hop
            next_hop = set()
            for neighbor in current_hop:
                next_hop.update(flow_neighbors[neighbor] - visited)
            visited.update(current_hop)
            current_hop = next_hop

        neighbor_counts.append(hop_counts)

    return neighbor_counts


def plot_cdf(data, labels, xlabel, ylabel, title=""):
    plt.figure(figsize=(6, 4))
    for i, values in enumerate(data):
        values = np.concatenate(values)
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cdf * 100, label=labels[i], linewidth=2)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()


res_total = []
dir_input = "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_train"
topo_type = "_topology_flows"
data_list = []
sampled_list = np.random.choice(np.arange(4000), 100, replace=False)
# sampled_list = np.arange(4000)
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

res_n_flows_active = []
for spec_idx, (spec, src_dst_pair_target, topo_type) in enumerate(data_list):
    dir_input_tmp = f"{dir_input}/{spec}"
    fat = np.load(f"{dir_input_tmp}/fat.npy")
    fct = fat + np.load(f"{dir_input_tmp}/fct{topo_type}.npy")

    link_info = np.load(f"{dir_input_tmp}/flow_to_path.npy", allow_pickle=True)

    # Create arrival and completion events
    arrival_events = [(fat[i], 1, set(link_info[i]), i) for i in range(len(fat))]
    completion_events = [(fct[i], -1, set(link_info[i]), i) for i in range(len(fct))]

    # Combine and sort events by time
    all_events = sorted(arrival_events + completion_events, key=lambda x: x[0])

    snapshots = process_events_with_neighbors(all_events)
    scenario_counts = [
        compute_neighbor_counts_by_hop(snapshot["flow_neighbors"])
        for snapshot in snapshots
    ]

    res_n_flows_active.append(scenario_counts)

res_total.append(res_n_flows_active)

dir_input = (
    "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_test_8k"
)
topo_type = "_topology_flows"
data_list = []
for shard in np.arange(100):
    for n_hosts in [32]:
        spec = f"{shard}/ns3"
        fid_path = f"{dir_input}/{spec}/fid{topo_type}.npy"
        if os.path.exists(fid_path):
            fid = np.load(f"{dir_input}/{spec}/fid{topo_type}.npy")
            if len(fid) == len(set(fid)) and np.all(fid[:-1] <= fid[1:]):
                data_list.append((spec, (0, n_hosts - 1), topo_type))
            # else:
            #     print(f"{spec}: {len(fid)}")

print(len(data_list))

res_n_flows_active = []
for spec_idx, (spec, src_dst_pair_target, topo_type) in enumerate(data_list):
    dir_input_tmp = f"{dir_input}/{spec}"
    fat = np.load(f"{dir_input_tmp}/fat.npy")
    fct = fat + np.load(f"{dir_input_tmp}/fct{topo_type}.npy")

    link_info = np.load(f"{dir_input_tmp}/flow_to_path.npy", allow_pickle=True)

    # Create arrival and completion events
    arrival_events = [(fat[i], 1, set(link_info[i]), i) for i in range(len(fat))]
    completion_events = [(fct[i], -1, set(link_info[i]), i) for i in range(len(fct))]

    # Combine and sort events by time
    all_events = sorted(arrival_events + completion_events, key=lambda x: x[0])

    snapshots = process_events_with_neighbors(all_events)
    scenario_counts = [
        compute_neighbor_counts_by_hop(snapshot["flow_neighbors"])
        for snapshot in snapshots
    ]
    res_n_flows_active.append(scenario_counts)

res_total.append(res_n_flows_active)

dir_input = "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_train"
data_path = "/data2/lichenni/output_perflow/final_param_shard4000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/data_list.json"
data_list = json.load(open(data_path))
data_list = data_list["train"]
flow_size_threshold = 100000000
res_n_flows_active = []
for spec_idx, (spec, src_dst_pair_target, topo_type, segment_id, _) in enumerate(
    data_list
):
    dir_input_tmp = f"{dir_input}/{spec}"
    busy_periods = np.load(
        f"{dir_input_tmp}/period{topo_type}_t{flow_size_threshold}.npy",
        allow_pickle=True,
    )
    fid = np.array(busy_periods[segment_id])

    fat = np.load(f"{dir_input_tmp}/fat.npy")[fid]
    fct = fat + np.load(f"{dir_input_tmp}/fct{topo_type}.npy")[fid]

    link_info = np.load(f"{dir_input_tmp}/flow_to_path.npy", allow_pickle=True)

    # Create arrival and completion events
    arrival_events = [(fat[i], 1, set(link_info[i]), i) for i in range(len(fat))]
    completion_events = [(fct[i], -1, set(link_info[i]), i) for i in range(len(fct))]

    # Combine and sort events by time
    all_events = sorted(arrival_events + completion_events, key=lambda x: x[0])

    snapshots = process_events_with_neighbors(all_events)
    scenario_counts = [
        compute_neighbor_counts_by_hop(snapshot["flow_neighbors"])
        for snapshot in snapshots
    ]
    res_n_flows_active.append(scenario_counts)

res_total.append(res_n_flows_active)
