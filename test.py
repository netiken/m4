import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_events_with_neighbors(all_events):
    """
    Process a sequence of flow arrival and completion events, tracking neighbors.

    Parameters:
    - all_events (list): List of tuples (event_time, flow_change, links).

    Returns:
    - flow_neighbors (dict): Neighbors of each flow at each hop level.
    """
    link_to_flows = defaultdict(set)  # Map links to active flows
    flow_neighbors = defaultdict(set)  # Track neighbors of each flow

    for event_time, flow_change, links, flow_id in all_events:
        if flow_change == 1:  # Flow arrival
            # Find current flows that share links with this event
            current_flows = set([flow_id])  # Include the new flow itself
            # current_flows.add(flow_id)
            for link in links:
                current_flows.update(link_to_flows[link])

            # Update neighbors for the current flow
            for flow in current_flows:
                flow_neighbors[flow].update(current_flows)
                flow_neighbors[flow].discard(flow)  # Remove self-loop

            # Update link-to-flows mapping
            for link in links:
                link_to_flows[link].add(flow_id)  # Use `event_idx` as a unique flow ID
        else:  # Flow completion
            # Remove the flow from the link mappings
            for link in links:
                if flow_id in link_to_flows[link]:
                    link_to_flows[link].remove(flow_id)
            # Cleanup neighbors (optional but recommended)
            if flow_id in flow_neighbors:
                del flow_neighbors[flow_id]
    return flow_neighbors


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
        visited = set([flow])
        current_hop = set(neighbors)
        hop_counts = []

        while current_hop:
            next_hop = set()
            for neighbor in current_hop:
                if neighbor not in visited:
                    next_hop.update(flow_neighbors[neighbor])
            next_hop -= visited  # Remove already visited flows
            hop_counts.append(len(current_hop))
            visited.update(current_hop)
            current_hop = next_hop

        neighbor_counts.append(hop_counts)

    return neighbor_counts


def plot_cdf(data, labels, xlabel, ylabel, title=""):
    """
    Plot the CDF of data.

    Parameters:
    - data (list): List of lists of values to plot.
    - labels (list): Labels for each dataset.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Optional title for the plot.
    """
    plt.figure(figsize=(6, 4))
    for i, values in enumerate(data):
        values = np.concatenate(values)
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


# Main script
dir_input = "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_7/data"
topo_type = "_topology_flows"
scenario_labels = []

neighbor_counts_per_scenario = []

for scenario_idx in range(3):
    dir_input_tmp = f"{dir_input}/{scenario_idx}/ns3"
    fat = np.load(f"{dir_input_tmp}/fat.npy")
    fct = fat + np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    link_info = np.load(f"{dir_input_tmp}/flow_to_path.npy", allow_pickle=True)

    # Create arrival and completion events
    arrival_events = [(fat[i], 1, set(link_info[i]), i) for i in range(len(fat))]
    completion_events = [(fct[i], -1, set(link_info[i]), i) for i in range(len(fct))]
    all_events = sorted(arrival_events + completion_events, key=lambda x: x[0])

    # Process events to track neighbors
    flow_neighbors = process_events_with_neighbors(all_events)

    # Compute neighbor counts
    neighbor_counts = compute_neighbor_counts_by_hop(flow_neighbors)
    neighbor_counts_per_scenario.append(neighbor_counts)
    scenario_labels.append(f"Scenario-{scenario_idx}")

# Plot CDF of neighbor counts
plot_cdf(
    neighbor_counts_per_scenario,
    scenario_labels,
    xlabel="# of Neighbors (First-hop, Second-hop, etc.)",
    ylabel="CDF (%)",
    title="CDF of Neighbor Counts per Scenario",
)
