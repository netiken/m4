# inference.py

import ctypes
import numpy as np
import os
from collections import defaultdict
import yaml

# Load the shared library
# Adjust the path if the library is located elsewhere
lib = ctypes.CDLL("../build/libinference_shared.so")  # Ensure the correct library name

config_file = "../../config/test_config_lstm_topo_cplusplus.yaml"
config = yaml.safe_load(open(config_file, "r"))
config_dataset = config["dataset"]
config_model = config["model"]

# Set parameters
gpu_id = 0
rtt = 0.0
hidden_size = config_model["hidden_size"]
enable_flowsim = config_dataset["enable_flowsim_diff"]
n_links = config_dataset["n_links_max"]
# n_flows = len(size)
n_flows = 10

# Define the argument and return types of the interactive_inference function
lib.interactive_inference.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # size
    ctypes.POINTER(ctypes.c_float),  # fat
    ctypes.POINTER(ctypes.c_float),  # i_fct
    ctypes.POINTER(ctypes.c_float),  # fct
    ctypes.POINTER(ctypes.c_float),  # sldn_flowsim
    ctypes.POINTER(ctypes.c_int),  # flowid_to_linkid_offsets
    ctypes.POINTER(ctypes.c_int),  # flowid_to_linkid_flat
    ctypes.POINTER(ctypes.c_int),  # edges_flow_ids (new)
    ctypes.POINTER(ctypes.c_int),  # edges_link_ids (new)
    ctypes.c_int,  # n_links
    ctypes.c_int,  # n_flows
    ctypes.c_int,  # n_edges
    ctypes.POINTER(ctypes.c_float),  # res_fct
    ctypes.POINTER(ctypes.c_float),  # res_sldn
    ctypes.c_int,  # gpu_id
    ctypes.c_int,  # h_vec_dim
    ctypes.c_float,  # rtt
    ctypes.c_bool,  # enable_flowsim
]
lib.interactive_inference.restype = ctypes.c_int

# Prepare input data
data_dir = "/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data_empirical/0/ns3"
config_str = "_topology_flows_dctcp"

# Load numpy arrays
size = np.load(os.path.join(data_dir, "fsize.npy")).astype(np.float32)
fat = np.load(os.path.join(data_dir, "fat.npy"))
fat = fat - fat[0]
fat = fat.astype(np.float32)
fct = np.load(os.path.join(data_dir, f"fct{config_str}.npy")).astype(np.float32)
i_fct = np.load(os.path.join(data_dir, f"fct_i{config_str}.npy")).astype(np.float32)
fct_flowsim = np.load(os.path.join(data_dir, "flowsim_fct.npy")).astype(np.float32)

sldn_flowsim = np.divide(fct_flowsim, i_fct)

# Load link_list and flow_to_path
link_list = np.load(os.path.join(data_dir, "flink.npy"))  # Shape: (n_links,)
link_dict = {link: idx for idx, link in enumerate(link_list)}
flow_to_path = np.load(os.path.join(data_dir, "flow_to_path.npy"), allow_pickle=True)
link_info = [
    flow_to_path[i] for i in range(len(size))
]  # Assuming flow_to_path[i] gives the list of link pairs for flow i

# Build flowid_to_linkid and edges_list
flowid_to_linkid = defaultdict(list)
edges_list = []
for flow_idx in range(len(size)):
    for link_pair in link_info[flow_idx]:
        link_id = link_dict.get(link_pair, -1)  # Handle missing links gracefully
        edges_list.append([flow_idx, link_id])
        flowid_to_linkid[flow_idx].append(link_id)

# Separate edges_list into flow_ids and link_ids
edges_flow_ids = np.array([pair[0] for pair in edges_list], dtype=np.int32)
edges_link_ids = np.array([pair[1] for pair in edges_list], dtype=np.int32)


# Calculate flowid_to_linkid_offsets and flowid_to_linkid_flat
def calculate_flowid_to_linkid_offsets_flat(flowid_to_linkid, n_flows):
    """
    Calculate flowid_to_linkid_offsets and flowid_to_linkid_flat from flowid_to_linkid.

    Args:
        flowid_to_linkid (dict): Mapping from flow_id to list of link_ids.
        n_flows (int): Total number of flows.

    Returns:
        tuple: (flowid_to_linkid_offsets, flowid_to_linkid_flat)
    """
    flowid_to_linkid_offsets = np.zeros(n_flows + 1, dtype=np.int32)
    flowid_to_linkid_flat = []

    current_offset = 0
    for flow_id in range(n_flows):
        flowid_to_linkid_offsets[flow_id] = current_offset
        links = flowid_to_linkid.get(flow_id, [])
        flowid_to_linkid_flat.extend(links)
        current_offset += len(links)
    flowid_to_linkid_offsets[n_flows] = current_offset

    flowid_to_linkid_flat = np.array(flowid_to_linkid_flat, dtype=np.int32)
    return flowid_to_linkid_offsets, flowid_to_linkid_flat


flowid_to_linkid_offsets, flowid_to_linkid_flat = (
    calculate_flowid_to_linkid_offsets_flat(flowid_to_linkid, len(size))
)

n_edges = len(flowid_to_linkid_flat)
size = size[:n_flows]
fat = fat[:n_flows]
fct = fct[:n_flows]
i_fct = i_fct[:n_flows]
sldn_flowsim = sldn_flowsim[:n_flows]
flowid_to_linkid_offsets = flowid_to_linkid_offsets[: n_flows + 1]

# Adjust fat to start from zero
# Prepare output arrays
res_fct = np.zeros(n_flows * 2, dtype=np.float32)
res_sldn = np.zeros(n_flows * 2, dtype=np.float32)

# Convert numpy arrays to ctypes pointers
size_ptr = size.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
fat_ptr = fat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
i_fct_ptr = i_fct.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
fct_ptr = fct.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
sldn_flowsim_ptr = sldn_flowsim.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
flowid_to_linkid_offsets_ptr = flowid_to_linkid_offsets.ctypes.data_as(
    ctypes.POINTER(ctypes.c_int)
)
flowid_to_linkid_flat_ptr = flowid_to_linkid_flat.ctypes.data_as(
    ctypes.POINTER(ctypes.c_int)
)
edges_flow_ids_ptr = edges_flow_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))  # New
edges_link_ids_ptr = edges_link_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))  # New
res_fct_ptr = res_fct.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
res_sldn_ptr = res_sldn.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Call the interactive_inference function
result = lib.interactive_inference(
    size_ptr,
    fat_ptr,
    i_fct_ptr,
    fct_ptr,
    sldn_flowsim_ptr,
    flowid_to_linkid_offsets_ptr,
    flowid_to_linkid_flat_ptr,
    edges_flow_ids_ptr,  # New
    edges_link_ids_ptr,  # New
    n_links,
    n_flows,
    n_edges,
    res_fct_ptr,
    res_sldn_ptr,
    gpu_id,
    hidden_size,
    rtt,
    enable_flowsim,
)

# Check the result
if result == 0:
    # Reshape the results
    res_fct = res_fct.reshape((n_flows, 2))
    res_sldn = res_sldn.reshape((n_flows, 2))

    # Display the first 10 results
    print("First 10 FCT Results:")
    for i in range(max(-10, -n_flows), 0):
        predicted_fct = res_fct[i, 0]
        actual_fct = res_fct[i, 1]
        print(
            f"Flow ID {i}: Predicted FCT = {predicted_fct}, Actual FCT = {actual_fct}"
        )

    print("\nFirst 10 SLDN Results:")
    for i in range(max(-10, -n_flows), 0):
        predicted_sldn = res_sldn[i, 0]
        actual_sldn = res_sldn[i, 1]
        print(
            f"Flow ID {i}: Predicted SLDN = {predicted_sldn}, Actual SLDN = {actual_sldn}"
        )

    # Save the results to a file
    # np.savez(
    #     "../../res/inference_link_cplusplus_empirical.npz",
    #     fct=res_fct,
    #     sldn=res_sldn,
    # )
    print("Results saved to '../../res/inference_link_cplusplus_empirical.npz'")
else:
    print("An error occurred during inference.")
