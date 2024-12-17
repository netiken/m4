import numpy as np

real = np.load("87/ns3/fct_topology_flows_dctcp.npy")
ideal = np.load("87/ns3/fct_i_topology_flows_dctcp.npy")

flowsim = np.load("87/ns3/flowsim_fct.npy")
new_flowsim = np.load("out.npy")

#print(real[:10])
#print(ideal[:10])
print(flowsim[:10])
print(new_flowsim[:10])
