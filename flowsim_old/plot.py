import matplotlib.pyplot as plt
import numpy as np

ns3 = np.load("88/ns3/fct_topology_flows_dctcp.npy")

flowsim = np.load("out.npy")

fig, ax = plt.subplots()

ax.ecdf(ns3, label="ns3")
ax.ecdf(flowsim, label="flowsim")
plt.savefig("test.png")
