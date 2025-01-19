import matplotlib.pyplot as plt
import numpy as np
import math

#new_m4 = np.load("test_m4.npy")
m4 = np.load("one.npy")
limit = np.load("two.npy")
#m4 = np.load("/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_test/2/ns3/m4_fct.npy")
ideal = np.load("1/ns3/fct_i_topology_flows.npy")
ns3 = np.load("1/ns3/fct_topology_flows.npy")

_, ax = plt.subplots()


#baseline = np.load("/data1/lichenni/projects/per-flow-sim/res/m4_noflowsim_7_large0.npz")
#baseline = np.load("/data1/lichenni/projects/per-flow-sim/res/new_loss01_mlp1_10_large0.npz")
#baseline = baseline["fct"][1, :, 0]

#test = np.load("large_test.npy")

ax.ecdf(ns3 / ideal, label="ns3")
ax.ecdf(m4 / ideal, label="m4")
#ax.ecdf(baseline / ideal, label="canon", linestyle='dashed')
#`ax.ecdf(limit / ideal, label="limit")
plt.xscale('log')

plt.legend()
plt.savefig("limit.png")
