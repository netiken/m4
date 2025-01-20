import matplotlib.pyplot as plt
import numpy as np
import math

#new_m4 = np.load("test_m4.npy")
#m4 = np.load("one.npy")
#limit = np.load("two.npy")
#m4 = np.load("/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_test/2/ns3/m4_fct.npy")
#ideal = np.load("1/ns3/fct_i_topology_flows.npy")
#ns3 = np.load("1/ns3/fct_topology_flows.npy")
#flowsim = np.load("valid_flowsim.npy")
#test_flowsim = np.load("zero.npy")
#two = np.load("flowsim_two.npy")
#valid = np.load("valid.npy")

ideal = np.load("app/ns3/fct_i_topology_flows.npy")
ns3 = np.load("app/ns3/fct_topology_flows.npy")
flowsim = np.load("app_flowsim.npy")
m4 = np.load("app_m4.npy")

_, ax = plt.subplots()

ax.ecdf(ns3 / ideal, label="ns3")
ax.ecdf(flowsim / ideal, label="flowsim")
ax.ecdf(m4 / ideal, label="m4")

#baseline = np.load("/data1/lichenni/projects/per-flow-sim/res/m4_noflowsim_7_large0.npz")
#baseline = np.load("/data1/lichenni/projects/per-flow-sim/res/new_loss01_mlp1_10_large0.npz")
#baseline = baseline["fct"][1, :, 0]

#test = np.load("large_test.npy")

#print(min(limit))

#ax.ecdf(ns3 / ideal, label="ns3")
#ax.ecdf(m4 / ideal, label="m4")
#ax.ecdf(flowsim / ideal, label="flowsim")
#ax.ecdf(test_flowsim / ideal, label="zero", linestyle='dashed')
#ax.ecdf(two / ideal, label="two")
#ax.ecdf(baseline / ideal, label="canon", linestyle='dashed')
#ax.ecdf(limit / ideal, label="limit", linestyle='dashed')
#ax.ecdf(valid / ideal, label="validation", linestyle='dashed')
plt.xscale('log')

plt.legend()
plt.savefig("limit.png")
