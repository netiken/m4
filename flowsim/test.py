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

fid = np.load("app/ns3/fid_topology_flows.npy")
ideal = np.load("app/ns3/fct_i_topology_flows.npy")
ns3 = np.load("app/ns3/fct_topology_flows.npy")
#flowsim = np.load("app_flowsim.npy")
#m4 = np.load("app_m4.npy")
m4 = np.load("re_app.npy")[fid]
flowsim = np.load("app_flowsim_1.npy")[fid]
#limit = np.load("app_flowsim_1.npy")

norm_flowsim = flowsim / ideal
norm_m4 = m4 / ideal

_, ax = plt.subplots()

ax.ecdf(ns3 / ideal, label="ns3")
ax.ecdf(np.where(norm_flowsim < 1.0, 1.0, norm_flowsim), label="flowsim")
ax.ecdf(np.where(norm_m4 < 1.0, 1.0, norm_m4), label="m4")
#ax.ecdf(flowsim / ideal, label="flowsim")
#ax.ecdf(m4 / ideal, label="m4")
#ax.ecdf(test / ideal, label="m4 again", linestyle='dashed')
#ax.ecdf(limit / ideal, label="flowsim limit", linestyle='dashed')
plt.xscale('log')

plt.legend()
plt.savefig("limit.png")
