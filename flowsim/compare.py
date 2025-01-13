import matplotlib.pyplot as plt
import numpy as np
import math

new_m4 = np.load("test_m4.npy")
m4 = np.load("validate.npy")
ns3 = np.load("eval_test/ns3/fct_topology_flows.npy")
test_m4_no = np.load("test_m4_no.npy")
#flowsim = np.load("eval_test/ns3/flowsim_fct.npy")
#script_flowsim = np.load("flowsim_fct.npy")
ideal = np.load("eval_test/ns3/fct_i_topology_flows.npy")
#m4_with_new_flowsim = np.load("validate_new_flowsim.npy")
#m4_with_flowsim = np.load("test_m4.npy")
#new_flowsim = np.load("new_fct.npy")
#print(len(new_m4))
#print(len(m4))
#print(len(ideal))

#new_m4 = new_m4[:2000]
#m4 = m4[:2000]
#flowsim = flowsim[:2000]
#ideal = ideal[:2000]
#ns3 = ns3[:2000]
#m4_with_new_flowsim = m4_with_new_flowsim[:2000]
#m4_with_flowsim = m4_with_flowsim[:2000]

#print(flowsim[:10])
#print(script_flowsim[:10])
#print(new_flowsim[:10])
#print(flowsim[:10] / ideal[:10])
#print(script_flowsim[:10] / ideal[:10])

print((m4 / ideal)[:10])
print((new_m4 / ideal)[:10])
#print(flowsim / ideal)

_, ax = plt.subplots()

#for i in range(len(new_m4)):
#    if math.isnan(new_m4[i]):
#        print(i)

print(len(new_m4))
print(len(m4))

baseline = np.load("/data1/lichenni/projects/per-flow-sim/res/m4_noflowsim_7eval_test.npz")
baseline = baseline["fct"][0, :, 0]

ax.ecdf(new_m4 / ideal, label="new m4")
ax.ecdf(ns3 / ideal, label="ns3")
#ax.ecdf(flowsim / ideal, label="flowsim")
ax.ecdf(m4 / ideal, label="m4")
ax.ecdf(test_m4_no / ideal, label="m4 no flowsim")
ax.ecdf(baseline / ideal, label="canon", linestyle='dashed')
#ax.ecdf(script_flowsim / ideal, label="script flowsim", linestyle='dashed')
plt.xscale('log')
#ax.ecdf(m4_with_flowsim / ideal, label="m4 with flowsim")
#ax.ecdf(m4_with_new_flowsim / ideal, label="m4 with new flowsim")

plt.legend()
plt.savefig("valid.png")
