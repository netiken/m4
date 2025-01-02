import matplotlib.pyplot as plt
import numpy as np

fsize = np.load("0/ns3/fsize.npy")
real = np.load("0/ns3/fct_topology_flows_dctcp.npy")
ideal = np.load("0/ns3/fct_i_topology_flows_dctcp.npy")

flowsim = np.load("flowsim.npy")
new_flowsim = np.load("out.npy")


#print(real[:10])
#print(ideal[:10])
#print(real[:10])
#print(flowsim[:10])
#print(flowsim[:10])
#print(new_flowsim[:10])

kb = []
fifkb = []
twokb = []
mb = []

for i in range(len(fsize)):
    size = fsize[i]
    if i <= 1000:
        kb.append(i)
    elif i <= 50000:
        fifkb.append(i)
    elif i <= 200000:
        twokb.append(i)
    else:
        mb.append(i)

_, ax = plt.subplots()

m4_slowdown = new_flowsim / ideal
real_slowdown = real / ideal
flowsim_slowdown = flowsim / ideal

print(len(kb))

baseline = np.load("baseline.npy")
new = np.load("new.npy")
test_link = np.load("test_link.npy")

ax.ecdf(m4_slowdown, label="new approach: flowsim and m4 together")
ax.ecdf(real_slowdown, label="ns3")
ax.ecdf(flowsim_slowdown, label="flowsim")
ax.ecdf(baseline / ideal, label="current m4")
ax.ecdf(test_link / ideal, label="new flowsim and m4")
#ax.ecdf(new / ideal, label="flowsim into m4")
plt.xlabel("Slowdown")
plt.title("CDF of slowdowns")
plt.legend()

plt.savefig("results_kb.png")
plt.clf()

_, ax = plt.subplots()
print(len(fifkb))

ax.ecdf(m4_slowdown[fifkb], label="m4+flowsim")
ax.ecdf(real_slowdown[fifkb], label="ns3")
ax.ecdf(flowsim_slowdown[fifkb], label="flowsim")
plt.legend()

plt.savefig("results_fifkb.png")

