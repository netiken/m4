import matplotlib.pyplot as plt
import numpy as np

fat = np.load("87/ns3/fsize.npy")
print(fat[0], fat[1])
real = np.load("87/ns3/fct_topology_flows_dctcp.npy")
ideal = np.load("87/ns3/fct_i_topology_flows_dctcp.npy")

flowsim = np.load("87/ns3/flowsim_fct.npy")
new_flowsim = np.load("out.npy")

#print(real[:10])
#print(ideal[:10])
print(real[:10])
#print(flowsim[:10])
print(new_flowsim[:10])

count = 0
#for i in range(0, len(ideal)):
    #if abs(real[i] - new_flowsim[i]) > 10000:
        #count += 1

for i in range(len(new_flowsim)):
    if abs(new_flowsim[i] - real[i]) > 10000:
        count += 1
#    if new_flowsim[i] < 0:
#        count += 1

print(count)

_, ax = plt.subplots()

ax.ecdf(new_flowsim, label="m4+flowsim")
ax.ecdf(real, label="ns3")
plt.legend()

plt.savefig("results.png")
