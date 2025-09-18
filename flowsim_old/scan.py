f = open("test.out", 'r')
flows = set()
dups = list()
for line in f.readlines():
    line = line.replace("\n", "").split(" ")
    if len(line) > 2 and line[2] == "completed":
        flow_id = int(line[3])
        if flow_id in flows:
            dups.append(flow_id)
        flows.add(int(line[3]))
f.close()


missing = []
for i in range(20000):
    if not i in flows:
        missing.append(i)


print(len(dups))
print(dups)
