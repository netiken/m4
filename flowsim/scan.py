f = open("test.out", 'r')
flows = set()
for line in f.readlines():
    line = line.replace("\n", "").split(" ")
    if line[1] == "arrived":
        flows.add(int(line[0]))
f.close()


missing = []
for i in range(20000):
    if not i in flows:
        missing.append(i)

print(missing)
