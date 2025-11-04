import os
import re
from collections import defaultdict

root_dir = "."
pattern = re.compile(r"wire_bytes=(\d+)")

# Map wire_bytes -> set of folders
wire_bytes_map = defaultdict(set)

for subdir, _, files in os.walk(root_dir):
    for fname in files:
        if fname == "flows_debug.txt":
            fpath = os.path.join(subdir, fname)
            with open(fpath, "r") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        val = int(match.group(1))
                        wire_bytes_map[val].add(subdir)

# Print results
for val in sorted(wire_bytes_map.keys()):
    if int(val) == 41:
        continue #ignore control
    print(f"{val}:")
    for folder in sorted(wire_bytes_map[val]):
        print(f"  {folder}")
