import os
import re

root_dir = "."

# Regex patterns
flow_start = re.compile(r"^flow_id=(\d+)")
clt_pattern = re.compile(r"clt=(\d+)")
network_time_pattern = re.compile(r"network_time=\[(\d+)\]")
rdma_time_pattern = re.compile(r"rdma_time=\[(\d+)\]")

for subdir, _, files in os.walk(root_dir):
    if "flows_debug.txt" in files:
        input_path = os.path.join(subdir, "flows_debug.txt")
        output_path = os.path.join(subdir, "real_world.txt")

        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            current_flow_id = None
            current_client = None
            network_time = None
            rdma_time = None

            for line in fin:
                # Capture flow id
                m = flow_start.match(line.strip())
                if m:
                    current_flow_id = m.group(1)
                    current_client = None
                    network_time = None
                    rdma_time = None
                    continue

                # Capture client id
                if "clt=" in line and current_client is None:
                    m = clt_pattern.search(line)
                    if m:
                        current_client = m.group(1)

                # Capture times
                m_net = network_time_pattern.search(line)
                if m_net:
                    network_time = m_net.group(1)

                m_rdma = rdma_time_pattern.search(line)
                if m_rdma:
                    rdma_time = m_rdma.group(1)

                # When both times found, write them out
                if network_time and rdma_time and current_flow_id and current_client:
                    fout.write(f"[ud] client={current_client} id={current_flow_id} dur_ns={network_time}\n")
                    fout.write(f"[rdma] client={current_client} id={current_flow_id} dur_ns={rdma_time}\n")

                    # Reset so we don't duplicate
                    current_flow_id = None
                    current_client = None
                    network_time = None
                    rdma_time = None

        print(f"Processed {input_path} -> {output_path}")
