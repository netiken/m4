import os
import glob
import sys

BASE_DIR = sys.argv[1]
CLIENT_LOG_NAMES = [f"client_{i}.log" for i in range(0, 12)]

def parse_line(line):
    """Parse a line into a dict"""
    parts = line.strip().split()
    event_type = parts[0].split("=")[1]
    data = {}
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            if k in ['ts_ns', 'id', 'clt', 'wrkr', 'slot', 'size', 'start_ns', 'dur_ns']:
                v = int(v)
            data[k] = v
    data['event'] = event_type
    return data

def format_event_ns3(e):
    """Format an event in ns3-style log line"""
    role_map = {
        "req_send": "client req_send",
        "req_recv": "server req_recv",
        "resp_send": "server resp_send",
        "resp_recv_ud": "client resp_recv",
        "hand_send": "client hand_send",
        "hand_recv": "server hand_recv",
        "resp_rdma_read": "client rdma_recv",
        "resp_rdma_send": "server rdma_send",
    }
    role_event = role_map.get(e['event'], e['event'])
    size_str = f"{e['size']}B" if 'size' in e else f"{e.get('wire_bytes',0)}B"
    return f"[{role_event}] t={e['ts_ns']} ns reqId={e['id']} size={size_str} client_node_id={e['clt']}"

def process_directory(subdir):
    # Read all client logs
    all_events = []
    for log_name in CLIENT_LOG_NAMES:
        log_file = os.path.join(subdir, log_name)
        if not os.path.exists(log_file):
            continue
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    all_events.append(parse_line(line))

    # Group events by client+id
    flows = {}
    for e in all_events:
        flow_key = (e['clt'], e['id'])
        if flow_key not in flows:
            flows[flow_key] = []
        flows[flow_key].append(e)

    # Write grouped flows in ns3 block format
    output_file = os.path.join(subdir, "grouped_flows.txt")
    with open(output_file, "w") as out:
        for (clt, reqId), events in sorted(flows.items()):
            out.write(f"### reqId={reqId} ###\n")
            for e in sorted(events, key=lambda x: x['ts_ns']):
                out.write(format_event_ns3(e) + "\n")
            out.write("\n")  # empty line between flows

    print(f"Processed {subdir}, wrote {output_file}")

# Loop over all subdirectories under sweeps
for subdir in glob.glob(os.path.join(BASE_DIR, "*")):
    if os.path.isdir(subdir):
        process_directory(subdir)
