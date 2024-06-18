import os
import numpy as np
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Define a pattern to match the log line
log_pattern = re.compile(r"(\d+)\s+n:(\d+)\s+(\d+):(\d+)\s+(\d+)\s+(\w+)\s+ecn:(\d+)\s+(0b[0-9a-f]+)\s+(0b[0-9a-f]+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\((\d+)\)\s+(\d+)")
# Create a DataFrame
columns = ["timestamp", "node", "src_port", "queue", "queue_length", "event", "ecn", "src_addr", "dst_addr", 
           "src_port_num", "dst_port_num", "packet_type", "seq_num", "tx_timestamp", "priority_group", "packet_size", 
           "payload_size", "flow_id"]

def process_spec(spec_idx, spec, topo_type, dir_input,flow_size):
    start_time = time.time()
    print(f"Processing spec_idx: {spec_idx}")
    input_tmp = f"{dir_input}/{spec}"

    # List to hold parsed log data
    log_data = []

    # Read the log file
    with open(f'{input_tmp}/mix{topo_type}.log', 'r') as file:
        log_data = [match.groups() for line in file if (match := log_pattern.match(line))]

    df = pd.DataFrame(log_data, columns=columns)

    # Convert relevant columns to appropriate data types
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["seq_num"] = pd.to_numeric(df["seq_num"])
    df["flow_id"] = pd.to_numeric(df["flow_id"])

    fid = np.load(f'{input_tmp}/fid{topo_type}.npy')
    fat = np.load(f'{input_tmp}/fat.npy')
    fct = np.load(f'{input_tmp}/fct{topo_type}.npy')
    
    # Create a dictionary for flow end times
    flow_end_time_dict={}
    for flow_idx, flow_id in enumerate(fid):
        flow_end_time_dict[flow_id] = fat[flow_id]+fct[flow_idx]

    # Identify busy periods
    busy_periods = []
    flows = []
    current_period = None

    for _, row in df.iterrows():
        seq_num = row["seq_num"]
        timestamp = row["timestamp"]
        flow_id = row["flow_id"]
        if seq_num == 1:
            if current_period is not None:
                raise AssertionError("Current period should be None")
            current_period = {"start": timestamp, "end": None}
        elif seq_num == 2 and current_period is not None:
            current_period["end"] = timestamp
            busy_periods.append(current_period)
            current_period = None
        elif seq_num == 0:
            end_time = flow_end_time_dict.get(flow_id, None)
            if end_time is not None:
                flows.append((flow_id, timestamp, end_time))
        else:
            raise AssertionError("Invalid seq_num")

    # Add the last period if still open
    if current_period is not None:
        current_period["end"] = 60000000000
        busy_periods.append(current_period)

    # Output the number of active flows for each busy period
    num_active_flows_list_tmp = []
    for period in busy_periods:
        start, end = period["start"], period["end"]
        active_flows = set(flow_id for flow_id, start_time, end_time in flows if start_time <= end and end_time >= start)
        num_active_flows = len(active_flows)
        num_active_flows_list_tmp.append(num_active_flows)

    print(f"Finished spec_idx: {spec_idx} with {len(num_active_flows_list_tmp)} events")
    return num_active_flows_list_tmp,flow_size

# Main function to run everything in parallel and save the result to a npy file
def main():
    topo_type = "_topo-pl-x_"
    lr = 10
    
    for target_str in ["_lognormal", "_empirical_lognormal", "_exp", "_empirical_exp"]:
        
        result_file = f'./res/num_active_flows_queue{target_str}.npy'

        # Check if result file already exists
        if os.path.exists(result_file):
            results = np.load(result_file, allow_pickle=True).item()
            num_active_flows_list = results["num_active_flows"]
            flow_sizes_list = results["flow_sizes"]
        else:
            dir_input = f"/data2/lichenni/path_perflow{target_str}"
            num_active_flows_list = []
            flow_sizes_list = []
            data_list = []
            flow_size_list=[]
            for shard in np.arange(1000):
            # for shard in [688]:
                for n_flows in [20000]:
                    for n_hosts in [3]:
                        for shard_seed in [0]:
                            topo_type_cur = topo_type.replace("-x_", f"-{n_hosts}_") + "s%d" % (shard_seed)
                            spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                            dir_input_tmp = f"{dir_input}/{spec}"

                            # fat = np.load(f'{dir_input}/{spec}/fat.npy')
                            # fct = np.load(f'{dir_input}/{spec}/fct{topo_type_cur}.npy')
                            fid = np.load(f'{dir_input}/{spec}/fid{topo_type_cur}.npy')
                            if len(fid) == len(set(fid)):
                                data_list.append((spec, topo_type_cur))
                                statss = np.load(f'{dir_input}/{spec}/stats.npy', allow_pickle=True)
                                flow_size_list.append(statss.item().get("size_dist_candidate"))
                                
            print(f"len(data_list): {len(data_list)}, {len(flow_size_list)}")
        
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_spec, spec_idx, spec, topo_type, dir_input, flow_size_list[spec_idx]) for spec_idx, (spec, topo_type) in enumerate(data_list)]

                for future in as_completed(futures):
                    num_active_flows, flow_sizes = future.result()
                    num_active_flows_list.append(num_active_flows)
                    flow_sizes_list.append(flow_sizes)

            # Save the results to a numpy file
            results = {"num_active_flows": num_active_flows_list, "flow_sizes": flow_sizes_list}
            np.save(result_file, results)

if __name__ == "__main__":
    main()
