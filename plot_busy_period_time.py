import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

target_str="_lognormal"
# target_str="_empirical_lognormal"
# target_str="_exp"
# target_str="_empirical_exp"
dir_input = f"/data2/lichenni/path_perflow{target_str}"
topo_type = "_topo-pl-x_"
lr = 10
data_list = []
for shard in np.arange(1000):
    for n_flows in [20000]:
        for n_hosts in [3]:
            for shard_seed in [0]:
                topo_type_cur = topo_type.replace("-x_", f"-{n_hosts}_") + "s%d" % (shard_seed)
                spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                dir_input_tmp = f"{dir_input}/{spec}"

                # fat = np.load(f'{dir_input}/{spec}/fat.npy')
                # fct = np.load(f'{dir_input}/{spec}/fct{topo_type_cur}.npy')
                # if len(fat) == len(fct):
                data_list.append((spec, topo_type_cur))
print(f"len(data_list): {len(data_list)}")

def process_spec(spec_idx, spec, topo_type, dir_input):
    start_time = time.time()
    print(f"Processing spec_idx: {spec_idx}")
    input_tmp = f"{dir_input}/{spec}"

    # Load FAT and FCT data
    fid = np.load(f'{input_tmp}/fid{topo_type}.npy')
    fat = np.load(f'{input_tmp}/fat.npy')
    fct = np.load(f'{input_tmp}/fct{topo_type}.npy')

    # Create a list of all start and end events
    events = []
    for flow_idx, flow_id in enumerate(fid):
        events.append((fat[flow_id], 'start', flow_id))
        events.append((fat[flow_id]+fct[flow_idx], 'end', flow_id))
    
    # Sort events by time
    events.sort()

    # Track active flows and busy periods
    active_flows = set()
    num_active_flows_list_tmp = []
    # last_event_time = None

    for event_time, event_type, flow_id in events:
        if event_type == 'start':
            active_flows.add(flow_id)
        elif event_type == 'end':
            active_flows.remove(flow_id)

        # Record the number of active flows for the current period
        # if last_event_time is not None and event_time != last_event_time:
        num_active_flows_list_tmp.append(len(active_flows))

        # last_event_time = event_time

    print(f"Finished spec_idx: {spec_idx} with {len(num_active_flows_list_tmp)}/{len(fat)} events")
    return num_active_flows_list_tmp

# Main function to run everything in parallel and save the result to a npy file
def main():
    result_file = f'num_active_flows_time{target_str}.npy'

    # Check if result file already exists
    if os.path.exists(result_file):
        num_active_flows_list = np.load(result_file, allow_pickle=True).tolist()
    else:
        num_active_flows_list = []

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_spec, spec_idx, spec, topo_type, dir_input) for spec_idx, (spec, topo_type) in enumerate(data_list)]

            for future in as_completed(futures):
                num_active_flows_list.append(future.result())

        # Save the results to a numpy file
        np.save(result_file, np.array(num_active_flows_list, dtype=object))

if __name__ == "__main__":
    main()
