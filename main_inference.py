import torch
import numpy as np
from util.model import FlowSimLstm, FlowSimTransformer
from util.consts import MTU, HEADER_SIZE, BYTE_TO_BIT, DELAY_PROPAGATION_BASE
import argparse
import yaml
from collections import deque

class Inference:
    def __init__(self, config_path, checkpoint_path, model_name, device='cuda'):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, checkpoint_path):
        model_config = self.config["model"]
        training_config = self.config["training"]
        
        if self.model_name == "lstm":
            model = FlowSimLstm.load_from_checkpoint(
                checkpoint_path, 
                map_location=self.device,
                n_layer=model_config["n_layer"],
                loss_fn_type=model_config["loss_fn_type"],
                learning_rate=training_config["learning_rate"],
                batch_size=training_config["batch_size"],
                hidden_size=model_config["hidden_size"],
                dropout=model_config["dropout"],
                enable_val=False,
                enable_dist=False,
                input_size=2,
                output_size=1,
                enable_bidirectional=model_config.get("enable_bidirectional", False),
            )
        elif self.model_name == "transformer":
            model = FlowSimTransformer.load_from_checkpoint(
                checkpoint_path, 
                map_location=self.device,
                n_layer=model_config["n_layer"],
                n_head=model_config["n_head"],
                n_embd=model_config["n_embd"],
                block_size=model_config["block_size"],
                vocab_size=model_config["vocab_size"],
                output_dim=model_config["output_dim"],
                dropout=model_config["dropout"],
                compile=model_config["compile"],
                loss_fn_type=model_config["loss_fn_type"],
                weight_decay=training_config["weight_decay"],
                learning_rate=training_config["learning_rate"],
                betas=training_config["betas"],
                batch_size=training_config["batch_size"],
                enable_position=model_config["enable_position"],
                enable_causal=model_config["enable_causal"],
                enable_val=False,
                enable_dist=False,
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        return model

    def preprocess(self, data):
        return torch.tensor(data, dtype=torch.float32).to(self.device)

    def postprocess(self, output):
        return output.cpu().detach().numpy()

    def infer(self, data):
        data = self.preprocess(data)
        with torch.no_grad():
            lengths = np.array([len(data)])
            if self.model_name == "lstm":
                output, _ = self.model(data.unsqueeze(0), lengths)
            elif self.model_name == "transformer":
                output, _ = self.model(data.unsqueeze(0), lengths)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        return self.postprocess(output)

def load_data(dir_input, spec, topo_type="_topo-pl-21_s0", lr=10,max_inflight_flows=0):
    topo_type+=f"_i{max_inflight_flows}"
    dir_input_tmp = f"{dir_input}/{spec}"
    
    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
    fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
    assert np.all(fid[:-1] <= fid[1:])
    
    size = sizes_flowsim[fid].astype(np.float32)
    fat = fats_flowsim[fid].astype(np.float32)
    
    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")[fid]
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")[fid]
    # sldn = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
    
    # pkt_head = np.clip(size, a_min=0, a_max=MTU)
    # delay_propagation = DELAY_PROPAGATION_BASE * 2
    # pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    # delay_transmission = pkt_size / lr
    # delay_propagation_perflow = delay_propagation + delay_transmission
    # fct_ideal = (
    #     size + np.ceil(size / MTU) * HEADER_SIZE
    # ) * BYTE_TO_BIT / lr + delay_propagation_perflow

    return size, fat, fid, fcts, i_fcts

def interactive_inference(inference, size, fat, fid, fcts, i_fcts, max_inflight_flows=5):
    if max_inflight_flows==0:
        max_inflight_flows=1000000
    n_flows_total = len(size)
    print(f"Total number of flows: {n_flows_total}")
    # n_flows_total = 1000
    active_flows = []
    completed_flow_idx_list=[]
    inflight_flows = 0
    flow_completion_times = {}
    flow_fct_sldn = {}
    current_time = 0  # Initialize the current time
    
    i = 0
    while i < n_flows_total or len(active_flows) > 0:
        # assert i==fid[i]
        flow_arrival_time = float('inf')
        flow_completion_time = float('inf')
        if i < n_flows_total:
            if inflight_flows < max_inflight_flows:
                flow_arrival_time = np.maximum(fat[i],current_time+1)
                # print(f"Flow {i} added to queue")

        if active_flows:
            active_flow_inputs = np.array([[f[0], f[1]] for f in active_flows])
            active_flow_ids = np.array([f[2] for f in active_flows])
            active_flow_inputs[:, 1] = np.diff(active_flow_inputs[:, 1], prepend=active_flow_inputs[0, 1])

            predictions = inference.infer(active_flow_inputs)
            sldn_est = predictions[0, :, 0]
            
            mask = np.isin(np.arange(len(sldn_est)), completed_flow_idx_list, invert=True)
            masked_sldn_est = np.ma.array(sldn_est, mask=~mask)
            sldn_est_min_idx = np.argmin(masked_sldn_est, axis=0)
            completed_flow_id = active_flow_ids[sldn_est_min_idx]
            fct_min = sldn_est[sldn_est_min_idx] * i_fcts[completed_flow_id]
            fat_min = active_flows[sldn_est_min_idx][1]
            flow_completion_time = fat_min + fct_min

        if flow_arrival_time < flow_completion_time:
            # Next event is flow arrival
            current_time = flow_arrival_time
            active_flows.append((size[i], flow_arrival_time, fid[i]))
            inflight_flows += 1
            # print(f"Event: Flow {i} Arrival at {current_time}")
            i += 1
        else:
            # Next event is flow completion
            current_time = flow_completion_time
            flow_completion_times[completed_flow_id] = fct_min
            flow_fct_sldn[completed_flow_id] = sldn_est
            completed_flow_idx_list.append(sldn_est_min_idx)
            inflight_flows -= 1
            # print(f"Event: Flow {completed_flow_id} Completion at {current_time}")

            if inflight_flows == 0:
                active_flows = []
                completed_flow_idx_list=[]
                # print("Busy period reset")
                
        print(f"Current time: {current_time}, Inflight flows: {inflight_flows}, Active flows: {len(active_flows)}")
    data_dict = {}
    # Compare recorded flow completion times with the ground truth
    for flow_id in flow_completion_times:
        predicted_completion_time = flow_completion_times[flow_id]
        actual_completion_time = fcts[flow_id]
        # print(f"Flow ID: {flow_id}, Predicted Completion Time: {predicted_completion_time}, Actual Completion Time: {actual_completion_time}")
        predicted_sldn = flow_fct_sldn[flow_id][0]
        actual_sldn = fcts[flow_id] / i_fcts[flow_id]
        # print(f"Flow ID: {flow_id}, Predicted SLDN: {predicted_sldn}, Actual SLDN: {actual_sldn}")

        data_dict[flow_id] = [predicted_completion_time, actual_completion_time, predicted_sldn, actual_sldn]

    sorted_flow_ids = sorted(data_dict.keys())
    res = np.array([data_dict[flow_id] for flow_id in sorted_flow_ids])
    return res[:, :2], res[:, 2:]
    # Saving the data to a .npz file
    # np.savez(f'./res/inference_{n_flows_total}_{max_inflight_flows}.npz', 
            #  fct=res[:, :2], 
            #  sldn=res[:, 2:])

def main():
    parser = argparse.ArgumentParser(description='Interactive Inference Script')
    parser.add_argument('--config', type=str, required=False, help='Path to the YAML configuration file', default='./config/test_config_lstm.yaml')
    parser.add_argument('--model', type=str, required=False, choices=['lstm', 'transformer'], help='Model type', default='lstm')
    parser.add_argument('--input', type=str, required=False, help='Path to the input data directory', default='/data2/lichenni/path_perflow_busy_close')
    parser.add_argument('--output', type=str, required=False, help='Path to save the output predictions', default='/data2/lichenni/output_perflow')

    args = parser.parse_args()
    args.checkpoint = f"{args.output}/fct_lstm_bi_uniform_shard5000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/best.ckpt"
    
    lr = 10
    inference = Inference(config_path=args.config, checkpoint_path=args.checkpoint, model_name=args.model)
    
    
    # for max_inflight_flows in [0, 4, 6, 15]:
    for max_inflight_flows in [6]:
        fct,sldn=[], []
        # for shard in np.arange(0, 500, 50):
        for shard in [0]:
            for n_flows in [2000]:
                for n_hosts in [21]:
                    spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                    size, fat, fid, fcts, i_fcts = load_data(args.input, spec=spec, lr=lr,max_inflight_flows=max_inflight_flows)
                    
                    # Perform interactive inference
                    fct_tmp,sldn_tmp=interactive_inference(inference, size, fat, fid, fcts, i_fcts, max_inflight_flows=max_inflight_flows)
                    print(f"Finished inference with workload={shard}. fct shape: {fct_tmp.shape}, sldn shape: {sldn_tmp.shape}")
                    fct.append(fct_tmp)
                    sldn.append(sldn_tmp)
        fct=np.array(fct)
        sldn=np.array(sldn)
        print(f"Finished inference with {max_inflight_flows} inflight flows. fct shape: {fct.shape}, sldn shape: {sldn.shape}")
        np.savez(f'./res/inference_{max_inflight_flows}.npz', fct=fct, sldn=sldn)
if __name__ == '__main__':
    main()
