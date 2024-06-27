import torch
import numpy as np
from pytorch_lightning import LightningModule
from util.model import FlowSimLstm, FlowSimTransformer
from util.consts import MTU, HEADER_SIZE, BYTE_TO_BIT, DELAY_PROPAGATION_BASE
import argparse
import yaml
import os

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
            model = FlowSimLstm(
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
            model = FlowSimTransformer(
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
        
        model = model.load_from_checkpoint(checkpoint_path, map_location=self.device)
        return model

    def preprocess(self, data):
        return torch.tensor(data, dtype=torch.float32).to(self.device)

    def postprocess(self, output):
        return output.cpu().detach().numpy()

    def infer(self, data):
        data = self.preprocess(data)
        with torch.no_grad():
            lengths = torch.tensor([len(data)], dtype=torch.long).to(self.device)
            if self.model_name == "lstm":
                output, _ = self.model(data.unsqueeze(0), lengths)
            elif self.model_name == "transformer":
                output, _ = self.model(data.unsqueeze(0), lengths)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        return self.postprocess(output)

def load_data(dir_input, spec, topo_type="_topo-pl-21_s0",lr=10):
    dir_input_tmp = f"{dir_input}/{spec}"
    
    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
    fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
    assert np.all(fid[:-1] <= fid[1:])
    
    size = sizes_flowsim[fid].astype(np.float32)
    fat = fats_flowsim[fid].astype(np.float32)
    
    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
    sldn = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)[fid]
    
    pkt_head = np.clip(size, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE*2
    pkt_size=(pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = pkt_size / lr
    DELAY_PROPAGATION_PERFLOW=delay_propagation+delay_transmission
    fct_ideal=(
                size + np.ceil(sizes_flowsim / MTU) * HEADER_SIZE
            ) * BYTE_TO_BIT / lr + DELAY_PROPAGATION_PERFLOW
    return size, fat, sldn, fid, fct_ideal

def interactive_inference(inference, size, fat, sldn, fid, fct_ideal):
    n_flows_total = len(size)
    active_flows = []
    n_active_flows = 0
    flow_completion_times = {}
    current_time = 0  # Initialize the current time

    i = 0
    while i < n_flows_total or len(active_flows) > 0:
        if i < n_flows_total:
            flow_size,flow_at,flow_id = size[i], fat[i], fid[i] 
        else:
            flow_arrival_time = float('inf')
        
        # Perform inference for all active flows
        if active_flows:
            active_flow_inputs = np.array([[f[0],f[1]] for f in active_flows])
            active_flow_ids = np.array([f[2] for f in active_flows])
            active_flow_inputs[1]=np.diff(active_flow_inputs[:,1],prepend=0)
            predictions = inference.infer(active_flow_inputs)
            sldn_est = predictions[:, 0]
            sldn_est_min_idx = np.argmin(sldn_est)
            completed_flow_id=active_flow_ids[sldn_est_min_idx]
            fct_min = sldn_est[sldn_est_min_idx]*fct_ideal[completed_flow_id]
            fat_min = active_flows[sldn_est_min_idx][1]
            flow_completion_time=fat_min+fct_min
        else:
            flow_completion_time = float('inf')

        if flow_arrival_time < flow_completion_time:
            # Next event is flow arrival
            current_time = flow_arrival_time
            active_flows.append((flow_size,flow_at, flow_id))
            n_active_flows+=1
            print(f"Next event: Flow arrival at time {current_time}")
            i += 1  # Move to the next flow
        else:
            # Next event is flow completion
            current_time = flow_completion_time
            flow_completion_times[completed_flow_id] = current_time
            n_active_flows-=1
            print(f"Next event: Flow completion at time {current_time}")
            
            # If all active flows are completed, end the busy period
            if n_active_flows==0:
                print(f"Busy period reset")
                active_flows = []

    # Compare recorded flow completion times with the ground truth
    for flow_id in flow_completion_times:
        predicted_completion_time = flow_completion_times[flow_id]
        actual_completion_time = ground_truth[flow_id]
        print(f"Flow ID: {flow_id}, Predicted Completion Time: {predicted_completion_time}, Actual Completion Time: {actual_completion_time}")

def main():
    parser = argparse.ArgumentParser(description='Interactive Inference Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'], help='Model type')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output predictions')

    args = parser.parse_args()
    args.checkpoint = f"{args.output}/fct_lstm_bi_large_shard10000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/best.ckpt"
    
    lr = 10
    inference = Inference(config_path=args.config, checkpoint_path=args.checkpoint, model_name=args.model)
    
    
    # for shard in np.arange(0, 1000):
    for shard in [0]:
        for n_flows in [2000]:
            for n_hosts in [21]:
                spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                size, fat, sldn, fid,fct_ideal = load_data(args.input, spec=spec,lr=lr)
                
                # Perform interactive inference
                interactive_inference(inference, size, fat, sldn, fid, fid,fct_ideal)

if __name__ == '__main__':
    main()
