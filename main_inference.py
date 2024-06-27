import torch
import numpy as np
from pytorch_lightning import LightningModule
from util.model import FlowSimLstm, FlowSimTransformer
import argparse
import yaml
import os

class Inference:
    def __init__(self, config_path, checkpoint_path, model_name, device='cuda'):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.device = torch.device(self.config["training"]["gpu"][0])
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

def load_data(dir_input, spec, topo_type="_topo-pl-21_s0"):
    dir_input_tmp = f"{dir_input}/{spec}"
    
    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
    fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
    assert np.all(fid[:-1] <= fid[1:])
    
    sizes_flowsim = sizes_flowsim[fid]
    fats_flowsim = fats_flowsim[fid]
    
    fats_ia_flowsim = np.diff(fats_flowsim)
    fats_ia_flowsim = np.insert(fats_ia_flowsim, 0, 0)
    
    input_data = np.column_stack((sizes_flowsim, fats_ia_flowsim)).astype(np.float32)
    
    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
    output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)[fid]
    
    return input_data, output_data, fid

def interactive_inference(inference, input_data, fid, ground_truth):
    active_flows = []
    flow_completion_times = {}
    busy_period_start = None

    i = 0
    while i < len(input_data):
        flow = input_data[i]
        flow_id = fid[i]
        
        # Add new flow to active flows
        active_flows.append((flow, flow_id))
        
        # Perform inference for all active flows
        active_flow_inputs = np.array([f[0] for f in active_flows])
        predictions = inference.infer(active_flow_inputs)
        
        # Get flow completion times from predictions
        completion_times = predictions[:, 0]
        
        # Determine busy period
        if busy_period_start is None:
            busy_period_start = flow[1]  # Flow arrival time

        # Find the next event: either a new flow arrival or the earliest flow completion
        if i < len(input_data) - 1:
            next_flow_arrival_time = input_data[i + 1][1]
        else:
            next_flow_arrival_time = float('inf')  # No more flows arriving
        
        min_completion_time = min(completion_times)

        if next_flow_arrival_time < min_completion_time:
            # Next event is flow arrival
            print(f"Next event: Flow arrival at time {next_flow_arrival_time}")
            i += 1  # Move to the next flow
        else:
            # Next event is flow completion
            print(f"Next event: Flow completion at time {min_completion_time}")
            
            # Record completion time for the flow that completed
            completed_flow_index = np.argmin(completion_times)
            completed_flow = active_flows[completed_flow_index]
            completed_flow_id = completed_flow[1]
            flow_completion_times[completed_flow_id] = min_completion_time

            # If all active flows are completed, end the busy period
            if len(active_flows) == completed_flow_index + 1:
                busy_period_end = min_completion_time
                print(f"Busy period from {busy_period_start} to {busy_period_end}")
                busy_period_start = None  # Reset busy period

    # Compare recorded flow completion times with the ground truth
    for flow_id in flow_completion_times:
        predicted_completion_time = flow_completion_times[flow_id]
        actual_completion_time = ground_truth[flow_id]
        print(f"Flow ID: {flow_id}, Predicted Completion Time: {predicted_completion_time}, Actual Completion Time: {actual_completion_time}")

def main():
    parser = argparse.ArgumentParser(description='Interactive Inference Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file', default='./config/test_config_lstm.yaml')
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'], help='Model type', default='lstm')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data directory', default='/data2/lichenni/path_perflow_busy_empirical')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output predictions', default='./res_inference')

    args = parser.parse_args()
    args.checkpoint = f"{args.output}/fct_lstm_bi_large_shard10000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/best.ckpt"
    
    inference = Inference(config_path=args.config, checkpoint_path=args.checkpoint, model_name=args.model)
    lr = 10
    
    for shard in np.arange(0, 1000):
        for n_flows in [2000]:
            for n_hosts in [21]:
                spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                input_data, output_data, fid = load_data(args.input, spec=spec)
                
                # Perform interactive inference
                interactive_inference(inference, input_data, fid, output_data)

if __name__ == '__main__':
    main()
