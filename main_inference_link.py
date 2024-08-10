import torch
import numpy as np
from util.model import FlowSimLstm, FlowSimTransformer
from util.consts import get_base_delay_link, get_base_delay_transmission
import argparse
import yaml
from ctypes import *

class FCTStruct(Structure):
    _fields_ = [
        ("estimated_fcts", POINTER(c_double)),
        ("t_flows", POINTER(c_double)),
        ("num_flows", POINTER(c_uint)),
        ("num_flows_enq", POINTER(c_uint)),
    ]

def make_array(ctype, arr):
    return (ctype * len(arr))(*arr)

C_LIB_PATH = "./clibs/get_fct_mmf.so"
C_LIB = CDLL(C_LIB_PATH)
C_LIB.get_fct_mmf.argtypes = [
    c_uint,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
    POINTER(c_int),
    c_int,
    POINTER(c_int),
    c_int,
    c_int,
    c_int,
    c_int,
]

C_LIB.get_fct_mmf.restype = FCTStruct
C_LIB.free_fctstruct.argtypes = [FCTStruct]
C_LIB.free_fctstruct.restype = None

class Inference:
    def __init__(self, model_config, training_config,checkpoint_path, lr, device='cuda'):
        self.model_config = model_config
        self.training_config = training_config
        self.lr = lr
        self.device = torch.device(device)
        self.model_name = model_config["model_name"]
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, checkpoint_path):
        model_config = self.model_config
        training_config = self.training_config
        
        if self.model_name == "lstm":
            model = FlowSimLstm.load_from_checkpoint(
                checkpoint_path, 
                map_location=self.device,
                 n_layer=model_config["n_layer"],
                gcn_n_layer=model_config["gcn_n_layer"],
                loss_fn_type=model_config["loss_fn_type"],
                learning_rate=training_config["learning_rate"],
                batch_size=training_config["batch_size"],
                hidden_size=model_config["hidden_size"],
                gcn_hidden_size=model_config["gcn_hidden_size"],
                dropout=model_config["dropout"],
                enable_val=training_config["enable_val"],
                enable_dist=training_config["enable_dist"],
                input_size=model_config["input_size"],
                output_size=1,
                enable_bidirectional=model_config.get("enable_bidirectional", False),
                enable_positional_encoding=model_config.get("enable_positional_encoding", False),
                enable_gnn=model_config.get("enable_gnn", False),
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

    def preprocess(self, data, fsds, fcts_flowsim_gt,nhosts, flag_from_last_period):
        sizes, fats = data[:, 0], data[:, 1]
        fats_ia = np.diff(fats)
        fats_ia = np.insert(fats_ia, 0, 0)
        
        # fcts_flowsim=self.run_flow_simulation(sizes, fats, fsds, nhosts)
        fcts_flowsim=fcts_flowsim_gt
        n_links_passed = np.ones_like(fcts_flowsim) * 2
        base_delay = get_base_delay_link(sizes, n_links_passed, self.lr)
        i_fcts_flowsim = get_base_delay_transmission(sizes,self.lr) + base_delay
        fcts_flowsim += base_delay
        sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
        
        sizes=np.log1p(sizes)
        fats_ia=np.log1p(fats_ia)
        input_data = np.column_stack((sizes, fats_ia, sldn_flowsim, flag_from_last_period))
        return torch.tensor(input_data, dtype=torch.float32).to(self.device)

    def postprocess(self, output):
        return output.cpu().detach().numpy()

    def infer(self, data, active_fsds, active_fcts_flowsim_gt,nhosts, flag_from_last_period):
        data = self.preprocess(data, active_fsds,active_fcts_flowsim_gt,nhosts,flag_from_last_period)
        with torch.no_grad():
            lengths = np.array([len(data)])
            if self.model_name == "lstm":
                output, _ = self.model(data.unsqueeze(0), lengths, None, None)
            elif self.model_name == "transformer":
                output, _ = self.model(data.unsqueeze(0), lengths)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        return self.postprocess(output)

    def run_flow_simulation(self, size, fat, fsd, nhosts=21):
        nflows = len(size)
        # Adjust nhosts and flow source/destination if nhosts is 21
        if nhosts == 21:
            nhosts = 3
            fsd[:, 0] = 0
            fsd[:, 1] = 2
        # Prepare data for the C function
        fats_pt = make_array(c_double, fat)
        sizes_pt = make_array(c_double, size)
        src_pt = make_array(c_int, fsd[:, 0])
        dst_pt = make_array(c_int, fsd[:, 1])
        topo_pt = make_array(c_int, np.array([1, 4]))

        # Run the flow simulation
        res = C_LIB.get_fct_mmf(nflows, fats_pt, sizes_pt, src_pt, dst_pt, nhosts, topo_pt, 2, 8, 2, self.lr)
        estimated_fcts = np.fromiter(res.estimated_fcts, dtype=np.float64, count=nflows)

        C_LIB.free_fctstruct(res)
        return estimated_fcts

def load_data(dir_input, spec, topo_type="_topo-pl-21_s0", lr=10,max_inflight_flows=0):
    topo_type+=f"_i{max_inflight_flows}"
    dir_input_tmp = f"{dir_input}/{spec}"
    
    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    size = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
    fat = np.load(f"{dir_input_tmp}/fat.npy")[fid]
    fsd=np.load(f"{dir_input_tmp}/fsd.npy")[fid]
    assert np.all(fid[:-1] <= fid[1:])
    
    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
    fcts_flowsim_gt=np.load(f"{dir_input_tmp}/fct_flowsim.npy")[fid]
    # sldn = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
    
    # pkt_head = np.clip(size, a_min=0, a_max=MTU)
    # delay_propagation = DELAY_PROPAGATION_BASE * 2
    # pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    # delay_transmission = pkt_size / lr
    # delay_propagation_perflow = delay_propagation + delay_transmission
    # fct_ideal = (
    #     size + np.ceil(size / MTU) * HEADER_SIZE
    # ) * BYTE_TO_BIT / lr + delay_propagation_perflow

    assert len(size) == len(fcts)
    return size, fat, fsd, fcts, i_fcts,fcts_flowsim_gt

def interactive_inference(inference, size, fat, fsd, fcts, i_fcts,fcts_flowsim_gt, max_inflight_flows=5, nhosts=21, flow_size_threshold=100000):
    if max_inflight_flows==0:
        max_inflight_flows=1000000
    n_flows_total = len(size)
    print(f"Total number of flows: {n_flows_total}")
    # n_flows_total = 1000
    active_flows = []
    completed_flow_idx_set=set()
    inflight_flows = 0
    flow_completion_times = {}
    flow_fct_sldn = {}
    current_time = 0  # Initialize the current time
    
    i = 0
    while i < n_flows_total or len(active_flows) > 0:
        if i%1000==0:
            print(f"Flow {i} processed")
        # assert i==fid[i]
        flow_arrival_time = float('inf')
        flow_completion_time = float('inf')
        sldn_est_min_idx=None
        if i < n_flows_total:
            if inflight_flows < max_inflight_flows:
                flow_arrival_time = np.maximum(fat[i],current_time)
                # print(f"Flow {i} added to queue")

        if active_flows:
            active_flow_inputs = np.array([[f[0], f[1]] for f in active_flows])
            active_flow_ids = np.array([f[2] for f in active_flows])
            active_fsds=np.array([f[3] for f in active_flows])
            flag_from_last_period=np.array([f[4] for f in active_flows])
            active_fcts_flowsim_gt=np.array([f[5] for f in active_flows])
            predictions = inference.infer(active_flow_inputs,active_fsds,active_fcts_flowsim_gt,nhosts,flag_from_last_period)
            sldn_est = predictions[0, :, 0]
            
            sldn_est = np.where(np.isin(np.arange(len(sldn_est)), list(completed_flow_idx_set)), np.inf, sldn_est)
            sldn_est_min_idx = np.argmin(sldn_est, axis=0)
            
            completed_flow_id = active_flow_ids[sldn_est_min_idx]
            sldn_min=sldn_est[sldn_est_min_idx]
            fct_min = sldn_min * i_fcts[completed_flow_id]
            fat_min = active_flows[sldn_est_min_idx][1]
            flow_completion_time = fat_min + fct_min

        if flow_arrival_time < flow_completion_time:
            # Next event is flow arrival
            current_time = flow_arrival_time
            inflight_flows += 1
            active_flows.append((size[i], flow_arrival_time, i, fsd[i], 0,fcts_flowsim_gt[i]))
            # print(f"Event: Flow {i} Arrival at {current_time}")
            i += 1
        else:
            # Next event is flow completion
            current_time = flow_completion_time
            flow_completion_times[completed_flow_id] = fct_min
            flow_fct_sldn[completed_flow_id] = sldn_min
            completed_flow_idx_set.add(sldn_est_min_idx)
            inflight_flows -= 1
            # print(f"Event: Flow {completed_flow_id} Completion at {current_time}")
            if inflight_flows == 0:
                active_flows = []
                completed_flow_idx_set=set()
                # print("Busy period reset")
            else:
                n_small_flows=len([f for f in active_flows if f[0]<flow_size_threshold])
                if n_small_flows==0:
                    active_flows_tmp = []
                    for active_flow_idx in range(len(active_flows)):
                        flow_tmp=active_flows[active_flow_idx]
                        if flow_tmp[0]>=flow_size_threshold and i not in completed_flow_idx_set:
                            active_flows_tmp.append((flow_tmp[0], flow_tmp[1], flow_tmp[2], flow_tmp[3], 1,flow_tmp[5]))
                    active_flows=active_flows_tmp
                    completed_flow_idx_set=set()
        # print(f"Current time: {current_time}, Inflight flows: {inflight_flows}, Active flows: {len(active_flows)}")
    data_dict = {}
    # Compare recorded flow completion times with the ground truth
    for flow_id in flow_completion_times:
        predicted_completion_time = flow_completion_times[flow_id]
        actual_completion_time = fcts[flow_id]
        # print(f"Flow ID: {flow_id}, Predicted Completion Time: {predicted_completion_time}, Actual Completion Time: {actual_completion_time}")
        predicted_sldn = flow_fct_sldn[flow_id]
        assert predicted_sldn != np.inf
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
    parser.add_argument('--input', type=str, required=False, help='Path to the input data directory', default='/data2/lichenni/perflow_link')
    parser.add_argument('--output', type=str, required=False, help='Path to save the output predictions', default='/data2/lichenni/output_perflow')

    args = parser.parse_args()
    
    
    with open(args.config, 'r') as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)
        model_config = config_info["model"]
        training_config = config_info["training"]
        data_config = config_info["dataset"]
    
    lr = data_config["lr"]
    flow_size_threshold=data_config["flow_size_threshold"]
    
    args.checkpoint = f"{args.output}/fct_link_{flow_size_threshold}_shard2000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/best.ckpt"
    inference = Inference(model_config,training_config, checkpoint_path=args.checkpoint, lr=lr)
    empirical_str='_empirical'
    # empirical_str=''
    args.input+=empirical_str
    
    # for max_inflight_flows in [0, 4, 6, 15]:
    for max_inflight_flows in [0]:
        fct,sldn=[], []
        for shard in np.arange(0, 1):
            for nflows in [2000]:
                for nhosts in [21]:
                    spec = f"shard{shard}_nflows{nflows}_nhosts{nhosts}_lr{lr}Gbps"
                    size, fat, fsd, fcts, i_fcts,fcts_flowsim_gt = load_data(args.input, spec=spec, lr=lr,max_inflight_flows=max_inflight_flows)
                    
                    # Perform interactive inference
                    fct_tmp,sldn_tmp=interactive_inference(inference, size, fat, fsd, fcts, i_fcts,fcts_flowsim_gt, max_inflight_flows=max_inflight_flows,nhosts=nhosts,flow_size_threshold=flow_size_threshold)
                    print(f"Finished inference with workload={shard}. fct shape: {fct_tmp.shape}, sldn shape: {sldn_tmp.shape}")
                    fct.append(fct_tmp)
                    sldn.append(sldn_tmp)
        fct=np.array(fct)
        sldn=np.array(sldn)
        print(f"Finished inference with {max_inflight_flows} inflight flows. fct shape: {fct.shape}, sldn shape: {sldn.shape}")
        np.savez(f'./res/inference_{max_inflight_flows}_t{flow_size_threshold}{empirical_str}.npz', fct=fct, sldn=sldn)
if __name__ == '__main__':
    main()
