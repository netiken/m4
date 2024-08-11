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
    def __init__(self, model_config, training_config, checkpoint_path, lr, device='cuda'):
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

    def preprocess(self, flows_info_active):
        sizes, fats, fcts_flowsim, flag_from_last_period = map(np.array, zip(*flows_info_active))
        fats_ia = np.diff(fats, prepend=fats[0]) 
        n_links_passed = np.ones_like(fcts_flowsim) * 2
        base_delay = get_base_delay_link(sizes, n_links_passed, self.lr)
        i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
        fcts_flowsim += base_delay
        sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
        
        sizes = np.log1p(sizes)
        fats_ia = np.log1p(fats_ia)
        input_data = np.column_stack((sizes, fats_ia, sldn_flowsim, flag_from_last_period))
        return torch.tensor(input_data, dtype=torch.float32).to(self.device)

    def postprocess(self, output):
        return output.cpu().numpy()

    def infer(self, flows_info_active):
        data = self.preprocess(flows_info_active)
        with torch.no_grad():
            lengths = np.array([len(data)])
            if self.model_name == "lstm":
                output, _ = self.model(data.unsqueeze(0), lengths, None, None)
            elif self.model_name == "transformer":
                output, _ = self.model(data.unsqueeze(0), lengths)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        return self.postprocess(output)

def run_flow_simulation(flows_info, nhosts=21, lr=10):
    size, fat, fsd = map(np.array, zip(*flows_info))
    nflows = len(size)
    if nhosts == 21:
        nhosts = 3
        fsd[:, 0] = 0
        fsd[:, 1] = 2
    fats_pt = make_array(c_double, fat)
    sizes_pt = make_array(c_double, size)
    src_pt = make_array(c_int, fsd[:, 0])
    dst_pt = make_array(c_int, fsd[:, 1])
    topo_pt = make_array(c_int, np.array([1, 4]))

    res = C_LIB.get_fct_mmf(nflows, fats_pt, sizes_pt, src_pt, dst_pt, nhosts, topo_pt, 2, 8, 2, lr)
    estimated_fcts = np.fromiter(res.estimated_fcts, dtype=np.float64, count=nflows)

    C_LIB.free_fctstruct(res)
    return estimated_fcts

def load_data(dir_input, spec, topo_type="_topo-pl-21_s0", lr=10, max_inflight_flows=0):
    topo_type += f"_i{max_inflight_flows}"
    dir_input_tmp = f"{dir_input}/{spec}"
    
    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    size = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
    fat = np.load(f"{dir_input_tmp}/fat.npy")[fid]
    fsd = np.load(f"{dir_input_tmp}/fsd.npy")[fid]
    assert np.all(fid[:-1] <= fid[1:])
    
    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
    
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
    return size, fat, fsd, fcts, i_fcts

def interactive_inference(inference, size, fat, fsd, fcts, i_fcts, max_inflight_flows=5, nhosts=21, flow_size_threshold=100000, lr=10):
    if max_inflight_flows == 0:
        max_inflight_flows = 1000000
    n_flows_total = len(size)
    # print(f"Total number of flows: {n_flows_total}")
    # n_flows_total = 1000
    
    # flow_id: [flag_complete_previous_period, flag_from_last_period]
    flows_period = {}
    flow_completion_times = {}
    flow_fct_sldn = {}
    current_time = 0  # Initialize the current time
    inflight_flows = 0
    
    flow_id_in_prop = 0
    while flow_id_in_prop < n_flows_total or flows_period:
        # if flow_id_in_prop%1000==0:
            # print(f"Flow {flow_id_in_prop} processed")
        flow_arrival_time = float('inf')
        flow_completion_time = float('inf')
        if flow_id_in_prop < n_flows_total and inflight_flows < max_inflight_flows:
            flow_arrival_time = max(fat[flow_id_in_prop], current_time)
            # print(f"Flow {flow_id_in_prop} added to queue")

        if flows_period:
            
            flows_info = [[size[flow_id], fat[flow_id], fsd[flow_id]] for flow_id in flows_period]
            fcts_flowsim = run_flow_simulation(flows_info, nhosts, lr)
            
            flows_info_active = []
            flow_id_info_active = []
            for flow_idx, flow_id in enumerate(flows_period):
                flow_info = flows_period[flow_id]
                if not flow_info[0]:
                    flow_id_info_active.append(flow_id)
                    flows_info_active.append([size[flow_id], fat[flow_id], fcts_flowsim[flow_idx], flow_info[1]])
            
            predictions = inference.infer(flows_info_active)
            sldn_est = predictions[0, :, 0]
            
            fct_stamp_est = []
            for idx, sldn_tmp in enumerate(sldn_est):
                flow_id=flow_id_info_active[idx]
                if flow_id in flow_completion_times:
                    fct_stamp_est.append(np.inf)
                else:
                    fct_stamp_est.append(fat[flow_id]+sldn_tmp * i_fcts[flow_id])
            
            min_idx = np.argmin(fct_stamp_est)
            completed_flow_id = flow_id_info_active[min_idx]
            flow_completion_time = fct_stamp_est[min_idx]

        if flow_arrival_time < flow_completion_time:
            # Next event is flow arrival
            current_time = flow_arrival_time
            inflight_flows += 1
            flows_period[flow_id_in_prop] = [0, 0]
            # print(f"Event: Flow {flow_id_in_prop} Arrival at {current_time}")
            flow_id_in_prop += 1
        else:
            # Next event is flow completion
            current_time = flow_completion_time
            flow_completion_times[completed_flow_id] = flow_completion_time - fat[completed_flow_id]
            flow_fct_sldn[completed_flow_id] = sldn_est[min_idx]
            inflight_flows -= 1
            if inflight_flows == 0:
                flows_period.clear()
                # print("Busy period reset")
            else:
                small_flows = [flow_id for flow_id in flows_period if size[flow_id] < flow_size_threshold and flow_id not in flow_completion_times]
                if not small_flows:
                    for flow_id in flows_period:
                        if not flows_period[flow_id][0]:
                            flows_period[flow_id][1] = 1
                            if flow_id in flow_completion_times:
                                flows_period[flow_id][0] = 1
        # print(f"Current time: {current_time}, Inflight flows: {inflight_flows}, Active flows: {len(active_flows)}")
    sorted_flow_ids = sorted(flow_completion_times.keys())
    res = np.array([
        [flow_completion_times[flow_id], fcts[flow_id], flow_fct_sldn[flow_id], fcts[flow_id] / i_fcts[flow_id]]
        for flow_id in sorted_flow_ids
    ])
    
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
    flow_size_threshold = data_config["flow_size_threshold"]
    
    args.checkpoint = f"{args.output}/fct_link_{flow_size_threshold}_shard2000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/best.ckpt"
    inference = Inference(model_config, training_config, checkpoint_path=args.checkpoint, lr=lr)
    empirical_str = '_empirical'
    # empirical_str=''
    args.input += empirical_str
    
    for max_inflight_flows in [0]:
        fct, sldn = [], []
        for shard in np.arange(0, 100):
            for nflows in [2000]:
                for nhosts in [21]:
                    spec = f"shard{shard}_nflows{nflows}_nhosts{nhosts}_lr{lr}Gbps"
                    size, fat, fsd, fcts, i_fcts = load_data(args.input, spec=spec, lr=lr, max_inflight_flows=max_inflight_flows)
                    
                    # Perform interactive inference
                    fct_tmp, sldn_tmp = interactive_inference(inference, size, fat, fsd, fcts, i_fcts, max_inflight_flows=max_inflight_flows, nhosts=nhosts, flow_size_threshold=flow_size_threshold, lr=lr)
                    print(f"Workload={shard}. fct shape: {fct_tmp.shape}, sldn shape: {sldn_tmp.shape}")
                    fct.append(fct_tmp)
                    sldn.append(sldn_tmp)
        fct = np.array(fct)
        sldn = np.array(sldn)
        print(f"Finished inference with {max_inflight_flows} inflight flows. fct shape: {fct.shape}, sldn shape: {sldn.shape}")
        np.savez(f'./res/inference_{max_inflight_flows}_t{flow_size_threshold}{empirical_str}.npz', fct=fct, sldn=sldn)
if __name__ == '__main__':
    main()
