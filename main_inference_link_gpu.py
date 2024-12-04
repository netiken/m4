import torch
import numpy as np
from util.model import FlowSimLstm, FlowSimTransformer
from util.consts import get_base_delay_link, get_base_delay_transmission
import argparse
import yaml
from ctypes import *
import time
from collections import defaultdict
import traceback

from torch.profiler import profile, record_function, ProfilerActivity

torch.set_float32_matmul_precision("high")


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
    def __init__(
        self,
        dataset_config,
        model_config,
        training_config,
        checkpoint_path,
        lr,
        device="cuda:0",
    ):
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.training_config = training_config
        self.lr = lr
        self.device = torch.device(f"cuda:{training_config['gpu'][0]}")
        # self.device = torch.device(f"cpu")
        self.hidden_size = model_config["hidden_size"]
        self.enable_flowsim_diff = dataset_config.get("enable_flowsim_diff", False)
        self.model_name = model_config["model_name"]
        (
            gcn_layers_tmp,
            self.lstmcell_rate,
            self.lstmcell_time,
            self.output_layer,
        ) = self.load_model(checkpoint_path)
        gcn_layers = []
        for gcn in gcn_layers_tmp:
            gcn = gcn.to(self.device)
            gcn.eval()
            gcn = torch.jit.script(gcn)
            # gcn = torch.compile(gcn)
            gcn_layers.append(gcn)
        self.gcn_layers = gcn_layers

        # self.lstmcell_rate = torch.compile(self.lstmcell_rate)
        # self.lstmcell_time = torch.compile(self.lstmcell_time)
        # self.output_layer = torch.compile(self.output_layer)

        self.lstmcell_rate = torch.jit.script(self.lstmcell_rate)
        self.lstmcell_time = torch.jit.script(self.lstmcell_time)
        self.output_layer = torch.jit.script(self.output_layer)

        # self.rtt = self.model.rtt
        self.rtt = 0
        self.z_t_link = torch.zeros((len([0]), self.hidden_size), device=self.device)
        self.z_t_link[0, 0] = 1.0
        self.one_hot_type_a = torch.tensor([1, 0], dtype=torch.float32).to(self.device)
        self.one_hot_type_b = torch.tensor([0, 1], dtype=torch.float32).to(self.device)

        # self.save_models()

    def save_models(self, directory="./models"):

        for idx, gcn_layer in enumerate(self.gcn_layers):
            torch.jit.script(gcn_layer).save(f"{directory}/gcn_layer_{idx}.pt")

        # Save LSTM/GRU cells and output layer
        torch.jit.script(self.lstmcell_rate).save(f"{directory}/lstmcell_rate.pt")
        torch.jit.script(self.lstmcell_time).save(f"{directory}/lstmcell_time.pt")
        torch.jit.script(self.output_layer).save(f"{directory}/output_layer.pt")

    def load_model(self, checkpoint_path):
        model_config = self.model_config
        training_config = self.training_config
        dataset_config = self.dataset_config
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
                enable_positional_encoding=model_config.get(
                    "enable_positional_encoding", False
                ),
                enable_gnn=model_config.get("enable_gnn", False),
                enable_lstm=model_config.get("enable_lstm", False),
                current_period_len_idx=dataset_config.get(
                    "current_period_len_idx", None
                ),
                enable_lstm_in_gnn=model_config.get("enable_lstm_in_gnn", False),
                enable_link_state=model_config.get("enable_link_state", False),
                enable_flowsim_diff=dataset_config.get("enable_flowsim_diff", False),
                enable_remainsize=dataset_config.get("enable_remainsize", False),
                enable_path=dataset_config.get("enable_path", False),
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

        gcn_layers = [gcn.to(self.device).eval() for gcn in model.gcn_layers]
        model.lstmcell_rate.to(self.device).eval()
        model.lstmcell_time.to(self.device).eval()
        model.output_layer.to(self.device).eval()
        return (
            gcn_layers,
            model.lstmcell_rate,
            model.lstmcell_time,
            model.output_layer,
        )

    def infer(self, h_vec):
        with torch.no_grad():
            h_vec_res = self.output_layer(h_vec)
        return h_vec_res

    def update_rate(self, h_vec, edges_a_to_b_active, active_links=[0]):
        with torch.no_grad():
            # Expand one-hot encodings for type_a and type_b
            one_hot_a_expanded = self.one_hot_type_a.expand(h_vec.size(0), -1)
            one_hot_b_expanded = self.one_hot_type_b.expand(self.z_t_link.size(0), -1)

            # Append one-hot encodings to h_vec (type_a) and z_t_link (type_b)
            x_combined = torch.cat(
                [
                    torch.cat([one_hot_a_expanded, h_vec], dim=-1),
                    torch.cat([one_hot_b_expanded, self.z_t_link], dim=-1),
                ],
                dim=0,
            )

            # Number of type_a nodes
            num_type_a = h_vec.size(0)

            # Adjust the indices for bidirectional edges
            edge_index_a_to_b = torch.stack(
                [edges_a_to_b_active[0], edges_a_to_b_active[1] + num_type_a], dim=0
            )
            edge_index_combined = torch.cat(
                [
                    edge_index_a_to_b,
                    torch.stack(
                        [
                            edges_a_to_b_active[1] + h_vec.size(0),
                            edges_a_to_b_active[0],
                        ],
                        dim=0,
                    ),
                ],
                dim=1,
            )
            for gcn in self.gcn_layers:
                x_combined = gcn(x_combined, edge_index_combined)
            z_t_tmp = x_combined[:num_type_a]
            h_vec_res = self.lstmcell_rate(z_t_tmp, h_vec)

        return h_vec_res

    def update_time(self, h_vec, time_deltas):
        with torch.no_grad():
            h_vec_res = self.lstmcell_time(
                time_deltas,
                h_vec,
            )
        return h_vec_res


def run_flow_simulation(size, fat, fsd, nhosts=21, lr=10):
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
    topo_pt = make_array(c_int, np.array([1, 1]))

    # Run the flow simulation
    res = C_LIB.get_fct_mmf(
        nflows, fats_pt, sizes_pt, src_pt, dst_pt, nhosts, topo_pt, 2, 8, 2, lr
    )
    estimated_fcts = np.fromiter(res.estimated_fcts, dtype=np.float64, count=nflows)

    C_LIB.free_fctstruct(res)
    return estimated_fcts


def get_flowsim_sldn(sizes, fct, lr):
    n_links_passed = np.ones_like(fct) * 2
    base_delay = get_base_delay_link(sizes, n_links_passed, lr)
    i_fcts_flowsim = get_base_delay_transmission(sizes, lr) + base_delay
    fcts_flowsim = fct + base_delay
    sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
    return sldn_flowsim


def load_data(dir_input, spec, topo_type="_topo-pl-21_s0", lr=10, max_inflight_flows=0):
    topo_type += f"_i{max_inflight_flows}"
    dir_input_tmp = f"{dir_input}/{spec}"

    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    size = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
    fat = np.load(f"{dir_input_tmp}/fat.npy")[fid]
    fsd = np.load(f"{dir_input_tmp}/fsd.npy")[fid]
    assert np.all(fid[:-1] <= fid[1:])

    fct = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fct = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")

    fat = fat - fat[0]
    assert len(size) == len(fct)
    return size, fat, fsd, fct, i_fct


def interactive_inference_flowsim(
    size,
    fat,
    fsd,
    fct,
    i_fct,
    nhosts=21,
    lr=10,
):
    time_start = time.time()
    fcts_flowsim = run_flow_simulation(
        size,
        fat,
        fsd,
        nhosts,
        lr,
    )
    sldn_flowsim = get_flowsim_sldn(size, fcts_flowsim, lr)
    flow_metric = {}
    for flowid in range(len(size)):
        flow_metric[flowid] = [fcts_flowsim[flowid], sldn_flowsim[flowid]]
    time_elapsed = time.time() - time_start
    print(f"Time elapsed: {time_elapsed}")

    data_dict = {}
    for flow_id in flow_metric:
        predicted_completion_time = flow_metric[flow_id][0]
        actual_completion_time = fct[flow_id]
        predicted_sldn = flow_metric[flow_id][1]
        assert predicted_sldn != np.inf
        actual_sldn = fct[flow_id] / i_fct[flow_id]
        data_dict[flow_id] = [
            predicted_completion_time,
            actual_completion_time,
            predicted_sldn,
            actual_sldn,
        ]

    sorted_flow_ids = sorted(data_dict.keys())
    res = np.array([data_dict[flow_id] for flow_id in sorted_flow_ids])

    return sldn_flowsim, res[:, :2], res[:, 2:]


def interactive_inference(
    inference,
    size,
    fat,
    fsd,
    fct,
    i_fct,
    sldn_flowsim_total,
    nhosts=21,
    lr=10,
    n_flows_active_max=3000,
):
    # n_flows_total = len(size)

    # flowid_total = {}
    # flowid_active = {}

    i_fct_tensor = torch.tensor(i_fct, dtype=torch.float32, device=inference.device)
    fat_tensor = torch.tensor(fat, dtype=torch.float32, device=inference.device)
    h_vec_dim = inference.hidden_size
    h_vec = torch.zeros(
        (n_flows_active_max, h_vec_dim),
        dtype=torch.float32,
        device=inference.device,
    )
    size_log = torch.tensor(
        np.log2(size / 1000.0 + 1), dtype=torch.float32, device=inference.device
    )
    edges_a_to_b = torch.tensor(
        [[i, 0] for i in range(n_flows_active_max)],
        dtype=torch.long,
        device=inference.device,
    ).T
    time_deltas = torch.zeros(
        (n_flows_active_max, 1),
        dtype=torch.float32,
        device=inference.device,
    )
    sldn_flowsim_total = torch.tensor(
        sldn_flowsim_total, dtype=torch.float32, device=inference.device
    )

    # Track active flows and completed flows using boolean masks and indices
    flow_active_mask = torch.zeros(
        n_flows_active_max, dtype=torch.bool, device=inference.device
    )

    n_flows_total = torch.tensor(len(size), device=inference.device)
    # n_flows_total = torch.tensor(100, device=inference.device)
    flow_metric = torch.zeros(n_flows_total, 2, device=inference.device)
    time_last = torch.tensor(0, device=inference.device)
    flow_id_in_prop = torch.tensor(0, device=inference.device)
    flow_id_start = torch.tensor(0, device=inference.device)
    time_clock = time.time()
    n_flow_completed = torch.tensor(0, device=inference.device)
    rtt = torch.tensor(0, device=inference.device)
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    # ) as prof:
    while n_flow_completed < n_flows_total:
        # if flow_id_in_prop % 1000 == 0:
        #     print(f"Flow {flow_id_in_prop} processed")
        flow_arrival_time = torch.tensor(float("inf"), device=inference.device)
        flow_completion_time = torch.tensor(float("inf"), device=inference.device)
        if flow_id_in_prop < n_flows_total:
            flow_arrival_time = fat_tensor[flow_id_in_prop]

        if flow_active_mask.any():
            # [0, 1, 3]
            flowidx_active_list = torch.nonzero(flow_active_mask).squeeze(-1)
            # [50, 51, 53]
            flowid_active_list = flowidx_active_list + flow_id_start

            # start_time = time.time()
            if inference.enable_flowsim_diff:
                sldn_flowsim = sldn_flowsim_total[flowid_active_list]

            predictions = inference.infer(h_vec[flowidx_active_list])
            sldn_est = predictions[:, 0]
            if inference.enable_flowsim_diff:
                sldn_est = sldn_est + sldn_flowsim
            else:
                sldn_est = sldn_est + 1.0

            fct_stamp_est = (
                fat_tensor[flowid_active_list]
                + sldn_est * i_fct_tensor[flowid_active_list]
            )

            min_idx = torch.argmin(fct_stamp_est, dim=0)

            completed_flow_id = flowid_active_list[min_idx]

            fat_min = fat_tensor[completed_flow_id]
            flow_completion_time = fct_stamp_est[min_idx]
            sldn_min = sldn_est[min_idx]

        if flow_arrival_time < flow_completion_time:
            time_delta = flow_arrival_time - time_last
            if flow_active_mask.any() and time_delta > rtt:
                time_deltas[: flow_active_mask.sum()] = time_delta / 1000.0
                h_vec[flow_active_mask] = inference.update_time(
                    h_vec[flow_active_mask],
                    time_deltas[: flow_active_mask.sum()],
                )
            h_vec[flow_id_in_prop - flow_id_start, 0] = size_log[flow_id_in_prop]
            flow_active_mask[flow_id_in_prop - flow_id_start] = True

            edges_a_to_b_active = edges_a_to_b[:, : flow_active_mask.sum()]
            h_vec[flow_active_mask] = inference.update_rate(
                h_vec[flow_active_mask], edges_a_to_b_active
            )
            flow_id_in_prop += 1
            time_last = flow_arrival_time
        else:
            if completed_flow_id >= n_flows_total:
                break
            flow_metric[completed_flow_id][0] = flow_completion_time - fat_min
            flow_metric[completed_flow_id][1] = sldn_min
            n_flow_completed += 1
            time_delta = flow_completion_time - time_last
            if time_delta > rtt:
                time_deltas[: flow_active_mask.sum()] = time_delta / 1000.0
                h_vec[flow_active_mask] = inference.update_time(
                    h_vec[flow_active_mask],
                    time_deltas[: flow_active_mask.sum()],
                )
            flow_active_mask[completed_flow_id - flow_id_start] = False
            if not flow_active_mask.any():
                h_vec.zero_()
                flow_id_start = flow_id_in_prop
            else:
                edges_a_to_b_active = edges_a_to_b[:, : flow_active_mask.sum()]
                h_vec[flow_active_mask] = inference.update_rate(
                    h_vec[flow_active_mask],
                    edges_a_to_b_active,
                )
            time_last = flow_completion_time

        # print(
        #     f"Current time: {time_last}, Total flows: {len(flowid_total)}"
        # )

    time_elapsed = time.time() - time_clock
    print(f"Time elapsed: {time_elapsed}")

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")

    data_dict = {}
    for flow_id in range(len(flow_metric)):
        predicted_completion_time = flow_metric[flow_id][0].item()
        actual_completion_time = fct[flow_id]
        predicted_sldn = flow_metric[flow_id][1].item()
        assert predicted_sldn != np.inf
        actual_sldn = fct[flow_id] / i_fct[flow_id]
        data_dict[flow_id] = [
            predicted_completion_time,
            actual_completion_time,
            predicted_sldn,
            actual_sldn,
        ]

    sorted_flow_ids = sorted(data_dict.keys())
    res = np.array([data_dict[flow_id] for flow_id in sorted_flow_ids])
    return res[:, :2], res[:, 2:]


def main():
    parser = argparse.ArgumentParser(description="Interactive Inference Script")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the YAML configuration file",
        default="./config/test_config_lstm_link.yaml",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input data directory",
        default="/data2/lichenni/perflow_link_size",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the output predictions",
        default="/data2/lichenni/output_perflow",
    )
    parser.add_argument(
        "--flowsim",
        action="store_true",
        help="If set, the simulation will run and output predictions",
        default=False,
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)
        model_config = config_info["model"]
        training_config = config_info["training"]
        data_config = config_info["dataset"]

    lr = data_config["lr"]

    if not args.flowsim:
        model_instance = "link"
        args.checkpoint = f"{args.output}/{model_instance}_shard1000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/last.ckpt"
        inference = Inference(
            data_config,
            model_config,
            training_config,
            checkpoint_path=args.checkpoint,
            lr=lr,
        )
    empirical_str = "_empirical"
    # empirical_str = ""
    args.input += empirical_str

    if args.flowsim:
        print("Running flow simulation")
        model_instance = "flowsim"
    else:
        print("Running m4's inference")
    # for max_inflight_flows in [0, 4, 6, 15]:
    for max_inflight_flows in [0]:
        fct_list, sldn_list = [], []
        for shard in np.arange(0, 1):
            for nflows in [2000]:
                for nhosts in [21]:
                    spec = f"shard{shard}_nflows{nflows}_nhosts{nhosts}_lr{lr}Gbps"
                    size, fat, fsd, fct, i_fct = load_data(
                        args.input,
                        spec=spec,
                        lr=lr,
                        max_inflight_flows=max_inflight_flows,
                    )

                    # Perform interactive inference
                    try:
                        sldn_flowsim, fct_tmp, sldn_tmp = interactive_inference_flowsim(
                            size,
                            fat,
                            fsd,
                            fct,
                            i_fct,
                            nhosts=nhosts,
                            lr=lr,
                        )
                        if not args.flowsim:
                            fct_tmp, sldn_tmp = interactive_inference(
                                inference,
                                size,
                                fat,
                                fsd,
                                fct,
                                i_fct,
                                sldn_flowsim,
                                nhosts=nhosts,
                                lr=lr,
                            )
                        print(
                            f"Finished workload={shard}. fct shape: {fct_tmp.shape}, sldn shape: {sldn_tmp.shape}"
                        )
                        fct_list.append(fct_tmp)
                        sldn_list.append(sldn_tmp)
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()  # This prints the full traceback of the exception
                        continue
            fct_arr = np.array(fct_list)
            sldn_arr = np.array(sldn_list)
            print(
                f"Finished inference. fct shape: {fct_arr.shape}, sldn shape: {sldn_arr.shape}"
            )
            # np.savez(
            #     f"./res/inference_{model_instance}{empirical_str}.npz",
            #     fct=fct_arr,
            #     sldn=sldn_arr,
            # )


if __name__ == "__main__":
    main()
