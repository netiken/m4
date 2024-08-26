import torch
import numpy as np
from util.model import FlowSimLstm, FlowSimTransformer
from util.consts import get_base_delay_link, get_base_delay_transmission
import argparse
import yaml
from ctypes import *
from collections import defaultdict


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
        self, model_config, training_config, checkpoint_path, lr, device="cuda:0"
    ):
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
        dataset_config = model_config["dataset"]
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

        return model

    def preprocess(self, flows_info_active):
        sizes, fats, fcts_flowsim, flag_from_last_period = map(
            np.array, zip(*flows_info_active)
        )
        fats_ia = np.diff(fats)
        fats_ia = np.insert(fats_ia, 0, 0)

        n_links_passed = np.ones_like(fcts_flowsim) * 2
        base_delay = get_base_delay_link(sizes, n_links_passed, self.lr)
        i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
        fcts_flowsim += base_delay
        sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)

        sizes = np.log1p(sizes)
        fats_ia = np.log1p(fats_ia)
        input_data = np.column_stack(
            (sizes, fats_ia, sldn_flowsim, flag_from_last_period)
        )
        return torch.tensor(input_data, dtype=torch.float32).to(self.device)

    def postprocess(self, output):
        return output.cpu().detach().numpy()

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


class OnlineBusyPeriodPathProcessor:
    def __init__(self, nhosts, flow_size_threshold):
        self.nhosts = nhosts
        self.flow_size_threshold = flow_size_threshold

        self.flows = {}
        self.active_graphs = {}
        self.link_to_graph = {}
        self.large_flow_to_info = {}
        self.flow_to_size = {}
        self.graph_id_new = 0

        self.busy_periods = []
        self.busy_periods_len = []
        self.busy_periods_duration = []
        self.busy_periods_unique = set()
        self.current_time = 0

    def process_event(self, time, event, flow_id, links, size):
        self.current_time = time

        if event == "start":
            self.flow_to_size[flow_id] = size
            if size > self.flow_size_threshold:
                self.large_flow_to_info[flow_id] = (time, links)
            else:
                self._process_small_flow_start(flow_id, links)

        elif event == "end":
            self.flow_to_size.pop(flow_id, None)
            if flow_id in self.large_flow_to_info:
                self.large_flow_to_info.pop(flow_id, None)
            else:
                self._process_small_flow_end(flow_id, links)

    def _process_small_flow_start(self, flow_id, links):
        new_active_links = defaultdict(set)
        new_all_links = set()
        new_flows = set()
        new_all_flows = set()

        involved_graph_ids = set()
        for link in links:
            if link in self.link_to_graph:
                involved_graph_ids.add(self.link_to_graph[link])

        if involved_graph_ids:
            for gid in involved_graph_ids:
                graph = self.active_graphs[gid]
                new_active_links.update(graph["active_links"])
                new_all_links.update(graph["all_links"])
                new_flows.update(graph["active_flows"])
                new_all_flows.update(graph["all_flows"])
                if self.current_time > graph["start_time"]:
                    self.current_time = graph["start_time"]

                for link in graph["active_links"]:
                    self.link_to_graph[link] = self.graph_id_new
                del self.active_graphs[gid]

        for link in links:
            new_active_links[link].add(flow_id)
            new_all_links.add(link)
            self.link_to_graph[link] = self.graph_id_new
        new_flows.add(flow_id)
        new_all_flows.add(flow_id)
        for large_flow_id in self.large_flow_to_info:
            _, links_tmp = self.large_flow_to_info[large_flow_id]
            if large_flow_id not in new_all_flows and not links_tmp.isdisjoint(
                new_all_links
            ):
                new_all_flows.add(large_flow_id)

        self.active_graphs[self.graph_id_new] = {
            "active_links": new_active_links,
            "all_links": new_all_links,
            "active_flows": new_flows,
            "all_flows": new_all_flows,
            "start_time": self.current_time,
        }
        self.graph_id_new += 1

    def _process_small_flow_end(self, flow_id, links):
        graph = None
        for link in links:
            if link in self.link_to_graph:
                graph_id = self.link_to_graph[link]
                graph = self.active_graphs.get(graph_id)
                break

        if graph:
            for link in links:
                if flow_id in graph["active_links"][link]:
                    graph["active_links"][link].remove(flow_id)
                    if not graph["active_links"][link]:
                        del graph["active_links"][link]
                        del self.link_to_graph[link]

            if flow_id in graph["active_flows"]:
                graph["active_flows"].remove(flow_id)

            n_small_flows = len(
                [
                    flow_id
                    for flow_id in graph["active_flows"]
                    if self.flow_to_size[flow_id] <= self.flow_size_threshold
                ]
            )

            if n_small_flows == 0:
                self.busy_periods.append(tuple(graph["all_flows"]))
                self.busy_periods_len.append(len(graph["all_flows"]))
                self.busy_periods_duration.append(
                    [graph["start_time"], self.current_time]
                )
                self.busy_periods_unique.update(graph["all_flows"])

                del self.active_graphs[graph_id]

    def finalize(self):
        return self.busy_periods, self.busy_periods_duration, self.busy_periods_unique


def run_flow_simulation(flows_info, nhosts=5, lr=10):
    size, fat, fsd = map(np.array, zip(*flows_info))
    nflows = len(size)

    fats_pt = make_array(c_double, fat)
    sizes_pt = make_array(c_double, size)
    src_pt = make_array(c_int, fsd[:, 0])
    dst_pt = make_array(c_int, fsd[:, 1])
    topo_pt = make_array(c_int, np.array([1, 4]))

    res = C_LIB.get_fct_mmf(
        nflows, fats_pt, sizes_pt, src_pt, dst_pt, nhosts, topo_pt, 2, 8, 2, lr
    )
    estimated_fcts = np.fromiter(res.estimated_fcts, dtype=np.float64, count=nflows)

    C_LIB.free_fctstruct(res)
    return estimated_fcts


def load_data(dir_input, spec, topo_type="_topo-pl-5_s0", lr=10, max_inflight_flows=0):
    topo_type += f"_i{max_inflight_flows}"
    dir_input_tmp = f"{dir_input}/{spec}"

    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    size = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
    fat = np.load(f"{dir_input_tmp}/fat.npy")[fid]
    fsd = np.load(f"{dir_input_tmp}/fsd.npy")[fid]
    assert np.all(fid[:-1] <= fid[1:]) and len(fid) % 10000 == 0

    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")

    assert len(size) == len(fcts)
    return size, fat, fsd, fcts, i_fcts


def interactive_inference_path(
    inference,
    size,
    fat,
    fsd,
    fcts,
    i_fcts,
    nhosts=5,
    flow_size_threshold=100000,
    lr=10,
    max_inflight_flows=5,
):
    if max_inflight_flows == 0:
        max_inflight_flows = 1000000
    n_flows_total = len(size)

    flows_period = {}

    flow_completion_times = {}
    flow_fct_sldn = {}
    current_time = 0
    inflight_flows = 0

    flow_id_in_prop = 0
    processor = OnlineBusyPeriodPathProcessor(nhosts, flow_size_threshold)

    while flow_id_in_prop < n_flows_total or len(flows_period) > 0:
        flow_arrival_time = float("inf")
        flow_completion_time = float("inf")
        if flow_id_in_prop < n_flows_total:
            if inflight_flows < max_inflight_flows:
                flow_arrival_time = np.maximum(fat[flow_id_in_prop], current_time)

        if flows_period:
            flows_info = [
                [size[flow_id], fat[flow_id], fsd[flow_id]] for flow_id in flows_period
            ]

            fcts_flowsim = run_flow_simulation(flows_info, nhosts, lr)

            flows_info_active = []
            flow_id_info_active = []
            for flow_idx, flow_id in enumerate(flows_period):
                flow_info = flows_period[flow_id]
                if not flow_info[0]:
                    flow_id_info_active.append(flow_id)
                    links = set()
                    links.add((fsd[flow_id][0], nhosts + fsd[flow_id][0]))
                    for link_idx in range(fsd[flow_id][0], fsd[flow_id][1]):
                        links.add((nhosts + link_idx, nhosts + link_idx + 1))
                    links.add((nhosts + fsd[flow_id][1], fsd[flow_id][1]))

                    flows_info_active.append(
                        [
                            size[flow_id],
                            fat[flow_id],
                            fcts_flowsim[flow_idx],
                            flow_info[1],
                            links,
                        ]
                    )

            predictions = inference.infer(flows_info_active)
            sldn_est = predictions[0, :, 0]

            fct_stamp_est = []
            for idx, sldn_tmp in enumerate(sldn_est):
                flow_id = flow_id_info_active[idx]
                if flow_id in flow_completion_times:
                    fct_stamp_est.append(np.inf)
                else:
                    fct_stamp_est.append(fat[flow_id] + sldn_tmp * i_fcts[flow_id])

            min_idx = np.argmin(fct_stamp_est, axis=0)

            completed_flow_id = flow_id_info_active[min_idx]
            sldn_min = sldn_est[min_idx]
            flow_completion_time = fct_stamp_est[min_idx]
            fat_min = fat[completed_flow_id]

        if flow_arrival_time < flow_completion_time:
            current_time = flow_arrival_time
            inflight_flows += 1
            flows_period[flow_id_in_prop] = [0, 0]
            flow_id_in_prop += 1
        else:
            current_time = flow_completion_time
            flow_completion_times[completed_flow_id] = flow_completion_time - fat_min
            flow_fct_sldn[completed_flow_id] = sldn_min
            inflight_flows -= 1

            if inflight_flows == 0:
                flows_period = {}
            else:
                n_small_flows = len(
                    [
                        flow_id
                        for flow_id in flows_period
                        if size[flow_id] < flow_size_threshold
                        and flow_id not in flow_completion_times
                    ]
                )
                if n_small_flows == 0:
                    for flow_id in flows_period:
                        if not flows_period[flow_id][0]:
                            flow_info = flows_period[flow_id]
                            flows_period[flow_id][1] = 1
                            if flow_id in flow_completion_times:
                                flows_period[flow_id][0] = 1

    data_dict = {}
    for flow_id in flow_completion_times:
        predicted_completion_time = flow_completion_times[flow_id]
        actual_completion_time = fcts[flow_id]
        predicted_sldn = flow_fct_sldn[flow_id]
        assert predicted_sldn != np.inf
        actual_sldn = fcts[flow_id] / i_fcts[flow_id]

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
    parser = argparse.ArgumentParser(
        description="Interactive Inference Script for Path Scenario"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the YAML configuration file",
        default="./config/test_config_lstm_path.yaml",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input data directory",
        default="/data2/lichenni/perflow_path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the output predictions",
        default="/data2/lichenni/output_perflow",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)
        model_config = config_info["model"]
        training_config = config_info["training"]
        data_config = config_info["dataset"]

    lr = data_config["lr"]
    flow_size_threshold = data_config["flow_size_threshold"]

    args.checkpoint = f"{args.output}/path_{flow_size_threshold}_shard1000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/last.ckpt"
    inference = Inference(
        model_config, training_config, checkpoint_path=args.checkpoint, lr=lr
    )
    empirical_str = "_empirical"
    args.input += empirical_str

    print(f"Start inference with flow_size_threshold={flow_size_threshold}")

    for max_inflight_flows in [0]:
        fct, sldn = [], []
        for shard in np.arange(0, 10):
            for nflows in [10000]:
                for nhosts in [5]:
                    spec = f"shard{shard}_nflows{nflows}_nhosts{nhosts}_lr{lr}Gbps"
                    size, fat, fsd, fcts, i_fcts = load_data(
                        args.input,
                        spec=spec,
                        lr=lr,
                        max_inflight_flows=max_inflight_flows,
                    )

                    fct_tmp, sldn_tmp = interactive_inference_path(
                        inference,
                        size,
                        fat,
                        fsd,
                        fcts,
                        i_fcts,
                        nhosts=nhosts,
                        flow_size_threshold=flow_size_threshold,
                        lr=lr,
                        max_inflight_flows=max_inflight_flows,
                    )
                    print(
                        f"Finished workload={shard}. fct shape: {fct_tmp.shape}, sldn shape: {sldn_tmp.shape}"
                    )
                    fct.append(fct_tmp)
                    sldn.append(sldn_tmp)
        fct = np.array(fct)
        sldn = np.array(sldn)
        print(
            f"Finished inference with {max_inflight_flows} inflight flows. fct shape: {fct.shape}, sldn shape: {sldn.shape}"
        )
        np.savez(
            f"./res/inference_{max_inflight_flows}_t{flow_size_threshold}{empirical_str}.npz",
            fct=fct,
            sldn=sldn,
        )


if __name__ == "__main__":
    main()
