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
        self.n_gnn_connection_limit = 100

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
                enable_positional_encoding=model_config.get(
                    "enable_positional_encoding", False
                ),
                enable_gnn=model_config.get("enable_gnn", False),
                enable_lstm=model_config.get("enable_lstm", False),
                enable_path=True,
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
                enable_val=training_config["enable_val"],
                enable_dist=training_config["enable_dist"],
                enable_path=True,
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        return model

    def preprocess(self, flows_info_active, nhosts):
        sizes, fats, fcts_flowsim, fsd, flag_from_last_period = map(
            np.array, zip(*flows_info_active)
        )
        fid_period = np.arange(len(sizes))
        dict_tmp = defaultdict(list)
        for i in fid_period:
            key = (fsd[i][0] - fsd[i][1], fsd[i][0])
            dict_tmp[key].append(i)
        sorted_keys = sorted(dict_tmp.keys())
        fid_period = np.concatenate([dict_tmp[key] for key in sorted_keys])
        lengths_per_path = np.array([len(dict_tmp[key]) for key in sorted_keys])
        n_paths_per_batch = len(sorted_keys)

        fats_ia = fats - np.min(fats)
        n_links_passed = abs(fsd[:, 1] - fsd[:, 0]) + 2
        base_delay = get_base_delay_link(sizes, n_links_passed, self.lr)
        i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
        fcts_flowsim += base_delay
        sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)

        sizes = np.log2(sizes / 1000.0 + 1)
        fats_ia = np.log2(fats_ia / 10000.0 + 1)
        input_data = np.column_stack(
            (sizes, fats_ia, sldn_flowsim, n_links_passed, flag_from_last_period)
        )

        edge_index = self.compute_edge_index(nhosts, fid_period, fsd)

        return (
            torch.tensor(input_data, dtype=torch.float32).to(self.device),
            edge_index,
            np.array([lengths_per_path]),
            np.array([n_paths_per_batch]),
        )

    def postprocess(self, output):
        return output.cpu().detach().numpy()

    def compute_edge_index(self, n_hosts, fid, fsd_flowsim):
        n_flows = len(fsd_flowsim)
        edge_index = [[0, 0, 1]]

        fid_idx = np.argsort(fid)
        fsd_flowsim = fsd_flowsim[fid_idx]

        src_dst_to_links = {}
        for src in range(n_hosts - 1):
            for dst in range(src + 1, n_hosts):
                link_set = set([(src, src + n_hosts), (dst + n_hosts, dst)])
                for link_idx in range(src, dst):
                    link_set.add((n_hosts + link_idx, n_hosts + link_idx + 1))
                src_dst_to_links[(src, dst)] = link_set

        for flow_node_idx in range(1, n_flows):
            n_gnn_connection = 0
            pair_target = (fsd_flowsim[flow_node_idx, 0], fsd_flowsim[flow_node_idx, 1])
            link_sets_head = src_dst_to_links[pair_target]

            other_flow_idx = flow_node_idx - 1
            while (
                other_flow_idx >= 0 and n_gnn_connection < self.n_gnn_connection_limit
            ):

                pair_other = (
                    fsd_flowsim[other_flow_idx, 0],
                    fsd_flowsim[other_flow_idx, 1],
                )

                # Check if other flow shares links with current flow
                if pair_other == pair_target:
                    break

                link_sets_tail = src_dst_to_links[pair_other]
                overlapping_links = len(link_sets_head.intersection(link_sets_tail))

                # Only consider flows that interact within the timespan
                if overlapping_links > 0:
                    edge_index.append(
                        [
                            fid_idx[other_flow_idx],
                            fid_idx[flow_node_idx],
                            overlapping_links,
                        ]
                    )
                    edge_index.append(
                        [
                            fid_idx[flow_node_idx],
                            fid_idx[other_flow_idx],
                            overlapping_links,
                        ]
                    )
                    n_gnn_connection += 1
                other_flow_idx -= 1
        edge_index = np.array(edge_index).T

        # Sort edge_index by destination node (second row)
        sorted_indices = np.argsort(edge_index[1, :])
        edge_index = edge_index[:, sorted_indices]

        return edge_index

    def infer(self, flows_info_active, nhosts):
        data, edge_index, lengths_per_path, n_paths_per_batch = self.preprocess(
            flows_info_active, nhosts
        )
        lengths = np.array([len(data)])
        edge_index_len = np.array([edge_index.shape[1]])
        with torch.no_grad():

            if self.model_name == "lstm":
                output = self.model(
                    data.unsqueeze(0),
                    lengths,
                    edge_index,
                    edge_index_len,
                    lengths_per_path,
                    n_paths_per_batch,
                )
            elif self.model_name == "transformer":
                output, _ = self.model(data.unsqueeze(0), lengths)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        return self.postprocess(output)


class OnlineTrafficGenerator:
    def __init__(
        self,
        nhosts,
        flow_size_threshold,
        sizes,
    ):
        self.nhosts = nhosts
        self.flow_size_threshold = flow_size_threshold
        self.active_graphs = {}
        self.link_to_graph = {}
        self.large_flow_to_info = {}
        self.large_flow_to_unassigned = set()
        self.flow_to_size = sizes
        self.graph_id_new = 0

    def _create_subgraph(self, cur_time):
        subgraph_id = self.graph_id_new
        self.graph_id_new += 1
        self.active_graphs[subgraph_id] = {
            "active_links": defaultdict(set),
            # "all_links": set(),
            "active_flows": set(),
            "all_flows": set(),
            "start_time": cur_time,
        }
        return subgraph_id

    def _assign_flow_to_subgraph(self, flow_id, links, cur_time):
        involved_graph_ids = set()
        for link in links:
            if link in self.link_to_graph:
                involved_graph_ids.add(self.link_to_graph[link])

        if involved_graph_ids:
            # Merge the involved subgraphs into one
            subgraph_id = min(involved_graph_ids)
            main_graph = self.active_graphs[subgraph_id]

            for gid in involved_graph_ids:
                if gid == subgraph_id:
                    continue
                # Merge other graphs into the main graph
                other_graph = self.active_graphs.pop(gid)
                main_graph["active_links"].update(other_graph["active_links"])
                # main_graph["all_links"].update(other_graph["all_links"])
                main_graph["active_flows"].update(other_graph["active_flows"])
                main_graph["all_flows"].update(other_graph["all_flows"])
                for link in other_graph["active_links"]:
                    self.link_to_graph[link] = subgraph_id

            if cur_time > main_graph["start_time"]:
                main_graph["start_time"] = cur_time

        else:
            # No overlap, create a new subgraph
            subgraph_id = self._create_subgraph(cur_time)

        # Update the subgraph with the new flow
        for link in links:
            self.active_graphs[subgraph_id]["active_links"][link].add(flow_id)
            # self.active_graphs[subgraph_id]["all_links"].add(link)
            self.link_to_graph[link] = subgraph_id

        self.active_graphs[subgraph_id]["active_flows"].add(flow_id)
        self.active_graphs[subgraph_id]["all_flows"].add(flow_id)

        # Check if any large flow intersects with this new subgraph
        for large_flow_id, (
            large_flow_time,
            large_flow_links,
        ) in self.large_flow_to_info.items():
            if large_flow_id not in self.active_graphs[subgraph_id][
                "all_flows"
            ] and not large_flow_links.isdisjoint(links):
                self.active_graphs[subgraph_id]["all_flows"].add(large_flow_id)
                if large_flow_id in self.large_flow_to_unassigned:
                    self.large_flow_to_unassigned.remove(large_flow_id)

        return subgraph_id

    def get_active_flows(self, subgraph_id):
        flows = self.active_graphs[subgraph_id]["active_flows"]
        for large_flow_id in self.large_flow_to_info:
            if large_flow_id in self.active_graphs[subgraph_id]["all_flows"]:
                flows.add(large_flow_id)
        flows = sorted(flows)
        return flows

    def process_event(
        self, cur_time, event, flow_id, links, size, subgraph_id_completed=None
    ):
        if event == "start":
            if size > self.flow_size_threshold:
                self.large_flow_to_info[flow_id] = (cur_time, links)
                self.large_flow_to_unassigned.add(flow_id)
            else:
                self._assign_flow_to_subgraph(flow_id, links, cur_time)
        elif event == "end":
            if flow_id in self.large_flow_to_info:
                self.large_flow_to_info.pop(flow_id)
                if flow_id in self.large_flow_to_unassigned:
                    self.large_flow_to_unassigned.remove(flow_id)
                    return
            else:
                graph_id = subgraph_id_completed
                if graph_id is not None:
                    graph = self.active_graphs[graph_id]

                    for link in links:
                        if flow_id in graph["active_links"][link]:
                            graph["active_links"][link].remove(flow_id)
                            if not graph["active_links"][link]:
                                del graph["active_links"][link]
                                del self.link_to_graph[link]

                    graph["active_flows"].remove(flow_id)

                    if not any(
                        self.flow_to_size[flow] <= self.flow_size_threshold
                        for flow in graph["active_flows"]
                    ):
                        del self.active_graphs[graph_id]

    def get_subgraphs(self):
        return self.active_graphs

    def get_large_flows_unassigned(self):
        return sorted(list(self.large_flow_to_unassigned))

    def has_flows(self):
        return bool(self.active_graphs) or bool(self.large_flow_to_unassigned)


def run_flow_simulation(flows_info, nhosts=5, lr=10):
    size, fat, fsd = map(np.array, zip(*flows_info))
    nflows = len(size)

    fats_pt = make_array(c_double, fat)
    sizes_pt = make_array(c_double, size)
    src_pt = make_array(c_int, fsd[:, 0])
    dst_pt = make_array(c_int, fsd[:, 1])
    topo_pt = make_array(c_int, np.array([1, 1]))

    res = C_LIB.get_fct_mmf(
        nflows, fats_pt, sizes_pt, src_pt, dst_pt, nhosts, topo_pt, 1, 8, 2, lr
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
    fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
    assert np.all(fid[:-1] <= fid[1:]) and len(fid) % 10000 == 0

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
    src_dst_to_links = {}
    for src in range(nhosts - 1):
        for dst in range(src + 1, nhosts):
            link_set = set([(src, src + nhosts), (dst + nhosts, dst)])
            for link_idx in range(src, dst):
                link_set.add((nhosts + link_idx, nhosts + link_idx + 1))
            src_dst_to_links[(src, dst)] = link_set

    n_flows_total = len(size)

    flow_completion_times = {}
    flow_fct_sldn = {}

    traffic_generator = OnlineTrafficGenerator(nhosts, flow_size_threshold, sizes=size)
    current_time = 0

    inflight_flows = 0
    flow_id_in_prop = 0
    while len(flow_completion_times) != n_flows_total:
        flow_arrival_time = float("inf")
        flow_completion_time = float("inf")
        completed_flow_id = None

        if flow_id_in_prop < n_flows_total:
            if inflight_flows < max_inflight_flows:
                flow_arrival_time = max(fat[flow_id_in_prop], current_time)

        # Process flow completions across all subgraphs
        if traffic_generator.has_flows():
            large_flow_unassigned = traffic_generator.get_large_flows_unassigned()
            if len(large_flow_unassigned) > 0:
                flows_info = [
                    [size[flow_id], fat[flow_id], fsd[flow_id]]
                    for flow_id in large_flow_unassigned
                ]
                fcts_flowsim = run_flow_simulation(flows_info, nhosts, lr)
                for idx, flow_id in enumerate(large_flow_unassigned):
                    estimated_completion_time = fat[flow_id] + fcts_flowsim[idx]

                    if (
                        flow_id not in flow_completion_times
                        and estimated_completion_time < flow_completion_time
                    ):
                        flow_completion_time = estimated_completion_time
                        completed_flow_id = flow_id
                        sldn_min = 1
                        subgraph_id_completed = None

            subgraphs = traffic_generator.get_subgraphs()

            for subgraph_id, subgraph in subgraphs.items():
                period_start_time = subgraph["start_time"]
                flows_info = [
                    [size[flow_id], fat[flow_id], fsd[flow_id]]
                    for flow_id in subgraph["all_flows"]
                ]
                fcts_flowsim = run_flow_simulation(flows_info, nhosts, lr)

                flows_info_active = []
                flow_id_info_active = []
                flows = traffic_generator.get_active_flows(subgraph_id)
                for flow_idx, flow_id in enumerate(flows):
                    if flow_id not in flow_completion_times:
                        flow_id_info_active.append(flow_id)
                        flows_info_active.append(
                            [
                                size[flow_id],
                                fat[flow_id],
                                fcts_flowsim[flow_idx],
                                fsd[flow_id],
                                fat[flow_id] < period_start_time,
                            ]
                        )

                predictions = inference.infer(flows_info_active, nhosts)
                sldn_est = predictions[0, :, 0]

                for idx, sldn_tmp in enumerate(sldn_est):
                    flow_id = flow_id_info_active[idx]
                    estimated_completion_time = (
                        fat[flow_id] + sldn_tmp * i_fcts[flow_id]
                    )

                    if (
                        flow_id not in flow_completion_times
                        and estimated_completion_time < flow_completion_time
                    ):
                        flow_completion_time = estimated_completion_time
                        completed_flow_id = flow_id
                        sldn_min = sldn_tmp
                        subgraph_id_completed = subgraph_id

        # Determine the next event (either flow arrival or flow completion)
        if flow_arrival_time < flow_completion_time:
            # Next event is flow arrival
            current_time = flow_arrival_time
            links = src_dst_to_links[fsd[flow_id_in_prop][0], fsd[flow_id_in_prop][1]]
            traffic_generator.process_event(
                fat[flow_id_in_prop],
                "start",
                flow_id_in_prop,
                links,
                size[flow_id_in_prop],
            )
            inflight_flows += 1
            flow_id_in_prop += 1
        elif completed_flow_id is not None:
            # Next event is flow completion
            current_time = flow_completion_time
            flow_completion_times[completed_flow_id] = (
                flow_completion_time - fat[completed_flow_id]
            )
            flow_fct_sldn[completed_flow_id] = sldn_min

            links = src_dst_to_links[
                fsd[completed_flow_id][0], fsd[completed_flow_id][1]
            ]
            traffic_generator.process_event(
                current_time,
                "end",
                completed_flow_id,
                links,
                size[completed_flow_id],
                subgraph_id_completed,
            )
            inflight_flows -= 1

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

    args.checkpoint = f"{args.output}/path_{flow_size_threshold}_lstm_shard1000_nflows1_nhosts1_nsamples1_lr10Gbps/version_0/checkpoints/last.ckpt"
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
