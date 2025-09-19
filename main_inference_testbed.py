import torch
import numpy as np
from util.model import FlowSimLstm
from util.consts import get_base_delay_path, get_base_delay_transmission
import argparse
import yaml
import time
from collections import defaultdict
import traceback
import os
import gc

torch.set_float32_matmul_precision("high")


class Inference:
    def __init__(
        self,
        dataset_config,
        model_config,
        training_config,
        checkpoint_path,
    ):
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.training_config = training_config
        self.lr = dataset_config["lr"]
        self.device = torch.device(f"cuda:{training_config['gpu'][0]}")
        self.hidden_size = model_config["hidden_size"]
        self.enable_link_state = model_config.get("enable_link_state", False)
        self.model_name = model_config["model_name"]
        self.enable_lstm = model_config.get("enable_lstm", False)
        self.enable_gnn = model_config.get("enable_gnn", False)
        
        # Load model components based on enabled flags
        (
            gcn_layers_tmp,
            self.lstmcell_rate,
            self.lstmcell_time,
            self.output_layer,
            self.lstmcell_rate_link,
            self.lstmcell_time_link,
            self.mlp_rate,
        ) = self.load_model(checkpoint_path)

        # Convert GNN layers if enabled
        if self.enable_gnn:
            self.gcn_layers = [
                torch.jit.script(gcn.to(self.device)) for gcn in gcn_layers_tmp
            ]
        else:
            self.gcn_layers = []

        # Convert LSTM components if enabled
        if self.enable_lstm:
            self.lstmcell_rate = torch.jit.script(self.lstmcell_rate.to(self.device))
            self.lstmcell_time = torch.jit.script(self.lstmcell_time.to(self.device))
            if self.enable_link_state:
                # Only initialize link state components if both LSTM and link state are enabled
                self.lstmcell_rate_link = torch.jit.script(
                    self.lstmcell_rate_link.to(self.device)
                )
                self.lstmcell_time_link = torch.jit.script(
                    self.lstmcell_time_link.to(self.device)
                )
        else:
            self.mlp_rate = torch.jit.script(self.mlp_rate.to(self.device))

        # Always convert output layer and set to eval mode
        self.output_layer = torch.jit.script(self.output_layer.to(self.device))
        self.output_layer.eval()

        self.rtt = 0
        self.n_link = dataset_config.get("n_links_max", 100)
        
        # Initialize link state tensors if enabled
        if self.enable_link_state:
            self.z_t_link = torch.zeros((self.n_link, self.hidden_size), device=self.device)
            self.z_t_link[:, 1] = 1.0
            self.z_t_link[:, 2] = 1.0
        else:
            self.z_t_link = None
        # self.save_models("./testbed/models_new_v3")
        # exit()
    def save_models(self, directory="./inference/models_topo"):
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.enable_gnn:
            for idx, gcn_layer in enumerate(self.gcn_layers):
                gcn_layer.save(f"{directory}/gnn_layer_{idx}.pt")

        if self.enable_lstm:
            self.lstmcell_rate.save(f"{directory}/lstmcell_rate.pt")
            self.lstmcell_time.save(f"{directory}/lstmcell_time.pt")
            if self.enable_link_state:
                self.lstmcell_rate_link.save(f"{directory}/lstmcell_rate_link.pt")
                self.lstmcell_time_link.save(f"{directory}/lstmcell_time_link.pt")
        else:
            self.mlp_rate.save(f"{directory}/mlp_rate.pt")

        self.output_layer.save(f"{directory}/output_layer.pt")

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
                enable_positional_encoding=model_config.get(
                    "enable_positional_encoding", False
                ),
                enable_gnn=model_config.get("enable_gnn", False),
                enable_lstm=model_config.get("enable_lstm", False),
                enable_link_state=model_config.get("enable_link_state", False),
                enable_remainsize=dataset_config.get("enable_remainsize", False),
                enable_queuelen=dataset_config.get("enable_queuelen", False),
                enable_topo=dataset_config.get("enable_topo", False),
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Only return link state components if both LSTM and link state are enabled
        should_use_link_state = self.enable_lstm and self.enable_link_state
        
        if should_use_link_state:
            return (
                model.gcn_layers if self.enable_gnn else [],
                model.lstmcell_rate if self.enable_lstm else None,
                model.lstmcell_time if self.enable_lstm else None,
                model.output_layer,
                model.lstmcell_rate_link,
                model.lstmcell_time_link,
                model.mlp_rate if not self.enable_lstm else None,
            )
        else:
            return (
                model.gcn_layers if self.enable_gnn else [],
                model.lstmcell_rate if self.enable_lstm else None,
                model.lstmcell_time if self.enable_lstm else None,
                model.output_layer,
                None,
                None,
                model.mlp_rate if not self.enable_lstm else None,
            )

    def infer(self, h_vec):
        with torch.no_grad():
            h_vec_res = self.output_layer(h_vec)
        return h_vec_res

    def update_rate(
        self, h_vec, edge_index_a_to_b, h_vec_link, n_flows, param_data=None
    ):
        with torch.no_grad():
            if self.enable_gnn:
                x_combined = torch.cat([h_vec, h_vec_link], dim=0)

                # Create bidirectional edges
                edge_index_combined = torch.cat(
                    [
                        edge_index_a_to_b,
                        torch.stack([edge_index_a_to_b[1], edge_index_a_to_b[0]], dim=0),
                    ],
                    dim=1,
                )

                for gcn in self.gcn_layers:
                    x_combined = gcn(x_combined, edge_index_combined)
                z_tmp = x_combined[:n_flows]
                z_tmp_link = x_combined[n_flows:]
            else:
                z_tmp = h_vec
                z_tmp_link = h_vec_link

            z_tmp = torch.cat([z_tmp, param_data], dim=1)

            if self.enable_lstm:
                h_vec_res = self.lstmcell_rate(z_tmp, h_vec)
                if self.enable_link_state:
                    h_vec_link_res = self.lstmcell_rate_link(z_tmp_link, h_vec_link)
                else:
                    h_vec_link_res = None
            else:
                h_vec_res = self.mlp_rate(z_tmp)
                if self.enable_link_state:
                    h_vec_link_res = z_tmp_link
                else:
                    h_vec_link_res = None

            return h_vec_res, h_vec_link_res

    def update_time(self, h_vec, time_deltas):
        with torch.no_grad():
            if self.enable_lstm:
                h_vec_res = self.lstmcell_time(time_deltas, h_vec)
            else:
                h_vec_res = h_vec  # No time update when LSTM is disabled
        return h_vec_res

    def update_time_link(self, h_vec, time_deltas):
        with torch.no_grad():
            if self.enable_lstm and self.enable_link_state:
                h_vec_res = self.lstmcell_time_link(time_deltas, h_vec)
            else:
                h_vec_res = h_vec  # No time update when LSTM is disabled or link state is disabled
        return h_vec_res


# def get_flowsim_sldn(sizes, fct, lr):
#     n_links_passed = np.ones_like(fct) * 2
#     base_delay = get_base_delay_link(sizes, n_links_passed, lr)
#     i_fcts_flowsim = get_base_delay_transmission(sizes, lr) + base_delay
#     fcts_flowsim = fct + base_delay
#     sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
#     return sldn_flowsim


def load_data(dir_input, spec, topo_type, lr=10, max_inflight_flows=0):
    dir_input_tmp = f"{dir_input}/{spec}"
    fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
    assert np.all(fid[:-1] <= fid[1:])
    size = np.load(f"{dir_input_tmp}/fsize.npy")
    fat = np.load(f"{dir_input_tmp}/fat.npy")
    fat = fat - fat[0]
    fct = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
    i_fct = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")

    # index_selected = fct/i_fct>50
    # index_selected = size<50
    # fct[index_selected] = i_fct[index_selected]
    
    link_list = np.load(
        f"{dir_input_tmp}/flink.npy",
    )
    link_dict = {link: idx for idx, link in enumerate(link_list)}
    link_info = np.load(
        f"{dir_input_tmp}/flow_to_path.npy",
        allow_pickle=True,
    )
    # link_info = [[link_dict[link] for link in link_info[i]] for i in fid]
    link_info = [[link_dict[link] for link in link_info[i]] for i in range(len(size))]

    flowid_to_linkid = defaultdict(list)
    edges_list = []
    for flow_idx in range(len(size)):
        for link_idx in link_info[flow_idx]:
            edges_list.append([flow_idx, link_idx])
            flowid_to_linkid[flow_idx].append(link_idx)
    edges_list = np.array(edges_list).T
    # assert len(size) == len(fct)

    n_links_passed = np.array([len(path) for path in link_info])
    base_delay = get_base_delay_path(size, n_links_passed, lr)
    i_fcts_flowsim = get_base_delay_transmission(size, lr) + base_delay
    fcts_flowsim_path = f"{dir_input_tmp}/flowsim_fct.npy"
    if os.path.exists(fcts_flowsim_path):
        fcts_flowsim = np.load(f"{dir_input_tmp}/flowsim_fct.npy") + base_delay
    else:
        fcts_flowsim = i_fcts_flowsim
    sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
    flowid_to_linkid = [flowid_to_linkid[i] for i in flowid_to_linkid]
    param_data_path = f"{dir_input_tmp}/param{topo_type}.npy"
    if os.path.exists(param_data_path):
        param_data = np.load(f"{dir_input_tmp}/param{topo_type}.npy")
    else:
        param_data = np.zeros(13)
    param_data_repeat = np.repeat(param_data[:, np.newaxis], len(size), axis=1).T
    return (
        size,
        fat,
        fct,
        i_fct,  # Use loaded ground truth ideal FCT
        edges_list,
        sldn_flowsim,
        flowid_to_linkid,
        param_data_repeat,
        fid,
    )


def interactive_inference(
    inference,
    fid,
    size,
    fat,
    fct,
    i_fct,
    sldn_flowsim,
    param_data,
    lr=10,
    n_flows_active_max=10000,
    edges_list=None,
    flowid_to_linkid=None,
    n_flows_total=1000,
):
    n_flows_total_min = min(len(size), n_flows_total)
    # print(f"n_flows_total_min: {n_flows_total_min}")
    n_flows_active_max = min(n_flows_active_max, n_flows_total_min)
    size = size[:n_flows_total_min]
    fat = fat[:n_flows_total_min]
    fct = fct[:n_flows_total_min]
    i_fct = i_fct[:n_flows_total_min]
    sldn_flowsim = sldn_flowsim[:n_flows_total_min]
    param_data = param_data[:n_flows_total_min]
    flowid_to_linkid = [flowid_to_linkid[i] for i in range(n_flows_total_min)]
    edge_mask = edges_list[0] < n_flows_total_min
    edges_list = edges_list[:, edge_mask]
    device = inference.device
    n_links = inference.n_link
    flowid_active_mask = torch.zeros(
        n_flows_active_max, dtype=torch.bool, device=device
    )

    i_fct_tensor = torch.tensor(i_fct, dtype=torch.float32, device=device)
    fat_tensor = torch.tensor(fat, dtype=torch.float32, device=device)
    size_tensor = torch.tensor(
        np.log2(size + 1), dtype=torch.float32, device=device
    )
    sldn_flowsim_tensor = torch.tensor(sldn_flowsim, dtype=torch.float32, device=device)
    param_data_tensor = torch.tensor(param_data, dtype=torch.float32, device=device)

    edges_list = torch.tensor(edges_list, dtype=torch.long, device=device)

    link_to_graph_id = -torch.ones(n_links, dtype=torch.int64, device=device)
    link_to_nflows = torch.zeros(n_links, dtype=torch.int64, device=device)
    flow_to_graph_id = -torch.ones(n_flows_total_min, dtype=torch.int64, device=device)
    flowid_to_linkid_tensor = [
        torch.tensor(x, dtype=torch.long, device=device) for x in flowid_to_linkid
    ]
    flowid_to_nlinks = torch.tensor(
        [len(x) for x in flowid_to_linkid], dtype=torch.int64, device=device
    )
    graph_id_counter = 0
    graph_id_cur = 0

    h_vec_dim = inference.hidden_size
    h_vec = torch.zeros(
        (n_flows_active_max, h_vec_dim), dtype=torch.float32, device=device
    )
    h_vec[:, 0] = 1.0
    h_vec[:, 2] = size_tensor
    h_vec[:, 3] = 6
    # h_vec[:, 4 : param_data_tensor.size(1) + 4] = param_data_tensor

    time_last = torch.zeros((n_flows_active_max, 1), dtype=torch.float32, device=device)

    flow_metric = torch.zeros(
        (n_flows_total_min, 2), dtype=torch.float32, device=device
    )

    flow_id_in_prop = 0
    n_flows_active = 0
    n_flows_completed = 0
    data_extra = torch.cat(
        (
            size_tensor.unsqueeze(1),
            sldn_flowsim_tensor.unsqueeze(1),
            torch.ones_like(flowid_to_nlinks).unsqueeze(1)*6,
            param_data_tensor,
        ),
        dim=1,
    )

    time_clock = time.time()
    while n_flows_completed < n_flows_total_min:
        flow_arrival_time = (
            fat[flow_id_in_prop]
            if flow_id_in_prop < n_flows_total_min
            else float("inf")
        )
        flow_completion_time = float("inf")

        if n_flows_active > 0:
            flowid_active_list = torch.where(flowid_active_mask)[0]
            input_tensor = torch.cat(
                (
                    data_extra[flowid_active_list, 2:],
                    h_vec[flowid_active_list],
                ),
                dim=1,
            )
            predictions = inference.infer(input_tensor)
            sldn_est = predictions[:, 0]
            
            sldn_est[sldn_est < 1.0] = 1.0
            fct_stamp_est = (
                fat_tensor[flowid_active_list]
                + sldn_est * i_fct_tensor[flowid_active_list]
            )
            min_idx = torch.argmin(fct_stamp_est)
            flow_completion_time = fct_stamp_est[min_idx].item()
            completed_flow_id = flowid_active_list[min_idx].item()
            sldn_min = sldn_est[min_idx].item()

            # Debug: Check prediction values
            if n_flows_completed < 5:  # Reduce debug output
                raw_pred = sldn_est[0].item() if len(sldn_est) > 0 else float('nan')
                input_mean = input_tensor.mean().item()
                actual_sldn = (fct[flowid_active_list[0].item()] / i_fct_tensor[flowid_active_list[0]].item()) if len(flowid_active_list) > 0 else float('nan')
                print(f"Flow {n_flows_completed}: pred={raw_pred:.4f}, actual={actual_sldn:.4f}, "
                      f"error={(abs(raw_pred-actual_sldn)/actual_sldn*100):.1f}%, "
                      f"input_mean={input_mean:.4f}")
                
        if flow_arrival_time < flow_completion_time:
            # New flow arrives
            flowid_active_mask[flow_id_in_prop] = True
            n_flows_active += 1
            time_cur = flow_arrival_time
            time_last[flow_id_in_prop] = time_cur
            flowid_cur = flow_id_in_prop
            flow_id_in_prop += 1

            linkid_list = flowid_to_linkid_tensor[flowid_cur]
            existing_graph_ids = torch.unique(
                link_to_graph_id[linkid_list][link_to_graph_id[linkid_list] != -1]
            )

            link_to_nflows[linkid_list] += 1
            if existing_graph_ids.numel() == 0:
                # Assign a new graph ID
                graph_id_cur = graph_id_counter
                flow_to_graph_id[flowid_cur] = graph_id_cur
                link_to_graph_id[linkid_list] = graph_id_cur
                graph_id_counter += 1
            elif existing_graph_ids.numel() == 1:
                # Assign the existing graph ID
                graph_id_cur = existing_graph_ids[0]
                flow_to_graph_id[flowid_cur] = graph_id_cur
                link_to_graph_id[linkid_list] = graph_id_cur
            else:
                # Merge graphs
                graph_id_cur = graph_id_counter
                # Update flows
                flow_mask = torch.isin(flow_to_graph_id, existing_graph_ids)
                flow_to_graph_id[flow_mask] = graph_id_cur
                # Update links
                link_mask = torch.isin(link_to_graph_id, existing_graph_ids)
                link_to_graph_id[link_mask] = graph_id_cur
                # Assign graph ID to current flow and links
                flow_to_graph_id[flowid_cur] = graph_id_cur
                link_to_graph_id[linkid_list] = graph_id_cur
                graph_id_counter += 1

        else:
            # Flow completes
            flow_metric[completed_flow_id, 0] = (
                flow_completion_time - fat_tensor[completed_flow_id]
            )
            flow_metric[completed_flow_id, 1] = sldn_min
            flowid_active_mask[completed_flow_id] = False
            n_flows_active -= 1
            n_flows_completed += 1
            time_cur = flow_completion_time
            flowid_cur = completed_flow_id

            graph_id_cur = flow_to_graph_id[flowid_cur].item()
            linkid_list = flowid_to_linkid_tensor[flowid_cur]

            link_to_nflows[linkid_list] -= 1
            flow_to_graph_id[flowid_cur] = -1
            no_flow_links = linkid_list[link_to_nflows[linkid_list] == 0]
            link_to_graph_id[no_flow_links] = -1

            if inference.enable_link_state:
                inference.z_t_link[no_flow_links] = 0.0
                inference.z_t_link[no_flow_links, 1] = 1.0
                inference.z_t_link[no_flow_links, 2] = 1.0

        flowid_active_mask_cur = torch.logical_and(
            flowid_active_mask, flow_to_graph_id == graph_id_cur
        )
        flowid_active_list_cur = torch.where(flowid_active_mask_cur)[0]
        # print(
        #     f"n_active_flows: {flowid_active_list_cur.numel()}, graph_id_cur: {graph_id_cur}, fat: {flow_arrival_time/1000.0}, fct: {flow_completion_time/1000.0}"
        # )
        if (
            flowid_active_list_cur is not None
            and flowid_active_list_cur.numel() > 0
            and flow_arrival_time < flow_completion_time
        ):
            # Get unique link indices
            edge_mask = flowid_active_mask_cur[edges_list[0]]
            edges_list_active = edges_list[:, edge_mask]

            active_link_idx, new_link_indices = torch.unique(
                edges_list_active[1], return_inverse=True, sorted=False
            )

            time_deltas = time_cur - time_last[flowid_active_list_cur]
            if time_deltas.max() > 0:
                time_deltas[:] = time_deltas.max()
                time_deltas_link = torch.full(
                    (active_link_idx.size(0), 1),
                    time_deltas.max(),
                    dtype=torch.float32,
                    device=device,
                )
                h_vec[flowid_active_list_cur] = inference.update_time(
                    h_vec[flowid_active_list_cur],
                    time_deltas / 1000.0,
                )
                if inference.enable_link_state:
                    inference.z_t_link[active_link_idx] = inference.update_time_link(
                        inference.z_t_link[active_link_idx],
                        time_deltas_link / 1000.0,
                    )
            time_last[flowid_active_list_cur] = time_cur

            n_flows_active_cur = flowid_active_list_cur.size(0)
            new_flow_indices = torch.searchsorted(
                flowid_active_list_cur, edges_list_active[0]
            )
            new_link_indices += n_flows_active_cur
            edges_list_active = torch.stack([new_flow_indices, new_link_indices], dim=0)

            h_vec_updated, h_vec_link_updated = inference.update_rate(
                h_vec[flowid_active_list_cur],
                edges_list_active,
                inference.z_t_link[active_link_idx],
                n_flows_active_cur,
                param_data_tensor[flowid_active_list_cur],
            )
            h_vec[flowid_active_list_cur] = h_vec_updated
            if inference.enable_link_state:
                inference.z_t_link[active_link_idx] = h_vec_link_updated
    time_elapsed = time.time() - time_clock
    print(f"Time elapsed: {time_elapsed}")

    # Prepare results
    res_fct = flow_metric[:, 0].cpu().numpy()[fid]
    res_sldn = flow_metric[:, 1].cpu().numpy()[fid]
    actual_fct = fct
    actual_sldn = fct / i_fct[fid]

    res_fct = np.stack([res_fct, actual_fct], axis=1)
    res_sldn = np.stack([res_sldn, actual_sldn], axis=1)

    return res_fct, res_sldn


def release_gpu_memory():
    """Utility function to release GPU memory."""
    gc.collect()  # Collect unused objects
    torch.cuda.empty_cache()  # Clear the GPU cache
    torch.cuda.ipc_collect()  # Clear shared memory if using multiprocessing


def main():
    parser = argparse.ArgumentParser(description="Interactive Inference Script")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the YAML configuration file",
        default="./config/test_config_testbed.yaml",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input data directory",
        default="./testbed/",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the output predictions",
        default="./testbed/output_perflow",
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

    max_inflight_flows = 0
    dataset_list = [
        ("eval_train", ["100_2","100_4","100_8","100_12","100_16"], 4000),
        # ("eval_train", None, 4000),
    ]
    model_list = [
        # ("m4", 35, 6),
        # ("m4_cut_all", 31, 6),
        # ("m4_cut_1500", 23, 6),
        # ("m4_cut_1500", 15, 6),
        # ("m4_cut_1500", 23, 6),
        # ("m4_cut_1500", 35, 6),
        # ("m4_cut_1500", 40, 6),
        # ("m4_cut_1500", 45, 6),
        # ("m4_cut_1500", 50, 6),
        # ("m4_cut_1500", 55, 6),
        # ("m4_cut_1500", 60, 6),
        # ("m4_cut_1500", 65, 6),
        # ("m4_cut_1500", 70, 6),
        # ("m4_cut_1500", 75, 6),
        # ("m4_cut_1500", 80, 6),
        # ("m4_cut_1500", 85, 6),
        # ("m4_cut_1500", 90, 6),
        # ("m4_cut_1500", 95, 6),
        # ("m4_cut_1500", 99, 6),
        # ("m4", 15, 6),
        # ("m4", 20, 6),
        # ("m4", 25, 6),
        # ("m4", 30, 6),
        # ("m4", 35, 6),
        # ("m4", 40, 6),
        # ("m4", 45, 6),
        # ("m4", 50, 6),
        # ("m4", 55, 6),
        # ("m4", 60, 6),
        # ("m4", 65, 6),
        # ("m4", 70, 6),
        # ("m4", 75, 6),
        # ("m4", 80, 6),
        # ("m4", 85, 6),
        # ("m4", 90, 6),
        ("m4", 95, 6),
        # ("m4", 99, 6),
        # ("m4_cut", 20, 6),
        # ("m4_cut", 25, 6),
        # ("m4_cut", 30, 6),
        # ("m4_cut", 35, 6),
        # ("m4_cut", 40, 6),
        # ("m4_cut", 45, 6),
        # ("m4_cut", 50, 6),
        # ("m4_cut", 55, 6),
        # ("m4_cut", 60, 6),
        # ("m4_cut", 65, 6),
        # ("m4_cut", 70, 6),
        # ("m4_cut", 75, 6),
        # ("m4_cut", 80, 6),
        # ("m4_cut", 85, 6),
        # ("m4_cut", 90, 6),
        # ("m4_cut", 95, 6),
        # ("m4_cut", 99, 6),
        # ("m4_cut_all", 11, 6),
        # ("m4_cut_all", 13, 6),
        # ("m4_cut_all", 15, 6),
        # ("m4_cut_all", 17, 6),
        # ("m4_cut_all", 19, 6),
        # ("m4_cut_all", 21, 6),
        # ("m4_cut_all", 23, 6),
        # ("m4_cut_all", 25, 6),
        # ("m4_cut_all", 27, 6),
        # ("m4_cut_all", 29, 6),
        # ("m4_cut_all", 31, 6),
        # ("m4_cut_all", 33, 6),
        # ("m4_cut_all", 35, 6),
        # ("m4_cut_all", 37, 6),
        # ("m4_cut_1500", 40, 6),
        # ("m4_cut_1500", 45, 6),
        # ("m4_cut_1500", 50, 6),
        # ("m4_cut_1500", 55, 6),
        # ("m4_cut_1500", 60, 6),
        # ("m4_cut_1500", 65, 6),
        # ("m4_cut_1500", 70, 6),
        # ("m4_cut_1500", 75, 6),
        # ("m4_cut_1500", 80, 6),
        # ("m4_cut_1500", 85, 6),
        # ("m4_cut_1500", 90, 6),
        # ("m4_cut_1500", 95, 6),
        # ("m4_cut_1500", 99, 6),
        # ("m4_1500", 10, 6),
        # ("m4_1500", 15, 6),
        # ("m4_1500", 20, 6),
        # ("m4_1500", 25, 6),
        # ("m4_1500", 30, 6),
        # ("m4_1500", 35, 6),
        # ("m4_1500", 40, 6),
        # ("m4_1500", 45, 6),
        # ("m4_1500", 50, 6),
        # ("m4_1500", 55, 6),
        # ("m4_1500", 60, 6),
        # ("m4_1500", 65, 6),
        # ("m4_1500", 70, 6),
        # ("m4_1500", 75, 6),
        # ("m4_1500", 80, 6),
        # ("m4_1500", 85, 6),
        # ("m4_1500", 90, 6),
        # ("m4_1500", 95, 6),
        # ("m4_1500", 99, 6),
    ]
    if args.flowsim:
        print("Running flow simulation")
        model_instance = "flowsim"
        for dataset_str, n_shards, n_flows_total in dataset_list:
            input_dir = args.input + dataset_str
            fct_list, sldn_list = [], []
            for shard in np.arange(n_shards):
                try:
                    spec = f"{shard}/ns3"
                    (
                        size,
                        fat,
                        fct,
                        i_fct,
                        edges_list,
                        sldn_flowsim,
                        flowid_to_linkid,
                        param_data,
                        fid,
                    ) = load_data(
                        input_dir,
                        topo_type=data_config["topo_type"],
                        spec=spec,
                        lr=data_config["lr"],
                        max_inflight_flows=max_inflight_flows,
                    )
                    res_fct = []
                    res_sldn = []
                    sldn_flowsim = sldn_flowsim[fid]
                    i_fct = i_fct[fid]
                    for flow_id in range(len(sldn_flowsim)):
                        predicted_completion_time = (
                            sldn_flowsim[flow_id] * i_fct[flow_id]
                        )
                        actual_completion_time = fct[flow_id]
                        predicted_sldn = sldn_flowsim[flow_id]
                        assert predicted_sldn != np.inf
                        actual_sldn = fct[flow_id] / i_fct[flow_id]
                        res_fct.append(
                            [
                                predicted_completion_time,
                                actual_completion_time,
                            ]
                        )
                        res_sldn.append([predicted_sldn, actual_sldn])
                    res_fct = np.array(res_fct)
                    res_sldn = np.array(res_sldn)
                    print(
                        f"Finished workload={shard}. fct shape: {res_fct.shape}, sldn shape: {res_sldn.shape}"
                    )
                    res_fct_tmp = np.zeros((n_flows_total, 2))
                    res_sldn_tmp = np.zeros((n_flows_total, 2))
                    res_fct_tmp[: res_fct.shape[0], :] = res_fct
                    res_sldn_tmp[: res_sldn.shape[0], :] = res_sldn
                    fct_list.append(res_fct_tmp)
                    sldn_list.append(res_sldn_tmp)
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()  # This prints the full traceback of the exception
                    continue
                fct_arr = np.array(fct_list)
                sldn_arr = np.array(sldn_list)
                print(
                    f"Finished inference. fct shape: {fct_arr.shape}, sldn shape: {sldn_arr.shape}"
                )
                np.savez(
                    f"./testbed/res/{model_instance}_{dataset_str}.npz",
                    fct=fct_arr,
                    sldn=sldn_arr,
                )
    else:
        print("Running m4's inference")
        for dataset_str, spec_list, n_flows_total in dataset_list:
            input_dir = args.input + dataset_str
            for model_instance, model_ckpt, model_shard in model_list:
                model_name_loaded = f"last_epoch={model_ckpt:03d}"
                checkpoint = f"{args.output}/{model_instance}_shard{model_shard}_nflows1_nhosts1_lr10Gbps/version_0/checkpoints/{model_name_loaded}.ckpt"
                inference = Inference(
                    data_config,
                    model_config,
                    training_config,
                    checkpoint_path=checkpoint,
                )
                print(f"Loaded model: {model_instance}/{model_name_loaded}")
                fct_list, sldn_list = [], []
                if spec_list is None:
                    spec_list=os.listdir(input_dir)
                for spec in spec_list:
                    try:
                        spec = f"{spec}/ns3"
                        (
                            size,
                            fat,
                            fct,
                            i_fct,
                            edges_list,
                            sldn_flowsim,
                            flowid_to_linkid,
                            param_data,
                            fid,
                        ) = load_data(
                            input_dir,
                            topo_type=data_config["topo_type"],
                            spec=spec,
                            lr=data_config["lr"],
                            max_inflight_flows=max_inflight_flows,
                        )
                        res_fct, res_sldn = interactive_inference(
                            inference,
                            fid,
                            size,
                            fat,
                            fct,
                            i_fct,
                            sldn_flowsim,
                            param_data,
                            lr=data_config["lr"],
                            n_flows_active_max=n_flows_total,
                            edges_list=edges_list,
                            flowid_to_linkid=flowid_to_linkid,
                            n_flows_total=n_flows_total,
                        )
                        # print(
                        #     f"Finished workload={shard}. fct shape: {res_fct.shape}, sldn shape: {res_sldn.shape}"
                        # )
                        res_fct_tmp = np.zeros((n_flows_total, 2))
                        res_sldn_tmp = np.zeros((n_flows_total, 2))
                        res_fct_tmp[: res_fct.shape[0], :] = res_fct
                        res_sldn_tmp[: res_sldn.shape[0], :] = res_sldn
                        fct_list.append(res_fct_tmp)
                        sldn_list.append(res_sldn_tmp)
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()  # This prints the full traceback of the exception
                        continue
                    fct_arr = np.array(fct_list)
                    sldn_arr = np.array(sldn_list)
                    # print(
                    #     f"Finished inference. fct shape: {fct_arr.shape}, sldn shape: {sldn_arr.shape}"
                    # )
                    np.savez(
                        f"./testbed/res/{model_instance}_{model_ckpt}_{dataset_str}.npz",
                        fct=fct_arr,
                        sldn=sldn_arr,
                    )
                del inference
                release_gpu_memory()


if __name__ == "__main__":
    main()
