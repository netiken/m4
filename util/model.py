import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from .func import serialize_fp32
import numpy as np
import logging
import struct
import os
from sortedcontainers import SortedSet

from .model_llama import Transformer, ModelArgs

import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, MessagePassing, SAGEConv
from torch_geometric.data import HeteroData, Batch, Data
from typing import Tuple


class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()
        return

    def forward(self, x):
        return torch.exp(x)


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, prediction, target, batch_index):
        n_batch = batch_index.max() + 1
        elementwise_loss = 0
        if n_batch == 1:
            elementwise_loss = torch.abs(prediction - target).mean()
        else:
            for i in range(n_batch):
                idx = batch_index == i
                elementwise_loss += torch.abs(prediction[idx] - target[idx]).mean()
            elementwise_loss /= n_batch
        return elementwise_loss


class TransformerBase(LightningModule):
    def __init__(
        self,
        n_layer,
        n_head,
        n_embd,
        block_size,
        vocab_size,
        output_dim,
        dropout,
        loss_fn_type,
        enable_val,
        enable_position,
        enable_causal,
        save_dir=None,
    ):
        super().__init__()
        if loss_fn_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_fn_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_fn_type}")

        conf = ModelArgs(
            dim=n_embd,
            n_layers=n_layer,
            n_heads=n_head,
            vocab_size=vocab_size,
            output_dim=output_dim,
            multiple_of=1,
            max_seq_len=block_size,
            dropout=dropout,
            enable_causal=enable_causal,
        )
        self.model_transformer = Transformer(conf)
        self.enable_val = enable_val
        self.save_dir = save_dir
        logging.info(
            f"model: FlowSimTransformer, loss_fn: {loss_fn_type}, n_layer: {n_layer}, n_head: {n_head}, n_embd: {n_embd}, block_size: {block_size}, vocab_size: {vocab_size}, output_dim: {output_dim}, dropout: {dropout}, enable_position: {enable_position}, enable_causal: {enable_causal}, enable_val: {enable_val}"
        )

    def export_to_bin_llama_v0(self, filepath):
        """Original export of llama2.c bin files, i.e. version v0"""
        model = self.model_transformer
        out_file = open(filepath, "wb")

        # first write out the header
        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        p = model.params
        shared_classifier = torch.equal(
            model.tok_embeddings.weight, model.output.weight
        )
        # legacy format uses negative/positive vocab size as a shared classifier flag
        if not shared_classifier:
            p.vocab_size = -p.vocab_size
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack(
            "iiiiiii",
            p.dim,
            hidden_dim,
            p.n_layers,
            p.n_heads,
            n_kv_heads,
            p.vocab_size,
            p.max_seq_len,
        )
        out_file.write(header)

        # next write out the embedding weights
        serialize_fp32(out_file, model.tok_embeddings.weight)
        serialize_fp32(out_file, model.tok_embeddings.bias)

        # now all the layers
        # attention weights
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wq.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wk.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wv.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wo.weight)
        # ffn weights
        for layer in model.layers:
            serialize_fp32(out_file, layer.ffn_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w1.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w2.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize_fp32(out_file, model.norm.weight)
        # freqs_cis
        serialize_fp32(out_file, model.freqs_cos[: p.max_seq_len])
        serialize_fp32(out_file, model.freqs_sin[: p.max_seq_len])

        # final classifier weights
        if not shared_classifier:
            serialize_fp32(out_file, model.output.weight)

        # write to binary file
        out_file.close()
        print(f"wrote {filepath}")

    def step(self, batch, batch_idx, tag=None):
        return None

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="train")

    def validation_step(self, batch, batch_idx):
        if self.enable_val:
            return self.step(batch, batch_idx, tag="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="test")


class FlowSimTransformer(TransformerBase):
    def __init__(
        self,
        n_layer=4,
        n_head=4,
        n_embd=64,
        block_size=64,
        vocab_size=50257,
        output_dim=None,
        dropout=0.0,
        compile=False,
        loss_fn_type="l1",
        enable_position=True,
        enable_causal=False,
        weight_decay=1e-2,
        learning_rate=6e-4,
        betas=[0.9, 0.95],
        batch_size=400,
        enable_dist=False,
        enable_val=True,
        save_dir=None,
    ):
        super().__init__(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            vocab_size=vocab_size,
            output_dim=output_dim,
            dropout=dropout,
            enable_position=enable_position,
            enable_causal=enable_causal,
            loss_fn_type=loss_fn_type,
            enable_val=enable_val,
            save_dir=save_dir,
        )
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas
        self.batch_size = batch_size
        self.enable_dist = enable_dist

    def step(self, batch, batch_idx, tag=None):
        input, output, lengths, spec, src_dst_pair_target_str = batch
        # Create mask based on lengths
        max_len = input.size(1)
        mask = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
        attention_mask = mask.to(input.device)

        # Pass the input through the transformer
        estimated, _ = self.model_transformer(input, attention_mask=attention_mask)

        # Mask the output and target
        # masked_estimated = estimated[mask]
        # masked_output = output[mask]
        est = torch.div(estimated, output).squeeze()
        gt = torch.ones_like(est)
        est = est.masked_fill(~attention_mask, 0)
        gt = gt.masked_fill(~attention_mask, 0)

        # Calculate the loss
        loss = self.loss_fn(est, gt)

        if self.enable_dist:
            self.log(
                f"{tag}_loss_sync",
                loss,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
        else:
            self.log(
                f"{tag}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )

        if tag == "test":
            test_dir = f"{self.save_dir}/{spec[0]}_{src_dst_pair_target_str[0]}"
            os.makedirs(test_dir, exist_ok=True)
            estimated = estimated.cpu().numpy()
            output = output.cpu().numpy()

            np.savez(f"{test_dir}/res.npz", queue_len_est=estimated, output=output)
        return loss

    def configure_optimizers(self):
        optimizer = self.model_transformer.configure_optimizers(
            self.weight_decay, self.learning_rate, self.betas
        )
        return optimizer


def MLP(input_size, hidden_size, output_size, dropout):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),  # First layer
        nn.Dropout(p=dropout),  # Dropout for regularization
        nn.ReLU(),  # Non-linearity
        nn.Linear(hidden_size, output_size),  # Second layer
        nn.ReLU(),  # Non-linearity
    )


class HomoGNNLayer(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.2):
        super(HomoGNNLayer, self).__init__()
        self.homogeneous_layer = HomoNetGNN(c_in=c_in, c_out=c_out, dropout=dropout)

    def forward(self, x, edge_index):
        # Combine type_a and type_b node features into a single homogeneous feature matrix
        # Apply the homogeneous GNN layer
        out_combined = self.homogeneous_layer(x, edge_index)

        return out_combined


# original
class HomoNetGNN(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.2):
        super(HomoNetGNN, self).__init__()
        self.conv = SAGEConv(
            c_in, c_out, aggr="sum", project=True
        )  # project=True is default
        # self.dropout = nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(c_out)
        # self.final_lin = nn.Linear(c_out, c_out)
        # self.final_lin = nn.Sequential(
        #     nn.Linear(c_out, c_out),  # First layer
        #     nn.ReLU(),  # Non-linearity
        #     # nn.Dropout(p=dropout),
        #     nn.Linear(c_out, c_out),  # Second layer
        # )

    def forward(self, x, edge_index):
        # Apply the SAGEConv layer (includes residual connection)
        out = self.conv(x, edge_index)
        out = self.norm(out)  # Apply normalization

        # Apply linear transformation, activation, and dropout
        # out = F.relu(self.final_lin(out))
        # out = self.dropout(out)

        return out


# v0
# class HomoNetGNN(nn.Module):
#     def __init__(self, c_in, c_out, dropout=0.2):
#         super(HomoNetGNN, self).__init__()
#         self.conv = SAGEConv(
#             c_in, c_out, aggr="sum", project=False
#         )  # project=True is default
#         # self.dropout = nn.Dropout(dropout)
#         self.norm = torch.nn.LayerNorm(c_out)

#     def forward(self, x, edge_index):
#         # Apply the SAGEConv layer (includes residual connection)
#         out = self.conv(x, edge_index)
#         out = self.norm(out)  # Apply normalization

#         return out


# v1
# class HomoNetGNN(nn.Module):
#     def __init__(self, c_in, c_out, dropout=0.2):
#         super(HomoNetGNN, self).__init__()
#         self.conv = SAGEConv(
#             c_in, c_out, aggr="mean", project=True
#         )  # project=True is default
#         self.norm = torch.nn.LayerNorm(c_out)

#     def forward(self, x, edge_index):
#         # Apply the SAGEConv layer (includes residual connection)
#         out = self.conv(x, edge_index)
#         out = self.norm(out)  # Apply normalization

#         return out


# class HomoNetGNN(nn.Module):
#     def __init__(self, c_in, c_out, dropout=0.2):
#         super(HomoNetGNN, self).__init__()
#         self.conv = SAGEConv(c_in, c_out, aggr="sum")
#         self.dropout = nn.Dropout(dropout)
#         self.norm = torch.nn.LayerNorm(c_out)
#         # self.final_lin = nn.Linear(c_in + c_out, c_out)
#         if c_in != c_out:
#             self.lin_proj = nn.Linear(c_in, c_out)
#         else:
#             self.lin_proj = nn.Identity()
#         self.final_lin = nn.Linear(c_out, c_out)

#     def forward(self, x, edge_index):
#         # Apply the GCNConv layer
#         out = self.conv(x, edge_index)
#         out = self.norm(out)  # Apply normalization
#         # Concatenate original features with the aggregated ones
#         # x = torch.cat([x, out], dim=-1)
#         x_proj = self.lin_proj(x)
#         x = x_proj + out

#         # Apply linear transformation, activation, and dropout
#         x = F.relu(self.final_lin(x))
#         x = self.dropout(x)

#         return x


# class HeterGNNLayer(nn.Module):
#     def __init__(
#         self, c_in_type_a, c_in_type_b, c_out, dropout=0.2, enable_lstm_in_gnn=False
#     ):
#         super(HeterGNNLayer, self).__init__()
#         self.c_in_type_a = c_in_type_a
#         self.c_in_type_b = c_in_type_b
#         self.c_out = c_out
#         self.dropout = nn.Dropout(dropout)
#         self.enable_lstm_in_gnn = enable_lstm_in_gnn

#         # Define custom message passing layers
#         self.conv_layers = HeteroConv(
#             {
#                 ("type_a", "a_to_b", "type_b"): HeterNetGNN(
#                     c_in_type_a, c_out, dropout, enable_lstm_in_gnn
#                 ),
#                 ("type_b", "b_to_a", "type_a"): HeterNetGNN(
#                     c_in_type_b, c_out, dropout, enable_lstm_in_gnn
#                 ),
#             },
#             aggr="sum",
#         )

#         # Linear layers for combining original and aggregated features
#         self.final_lin_type_a = nn.Linear(c_in_type_a + c_out, c_out)
#         self.final_lin_type_b = nn.Linear(c_in_type_b + c_out, c_out)

#     def forward(self, x_dict, edge_index_dict):
#         x_type_a, x_type_b = x_dict["type_a"], x_dict["type_b"]
#         # Edge index from type_a to type_b
#         out_dict = self.conv_layers(x_dict, edge_index_dict)

#         # Concatenate original and aggregated features
#         x_type_a = torch.cat([x_type_a, out_dict["type_a"]], dim=-1)
#         x_type_b = torch.cat([x_type_b, out_dict["type_b"]], dim=-1)

#         # Apply final linear layers, activation, and dropout
#         x_type_a = F.relu(self.final_lin_type_a(x_type_a))
#         x_type_a = self.dropout(x_type_a)

#         x_type_b = F.relu(self.final_lin_type_b(x_type_b))
#         x_type_b = self.dropout(x_type_b)

#         output_dict = {"type_a": x_type_a, "type_b": x_type_b}
#         return output_dict


# class HeterNetGNN(MessagePassing):
#     def __init__(self, in_channels, out_channels, dropout, enable_lstm_in_gnn):
#         super(HeterNetGNN, self).__init__(aggr="add")
#         self.enable_lstm_in_gnn = enable_lstm_in_gnn

#         if enable_lstm_in_gnn:
#             self.message_func = nn.LSTM(in_channels, out_channels, batch_first=True)
#         else:
#             self.message_func = nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         return self.propagate(edge_index, x=x)

#     def message(self, x_j):
#         if self.enable_lstm_in_gnn:
#             x_j, _ = self.message_func(x_j.unsqueeze(1))
#             return x_j.squeeze(1)
#         else:
#             return self.message_func(x_j)


class SeqCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SeqCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.seq_cell = nn.LSTMCell(input_size, hidden_size)
        self.seq_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, x, h_t):
        # h_t, c_t = self.seq_cell(x, (h_t, c_t))
        h_t = self.seq_cell(x, h_t)
        return h_t


class FlowSimLstm(LightningModule):
    def __init__(
        self,
        n_layer=2,
        gcn_n_layer=2,
        gcn_hidden_size=32,
        loss_fn_type="l1",
        learning_rate=1e-3,
        batch_size=400,
        hidden_size=32,
        dropout=0.2,
        enable_dist=False,
        enable_val=True,
        output_size=1,
        input_size=2,
        current_period_len_idx=None,
        enable_bidirectional=False,
        enable_positional_encoding=False,
        enable_gnn=False,
        enable_lstm=False,
        enable_lstm_in_gnn=False,
        enable_link_state=False,
        enable_flowsim_diff=False,
        enable_remainsize=False,
        enable_log_norm=True,
        enable_path=False,
        enable_topo=False,
        loss_average="perflow",  # perflow, perperiod
        save_dir=None,
    ):
        super(FlowSimLstm, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = self._get_loss_fn(loss_fn_type)
        self.enable_gnn = enable_gnn
        self.enable_lstm = enable_lstm
        self.current_period_len_idx = current_period_len_idx
        self.hidden_size = hidden_size
        self.enable_link_state = enable_link_state
        self.enable_flowsim_diff = enable_flowsim_diff
        self.enable_remainsize = enable_remainsize
        self.enable_log_norm = enable_log_norm
        if enable_path:
            self.n_links = 12
        elif enable_topo:
            self.n_links = 96
        else:
            self.n_links = 1
        # GCN layers
        if enable_lstm and enable_gnn:
            logging.info(
                f"GNN and LSTM enabled, enable_lstm_in_gnn={enable_lstm_in_gnn}, enable_link_state={enable_link_state}, enable_flowsim_diff={enable_flowsim_diff}, enable_remainsize={enable_remainsize}"
            )
            # link_state_size = self.hidden_size if enable_link_state else 1
            # link_state_size = self.hidden_size
            # self.gcn_layers = nn.ModuleList(
            #     [
            #         HeterGNNLayer(
            #             c_in_type_a=hidden_size,
            #             c_in_type_b=link_state_size if i == 0 else hidden_size,
            #             c_out=hidden_size,
            #             dropout=dropout,
            #             enable_lstm_in_gnn=enable_lstm_in_gnn,
            #         )
            #         for i in range(gcn_n_layer)
            #     ]
            # )
            self.gcn_layers = nn.ModuleList(
                [
                    HomoGNNLayer(
                        c_in=hidden_size if i == 0 else gcn_hidden_size,
                        c_out=hidden_size if i == gcn_n_layer - 1 else gcn_hidden_size,
                        dropout=dropout,
                    )
                    for i in range(gcn_n_layer)
                ]
            )
            # lstmcell_rate_extra = 0
            lstmcell_rate_extra = 13
            self.lstmcell_rate = SeqCell(
                input_size=hidden_size + lstmcell_rate_extra, hidden_size=hidden_size
            )
            self.lstmcell_time = SeqCell(input_size=1, hidden_size=hidden_size)

            if self.enable_link_state:
                self.lstmcell_rate_link = SeqCell(
                    input_size=hidden_size, hidden_size=hidden_size
                )
                # self.lstmcell_time_link = SeqCell(input_size=1, hidden_size=hidden_size)
            dim_flowsim = 16 if self.enable_flowsim_diff else 0
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size + dim_flowsim, hidden_size // 2),  # First layer
                nn.ReLU(),  # Non-linearity
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size // 2, output_size),  # Second layer
            )
            if self.enable_remainsize:
                self.remain_size_layer = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 8),  # First layer
                    nn.ReLU(),  # Non-linearity
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_size // 8, 1),  # Second layer
                )
        elif enable_gnn:
            logging.info(f"GNN enabled")
            self.gcn_layers = nn.ModuleList(
                [
                    HomoGNNLayer(
                        input_size if i == 0 else gcn_hidden_size,
                        (gcn_hidden_size if i != gcn_n_layer - 1 else output_size),
                        dropout=dropout,
                        enable_lstm_in_gnn=enable_lstm_in_gnn,
                    )
                    for i in range(gcn_n_layer)
                ]
            )
        elif enable_lstm:
            logging.info(f"LSTM enabled")
            self.model_lstm = LSTMModel(
                input_size,
                hidden_size,
                output_size,
                n_layer,
                dropout=dropout,
                enable_bidirectional=enable_bidirectional,
                enable_positional_encoding=enable_positional_encoding,
            )
        else:
            assert False, "Either GNN or LSTM must be enabled"
        self.enable_dist = enable_dist
        self.enable_val = enable_val
        self.save_dir = save_dir
        self.loss_average = loss_average
        logging.info(
            f"Call FlowSimLstm. model: {n_layer}, input_size: {input_size}, loss_fn: {loss_fn_type}, learning_rate: {learning_rate}, batch_size: {batch_size}, hidden_size: {hidden_size}, gcn_hidden_size: {gcn_hidden_size}, enable_bidirectional: {enable_bidirectional}, enable_positional_encoding: {enable_positional_encoding}, dropout: {dropout}, loss_average: {loss_average}"
        )
        self.rtt = 0

    def _get_loss_fn(self, loss_fn_type):
        if loss_fn_type == "l1":
            # return nn.L1Loss()
            return WeightedL1Loss()
        elif loss_fn_type == "mse":
            return WeightedMSELoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_fn_type}")

    def forward(
        self,
        x,
        batch_index,
        remainsize_matrix,
        flow_active_matrix,  # (n_flows,2)
        time_delta_matrix,  # (batch,n_events)
        edges_a_to_b,  # (2, n_edges)
    ):
        loss_size = None
        if self.enable_gnn and self.enable_lstm:
            batch_size, n_events, _ = time_delta_matrix.size()
            n_flows = x.shape[0]
            batch_h_state = torch.zeros((n_flows, self.hidden_size), device=x.device)

            batch_h_state[:, 0] = 1.0
            batch_h_state[:, 2] = x[:, 0]
            batch_h_state[:, 3] = x[:, 2]
            # batch_h_state[:, 3 : 1 + x.shape[1]] = x[:, 2:]

            batch_h_state_link = torch.zeros(
                (batch_size * self.n_links, self.hidden_size), device=x.device
            )
            batch_h_state_link[:, 1] = 1.0
            batch_h_state_link[:, 2] = 1.0

            if self.enable_remainsize:
                loss_size = torch.zeros((n_flows, 1), device=x.device)
                loss_size_num = torch.ones_like(loss_size)

            flow_start = flow_active_matrix[:, 0].unsqueeze(1)  # (n_flows, 1)
            flow_end = flow_active_matrix[:, 1].unsqueeze(1)  # (n_flows, 1)
            event_indices = torch.arange(n_events, device=x.device).unsqueeze(
                0
            )  # (1, n_events)

            # Compute activity mask for all flows and events at once
            flow_activity_mask = (flow_start <= event_indices) & (
                flow_end > event_indices
            )  # (n_flows, n_events)

            time_deltas_full = time_delta_matrix[batch_index, :]  # (n_flows, n_events)

            for j in range(n_events):
                active_flow_mask = flow_activity_mask[:, j]  # (n_flows,)
                if active_flow_mask.any():
                    active_flow_idx = torch.where(active_flow_mask)[0]
                    time_deltas = time_deltas_full[active_flow_idx, j]
                    if (time_deltas > self.rtt).all():
                        batch_h_state[active_flow_idx, :] = self.lstmcell_time(
                            time_deltas, batch_h_state[active_flow_idx, :]
                        )

                    if self.enable_remainsize and len(remainsize_matrix[j]) == len(
                        active_flow_idx
                    ):
                        remain_size_est = self.remain_size_layer(
                            batch_h_state[active_flow_idx, :]
                        )[:, 0]

                        remain_size_gt = remainsize_matrix[j]

                        loss_size[active_flow_idx, 0] += torch.abs(
                            remain_size_est - remain_size_gt
                        )
                        # loss_size[active_flow_idx, 0] += torch.abs(
                        #     1 - remain_size_est / remain_size_gt
                        # )
                        loss_size_num[active_flow_idx, 0] += 1

                    edge_mask = active_flow_mask[edges_a_to_b[0]]
                    edge_index_a_to_b = edges_a_to_b[:, edge_mask]

                    n_flows_active = active_flow_idx.size(0)
                    new_flow_indices = torch.searchsorted(
                        active_flow_idx, edge_index_a_to_b[0]
                    )
                    active_link_idx, new_link_indices = torch.unique(
                        edge_index_a_to_b[1], return_inverse=True, sorted=False
                    )

                    new_link_indices += n_flows_active
                    edge_index_a_to_b = torch.stack(
                        [new_flow_indices, new_link_indices], dim=0
                    )

                    x_combined = torch.cat(
                        [
                            batch_h_state[active_flow_idx],
                            batch_h_state_link[active_link_idx],
                        ],
                        dim=0,
                    )

                    edge_index_b_to_a = torch.stack(
                        [edge_index_a_to_b[1], edge_index_a_to_b[0]], dim=0
                    )
                    edge_index = torch.cat(
                        [edge_index_a_to_b, edge_index_b_to_a], dim=1
                    )

                    for gcn in self.gcn_layers:
                        x_combined = gcn(x_combined, edge_index)

                    z_t_tmp = x_combined[:n_flows_active]
                    z_t_tmp_link = x_combined[n_flows_active:]

                    z_t_tmp = torch.cat([z_t_tmp, x[active_flow_idx, 3:]], dim=1)
                    batch_h_state[active_flow_idx, :] = self.lstmcell_rate(
                        z_t_tmp, batch_h_state[active_flow_idx, :]
                    )
                    if self.enable_link_state:

                        batch_h_state_link[active_link_idx, :] = (
                            self.lstmcell_rate_link(
                                z_t_tmp_link,
                                batch_h_state_link[active_link_idx, :],
                            )
                        )

            if self.enable_flowsim_diff:
                input_tmp = torch.cat([x, batch_h_state], dim=1)
                res = self.output_layer(input_tmp)
                # res = self.output_layer(batch_h_state) + x[:, 1:2]
            else:
                res = self.output_layer(batch_h_state) + 1.0
            if self.enable_remainsize:
                loss_size = torch.div(loss_size, loss_size_num)

        elif self.enable_lstm:
            res, _ = self.model_lstm(x, lengths)
            # res, _ = self.model_lstm(x[:, :, [0, 1]], lengths)
        else:
            assert False, "Either GNN or LSTM must be enabled"
        return res, loss_size

    def step(self, batch, batch_idx, tag=None):
        (
            input,
            output,
            batch_index,
            spec,
            remainsize_matrix,
            flow_active_matrix,
            time_delta_matrix,
            edges_a_to_b_matrix,
        ) = batch

        estimated, loss_size = self(
            input,
            batch_index,
            remainsize_matrix,
            flow_active_matrix,
            time_delta_matrix,
            edges_a_to_b_matrix,
        )

        # Generate a mask based on lengths

        est = torch.div(estimated, output).squeeze()
        gt = torch.ones_like(est)

        # Calculate the loss
        loss = self.loss_fn(est, gt, batch_index)
        self._log_loss(loss, tag)

        if self.enable_remainsize:
            # loss_size_mean = loss_size[loss_size > 0]
            if loss_size.size(0) == 0:
                loss_size_mean = 0
            else:
                loss_size_mean = 0
                n_batch = batch_index.max() + 1
                for i in range(n_batch):
                    idx = batch_index == i
                    loss_size_mean += loss_size[idx].nanmean()
                loss_size_mean /= n_batch
            self._log_loss(loss_size_mean, f"{tag}_size")
            loss = loss + 0.1 * loss_size_mean
        self._save_test_results(tag, spec, estimated, output)

        return loss

    def _log_loss(self, loss, tag):
        loss_tag = f"{tag}_loss"
        if self.enable_dist:
            loss_tag += "_sync"
        self.log(
            loss_tag,
            loss,
            sync_dist=self.enable_dist,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def _save_test_results(self, tag, spec, estimated, output):
        if tag == "test":
            test_dir = f"{self.save_dir}/{spec[0]}"
            os.makedirs(test_dir, exist_ok=True)
            np.savez(
                f"{test_dir}/res.npz",
                est=estimated.cpu().numpy(),
                output=output.cpu().numpy(),
            )

    def _log_gradient_norms(self, tag):
        # Compute and log gradient norms for GNN layers
        if self.enable_gnn:
            gnn_norms = []
            for layer in self.gcn_layers:
                for param in layer.parameters():
                    if param.grad is not None:
                        gnn_norms.append(param.grad.norm().item())
            if gnn_norms:
                avg_gnn_grad = sum(gnn_norms) / len(gnn_norms)
                self.log(
                    f"{tag}_gnn_grad_norm",
                    avg_gnn_grad,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    batch_size=self.batch_size,
                    sync_dist=self.enable_dist,
                )

        # Compute and log gradient norms for LSTM layers
        if self.enable_lstm:
            lstm_norms = []
            for param in self.lstmcell_rate.parameters():
                if param.grad is not None:
                    lstm_norms.append(param.grad.norm().item())
            for param in self.lstmcell_time.parameters():
                if param.grad is not None:
                    lstm_norms.append(param.grad.norm().item())
            if lstm_norms:
                avg_lstm_grad = sum(lstm_norms) / len(lstm_norms)
                self.log(
                    f"{tag}_lstm_grad_norm",
                    avg_lstm_grad,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    batch_size=self.batch_size,
                    sync_dist=self.enable_dist,
                )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, tag="train")
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model_lstm.parameters(), max_norm=1.0)
        if self.enable_log_norm:
            self._log_gradient_norms("train")

        return loss

    def validation_step(self, batch, batch_idx):
        if self.enable_val:
            return self.step(batch, batch_idx, tag="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="test")

    def configure_optimizers(self):
        if self.enable_lstm and self.enable_gnn:
            parameters = (
                list(self.lstmcell_rate.parameters())
                + list(self.lstmcell_time.parameters())
                + list(self.output_layer.parameters())
            )
            for gcn_layer in self.gcn_layers:
                parameters += list(gcn_layer.parameters())
            if self.enable_link_state:
                parameters += list(self.lstmcell_rate_link.parameters())
                # parameters += list(self.lstmcell_time_link.parameters())
            if self.enable_remainsize:
                parameters += list(self.remain_size_layer.parameters())
        else:
            parameters = []
            if self.enable_lstm:
                parameters = list(self.model_lstm.parameters())
            if self.enable_gnn:
                for gcn_layer in self.gcn_layers:
                    parameters += list(gcn_layer.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": (
                    "val_loss_sync" if self.enable_dist else "val_loss"
                ),  # Adjust according to the relevant metric
            },
        }
