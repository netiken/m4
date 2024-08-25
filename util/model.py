import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from .func import serialize_fp32
import numpy as np
import logging
import struct
import os

from .model_llama import Transformer, ModelArgs

# from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphConv
import torch.nn.functional as F

# from .consts import N_BACKGROUND


class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()
        return

    def forward(self, x):
        return torch.exp(x)


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, prediction, target, loss_average="perflow"):
        if loss_average == "perflow":
            elementwise_loss = torch.abs(prediction - target).sum()
            weighted_loss = elementwise_loss / target.sum()
        elif loss_average == "perperiod":
            if prediction.dim() > 1:
                sequencewise_loss = torch.abs(prediction - target).sum(
                    dim=1
                ) / target.sum(dim=1).to(prediction.device)
            else:
                sequencewise_loss = torch.abs(
                    prediction - target
                ).sum() / target.sum().to(prediction.device)
            weighted_loss = torch.mean(sequencewise_loss)
        else:
            raise ValueError(f"Unsupported loss average type: {loss_average}")
        return weighted_loss


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, loss_average="perflow"):
        if loss_average == "perflow":
            elementwise_loss = ((prediction - target) ** 2).sum()
            weighted_loss = elementwise_loss / target.sum()
        elif loss_average == "perperiod":
            if prediction.dim() > 1:
                sequencewise_loss = ((prediction - target) ** 2).sum(
                    dim=1
                ) / target.sum(dim=1).to(prediction.device)
            else:
                sequencewise_loss = (
                    (prediction - target) ** 2
                ).sum() / target.sum().to(prediction.device)
            weighted_loss = torch.mean(sequencewise_loss)
        else:
            raise ValueError(f"Unsupported loss average type: {loss_average}")
        return weighted_loss


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


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, lstm_out):
        seq_len = lstm_out.size(1)

        # Repeat hidden state for each time step
        hidden = lstm_out[:, -1, :].unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate hidden state with LSTM output
        energy = torch.tanh(self.attn(torch.cat((hidden, lstm_out), dim=2)))

        # Compute attention scores
        attn_weights = energy.matmul(self.v)  # [batch_size, seq_len]
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(
            -1
        )  # [batch_size, seq_len, 1]

        # Compute context vector
        context = attn_weights * lstm_out  # [batch_size, seq_len, hidden_size]

        return context, attn_weights


# class GRUConv(MessagePassing):
#     def __init__(self, c_in, c_out):
#         super(GRUConv, self).__init__(aggr="add")
#         self.gru = nn.GRU(input_size=c_in, hidden_size=c_out, batch_first=True)
#         self.fc = nn.Linear(c_out, c_out)

#     def forward(self, x, edge_index):
#         return self.propagate(edge_index, x=x)

#     def message(self, x_j):
#         return x_j

#     def aggregate(self, inputs, index):
#         # Group inputs by target node index
#         unique_nodes, inverse_indices = torch.unique(
#             index, sorted=True, return_inverse=True
#         )
#         num_nodes = unique_nodes.size(0)

#         # Determine number of neighbors for each node
#         counts = scatter(torch.ones_like(index), index, dim=0, reduce="sum").long()

#         # Filter out nodes with no neighbors
#         valid_node_mask = counts > 0
#         if not valid_node_mask.any():
#             # Handle the case where all nodes have no neighbors
#             return torch.zeros((num_nodes, inputs.size(-1)), device=inputs.device)

#         valid_counts = counts[valid_node_mask]
#         valid_nodes = unique_nodes[valid_node_mask]

#         max_neighbors = valid_counts.max().item()
#         padded_seq = torch.zeros(
#             (valid_nodes.size(0), max_neighbors, inputs.size(-1)), device=inputs.device
#         )

#         current_pos = torch.zeros(
#             valid_nodes.size(0), dtype=torch.long, device=inputs.device
#         )
#         for i in range(inputs.size(0)):
#             node_idx = inverse_indices[i]
#             if valid_node_mask[node_idx]:
#                 position = current_pos[valid_node_mask[node_idx]]
#                 padded_seq[valid_node_mask[node_idx], position, :] = inputs[i]
#                 current_pos[valid_node_mask[node_idx]] += 1

#         # Pack the sequences for GRU
#         packed_seq = nn.utils.rnn.pack_padded_sequence(
#             padded_seq, valid_counts.cpu(), batch_first=True, enforce_sorted=False
#         )

#         # Apply GRU
#         packed_out, _ = self.gru(packed_seq)
#         out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

#         # Use the last hidden state for each valid node
#         aggr_out = torch.zeros((num_nodes, out.size(-1)), device=inputs.device)
#         aggr_out[valid_node_mask] = out[
#             torch.arange(valid_nodes.size(0), device=inputs.device), valid_counts - 1
#         ]

#         return aggr_out

#     def update(self, aggr_out):
#         return self.fc(aggr_out)


# GNN model
class GNNLayer(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.2, enable_lstm=False):
        super(GNNLayer, self).__init__()
        # self.conv = GraphConv(c_in, c_out, aggr="mean")
        self.conv = SAGEConv(c_in, c_out, aggr="lstm")  # using mean aggregation
        # self.conv = GCNConv(c_in, c_out)
        # self.conv = GRUConv(c_in, c_out)  # using GRU-based aggregation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_feats, edge_index):
        # node_feats = self.conv(node_feats, edge_index[:2], edge_weight=edge_index[2])
        node_feats = self.conv(node_feats, edge_index[:2])
        node_feats = F.relu(node_feats)
        node_feats = self.dropout(node_feats)
        return node_feats


# class GNNLayer(nn.Module):
#     def __init__(
#         self, c_in, c_out, heads=4, concat=True, dropout=0.2, enable_lstm=False
#     ):
#         super(GNNLayer, self).__init__()
#         # Initialize the GATConv layer
#         self.enable_lstm = enable_lstm
#         if enable_lstm:
#             self.gat_conv = GATv2Conv(
#                 c_in,
#                 c_out // heads if concat else c_out,
#                 heads=heads,
#                 concat=concat,
#                 dropout=dropout,
#             )
#         else:
#             self.gat_conv = GATv2Conv(
#                 c_in,
#                 c_in // heads if concat else c_in,
#                 heads=heads,
#                 concat=concat,
#                 dropout=dropout,
#             )
#             self.fc = nn.Linear(c_in, c_out)

#     def forward(self, node_feats, edge_index):
#         # Apply GAT convolution
#         node_feats = self.gat_conv(node_feats, edge_index[:2])
#         node_feats = F.relu(node_feats)
#         if not self.enable_lstm:
#             node_feats = self.fc(node_feats)
#         return node_feats


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        enable_bidirectional=False,
        enable_positional_encoding=False,
        enable_attention=False,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = enable_bidirectional
        self.num_directions = 2 if enable_bidirectional else 1
        self.enable_attention = enable_attention
        # Adjust the fully connected layer based on bidirectionality
        self.lstm = nn.LSTM(
            input_size * 2 if enable_positional_encoding else input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=enable_bidirectional,
            dropout=dropout,
        )
        if enable_attention:
            logging.info("Attention enabled")
            self.attention = Attention(hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        # self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

    def init_hidden(self, batch_size, device):
        h_t = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(device)
        c_t = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ).to(device)
        return h_t, c_t

    def forward(self, x, lengths, lengths_per_path, n_paths_per_batch):
        batch_size = x.size(0)
        n_feats = x.size(2)
        n_longest_seq = np.max(lengths_per_path)
        n_batches_total = np.sum(n_paths_per_batch)

        x_adj = torch.zeros((n_batches_total, n_longest_seq, n_feats), device=x.device)

        n_batches = 0
        for i in range(batch_size):
            n_seq_accumulated = 0
            for j in range(n_paths_per_batch[i]):
                seq_len = lengths_per_path[i][j]
                x_adj[n_batches, : lengths_per_path[i][j]] = x[
                    i, n_seq_accumulated : n_seq_accumulated + seq_len
                ]
                n_seq_accumulated += seq_len
                n_batches += 1

        lengths_per_path_new = np.concatenate(
            [lengths_per_path[i, : n_paths_per_batch[i]] for i in range(batch_size)]
        )

        h_t, c_t = self.init_hidden(n_batches_total, x_adj.device)
        # Pack the padded batch of sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_adj, lengths_per_path_new, batch_first=True, enforce_sorted=False
        )
        packed_output, (h_t, c_t) = self.lstm(packed_input, (h_t, c_t))

        # Unpack the output sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply Layer Normalization
        # lstm_out = self.layer_norm(lstm_out)

        # Apply attention
        if self.enable_attention:
            lstm_out, _ = self.attention(lstm_out)
        # Apply the fully connected layer to each time step
        out = self.fc(lstm_out)

        # Initialize the final output tensor with the same shape as x
        out_final = torch.zeros((x.size(0), x.size(1), out.size(2)), device=out.device)
        n_batches = 0
        for i in range(batch_size):
            n_seq_accumulated = 0
            for j in range(n_paths_per_batch[i]):
                seq_len = lengths_per_path[i][j]
                out_final[i, n_seq_accumulated : n_seq_accumulated + seq_len] = out[
                    n_batches, :seq_len
                ]
                n_seq_accumulated += seq_len
                n_batches += 1

        return out_final, (h_t, c_t)


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
        enable_bidirectional=False,
        enable_positional_encoding=False,
        enable_gnn=False,
        enable_lstm=False,
        enable_path=False,
        loss_average="perflow",  # perflow, perperiod
        save_dir=None,
    ):
        super(FlowSimLstm, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = self._get_loss_fn(loss_fn_type)
        self.enable_gnn = enable_gnn
        self.enable_lstm = enable_lstm
        self.gcn_hidden_size = gcn_hidden_size
        self.enable_path = enable_path
        # input_size += N_BACKGROUND
        # GCN layers
        if enable_lstm and enable_gnn:
            logging.info(f"GNN and LSTM enabled")
            self.gcn_layers = nn.ModuleList(
                [
                    GNNLayer(
                        input_size if i == 0 else gcn_hidden_size,
                        gcn_hidden_size if i != gcn_n_layer - 1 else input_size,
                        dropout=dropout,
                        enable_lstm=enable_lstm,
                    )
                    for i in range(gcn_n_layer)
                ]
            )
            self.model_lstm = LSTMModel(
                input_size * 2,
                hidden_size,
                output_size,
                n_layer,
                dropout=dropout,
                enable_bidirectional=enable_bidirectional,
                enable_positional_encoding=enable_positional_encoding,
            )
        elif enable_gnn:
            logging.info(f"GNN enabled")
            self.gcn_layers = nn.ModuleList(
                [
                    GNNLayer(
                        input_size if i == 0 else gcn_hidden_size,
                        (gcn_hidden_size if i != gcn_n_layer - 1 else output_size),
                        dropout=dropout,
                        enable_lstm=enable_lstm,
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
            f"model: {n_layer}, input_size: {input_size}, loss_fn: {loss_fn_type}, learning_rate: {learning_rate}, batch_size: {batch_size}, hidden_size: {hidden_size}, gcn_hidden_size: {gcn_hidden_size}, enable_bidirectional: {enable_bidirectional}, enable_positional_encoding: {enable_positional_encoding}, dropout: {dropout}, loss_average: {loss_average}"
        )

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
        lengths,
        edge_index,
        edge_index_len,
        lengths_per_path,
        n_paths_per_batch,
    ):
        if self.enable_gnn and self.enable_lstm:
            batch_size = x.size(0)
            feature_dim = x.size(2)

            batch_gnn_output = torch.zeros(
                (batch_size, x.size(1), feature_dim), device=x.device
            )
            for i in range(batch_size):
                num_flow_nodes = lengths[i]
                edge_index_trimmed = edge_index[i, :, : edge_index_len[i]]

                x_gnn_input = x[i, :num_flow_nodes, :]
                for gcn in self.gcn_layers:
                    x_gnn_input = gcn(x_gnn_input, edge_index_trimmed)

                batch_gnn_output[i, :num_flow_nodes, :] = x_gnn_input[
                    :num_flow_nodes, :
                ]

            x = torch.cat((x, batch_gnn_output), dim=-1)
            res, _ = self.model_lstm(x, lengths, lengths_per_path, n_paths_per_batch)
        elif self.enable_gnn:
            batch_size = x.size(0)

            res = torch.zeros((batch_size, x.size(1), 1), device=x.device)
            for i in range(batch_size):
                num_flow_nodes = lengths[i]
                edge_index_trimmed = edge_index[i, :, : edge_index_len[i]]
                x_gnn_input = x[i, :num_flow_nodes, :]

                for gcn in self.gcn_layers:
                    x_gnn_input = gcn(x_gnn_input, edge_index_trimmed)

                res[i, :num_flow_nodes, :] = x_gnn_input[:num_flow_nodes, :]
        elif self.enable_lstm:
            res, _ = self.model_lstm(x, lengths, lengths_per_path, n_paths_per_batch)
        else:
            assert False, "Either GNN or LSTM must be enabled"
        return res

    def step(self, batch, batch_idx, tag=None):
        (
            input,
            output,
            lengths,
            spec,
            src_dst_pair_target_str,
            edge_index,
            edge_index_len,
            lengths_per_path,
            n_paths_per_batch,
        ) = batch

        estimated = self(
            input,
            lengths,
            edge_index,
            edge_index_len,
            lengths_per_path,
            n_paths_per_batch,
        )

        # Generate a mask based on lengths
        attention_mask = output.squeeze() >= 1.0

        est = torch.div(estimated, output).squeeze()
        gt = torch.ones_like(est)
        est = est.masked_fill(~attention_mask, 0.0)
        gt = gt.masked_fill(~attention_mask, 0.0)
        # Calculate the loss
        loss = self.loss_fn(est, gt, self.loss_average)

        self._log_loss(loss, tag)
        # if estimated.size(0) == 1:
        #     estimated[0, :, 0][~attention_mask] = input[0, :, 4][~attention_mask]
        # else:
        #     estimated[:, :, 0][~attention_mask] = input[:, :, 4][~attention_mask]
        self._save_test_results(tag, spec, src_dst_pair_target_str, estimated, output)

        return loss

    def _log_loss(self, loss, tag):
        loss_tag = f"{tag}_loss"
        if self.enable_dist:
            loss_tag += "_sync"
            self.log(
                loss_tag,
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
                loss_tag,
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )

    def _save_test_results(self, tag, spec, src_dst_pair_target_str, estimated, output):
        if tag == "test":
            test_dir = f"{self.save_dir}/{spec[0]}_{src_dst_pair_target_str[0]}"
            os.makedirs(test_dir, exist_ok=True)
            np.savez(
                f"{test_dir}/res.npz",
                est=estimated.cpu().numpy(),
                output=output.cpu().numpy(),
            )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, tag="train")
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model_lstm.parameters(), max_norm=1.0)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.enable_val:
            return self.step(batch, batch_idx, tag="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="test")

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.model_lstm.parameters(), lr=self.learning_rate
        # )
        # return optimizer
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
                "monitor": "val_loss_sync",  # Adjust according to the relevant metric
            },
        }
