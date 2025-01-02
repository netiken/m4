import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import numpy as np
import logging
import os
from torch_geometric.nn import SAGEConv


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
        out_combined = self.homogeneous_layer(x, edge_index)

        return out_combined


class HomoNetGNN(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.2):
        super(HomoNetGNN, self).__init__()
        self.conv = SAGEConv(c_in, c_out, aggr="sum", project=True)
        self.norm = torch.nn.LayerNorm(c_out)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.norm(out)  # Apply normalization
        return out


class SeqCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SeqCell, self).__init__()
        self.seq_cell = nn.GRUCell(input_size, hidden_size)
        # self.norm_layer = nn.LayerNorm(hidden_size)  # Normalize the hidden state

    def forward(self, x, h_t):
        h_t = self.seq_cell(x, h_t)
        # h_t = self.norm_layer(h_t)  # Apply normalization
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
        enable_queuelen=False,
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
        self.enable_queuelen = enable_queuelen
        self.enable_log_norm = enable_log_norm
        self.loss_efficiency_size = 0.1
        self.loss_efficiency_queue = 0.1
        if enable_path:
            self.n_links = 12
        elif enable_topo:
            self.n_links = 96
        else:
            self.n_links = 1
        if enable_lstm and enable_gnn:
            logging.info(
                f"GNN and LSTM enabled, enable_lstm_in_gnn={enable_lstm_in_gnn}, enable_link_state={enable_link_state}, enable_flowsim_diff={enable_flowsim_diff}, enable_remainsize={enable_remainsize}, enable_queuelen={enable_queuelen}"
            )

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
            lstmcell_rate_extra = 13
            self.lstmcell_rate = SeqCell(
                input_size=hidden_size + lstmcell_rate_extra, hidden_size=hidden_size
            )
            self.lstmcell_time = SeqCell(input_size=1, hidden_size=hidden_size)

            if self.enable_link_state:
                self.lstmcell_rate_link = SeqCell(
                    input_size=hidden_size, hidden_size=hidden_size
                )
                self.lstmcell_time_link = SeqCell(input_size=1, hidden_size=hidden_size)
            dim_flowsim = 16 if self.enable_flowsim_diff else 14
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size + dim_flowsim, hidden_size // 2),  # First layer
                nn.ReLU(),  # Non-linearity
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size // 2, output_size),  # Second layer
            )
            model_scaling_factor = 8
            if self.enable_remainsize:
                self.remain_size_layer = nn.Sequential(
                    nn.Linear(
                        hidden_size, hidden_size // model_scaling_factor
                    ),  # First layer
                    nn.ReLU(),  # Non-linearity
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_size // model_scaling_factor, 1),  # Second layer
                )
            if self.enable_queuelen:
                self.queue_len_layer = nn.Sequential(
                    nn.Linear(
                        hidden_size, hidden_size // model_scaling_factor
                    ),  # First layer
                    nn.ReLU(),  # Non-linearity
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_size // model_scaling_factor, 1),  # Second layer
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
            return WeightedL1Loss()
        elif loss_fn_type == "mse":
            return WeightedMSELoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_fn_type}")

    def forward(
        self,
        x,
        batch_index,
        batch_index_link,
        remainsize_matrix,
        queuelen_matrix,
        queuelen_link_matrix,
        flow_active_matrix,  # (n_flows,2)
        time_delta_matrix,  # (batch,n_events)
        edges_a_to_b,  # (2, n_edges)
        enable_test=False,
    ):
        loss_size = None
        loss_queue = None
        res_size = None
        res_queue = None
        if self.enable_gnn and self.enable_lstm:
            batch_size, n_events, _ = time_delta_matrix.size()
            n_flows = x.shape[0]
            batch_h_state = torch.zeros((n_flows, self.hidden_size), device=x.device)

            batch_h_state[:, 0] = 1.0
            batch_h_state[:, 2] = x[:, 0]
            batch_h_state[:, 3] = x[:, 2]

            batch_h_state_link = torch.zeros(
                (batch_size * self.n_links, self.hidden_size), device=x.device
            )
            batch_h_state_link[:, 1] = 1.0
            batch_h_state_link[:, 2] = 1.0

            if self.enable_remainsize:
                loss_size = torch.zeros((n_flows, 1), device=x.device)
                loss_size_num = torch.ones_like(loss_size)
                if enable_test:
                    res_size_est = []
                    res_size_gt = []
            if self.enable_queuelen:
                loss_queue = torch.zeros(
                    (batch_size * self.n_links, 1), device=x.device
                )
                loss_queue_num = torch.ones_like(loss_queue)
                if enable_test:
                    res_queue_est = []
                    res_queue_gt = []

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
            time_deltas_full_link = time_delta_matrix[batch_index_link, :]
            for j in range(n_events):
                active_flow_mask = flow_activity_mask[:, j]  # (n_flows,)
                if active_flow_mask.any():
                    active_flow_idx = torch.where(active_flow_mask)[0]

                    edge_mask = active_flow_mask[edges_a_to_b[0]]
                    edge_index_a_to_b = edges_a_to_b[:, edge_mask]
                    active_link_idx, new_link_indices = torch.unique(
                        edge_index_a_to_b[1], return_inverse=True, sorted=False
                    )

                    time_deltas = time_deltas_full[active_flow_idx, j]

                    if (time_deltas > self.rtt).all():
                        batch_h_state[active_flow_idx, :] = self.lstmcell_time(
                            time_deltas, batch_h_state[active_flow_idx, :]
                        )
                        if self.enable_link_state:
                            time_deltas_link = time_deltas_full_link[active_link_idx, j]
                            batch_h_state_link[active_link_idx, :] = (
                                self.lstmcell_time_link(
                                    time_deltas_link,
                                    batch_h_state_link[active_link_idx, :],
                                )
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
                        loss_size_num[active_flow_idx, 0] += 1
                        if enable_test:
                            res_size_est.extend(
                                remain_size_est.cpu().detach().numpy().tolist()
                            )
                            res_size_gt.extend(
                                remain_size_gt.cpu().detach().numpy().tolist()
                            )
                    if self.enable_queuelen:
                        queue_link_idx = queuelen_link_matrix[j]
                        if len(queue_link_idx) > 0:
                            queue_len_est = self.queue_len_layer(
                                batch_h_state_link[queue_link_idx, :]
                            )[:, 0]

                            queue_len_gt = queuelen_matrix[j]
                            if (
                                len(queue_len_gt)
                                == len(queue_len_est)
                                == len(queue_link_idx)
                            ):
                                loss_queue[queue_link_idx, 0] += torch.abs(
                                    queue_len_est - queue_len_gt
                                )
                                loss_queue_num[queue_link_idx, 0] += 1
                                if enable_test:
                                    res_queue_est.extend(
                                        queue_len_est.cpu().detach().numpy().tolist()
                                    )
                                    res_queue_gt.extend(
                                        queue_len_gt.cpu().detach().numpy().tolist()
                                    )

                    n_flows_active = active_flow_idx.size(0)
                    new_flow_indices = torch.searchsorted(
                        active_flow_idx, edge_index_a_to_b[0]
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
            else:
                input_tmp = torch.cat([x[:, 2:], batch_h_state], dim=1)
                res = self.output_layer(input_tmp)
                # res = self.output_layer(batch_h_state) + 1.0
            if self.enable_remainsize:
                loss_size = torch.div(loss_size, loss_size_num)
            if self.enable_queuelen:
                loss_queue = torch.div(loss_queue, loss_queue_num)

        elif self.enable_lstm:
            res, _ = self.model_lstm(x, lengths)
            # res, _ = self.model_lstm(x[:, :, [0, 1]], lengths)
        else:
            assert False, "Either GNN or LSTM must be enabled"
        if self.enable_remainsize and enable_test:
            res_size = np.array([res_size_est, res_size_gt])
        if self.enable_queuelen and enable_test:
            res_queue = np.array([res_queue_est, res_queue_gt])
        return res, loss_size, loss_queue, res_size, res_queue

    def step(self, batch, batch_idx, tag=None):
        (
            input,
            output,
            batch_index,
            batch_index_link,
            spec,
            remainsize_matrix,
            queuelen_matrix,
            queuelen_link_matrix,
            flow_active_matrix,
            time_delta_matrix,
            edges_a_to_b_matrix,
        ) = batch
        enable_test = tag == "test"

        estimated, loss_size, loss_queue, res_size, res_queue = self(
            input,
            batch_index,
            batch_index_link,
            remainsize_matrix,
            queuelen_matrix,
            queuelen_link_matrix,
            flow_active_matrix,
            time_delta_matrix,
            edges_a_to_b_matrix,
            enable_test=enable_test,
        )

        est = torch.div(estimated, output).squeeze()
        gt = torch.ones_like(est)

        # Calculate the loss
        loss = self.loss_fn(est, gt, batch_index)
        self._log_loss(loss, tag)

        if self.enable_remainsize:
            loss_size_mean = 0
            if loss_size.size(0) != 0:
                n_batch = batch_index.max() + 1
                for i in range(n_batch):
                    idx = batch_index == i
                    loss_size_mean += loss_size[idx].nanmean()
                loss_size_mean /= n_batch
            self._log_loss(loss_size_mean, f"{tag}_size")
            loss = loss + loss_size_mean * self.loss_efficiency_size
        if self.enable_queuelen:
            loss_queue_mean = 0
            if loss_queue.size(0) != 0:
                loss_queue_mean = loss_queue.nanmean()
            self._log_loss(loss_queue_mean, f"{tag}_queue")
            loss = loss + loss_queue_mean * self.loss_efficiency_queue
        if enable_test:
            self._save_test_results(spec, estimated, output, res_size, res_queue)

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

    def _save_test_results(self, spec, estimated, output, res_size, res_queue):
        test_dir = f"{self.save_dir}/{spec[0]}"
        os.makedirs(test_dir, exist_ok=True)
        np.savez(
            f"{test_dir}/res.npz",
            est=estimated.cpu().numpy(),
            output=output.cpu().numpy(),
            res_size=np.array(res_size),
            res_queue=np.array(res_queue),
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
            if self.enable_link_state:
                lstm_norms_link = []
                for param in self.lstmcell_rate_link.parameters():
                    if param.grad is not None:
                        lstm_norms_link.append(param.grad.norm().item())
                for param in self.lstmcell_time_link.parameters():
                    if param.grad is not None:
                        lstm_norms_link.append(param.grad.norm().item())
                if lstm_norms_link:
                    avg_lstm_grad_link = sum(lstm_norms_link) / len(lstm_norms_link)
                    self.log(
                        f"{tag}_lstm_grad_norm_link",
                        avg_lstm_grad_link,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                        prog_bar=True,
                        batch_size=self.batch_size,
                        sync_dist=self.enable_dist,
                    )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, tag="train")
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
                parameters += list(self.lstmcell_time_link.parameters())
            if self.enable_remainsize:
                parameters += list(self.remain_size_layer.parameters())
            if self.enable_queuelen:
                parameters += list(self.queue_len_layer.parameters())
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
