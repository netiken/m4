class GNNLayer_bk(MessagePassing):
    def __init__(
        self, c_in_type_a, c_in_type_b, c_out, dropout=0.2, enable_lstm_in_gnn=False
    ):
        super(GNNLayer, self).__init__(
            aggr="add"
        )  # Aggregation method: "add", "mean", "max"
        self.c_in_type_a = c_in_type_a
        self.c_in_type_b = c_in_type_b
        self.c_out = c_out
        self.enable_lstm_in_gnn = enable_lstm_in_gnn

        if enable_lstm_in_gnn:
            self.lstm_type_a = nn.LSTM(
                c_in_type_a, c_out, batch_first=True, bidirectional=False
            )
            self.lstm_type_b = nn.LSTM(
                c_in_type_b, c_out, batch_first=True, bidirectional=False
            )
        else:
            # Define MLP with multiple layers, non-linearity, and dropout
            self.mlp_type_a = MLP(c_in_type_a, c_out, c_out, dropout)

            self.mlp_type_b = MLP(c_in_type_b, c_out, c_out, dropout)

        self.final_lin_type_a = nn.Linear(c_in_type_a + c_out, c_out)
        self.final_lin_type_b = nn.Linear(c_in_type_b + c_out, c_out)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_type_a, x_type_b, edge_index_a_to_b):
        edge_index_b_to_a = torch.stack(
            [edge_index_a_to_b[1], edge_index_a_to_b[0]], dim=0
        )
        # Separate features and perform message passing
        aggr_feats_type_b = self.propagate(
            edge_index_a_to_b,
            x=(x_type_a, x_type_b),
            size=(x_type_a.size(0), x_type_b.size(0)),
            node_type="type_a",
        )

        aggr_feats_type_a = self.propagate(
            edge_index_b_to_a,
            x=(x_type_b, x_type_a),
            size=(x_type_b.size(0), x_type_a.size(0)),
            node_type="type_b",
        )

        # Concatenate original and aggregated features
        x_type_a = torch.cat([x_type_a, aggr_feats_type_a], dim=-1)
        x_type_b = torch.cat([x_type_b, aggr_feats_type_b], dim=-1)

        # Apply linear transformation, activation, and dropout
        x_type_a = F.relu(self.final_lin_type_a(x_type_a))
        x_type_a = self.dropout(x_type_a)

        x_type_b = F.relu(self.final_lin_type_b(x_type_b))
        x_type_b = self.dropout(x_type_b)

        return x_type_a, x_type_b

    def message(self, x_j, node_type):
        # x_j contains the source node features for each edge
        if self.enable_lstm_in_gnn:
            if node_type == "type_a":
                x_j, _ = self.lstm_type_a(x_j.unsqueeze(0))
            else:
                x_j, _ = self.lstm_type_b(x_j.unsqueeze(0))
            return x_j.squeeze(0)
        else:
            if node_type == "type_a":
                return self.mlp_type_a(x_j)
            else:
                return self.mlp_type_b(x_j)

    # def aggregate(self, inputs, index, ptr=None, dim_size=None):
    #     # Use built-in aggregation or custom LSTM-based aggregation
    #     return super().aggregate(inputs, index, ptr, dim_size)

    # def update(self, aggr_out):
    #     # The update step is handled in the forward method
    #     return aggr_out

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
# class GNNLayer(nn.Module):
#     def __init__(self, c_in, c_out, dropout=0.2, enable_lstm=False):
#         super(GNNLayer, self).__init__()
#         # self.conv = GraphConv(c_in, c_out, aggr="mean")
#         self.conv = SAGEConv(c_in, c_out, aggr="lstm")  # using mean aggregation
#         # self.conv = GCNConv(c_in, c_out)
#         # self.conv = GRUConv(c_in, c_out)  # using GRU-based aggregation
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, node_feats, edge_index):
#         # node_feats = self.conv(node_feats, edge_index[:2], edge_weight=edge_index[2])
#         node_feats = self.conv(node_feats, edge_index[:2])
#         # node_feats = F.relu(node_feats)
#         node_feats = self.dropout(node_feats)
#         return node_feats


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


class FlowSimLstm_bk(LightningModule):
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
        self.enable_path = enable_path
        # GCN layers
        if enable_lstm and enable_gnn:
            logging.info(
                f"GNN and LSTM enabled, enable_lstm_in_gnn={enable_lstm_in_gnn}, enable_link_state={enable_link_state}, enable_flowsim_diff={enable_flowsim_diff}, enable_remainsize={enable_remainsize}"
            )
            link_state_size = self.hidden_size if enable_link_state else 1
            self.gcn_layers = nn.ModuleList(
                [
                    torch.compile(
                        GNNLayer(
                            c_in_type_a=hidden_size,
                            c_in_type_b=link_state_size if i == 0 else hidden_size,
                            c_out=hidden_size,
                            dropout=dropout,
                            enable_lstm_in_gnn=enable_lstm_in_gnn,
                        )
                    )
                    for i in range(gcn_n_layer)
                ]
            )
            self.lstmcell_rate = torch.compile(
                LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
            )
            self.lstmcell_time = torch.compile(
                LSTMCell(input_size=1, hidden_size=hidden_size)
            )
            if self.enable_link_state:
                self.lstmcell_rate_link = torch.compile(
                    LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                )
                self.lstmcell_time_link = torch.compile(
                    LSTMCell(input_size=1, hidden_size=hidden_size)
                )
            # flowsim_dim = 2 if self.enable_flowsim_diff else 0
            # flowsim_dim = 0
            self.output_layer = torch.compile(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),  # First layer
                    nn.ReLU(),  # Non-linearity
                    nn.Linear(hidden_size, output_size),  # Second layer
                )
            )
            if self.enable_remainsize:
                self.remain_size_layer = torch.compile(
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),  # First layer
                        nn.ReLU(),  # Non-linearity
                        nn.Linear(hidden_size, 1),  # Second layer
                    )
                )
        elif enable_gnn:
            logging.info(f"GNN enabled")
            self.gcn_layers = nn.ModuleList(
                [
                    GNNLayer(
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
        spec,
        event_info_tuples,
        lengths,
        active_flowid_tuples,
        active_linkid_tuple,
        active_edges_a_to_b_tuples,
        remainsize_tuples,
    ):
        loss_size = None
        if self.enable_gnn and self.enable_lstm:
            batch_size, seq_len, feature_dim = x.size()

            # Preallocate tensors
            batch_h_state = torch.zeros(
                (batch_size, seq_len, self.hidden_size), device=x.device
            )
            batch_c_state = torch.zeros_like(batch_h_state)
            batch_h_state_final = torch.zeros_like(batch_h_state)

            # Initialize first feature dimension
            # batch_h_state[:, :] = (
            #     x[:, :, 0].unsqueeze(2).expand(-1, -1, self.hidden_size)
            # )
            batch_h_state[:, :, 0] = x[:, :, 0]

            if self.enable_link_state:
                if not self.enable_path:
                    batch_h_state_link = torch.ones(
                        (batch_size, 1, self.hidden_size), device=x.device
                    )
                    batch_c_state_link = torch.ones_like(batch_h_state_link)
                else:
                    n_hosts = int(spec[0].split("_")[2][6:])
                    n_links = 3 * (n_hosts - 1)
                    batch_h_state_link = torch.ones(
                        (batch_size, n_links, self.hidden_size), device=x.device
                    )
                    batch_c_state_link = torch.ones_like(batch_h_state_link)
            if self.enable_remainsize:
                loss_size = torch.zeros((batch_size, seq_len), device=x.device)
                loss_size_num = torch.ones_like(loss_size)

            # streams = [torch.cuda.Stream() for _ in range(batch_size)]

            def process_batch_element(i):
                n_flows = lengths[i]
                if self.enable_remainsize:
                    remain_size_gt_list = list(remainsize_tuples[i])
                event_info_list = list(event_info_tuples[i])
                event_time_list = event_info_list[0]
                event_departure_list = event_info_list[1]
                active_flowids_list = list(active_flowid_tuples[i])
                active_linkid_list = list(active_linkid_tuple[i])
                active_edges_a_to_b_list = list(active_edges_a_to_b_tuples[i])

                time_last = 0
                for j in range(n_flows * 2):
                    if event_departure_list[j] >= 0:
                        flow_id_complete = event_departure_list[j]
                        batch_h_state_final[i, flow_id_complete, :] = batch_h_state[
                            i, flow_id_complete, :
                        ]
                    cur_time = event_time_list[j]
                    active_flows = active_flowids_list[j]

                    if active_flows and cur_time - time_last > self.rtt:
                        time_deltas = torch.full(
                            (len(active_flows),),
                            (cur_time - time_last) / 1000.0,
                            device=x.device,
                        ).unsqueeze(1)

                        h_t_tmp = batch_h_state[i, active_flows, :]
                        c_t_tmp = batch_c_state[i, active_flows, :]

                        h_t, c_t = self.lstmcell_time(
                            time_deltas,
                            h_t_tmp,
                            c_t_tmp,
                        )
                        batch_h_state[i, active_flows, :] = h_t
                        batch_c_state[i, active_flows, :] = c_t

                    # if j == idx_completion_event[idx_completion_cnt]:
                    #     idx_completion_cnt += 1
                    # else:
                    #     flow_id_active.add(idx_arrive_cnt)
                    #     idx_arrive_cnt += 1
                    time_last = cur_time

                    if active_flows:  # Process only if there are active flows
                        active_edges_a_to_b = active_edges_a_to_b_list[j]
                        active_links = active_linkid_list[j]

                        h_t_tmp = batch_h_state[i, active_flows, :]
                        c_t_tmp = batch_c_state[i, active_flows, :]
                        z_t_tmp = h_t_tmp

                        if self.enable_link_state:
                            h_t_tmp_link = batch_h_state_link[i, active_links, :]
                            c_t_tmp_link = batch_c_state_link[i, active_links, :]
                            z_t_tmp_link = h_t_tmp_link
                        else:
                            z_t_tmp_link = torch.ones(
                                (len(active_links), 1), device=x.device
                            )

                        for gcn in self.gcn_layers:
                            z_t_tmp, z_t_tmp_link = gcn(
                                z_t_tmp, z_t_tmp_link, active_edges_a_to_b
                            )

                        h_t, c_t = self.lstmcell_rate(
                            z_t_tmp,
                            h_t_tmp,
                            c_t_tmp,
                        )
                        batch_h_state[i, active_flows, :] = h_t
                        batch_c_state[i, active_flows, :] = c_t

                    # if active_flowids_list[j]:
                    #     active_flows = active_flowids_list[j]
                    #     active_links = active_linkid_list[j]
                    #     active_edges_a_to_b = active_edges_a_to_b_list[j]

                    #     h_t_tmp = batch_h_state[i, active_flows, :]
                    #     c_t_tmp = batch_c_state[i, active_flows, :]
                    #     z_t_tmp = h_t_tmp

                    #     if self.enable_link_state:
                    #         h_t_tmp_link = batch_h_state_link[i, active_links, :]
                    #         c_t_tmp_link = batch_c_state_link[i, active_links, :]
                    #         z_t_tmp_link = h_t_tmp_link
                    #     else:
                    #         z_t_tmp_link = torch.ones(
                    #             (len(active_links), 1), device=x.device
                    #         )

                    #     for gcn in self.gcn_layers:
                    #         z_t_tmp, z_t_tmp_link = gcn(
                    #             z_t_tmp, z_t_tmp_link, active_edges_a_to_b
                    #         )
                    #     if self.enable_link_state:
                    #         h_t_link, c_t_link = self.lstmcell_rate_link(
                    #             z_t_tmp_link,
                    #             h_t_tmp_link,
                    #             c_t_tmp_link,
                    #         )
                    #         batch_h_state_link[i, active_links, :] = h_t_link
                    #         batch_c_state_link[i, active_links, :] = c_t_link

                    #     h_t, c_t = self.lstmcell_rate(
                    #         z_t_tmp,
                    #         h_t_tmp,
                    #         c_t_tmp,
                    #     )
                    #     batch_h_state[i, active_flows, :] = h_t
                    #     batch_c_state[i, active_flows, :] = c_t

                    #     if event_departure_list[j] >= 0:
                    #         flowid_complete = event_departure_list[j]
                    #         batch_h_state_final[i, flowid_complete, :] = batch_h_state[
                    #             i, flowid_complete, :
                    #         ]

                    #     if cur_time - time_last > self.rtt:
                    #         time_deltas = torch.full(
                    #             (len(active_flows),),
                    #             (cur_time - time_last) / 1000.0,
                    #             device=x.device,
                    #         ).unsqueeze(1)

                    #         h_t_tmp = batch_h_state[i, active_flows, :]
                    #         c_t_tmp = batch_c_state[i, active_flows, :]

                    #         h_t, c_t = self.lstmcell_time(
                    #             time_deltas,
                    #             h_t_tmp,
                    #             c_t_tmp,
                    #         )
                    #         batch_h_state[i, active_flows, :] = h_t
                    #         batch_c_state[i, active_flows, :] = c_t

                    #         if self.enable_link_state:
                    #             time_delta_link = torch.full(
                    #                 (len(active_links),),
                    #                 (cur_time - time_last) / 1000.0,
                    #                 device=x.device,
                    #             ).unsqueeze(1)

                    #             h_t_tmp_link = batch_h_state_link[i, active_links, :]
                    #             c_t_tmp_link = batch_c_state_link[i, active_links, :]
                    #             h_t_link, c_t_link = self.lstmcell_time_link(
                    #                 time_delta_link,
                    #                 h_t_tmp_link,
                    #                 c_t_tmp_link,
                    #             )
                    #             batch_h_state_link[i, active_links, :] = h_t_link
                    #             batch_c_state_link[i, active_links, :] = c_t_link

                    #         if self.enable_remainsize and len(remain_size_gt_list[j]):
                    #             flow_id_update_size = batch_h_state[i, active_flows, :]
                    #             remain_size_est = self.remain_size_layer(
                    #                 flow_id_update_size
                    #             )[:, 0]
                    #             remain_size_gt = remain_size_gt_list[j]
                    #             # est_tmp = torch.div(
                    #             #     remain_size_est + 1.0, remain_size_gt + 1.0
                    #             # ).squeeze()
                    #             # gt_tmp = torch.ones_like(est_tmp)
                    #             # loss_size[i, j] = torch.nansum(
                    #             #     torch.abs(est_tmp - gt_tmp)
                    #             # )
                    #             loss_size[i, active_flows] += torch.abs(
                    #                 remain_size_est - remain_size_gt
                    #             )
                    #             loss_size_num[i, active_flows] += 1

                    # time_last = cur_time

            for i in range(batch_size):
                # with torch.cuda.stream(streams[i]):
                process_batch_element(i)

            # Synchronize streams
            # torch.cuda.synchronize()

            if self.enable_flowsim_diff:
                # input_tmp = torch.cat([x[:, :, [0, 2]], batch_h_state_final], dim=2)
                # res = self.output_layer(input_tmp)
                res = self.output_layer(batch_h_state_final) + x[:, :, 2:3]
            else:
                res = self.output_layer(batch_h_state_final) + 1.0
            if self.enable_remainsize:
                loss_size = torch.div(loss_size, loss_size_num)
        elif self.enable_gnn:
            batch_size = x.size(0)
            feature_dim = x.size(2)

            res = torch.zeros((batch_size, x.size(1), 1), device=x.device)
            for i in range(batch_size):
                num_flow_nodes = lengths[i]
                edge_index_trimmed = edge_index[i, :, : edge_index_len[i]]

                max_node_index = edge_index_trimmed.max().item()
                num_link_nodes = max_node_index + 1 - num_flow_nodes

                link_node_feats = torch.full(
                    (num_link_nodes, feature_dim), 1.0, device=x.device
                )
                x_gnn_input = torch.cat(
                    [x[i, :num_flow_nodes, :], link_node_feats], dim=0
                )
                for gcn in self.gcn_layers:
                    x_gnn_input = gcn(x_gnn_input, edge_index_trimmed)

                res[i, :num_flow_nodes, :] = x_gnn_input[:num_flow_nodes, :]
        elif self.enable_lstm:
            # res, _ = self.model_lstm(x[:, :, [0, 1, 4, 5]], lengths)
            res, _ = self.model_lstm(x[:, :, [0, 1]], lengths)
        else:
            assert False, "Either GNN or LSTM must be enabled"
        return res, loss_size

    def step(self, batch, batch_idx, tag=None):
        (
            input,
            output,
            event_info_tuples,
            lengths,
            spec,
            src_dst_pair_target_str,
            active_flowid_tuples,
            active_linkid_tuple,
            active_edges_a_to_b_tuples,
            remainsize_tuples,
        ) = batch

        estimated, loss_size = self(
            input,
            spec,
            event_info_tuples,
            lengths,
            active_flowid_tuples,
            active_linkid_tuple,
            active_edges_a_to_b_tuples,
            remainsize_tuples,
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

        if self.enable_remainsize:
            loss_size_mean = loss_size[loss_size > 0]
            if loss_size_mean.size(0) == 0:
                loss_size_mean = 0
            else:
                loss_size_mean = loss_size_mean.nanmean()
            self._log_loss(loss_size_mean, f"{tag}_size")
            loss += loss_size_mean
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
            sync_dist=self.enable_dist,
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
                parameters += list(self.lstmcell_time_link.parameters())
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
                "monitor": "val_loss_sync",  # Adjust according to the relevant metric
            },
        }


class LinkFctSldnSegment_bk(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
        enable_positional_encoding,
        flow_size_threshold,
        enable_gnn,
        enable_abstime,
        enable_flowsim_gt=False,
        enable_remainsize=False,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        self.use_first_epoch_logic = True
        self.lr = 10.0
        self.enable_positional_encoding = enable_positional_encoding
        self.flow_size_threshold = flow_size_threshold
        self.enable_gnn = enable_gnn
        self.enable_abstime = enable_abstime
        self.enable_flowsim_gt = enable_flowsim_gt
        self.enable_remainsize = enable_remainsize
        logging.info(
            f"call LinkFctSldnSegment. data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}, enable_positional_encoding={enable_positional_encoding}, flow_size_threshold={flow_size_threshold}, enable_gnn={enable_gnn},enable_abstime={enable_abstime}, enable_flowsim_gt={enable_flowsim_gt}, enable_remainsize={enable_remainsize}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type, segment_id = self.data_list[idx]
        src_dst_pair_target_str = (
            "_".join([str(x) for x in src_dst_pair_target]) + f"_seg{segment_id}"
        )

        dir_input_tmp = f"{self.dir_input}/{spec}"

        busy_periods = np.load(
            f"{dir_input_tmp}/period{topo_type}_t{self.flow_size_threshold}.npy",
            allow_pickle=True,
        )
        # busy_periods_time = np.load(
        #     f"{dir_input_tmp}/period_time{topo_type}_t{self.flow_size_threshold}.npy"
        # )
        # assert len(busy_periods) == len(busy_periods_time)

        fid = np.array(busy_periods[segment_id])
        # period_start_time, period_end_time = busy_periods_time[segment_id]
        assert np.all(fid[:-1] <= fid[1:])

        n_flows = len(fid)
        sizes = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
        fats = np.load(f"{dir_input_tmp}/fat.npy")[fid]
        fcts_flowsim = np.load(f"{dir_input_tmp}/fct_flowsim.npy")[fid]

        n_links_passed = np.ones_like(fcts_flowsim) * 2
        base_delay = get_base_delay_link(sizes, n_links_passed, self.lr)
        i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
        fcts_flowsim += base_delay

        sizes = sizes
        fats = fats
        i_fcts_flowsim = i_fcts_flowsim
        fcts_flowsim = fcts_flowsim
        sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)

        if self.enable_flowsim_gt:
            fcts = fcts_flowsim
            i_fcts = i_fcts_flowsim
            # flag_flow_incomplete = np.zeros_like(fats)
        else:
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")[fid]
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")[fid]
            # flag_flow_incomplete = np.array(fats + fcts > period_end_time)
        # assert not flag_flow_incomplete.all()
        # flag_from_last_period = np.array(fats < period_start_time)

        fats = fats - fats[0]
        fcts_stamp = fats + fcts

        # Concatenate the arrays
        event_time_list = np.concatenate((fats, fcts_stamp))
        sorted_indices = np.argsort(event_time_list)
        idx_completion_events = np.where(sorted_indices >= n_flows)[0]
        idx_completion_ranks = np.argsort(fcts_stamp)
        event_departure_list = -np.ones_like(event_time_list, dtype=int)
        event_departure_list[idx_completion_events] = idx_completion_ranks
        event_info_tuple = tuple([event_time_list, event_departure_list])

        active_flowid_tuple = []
        active_flowid = SortedSet()
        idx_arrive_cnt = 0
        for i in range(n_flows * 2):
            active_flowid_tuple.append(list(active_flowid))
            if event_departure_list[i] >= 0:
                flow_id_complete = event_departure_list[i]
                active_flowid.remove(flow_id_complete)
            else:
                active_flowid.add(idx_arrive_cnt)
                idx_arrive_cnt += 1

        if self.enable_remainsize:
            busy_periods_remainsize = np.load(
                f"{dir_input_tmp}/period_remainsize{topo_type}_t{self.flow_size_threshold}.npy",
                allow_pickle=True,
            )
            receivedsize_list = busy_periods_remainsize[segment_id]
            receivedsize_list = [
                np.array(receivedsize) for receivedsize in receivedsize_list
            ]
            assert (
                len(receivedsize_list) == 2 * n_flows
            ), f"len(remain_size): {len(receivedsize_list)}, len(fid): {n_flows}"

            remainsize_tuple = []
            for idx, receivedsize in enumerate(receivedsize_list):
                active_flowid = active_flowid_tuple[idx]
                if len(active_flowid):
                    if len(active_flowid_tuple[idx]) == len(receivedsize):
                        total_size = sizes[active_flowid_tuple[idx]]
                        remainsize_list = (total_size - receivedsize) / total_size
                        assert (remainsize_list >= 0).all()
                        remainsize_tuple.append(torch.tensor(remainsize_list))
                    else:
                        remainsize_tuple.append(torch.tensor([]))
                else:
                    remainsize_tuple.append(torch.tensor([]))
            remainsize_tuple = tuple(remainsize_tuple)
        else:
            remainsize_tuple = None

        active_flowid_tuple = tuple(active_flowid_tuple)

        output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
        assert (output_data >= 1.0).all()

        sizes = np.log2(sizes / 1000.0 + 1)
        if not self.enable_gnn:
            fats = np.diff(fats)
            fats = np.insert(fats, 0, 0)

        # sldn_flowsim[flag_flow_incomplete] = 0
        # output_data[flag_flow_incomplete] = PLACEHOLDER
        # Generate positional encoding

        if self.enable_positional_encoding:
            positional_encodings = self.get_positional_encoding(len(fid), 3)
            input_data = np.column_stack(
                (
                    sizes,
                    fats,
                    sldn_flowsim,
                    # flag_from_last_period,
                    positional_encodings,
                )
            ).astype(np.float32)
        else:
            input_data = np.column_stack(
                (
                    sizes,
                    fats,
                    # fcts_stamp,
                    # i_fcts,
                    sldn_flowsim,
                    # flag_from_last_period,
                )
            ).astype(np.float32)

        # Compute the adjacency matrix for the bipartite graph
        if self.enable_gnn:
            edge_index = self.compute_edge_index(fid)
            flow_id_to_edge = {k: [] for k in range(n_flows)}
            for j in range(edge_index.shape[1]):
                # type_a (flow node) to type_b (link node)
                flow_id_to_edge[edge_index[0, j]].append(edge_index[1, j])
            active_edges_a_to_b_tuple = []
            active_linkid_tuple = []
            for active_flows in active_flowid_tuple:
                link_id_active = set()
                for idx, flow_id in enumerate(active_flows):
                    link_id_active.update(flow_id_to_edge[flow_id])
                active_links = sorted(link_id_active)
                active_linkid_tuple.append(list(active_links))

                active_edges_a_to_b = []
                for idx, flow_id in enumerate(active_flows):
                    for edge in flow_id_to_edge[flow_id]:
                        active_edges_a_to_b.append([idx, edge])
                active_edges_a_to_b_tuple.append(
                    torch.tensor(np.array(active_edges_a_to_b).T)
                )
            active_edges_a_to_b_tuple = tuple(active_edges_a_to_b_tuple)
            active_linkid_tuple = tuple(active_linkid_tuple)
        else:
            active_flowid_tuple = None
            active_linkid_tuple = None
            active_edges_a_to_b_tuple = None

        # assert (input_data >= 0.0).all()

        return (
            input_data,
            output_data,
            event_info_tuple,
            spec + topo_type,
            src_dst_pair_target_str,
            active_flowid_tuple,
            active_linkid_tuple,
            active_edges_a_to_b_tuple,
            remainsize_tuple,
        )

    def get_positional_encoding(self, seq_len, d_model):
        pe = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term[: d_model // 2])
        pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])
        return pe

    def compute_edge_index(self, fid):
        edge_index = []
        for i in range(0, len(fid)):
            # from type_a (flow node) to type_b (link node)
            edge_index.append([i, 0])

        edge_index = np.array(edge_index).T

        # Sort edge_index by destination node (second row)
        sorted_indices = np.lexsort((edge_index[0, :], edge_index[1, :]))
        edge_index = edge_index[:, sorted_indices]
        return edge_index


def collate_fn_bk(batch):
    (
        inputs,
        outputs,
        event_info_tuples,
        specs,
        src_dst_pairs,
        active_flowid_tuples,
        active_linkid_tuples,
        active_edges_a_to_b_tuples,
        remainsize_tuples,
    ) = zip(*batch)

    # Get lengths of each sequence in the batch
    lengths = np.array([x.shape[0] for x in inputs]).astype(np.int64)

    # Pad sequences
    max_len = max(lengths)
    padded_inputs = np.zeros(
        (len(inputs), max_len, inputs[0].shape[1]), dtype=np.float32
    )
    padded_outputs = (
        np.ones((len(outputs), max_len, outputs[0].shape[1]), dtype=np.float32)
        * PLACEHOLDER
    )

    for i, (input, output) in enumerate(zip(inputs, outputs)):
        padded_inputs[i, : input.shape[0], :] = input
        padded_outputs[i, : output.shape[0], :] = output
    # padded_inputs[:, :, 3][padded_inputs[:, :, 3] == 0] = 1

    return (
        torch.tensor(padded_inputs),
        torch.tensor(padded_outputs),
        event_info_tuples,
        lengths,
        specs,
        src_dst_pairs,
        active_flowid_tuples,
        active_linkid_tuples,
        active_edges_a_to_b_tuples,
        remainsize_tuples,
    )
