import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, HeteroConv
from typing import Tuple


# Define custom message passing layers
class TypeAToBConv(MessagePassing):
    def __init__(self, in_channels_a, in_channels_b, out_channels, dropout=0.2):
        super(TypeAToBConv, self).__init__(aggr="add")
        self.lstm = nn.LSTM(in_channels_a, out_channels, batch_first=True)
        self.lin = nn.Linear(in_channels_b + out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], edge_index: torch.Tensor):
        x_src, x_dst = (
            x  # x_src: Features of type A nodes (source), x_dst: Features of type B nodes (target)
        )
        size = (x_src.size(0), x_dst.size(0))
        out = self.propagate(edge_index, x_src=x_src, x_dst=x_dst, size=size)
        return out  # Return the updated features

    def message(self, x_src_j):
        # x_src_j: Features of neighboring type A nodes (source nodes)
        x_j = x_src_j.unsqueeze(1)  # Add sequence dimension for LSTM
        x_j, _ = self.lstm(x_j)  # Pass through LSTM
        return x_j.squeeze(1)  # Remove sequence dimension

    def update(self, aggr_out, x_dst):
        # Combine the aggregated messages with the target node features
        x_b = torch.cat([x_dst, aggr_out], dim=-1)
        x_b = F.relu(self.lin(x_b))
        x_b = self.dropout(x_b)
        return x_b


class TypeBToAConv(MessagePassing):
    def __init__(self, in_channels_b, in_channels_a, out_channels, dropout=0.2):
        super(TypeBToAConv, self).__init__(aggr="add")
        self.lstm = nn.LSTM(in_channels_b, out_channels, batch_first=True)
        self.lin = nn.Linear(in_channels_a + out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], edge_index: torch.Tensor):
        x_src, x_dst = (
            x  # x_src: Features of type B nodes (source), x_dst: Features of type A nodes (target)
        )
        size = (x_src.size(0), x_dst.size(0))
        out = self.propagate(edge_index, x_src=x_src, x_dst=x_dst, size=size)
        return out  # Return the updated features

    def message(self, x_src_j):
        # x_src_j: Features of neighboring type B nodes (source nodes)
        x_j = x_src_j.unsqueeze(1)
        x_j, _ = self.lstm(x_j)
        return x_j.squeeze(1)

    def update(self, aggr_out, x_dst):
        # Combine the aggregated messages with the target node features
        x_a = torch.cat([x_dst, aggr_out], dim=-1)
        x_a = F.relu(self.lin(x_a))
        x_a = self.dropout(x_a)
        return x_a


# Define the Heterogeneous GNN model
class HeteroGNN(torch.nn.Module):
    def __init__(self, c_in_type_a, c_in_type_b, c_out, dropout=0.2):
        super(HeteroGNN, self).__init__()
        self.conv = HeteroConv(
            {
                ("type_a", "a_to_b", "type_b"): TypeAToBConv(
                    c_in_type_a, c_in_type_b, c_out, dropout
                ),
                ("type_b", "b_to_a", "type_a"): TypeBToAConv(
                    c_in_type_b, c_in_type_a, c_out, dropout
                ),
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        return x_dict


# Define your data (as per your original code)
# Graph 1
x_type_a_1 = torch.tensor([[1.0, 0.0]], dtype=torch.float)
x_type_b_1 = torch.tensor([[2.0, 0.0]], dtype=torch.float)
edge_index_a_to_b_1 = torch.tensor([[0], [0]], dtype=torch.long)

# Graph 2
x_type_a_2 = torch.tensor([[0.0, 1.0]], dtype=torch.float)
x_type_b_2 = torch.tensor([[0.0, 2.0]], dtype=torch.float)
edge_index_a_to_b_2 = torch.tensor([[0], [0]], dtype=torch.long)

# Create HeteroData objects
data1 = HeteroData()
data1["type_a"].x = x_type_a_1
data1["type_b"].x = x_type_b_1
data1["type_a", "a_to_b", "type_b"].edge_index = edge_index_a_to_b_1
data1["type_b", "b_to_a", "type_a"].edge_index = edge_index_a_to_b_1.flip(0)

data2 = HeteroData()
data2["type_a"].x = x_type_a_2
data2["type_b"].x = x_type_b_2
data2["type_a", "a_to_b", "type_b"].edge_index = edge_index_a_to_b_2
data2["type_b", "b_to_a", "type_a"].edge_index = edge_index_a_to_b_2.flip(0)

# Create a list of graphs and batch them
data_list = [data1, data2]
loader = DataLoader(data_list, batch_size=2)

# Instantiate the model
model = HeteroGNN(c_in_type_a=2, c_in_type_b=2, c_out=4, dropout=0.2)

# Process the batched data
for batch_data in loader:
    x_dict = batch_data.x_dict
    edge_index_dict = batch_data.edge_index_dict
    out = model(x_dict, edge_index_dict)

    x_type_a = out["type_a"]
    x_type_b = out["type_b"]

    print("Updated features for type A nodes:", x_type_a)
    print("Updated features for type B nodes:", x_type_b)
