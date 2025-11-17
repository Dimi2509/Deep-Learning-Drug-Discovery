import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x

class MPNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features ,hidden_channels=64):
        super().__init__()

        self.edge_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_node_features * hidden_channels),
        )

        self.conv1 = NNConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            nn=self.edge_mlp1,
            aggr="add",
        )

        self.edge_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels * hidden_channels),
        )

        self.conv2 = NNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            nn=self.edge_mlp2,
            aggr="add",
        )

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, data):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Edge-aware message passing
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Graph-level readout
        x = global_mean_pool(x, batch)

        # MLP prediction head
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

