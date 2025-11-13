import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, global_mean_pool, BatchNorm


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def init_weights(self):
        pass # Weight initialization for GCN??


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x_res = x

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x + x_res)  # Residual connection

        x_res = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x + x_res)  # Residual connection

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x


class GCNIIBlock(torch.nn.Module):
    """Single GCNII layer block."""
    def __init__(self, hidden_channels, alpha=0.1, theta=0.5, layer=1):
        super().__init__()
        self.gcn2 = GCN2Conv(
            hidden_channels,
            alpha=alpha,
            theta=theta,
            layer=layer,
            shared_weights=False
        )

    def forward(self, x, edge_index):
        x = self.gcn2(x, edge_index)
        return F.relu(x)

class GCNIIModel(torch.nn.Module):
    """Scalable GCNII architecture for graph-level regression."""
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=12,    # QM9 has 12 regression targets
        num_layers=8,
        alpha=0.1,
        theta=0.5,
        dropout=0.5
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Initial projection
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)

        # Stack of GCNII blocks
        self.layers = torch.nn.ModuleList([
            GCNIIBlock(hidden_channels, alpha=alpha, theta=theta, layer=i+1)
            for i in range(num_layers)
        ])

        # Output regression head
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial projection
        x = self.lin1(x)
        x = F.relu(x)

        # Stacked GCNII layers
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Regression head
        return self.lin2(x)