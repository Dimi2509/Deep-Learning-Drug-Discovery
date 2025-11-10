import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv


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

class QM9GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_heads=4, num_layers=3):
        super(QM9GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_channels * num_heads
            self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True))
        
        self.lin = torch.nn.Linear(hidden_channels * num_heads, 1)  # Single target regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # non-linearity after each GAT layer

        # Global pooling for graph-level regression
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x