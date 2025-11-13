import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, global_add_pool, global_max_pool, global_sort_pool
import torch.nn as nn


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
    

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels = 64):
        super(GIN,self).__init__()

        # GIN needs Multi-Layer Perceptron (MLP) for each layer

        mlp1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels,hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels,hidden_channels)
        )


        #GINConv sums is a wrapper learns from their neighbors
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)

        #Adding BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        #Linear Layer
        self.linear = nn.Linear(hidden_channels,1)

        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #1 Gin layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()

        x = self.conv2(x,edge_index)
        x = self.bn2(x)
        x = x.relu()


        #2 readout
        x = global_max_pool(x, batch)

        #3 Prediction
        x = self.linear(x)

        return x