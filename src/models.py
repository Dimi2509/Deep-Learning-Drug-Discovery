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
    def __init__(self, num_node_features, hidden_channels = 512, dropout = 0.0):
        super(GIN,self).__init__()

        # GIN needs Multi-Layer Perceptron (MLP) for each layer

        # Dropout
        self.dropout = dropout

        # Projection to make the input matches the hidden channels, for residual connection
        self.input_proj = nn.Linear(num_node_features, hidden_channels)

        mlp1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),  #If we remove residual connection add num_node_features at the first input 
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels,hidden_channels),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_channels,hidden_channels)
        )

        mlp3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_channels,hidden_channels)
        )

        mlp4 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_channels,hidden_channels)
        )

        mlp5 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_channels,hidden_channels)
        )

        mlp6 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_channels,hidden_channels)
        )

        


        #GINConv sums is a wrapper learns from their neighbors
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.conv3 = GINConv(mlp3, train_eps=True)
        self.conv4 = GINConv(mlp4, train_eps=True)
        self.conv5 = GINConv(mlp5, train_eps=True)
        #self.conv6 = GINConv(mlp6, train_eps=True)


        #Adding BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        #self.bn6 = nn.BatchNorm1d(hidden_channels)

        #Linear Layer
        self.linear = nn.Linear(hidden_channels,1)

        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        #Input Proyection
        x = self.input_proj(x)
        

        #1 Gin layers
        identity = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = x + identity  # ← Residual connection
        #x = F.dropout(x, p=self.dropout, training=self.training)
        
        #2 Gin Layers
        #identity = x
        x = self.conv2(x,edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = x + identity
        #x = F.dropout(x, p=self.dropout, training=self.training)

        #3 Gin Layers
        #identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = x + identity
        #x = F.dropout(x, p=self.dropout, training=self.training)

        #4 Gin Layers
        #identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = x.relu()
        x = x + identity
        #x = F.dropout(x, p=self.dropout, training=self.training)

        #5 Gin Layers
        #identity = x
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = x.relu()
        x = x + identity
        #x = F.dropout(x, p=self.dropout, training=self.training)

        #6 Gin Layer
        #x = self.conv6(x, edge_index)
        #x = self.bn6(x)
        #x = x.relu()
        #x = x + identity




        #2 readout
        x = global_max_pool(x, batch)

        #3 Prediction
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x