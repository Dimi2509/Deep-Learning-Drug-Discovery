import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, global_mean_pool, BatchNorm


class GCNBlock(torch.nn.Module):
    """Single GCN layer block with BatchNorm, ReLU, and residual connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = BatchNorm(out_channels)
        
        # Projection for residual if dimensions don't match
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        identity = x
        
        x = self.conv(x, edge_index)
        x = self.norm(x)
        
        # Project residual if needed
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        # Add residual and apply activation
        x = F.relu(x + identity)
        return x

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=4, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        
        # First block: input -> hidden
        self.blocks = torch.nn.ModuleList([
            GCNBlock(num_node_features, hidden_channels)
        ])
        
        # Remaining blocks: hidden -> hidden
        for _ in range(num_layers - 1):
            self.blocks.append(GCNBlock(hidden_channels, hidden_channels))
        
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN blocks
        for block in self.blocks:
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = block(x, edge_index)
        
        # Readout and classification
        x = global_mean_pool(x, batch)
        return self.linear(x)
