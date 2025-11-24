"""
Graph augmentation utilities for Mean Teacher implementation.

These augmentations add gaussian noise to node features, a crucial step to avoid model collapse (where the student and teacher produce identical outputs without learning meaningful representations)
when training with consistency loss between student and teacher models.
"""

import torch
from torch_geometric.data import Data


class GraphAugmentor:   
    def __init__(
        self,
        use_feature_noise: bool = True,
        feature_noise_std: float = 0.1,
        use_edge_dropout: bool = True,
        edge_dropout_rate: float = 0.1,
    ):
        """
        Args:
            use_feature_noise: Whether to add Gaussian noise to node features
            feature_noise_std: Standard deviation of the Gaussian noise
            use_edge_dropout: Whether to randomly drop edges
            edge_dropout_rate: Probability of dropping each edge
        """
        self.use_feature_noise = use_feature_noise
        self.feature_noise_std = feature_noise_std
        self.use_edge_dropout = use_edge_dropout
        self.edge_dropout_rate = edge_dropout_rate
    
    def augment(self, data: Data, training: bool = True) -> Data:
        """
        Apply augmentations to a graph data object.
        
        Args:
            data: PyG Data object containing the graph
            training: If False, no augmentation is applied (for validation)
        
        Returns:
            Augmented Data object (new copy, original is unchanged)
        """
        if not training:
            return data
        
        # Create a copy to avoid modifying the original
        augmented_data = data.clone()
        
        # Apply feature noise
        if self.use_feature_noise and self.feature_noise_std > 0:
            noise = torch.randn_like(augmented_data.x) * self.feature_noise_std
            augmented_data.x = augmented_data.x + noise

        if self.use_edge_dropout and self.edge_dropout_rate > 0:
            augmented_data = self._edge_dropout(augmented_data)
        
        return augmented_data
    
    def _edge_dropout(self, data: Data) -> Data:
        """
        Randomly drop edges from the graph.
        
        Implementation note: We keep both directions of undirected edges together
        to maintain graph symmetry (if edge i->j is dropped, j->i is also dropped).
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Create mask for which edges to keep
        # Assuming edges come in pairs (undirected graph), we process pairs together
        if num_edges % 2 == 0:
            # For undirected graphs, edges come in pairs: (i,j) and (j,i)
            # We drop both or keep both to maintain symmetry
            num_pairs = num_edges // 2
            pair_mask = torch.rand(num_pairs, device=edge_index.device) > self.edge_dropout_rate
            # Repeat each element twice for both directions
            edge_mask = pair_mask.repeat_interleave(2)
        else:
            # Fallback for directed or odd number of edges
            edge_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_dropout_rate
        
        # Apply mask
        data.edge_index = edge_index[:, edge_mask]
        #print(f"Dropped {num_edges - data.edge_index.size(1)} edges out of {num_edges}")
        
        # If edge attributes exist, drop them too
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]
        
        return data
    
    def __call__(self, data: Data, training: bool = True) -> Data:
        """Allow the augmentor to be called as a function."""
        return self.augment(data, training)
    

def augment_batch(batch: Data, augmentor: GraphAugmentor, training: bool = True) -> Data:
    """
    Apply augmentation to a batched graph.
    
    Args:
        batch: Batched PyG Data object
        augmentor: GraphAugmentor instance
        training: Whether we're in training mode
    
    Returns:
        Augmented batch
    """
    return augmentor.augment(batch, training=training)