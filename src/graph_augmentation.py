"""
Graph augmentation strategies for semi-supervised learning with molecular graphs.
Implements weak and strong augmentations for FixMatch.
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge
import random
from copy import deepcopy


class WeakAugmentation:
    """
    Weak augmentation for graphs - minimal perturbation.
    Used to generate pseudo-labels in FixMatch.
    """
    def __init__(self, node_drop_rate=0.0, edge_drop_rate=0.1, feature_noise=0.01):
        self.node_drop_rate = node_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.feature_noise = feature_noise
    
    def __call__(self, data):
        """Apply weak augmentation to a batch."""
        data = deepcopy(data)
        
        # Light edge dropout
        if self.edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                data.edge_index, 
                p=self.edge_drop_rate,
                training=True
            )
            data.edge_index = edge_index
        
        # Small feature noise
        if self.feature_noise > 0:
            noise = torch.randn_like(data.x) * self.feature_noise
            data.x = data.x + noise
        
        return data


class StrongAugmentation:
    """
    Strong augmentation for graphs - significant perturbation.
    Used for consistency regularization in FixMatch.
    
    Note: Node dropout is disabled for batched graphs to avoid index misalignment issues.
    """
    def __init__(
        self, 
        node_drop_rate=0.0,  # Disabled for batched graphs
        edge_drop_rate=0.3,
        feature_noise=0.1,
        feature_mask_rate=0.1,
        subgraph_rate=0.0
    ):
        if node_drop_rate > 0:
            print("WARNING: node_drop_rate is not supported for batched graphs and will be ignored")
        self.node_drop_rate = 0.0  # Force disable
        self.edge_drop_rate = edge_drop_rate
        self.feature_noise = feature_noise
        self.feature_mask_rate = feature_mask_rate
        self.subgraph_rate = subgraph_rate
    
    def __call__(self, data):
        """Apply strong augmentation to a batch."""
        data = deepcopy(data)
        
        # Strong edge dropout
        if self.edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                data.edge_index, 
                p=self.edge_drop_rate,
                training=True
            )
            data.edge_index = edge_index
        
        # Feature noise
        if self.feature_noise > 0:
            noise = torch.randn_like(data.x) * self.feature_noise
            data.x = data.x + noise
        
        # Feature masking (set random features to 0)
        if self.feature_mask_rate > 0:
            mask = torch.rand_like(data.x) > self.feature_mask_rate
            data.x = data.x * mask.float()
        
        return data


class GraphAugmentor:
    """
    Main augmentor class that provides both weak and strong augmentations.
    """
    def __init__(self, weak_config=None, strong_config=None):
        # Default weak augmentation config
        if weak_config is None:
            weak_config = {
                'node_drop_rate': 0.0,
                'edge_drop_rate': 0.1,
                'feature_noise': 0.01
            }
        
        # Default strong augmentation config
        if strong_config is None:
            strong_config = {
                'node_drop_rate': 0.0,  # Disabled for batched graphs
                'edge_drop_rate': 0.3,
                'feature_noise': 0.1,
                'feature_mask_rate': 0.1,
                'subgraph_rate': 0.0
            }
        
        # Force node_drop_rate to 0 for safety
        if 'node_drop_rate' in strong_config and strong_config['node_drop_rate'] > 0:
            print(f"WARNING: Setting node_drop_rate to 0 (was {strong_config['node_drop_rate']})")
            strong_config['node_drop_rate'] = 0.0
        
        self.weak_aug = WeakAugmentation(**weak_config)
        self.strong_aug = StrongAugmentation(**strong_config)
    
    def weak(self, data):
        """Apply weak augmentation."""
        return self.weak_aug(data)
    
    def strong(self, data):
        """Apply strong augmentation."""
        return self.strong_aug(data)