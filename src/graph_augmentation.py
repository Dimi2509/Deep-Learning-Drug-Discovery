"""
Fixed Graph augmentation strategies for semi-supervised learning with molecular graphs.
Key improvements:
1. Only adds noise to CONTINUOUS features, not one-hot categorical features
2. Uses feature masking instead of noise for categorical features
3. Reduces/removes edge dropout (bonds are chemistry!)
4. Better defaults for molecular graphs

QM9 node features (11 dimensions):
- [0:5]: One-hot atom type (H, C, N, O, F) - CATEGORICAL, don't add noise!
- [5:6]: Atomic number (normalized)
- [6:7]: Aromatic (binary)
- [7:8]: Hybridization sp (binary)  
- [8:9]: Hybridization sp2 (binary)
- [9:10]: Hybridization sp3 (binary)
- [10:11]: Number of hydrogens (normalized)

Strategy: Only add noise to truly continuous features (indices 5, 10)
Use masking for categorical features if needed.
"""
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge
from copy import deepcopy


# QM9 feature indices
CATEGORICAL_INDICES = list(range(0, 5)) + [6, 7, 8, 9]  # One-hot atom type + binary features
CONTINUOUS_INDICES = [5, 10]  # Atomic number, num hydrogens (normalized continuous)


class WeakAugmentation:
    """
    Weak augmentation for molecular graphs - minimal, chemically-valid perturbation.
    
    Key changes:
    - NO noise on one-hot/categorical features
    - Very small noise only on continuous features
    - NO edge dropout (bonds define chemistry)
    """
    def __init__(self, continuous_noise=0.01, edge_drop_rate=0.0):
        self.continuous_noise = continuous_noise
        self.edge_drop_rate = edge_drop_rate  # Recommend keeping at 0
        
    def __call__(self, data):
        """Apply weak augmentation to a batch."""
        data = deepcopy(data)
        
        # Only add noise to continuous features
        if self.continuous_noise > 0 and len(CONTINUOUS_INDICES) > 0:
            noise = torch.zeros_like(data.x)
            for idx in CONTINUOUS_INDICES:
                if idx < data.x.shape[1]:
                    noise[:, idx] = torch.randn(data.x.shape[0], device=data.x.device) * self.continuous_noise
            data.x = data.x + noise
        
        # Very light edge dropout (optional, but risky for molecules)
        if self.edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                data.edge_index, 
                p=self.edge_drop_rate,
                training=True
            )
            data.edge_index = edge_index
        
        return data


class StrongAugmentation:
    """
    Strong augmentation for molecular graphs.
    
    Key changes:
    - Uses FEATURE MASKING instead of noise for categorical features
    - Only adds noise to continuous features
    - Reduced edge dropout (consider 0.1-0.2 max, or 0)
    """
    def __init__(
        self, 
        continuous_noise=0.05,
        categorical_mask_rate=0.15,  # Mask 15% of categorical features
        continuous_mask_rate=0.1,    # Mask 10% of continuous features  
        edge_drop_rate=0.1,          # Much lower than before!
    ):
        self.continuous_noise = continuous_noise
        self.categorical_mask_rate = categorical_mask_rate
        self.continuous_mask_rate = continuous_mask_rate
        self.edge_drop_rate = edge_drop_rate
        
    def __call__(self, data):
        """Apply strong augmentation to a batch."""
        data = deepcopy(data)
        
        # 1. Add noise ONLY to continuous features
        if self.continuous_noise > 0:
            for idx in CONTINUOUS_INDICES:
                if idx < data.x.shape[1]:
                    noise = torch.randn(data.x.shape[0], device=data.x.device) * self.continuous_noise
                    data.x[:, idx] = data.x[:, idx] + noise
        
        # 2. Mask categorical features (set to 0, don't add noise)
        if self.categorical_mask_rate > 0:
            for idx in CATEGORICAL_INDICES:
                if idx < data.x.shape[1]:
                    mask = torch.rand(data.x.shape[0], device=data.x.device) > self.categorical_mask_rate
                    data.x[:, idx] = data.x[:, idx] * mask.float()
        
        # 3. Mask continuous features
        if self.continuous_mask_rate > 0:
            for idx in CONTINUOUS_INDICES:
                if idx < data.x.shape[1]:
                    mask = torch.rand(data.x.shape[0], device=data.x.device) > self.continuous_mask_rate
                    data.x[:, idx] = data.x[:, idx] * mask.float()
        
        # 4. Light edge dropout (keep low for molecules!)
        if self.edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                data.edge_index, 
                p=self.edge_drop_rate,
                training=True
            )
            data.edge_index = edge_index
        
        return data


class GraphAugmentor:
    """
    Fixed augmentor class for molecular graphs.
    """
    def __init__(self, weak_config=None, strong_config=None):
        # Safe defaults for molecular graphs
        if weak_config is None:
            weak_config = {
                'continuous_noise': 0.01,
                'edge_drop_rate': 0.0,  # No edge dropout for weak aug
            }
        
        if strong_config is None:
            strong_config = {
                'continuous_noise': 0.05,
                'categorical_mask_rate': 0.15,
                'continuous_mask_rate': 0.1,
                'edge_drop_rate': 0.1,  # Low edge dropout
            }
        
        self.weak_aug = WeakAugmentation(**weak_config)
        self.strong_aug = StrongAugmentation(**strong_config)
    
    def weak(self, data):
        """Apply weak augmentation."""
        return self.weak_aug(data)
    
    def strong(self, data):
        """Apply strong augmentation."""
        return self.strong_aug(data)


# Alternative: Even simpler approach - NO structural augmentation at all
class DropoutOnlyAugmentation:
    """
    Safest augmentation: Just use dropout on features, no structural changes.
    This is essentially what happens during training anyway.
    """
    def __init__(self, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    
    def __call__(self, data):
        data = deepcopy(data)
        # Apply dropout to all features (works for both categorical and continuous)
        if self.training:
            data.x = self.dropout(data.x)
        return data


class GraphAugmentorSimple:
    """
    Simplest possible augmentor - just feature dropout.
    Avoids all the complexity of different feature types.
    """
    def __init__(self, weak_dropout=0.05, strong_dropout=0.2):
        self.weak_dropout = weak_dropout
        self.strong_dropout = strong_dropout
    
    def weak(self, data):
        data = deepcopy(data)
        if self.weak_dropout > 0:
            mask = torch.rand_like(data.x) > self.weak_dropout
            data.x = data.x * mask.float()
        return data
    
    def strong(self, data):
        data = deepcopy(data)
        if self.strong_dropout > 0:
            mask = torch.rand_like(data.x) > self.strong_dropout
            data.x = data.x * mask.float()
        return data