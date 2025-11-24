"""
Simple script to visualize QM9 dataset features.

This will show you:
1. What node features look like (are they categorical or continuous?)
2. Feature value distributions
3. Example molecules and their features
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.datasets import QM9

# Load QM9 dataset
print("Loading QM9 dataset...")
dataset = QM9(root='./data')
print(f"Dataset loaded: {len(dataset)} molecules\n")

# Get first molecule as example
mol = dataset

print("=" * 60)
print("MOLECULE STRUCTURE")
print("=" * 60)
print(f"Number of atoms (nodes): {mol.x.shape[0]}")
print(f"Number of features per atom: {mol.x.shape[1]}")
print(f"Number of bonds (edges): {mol.edge_index.shape[1] // 2}")  # Divided by 2 for undirected
print(f"Target properties shape: {mol.y.shape}")
print()

print("=" * 60)
print("NODE EDGES")
print("=" * 60)
print("Edge Index (first 10 edges):")
print(mol.edge_index)
print(mol.edge_index.size(1)) # Number of edges, both directions
print(mol.edge_index.size(1) // 2) # Number of undirected edges
print(mol.edge_attr)  # First 10 edge attributes
pair_mask = torch.rand(mol.edge_index.size(1) // 2, device=mol.edge_index.device) > 0.99
edge_mask = pair_mask.repeat_interleave(2)
edge_index = mol.edge_index
# Apply mask
print(edge_mask)
print(f"Number of true in edge_mask: {edge_mask.sum()}")
print(f"Number of false in edge_mask: {(~edge_mask).sum()}")
dataset.edge_index = edge_index[:, edge_mask]

print(dataset.edge_index.size(1)) # Number of edges, both directions
print(dataset.edge_index.size(1) // 2) # Number of undirected edges