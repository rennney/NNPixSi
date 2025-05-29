import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv
import torch_geometric
from torch_geometric.data import Batch
from contextlib import nullcontext
import math, time

# === Model ===
class SampleEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(2, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, samples):  # samples: (N, L, 2)
        t = samples[:, :, 0:1] / 10000.0
        a = samples[:, :, 1:2] / 20000.0
        x = torch.cat([t, a], dim=2)  # (N, L, 2)
        x = F.relu(self.fc1(x))       # (N, L, D)
        x = self.norm(F.relu(self.fc2(x)))  # (N, L, D)

        mask = (samples.abs().sum(2) > 0).float()  # (N, L)
        denom = mask.sum(1, keepdim=True).clamp(min=1e-6)  # (N, 1)
        x = (x * mask.unsqueeze(2)).sum(1) / denom  # (N, D)
        return x

class PixelGNN(nn.Module):
    def __init__(self, embed_dim=32, gnn_dim=64, spline_dim=8):
        super().__init__()
        self.enc = SampleEncoder(embed_dim)
        self.gnn = nn.ModuleList([
            GATv2Conv(embed_dim, gnn_dim, heads=4, concat=False, edge_dim=1),
            GATv2Conv(gnn_dim, gnn_dim, heads=4, concat=False, edge_dim=1)
        ])
        #self.head_I = nn.Linear(gnn_dim, spline_dim)
        self.head_Q = nn.Linear(gnn_dim, 2)

    def forward(self, g):
        samples = torch.stack(g.x_raw)
        x = self.enc(samples)
        if torch.isnan(x).any():
            print("NaN in node encodings BEFORE GNN!")
        assert g.edge_index.max() < x.size(0), f"edge_index.max={g.edge_index.max().item()}, x.size(0)={x.size(0)}"
        assert g.edge_attr.size(0) == g.edge_index.size(1), f"edge_attr mismatch: {g.edge_attr.size(0)} vs {g.edge_index.size(1)}"
        for layer in self.gnn:
            try:
                x = F.relu(layer(x, g.edge_index, g.edge_attr))
            except Exception as e:
                print("Exception during GATv2Conv:")
                print("x:", x.shape)
                print("edge_index:", g.edge_index.shape, "max:", g.edge_index.max())
                print("edge_attr:", g.edge_attr.shape)
                raise
        if torch.isnan(x).any():
            print("NaN in GNN output features!")
        #I = F.softplus(self.head_I(x))
        Q, logv = self.head_Q(x).chunk(2, 1)
        logv = torch.clamp(logv, min=-5.0, max=10.0)
        return {"Q": Q.squeeze(1), "log_var": logv.squeeze(1)} #{"I": I, "Q": Q.squeeze(1), "log_var": logv.squeeze(1)}




        
