import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv
import torch_geometric
from torch_geometric.data import Batch
from contextlib import nullcontext
import math, time

# === Loss terms ===
def charge_conservation_loss(pred, g):
    if not hasattr(g, "Q_true"): return torch.tensor(0.0, device=pred["Q"].device)
    return (pred["Q"].sum() - g.Q_true.sum().to(pred["Q"].device)).pow(2)

def unipolarity_loss(I_coeff):
    return F.relu(-I_coeff).mean()

def timing_smoothness_loss(I_coeff, time_grid):
    peak_t = (I_coeff * time_grid).sum(dim=1) / I_coeff.sum(dim=1).clamp(min=1e-6)
    return ((peak_t[:, None] - peak_t[None, :])**2).mean()

def heteroscedastic_nll(Q_pred, logvar, Q_true):
    var = logvar.exp()
    return ((Q_pred - Q_true)**2 / var + logvar).mean() + 1e-3 * logvar.pow(2).mean()

def mean_bias_loss(Q_pred, Q_true):
    return 0.01*(Q_pred.mean() - Q_true.mean()).abs()


def pixel_mse(Q_pred, Q_true):
    w = (Q_true / Q_true.max()).clamp(min=0.05).detach()
    return ((Q_pred - Q_true)**2 * w).mean()


def total_loss(pred, g, time_grid):
    losses = {}
    if hasattr(g, "Q_true"):
        losses["nll"] = heteroscedastic_nll(pred["Q"], pred["log_var"], g.Q_true.to(pred["Q"].device))
    #losses["conservation"] = 0.1*charge_conservation_loss(pred, g)
   # losses["shape"] = unipolarity_loss(pred["I"])
   # losses["timing"] = timing_smoothness_loss(pred["I"], time_grid.to(pred["I"].device))
    #losses["L1"] = mean_bias_loss(pred["Q"],g.Q_true.to(pred["Q"].device))
    losses["mse_w"] = pixel_mse(pred["Q"],g.Q_true.to(pred["Q"].device))
    Q, Qtrue, logv = pred["Q"], g.Q_true, pred["log_var"]
    if torch.isnan(Q).any() or torch.isnan(logv).any():
        print(" NaN in predictions:")
        print("Q:", Q)
        print("log_var:", logv)
    return sum(losses.values()), losses

# === Training loop with evaluation ===
def train_with_regularization(model, train_loader, test_loader, time_grid,nepochs=1, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(nepochs):
        model.train(); total = 0
        for g in train_loader:
            g = g.to(device)
            pred = model(g)
            loss, _ = total_loss(pred, g, time_grid)
            loss.backward(); total += loss.item()
            opt.step(); opt.zero_grad()
        model.eval(); errors = []
        with torch.no_grad():
            for g in test_loader:
                g = g.to(device)
                pred = model(g)
                errors.append((pred["Q"] - g.Q_true.to(device)).cpu())
        err = torch.cat(errors)
        print(f"Epoch {ep+1}: loss={total/len(train_loader):.3f}  MAE={err.abs().mean():.2f}  RMSE={err.pow(2).mean().sqrt():.2f}")
    plt.hist(err.numpy(), bins=40, alpha=0.8)
    plt.title("Q_pred − Q_true residuals"); plt.xlabel("ADC counts"); plt.ylabel("count")
    plt.tight_layout(); plt.show()




        
