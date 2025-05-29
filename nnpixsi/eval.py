import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv
import torch_geometric
from torch_geometric.data import Batch
from contextlib import nullcontext
import math, time

def build_patch(center_yz, samples_by_pixel, truth_map, patch_size=5, L=10):
    """Return torch_geometric.data.Data for one 5×5 patch centred on `center_yz`."""
    radius = patch_size // 2
    node_idx, x_raw, q_true = {}, [], []
    node_pos = {}
    idx = 0
    cy, cz = center_yz

    # -------- nodes ----------
    for dy in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            y, z = cy + dy, cz + dz
            node_idx[(y, z)] = idx
            node_pos[idx] = (dy,dz)
            idx += 1
            entries = samples_by_pixel.get((y, z), [])
            tensor = torch.tensor(entries, dtype=torch.float32)
            tensor = pad_or_trim(tensor, L)  # shape (L, 2)
            delta_pos = torch.tensor([dy, dz], dtype=torch.float32).expand(L, 2)
            #tensor_with_pos = torch.cat([tensor, delta_pos], dim=1)  # shape (L, 4)
            x_raw.append(tensor)
            #tensor = torch.tensor(entries, dtype=torch.float32)
            #x_raw.append(pad_or_trim(tensor, L))
            q_true.append(truth_map.get((y, z), 0.0))
    
    # -------- edges ----------
    lambda_decay=0.1
    edge_src, edge_dst, edge_wt = [], [], []
    for (y1, z1), i in node_idx.items():
        for (y2, z2), j in node_idx.items():
            if i != j and abs(y1 - y2) <= 1 and abs(z1 - z2) <= 1:
                dy1, dz1 = node_pos[i]
                dy2, dz2 = node_pos[j]
                dist = math.sqrt((dy1 - dy2)**2 + (dz1 - dz2)**2)
                wt = max(math.exp(-lambda_decay * dist), 1e-3)
                edge_src.append(i); edge_dst.append(j); edge_wt.append(wt)

    g = torch_geometric.data.Data()
    g.x_raw      = x_raw
    g.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    g.edge_attr  = torch.tensor(edge_wt).unsqueeze(1)
    g.Q_true     = torch.tensor(q_true, dtype=torch.float32)
    return g

def evaluate_single_event(model,
                          event_hits,
                          truth_map,
                          device="cuda",
                          patch_size=5,
                          L_samples=10,
                          time_grid=None,
                          save_dir="event_eval",
                            ploting = True):
    """
    event_hits : list[ ((y,z), t_tick, adc) ]
    truth_map  : dict{ (y,z): q_true }
    """

    # ---------- organise raw samples ----------
    samples_by_pixel = defaultdict(list)
    for (y, z), t, adc in event_hits:
        samples_by_pixel[(y, z)].append((t, adc))

    # ---------- choose a reproducible pixel ordering ----------
    # sort by increasing y then z (or replace by your track ordering)
    pixel_order = sorted(samples_by_pixel.keys())

    q_true, q_meas, q_pred, q_pred_err = [], [], [], []

    model.to(device).eval()
    with torch.no_grad():
        for yz in pixel_order:
            g = build_patch(yz, samples_by_pixel,truth_map,
                            patch_size=patch_size, L=L_samples).to(device)
            out = model(g)

            idx_center = (patch_size**2) // 2                      # centre node
            q_pred.append(out["Q"][idx_center].item())
            q_pred_err.append(out["log_var"][idx_center].exp().sqrt().item())

            q_true.append(truth_map.get(yz, 0.0))
            # raw measured = Σ ADC for this pixel
            q_meas.append(sum(adc for _, adc in samples_by_pixel[yz]))

    q_true = np.asarray(q_true)
    q_meas = np.asarray(q_meas)
    q_pred = np.asarray(q_pred)
    q_pred_err = np.asarray(q_pred_err)

    # ---------- residuals ----------
    eps = 1e-6          # protect against divide-by-zero for empty pixels
    rel_meas = (q_meas - q_true) / np.clip(q_true, eps, None)
    rel_pred = (q_pred - q_true) / np.clip(q_true, eps, None)
    if (ploting):
        # ------------------------------------------------------------------
        #  1 + 2.   Per-pixel charge + residuals (three-pane figure)
        # ------------------------------------------------------------------
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
            3, 1, figsize=(10, 6),
            gridspec_kw={"height_ratios": [2, 1, 1]}, sharex=True)
    
        pix_id = np.arange(len(pixel_order))
    
        ax_top.plot(pix_id, q_true, "k-", lw=2, label="True (sim)")
        ax_top.plot(pix_id, q_meas, "C1o", ms=4, label="Measured")
        ax_top.errorbar(pix_id, q_pred, yerr=q_pred_err,
                        fmt="C2s", ms=4, lw=1, label="Predicted")
        ax_top.set_ylabel("Charge")
        ax_top.set_title("Per-pixel charge along track")
        ax_top.legend()
    
        ax_mid.axhline(0, color="k", lw=0.8)
        ax_mid.plot(pix_id, rel_meas, "C1o", ms=4, label="(meas − true)/true")
        ax_mid.set_ylabel("Relative error")
        ax_mid.legend(loc="upper right")
    
        ax_bot.axhline(0, color="k", lw=0.8)
        ax_bot.errorbar(pix_id, rel_pred, yerr=q_pred_err/np.clip(q_true, eps, None),
                        fmt="C2s", ms=4, lw=1, label="(pred − true)/true")
        ax_bot.set_xlabel("Pixel index along track")
        ax_bot.set_ylabel("Relative error")
        ax_bot.legend(loc="upper right")
    
        #Path(save_dir).mkdir(exist_ok=True)
        plt.tight_layout()
        #plt.savefig(Path(save_dir) / "per_pixel_charge_and_residuals.png", dpi=120)
        plt.show()
    
        # ------------------------------------------------------------------
        #  3.   Two-population histogram
        # ------------------------------------------------------------------
        plt.figure(figsize=(6, 4))
        plt.hist(rel_meas, bins=50, range=(-10,10), alpha=0.6, label=f"(meas − true)/true, $\mu=${rel_meas.mean():.3f}, $\sigma=${rel_meas.std():.3f}")
        plt.hist(rel_pred, bins=50, range=(-10,10), alpha=0.6, label=f"(pred − true)/true, $\mu=${rel_pred.mean():.3f}, $\sigma=${rel_pred.std():.3f}")
        plt.xlabel("Relative error")
        plt.ylabel("Pixel count")
        plt.legend()
        plt.tight_layout()
        #plt.savefig(Path(save_dir) / "residual_histogram.png", dpi=120)
        plt.show()

    return rel_meas.mean() , rel_pred.mean()
