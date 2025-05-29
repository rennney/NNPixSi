import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv
import torch_geometric
from torch_geometric.data import Batch
from contextlib import nullcontext
import math, time

from tqdm import tqdm
def pad_or_trim(tensor: torch.Tensor, L: int):
    """
    Pads or trims a (k,2) tensor of (time, ADC) samples to shape (L,2).
    If k < L → pads with zeros. If k > L → trims the latest samples.
    """
    k = tensor.shape[0]
    if k == L:
        return tensor
    elif k > L:
        return tensor[:L]
    else:
        pad = torch.zeros((L - k, 2), dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

class PixelPatchDataset(torch.utils.data.Dataset):
    def __init__(self, event_list, truth_list, patch_size=5):
        self.graphs = []
        radius = patch_size // 2

        for evt,qmap in tqdm(zip(event_list,truth_list)):
            # Group all ADC samples per pixel
            samples_by_pixel = {}
            if(len(evt)<50): continue # lets train only on longer tracks
            for (y, z), t, adc in evt:
                samples_by_pixel.setdefault((y, z), []).append((t, adc))

            # For every pixel with signal, build a 5×5 patch
            for cy, cz in samples_by_pixel:
                x_raw = []
                q_true = []
                node_index = {}
                node_pos = {}
                idx = 0
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        y, z = cy + dy, cz + dz
                        node_index[(y, z)] = idx
                        node_pos[idx] = (dy, dz)
                        idx += 1
                        entries = samples_by_pixel.get((y, z), [])
                        if entries:
                            tensor = torch.tensor(entries, dtype=torch.float32)  # shape (k, 2)
                        else:
                            tensor = torch.zeros((0, 2), dtype=torch.float32)     # force (0, 2)
                        #tensor = torch.tensor(entries, dtype=torch.float32)
                        # Add dy, dz as extra columns (same value for each row)
                        tensor = pad_or_trim(tensor, 10)  # Ensure shape is (10, 2)
                        delta_pos = torch.tensor([abs(dy), abs(dz)], dtype=torch.float32).expand(10, 2)
                        #tensor_with_pos = torch.cat([tensor, delta_pos], dim=1)  # shape (10, 4)
                        x_raw.append(tensor)
                        #x_raw.append(pad_or_trim(tensor, 10))  # 10 samples per pixel
                        q_true.append(qmap.get((y, z), 0.0))
                lambda_decay = 0.1
                # Create edges (e.g., 4- or 8-neighbour grid)
                edge_src, edge_dst, edge_wt = [], [], []
                for (y1, z1), i in node_index.items():
                    for (y2, z2), j in node_index.items():
                        if i != j and abs(y1 - y2) <= 1 and abs(z1 - z2) <= 1:
                            dy1, dz1 = node_pos[i]
                            dy2, dz2 = node_pos[j]
                            dist = math.sqrt((dy1 - dy2)**2 + (dz1 - dz2)**2)
                            wt = max(math.exp(-lambda_decay * dist), 1e-3)
                            edge_src.append(i)
                            edge_dst.append(j)
                            edge_wt.append(wt)  # or plug in from your Ramo kernel

                g = torch_geometric.data.Data()
                g.x_raw = x_raw
                g.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                g.edge_attr = torch.tensor(edge_wt).unsqueeze(1)
                g.Q_true = torch.tensor(q_true, dtype=torch.float32)
                g.num_nodes = len(x_raw)
                #print([x.shape for x in x_raw])
                assert g.edge_index.max() < g.num_nodes , "g.edge_index.max() < g.num_nodes Fail"
                assert all(x.shape == (10, 2) for x in x_raw),  "all(x.shape == (10, 2) for x in x_raw) Fail"
                assert tensor.shape == (10, 2), f"Bad shape: {tensor_with_pos.shape}"
                assert g.edge_index.max() < g.num_nodes, f"edge_index.max()={g.edge_index.max()} vs num_nodes={g.num_nodes}"
                assert g.edge_index.min() >= 0, "Negative index in edge_index"
                self.graphs.append(g)

    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]
