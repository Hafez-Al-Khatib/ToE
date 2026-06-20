"""
exp_flow_matching_cifar.py
==========================
Trains a parameter-matched Flow Matching baseline on CIFAR-10.
Compares against the KAN-EBM results.
"""

import sys
import time
import math
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class UNet32(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1, stride=2)
        self.bottleneck = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1))
        self.upconv = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim * 2, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
        x1 = F.silu(self.conv1(x) + t_emb)
        x2 = F.silu(self.conv2(x1) + t_emb.repeat(1, 2, 1, 1))
        x_mid = self.bottleneck(x2) + x2
        x_up = F.silu(self.upconv(x_mid) + t_emb)
        x_cat = torch.cat([x_up, x1], dim=1)
        return self.conv_out(x_cat)

def train_flow_matching(model, loader, device, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_clean, _ in loader:
            x_clean = x_clean.to(device)
            B = x_clean.shape[0]
            t = torch.rand(B, 1, device=device)
            t_exp = t.view(B, 1, 1, 1)
            x_noise = torch.randn_like(x_clean)
            x_t = (1 - t_exp) * x_noise + t_exp * x_clean
            v_target = x_clean - x_noise
            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

@torch.no_grad()
def evaluate_flow_matching(model, loader, device, sigma=0.2, k_list=[1, 2, 5, 10, 20]):
    model.eval()
    results = {k: [] for k in k_list}
    for batch in loader:
        x_clean = batch[0].to(device)
        x_noisy = x_clean + torch.randn_like(x_clean) * sigma
        for k in k_list:
            dt = 1.0 / k
            x_t = x_noisy.clone()
            t = torch.zeros(x_t.shape[0], 1, device=device)
            for _ in range(k):
                v_pred = model(x_t, t)
                x_t = x_t + v_pred * dt
                t = t + dt
            mse = F.mse_loss(x_t.clamp(-1,1), x_clean).item()
            results[k].append(10 * math.log10(4.0 / mse))
    return {k: float(np.mean(v)) for k, v in results.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    
    # ── Model (Match KAN-EBM params ~32K) ──
    # hidden_dim=22 gives ~32K parameters
    model = UNet32(in_channels=3, hidden_dim=22).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flow-Matching Params: {n_params:,}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_ds = datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_subset = torch.utils.data.Subset(test_ds, range(500))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    print("\nTraining Flow-Matching...")
    train_flow_matching(model, train_loader, device, epochs=5 if args.quick else 50)
    
    print("\nEvaluating K-scaling...")
    k_list = [1, 2, 5, 10, 20, 50]
    fm_results = evaluate_flow_matching(model, test_loader, device, sigma=0.2, k_list=k_list)
    
    # -- Load KAN-EBM results --
    kan_results = {
        "1": 21.89, "2": 23.39, "5": 25.92, "10": 24.10, "20": 20.99, "50": 18.69
    }
    
    # -- Plot --
    plt.figure(figsize=(8, 5))
    ks = [1, 2, 5, 10, 20, 50]
    plt.plot(ks, [fm_results[k] for k in ks], 'r-s', label='Flow-Matching (32K params)')
    plt.plot(ks, [kan_results[str(k)] for k in ks], 'b-o', label='KAN-EBM (32K params)')
    plt.xlabel('Inference Steps (K)')
    plt.ylabel('PSNR (dB)')
    plt.title('CIFAR-10: KAN-EBM vs Flow-Matching (Parameter-Matched)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    out_dir = ROOT / 'outputs' / 'cifar10'
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'kan_vs_fm_cifar.png', dpi=200)
    plt.close()
    
    print(f"\nFinal Comparison (sigma=0.2):")
    for k in ks:
        print(f"  K={k:>2} | KAN: {kan_results[str(k)]:.2f} | FM: {fm_results[k]:.2f}")

if __name__ == '__main__':
    main()
