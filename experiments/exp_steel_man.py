"""
exp_steel_man.py
================
The "Steel-Man" Baseline: KAN-EBM vs. Modern Smooth MLP-EBM (SiLU).
Matches the "smoothness" of the activation to isolate the KAN's 
architectural advantage (Locality/Compositionality).
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel as KANEBM
from metrics import compute_psnr

class SmoothMLPEBM(nn.Module):
    """
    Modern Smooth MLP Baseline using SiLU (Swish).
    Matches the 'smoothness' of B-splines to isolate KAN's 
    compositional advantage.
    """
    def __init__(self, n_channels=1, n_filters=16, filter_size=5, hidden_dim=128):
        super().__init__()
        self.filters = nn.Parameter(torch.randn(n_filters, n_channels, filter_size, filter_size) * 0.01)
        self.log_precision = nn.Parameter(torch.zeros(n_filters))
        
        # Using SiLU (standard in modern EBMs/Diffusion)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def energy(self, x):
        B = x.shape[0]
        f = F.conv2d(x, self.filters, padding='same')
        prec = F.softplus(self.log_precision).view(1, -1, 1, 1)
        f = torch.tanh(f * prec)
        f_flat = f.permute(0, 2, 3, 1) # B, H, W, C
        e_pixel = self.mlp(f_flat)
        e = e_pixel.permute(0, 3, 1, 2)
        return e.sum(dim=(1, 2, 3))

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi = xn.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.energy(xi).sum(), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma**2 * grad, x_clean)

def refine(model, x_noisy, n_steps=20, dt=0.05):
    u = x_noisy.clone()
    for i in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            g = torch.autograd.grad(model.energy(ui).sum(), ui)[0].detach()
        u = (u.detach() - dt * (0.97**i) * g.clamp(-1, 1))
        u = torch.clamp(u, -1, 1)
    return u

def train_baseline(model, device, loader, epochs=10, sigma=0.3):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        losses = []
        for x, _ in loader:
            x = x.to(device)
            loss = model.loss(x, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"  Epoch {ep+1} | Loss: {np.mean(losses):.4f}")

def run_steel_man():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models
    # KAN-EBM (~5.8K params)
    kan = KANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    kan_ckpt = ROOT / 'results' / 'kan_ebm_multitask.pt'
    if kan_ckpt.exists(): kan.load_state_dict(torch.load(kan_ckpt, map_location=device))
    
    # Smooth MLP (~20K params - intentionally larger to be a 'Steel Man')
    mlp = SmoothMLPEBM(n_channels=1, n_filters=16, filter_size=5, hidden_dim=128).to(device)
    
    # 2. Data
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_ds, range(200)), batch_size=32)
    
    # 3. Train the Smooth MLP
    print("\nTraining Steel-Man Baseline (SiLU-MLP, 20K params)...")
    train_baseline(mlp, device, train_loader, epochs=5)
    
    # 4. Evaluate K-scaling
    sigma = 0.3
    K_vals = [0, 1, 5, 10, 20, 50]
    
    results = {'kan': [], 'mlp': []}
    
    print(f"\nEvaluating K-scaling (sigma={sigma})")
    for K in K_vals:
        kan_psnrs, mlp_psnrs = [], []
        for x, _ in test_loader:
            x = x.to(device)
            xn = x + torch.randn_like(x) * sigma
            xn = torch.clamp(xn, -1, 1)
            
            with torch.no_grad():
                xp_kan = refine(kan, xn, n_steps=K) if K > 0 else xn
                xp_mlp = refine(mlp, xn, n_steps=K) if K > 0 else xn
                kan_psnrs.append(compute_psnr(xp_kan, x))
                mlp_psnrs.append(compute_psnr(xp_mlp, x))
        
        avg_kan = np.mean(kan_psnrs)
        avg_mlp = np.mean(mlp_psnrs)
        results['kan'].append(avg_kan)
        results['mlp'].append(avg_mlp)
        print(f"  K={K:>2} | KAN: {avg_kan:.2f} dB | SiLU-MLP: {avg_mlp:.2f} dB")

    # 5. Summary
    kan_gain = results['kan'][K_vals.index(10)] - results['kan'][0]
    mlp_gain = results['mlp'][K_vals.index(10)] - results['mlp'][0]
    print(f"\nRefinement Gain (K=10):")
    print(f"  KAN-EBM:  {kan_gain:+.2f} dB")
    print(f"  SiLU-MLP: {mlp_gain:+.2f} dB")

if __name__ == '__main__':
    run_steel_man()
