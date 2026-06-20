"""
exp_triple_role_steel.py
========================
The "Perceptual Transfer" Battle: KAN-EBM vs. Large SiLU-MLP.
Tests if the learned energy field can act as a refractive index for planning.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel as KANEBM
from exp_steel_man import SmoothMLPEBM
from wave_solver import EikonalSolver, extract_path
from train_phase3 import generate_maze_with_path

def is_valid_path(path, maze, threshold=0.5):
    if path is None or len(path) < 2: return False
    for r, c in path:
        r_idx, c_idx = int(round(r)), int(round(c))
        if r_idx < 0 or r_idx >= maze.shape[2] or c_idx < 0 or c_idx >= maze.shape[3]:
            return False
        if maze[0, 0, r_idx, c_idx] < threshold: # Wall
            return False
    return True

def get_slowness(model, maze, alpha=0.5, beta=4.0):
    """Zero-shot extraction of n(x) from energy."""
    model.eval()
    with torch.no_grad():
        # For KAN, we can get per-pixel contribution
        # For MLP, we use a sliding window or just the pixelwise feature energy if it's a CNN-EBM
        # Since SmoothMLPEBM is a CNN-based EBM (from my implementation in exp_steel_man),
        # we can extract local energy.
        x = maze.detach()
        # We need the local energy map before the global sum
        # Re-implementing energy for local extraction
        f = F.conv2d(x, model.filters, padding='same')
        prec = F.softplus(model.log_precision).view(1, -1, 1, 1)
        f = torch.tanh(f * prec)
        B, C, H, W = f.shape
        f_flat = f.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        if hasattr(model, 'kan'):
            e_pixel = model.kan(f_flat).abs().view(B, H, W)
        else:
            e_pixel = model.mlp(f_flat).abs().view(B, H, W)
            
        e = e_pixel # (B, H, W)
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)
        return alpha + beta * e_norm

def run_battle(device_str='auto'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Models (MNIST checkpoints)
    kan = KANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    kan_ckpt = ROOT / 'results' / 'kan_ebm_multitask.pt'
    if kan_ckpt.exists(): kan.load_state_dict(torch.load(kan_ckpt, map_location=device))
    
    # Large Smooth MLP (SiLU)
    mlp = SmoothMLPEBM(n_channels=1, n_filters=16, filter_size=5, hidden_dim=128).to(device)
    # Train it quickly on MNIST
    print("Pre-training Large SiLU-MLP on MNIST...")
    from exp_steel_man import train_baseline
    import torchvision.transforms as T
    from torchvision import datasets
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    train_baseline(mlp, device, loader, epochs=3)
    
    # 2. Planning Benchmark
    solver = EikonalSolver(grid_size=(32, 32), n_sweeps=8).to(device)
    n_test = 10
    kan_succ, mlp_succ = 0, 0
    
    print(f"\nEvaluating Perceptual Transfer (Zero-Shot Maze Solving)...")
    for i in range(n_test):
        maze, source, target, _ = generate_maze_with_path(size=32, device=device)
        
        # KAN Path
        n_kan = get_slowness(kan, maze)
        u_kan = solver(n_kan, source)
        path_kan = extract_path(u_kan, source, target)
        if is_valid_path(path_kan, maze): kan_succ += 1
        
        # MLP Path
        n_mlp = get_slowness(mlp, maze)
        u_mlp = solver(n_mlp, source)
        path_mlp = extract_path(u_mlp, source, target)
        if is_valid_path(path_mlp, maze): mlp_succ += 1
        
    print(f"\nResults (Success Rate):")
    print(f"  KAN-EBM (32K params):  {kan_succ/n_test:.2%}")
    print(f"  SiLU-MLP (20K params): {mlp_succ/n_test:.2%}")
    
    if kan_succ > mlp_succ:
        print(f"RESULT: KAN-EBM is { (kan_succ - mlp_succ)/n_test:.2% } better at zero-shot planning transfer.")

if __name__ == '__main__':
    run_battle()
