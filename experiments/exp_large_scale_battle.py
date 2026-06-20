"""
exp_large_scale_battle.py
========================
High-Param Battle: KAN-EBM vs. Smooth-MLP at the 250K-500K Parameter Scale.
Tests the hypothesis: "Does the KAN advantage disappear or widen as we scale up?"
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
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel, train_model, evaluate, psnr
from exp_steel_man import SmoothMLPEBM

def run_large_battle(quick=False, device_str='auto'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_str == 'auto' else torch.device(device_str)
    
    # --- Scaling Configuration ---
    # We want to match around 250K - 500K parameters.
    # KAN Scale-up: Increase filters and grid size
    n_filters_kan = 32 # 3 channels * 32 = 96 input dims
    kan_hidden = [96, 64, 32]
    
    # MLP Scale-up: Increase hidden dimension
    hidden_dim_mlp = 512 # Will result in ~large param count
    
    out_dir = ROOT / 'outputs' / 'large_scale_battle'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_ds = datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    
    if quick:
        train_ds = torch.utils.data.Subset(train_ds, range(2000))
        test_ds = torch.utils.data.Subset(test_ds, range(200))
        
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 2. Build Large Models
    kan = KANEnergyModel(n_filters=n_filters_kan, filter_size=5, kan_hidden=kan_hidden, n_channels=3).to(device)
    mlp = SmoothMLPEBM(n_channels=3, n_filters=n_filters_kan, filter_size=5, hidden_dim=hidden_dim_mlp).to(device)
    
    print(f"\nModel Parameter Counts:")
    print(f"  Large KAN-EBM:  {sum(p.numel() for p in kan.parameters()):,}")
    print(f"  Large SiLU-MLP: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # 3. Train
    sigma = 0.2
    n_epochs = 5 if quick else 30
    
    print(f"\nTraining Large KAN-EBM...")
    train_model(kan, train_loader, device, n_epochs, sigma, tag='Large-KAN')
    
    print(f"\nTraining Large SiLU-MLP...")
    train_model(mlp, train_loader, device, n_epochs, sigma, tag='Large-MLP')
    
    # 4. Evaluate K-Scaling
    K_LIST = [1, 5, 10, 20, 50]
    print(f"\nEvaluating Large-Scale K-Scaling (K={K_LIST})")
    
    kan_results = evaluate(kan, test_loader, device, sigma, K_LIST)
    mlp_results = evaluate(mlp, test_loader, device, sigma, K_LIST)
    
    # 5. Summary & Plot
    print(f"\n{'K':>4} | {'KAN PSNR':>10} | {'MLP PSNR':>10}")
    print("-" * 30)
    for k in K_LIST:
        print(f"{k:>4} | {kan_results[k]:>10.2f} | {mlp_results[k]:>10.2f}")
        
    # Save results
    final_res = {
        'kan': {str(k): v for k,v in kan_results.items()},
        'mlp': {str(k): v for k,v in mlp_results.items()},
        'params': {
            'kan': sum(p.numel() for p in kan.parameters()),
            'mlp': sum(p.numel() for p in mlp.parameters())
        }
    }
    with open(out_dir / 'large_battle_results.json', 'w') as f:
        json.dump(final_res, f, indent=2)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_LIST, [kan_results[k] for k in K_LIST], 'b-o', label='Large KAN-EBM')
    plt.plot(K_LIST, [mlp_results[k] for k in K_LIST], 'r-s', label='Large SiLU-MLP')
    plt.xlabel('Inference Steps (K)')
    plt.ylabel('PSNR (dB)')
    plt.title('CIFAR-10: Large-Scale Battle (Parameter-Matched)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'large_battle_plot.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    run_large_battle(quick=args.quick, device_str=args.device)
