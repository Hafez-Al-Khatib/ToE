"""
exp_transfer.py
===============
Evaluates Zero-Shot Cross-Domain Generalization.
Tests whether the model learned the "physics of strokes" (MNIST) 
or just memorized the manifold.

Experiment:
1. Load KAN-EBM and MLP-EBM trained ONLY on MNIST.
2. Evaluate denoising performance on Fashion-MNIST (unseen domain).
3. Measure PSNR gain vs. K steps to see if the energy landscape is "universal".
"""

import sys
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from exp_steel_man import SmoothMLPEBM as MLPEBM
from exp_cifar10 import KANEnergyModel as KANEBM
from metrics import compute_psnr

def load_models(device):
    kan_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
    
    kan = KANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    if kan_path.exists():
        kan.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
        print(f"Loaded MNIST-trained KAN-EBM")
        
    mlp = MLPEBM(n_channels=1, n_filters=16, filter_size=5, hidden_dim=128).to(device)
    # Train it quickly on MNIST to ensure it's a valid prior
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    
    print("Pre-training Steel-Man MLP baseline (SiLU) for transfer test...")
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
    mlp.train()
    for _ in range(20):
        for x, _ in loader:
            x = x.to(device)
            loss = mlp.loss(x, 0.3)
            opt.zero_grad(); loss.backward(); opt.step()
            
    kan.eval()
    mlp.eval()
    return kan, mlp

def refine(model, x_noisy, n_steps=10, dt=0.05):
    u = x_noisy.clone()
    for i in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            g = torch.autograd.grad(model.energy(ui).sum(), ui)[0].detach()
        u = (u.detach() - dt * (0.97**i) * g.clamp(-1, 1))
        u = torch.clamp(u, -1, 1)
    return u

def run_transfer_test(quick=False, device_str='auto'):
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
             if device_str == 'auto' else torch.device(device_str)
    
    kan, mlp = load_models(device)
    
    # Load Fashion-MNIST (The UNSEEN domain)
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    fashion_ds = torchvision.datasets.FashionMNIST(ROOT / 'data', train=False, download=True, transform=tf)
    
    n_test = 50 if quick else 200
    loader = torch.utils.data.DataLoader(fashion_ds, batch_size=32, shuffle=False)
    
    sigma = 0.3
    K_values = [0, 1, 5, 10, 20, 50] if not quick else [0, 1, 10]
    
    results = {
        'K_values': K_values,
        'kan_psnr': [],
        'mlp_psnr': []
    }
    
    print(f"\nEvaluating Zero-Shot Transfer: MNIST -> Fashion-MNIST (sigma={sigma})")
    
    for K in K_values:
        kan_batch = []
        mlp_batch = []
        processed = 0
        
        for x, _ in loader:
            if processed >= n_test: break
            x = x.to(device)
            xn = x + torch.randn_like(x) * sigma
            xn = torch.clamp(xn, -1, 1)
            
            with torch.no_grad():
                # For K=0, just use noisy image
                if K == 0:
                    xp_kan, xp_mlp = xn, xn
                else:
                    xp_kan = refine(kan, xn, n_steps=K)
                    xp_mlp = refine(mlp, xn, n_steps=K)
                
                kan_batch.append(compute_psnr(xp_kan, x).item())
                mlp_batch.append(compute_psnr(xp_mlp, x).item())
            
            processed += x.shape[0]
            
        avg_kan = np.mean(kan_batch)
        avg_mlp = np.mean(mlp_batch)
        results['kan_psnr'].append(avg_kan)
        results['mlp_psnr'].append(avg_mlp)
        print(f"  K={K:>2} | KAN PSNR: {avg_kan:.2f} | MLP PSNR: {avg_mlp:.2f}")

    # ── Saving and Plotting ──────────────────────────────────────────────────
    out_dir = ROOT / 'outputs' / 'transfer'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'transfer_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(K_values, results['kan_psnr'], 'b-o', label='KAN-EBM (MNIST trained)')
    plt.plot(K_values, results['mlp_psnr'], 'r-s', label='MLP-EBM (MNIST trained)')
    plt.xlabel('Inference Steps (K)')
    plt.ylabel('PSNR on Fashion-MNIST (dB)')
    plt.title('Zero-Shot Transfer: Trained on MNIST, Evaluated on Fashion-MNIST')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'transfer_plot.png', dpi=300)
    plt.close()

    print(f"Results saved to {out_dir}")
    
    gain_kan = results['kan_psnr'][-1] - results['kan_psnr'][0]
    gain_mlp = results['mlp_psnr'][-1] - results['mlp_psnr'][0]
    print(f"\nTransfer Summary:")
    print(f"  KAN Zero-Shot Gain: {gain_kan:.2f} dB")
    print(f"  MLP Zero-Shot Gain: {gain_mlp:.2f} dB")
    if gain_kan > gain_mlp:
        print(f"  RESULT: KAN-EBM transfers {gain_kan - gain_mlp:.2f} dB better than MLP.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    run_transfer_test(quick=args.quick, device_str=args.device)
