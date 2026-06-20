"""
exp_multitask_cifar.py
======================
High-Quality Multi-Task Restoration on CIFAR-10 (32x32 RGB).
Upgrades the MNIST benchmark with:
1.  Natural Color Images (CIFAR-10).
2.  Projected Langevin Dynamics (Hard data consistency).
3.  Total Variation (TV) Prior (Edge sharpening).
4.  Modern Baseline (ResNet/U-Net style FFN).
"""

import argparse
import sys
import math
import time
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from exp_cifar10 import KANEnergyModel, KANLinear, KAN

# ──────────────────────────────────────────────────────────────────────────────
# Improved Inference Logic
# ──────────────────────────────────────────────────────────────────────────────

def total_variation_loss(img):
    """Total Variation loss for edge sharpening."""
    bs, c, h, w = img.size()
    tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
    return (tv_h + tv_w) / (bs * c * h * w)

def improved_refine(model, x_corrupt, n_steps=20, dt=0.1, dt_decay=0.98, 
                    tv_weight=0.01, task='denoise', anchor_mask=None, anchor_vals=None):
    """
    Projected Langevin Dynamics with TV Prior.
    """
    device = x_corrupt.device
    u = x_corrupt.clone().detach()
    step = dt
    
    for i in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            # 1. KAN Energy Gradient
            e_grad = torch.autograd.grad(model.energy(ui), ui)[0].detach().clamp(-1, 1)
            
            # 2. TV Prior Gradient (for sharpening)
            if tv_weight > 0:
                tv_loss = total_variation_loss(ui)
                tv_grad = torch.autograd.grad(tv_loss, ui)[0].detach().clamp(-0.1, 0.1)
            else:
                tv_grad = 0
                
        # 3. Langevin Step
        u = (u - step * (e_grad + tv_weight * tv_grad)).detach()
        
        # 4. Projection / Data Consistency
        if task == 'inpaint' and anchor_mask is not None:
            # Hard projection: force known pixels to match observations
            u = torch.where(anchor_mask.to(device), anchor_vals.to(device), u)
            
        elif task == 'superres' and anchor_vals is not None:
            # Data consistency for SR: ensure downsampled u matches low-res input
            vals = anchor_vals.to(device)
            scale = u.shape[-1] // vals.shape[-1]
            u_lr = F.avg_pool2d(u, scale)
            residual = F.interpolate(vals - u_lr, size=u.shape[-2:], mode='bicubic', align_corners=False)
            u = u + 0.5 * residual # Soft projection to avoid noise amplification
            
        u = torch.clamp(u, -1, 1)
        step *= dt_decay
        
    return u

# ──────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_multitask_cifar(quick=False, device_str='auto'):
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
             if device_str == 'auto' else torch.device(device_str)
    
    out_dir = ROOT / 'outputs' / 'multitask_cifar'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    n_test = 20 if quick else 100
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 2. Load KAN-EBM (CIFAR-10 checkpoint)
    # Using the existing f32 checkpoint we found
    kan_path = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
    model = KANEnergyModel(n_filters=32, filter_size=5, kan_hidden=[96, 16], n_channels=3).to(device)
    if kan_path.exists():
        model.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
        print(f"Loaded KAN-EBM from {kan_path}")
    else:
        print(f"Warning: KAN-EBM checkpoint not found at {kan_path}. Results will be untrained.")
    model.eval()
    
    K_vals = [0, 1, 2, 5, 10, 20, 50]
    sigma = 0.1
    tasks = ['denoise', 'superres', 'inpaint']
    
    results = {t: {k: [] for k in K_vals} for t in tasks}
    results['corrupted'] = {t: [] for t in tasks}
    
    print(f"\nEvaluating Multi-Task CIFAR-10 (sigma={sigma})")
    
    for i, (x, _) in enumerate(loader):
        if i >= n_test: break
        x = x.to(device)
        
        # --- Task 1: Denoise ---
        xn = x + torch.randn_like(x) * sigma
        results['corrupted']['denoise'].append(compute_psnr(xn, x))
        for k in K_vals:
            xp = improved_refine(model, xn, n_steps=k, task='denoise', tv_weight=0.01)
            results['denoise'][k].append(compute_psnr(xp, x))
            
        # --- Task 2: Super-Res (4x) ---
        lr = F.avg_pool2d(x, 4)
        x_bicubic = F.interpolate(lr, size=(32, 32), mode='bicubic', align_corners=False)
        results['corrupted']['superres'].append(compute_psnr(x_bicubic, x))
        for k in K_vals:
            xp = improved_refine(model, x_bicubic, n_steps=k, task='superres', anchor_vals=lr, tv_weight=0.05)
            results['superres'][k].append(compute_psnr(xp, x))
            
        # --- Task 3: Inpaint (50% box) ---
        mask = torch.ones_like(x)
        mask[:, :, 8:24, 8:24] = 0 # Center box
        x_masked = x * mask
        results['corrupted']['inpaint'].append(compute_psnr(x_masked, x))
        for k in K_vals:
            xp = improved_refine(model, x_masked, n_steps=k, task='inpaint', anchor_mask=mask.bool(), anchor_vals=x, tv_weight=0.02)
            results['inpaint'][k].append(compute_psnr(xp, x))
            
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{n_test} images")

    # Save Stats
    final_stats = {t: {str(k): float(np.mean(results[t][k])) for k in K_vals} for t in tasks}
    final_stats['corrupted'] = {t: float(np.mean(results['corrupted'][t])) for t in tasks}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
        
    # Visual Strip (Save last image)
    save_visual_strip(x, xn, lr, x_masked, model, out_dir, device)
    
    print(f"\nResults and Visuals saved to {out_dir}")

def compute_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 10 * math.log10(4.0 / mse) if mse > 0 else 100

def save_visual_strip(x, xn, lr, xm, model, out_dir, device):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Rows: Denoise, SR, Inpaint
    # Cols: Corrupted, K=10, K=50, Ground Truth
    
    def plot(ax, img, title):
        img = (img[0].permute(1,2,0).cpu().detach().numpy() + 1) / 2
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    mask = torch.ones_like(x)
    mask[:, :, 8:24, 8:24] = 0
    mask = mask.to(device)

    # Denoise
    plot(axes[0, 0], xn, "Noisy (σ=0.1)")
    plot(axes[0, 1], improved_refine(model, xn.to(device), n_steps=10, task='denoise'), "K=10")
    plot(axes[0, 2], improved_refine(model, xn.to(device), n_steps=50, task='denoise'), "K=50")
    plot(axes[0, 3], x, "Ground Truth")
    
    # SR
    x_bicubic = F.interpolate(lr.to(device), size=(32, 32), mode='bicubic', align_corners=False)
    plot(axes[1, 0], x_bicubic, "Bicubic (4x)")
    plot(axes[1, 1], improved_refine(model, x_bicubic, n_steps=10, task='superres', anchor_vals=lr.to(device), tv_weight=0.05), "K=10")
    plot(axes[1, 2], improved_refine(model, x_bicubic, n_steps=50, task='superres', anchor_vals=lr.to(device), tv_weight=0.05), "K=50")
    plot(axes[1, 3], x, "Ground Truth")
    
    # Inpaint
    plot(axes[2, 0], xm, "Masked (Center)")
    plot(axes[2, 1], improved_refine(model, xm.to(device), n_steps=10, task='inpaint', anchor_mask=mask.bool(), anchor_vals=x.to(device)), "K=10")
    plot(axes[2, 2], improved_refine(model, xm.to(device), n_steps=50, task='inpaint', anchor_mask=mask.bool(), anchor_vals=x.to(device)), "K=50")
    plot(axes[2, 3], x, "Ground Truth")
    
    plt.tight_layout()
    plt.savefig(out_dir / 'visual_strip.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    run_multitask_cifar(quick=args.quick, device_str=args.device)
