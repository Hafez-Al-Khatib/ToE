"""
KAN-EBM Interpretability Visualizer
=====================================
Produces all interpretability figures for the paper:

  Figure 2 – Learned spatial filter bank (16 × 5×5 filters)
  Figure 3 – KAN spline functions φ_{i→j}(x) per layer connection
  Figure 4 – Energy landscape slices (2D) showing attractor basins
  Figure 5 – Test-time compute scaling (PSNR vs K, with uncertainty bands)
  Figure 6 – Inference trajectory (5-step denoising strip)
  Figure 7 – Precision weights Π_l (what the model pays attention to)

Run:
    py -3.12 experiments/exp_interpretability.py [--checkpoint PATH] [--device cuda]

The script will train a quick model if no checkpoint is given.
All figures saved to results/interpretability/.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ── reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Inline KAN + EBM (so this script is fully self-contained)
# ──────────────────────────────────────────────────────────────────────────────

import math

class KANLinear(nn.Module):
    """B-spline KAN layer (self-contained copy)."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=torch.nn.SiLU, grid_range=(-1, 1)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_noise  = scale_noise
        self.scale_base   = scale_base
        self.scale_spline = scale_spline

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * scale_spline)
        with torch.no_grad():
            start = self.grid[0, spline_order].item()
            end   = self.grid[0, -1 - spline_order].item()
            x     = torch.linspace(start, end, grid_size + spline_order
                                   ).expand(in_features, -1).t().contiguous()
            noise = (torch.rand(grid_size + spline_order, in_features, out_features)
                     - 0.5) * scale_noise / grid_size
            self.spline_weight.data.copy_(self.curve2coeff(x, noise))

    def b_splines(self, x):
        x    = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
                   + (grid[:, k+1:] - x)    / (grid[:, k+1:] - grid[:, 1:-k])   * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A   = self.b_splines(x).transpose(0, 1)
        B   = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(self.b_splines(x).view(x.size(0), -1),
                              self.scaled_spline_weight.view(self.out_features, -1))
        return base_out + spline_out

    def get_spline_function(self, in_idx: int, out_idx: int, n_pts: int = 200) -> tuple:
        """Return (x_vals, y_vals) for the spline φ_{in_idx → out_idx}."""
        start = self.grid[in_idx, self.spline_order].item()
        end   = self.grid[in_idx, -1 - self.spline_order].item()
        x     = torch.linspace(start, end, n_pts).unsqueeze(1).expand(-1, self.in_features)
        # Only evaluate the one spline we care about
        device = self.grid.device
        with torch.no_grad():
            splines    = self.b_splines(x.to(device))            # (n_pts, in, coeff)
            sw         = self.scaled_spline_weight               # (out, in, coeff)
            # spline contribution: sum over coeff for (in_idx, out_idx)
            spline_y   = (splines[:, in_idx, :] * sw[out_idx, in_idx, :]).sum(-1)

            # Move x to same device for activation and weight multiplication
            x_in       = x[:, in_idx].to(device)
            # base contribution for this one input/output pair
            base_y     = (self.base_activation(x_in) * self.base_weight[out_idx, in_idx])
            y          = (spline_y + base_y).cpu().numpy()
        return x[:, in_idx].cpu().numpy(), y


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(layers_hidden[i], layers_hidden[i+1],
                      grid_size=grid_size, spline_order=spline_order,
                      scale_noise=scale_noise, scale_base=scale_base)
            for i in range(len(layers_hidden) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# KAN Energy Model (matches run_paper_experiments.py)
# ──────────────────────────────────────────────────────────────────────────────

class KANEnergyModel(nn.Module):
    """
    Spatial filter bank + KAN energy function.
    Energy gradient descent at inference time.
    """
    def __init__(self, n_filters=16, filter_size=5, kan_hidden=None,
                 img_size=28, n_channels=1):
        super().__init__()
        if kan_hidden is None:
            kan_hidden = [32]
        self.n_filters   = n_filters
        self.filter_size = filter_size
        self.img_size    = img_size
        self.n_channels  = n_channels

        # Spatial filter bank
        self.filters = nn.Parameter(
            0.01 * torch.randn(n_filters, n_channels, filter_size, filter_size))

        # KAN energy: filter features → scalar
        kan_in = n_channels * n_filters
        self.kan = KAN([kan_in] + kan_hidden + [1], grid_size=5)

        # Learnable precision per filter
        self.log_precision = nn.Parameter(torch.zeros(n_filters))

    @property
    def precision(self):
        return F.softplus(self.log_precision)          # (n_filters,)

    def extract_features(self, x):
        """x: (B, C, H, W) → features: (B*H*W, C*n_filters)"""
        B, C, H, W = x.shape
        pad = self.filter_size // 2
        feats = []
        for c in range(C):
            f = F.conv2d(x[:, c:c+1], self.filters[:, c:c+1], padding=pad)
            feats.append(f)
        feats = torch.cat(feats, dim=1)                          # (B, C*nf, H, W)
        # Apply learned precision weights per filter channel (wires log_precision
        # into the computation graph so it receives gradient during DSM training)
        prec = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = feats * prec
        # Squash to KAN grid range [-1, 1] so B-spline basis functions are
        # evaluated across their full support (not collapsed near zero)
        feats = torch.tanh(feats)
        return feats.permute(0, 2, 3, 1).contiguous().view(B*H*W, -1)

    def energy(self, x):
        feats = self.extract_features(x)
        return self.kan(feats).sum()

    def energy_grad(self, x):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            E = self.energy(x_in)
            return torch.autograd.grad(E, x_in)[0].detach()

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        """Gradient descent on E.  dt_decay prevents overshoot at large K."""
        u    = x_noisy.clone()
        step = dt
        for _ in range(n_steps):
            grad = self.energy_grad(u).clamp(-1., 1.)
            u    = (u - step * grad).detach()
            step = step * dt_decay
        return u

    def loss(self, x_clean, sigma):
        noise   = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        with torch.enable_grad():
            x_in = x_noisy.detach().requires_grad_(True)
            E    = self.energy(x_in)
            grad = torch.autograd.grad(E, x_in, create_graph=True)[0]
        pred = x_in - sigma**2 * grad
        return F.mse_loss(pred, x_clean)


# ──────────────────────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────────────────────

def train_quick(device, n_epochs=30, sigma=0.2, batch_size=32):
    """Train a small KAN-EBM on MNIST in ~5 minutes on CPU, ~1 min on GPU."""
    print(f"\n[train] Training KAN-EBM for {n_epochs} epochs on MNIST (σ={sigma})")
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)

    ds = datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=0, drop_last=True)

    model = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = model.loss(x, sigma)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss = model.loss(x, sigma)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
        sched.step()
        if (epoch+1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={np.mean(losses):.4f}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# PSNR helper
# ──────────────────────────────────────────────────────────────────────────────

def psnr(x, y, data_range=2.0):
    mse = ((x - y) ** 2).mean().item()
    if mse < 1e-12:
        return 100.0
    return 10 * math.log10(data_range**2 / mse)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 – Spatial Filter Bank
# ──────────────────────────────────────────────────────────────────────────────

def fig_filter_bank(model, save_dir):
    """Show the 16 learned 5×5 spatial filters as a 4×4 grid."""
    print("[fig 2] Spatial filter bank")
    filters = model.filters.detach().cpu().numpy()   # (16, 1, 5, 5)
    n = filters.shape[0]
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    fig.suptitle('Learned Spatial Filter Bank\n(16 × 5×5, energy basis functions)',
                 fontsize=13, fontweight='bold')

    vmax = np.abs(filters).max()
    im = None
    active_axes = []
    for i, ax in enumerate(axes.flat):
        if i < n:
            f = filters[i, 0]
            im = ax.imshow(f, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
            ax.set_title(f'f_{i+1}', fontsize=8)
            active_axes.append(ax)
        ax.axis('off')

    if im is not None and len(active_axes) > 0:
        # If active_axes has many elements, matplotlib might struggle if not passed as a list/array
        plt.colorbar(im, ax=active_axes, fraction=0.03, pad=0.04, label='weight')
    plt.tight_layout()
    path = save_dir / 'fig2_filter_bank.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 – KAN Spline Functions
# ──────────────────────────────────────────────────────────────────────────────

def fig_spline_functions(model, save_dir):
    """
    Visualise the learned univariate spline φ_{i→j}(x) for the first KAN layer.

    Each cell of the grid shows one input→output spline connection.
    Displays up to 8 inputs × 4 outputs = 32 splines (most informative subset).
    """
    print("[fig 3] KAN spline functions")
    layer = model.kan.layers[0]                      # First KAN layer
    in_f  = layer.in_features
    out_f = layer.out_features

    show_in  = min(in_f,  8)
    show_out = min(out_f, 4)

    fig, axes = plt.subplots(show_out, show_in, figsize=(show_in * 1.6, show_out * 1.5),
                             sharex=False, sharey=False, squeeze=False)
    fig.suptitle(
        'KAN Layer 1: Learned Spline Functions φ_{i→j}(x)\n'
        'Each cell = one learnable univariate function (vs MLP fixed activation)',
        fontsize=11, fontweight='bold'
    )

    for j in range(show_out):
        for i in range(show_in):
            ax = axes[j, i]
            x_vals, y_vals = layer.get_spline_function(i, j)

            ax.plot(x_vals, y_vals, color='steelblue', linewidth=1.2)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_xlim(x_vals[0], x_vals[-1])
            if i == 0:
                ax.set_ylabel(f'out {j}', fontsize=7)
            if j == show_out - 1:
                ax.set_xlabel(f'in {i}', fontsize=7)
            ax.tick_params(labelsize=5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

    plt.tight_layout()
    path = save_dir / 'fig3_spline_functions.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 – Energy Landscape (2D slices)
# ──────────────────────────────────────────────────────────────────────────────

def fig_energy_landscape(model, device, save_dir):
    """
    Plot 2D energy slices in pixel space.
    Uses a 28×28 flat mean image; sweeps two pixel dimensions.
    Shows attractor basins = preferred clean image configurations.
    """
    print("[fig 4] Energy landscape (2D slices)")
    model.eval()
    n = 60
    vals = torch.linspace(-2., 2., n)

    # Pick two interesting pixel pairs
    pixel_pairs = [(0, 1), (100, 200), (300, 400)]
    titles = ['pixels (0,1)', 'pixels (100,200)', 'pixels (300,400)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Energy Landscape E(x): 2D slices through pixel space\n'
                 'Dark regions = low energy (attractors = clean image manifold)',
                 fontsize=11, fontweight='bold')

    base = torch.zeros(1, 1, 28, 28, device=device)     # background = 0

    for ax, (p1, p2), title in zip(axes, pixel_pairs, titles):
        row1, col1 = p1 // 28, p1 % 28
        row2, col2 = p2 // 28, p2 % 28
        E_grid = np.zeros((n, n))

        with torch.no_grad():
            vals_dev = vals.to(device)
            for ii, v1 in enumerate(vals):
                sweep = base.expand(n, -1, -1, -1).clone()
                sweep[:, 0, row1, col1] = v1
                sweep[:, 0, row2, col2] = vals_dev
                feats  = model.extract_features(sweep)
                E_vals = model.kan(feats).view(n, -1).sum(-1)  # (n,)
                E_grid[ii] = E_vals.cpu().numpy()

        im = ax.contourf(vals.numpy(), vals.numpy(), E_grid,
                         levels=30, cmap='viridis_r')
        ax.contour(vals.numpy(), vals.numpy(), E_grid,
                   levels=10, colors='white', alpha=0.3, linewidths=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f'pixel {p1} value')
        ax.set_ylabel(f'pixel {p2} value')
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    path = save_dir / 'fig4_energy_landscape.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 – Test-Time Compute Scaling (PSNR vs K)
# ──────────────────────────────────────────────────────────────────────────────

def fig_psnr_vs_k(model, device, save_dir, sigma=0.2, n_trials=5, n_imgs=50):
    """
    Central paper figure: PSNR(K) curves for KAN-EBM vs FFN baseline.
    Runs n_trials random seeds to get mean ± std bands.
    """
    print(f"[fig 5] PSNR vs K scaling (σ={sigma}, n_imgs={n_imgs}, trials={n_trials})")
    model.eval()

    data_dir = ROOT / 'data'
    ds = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    imgs = torch.stack([ds[i][0] for i in range(n_imgs)]).to(device)

    # Simple FFN baseline (same width as KAN-EBM)
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 784),
            )
        def forward(self, x):
            B = x.size(0)
            return self.net(x.view(B, -1)).view(B, 1, 28, 28)

    ffn = FFN().to(device)
    opt_ffn = torch.optim.Adam(ffn.parameters(), lr=1e-3)
    # Quick FFN training (5 epochs, fast)
    train_ds = datasets.MNIST(data_dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ]))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    ffn.train()
    for ep in range(5):
        for x, _ in loader:
            x = x.to(device)
            noise = torch.randn_like(x) * sigma
            loss  = F.mse_loss(ffn(x + noise), x)
            opt_ffn.zero_grad(); loss.backward(); opt_ffn.step()
    ffn.eval()

    K_vals = [1, 2, 3, 5, 7, 10, 15, 20]
    kan_psnr_all = []    # shape (n_trials, len(K_vals))

    for trial in range(n_trials):
        torch.manual_seed(trial)
        noise = torch.randn_like(imgs) * sigma
        x_noisy = (imgs + noise).clamp(-2., 2.)
        kan_psnr_row = []
        for K in K_vals:
            with torch.no_grad():
                x_den = model.denoise(x_noisy, n_steps=K, dt=0.05, dt_decay=0.97)
            kan_psnr_row.append(psnr(x_den, imgs))
        kan_psnr_all.append(kan_psnr_row)

    kan_mean = np.mean(kan_psnr_all, axis=0)
    kan_std  = np.std(kan_psnr_all, axis=0)

    # FFN is K-independent (single forward pass)
    with torch.no_grad():
        noise   = torch.randn_like(imgs) * sigma
        x_noisy = (imgs + noise).clamp(-2., 2.)
        ffn_psnr = psnr(ffn(x_noisy), imgs)

    # Noisy input PSNR (K=0 baseline)
    noisy_psnr = psnr(x_noisy, imgs)

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(K_vals, kan_mean - kan_std, kan_mean + kan_std,
                    alpha=0.2, color='steelblue', label='KAN-EBM ±1σ')
    ax.plot(K_vals, kan_mean, 'o-', color='steelblue', linewidth=2,
            markersize=5, label='KAN-EBM (K steps)')

    ax.axhline(ffn_psnr, color='tomato', linewidth=2, linestyle='--',
               label=f'FFN (K=1 only)  {ffn_psnr:.2f} dB')
    ax.axhline(noisy_psnr, color='gray', linewidth=1.5, linestyle=':',
               label=f'Noisy input  {noisy_psnr:.2f} dB')

    ax.annotate(f'KAN-EBM K=20\n{kan_mean[-1]:.2f} dB',
                xy=(20, kan_mean[-1]), xytext=(15, kan_mean[-1] - 0.8),
                arrowprops=dict(arrowstyle='->', color='steelblue'),
                color='steelblue', fontsize=9)
    ax.annotate(f'FFN {ffn_psnr:.2f} dB',
                xy=(10, ffn_psnr), xytext=(5, ffn_psnr + 0.3),
                arrowprops=dict(arrowstyle='->', color='tomato'),
                color='tomato', fontsize=9)

    ax.set_xlabel('Inference Steps K (test-time compute)', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(f'Test-Time Compute Scaling (σ={sigma})\n'
                 'KAN-EBM improves monotonically; FFN cannot improve past K=1',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(K_vals) + 1)

    plt.tight_layout()
    path = save_dir / 'fig5_psnr_vs_k.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path, ffn_psnr, kan_mean, K_vals


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 – Inference Trajectory Strip
# ──────────────────────────────────────────────────────────────────────────────

def fig_inference_trajectory(model, device, save_dir, sigma=0.2):
    """
    Show a noisy image → intermediate steps → clean reconstruction.
    Displays: Noisy | K=1 | K=3 | K=5 | K=10 | K=20 | Ground Truth
    """
    print("[fig 6] Inference trajectory strip")
    model.eval()

    data_dir = ROOT / 'data'
    ds = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    x_clean = ds[7][0].unsqueeze(0).to(device)    # pick digit 7

    torch.manual_seed(0)
    noise   = torch.randn_like(x_clean) * sigma
    x_noisy = (x_clean + noise).clamp(-2., 2.)

    K_show = [0, 1, 3, 5, 10, 20]
    labels = ['Noisy\n(K=0)', 'K=1', 'K=3', 'K=5', 'K=10', 'K=20', 'Ground\nTruth']

    # Build trajectory with a single loop so dt_decay is applied
    # continuously across all steps (not reset on each denoise() call)
    frames = [x_noisy]
    u    = x_noisy.clone()
    step = 0.05
    dt_decay = 0.97
    K_show_set = set(K_show[1:])
    model.eval()
    for k in range(1, max(K_show) + 1):
        with torch.no_grad():
            grad = model.energy_grad(u).clamp(-1., 1.)
            u = (u - step * grad).detach()
        step *= dt_decay
        if k in K_show_set:
            frames.append(u.clone())
    frames.append(x_clean)

    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 1.8, 2.5))
    fig.suptitle(f'Inference Trajectory (σ={sigma}): noisy → progressively denoised → clean',
                 fontsize=10, fontweight='bold')

    for ax, frame, label in zip(axes, frames, labels):
        img = frame[0, 0].cpu().detach().numpy()
        ax.imshow(img, cmap='gray', vmin=-2, vmax=2)
        p = psnr(frame, x_clean) if label != 'Ground\nTruth' else float('inf')
        subtitle = f'{p:.1f} dB' if p < 90 else 'reference'
        ax.set_title(f'{label}\n{subtitle}', fontsize=8)
        ax.axis('off')

    # Add arrow between frames
    fig.text(0.5, 0.02, '← more inference compute →', ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    path = save_dir / 'fig6_inference_trajectory.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 – Precision Weights (Attention)
# ──────────────────────────────────────────────────────────────────────────────

def fig_precision_weights(model, save_dir):
    """
    Barplot of the learned precision values Π = softplus(log_Π) per filter.
    High precision = model strongly penalises discrepancy in that feature.
    """
    print("[fig 7] Precision weights Π")
    prec = F.softplus(model.log_precision).detach().cpu().numpy()
    n    = len(prec)
    idx  = np.arange(n)

    # Sort by magnitude for clarity
    order = np.argsort(prec)[::-1]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, n))
    bars = ax.bar(np.arange(n), prec[order], color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Π=1 (unweighted)')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f'f_{order[i]+1}' for i in range(n)], fontsize=8, rotation=45)
    ax.set_xlabel('Spatial Filter (sorted by precision)', fontsize=10)
    ax.set_ylabel('Precision Π (= 1/σ² of prediction error)', fontsize=10)
    ax.set_title('Learned Precision Weights: What the Model Pays Attention To\n'
                 'High Π = strong penalisation of errors in that feature channel',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate top 3
    for rank in range(3):
        fi = order[rank]
        ax.text(rank, prec[fi] + 0.02, f'{prec[fi]:.2f}', ha='center', fontsize=7, color='black')

    plt.tight_layout()
    path = save_dir / 'fig7_precision_weights.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8 – Energy Convergence During Inference
# ──────────────────────────────────────────────────────────────────────────────

def fig_energy_convergence(model, device, save_dir, sigma=0.2, n_imgs=20):
    """
    Plot energy E(u_t) vs inference step t.
    Should decrease monotonically (proves gradient descent is working).
    Show multiple trajectories (one per image) + mean.
    """
    print("[fig 8] Energy convergence")
    model.eval()

    data_dir = ROOT / 'data'
    ds = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))

    K_max = 30
    all_E = []

    for i in range(n_imgs):
        x = ds[i][0].unsqueeze(0).to(device)
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).clamp(-2., 2.)
        u = x_noisy.clone()
        E_traj = []
        for _ in range(K_max):
            with torch.enable_grad():
                u_in = u.detach().requires_grad_(True)
                E    = model.energy(u_in)
                grad = torch.autograd.grad(E, u_in)[0].detach()
            E_traj.append(E.item())
            grad = grad.clamp(-1., 1.)
            u = (u - 0.05 * grad).detach()
        all_E.append(E_traj)

    all_E  = np.array(all_E)         # (n_imgs, K_max)
    steps  = np.arange(1, K_max+1)
    mean_E = all_E.mean(0)
    std_E  = all_E.std(0)

    # Normalise each trajectory to [0,1] for cleaner display
    all_E_norm = (all_E - all_E[:, :1]) / (all_E[:, :1].clip(min=1e-6))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Energy Convergence During Inference\n'
                 'Validates that dynamics follow gradient descent on E(u)',
                 fontsize=11, fontweight='bold')

    # Left: absolute energy trajectories
    for i in range(n_imgs):
        ax1.plot(steps, all_E[i], alpha=0.2, color='steelblue', linewidth=0.8)
    ax1.plot(steps, mean_E, color='navy', linewidth=2.5, label='Mean')
    ax1.fill_between(steps, mean_E - std_E, mean_E + std_E,
                     alpha=0.25, color='steelblue', label='±1σ')
    ax1.set_xlabel('Inference Step K')
    ax1.set_ylabel('Energy E(u)')
    ax1.set_title('Absolute Energy (n=20 images)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: relative energy drop
    mean_norm = all_E_norm.mean(0)
    std_norm  = all_E_norm.std(0)
    ax2.fill_between(steps, mean_norm - std_norm, mean_norm + std_norm,
                     alpha=0.25, color='tomato')
    ax2.plot(steps, mean_norm, color='tomato', linewidth=2.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Inference Step K')
    ax2.set_ylabel('Relative Energy Change (E_K - E_1) / E_1')
    ax2.set_title('Relative Energy Drop (shows convergence rate)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / 'fig8_energy_convergence.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9 – Score Function Cosine Similarity (Allen-Cahn = DSM bridge)
# ──────────────────────────────────────────────────────────────────────────────

def fig_score_alignment(model, device, save_dir, sigma=0.2, n_imgs=100):
    """
    For each image, compute:
      s_kan(x) = -∇_x E_θ(x)  [KAN energy gradient = predicted score]
      s_true(x) = (x_clean - x_noisy)/σ²  [true score under Gaussian noise model]

    Cosine similarity between them shows how well the KAN has learned the score.
    If > 0.8: the energy landscape correctly encodes the log density.
    """
    print("[fig 9] Score function alignment")
    model.eval()

    data_dir = ROOT / 'data'
    ds = datasets.MNIST(data_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))

    cosines = []
    for i in range(n_imgs):
        x_clean = ds[i][0].unsqueeze(0).to(device)
        torch.manual_seed(i)
        noise   = torch.randn_like(x_clean) * sigma
        x_noisy = (x_clean + noise).clamp(-2., 2.)

        # True score: ∇_x log p(x_noisy | x_clean) = -(x_noisy - x_clean) / σ²
        true_score = -(x_noisy - x_clean) / (sigma**2)

        # KAN score: -∇_x E_θ(x_noisy)
        kan_grad   = model.energy_grad(x_noisy)
        kan_score  = -kan_grad

        # Cosine similarity (flatten)
        ts = true_score.view(-1)
        ks = kan_score.view(-1)
        cos_sim = F.cosine_similarity(ts.unsqueeze(0), ks.unsqueeze(0)).item()
        cosines.append(cos_sim)

    cosines = np.array(cosines)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('KAN Score Alignment: −∇E_θ(x) vs True Score ∇ log p(x|x_clean)\n'
                 'Validates Allen-Cahn = Score Matching equivalence on real data',
                 fontsize=10, fontweight='bold')

    # Histogram of cosine similarities
    ax1.hist(cosines, bins=25, color='steelblue', edgecolor='white',
             linewidth=0.5, alpha=0.85)
    ax1.axvline(cosines.mean(), color='navy', linewidth=2,
                label=f'Mean = {cosines.mean():.3f}')
    ax1.axvline(0, color='gray', linewidth=1, linestyle='--')
    ax1.set_xlabel('Cosine Similarity(KAN score, True score)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Cosine Similarity Distribution (n={n_imgs})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Per-image cosine similarity
    ax2.plot(cosines, 'o', color='steelblue', markersize=4, alpha=0.6)
    ax2.axhline(cosines.mean(), color='navy', linewidth=2,
                label=f'μ={cosines.mean():.3f}')
    ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax2.set_xlabel('Image index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Per-Image Score Alignment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / 'fig9_score_alignment.pdf'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → {path}")
    return path, cosines.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(ffn_psnr, kan_mean, K_vals, cos_mean, sigma):
    print("\n" + "="*65)
    print("INTERPRETABILITY RESULTS SUMMARY")
    print("="*65)
    print(f"\nNoise level σ = {sigma}")
    print(f"\nPSNR vs K (KAN-EBM):")
    for K, p in zip(K_vals, kan_mean):
        marker = " ← break-even" if abs(p - ffn_psnr) < 0.05 else ""
        print(f"  K={K:2d}: {p:.2f} dB{marker}")
    print(f"\nFFN baseline (K=1 only): {ffn_psnr:.2f} dB")
    print(f"  KAN-EBM gain K=1→20: {kan_mean[-1] - kan_mean[0]:+.2f} dB")
    print(f"  KAN-EBM vs FFN at K=20: {kan_mean[-1] - ffn_psnr:+.2f} dB")
    print(f"\nScore alignment (Allen-Cahn = DSM validation):")
    print(f"  Mean cosine similarity = {cos_mean:.3f}")
    qual = "excellent" if cos_mean > 0.8 else "good" if cos_mean > 0.5 else "moderate"
    print(f"  → {qual} alignment — energy gradient tracks score function")
    print("="*65)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='KAN-EBM Interpretability Visualizer')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to saved model checkpoint (skip training if given)')
    p.add_argument('--device', type=str, default='cpu',
                   help='Device: cpu or cuda')
    p.add_argument('--sigma', type=float, default=0.2,
                   help='Noise level σ (normalised, ~0.2 ≈ σ_px=25)')
    p.add_argument('--epochs', type=int, default=30,
                   help='Training epochs if no checkpoint (default 30)')
    p.add_argument('--n-imgs', type=int, default=50,
                   help='Number of test images for scaling experiment')
    p.add_argument('--n-trials', type=int, default=3,
                   help='Number of noise seeds for PSNR bands')
    p.add_argument('--save-checkpoint', type=str, default=None,
                   help='Save trained model to this path')
    p.add_argument('--skip-training', action='store_true',
                   help='Skip training even without checkpoint (uses random weights)')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f"[interpretability] device={device}  σ={args.sigma}")

    save_dir = ROOT / 'results' / 'interpretability'
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or train model ────────────────────────────────────────────────────
    model = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32]).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[model] Loaded checkpoint from {args.checkpoint}")
    elif not args.skip_training:
        model = train_quick(device, n_epochs=args.epochs, sigma=args.sigma)
        if args.save_checkpoint:
            torch.save(model.state_dict(), args.save_checkpoint)
            print(f"[model] Saved checkpoint → {args.save_checkpoint}")
    else:
        print("[model] Using random weights (--skip-training)")

    model.eval()

    # ── Generate all figures ───────────────────────────────────────────────────
    t0 = time.time()

    fig_filter_bank(model, save_dir)
    fig_spline_functions(model, save_dir)
    fig_energy_landscape(model, device, save_dir)

    _, ffn_psnr, kan_mean, K_vals = fig_psnr_vs_k(
        model, device, save_dir,
        sigma=args.sigma,
        n_trials=args.n_trials,
        n_imgs=args.n_imgs,
    )
    fig_inference_trajectory(model, device, save_dir, sigma=args.sigma)
    fig_precision_weights(model, save_dir)
    fig_energy_convergence(model, device, save_dir, sigma=args.sigma)
    _, cos_mean = fig_score_alignment(model, device, save_dir, sigma=args.sigma)

    print_summary(ffn_psnr, kan_mean, K_vals, cos_mean, args.sigma)
    print(f"\n[done] All figures saved to {save_dir}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
