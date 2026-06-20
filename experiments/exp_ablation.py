"""
exp_ablation.py
===============
Ablation study isolating each component of KAN-EBM.

Run:
  py -3.12 experiments/exp_ablation.py [--quick] [--device cuda]

Ablation conditions (all trained with identical settings):

  A. KAN-EBM (full)          — spatial filters + KAN energy     [proposed]
  B. MLP-EBM-filtered        — spatial filters + MLP energy      [filter bank only]
  C. KAN-EBM-raw             — no filter bank, KAN on raw pixels [KAN only]
  D. FFN-DSM                 — FFN trained with DSM loss, tested at K=1 [fair baseline]
  E. KAN-EBM n_filters=8     — half filters                      [filter count ablation]
  F. KAN-EBM n_filters=32    — double filters
  G. KAN-EBM filter_size=3   — smaller receptive field
  H. KAN-EBM filter_size=7   — larger receptive field

Key questions answered:
  - Is the gain from KAN or from the filter bank? (compare A vs B vs C)
  - Is the gain from iterative inference or from DSM training? (compare A@K=1 vs D@K=1)
  - How does filter count affect performance? (E vs A vs F)
  - How does receptive field size matter? (G vs A vs H)
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Metrics
# =============================================================================

def psnr(clean, pred, data_range=2.0):
    pred = pred.clamp(-1, 1)
    mse = F.mse_loss(pred, clean).item()
    return 10.0 * math.log10(data_range ** 2 / mse) if mse > 1e-10 else 100.0


# =============================================================================
# Ablation models
# =============================================================================

class MLPEnergyFiltered(nn.Module):
    """
    Ablation B: Spatial filter bank + MLP energy.
    Same filter bank as KAN-EBM but MLP energy instead of KAN.
    Tests whether the filter bank alone explains the gain.
    """
    def __init__(self, n_channels=1, n_filters=16, filter_size=5,
                 height=28, width=28, hidden=256):
        super().__init__()
        # Same filter bank as ThermodynamicField
        self.spatial_filters = nn.ParameterList([
            nn.Parameter(torch.randn(n_filters, 1, filter_size, filter_size) * 0.05)
            for _ in range(n_channels)
        ])
        self.n_filters   = n_filters
        self.filter_size = filter_size
        feat_dim = n_channels * n_filters
        self.energy_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def compute_energy(self, u):
        B, C, H, W = u.shape
        pad = self.filter_size // 2
        feats = []
        for c in range(C):
            f = F.conv2d(u[:, c:c+1], self.spatial_filters[c], padding=pad)
            feats.append(f.mean(dim=[-2, -1]))  # (B, n_filters)
        feat = torch.cat(feats, dim=1)          # (B, C*n_filters)
        return self.energy_mlp(feat).squeeze(-1) # (B,)

    def _grad(self, u):
        with torch.enable_grad():
            u_in = u.detach().requires_grad_(True)
            E = self.compute_energy(u_in).sum()
            return torch.autograd.grad(E, u_in)[0].detach().clamp(-1, 1)

    def denoise(self, x_noisy, n_steps=10, dt=0.05):
        u = x_noisy.clone()
        for _ in range(n_steps):
            u = (u.detach() - dt * self._grad(u))
        return u

    def loss(self, x_clean, sigma):
        noise  = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        x_in = x_noisy.detach().requires_grad_(True)
        E    = self.compute_energy(x_in).sum()
        grad = torch.autograd.grad(E, x_in, create_graph=True)[0]
        pred = x_in - sigma**2 * grad
        return F.mse_loss(pred, x_clean) + 0.01 * (grad**2).mean()

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class KANEnergyRaw(nn.Module):
    """
    Ablation C: KAN energy on raw flattened pixels, no spatial filter bank.
    Tests whether KAN alone (without spatial structure) gives the gain.
    """
    def __init__(self, obs_dim=784, kan_hidden: List[int] = None):
        super().__init__()
        from kan import KAN
        if kan_hidden is None:
            kan_hidden = [256, 128]
        self.kan_energy = KAN(
            layers_hidden=[obs_dim] + kan_hidden + [1],
            grid_size=5, spline_order=3,
        )

    def compute_energy(self, u):
        return self.kan_energy(u.view(u.shape[0], -1)).squeeze(-1)

    def _grad(self, u):
        with torch.enable_grad():
            u_in = u.detach().requires_grad_(True)
            E = self.compute_energy(u_in).sum()
            return torch.autograd.grad(E, u_in)[0].detach().clamp(-1, 1)

    def denoise(self, x_noisy, n_steps=10, dt=0.05):
        u = x_noisy.clone()
        for _ in range(n_steps):
            u = (u.detach() - dt * self._grad(u))
        return u

    def loss(self, x_clean, sigma):
        noise   = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        x_in = x_noisy.detach().requires_grad_(True)
        E    = self.compute_energy(x_in).sum()
        grad = torch.autograd.grad(E, x_in, create_graph=True)[0]
        pred = x_in - sigma**2 * grad
        return F.mse_loss(pred, x_clean) + 0.01 * (grad**2).mean()

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class FFNwithDSM(nn.Module):
    """
    Ablation D: FFN trained with DSM loss (same as EBMs), tested at K=1.
    This is the fairest single-pass baseline — same loss, same architecture depth.
    If KAN-EBM K=1 beats this, the structured energy is doing real work.
    """
    def __init__(self, obs_dim=784, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1)).view(x.shape)

    def denoise(self, x_noisy, n_steps=1):
        # K=1: apply network as a one-step denoiser
        return self.forward(x_noisy)

    def loss(self, x_clean, sigma):
        # DSM: predict score * sigma^2, supervision from noise direction
        noise   = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        # Network predicts the denoised image (implicit score estimator)
        pred = self.forward(x_noisy)
        return F.mse_loss(pred, x_clean)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class KANEBMConfigured(nn.Module):
    """KAN-EBM with configurable n_filters and filter_size for ablations E-H."""
    def __init__(self, n_filters=16, filter_size=5, kan_hidden=None,
                 n_channels=1, height=28, width=28):
        super().__init__()
        from predictive_coding_field import PredictiveCodingField
        if kan_hidden is None:
            kan_hidden = [64, 32]
        self.field = PredictiveCodingField(
            n_channels=n_channels, n_filters=n_filters,
            filter_size=filter_size, kan_hidden=kan_hidden,
            height=height, width=width,
        )
        self.config = {'n_filters': n_filters, 'filter_size': filter_size}

    def denoise(self, x_noisy, n_steps=10, dt=0.05):
        u = x_noisy.clone()
        for _ in range(n_steps):
            with torch.enable_grad():
                u_in = u.detach().requires_grad_(True)
                E = self.field.compute_energy(u_in).sum()
                grad = torch.autograd.grad(E, u_in)[0].detach().clamp(-1, 1)
            u = u.detach() - dt * grad
        return u

    def loss(self, x_clean, sigma):
        noise = torch.randn_like(x_clean) * sigma
        return self.field.denoising_loss(x_clean, x_clean + noise)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


# =============================================================================
# Data and training
# =============================================================================

def get_mnist_tensors(quick=False):
    try:
        import torchvision, torchvision.transforms as T
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        n = 5000 if quick else 50000
        ds = torchvision.datasets.MNIST(ROOT/'data', train=True, download=True, transform=tf)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, range(n)), batch_size=512, shuffle=False)
        xs = [b[0] for b in loader]
        train = torch.cat(xs, 0)
        ds_t = torchvision.datasets.MNIST(ROOT/'data', train=False, download=True, transform=tf)
        loader_t = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds_t, range(500 if quick else 2000)),
            batch_size=256, shuffle=False)
        xs_t = [b[0] for b in loader_t]
        test = torch.cat(xs_t, 0)
        print(f"  MNIST: {train.shape[0]} train / {test.shape[0]} test")
        return train, test
    except Exception as e:
        print(f"  MNIST unavailable ({e}), using synthetic")
        return torch.randn(5000,1,28,28), torch.randn(500,1,28,28)


def train_model(model, train_data, device, n_epochs, sigma, batch_size=128, name=''):
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    N     = train_data.shape[0]
    for ep in range(1, n_epochs+1):
        idx = torch.randperm(N)
        ep_loss, nb = 0.0, 0
        for i in range(0, N, batch_size):
            x = train_data[idx[i:i+batch_size]].to(device)
            opt.zero_grad()
            loss = model.loss(x, sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        if ep % max(1, n_epochs//3) == 0:
            print(f"    [{name:20s}] ep {ep:3d}/{n_epochs}  "
                  f"loss={ep_loss/nb:.5f}  params={model.n_params:,}")


@torch.no_grad()
def evaluate(model, test_data, device, sigma, k_list):
    model.eval()
    results = {}
    for k in k_list:
        psnrs = []
        for i in range(0, test_data.shape[0], 128):
            x_clean = test_data[i:i+128].to(device)
            x_noisy = x_clean + torch.randn_like(x_clean) * sigma
            try:
                x_pred = model.denoise(x_noisy, n_steps=k)
            except TypeError:
                x_pred = model.denoise(x_noisy)
            psnrs.append(psnr(x_clean, x_pred))
        results[k] = float(np.mean(psnrs))
    return results


# =============================================================================
# Main ablation runner
# =============================================================================

def run_ablation(quick=False, device_str='auto', n_epochs_override=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if device_str == 'auto' else torch.device(device_str)

    n_epochs = n_epochs_override if n_epochs_override is not None \
               else (5 if quick else 30)
    print(f"Device: {device}  |  epochs: {n_epochs}")

    sigma    = 0.2
    K_LIST   = [1, 5, 10, 20]
    out_dir  = ROOT / 'outputs' / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading MNIST...")
    train_data, test_data = get_mnist_tensors(quick=quick)

    # ── Define all ablation conditions ──────────────────────────────────
    conditions = {
        # name → (model_constructor, description)
        'A_KAN-EBM-full':     (
            lambda: KANEBMConfigured(n_filters=16, filter_size=5),
            'Proposed: spatial filters (5×5, ×16) + KAN energy'),
        'B_MLP-EBM-filtered': (
            lambda: MLPEnergyFiltered(n_filters=16, filter_size=5),
            'Ablation: same filter bank, MLP energy instead of KAN'),
        'C_KAN-EBM-raw':      (
            lambda: KANEnergyRaw(obs_dim=784, kan_hidden=[256, 128]),
            'Ablation: KAN on raw pixels, no spatial filter bank'),
        'D_FFN-DSM':          (
            lambda: FFNwithDSM(obs_dim=784, hidden=512),
            'Fair baseline: FFN trained with DSM, K=1 (no iteration)'),
        'E_KAN-EBM-8filt':   (
            lambda: KANEBMConfigured(n_filters=8,  filter_size=5),
            'Ablation: n_filters=8 (half)'),
        'F_KAN-EBM-32filt':  (
            lambda: KANEBMConfigured(n_filters=32, filter_size=5),
            'Ablation: n_filters=32 (double)'),
        'G_KAN-EBM-filt3':   (
            lambda: KANEBMConfigured(n_filters=16, filter_size=3),
            'Ablation: filter_size=3 (smaller receptive field)'),
        'H_KAN-EBM-filt7':   (
            lambda: KANEBMConfigured(n_filters=16, filter_size=7),
            'Ablation: filter_size=7 (larger receptive field)'),
    }
    if quick:
        # Subset for fast run
        conditions = {k: v for k, v in conditions.items()
                      if k in ('A_KAN-EBM-full', 'B_MLP-EBM-filtered',
                                'C_KAN-EBM-raw', 'D_FFN-DSM')}

    results = {}
    t0 = time.time()

    for name, (constructor, desc) in conditions.items():
        print(f"\n{'─'*60}")
        print(f"  [{name}]")
        print(f"  {desc}")
        model = constructor().to(device)
        train_model(model, train_data, device, n_epochs, sigma, name=name[:20])
        res = evaluate(model, test_data, device, sigma, K_LIST)
        results[name] = {'desc': desc, 'params': model.n_params, 'psnr': res}
        print(f"  Params: {model.n_params:,}")
        for k, v in res.items():
            print(f"  PSNR@K={k:>2}: {v:.2f} dB")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Condition':<25}  {'Params':>8}  " + \
             "  ".join(f"K={k:>2}" for k in K_LIST)
    print(header)
    print("─" * len(header))
    for name, res in results.items():
        psnrs = "  ".join(f"{res['psnr'][k]:>6.2f}" for k in K_LIST)
        print(f"{name:<25}  {res['params']:>8,}  {psnrs}")

    # ── LaTeX ablation table ─────────────────────────────────────────────
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study. MNIST denoising PSNR (dB) at $\sigma=0.2$. "
        r"Condition A is the proposed KAN-EBM. "
        r"B isolates the filter bank contribution. "
        r"C isolates the KAN contribution. "
        r"D is the fairest single-pass baseline (same DSM loss). "
        r"E-H vary filter bank hyperparameters.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lc" + "c"*len(K_LIST) + r"}",
        r"\toprule",
        f"Condition & Params & " + " & ".join(f"K={k}" for k in K_LIST) + r" \\",
        r"\midrule",
    ]
    for name, res in results.items():
        short = name.split('_', 1)[1].replace('-', r'\text{-}')
        psnr_cells = " & ".join(f"{res['psnr'][k]:.2f}" for k in K_LIST)
        lines.append(f"{short} & {res['params']:,} & {psnr_cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / 'table_ablation.txt').write_text('\n'.join(lines))

    # ── Save JSON ────────────────────────────────────────────────────────
    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump({'results': results, 'sigma': sigma, 'K_LIST': K_LIST,
                   'n_epochs': n_epochs, 'elapsed': time.time()-t0}, f, indent=2)
    print(f"\nSaved to {out_dir}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',  action='store_true',
                        help='5-epoch fast run (overridden by --epochs)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count (default: 5 if --quick, else 30)')
    args = parser.parse_args()
    run_ablation(quick=args.quick, device_str=args.device,
                 n_epochs_override=args.epochs)
