"""
rebuttal_smooth_mlp.py
======================
Wave 2: The bombshell experiment.

Reviewer R2: "MLP-EBM uses piecewise-linear activations, which is the worst
possible choice for an iterative-inference energy. A GELU or SiLU MLP would
be a far more informative ablation -- the paper's central mechanistic claim
(B-spline smoothness is the cause) requires ruling out activation smoothness
as an equally effective substitute."

This script trains three matched-architecture MLP-EBM variants:
  1. ConvMLP-EBM-GELU  (smooth, C^infty)
  2. ConvMLP-EBM-SiLU  (smooth, C^infty)
  3. ConvMLP-EBM-Tanh  (smooth, C^infty)

All share the SAME conv backbone as KAN-EBM (16 filters 5x5, learnable
precisions) and have ~32K params each (matched to KAN-EBM-small). The ONLY
difference vs KAN-EBM is the head: B-spline KAN [48,48,16,1] is replaced by
smooth-MLP [48,160,160,1] with the chosen activation.

After training each, we run the standard K* protocol (5 sigmas, 3 seeds)
and fit the K*(sigma) power law.  If alpha approx 1.5 with R^2 > 0.9 for
ANY of the smooth-MLP variants, the central claim collapses and we must
reframe.  If they all give MLP-style (alpha approx 0, K*=1, R^2 < 0.5),
the smoothness-as-cause hypothesis is ruled out and the paper's claim is
isolated to KAN-specific structure (basis functions per edge).

Outputs:
  outputs/finalization/rebuttal/smooth_mlp_ablation.json
  outputs/finalization/rebuttal/smooth_mlp_ablation.png
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_cifar10 import KANEnergyModel as KAN_CIFAR  # noqa: E402

OUT_DIR = ROOT / 'outputs' / 'finalization' / 'rebuttal'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Architecture: same conv backbone as KAN-EBM, smooth-MLP head with matched params
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATIONS = {
    'gelu':  nn.GELU,
    'silu':  nn.SiLU,
    'tanh':  nn.Tanh,
    'relu':  nn.ReLU,   # also test ReLU as a "rough" control
}


class ConvSmoothMLPEBM(nn.Module):
    """
    Same per-pixel conv backbone as KANEnergyModel, but the per-pixel head
    is a smooth MLP instead of a B-spline KAN. Matched parameter count to
    KAN-EBM-small (~32K params) at hidden=160.
    """
    def __init__(self, n_filters=16, filter_size=5, mlp_hidden=160,
                 n_channels=3, activation='gelu'):
        super().__init__()
        self.n_filters   = n_filters
        self.filter_size = filter_size
        self.n_channels  = n_channels
        self.activation_name = activation
        # Same convolutional filter bank as KAN-EBM
        self.filters = nn.Parameter(
            0.01 * torch.randn(n_filters, n_channels, filter_size, filter_size))
        self.log_precision = nn.Parameter(torch.zeros(n_filters))

        # Per-pixel smooth-MLP head: [48, h, h, 1]
        in_dim = n_channels * n_filters
        Act = ACTIVATIONS[activation.lower()]
        self.head = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), Act(),
            nn.Linear(mlp_hidden, mlp_hidden), Act(),
            nn.Linear(mlp_hidden, 1),
        )

    @property
    def precision(self):
        return F.softplus(self.log_precision)

    def extract_features(self, x):
        B, C, H, W = x.shape
        pad = self.filter_size // 2
        feats = []
        for c in range(C):
            feats.append(F.conv2d(x[:, c:c+1], self.filters[:, c:c+1], padding=pad))
        feats = torch.cat(feats, dim=1)
        prec = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = torch.tanh(feats * prec)
        return feats.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

    def energy(self, x):
        return self.head(self.extract_features(x)).sum()

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi = xn.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.energy(xi), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma ** 2 * grad, x_clean)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# K* sweep helper (matches existing protocol)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_kstar_seeded(model, batch_clean, sigmas, K_list, device,
                       n_seeds=3, dt=0.05, dt_decay=0.97):
    out = {s: [] for s in sigmas}
    psnr_table = {s: {K: [] for K in K_list} for s in sigmas}
    for seed in range(n_seeds):
        torch.manual_seed(SEED + seed * 11)
        for sigma in sigmas:
            xc = batch_clean.to(device)
            xn = xc + torch.randn_like(xc) * sigma
            psnrs = {}
            for K in K_list:
                with torch.enable_grad():
                    u = xn.clone()
                    step = dt
                    for _ in range(K):
                        ui = u.detach().requires_grad_(True)
                        E = model.energy(ui)
                        E = E.sum() if E.ndim > 0 else E
                        g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
                        u = (u - step * g).detach()
                        step *= dt_decay
                mse = F.mse_loss(u.clamp(-1, 1), xc).item()
                psnrs[K] = 100.0 if mse < 1e-10 else 10 * math.log10(4.0 / mse)
                psnr_table[sigma][K].append(psnrs[K])
            kstar = max(psnrs.items(), key=lambda kv: kv[1])[0]
            out[sigma].append(kstar)
    return out, psnr_table


def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one(model, train_loader, device, n_epochs, sigma_train, tag, max_seconds=None):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    losses = []
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = model.loss(x, sigma_train)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = model.loss(x, sigma_train)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        avg = ep_loss / max(1, nb)
        losses.append(avg)
        elapsed = time.time() - t0
        if ep <= 3 or ep % 5 == 0 or ep == n_epochs:
            print(f"  [{tag}] ep {ep:>3d}/{n_epochs}  loss={avg:.5f}  elapsed={elapsed:.1f}s",
                  flush=True)
        if max_seconds is not None and elapsed > max_seconds:
            print(f"  [{tag}] wall-clock cap reached at ep {ep}", flush=True)
            break
    return losses, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40,
                   help='Training epochs per smooth-MLP variant (default 40)')
    p.add_argument('--n_train', type=int, default=20000,
                   help='CIFAR train images (default 20000)')
    p.add_argument('--n_eval', type=int, default=24,
                   help='Eval images for K* (default 24)')
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--max_seconds_per_model', type=float, default=2400.0,
                   help='Wall-clock cap per smooth-MLP training (default 40 min)')
    p.add_argument('--variants', nargs='+',
                   default=['gelu', 'silu', 'tanh', 'relu'],
                   help='Activation variants to train')
    p.add_argument('--quick', action='store_true',
                   help='Smoke test: 3 epochs, 1500 train images, 8 eval, 1 seed')
    args = p.parse_args()

    if args.quick:
        args.epochs = 3
        args.n_train = 1500
        args.n_eval = 8
        args.n_seeds = 1
        args.variants = ['gelu']
        args.max_seconds_per_model = 600.0

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")
    if device.type == 'cuda':
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # ─── Data ─────────────────────────────────────────────────────────────────
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_ds = datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds  = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    train_subset = torch.utils.data.Subset(train_ds, list(range(args.n_train)))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128,
                                               shuffle=True, num_workers=0)
    eval_batch = torch.stack([test_ds[i][0] for i in range(args.n_eval)]).to(device)
    print(f"[data] train n={args.n_train}  eval n={args.n_eval}")

    # ─── K* sweep settings ────────────────────────────────────────────────────
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    K_list = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    record = {
        'config': {
            'epochs': args.epochs, 'n_train': args.n_train,
            'n_eval': args.n_eval, 'n_seeds': args.n_seeds,
            'sigmas': sigmas, 'K_list': K_list,
            'mlp_hidden': 160, 'sigma_train': 0.15,
        },
        'variants': {},
    }

    for variant in args.variants:
        print(f"\n{'='*70}")
        print(f"Variant: ConvMLP-EBM-{variant.upper()}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        model = ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                  n_channels=3, activation=variant).to(device)
        n_params = model.n_params
        print(f"  Params: {n_params:,}")

        # Train
        losses, train_seconds = train_one(model, train_loader, device,
                                          n_epochs=args.epochs, sigma_train=0.15,
                                          tag=f"MLP-{variant}",
                                          max_seconds=args.max_seconds_per_model)
        ckpt_path = OUT_DIR / f'conv_mlp_{variant}.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved {ckpt_path}")

        # K* sweep
        print(f"  Running K* protocol  (sigmas={sigmas}  seeds={args.n_seeds})")
        kstar_seeds, psnr_per_K = sweep_kstar_seeded(model, eval_batch, sigmas,
                                                     K_list, device, n_seeds=args.n_seeds)
        kstar_means = {s: float(np.mean(kstar_seeds[s])) for s in sigmas}
        kstar_stds  = {s: float(np.std(kstar_seeds[s], ddof=1)
                                 if len(kstar_seeds[s]) > 1 else 0.0) for s in sigmas}
        unique_kstars = sorted(set(int(round(v)) for v in kstar_means.values()))
        if len(unique_kstars) >= 2:
            a, b, r2 = fit_power_law(sigmas, [max(1.0, kstar_means[s]) for s in sigmas])
            C = math.exp(a)
        else:
            a, b, r2 = float('nan'), float('nan'), float('nan')
            C = float('nan')
        print(f"  K* values per sigma:")
        for s in sigmas:
            print(f"    sigma={s}: K* = {kstar_means[s]:.2f} +- {kstar_stds[s]:.2f}  "
                  f"(seeds={kstar_seeds[s]})")
        print(f"  Power-law fit:  alpha = {b:.3f}   R^2 = {r2:.3f}   C = {C:.2f}")
        # Verdict line
        if not math.isnan(r2):
            if r2 > 0.9 and 1.0 < b < 2.0:
                verdict = "LAW REPRODUCED -- central claim WEAKENED"
            elif r2 > 0.5:
                verdict = "PARTIAL law"
            else:
                verdict = "NO law -- central claim PRESERVED"
        else:
            verdict = "DEGENERATE (constant K*) -- central claim PRESERVED"
        print(f"  Verdict: {verdict}")

        record['variants'][variant] = {
            'n_params': n_params,
            'train_seconds': train_seconds,
            'final_loss': losses[-1] if losses else None,
            'epochs_completed': len(losses),
            'kstar_per_seed': {str(s): kstar_seeds[s] for s in sigmas},
            'kstar_means': {str(s): kstar_means[s] for s in sigmas},
            'kstar_stds': {str(s): kstar_stds[s] for s in sigmas},
            'unique_kstars': unique_kstars,
            'fit': {'C': C, 'alpha': b, 'r_squared': r2},
            'psnr_per_K_avg': {str(s): {str(k): float(np.mean(psnr_per_K[s][k]))
                                        for k in K_list} for s in sigmas},
            'verdict': verdict,
        }

        # Save incrementally so we don't lose work on crash
        out_path = OUT_DIR / 'smooth_mlp_ablation.json'
        out_path.write_text(json.dumps(record, indent=2))
        print(f"  Wrote partial results to {out_path}")

    # ─── Cross-comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Smooth-activation MLP-EBM ablation summary")
    print("=" * 70)
    print(f"{'Activation':<10} {'Params':>8} {'alpha':>8} {'R^2':>6} {'K*(0.30)':>10}  {'Verdict'}")
    print("-" * 90)
    # Add KAN reference row from existing JSON
    try:
        kan_ref = json.loads((ROOT / 'outputs' / 'finalization' / 'kstar_validation' /
                              'kstar_validation.json').read_text())
        kr = kan_ref['experiments']['cifar_kan_seeded']
        print(f"{'KAN (ref)':<10} {32096:>8} {kr['fit']['alpha']:>8.3f} "
              f"{kr['fit']['r_squared']:>6.3f} {kr['mean']['0.3']:>10.1f}  "
              f"REFERENCE")
    except Exception as e:
        print(f"  (could not load KAN reference: {e})")
    for variant, r in record['variants'].items():
        a = r['fit']['alpha']
        r2 = r['fit']['r_squared']
        k30 = r['kstar_means']['0.3']
        print(f"{variant:<10} {r['n_params']:>8} "
              f"{a if not math.isnan(a) else float('nan'):>8.3f} "
              f"{r2 if not math.isnan(r2) else float('nan'):>6.3f} "
              f"{k30:>10.1f}  {r['verdict']}")

    out_path = OUT_DIR / 'smooth_mlp_ablation.json'
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nFinal results: {out_path}")


if __name__ == "__main__":
    main()
