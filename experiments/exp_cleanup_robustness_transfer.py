"""
exp_cleanup_robustness_transfer.py
==================================
Re-runs the Section E claims (adversarial robustness + cross-domain transfer)
with a *properly-trained* steel-man MLP baseline.

The original exp_robustness.py and exp_transfer.py both train the
SmoothMLPEBM for only 3 mini-epochs on MNIST while the KAN-EBM was trained
to convergence. This stacks the comparison and produced misleading
"5.36 dB more resilient" / "26x parameter efficiency" claims.

This script:
  1. Trains a SmoothMLPEBM on MNIST for the same number of epochs as the
     KAN-EBM (default 30) with matched hyperparameters.
  2. Runs the FGSM robustness experiment with the fairly-trained MLP.
  3. Runs the Fashion-MNIST cross-domain transfer experiment with the
     same fairly-trained MLP.
  4. Saves an honest comparison JSON and a console report.

Design choices to make the comparison fair:
  - Same DSM training objective for both (KAN and MLP heads on a filter bank).
  - Same number of epochs.
  - Same noise level during training (sigma=0.3, matching the original).
  - Same inference dynamics (decaying step size, gradient clamp).
  - Steel-man MLP intentionally has MORE parameters (~10K-20K) than KAN
    so any KAN advantage cannot be dismissed as "more capacity wins".

Output: outputs/cleanup/robustness_and_transfer.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_cifar10 import KANEnergyModel  # noqa: E402
from exp_steel_man import SmoothMLPEBM  # noqa: E402

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def psnr(a, b, data_range=2.0):
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)


def refine(model, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
    """Inference loop matched to the main paper's KAN-EBM."""
    u = x_noisy.clone()
    step = dt
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            g = torch.autograd.grad(model.energy(ui).sum(), ui)[0].detach().clamp(-1, 1)
        u = (u.detach() - step * g).clamp(-1, 1)
        step *= dt_decay
    return u


def fgsm_attack(model, x, epsilon):
    """FGSM that maximizes energy (pushes off the data manifold)."""
    x_adv = x.clone().detach().requires_grad_(True)
    e = model.energy(x_adv).sum()
    g = torch.autograd.grad(e, x_adv)[0]
    x_adv = (x + epsilon * g.sign()).clamp(-1, 1)
    return x_adv.detach()


def train_mlp_baseline(model, device, train_loader, epochs, sigma, lr=1e-3,
                       log_every=1):
    """Full DSM training of the steel-man MLP-EBM."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    losses_per_epoch = []
    for ep in range(epochs):
        ep_losses = []
        t0 = time.time()
        for x, _ in train_loader:
            x = x.to(device)
            loss = model.loss(x, sigma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        avg = float(np.mean(ep_losses))
        losses_per_epoch.append(avg)
        if (ep + 1) % log_every == 0:
            print(f"  MLP epoch {ep+1:2d}/{epochs} | loss={avg:.5f} | "
                  f"{time.time()-t0:.1f}s")
    return losses_per_epoch


def run_robustness(kan, mlp, device, eps_list, K_list, n_test=200):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    test = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    out = {
        'eps_list': list(eps_list), 'K_list': list(K_list), 'n_test': n_test,
        'kan': {'attacked': [], 'refined': {str(k): [] for k in K_list}},
        'mlp': {'attacked': [], 'refined': {str(k): [] for k in K_list}},
    }

    for eps in eps_list:
        att_kan, att_mlp = [], []
        ref_kan = {k: [] for k in K_list}
        ref_mlp = {k: [] for k in K_list}
        seen = 0
        for x, _ in loader:
            if seen >= n_test:
                break
            x = x.to(device)
            x_kan = fgsm_attack(kan, x, eps) if eps > 0 else x
            x_mlp = fgsm_attack(mlp, x, eps) if eps > 0 else x
            att_kan.append(psnr(x_kan, x))
            att_mlp.append(psnr(x_mlp, x))
            for k in K_list:
                rk = refine(kan, x_kan, n_steps=k)
                rm = refine(mlp, x_mlp, n_steps=k)
                ref_kan[k].append(psnr(rk, x))
                ref_mlp[k].append(psnr(rm, x))
            seen += x.shape[0]
        out['kan']['attacked'].append(float(np.mean(att_kan)))
        out['mlp']['attacked'].append(float(np.mean(att_mlp)))
        for k in K_list:
            out['kan']['refined'][str(k)].append(float(np.mean(ref_kan[k])))
            out['mlp']['refined'][str(k)].append(float(np.mean(ref_mlp[k])))
        print(f"  eps={eps:.2f}: KAN att={out['kan']['attacked'][-1]:.2f} "
              f"-> ref(K=10)={out['kan']['refined']['10'][-1]:.2f} | "
              f"MLP att={out['mlp']['attacked'][-1]:.2f} "
              f"-> ref(K=10)={out['mlp']['refined']['10'][-1]:.2f}")
    return out


def run_transfer(kan, mlp, device, K_list, sigma=0.3, n_test=500):
    """Train on MNIST, evaluate denoising on Fashion-MNIST."""
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    fmnist = torchvision.datasets.FashionMNIST(ROOT / 'data', train=False,
                                               download=True, transform=tf)
    loader = torch.utils.data.DataLoader(fmnist, batch_size=32, shuffle=False)

    out = {'K_list': list(K_list), 'sigma': sigma, 'n_test': n_test,
           'kan': {str(k): [] for k in K_list},
           'mlp': {str(k): [] for k in K_list},
           'corrupted': []}

    seen = 0
    for x, _ in loader:
        if seen >= n_test:
            break
        x = x.to(device)
        xn = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
        out['corrupted'].append(psnr(xn, x))
        for k in K_list:
            if k == 0:
                rk = xn; rm = xn
            else:
                rk = refine(kan, xn, n_steps=k)
                rm = refine(mlp, xn, n_steps=k)
            out['kan'][str(k)].append(psnr(rk, x))
            out['mlp'][str(k)].append(psnr(rm, x))
        seen += x.shape[0]

    summary = {
        'K_list': out['K_list'], 'sigma': sigma, 'n_test': n_test,
        'corrupted_mean': float(np.mean(out['corrupted'])),
        'kan_mean_per_K': {k: float(np.mean(v)) for k, v in out['kan'].items()},
        'mlp_mean_per_K': {k: float(np.mean(v)) for k, v in out['mlp'].items()},
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30,
                    help='MLP training epochs (matched to KAN typical training)')
    ap.add_argument('--sigma', type=float, default=0.3,
                    help='Training noise level (matches existing KAN checkpoint)')
    ap.add_argument('--mlp_hidden', type=int, default=128)
    ap.add_argument('--n_test_robust', type=int, default=200)
    ap.add_argument('--n_test_transfer', type=int, default=500)
    args = ap.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    out_dir = ROOT / 'outputs' / 'cleanup'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load KAN-EBM ──
    kan_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
    kan = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    kan.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
    kan.eval()
    n_kan = sum(p.numel() for p in kan.parameters())
    print(f"Loaded KAN-EBM: {n_kan:,} params from {kan_path}")

    # ── Build + train SmoothMLPEBM with matched config ──
    mlp = SmoothMLPEBM(n_channels=1, n_filters=16, filter_size=5,
                       hidden_dim=args.mlp_hidden).to(device)
    n_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"Steel-man MLP-EBM: {n_mlp:,} params (hidden_dim={args.mlp_hidden})")
    print(f"  Param ratio MLP/KAN = {n_mlp/n_kan:.1f}x  -> MLP intentionally larger")

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    print(f"\nTraining steel-man MLP for {args.epochs} epochs at sigma={args.sigma}...")
    t0 = time.time()
    losses = train_mlp_baseline(mlp, device, train_loader, args.epochs, args.sigma)
    train_s = time.time() - t0
    print(f"  Training done in {train_s/60:.1f} min, final loss={losses[-1]:.5f}")

    mlp_ckpt = out_dir / f'smooth_mlp_ebm_e{args.epochs}_s{args.sigma}.pt'
    torch.save(mlp.state_dict(), mlp_ckpt)
    print(f"  Saved MLP checkpoint -> {mlp_ckpt}")
    mlp.eval()

    # ── Robustness ──
    print("\n=== Adversarial robustness (FGSM) ===")
    eps_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    K_list_rob = [1, 5, 10, 20]
    rob = run_robustness(kan, mlp, device, eps_list, K_list_rob, args.n_test_robust)

    # ── Transfer ──
    print("\n=== Cross-domain transfer (Fashion-MNIST) ===")
    K_list_tr = [0, 1, 5, 10, 20]
    tr = run_transfer(kan, mlp, device, K_list_tr,
                      sigma=args.sigma, n_test=args.n_test_transfer)
    print(f"  Corrupted: {tr['corrupted_mean']:.2f} dB")
    for k in K_list_tr:
        print(f"  K={k:>2d}: KAN {tr['kan_mean_per_K'][str(k)]:.2f} dB | "
              f"MLP {tr['mlp_mean_per_K'][str(k)]:.2f} dB")

    # ── Aggregate + save ──
    out = {
        'config': {
            'mlp_epochs': args.epochs, 'mlp_train_sigma': args.sigma,
            'mlp_hidden_dim': args.mlp_hidden,
            'kan_params': n_kan, 'mlp_params': n_mlp,
            'train_seconds': train_s,
            'final_mlp_loss': losses[-1],
            'mlp_ckpt': str(mlp_ckpt),
        },
        'robustness': rob,
        'transfer': tr,
    }
    out_path = out_dir / 'robustness_and_transfer.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")

    # ── Honest summary report ──
    print("\n" + "=" * 78)
    print("HONEST SUMMARY: Section E claims after fair-baseline retraining")
    print("=" * 78)

    # E1: refinement gain at eps=0.3, K=10
    if 0.3 in eps_list:
        idx = eps_list.index(0.3)
        kan_ref = rob['kan']['refined']['10'][idx]
        mlp_ref = rob['mlp']['refined']['10'][idx]
        kan_att = rob['kan']['attacked'][idx]
        mlp_att = rob['mlp']['attacked'][idx]
        print(f"E1 ROBUSTNESS (FGSM eps=0.3, K=10 refinement):")
        print(f"  KAN:  attacked={kan_att:.2f} -> refined={kan_ref:.2f} "
              f"(gain {kan_ref-kan_att:+.2f} dB)")
        print(f"  MLP:  attacked={mlp_att:.2f} -> refined={mlp_ref:.2f} "
              f"(gain {mlp_ref-mlp_att:+.2f} dB)")
        print(f"  ABSOLUTE refined-PSNR gap (KAN - MLP) = {kan_ref-mlp_ref:+.2f} dB")
        print(f"  (Refinement-gain difference is misleading because it conflates")
        print(f"   attack-PSNR start point. Absolute refined-PSNR is the honest metric.)")

    # E2: Fashion-MNIST K=10 absolute PSNR
    print(f"\nE2 TRANSFER (Fashion-MNIST denoising, sigma={args.sigma}, K=10):")
    print(f"  KAN K=10:  {tr['kan_mean_per_K']['10']:.2f} dB  ({n_kan:,} params)")
    print(f"  MLP K=10:  {tr['mlp_mean_per_K']['10']:.2f} dB  ({n_mlp:,} params)")
    delta = tr['kan_mean_per_K']['10'] - tr['mlp_mean_per_K']['10']
    print(f"  KAN advantage: {delta:+.2f} dB")
    if delta < 0:
        print("  *** KAN does NOT win this metric. The 'verified' Section E2 claim fails.")
    else:
        kan_g = tr['kan_mean_per_K']['10'] - tr['kan_mean_per_K']['0']
        mlp_g = tr['mlp_mean_per_K']['10'] - tr['mlp_mean_per_K']['0']
        print(f"  K=0->K=10 gain: KAN {kan_g:+.2f} | MLP {mlp_g:+.2f}")
    print("=" * 78)


if __name__ == '__main__':
    main()
