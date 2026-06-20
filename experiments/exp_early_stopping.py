"""
exp_early_stopping.py
=====================
Implements and evaluates an adaptive energy-based stopping criterion
for KAN-EBM inference. Transforms the "peak-and-degrade" pathology
from a bug into a feature: the model knows when to stop thinking.

The criterion: halt when |Delta E_t / E_t| < tau

Run:
  python experiments/exp_early_stopping.py --device cuda
"""

import sys
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def psnr(clean, pred, data_range=2.0):
    pred = pred.clamp(-1, 1)
    mse = F.mse_loss(pred, clean).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(data_range ** 2 / mse)


def denoise_with_energy_tracking(model, x_noisy, max_k=100, dt=0.1, decay=0.97):
    """Run gradient descent and track energy + PSNR at each step."""
    u = x_noisy.clone()
    energies = []
    trajectory = [u.detach().clone()]

    for k in range(max_k):
        with torch.enable_grad():
            u_in = u.detach().requires_grad_(True)
            E = model.energy(u_in)
            grad = torch.autograd.grad(E, u_in)[0].detach()

        energies.append(E.item())
        grad = grad.clamp(-1.0, 1.0)
        step_size = dt * (decay ** k)
        u = u.detach() - step_size * grad
        trajectory.append(u.detach().clone())

    return trajectory, energies


def find_adaptive_stop(energies, tau):
    """Find the step where relative energy change drops below tau."""
    for i in range(1, len(energies)):
        E_prev = abs(energies[i-1])
        if E_prev < 1e-10:
            continue
        rel_change = abs(energies[i] - energies[i-1]) / E_prev
        if rel_change < tau:
            return i
    return len(energies)  # never stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max_k', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    args = parser.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
             if args.device == 'auto' else torch.device(args.device)

    print(f"Device: {device}")

    out_dir = ROOT / 'outputs' / 'early_stopping'
    out_dir.mkdir(parents=True, exist_ok=True)

    from exp_cifar10 import KANEnergyModel as CifarKANEBM
    import torchvision
    import torchvision.transforms as T
    
    if args.dataset == 'cifar10':
        ckpt_path = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
        if not ckpt_path.exists():
            print(f"Error: No checkpoint found at {ckpt_path}")
            return
        model = CifarKANEBM(n_filters=32, filter_size=5, kan_hidden=[96, 16], n_channels=3).to(device)
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    else:
        ckpt_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
        if not ckpt_path.exists():
            ckpt_path = ROOT / 'results' / 'kan_ebm.pt'
        if not ckpt_path.exists():
            print(f"Error: No checkpoint found in results/")
            return
        model = CifarKANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        test_ds = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, range(200)),
        batch_size=1, shuffle=False
    )

    # ── Run inference with energy tracking ──
    print(f"\nRunning energy-tracked inference (max K={args.max_k}, sigma={args.sigma})...")

    tau_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    all_oracle_psnrs = []  # best PSNR at any K
    all_oracle_stops = []  # K that gives best PSNR
    tau_results = {tau: {'stops': [], 'psnrs': []} for tau in tau_values}

    n_images = 0
    for batch in test_loader:
        x_clean = batch[0].to(device)
        x_noisy = x_clean + torch.randn_like(x_clean) * args.sigma

        trajectory, energies = denoise_with_energy_tracking(
            model, x_noisy, max_k=args.max_k
        )

        # Compute PSNR at each step
        step_psnrs = [psnr(x_clean, t) for t in trajectory]

        # Oracle: best PSNR across all K
        best_k = int(np.argmax(step_psnrs))
        best_psnr = step_psnrs[best_k]
        all_oracle_psnrs.append(best_psnr)
        all_oracle_stops.append(best_k)

        # Adaptive stopping for each tau
        for tau in tau_values:
            stop_k = find_adaptive_stop(energies, tau)
            stop_k = min(stop_k, len(trajectory) - 1)
            stopped_psnr = step_psnrs[stop_k]
            tau_results[tau]['stops'].append(stop_k)
            tau_results[tau]['psnrs'].append(stopped_psnr)

        n_images += 1
        if n_images % 50 == 0:
            print(f"  Processed {n_images} images...")

    # ── Summarize results ──
    oracle_mean = np.mean(all_oracle_psnrs)
    oracle_k_mean = np.mean(all_oracle_stops)

    print(f"\n{'='*60}")
    print(f"  ADAPTIVE EARLY STOPPING RESULTS ({n_images} images)")
    print(f"{'='*60}")
    print(f"  Oracle (best K per image):  {oracle_mean:.2f} dB  (mean K*={oracle_k_mean:.1f})")
    print(f"")
    print(f"  {'tau':>8}  {'Mean PSNR':>10}  {'vs Oracle':>10}  {'Mean K_stop':>12}  {'Recovery':>10}")
    print(f"  {'-'*54}")

    summary = {}
    for tau in tau_values:
        mean_psnr = np.mean(tau_results[tau]['psnrs'])
        mean_stop = np.mean(tau_results[tau]['stops'])
        gap = oracle_mean - mean_psnr
        recovery = mean_psnr / oracle_mean * 100 if oracle_mean > 0 else 0
        print(f"  {tau:>8.3f}  {mean_psnr:>9.2f}  {gap:>+9.2f}  {mean_stop:>11.1f}  {recovery:>9.1f}%")
        summary[str(tau)] = {
            'mean_psnr': float(mean_psnr),
            'mean_stop_k': float(mean_stop),
            'gap_vs_oracle': float(gap),
            'recovery_pct': float(recovery)
        }

    # Save
    output = {
        'oracle_mean_psnr': float(oracle_mean),
        'oracle_mean_k': float(oracle_k_mean),
        'n_images': n_images,
        'sigma': args.sigma,
        'max_k': args.max_k,
        'tau_results': summary
    }
    (out_dir / 'early_stopping_results.json').write_text(json.dumps(output, indent=2))

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Recovery % vs tau
    taus = [float(t) for t in summary.keys()]
    recoveries = [summary[str(t)]['recovery_pct'] for t in taus]
    stops = [summary[str(t)]['mean_stop_k'] for t in taus]

    ax1.plot(taus, recoveries, 'o-', color='#4C6EF5', linewidth=2.5, markersize=8)
    ax1.axhline(y=100, color='#12B886', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Oracle (best K per image)')
    ax1.axhline(y=90, color='#FA5252', linestyle=':', linewidth=1.5, alpha=0.7,
                label='90% Recovery Threshold')
    ax1.set_xlabel('Stopping threshold tau', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PSNR Recovery (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Adaptive Stopping: Quality Recovery', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Mean K_stop vs tau
    ax2.plot(taus, stops, 's-', color='#E64980', linewidth=2.5, markersize=8)
    ax2.axhline(y=oracle_k_mean, color='#12B886', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Oracle mean K*={oracle_k_mean:.1f}')
    ax2.set_xlabel('Stopping threshold tau', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Stopping Step', fontsize=13, fontweight='bold')
    ax2.set_title('Adaptive Stopping: Compute Usage', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'early_stopping.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'early_stopping.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll outputs -> {out_dir}/")


if __name__ == '__main__':
    main()
