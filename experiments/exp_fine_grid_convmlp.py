"""
exp_fine_grid_convmlp.py
========================
Fine-grid K* sweep for ConvMLP-GELU on 500 CIFAR-10 images.

Tests whether the power-law scaling generalizes beyond KAN-EBM
to other conv-backbone architectures.
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_scaled_eval_cifar10 import load_conv_mlp

OUT_DIR = ROOT / 'outputs' / 'fine_grid_kstar'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def evaluate_fine_grid(model, images_clean, sigma, K_max=30, dt=0.05, dt_decay=0.97):
    noisy = (images_clean + torch.randn_like(images_clean) * sigma).clamp(-1, 1)
    per_step_psnr = []
    u = noisy.clone()
    step = dt
    for k in range(K_max):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            E = model.energy(ui)
            E = E.sum() if E.ndim > 0 else E
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1., 1.)
        u = (u - step * g).detach()
        mse = F.mse_loss(u.clamp(-1, 1), images_clean, reduction='none').mean(dim=[1, 2, 3])
        psnr = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
        per_step_psnr.append(psnr)
    return np.stack(per_step_psnr, axis=0)


def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_images', type=int, default=500)
    parser.add_argument('--K_max', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--out_name', type=str, default='fine_grid_convmlp_gelu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")

    ckpt = ROOT / 'outputs' / 'finalization' / 'rebuttal' / f'conv_mlp_{args.activation}.pt'
    model = load_conv_mlp(ckpt, device, args.activation)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ConvMLP-{args.activation.upper()} ({n_params:,} params)")

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    images_all = torch.stack([test_ds[i][0] for i in range(min(args.n_images, len(test_ds)))])
    print(f"[data] Loaded {len(images_all)} test images")

    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    K_max = args.K_max
    n_seeds = 3
    all_results = {}

    for seed_idx in range(n_seeds):
        torch.manual_seed(SEED + seed_idx * 100)
        np.random.seed(SEED + seed_idx * 100)
        print(f"\n=== Seed {seed_idx} ===")
        all_results[seed_idx] = {}
        for sigma in sigmas:
            print(f"  sigma={sigma:.2f} ...", end='', flush=True)
            t0 = time.time()
            per_step_psnr = []
            for b in range(0, len(images_all), args.batch_size):
                batch = images_all[b:b + args.batch_size].to(device)
                psnr = evaluate_fine_grid(model, batch, sigma, K_max)
                per_step_psnr.append(psnr)
            per_step_psnr = np.concatenate(per_step_psnr, axis=1)
            all_results[seed_idx][sigma] = per_step_psnr
            kstar_mean = int(np.argmax(per_step_psnr.mean(axis=1))) + 1
            print(f" done in {time.time()-t0:.1f}s | K*={kstar_mean}")

    # Aggregate
    summary = {'per_sigma': {}}
    for sigma in sigmas:
        stacked = np.stack([all_results[s][sigma] for s in range(n_seeds)], axis=0)
        mean_psnr = stacked.mean(axis=0)
        kstar_per_image = np.argmax(mean_psnr, axis=0) + 1
        kstar_mean = float(np.mean(kstar_per_image))
        kstar_std = float(np.std(kstar_per_image, ddof=1))
        print(f"\nsigma={sigma:.2f}: K*={kstar_mean:.2f} ± {kstar_std:.2f}")
        summary['per_sigma'][f'{sigma:.3f}'] = {
            'kstar_mean': kstar_mean,
            'kstar_std': kstar_std,
            'mean_psnr_per_K': mean_psnr.mean(axis=1).tolist(),
        }

    K_vals = [summary['per_sigma'][f'{s:.3f}']['kstar_mean'] for s in sigmas]
    a, alpha, r2 = fit_power_law(sigmas, K_vals)
    C = math.exp(a)
    print(f"\nPower-law fit: alpha={alpha:.3f}, C={C:.1f}, R^2={r2:.3f}")
    summary['power_law'] = {'alpha': alpha, 'C': C, 'R2': r2}

    out_path = OUT_DIR / f'{args.out_name}.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] Results saved to {out_path}")


if __name__ == '__main__':
    main()
