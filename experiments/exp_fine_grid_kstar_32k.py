"""
Fine-grid K* sweep for KAN-EBM 32K (matched parameters vs ConvMLP 35K).
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

from metrics import compute_ssim
from exp_cifar10 import KANEnergyModel
from exp_scaled_eval_cifar10 import load_kan

OUT_DIR = ROOT / 'outputs' / 'fine_grid_kstar'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def sequential_denoise_record(model, noisy, K_max=30, dt=0.05, dt_decay=0.97):
    states = []
    u = noisy.clone()
    step = dt
    for _ in range(K_max):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            if hasattr(model, 'energy_grad'):
                g = model.energy_grad(ui).detach().clamp(-1., 1.)
            else:
                E = model.energy(ui)
                E = E.sum() if E.ndim > 0 else E
                g = torch.autograd.grad(E, ui)[0].detach().clamp(-1., 1.)
        u = (u - step * g).detach()
        states.append(u.clone())
        step *= dt_decay
    return states


def evaluate_fine_grid(model, images_clean, sigma, K_max=30, dt=0.05, dt_decay=0.97):
    noisy = (images_clean + torch.randn_like(images_clean) * sigma).clamp(-1, 1)
    states = sequential_denoise_record(model, noisy, K_max, dt, dt_decay)
    per_step_psnr = []
    for k in range(K_max):
        denoised = states[k]
        mse = F.mse_loss(denoised, images_clean, reduction='none').mean(dim=[1, 2, 3])
        psnr = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
        per_step_psnr.append(psnr)
    per_step_psnr = np.stack(per_step_psnr, axis=0)
    return per_step_psnr


def bootstrap_ci_kstar(kstar_per_image, B=10000, rng_seed=20260502):
    n = len(kstar_per_image)
    rng = np.random.default_rng(rng_seed)
    boot_means = []
    for _ in range(B):
        sample = rng.choice(kstar_per_image, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    boot_means = np.array(boot_means)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def parametric_bootstrap_ci(sigmas, K_means, B=10000, rng_seed=20260502):
    a, b, r2 = fit_power_law(sigmas, K_means)
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    yh = a + b * xs
    resid = ys - yh
    resid_c = resid - resid.mean()
    rng = np.random.default_rng(rng_seed)
    boot_alphas = []
    for _ in range(B):
        e = rng.choice(resid_c, size=len(xs), replace=True)
        ys_boot = yh + e
        slope, _ = np.polyfit(xs, ys_boot, 1)
        boot_alphas.append(float(slope))
    boot_alphas = np.array(boot_alphas)
    return float(np.percentile(boot_alphas, 2.5)), float(np.percentile(boot_alphas, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_images', type=int, default=500)
    parser.add_argument('--K_max', type=int, default=30)
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")
    print(f"[config] n_images={args.n_images}, K_max={args.K_max}, sigmas={args.sigmas}, n_seeds={args.n_seeds}")

    # Load KAN-EBM 32K
    ckpt = ROOT / 'outputs' / 'finalization' / 'kstar_param_scaling' / 'kan_small.pt'
    model = load_kan(ckpt, device, n_filters=16, kan_hidden=[48, 16])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Loaded KAN-EBM 32K ({n_params:,} params)")

    # Load CIFAR-10 test set
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    images_all = torch.stack([test_ds[i][0] for i in range(min(args.n_images, len(test_ds)))])
    print(f"[data] Loaded {len(images_all)} test images")

    sigmas = args.sigmas
    K_max = args.K_max
    n_seeds = args.n_seeds
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
        kstar_lo, kstar_hi = bootstrap_ci_kstar(kstar_per_image)
        print(f"\nsigma={sigma:.2f}: K*={kstar_mean:.2f} +- {kstar_std:.2f}  [95% CI: {kstar_lo:.2f}, {kstar_hi:.2f}]")
        summary['per_sigma'][f'{sigma:.3f}'] = {
            'kstar_mean': kstar_mean,
            'kstar_std': kstar_std,
            'kstar_ci_lo': kstar_lo,
            'kstar_ci_hi': kstar_hi,
            'mean_psnr_per_K': mean_psnr.mean(axis=1).tolist(),
        }

    # Power-law fit
    K_vals = [summary['per_sigma'][f'{s:.3f}']['kstar_mean'] for s in sigmas]
    a, alpha, r2 = fit_power_law(sigmas, K_vals)
    C = math.exp(a)
    alpha_lo, alpha_hi = parametric_bootstrap_ci(sigmas, K_vals)
    print(f"\nPower-law fit: alpha={alpha:.4f}, C={C:.2f}, R^2={r2:.4f}")
    print(f"95% CI alpha: [{alpha_lo:.4f}, {alpha_hi:.4f}]")
    summary['power_law'] = {
        'alpha': alpha,
        'C': C,
        'R2': r2,
        'alpha_ci_lo': alpha_lo,
        'alpha_ci_hi': alpha_hi,
    }

    out_path = OUT_DIR / 'fine_grid_kstar_32k.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] Results saved to {out_path}")


if __name__ == '__main__':
    main()
