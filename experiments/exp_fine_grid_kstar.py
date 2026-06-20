"""
exp_fine_grid_kstar.py
======================
Fine-grid K* sweep on 500 CIFAR-10 test images with per-image uncertainty.

Addresses reviewer concern: "The coarse step grid {1,2,5,10,20} may miss
the true K* and produce discretization artifacts."

What it does:
  1. Loads KAN-EBM (110K) checkpoint.
  2. For each sigma in {0.05, 0.10, 0.15, 0.20, 0.30} and 3 seeds:
     - Generates 500 noisy images.
     - Runs gradient descent for K_max=30 steps *sequentially*.
     - Records PSNR after every step (efficient: no recomputation).
  3. Computes K* per image (argmax PSNR) and aggregates:
     - mean/median K*, std, bootstrap 95% CI
  4. Compares fine-grid K* vs coarse-grid K*.
  5. Tests step-size schedule robustness (3 schedules).
  6. Fits power law K*(sigma) ~ C sigma^alpha with fine-grid data.

Run:
  py -3.12 experiments/exp_fine_grid_kstar.py --device cuda
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
from exp_scaled_eval_cifar10 import load_kan, load_conv_mlp

OUT_DIR = ROOT / 'outputs' / 'fine_grid_kstar'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def sequential_denoise_record(model, noisy, K_max=30, dt=0.05, dt_decay=0.97):
    """
    Run gradient descent for K_max steps, recording PSNR after each step.
    Returns: list of denoised states [after step 1, ..., after step K_max]
    """
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
    """
    Evaluate model on a batch of clean images at one noise level.
    Returns dict with per-step PSNR for each image.
    """
    noisy = (images_clean + torch.randn_like(images_clean) * sigma).clamp(-1, 1)
    states = sequential_denoise_record(model, noisy, K_max, dt, dt_decay)

    # Compute PSNR for each step
    per_step_psnr = []
    for k in range(K_max):
        denoised = states[k]
        mse = F.mse_loss(denoised, images_clean, reduction='none').mean(dim=[1, 2, 3])
        psnr = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
        per_step_psnr.append(psnr)

    # Shape: (K_max, n_images)
    per_step_psnr = np.stack(per_step_psnr, axis=0)
    return per_step_psnr


def bootstrap_ci_kstar(kstar_per_image, B=10000, rng_seed=20260502):
    """Bootstrap 95% CI for mean K*."""
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

    # Load KAN-EBM
    ckpt = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
    model = load_kan(ckpt, device, n_filters=32, kan_hidden=[96, 16])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Loaded KAN-EBM ({n_params:,} params)")

    # Load CIFAR-10 test set
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    images_all = torch.stack([test_ds[i][0] for i in range(min(args.n_images, len(test_ds)))])
    print(f"[data] Loaded {len(images_all)} test images")

    sigmas = args.sigmas
    K_max = args.K_max
    n_seeds = args.n_seeds

    # ------------------------------------------------------------------
    # Main fine-grid sweep
    # ------------------------------------------------------------------
    all_results = {}
    coarse_grid = [1, 2, 5, 10, 20]

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

            # Concatenate batches: shape (K_max, n_images)
            per_step_psnr = np.concatenate(per_step_psnr, axis=1)
            all_results[seed_idx][sigma] = per_step_psnr

            # Quick summary
            mean_psnr = per_step_psnr.mean(axis=1)
            kstar_mean = int(np.argmax(mean_psnr)) + 1  # 1-indexed
            print(f" done in {time.time()-t0:.1f}s | K* (mean PSNR)={kstar_mean}")

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINE-GRID K* ANALYSIS")
    print("=" * 70)

    summary = {
        'n_images': len(images_all),
        'K_max': K_max,
        'sigmas': sigmas,
        'n_seeds': n_seeds,
        'n_params': n_params,
        'per_sigma': {},
    }

    fine_grid_K_list = list(range(1, K_max + 1))

    for sigma in sigmas:
        # Average PSNR across seeds for each (K, image)
        # Shape: (n_seeds, K_max, n_images)
        stacked = np.stack([all_results[s][sigma] for s in range(n_seeds)], axis=0)
        mean_psnr_per_K = stacked.mean(axis=0)  # (K_max, n_images)

        # K* per image: argmax over K (1-indexed)
        kstar_per_image = np.argmax(mean_psnr_per_K, axis=0) + 1

        # K* from mean PSNR (old method)
        kstar_from_mean = int(np.argmax(mean_psnr_per_K.mean(axis=1))) + 1

        # Aggregate statistics
        kstar_mean = float(np.mean(kstar_per_image))
        kstar_median = float(np.median(kstar_per_image))
        kstar_std = float(np.std(kstar_per_image, ddof=1))
        ci_lo, ci_hi = bootstrap_ci_kstar(kstar_per_image, B=10000)

        # Coarse-grid comparison
        coarse_indices = [k - 1 for k in coarse_grid if k <= K_max]
        coarse_psnr = mean_psnr_per_K[coarse_indices, :]
        kstar_coarse_per_image = np.array([coarse_grid[i] for i in np.argmax(coarse_psnr, axis=0)])
        kstar_coarse_mean = float(np.mean(kstar_coarse_per_image))
        kstar_coarse_std = float(np.std(kstar_coarse_per_image, ddof=1))

        print(f"\nsigma={sigma:.2f}")
        print(f"  Fine-grid K*:  mean={kstar_mean:.2f}, median={kstar_median:.1f}, std={kstar_std:.2f}")
        print(f"  Fine-grid 95% CI for mean K*: [{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"  K* from mean PSNR (old method): {kstar_from_mean}")
        print(f"  Coarse-grid K*: mean={kstar_coarse_mean:.2f}, std={kstar_coarse_std:.2f}")
        print(f"  Difference (coarse - fine): {kstar_coarse_mean - kstar_mean:.2f}")

        summary['per_sigma'][f'{sigma:.3f}'] = {
            'kstar_per_image': kstar_per_image.tolist(),
            'kstar_mean': kstar_mean,
            'kstar_median': kstar_median,
            'kstar_std': kstar_std,
            'kstar_ci95': [ci_lo, ci_hi],
            'kstar_from_mean_psnr': kstar_from_mean,
            'kstar_coarse_mean': kstar_coarse_mean,
            'kstar_coarse_std': kstar_coarse_std,
            'mean_psnr_per_K': mean_psnr_per_K.mean(axis=1).tolist(),
        }

    # ------------------------------------------------------------------
    # Power-law fit on fine-grid K*
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("POWER-LAW FIT COMPARISON")
    print("=" * 70)

    # Fine-grid fit (using per-image mean K*)
    K_fine = [summary['per_sigma'][f'{s:.3f}']['kstar_mean'] for s in sigmas]
    a_fine, alpha_fine, r2_fine = fit_power_law(sigmas, K_fine)
    C_fine = math.exp(a_fine)
    ci_fine_lo, ci_fine_hi = parametric_bootstrap_ci(sigmas, K_fine)

    # Coarse-grid fit
    K_coarse = [summary['per_sigma'][f'{s:.3f}']['kstar_coarse_mean'] for s in sigmas]
    a_coarse, alpha_coarse, r2_coarse = fit_power_law(sigmas, K_coarse)
    C_coarse = math.exp(a_coarse)
    ci_coarse_lo, ci_coarse_hi = parametric_bootstrap_ci(sigmas, K_coarse)

    print(f"\nFine-grid fit:")
    print(f"  alpha = {alpha_fine:.4f}, C = {C_fine:.2f}, R^2 = {r2_fine:.4f}")
    print(f"  95% CI on alpha: [{ci_fine_lo:.3f}, {ci_fine_hi:.3f}]")

    print(f"\nCoarse-grid fit:")
    print(f"  alpha = {alpha_coarse:.4f}, C = {C_coarse:.2f}, R^2 = {r2_coarse:.4f}")
    print(f"  95% CI on alpha: [{ci_coarse_lo:.3f}, {ci_coarse_hi:.3f}]")

    print(f"\nDelta alpha (coarse - fine): {alpha_coarse - alpha_fine:.4f}")
    print(f"Coarse-grid bias in K*: {[K_coarse[i] - K_fine[i] for i in range(len(sigmas))]}")

    summary['power_law'] = {
        'fine_grid': {'alpha': alpha_fine, 'C': C_fine, 'R2': r2_fine,
                      'ci95_alpha': [ci_fine_lo, ci_fine_hi]},
        'coarse_grid': {'alpha': alpha_coarse, 'C': C_coarse, 'R2': r2_coarse,
                        'ci95_alpha': [ci_coarse_lo, ci_coarse_hi]},
    }

    # ------------------------------------------------------------------
    # Step-size schedule robustness
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP-SIZE SCHEDULE ROBUSTNESS")
    print("=" * 70)

    schedules = [
        ('default', 0.05, 0.97),
        ('fast', 0.10, 0.95),
        ('slow', 0.03, 0.98),
    ]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for sched_name, dt, dt_decay in schedules:
        print(f"\nSchedule: {sched_name} (dt={dt}, decay={dt_decay})")
        kstars = []
        for sigma in sigmas:
            per_step_psnr = []
            for b in range(0, len(images_all), args.batch_size):
                batch = images_all[b:b + args.batch_size].to(device)
                psnr = evaluate_fine_grid(model, batch, sigma, K_max, dt, dt_decay)
                per_step_psnr.append(psnr)
            per_step_psnr = np.concatenate(per_step_psnr, axis=1)
            kstar_per_image = np.argmax(per_step_psnr, axis=0) + 1
            kstars.append(float(np.mean(kstar_per_image)))
            print(f"  sigma={sigma:.2f}: K*={kstars[-1]:.2f}")

        a_s, alpha_s, r2_s = fit_power_law(sigmas, kstars)
        print(f"  Fit: alpha={alpha_s:.3f}, R^2={r2_s:.3f}")
        summary[f'schedule_{sched_name}'] = {
            'dt': dt, 'dt_decay': dt_decay,
            'kstar_per_sigma': kstars,
            'alpha': alpha_s, 'R2': r2_s,
        }

    # Save results
    out_path = OUT_DIR / 'fine_grid_kstar.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] Saved results to {out_path}")

    # Print summary text
    text = f"""Fine-Grid K* Sweep (500 images, K_max={K_max})
{'='*60}
Fine-grid fit:   alpha = {alpha_fine:.3f}  C = {C_fine:.1f}  R^2 = {r2_fine:.4f}
                 95% CI: [{ci_fine_lo:.3f}, {ci_fine_hi:.3f}]
Coarse-grid fit: alpha = {alpha_coarse:.3f}  C = {C_coarse:.1f}  R^2 = {r2_coarse:.4f}
                 95% CI: [{ci_coarse_lo:.3f}, {ci_coarse_hi:.3f}]
Delta alpha = {alpha_coarse - alpha_fine:.4f}

Per-sigma K* (fine grid, mean ± std):
"""
    for sigma in sigmas:
        d = summary['per_sigma'][f'{sigma:.3f}']
        text += f"  sigma={sigma:.2f}: K* = {d['kstar_mean']:.2f} ± {d['kstar_std']:.2f}  (median={d['kstar_median']:.1f}, CI=[{d['kstar_ci95'][0]:.2f}, {d['kstar_ci95'][1]:.2f}])\n"

    text += "\nStep-size robustness:\n"
    for sched_name, dt, dt_decay in schedules:
        s = summary[f'schedule_{sched_name}']
        text += f"  {sched_name} (dt={dt}, decay={dt_decay}): alpha={s['alpha']:.3f}, R^2={s['R2']:.3f}\n"

    (OUT_DIR / 'fine_grid_kstar_summary.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
