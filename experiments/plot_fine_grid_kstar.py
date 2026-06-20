"""
plot_fine_grid_kstar.py
=======================
Visualise fine-grid K* results.

Produces:
  - Fine-grid vs coarse-grid PSNR curves per sigma
  - K* distribution per sigma (histogram)
  - Power-law fit comparison (fine vs coarse)
  - Step-size schedule robustness plot
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'outputs' / 'fine_grid_kstar'
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data_path = OUT_DIR / 'fine_grid_kstar.json'
    if not data_path.exists():
        print(f"[error] Data not found: {data_path}")
        print("        Run exp_fine_grid_kstar.py first.")
        return

    with open(data_path) as f:
        data = json.load(f)

    sigmas = [float(s) for s in data['per_sigma'].keys()]
    K_max = data['K_max']
    fine_grid_K = list(range(1, K_max + 1))
    coarse_grid = [1, 2, 5, 10, 20]

    # ── Plot 1: PSNR curves per sigma (fine vs coarse) ────────────────────────
    fig, axes = plt.subplots(1, len(sigmas), figsize=(20, 3.5))
    for ax, sigma in zip(axes, sigmas):
        d = data['per_sigma'][f'{sigma:.3f}']
        mean_psnr = d['mean_psnr_per_K']
        ax.plot(fine_grid_K, mean_psnr, 'b-', linewidth=2, label='Fine grid')
        coarse_indices = [k - 1 for k in coarse_grid if k <= K_max]
        coarse_psnr = [mean_psnr[i] for i in coarse_indices]
        ax.plot(coarse_grid[:len(coarse_psnr)], coarse_psnr, 'ro', markersize=8, label='Coarse grid')
        kstar_fine = d['kstar_mean']
        kstar_coarse = d['kstar_coarse_mean']
        ax.axvline(kstar_fine, color='b', linestyle='--', alpha=0.5)
        ax.axvline(kstar_coarse, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'$\\sigma$={sigma:.2f}\nFine K*={kstar_fine:.1f}, Coarse K*={kstar_coarse:.1f}')
        ax.set_xlabel('K (inference steps)')
        ax.set_ylabel('PSNR (dB)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Fine-grid vs Coarse-grid PSNR curves (KAN-EBM, 500 images)', fontsize=12, y=1.05)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'psnr_curves_fine_vs_coarse.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'psnr_curves_fine_vs_coarse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved PSNR curves")

    # ── Plot 2: K* distribution per sigma ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(sigmas), figsize=(20, 3.5))
    for ax, sigma in zip(axes, sigmas):
        d = data['per_sigma'][f'{sigma:.3f}']
        kstars = d['kstar_per_image']
        ax.hist(kstars, bins=np.arange(0.5, K_max + 1.5, 1), color='steelblue', edgecolor='white')
        ax.axvline(d['kstar_mean'], color='r', linestyle='--', linewidth=2, label=f"Mean={d['kstar_mean']:.1f}")
        ax.axvline(d['kstar_median'], color='g', linestyle='--', linewidth=2, label=f"Median={d['kstar_median']:.1f}")
        ax.set_title(f'$\\sigma$={sigma:.2f}')
        ax.set_xlabel('K* (per image)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.set_xlim(0, K_max + 1)
    plt.suptitle('Per-image K* distribution (KAN-EBM, 500 images, 3 seeds)', fontsize=12, y=1.05)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'kstar_distribution.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'kstar_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved K* distribution")

    # ── Plot 3: Power-law fit comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))

    pl_fine = data['power_law']['fine_grid']
    pl_coarse = data['power_law']['coarse_grid']

    K_fine = [data['per_sigma'][f'{s:.3f}']['kstar_mean'] for s in sigmas]
    K_coarse = [data['per_sigma'][f'{s:.3f}']['kstar_coarse_mean'] for s in sigmas]

    ax.loglog(sigmas, K_fine, 'bo-', markersize=10, linewidth=2,
              label=f"Fine grid: $\\alpha$={pl_fine['alpha']:.2f}, $R^2$={pl_fine['R2']:.3f}")
    ax.loglog(sigmas, K_coarse, 'rs--', markersize=8, linewidth=2,
              label=f"Coarse grid: $\\alpha$={pl_coarse['alpha']:.2f}, $R^2$={pl_coarse['R2']:.3f}")

    # Fit lines
    s_fit = np.linspace(min(sigmas), max(sigmas), 100)
    ax.loglog(s_fit, pl_fine['C'] * s_fit ** pl_fine['alpha'], 'b-', alpha=0.3)
    ax.loglog(s_fit, pl_coarse['C'] * s_fit ** pl_coarse['alpha'], 'r-', alpha=0.3)

    ax.set_xlabel('$\\sigma$ (noise level)')
    ax.set_ylabel('$K^*$ (optimal steps)')
    ax.set_title('Power-law fit: Fine-grid vs Coarse-grid K*')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'powerlaw_fine_vs_coarse.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'powerlaw_fine_vs_coarse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved power-law comparison")

    # ── Plot 4: Step-size schedule robustness ─────────────────────────────────
    schedules = ['schedule_default', 'schedule_fast', 'schedule_slow']
    sched_labels = {'schedule_default': 'Default (dt=0.05, decay=0.97)',
                    'schedule_fast': 'Fast (dt=0.10, decay=0.95)',
                    'schedule_slow': 'Slow (dt=0.03, decay=0.98)'}

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'schedule_default': 'blue', 'schedule_fast': 'green', 'schedule_slow': 'orange'}

    for sched in schedules:
        if sched in data:
            K_vals = data[sched]['kstar_per_sigma']
            alpha = data[sched]['alpha']
            r2 = data[sched]['R2']
            ax.loglog(sigmas, K_vals, 'o-', color=colors[sched], markersize=8, linewidth=2,
                      label=f"{sched_labels[sched]}: $\\alpha$={alpha:.2f}, $R^2$={r2:.3f}")

    ax.set_xlabel('$\\sigma$ (noise level)')
    ax.set_ylabel('$K^*$ (optimal steps)')
    ax.set_title('Step-size schedule robustness')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'step_schedule_robustness.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'step_schedule_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved step-schedule robustness")

    print(f"\n[done] All plots saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
