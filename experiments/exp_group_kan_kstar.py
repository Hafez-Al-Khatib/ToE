# experiments/exp_group_kan_kstar.py
"""Fine-grid K* sweep for GroupKAN-32K -> alpha for the paper's
architecture table. Protocol identical to exp_fine_grid_kstar.py
(5 sigmas, 3 seeds, 500 images, K_max=30).

Run: py -3.12 experiments/exp_group_kan_kstar.py --device cuda
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from group_kan import GroupKANEnergyModel
from exp_fine_grid_kstar import (evaluate_fine_grid, fit_power_law,
                                 parametric_bootstrap_ci, bootstrap_ci_kstar)

OUT = ROOT / 'outputs' / 'group_kan'
SEED = 42
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--hidden', type=int, default=640)
    p.add_argument('--tag', default='group_kan_32k')
    p.add_argument('--n_images', type=int, default=500)
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--k_max', type=int, default=30)
    p.add_argument('--batch', type=int, default=100)
    args = p.parse_args()
    device = torch.device(args.device)

    model = GroupKANEnergyModel(hidden=args.hidden).to(device)
    model.load_state_dict(torch.load(OUT / f"{args.tag}.pt",
                                     map_location=device, weights_only=False))
    model.eval()

    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False,
                                      download=True, transform=tf)
    images = torch.stack([ds[i][0] for i in range(args.n_images)])

    summary = {'tag': args.tag, 'n_params': model.n_params,
               'sigmas': SIGMAS, 'per_sigma': {}}
    K_means = []
    for sigma in SIGMAS:
        stacked = []
        for s in range(args.n_seeds):
            torch.manual_seed(SEED + s * 100)
            np.random.seed(SEED + s * 100)
            per_step = []
            for b in range(0, len(images), args.batch):
                batch = images[b:b + args.batch].to(device)
                per_step.append(evaluate_fine_grid(model, batch, sigma,
                                                   args.k_max))
            stacked.append(np.concatenate(per_step, axis=1))
        mean_psnr = np.stack(stacked, axis=0).mean(axis=0)  # (K, n_img)
        kstar = np.argmax(mean_psnr, axis=0) + 1
        lo, hi = bootstrap_ci_kstar(kstar)
        K_means.append(float(np.mean(kstar)))
        summary['per_sigma'][f'{sigma:.3f}'] = {
            'kstar_mean': float(np.mean(kstar)),
            'kstar_std': float(np.std(kstar, ddof=1)),
            'kstar_ci95': [lo, hi]}
        print(f"sigma={sigma:.2f}: K*={np.mean(kstar):.2f}")

    a, alpha, r2 = fit_power_law(SIGMAS, K_means)
    ci_lo, ci_hi = parametric_bootstrap_ci(SIGMAS, K_means)
    summary['power_law'] = {'alpha': alpha, 'C': math.exp(a), 'R2': r2,
                            'ci95_alpha': [ci_lo, ci_hi]}
    out = OUT / f'fine_grid_{args.tag}.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f"alpha={alpha:.4f}  C={math.exp(a):.1f}  R2={r2:.4f}  "
          f"CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"[done] {out}")


if __name__ == '__main__':
    main()
