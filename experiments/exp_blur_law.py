# experiments/exp_blur_law.py
"""Blur-trained control (panel domain-Q2, DA alternative-explanation 5):
is the corruption phase boundary a property of the corruption class, or of
the training objective?

Paper finding: DSM-trained (additive-noise) descent gives K* ~ 0 for blur.
Control: train the SAME ConvSmoothMLP-GELU head (matched recipe hyper-
parameters) on a blur-residual objective -- gradient of E regresses the
blur residual, mirroring DSM's structure with the blur residual replacing
the noise residual:

    x_b = gaussian_blur(x, sigma_b),  sigma_b ~ U{0.5, 1.0, 1.5, 2.0}
    loss = MSE( x_b - grad_x E(x_b),  x )

Evaluation: descent protocol (dt=0.05, decay 0.97, clamp +/-1) on held-out
images at each blur severity; K* = argmax mean PSNR over K=0..30; fit
log K* vs log sigma_b. If K* grows with severity (law-like), the boundary
is training-objective-conditioned, not corruption-conditioned -- the
honest scoping the revised Sec. 6 states.

EXPLORATORY: one seed, one head. Writes outputs/blur_law/blur_law.json.
Run: py experiments/exp_blur_law.py --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from rebuttal_smooth_mlp import ConvSmoothMLPEBM
from exp_fine_grid_kstar import sequential_denoise_record
from exp_inference_frontier import load_test_images, psnr_batch

SEVERITIES = [0.5, 1.0, 1.5, 2.0]
K_MAX = 30
N_EVAL = 200


def gaussian_blur(x, sigma_b):
    k = max(3, int(2 * round(2.5 * sigma_b) + 1))
    return T.functional.gaussian_blur(x, kernel_size=k, sigma=sigma_b)


def blur_loss(model, x_clean):
    sig = SEVERITIES[torch.randint(len(SEVERITIES), (1,)).item()]
    xb = gaussian_blur(x_clean, sig)
    with torch.enable_grad():
        xi = xb.detach().requires_grad_(True)
        grad = torch.autograd.grad(model.energy(xi), xi, create_graph=True)[0]
    return F.mse_loss(xi - grad, x_clean)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    args = p.parse_args()
    device = torch.device(args.device)
    out_dir = ROOT / 'outputs' / 'blur_law'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / 'blur_ebm.pt'

    torch.manual_seed(0)
    model = ConvSmoothMLPEBM(n_filters=16, mlp_hidden=160,
                             activation='gelu').to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        print('[skip] blur EBM already trained')
    else:
        tf = T.Compose([T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True,
                                          download=True, transform=tf)
        subset = torch.utils.data.Subset(ds, list(range(args.n_train)))
        loader = torch.utils.data.DataLoader(subset, batch_size=128,
                                             shuffle=True, num_workers=0,
                                             pin_memory=True, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs)
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            tot, nb = 0.0, 0
            for batch in loader:
                x = batch[0].to(device)
                opt.zero_grad()
                loss = blur_loss(model, x)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                nb += 1
            sched.step()
            if ep <= 3 or ep % 10 == 0 or ep == args.epochs:
                print(f'  [blur_ebm] ep {ep:>3d}/{args.epochs} '
                      f'loss={tot / nb:.5f}')
        torch.save(model.state_dict(), ckpt)
        print(f'[saved] {ckpt} ({time.time() - t0:.0f}s)')

    model.eval()
    images = load_test_images(N_EVAL)
    result = {'severities': {}, 'protocol': 'K*=argmax mean PSNR, K=0..30, '
              f'{N_EVAL} held-out images, descent dt=0.05 decay=0.97'}
    kstars = []
    for sig in SEVERITIES:
        xb = gaussian_blur(images, sig).clamp(-1, 1).to(device)
        clean = images.to(device)
        states = sequential_denoise_record(model, xb, K_max=K_MAX)
        curve = [float(np.mean(psnr_batch(xb, clean)))] + \
                [float(np.mean(psnr_batch(s, clean))) for s in states]
        kstar = int(np.argmax(curve))          # 0 = no iteration helps
        kstars.append(max(kstar, 1e-9))
        result['severities'][str(sig)] = {'kstar': kstar,
                                          'psnr_per_k': curve}
        print(f'  severity {sig}: K*={kstar}, peak {max(curve):.2f} dB, '
              f'K=0 {curve[0]:.2f} dB')
    ls, lk = np.log(SEVERITIES), np.log(kstars)
    slope, intercept = np.polyfit(ls, lk, 1)
    pred = slope * ls + intercept
    ss_res = float(np.sum((lk - pred) ** 2))
    ss_tot = float(np.sum((lk - lk.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    result['power_law'] = {'alpha': float(slope), 'R2': r2}
    result['verdict'] = ('LAW-LIKE (K* grows with severity)' if slope > 0.3
                         and r2 > 0.8 else 'NO LAW (K* flat or erratic)')
    print(f'blur law fit: alpha={slope:.3f} R2={r2:.3f} -> '
          f'{result["verdict"]}')
    (out_dir / 'blur_law.json').write_text(json.dumps(result, indent=1))
    print('wrote', out_dir / 'blur_law.json')


if __name__ == '__main__':
    main()
