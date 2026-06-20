"""
exp_wav_kan_pilot.py
====================
Pilot training of the 4 wavelet-variant KAN-EBMs at matched ~32K parameters.

Variants:
  V1  front=free  head=bspline   -- original KAN-EBM (control)
  V2  front=free  head=wavkan    -- Wav-KAN head
  V3  front=gabor head=bspline   -- parametric Gabor front
  V4  front=gabor head=wavkan    -- fully wavelet

All trained with DSM on CIFAR-10 subset, with the standard protocol shared
with exp_cifar10.py:
  loss = ||x - (xn + sigma^2 * s_theta(xn))||^2,  s_theta = -grad E

CPU-friendly defaults (small subset, fewer epochs). For a full
fine-grid alpha comparison, rerun with --n_train 50000 --epochs 100 on GPU.

Outputs:
  outputs/wav_kan_pilot/<variant>.pt
  outputs/wav_kan_pilot/results.json
  outputs/wav_kan_pilot/results.txt
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

from wav_kan_ebm import WavKANEnergyModel  # noqa: E402

OUT_DIR = ROOT / 'outputs' / 'wav_kan_pilot'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    'free_bspline':  dict(front='free',  head='bspline'),
    'free_wavkan':   dict(front='free',  head='wavkan', n_atoms=3),
    'gabor_bspline': dict(front='gabor', head='bspline'),
    'gabor_wavkan':  dict(front='gabor', head='wavkan', n_atoms=3),
}


def dsm_loss(model, x, sigma):
    eps = torch.randn_like(x)
    xn = (x + sigma * eps).clamp(-1, 1)
    with torch.enable_grad():
        xn_g = xn.detach().requires_grad_(True)
        E = model.energy(xn_g)
        score = -torch.autograd.grad(E, xn_g, create_graph=True)[0]
    target = x
    pred = xn + sigma * sigma * score
    return F.mse_loss(pred, target)


def kstar_per_image(model, batch_clean, sigma, K_max=20, dt=0.05, dt_decay=0.97):
    n = batch_clean.size(0)
    xn = (batch_clean + torch.randn_like(batch_clean) * sigma).clamp(-1, 1)
    psnr_at_K = np.zeros((n, K_max), dtype=np.float32)
    u = xn.clone()
    step = dt
    for k in range(1, K_max + 1):
        u = u.detach().requires_grad_(True)
        E = model.energy(u)
        g = torch.autograd.grad(E.sum(), u)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach()
        step *= dt_decay
        diff = (u.clamp(-1, 1) - batch_clean) ** 2
        mse = diff.view(n, -1).mean(dim=1)
        psnr_at_K[:, k - 1] = (10.0 * torch.log10(4.0 / mse.clamp(min=1e-12))
                                ).cpu().numpy()
    kstar = (psnr_at_K.argmax(axis=1) + 1).astype(int).tolist()
    mean_psnr = psnr_at_K.mean(axis=0).tolist()
    return kstar, mean_psnr


def fit_powerlaw(sigmas, Ks):
    xs = np.log(np.asarray(sigmas, float))
    ys = np.log(np.asarray(Ks, float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(math.exp(intercept)), float(slope), float(r2)


def train(model, train_x, n_epochs, batch_size, lr, sigmas, log_every=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = train_x.size(0)
    history = []
    step = 0
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_losses = []
        t0 = time.time()
        for i in range(0, N, batch_size):
            xb = train_x[perm[i:i + batch_size]]
            sigma = float(np.random.choice(sigmas))
            loss = dsm_loss(model, xb, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss))
            step += 1
            if step % log_every == 0:
                print(f'    step {step:5d} loss={loss.item():.4f}', flush=True)
        elapsed = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))
        history.append({'epoch': epoch, 'mean_loss': mean_loss, 'sec': elapsed})
        print(f'  epoch {epoch+1}/{n_epochs} mean_loss={mean_loss:.4f} ({elapsed:.1f}s)',
              flush=True)
    return history


def evaluate(model, test_x, sigmas, K_max):
    model.eval()
    per_sigma = {}
    Ks_mean = []
    for sigma in sigmas:
        kstar_list, mean_psnr = kstar_per_image(model, test_x, sigma, K_max=K_max)
        kmean = float(np.mean(kstar_list))
        Ks_mean.append(kmean)
        per_sigma[f'{sigma:.3f}'] = {
            'kstar_mean': kmean,
            'kstar_std': float(np.std(kstar_list, ddof=1)) if len(kstar_list) > 1 else 0.0,
            'mean_psnr_per_K': mean_psnr,
            'peak_psnr': float(max(mean_psnr)),
            'peak_K': int(np.argmax(mean_psnr) + 1),
        }
    C, alpha, r2 = fit_powerlaw(sigmas, Ks_mean)
    return per_sigma, {'C': C, 'alpha': alpha, 'R2': r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=2000)
    ap.add_argument('--n_test', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()))
    ap.add_argument('--sigmas_train', nargs='+', type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument('--sigmas_eval', nargs='+', type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument('--K_max', type=int, default=20)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    print(f'[device] {device}')
    print(f'[args] {vars(args)}')

    # ── Data ───────────────────────────────────────────────────────
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    train_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True, download=True,
                                            transform=tf)
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True,
                                           transform=tf)
    train_x = torch.stack([train_ds[i][0] for i in range(args.n_train)]).to(device)
    test_x = torch.stack([test_ds[i][0] for i in range(args.n_test)]).to(device)
    print(f'[data] train {tuple(train_x.shape)}  test {tuple(test_x.shape)}')

    all_results = {'config': vars(args), 'variants': {}}
    total_t0 = time.time()
    for vname in args.variants:
        print(f'\n=== {vname} ===')
        cfg = dict(VARIANTS[vname])
        cfg.update(dict(n_filters=16, kan_hidden=[48, 16]))
        model = WavKANEnergyModel(**cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  config: {cfg}')
        print(f'  params: {n_params:,}')

        t0 = time.time()
        history = train(model, train_x, args.epochs, args.batch, args.lr,
                        args.sigmas_train)
        train_time = time.time() - t0
        print(f'  train time: {train_time/60:.1f} min')

        ckpt_path = OUT_DIR / f'{vname}.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f'  ckpt: {ckpt_path}')

        t0 = time.time()
        per_sigma, power_law = evaluate(model, test_x, args.sigmas_eval, args.K_max)
        eval_time = time.time() - t0
        print(f'  eval time: {eval_time:.1f}s')
        print(f'  alpha = {power_law["alpha"]:.3f}  C = {power_law["C"]:.2f}  '
              f'R2 = {power_law["R2"]:.4f}')

        all_results['variants'][vname] = {
            'config': cfg,
            'n_params': n_params,
            'train_time_sec': train_time,
            'eval_time_sec': eval_time,
            'history': history,
            'per_sigma': per_sigma,
            'power_law': power_law,
        }
        out_json = OUT_DIR / 'results.json'
        out_json.write_text(json.dumps(all_results, indent=2))

    total_time = time.time() - total_t0
    print(f'\nTotal time: {total_time/60:.1f} min')

    # Text summary
    lines = ['Wav-KAN pilot results (small-scale, CPU)',
             '=' * 60,
             f'n_train={args.n_train}  n_test={args.n_test}  epochs={args.epochs}',
             '']
    lines.append(f'{"variant":18s} {"params":>8s}  {"alpha":>7s}  {"C":>7s}  {"R2":>6s}  '
                 f'{"peak_dB@0.10":>12s}  {"peak_K@0.10":>11s}')
    for vname, r in all_results['variants'].items():
        pl = r['power_law']
        s10 = r['per_sigma'].get('0.100', {})
        lines.append(f'{vname:18s} {r["n_params"]:8,d}  {pl["alpha"]:7.3f}  '
                     f'{pl["C"]:7.2f}  {pl["R2"]:6.3f}  '
                     f'{s10.get("peak_psnr",0):12.2f}  {s10.get("peak_K",0):11d}')
    out_txt = OUT_DIR / 'results.txt'
    out_txt.write_text('\n'.join(lines))
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
