"""
K*(sigma) sweep on FashionMNIST and MNIST — validates predictive-alpha.
========================================================================
The predictive-alpha analysis (predictive_alpha.py) predicted:
  Fashion: alpha_pred = 1.837  (from beta_hat=2.278, gamma_bar=2.092)

This script trains EBM (ConvMLP-GELU) and score (ScoreNet) denoisers on
grayscale 28x28 images, runs the same K-sweep protocol as tier0_seeds.py,
fits alpha, and compares to the prediction.

These are 1-channel datasets, so models use n_channels=1.

Run:
    py -3.12 theory/fashion_mnist_kstar.py --dataset fashion --epochs 40
    py -3.12 theory/fashion_mnist_kstar.py --dataset mnist --epochs 40
    py -3.12 theory/fashion_mnist_kstar.py --dataset fashion --quick

Chain overnight (fashion then mnist):
    py -3.12 theory/fashion_mnist_kstar.py --dataset fashion --epochs 40 && py -3.12 theory/fashion_mnist_kstar.py --dataset mnist --epochs 40

Outputs: outputs/theory/fashion_results.json, outputs/theory/mnist_results.json
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'theory'))

from torchvision import datasets, transforms
from rebuttal_smooth_mlp import ConvSmoothMLPEBM, train_one
from phase_b_diffusion import ScoreNet, train_score
from tier0_seeds import fit_alpha, psnr_per_image, SIGMAS, K_MAX

OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)


def get_data(dataset_name, n_train, batch_size=32):
    """Load FashionMNIST or MNIST, grayscale 28x28, normalized to [-1, 1]."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    DS = datasets.FashionMNIST if dataset_name == 'fashion' else datasets.MNIST
    train_ds = DS(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = DS(ROOT / 'data', train=False, download=True, transform=tf)
    subset = torch.utils.data.Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, test_ds


def kstar_ebm(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Per-image continuous K* for EBM (gradient descent on energy)."""
    per_image = []
    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        u = (clean + sigma * torch.randn_like(clean)).to(device)
        series = [psnr_per_image(u, clean)]
        step = dt
        for _ in range(K_MAX):
            with torch.enable_grad():
                xi = u.detach().requires_grad_(True)
                E = model.energy(xi)
                if E.ndim > 0:
                    E = E.sum()
                g = torch.autograd.grad(E, xi)[0].detach()
            u = (u - step * g.clamp(-1, 1)).detach()
            step *= decay
            series.append(psnr_per_image(u, clean))
        arr = np.stack(series, axis=0)
        per_image.append(arr.argmax(axis=0))
    return float(np.mean(per_image))


def kstar_score(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Per-image continuous K* for score model (score ascent)."""
    per_image = []
    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        u = (clean + sigma * torch.randn_like(clean)).to(device)
        series = [psnr_per_image(u, clean)]
        step = dt
        with torch.no_grad():
            for _ in range(K_MAX):
                s = model(u).clamp(-1, 1)
                u = u + step * s
                step *= decay
                series.append(psnr_per_image(u, clean))
        arr = np.stack(series, axis=0)
        per_image.append(arr.argmax(axis=0))
    return float(np.mean(per_image))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='fashion', choices=['fashion', 'mnist'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--n_eval', type=int, default=64)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--families', nargs='+', default=['convmlp_gelu', 'score'])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_train, args.n_eval = 5, 5000, 16
        args.seeds = [0]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}"
          + (f" [{torch.cuda.get_device_name(0)}]" if device.type == 'cuda' else ""))
    print(f"[dataset] {args.dataset} (1-channel, 28x28)")

    loader, test_ds = get_data(args.dataset, args.n_train, batch_size=args.batch)
    clean = torch.stack([test_ds[i][0] for i in range(args.n_eval)]).to(device)
    print(f"[data] train={args.n_train}, eval={args.n_eval}, shape={clean.shape}")

    res_path = OUT / f'{args.dataset}_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    FAMILIES = {
        'convmlp_gelu': lambda: ConvSmoothMLPEBM(
            n_filters=16, filter_size=5, mlp_hidden=160,
            n_channels=1, activation='gelu'),
        'score': lambda: ScoreNet(ch=1, h=64),
    }

    for fam in args.families:
        is_score = (fam == 'score')
        results[fam] = {'alpha': [], 'r2': [], 'kstars': []}

        for seed in args.seeds:
            tag = f"{fam}-{args.dataset}-s{seed}"
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = FAMILIES[fam]().to(device)
            n_p = model.n_params
            print(f"\n=== {tag} ({n_p:,} params) ===", flush=True)

            if is_score:
                train_score(model, loader, device, n_epochs=args.epochs,
                            sigma_train=0.15)
            else:
                train_one(model, loader, device, n_epochs=args.epochs,
                          sigma_train=0.15, tag=tag)

            kstars = []
            for sig in SIGMAS:
                if is_score:
                    k = kstar_score(model, clean, sig, device)
                else:
                    k = kstar_ebm(model, clean, sig, device)
                kstars.append(k)

            alpha, r2 = fit_alpha(SIGMAS, kstars)
            results[fam]['alpha'].append(alpha)
            results[fam]['r2'].append(r2)
            results[fam]['kstars'].append(kstars)
            results[fam]['n_params'] = n_p

            print(f"  {tag}: K*={[round(k,2) for k in kstars]}  "
                  f"alpha={alpha:.3f}  R^2={r2:.3f}", flush=True)

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        res_path.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "=" * 72)
    print(f"{args.dataset.upper()} -- K*(sigma) power-law results")
    print("-" * 72)
    for fam, r in results.items():
        a = np.array(r['alpha'])
        a = a[np.isfinite(a)]
        if len(a) == 0:
            continue
        m = a.mean()
        s = a.std(ddof=1) if len(a) > 1 else 0
        ps = ", ".join(f"{x:.3f}" for x in r['alpha'])
        print(f"  {fam:>14}: alpha = {m:.3f} +/- {s:.3f}   [{ps}]")

    # Compare to prediction
    pred_path = OUT / 'predictive_alpha.json'
    if pred_path.exists():
        pa = json.loads(pred_path.read_text())
        up = pa.get('unmeasured_predictions', {})
        if args.dataset in up:
            pred = up[args.dataset]['alpha_pred']
            all_alpha = []
            for fam, r in results.items():
                all_alpha.extend([x for x in r['alpha'] if np.isfinite(x)])
            if all_alpha:
                measured = float(np.mean(all_alpha))
                err = measured - pred
                rel = err / pred
                print(f"\n  PREDICTED alpha = {pred:.3f}")
                print(f"  MEASURED  alpha = {measured:.3f}")
                print(f"  ERROR           = {err:+.3f} ({rel:+.1%})")
                ok = "CONFIRMED" if abs(rel) < 0.10 else "MARGINAL" if abs(rel) < 0.20 else "MISSED"
                print(f"  VERDICT: {ok}")

    print(f"\n[json] {res_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
