"""
Tier-0 rigor: independent TRAINING-seed variance for the inference exponent alpha.
==================================================================================
The credibility smell in the paper ("alpha identical to 16 digits, std=0.0") comes
from training every variant with the SAME seed and only varying the eval-noise seed.
Here we train each architecture with >=3 INDEPENDENT training seeds and report alpha
as mean +/- std across those runs, so the KAN>ConvMLP gap is shown to be real and
separable, not a fitting artifact.

Run (GPU):  py -3.12 theory/tier0_seeds.py --epochs 40 --seeds 0 1 2 --archs kan convmlp_gelu
Timing:     py -3.12 theory/tier0_seeds.py --time_only --epochs 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from torchvision import datasets, transforms                 # noqa: E402
from exp_cifar10 import KANEnergyModel                        # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM, train_one   # noqa: E402

OUT = ROOT / 'outputs' / 'tier0_seeds'
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = {
    'kan':          lambda: KANEnergyModel(n_filters=16, filter_size=5,
                                           kan_hidden=[48, 16], n_channels=3),
    'convmlp_gelu': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                             n_channels=3, activation='gelu'),
    'convmlp_silu': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                             n_channels=3, activation='silu'),
    'convmlp_tanh': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                             n_channels=3, activation='tanh'),
}
SIGMAS = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
K_MAX = 30


def get_data(n_train, device, batch_size=32):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    subset = torch.utils.data.Subset(train_ds, list(range(n_train)))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, test_ds


def psnr_per_image(u, clean):
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).cpu().numpy()   # (N,)


def kstar_at_sigma(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Continuous K* = mean over images of the PER-IMAGE argmax-PSNR step.

    Averaging per-image K* (not argmax of the mean curve) keeps K* continuous, so
    genuine per-seed model differences surface as real variance in alpha.
    """
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
        arr = np.stack(series, axis=0)            # (K+1, N)
        per_image.append(arr.argmax(axis=0))      # (N,) per-image integer K*
    return float(np.mean(per_image))              # continuous mean K*


def fit_alpha(sigmas, kstars):
    sigmas, kstars = np.array(sigmas, float), np.array(kstars, float)
    ok = kstars > 0
    if ok.sum() < 4 or np.std(kstars[ok]) < 1e-6:
        return float('nan'), float('nan')
    x, y = np.log(sigmas[ok]), np.log(kstars[ok])
    slope, intercept = np.polyfit(x, y, 1)
    yh = intercept + slope * x
    r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(r2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--batch', type=int, default=32, help='train batch (32 fits a 10GB GPU)')
    p.add_argument('--n_eval', type=int, default=64)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--archs', nargs='+', default=['kan', 'convmlp_gelu'])
    p.add_argument('--time_only', action='store_true', help='time a few epochs and exit')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}"
          + (f"  [{torch.cuda.get_device_name(0)}]" if device.type == 'cuda' else ""))
    loader, test_ds = get_data(args.n_train, device, batch_size=args.batch)
    clean = torch.stack([test_ds[i][0] for i in range(args.n_eval)]).to(device)

    if args.time_only:
        for arch in args.archs:
            model = ARCHS[arch]().to(device)
            t0 = time.time()
            train_one(model, loader, device, n_epochs=args.epochs,
                      sigma_train=0.15, tag=f"time-{arch}")
            dt = time.time() - t0
            print(f"[time] {arch}: {dt/args.epochs:.1f}s/epoch  ->  "
                  f"40ep~{dt/args.epochs*40/60:.1f}min, 3seeds~{dt/args.epochs*40*3/60:.1f}min")
        return

    results = {}
    for arch in args.archs:
        results[arch] = {'alpha': [], 'r2': [], 'kstars': []}
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = ARCHS[arch]().to(device)
            print(f"\n=== {arch} seed={seed} ({model.n_params:,} params) ===", flush=True)
            train_one(model, loader, device, n_epochs=args.epochs,
                      sigma_train=0.15, tag=f"{arch}-s{seed}")
            torch.save(model.state_dict(), OUT / f"{arch}_seed{seed}.pt")
            kstars = [kstar_at_sigma(model, clean, s, device) for s in SIGMAS]
            alpha, r2 = fit_alpha(SIGMAS, kstars)
            results[arch]['alpha'].append(alpha)
            results[arch]['r2'].append(r2)
            results[arch]['kstars'].append(kstars)
            print(f"  seed {seed}: K*={kstars}  alpha={alpha:.3f}  R^2={r2:.3f}", flush=True)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    (OUT / 'tier0_results.json').write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 64)
    print(f"{'arch':>14} | {'alpha (mean +/- std)':>22} | {'per-seed alpha':>22}")
    print("-" * 64)
    for arch, r in results.items():
        a = np.array(r['alpha'])
        a = a[np.isfinite(a)]
        ps = ", ".join(f"{x:.3f}" for x in r['alpha'])
        print(f"{arch:>14} | {a.mean():>10.3f} +/- {a.std(ddof=1) if len(a)>1 else 0:>7.3f} | {ps:>22}")
    print(f"\nsaved {OUT/'tier0_results.json'}")


if __name__ == '__main__':
    main()
