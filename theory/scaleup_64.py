"""
Scale-up: does the universal K*(sigma) law hold beyond 32px?  STL10 @ 64px.
===========================================================================
Trains energy (KAN, ConvMLP-GELU) and score (diffusion-style) denoisers on
64x64 STL10 patches with the same single-sigma recipe, and checks that the
inference-depth exponent is (a) a clean power law and (b) consistent across
model families at higher resolution. Per-arch batch sizes keep the KAN
second-order autograd within the 10GB 3080.

Designed to run unattended (~3-4h). Saves outputs/theory/scaleup64_results.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))

from torchvision import datasets, transforms          # noqa: E402
from exp_cifar10 import KANEnergyModel                 # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM, train_one  # noqa: E402
from phase_b_diffusion import ScoreNet, train_score    # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)
SIGMAS = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
K_MAX = 50   # 64px K* is ~2.5x the 32px values; avoid clipping the high-sigma fit

# per-arch: (constructor, is_score, batch, n_train, epochs, seeds)
CONFIGS = {
    'score':        (lambda: ScoreNet(ch=3, h=64), True, 128, 15000, 30, [0, 1, 2]),
    'convmlp_gelu': (lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                              n_channels=3, activation='gelu'),
                     False, 32, 12000, 30, [0, 1, 2]),
    'kan':          (lambda: KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[48, 16],
                                            n_channels=3),
                     False, 8, 8000, 18, [0, 1]),
}


def get_celeba_64(n_train, bs):
    """CelebA @ 64x64, RGB, [-1,1] -- local (no download), genuinely higher-res than CIFAR."""
    tf = transforms.Compose([transforms.Resize(64, antialias=True),
                             transforms.CenterCrop(64),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tr = datasets.CelebA(str(ROOT / 'data'), split='train', download=False, transform=tf)
    te = datasets.CelebA(str(ROOT / 'data'), split='test', download=False, transform=tf)
    sub = torch.utils.data.Subset(tr, list(range(n_train)))
    loader = torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=True, num_workers=0)
    return loader, te


def psnr_img(u, clean):
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).cpu().numpy()


def kstar(model, clean, sigma, is_score, dt=0.05, decay=0.97, eval_seeds=2, eval_bs=8):
    """Per-image continuous K*, evaluated in mini-batches so KAN's 64px
    energy+autograd stays within the 10GB GPU (full-batch eval OOM-thrashes)."""
    per_img = []
    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        seed_kstar = []
        for b in range(0, clean.shape[0], eval_bs):
            cb = clean[b:b + eval_bs]
            u = cb + sigma * torch.randn_like(cb)
            series = [psnr_img(u, cb)]
            step = dt
            for _ in range(K_MAX):
                if is_score:
                    with torch.no_grad():
                        s = model(u).clamp(-1, 1)
                    u = u + step * s
                else:
                    with torch.enable_grad():
                        xi = u.detach().requires_grad_(True)
                        E = model.energy(xi)
                        if E.ndim > 0:
                            E = E.sum()
                        g = torch.autograd.grad(E, xi)[0].detach()
                    u = (u - step * g.clamp(-1, 1)).detach()
                series.append(psnr_img(u, cb))
            seed_kstar.append(np.stack(series, 0).argmax(0))
        per_img.append(np.concatenate(seed_kstar))
    return float(np.mean(per_img))


def fit_alpha(sig, kst):
    sig, kst = np.array(sig, float), np.array(kst, float)
    ok = kst > 0
    sl, ic = np.polyfit(np.log(sig[ok]), np.log(kst[ok]), 1)
    yh = ic + sl * np.log(sig[ok])
    r2 = 1 - np.sum((np.log(kst[ok]) - yh) ** 2) / np.sum((np.log(kst[ok]) - np.log(kst[ok]).mean()) ** 2)
    return float(sl), float(r2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', nargs='+', default=list(CONFIGS.keys()))
    args = ap.parse_args()

    print(f"[device] {DEVICE}"
          + (f" [{torch.cuda.get_device_name(0)}]" if DEVICE.type == 'cuda' else ""), flush=True)
    t_start = time.time()
    res_path = OUT / 'scaleup64_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}  # merge, don't clobber
    for arch in args.archs:
        ctor, is_score, bs, n_tr, eps, seeds = CONFIGS[arch]
        loader, te = get_celeba_64(n_tr, bs)
        clean = torch.stack([te[i][0] for i in range(48)]).to(DEVICE)
        results[arch] = {'alpha': [], 'r2': [], 'kstars': []}
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = ctor().to(DEVICE)
            tag = f"{arch}-s{seed}"
            print(f"\n=== {tag} ({model.n_params:,} p, bs={bs}, {n_tr} imgs, {eps} ep) "
                  f"[{time.time()-t_start:.0f}s] ===", flush=True)
            if is_score:
                train_score(model, loader, DEVICE, n_epochs=eps)
            else:
                train_one(model, loader, DEVICE, n_epochs=eps, sigma_train=0.15, tag=tag)
            ks = [kstar(model, clean, s, is_score) for s in SIGMAS]
            a, r2 = fit_alpha(SIGMAS, ks)
            results[arch]['alpha'].append(a)
            results[arch]['r2'].append(r2)
            results[arch]['kstars'].append(ks)
            print(f"  {tag}: K*={[round(k,2) for k in ks]}  alpha={a:.3f}  R^2={r2:.3f}", flush=True)
            (OUT / 'scaleup64_results.json').write_text(json.dumps(results, indent=2))
            del model
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"{'arch':>14} | {'alpha (mean+/-std)':>20}")
    for arch, r in results.items():
        a = np.array(r['alpha'])
        print(f"{arch:>14} | {a.mean():.3f} +/- {(a.std(ddof=1) if len(a)>1 else 0):.3f}")
    print(f"\n[done in {time.time()-t_start:.0f}s]  saved {OUT/'scaleup64_results.json'}")


if __name__ == '__main__':
    main()
