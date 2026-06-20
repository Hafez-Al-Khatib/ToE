"""
Data power-spectrum exponent (beta-hat) via radially-averaged 2D FFT spectra.
================================================================================
The spectral-shrinkage account of the inference-depth law predicts alpha = 2*gamma/beta,
where beta is the data power-spectrum decay exponent P(f) ~ f^{-beta}. This module:

(a) measures beta-hat for real datasets (CIFAR-10, CelebA-64, MNIST, Fashion, CBSD68),
    so the measured alphas (CIFAR ~1.38, CelebA-64 ~1.26) can be tested against the
    2*gamma/beta ordering ("in-the-wild" corroboration of the controlled beta-sweep);
(b) provides the synthetic Gaussian-random-field generator (P(f) = f^{-beta} by
    construction) used by theory/beta_sweep.py;
(c) self-validates the generator: `--datasets grf:2.0` must return beta-hat ~= 2.

Run:
    py -3.12 theory/measure_beta.py --datasets grf:1.0 grf:2.0 grf:3.0 cifar10 mnist
    py -3.12 theory/measure_beta.py --datasets cifar10 celeba64 --alphas cifar10=1.376,celeba64=1.262

Outputs: outputs/theory/data_beta.json, outputs/theory/data_beta_spectra.(pdf|png)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic 1/f^beta Gaussian random fields (shared with beta_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────

def grf_images(n, size=32, beta=2.0, channels=3, seed=0, std=0.35, chunk=4096):
    """Gaussian random fields with isotropic power spectrum P(f) ∝ f^{-beta}.

    Per-channel zero mean, std `std`, clamped to [-1, 1] (std=0.35 clips ~0.4% of
    pixels at ~2.9 sigma — negligible spectral distortion, and the realized beta-hat
    is re-measured downstream anyway). Matches the [-1,1] range / PSNR-peak-4
    convention of the K* harness (tier0_seeds.py).
    """
    g = torch.Generator().manual_seed(seed)
    ky = torch.fft.fftfreq(size)
    kx = torch.fft.rfftfreq(size)
    f = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    amp = torch.where(f > 0, f ** (-beta / 2.0), torch.zeros_like(f))  # zero DC
    outs = []
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        noise = torch.randn(m, channels, size, size, generator=g)
        img = torch.fft.irfft2(torch.fft.rfft2(noise) * amp, s=(size, size))
        img = img - img.mean(dim=(-2, -1), keepdim=True)
        img = img / (img.std(dim=(-2, -1), keepdim=True) + 1e-8) * std
        outs.append(img.clamp(-1, 1))
    return torch.cat(outs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Radially-averaged power spectrum + beta fit
# ─────────────────────────────────────────────────────────────────────────────

def radial_spectrum(images, window=True):
    """images: (N, C, H, W) tensor, H == W. Returns (f, P): f in cycles/pixel,
    ring-averaged power, averaged over images and channels.

    A 2D Hann window suppresses leakage from non-periodic image boundaries
    (important for real photos; harmless for periodic GRFs)."""
    x = images.float().cpu()
    N, C, H, W = x.shape
    assert H == W, "radial_spectrum assumes square images"
    if window:
        w1 = torch.hann_window(H, periodic=False)
        w2 = w1[:, None] * w1[None, :]
        x = x * w2
        norm = float((w2 ** 2).mean())
    else:
        norm = 1.0
    spec = torch.fft.fft2(x)
    P = (spec.real ** 2 + spec.imag ** 2).mean(dim=(0, 1)) / norm    # (H, W)
    ky = torch.fft.fftfreq(H) * H
    kx = torch.fft.fftfreq(W) * W
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kbin = k.round().long().flatten()
    Pf = P.flatten()
    f, Pr = [], []
    for ring in range(1, H // 2):
        m = kbin == ring
        if m.any():
            f.append(ring / H)
            Pr.append(float(Pf[m].mean()))
    return np.array(f), np.array(Pr)


def fit_beta(f, P, fmin=0.05, fmax=0.40):
    """Log-log OLS over the band f in [fmin, fmax] cycles/pixel. Returns (beta_hat, R^2)."""
    f, P = np.asarray(f, float), np.asarray(P, float)
    m = (f >= fmin) & (f <= fmax) & (P > 0)
    if m.sum() < 4:
        return float('nan'), float('nan')
    x, y = np.log(f[m]), np.log(P[m])
    slope, intercept = np.polyfit(x, y, 1)
    yh = intercept + slope * x
    r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(-slope), float(r2)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders (local data only; each loader degrades gracefully if absent)
# ─────────────────────────────────────────────────────────────────────────────

def load_images(name, n):
    """Returns (N, C, H, W) tensor in [-1, 1] for a dataset spec.

    Specs: cifar10 | mnist | fashion | celeba64 | cbsd68 | grf:<beta>[:<size>]
    """
    if name.startswith('grf:'):
        parts = name.split(':')
        beta = float(parts[1])
        size = int(parts[2]) if len(parts) > 2 else 32
        return grf_images(n, size=size, beta=beta, channels=3, seed=4242)

    from torchvision import datasets, transforms
    norm3 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    norm1 = transforms.Normalize((0.5,), (0.5,))

    if name == 'cifar10':
        tf = transforms.Compose([transforms.ToTensor(), norm3])
        ds = datasets.CIFAR10(ROOT / 'data', train=True, download=False, transform=tf)
    elif name == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), norm1])
        ds = datasets.MNIST(ROOT / 'data', train=True, download=False, transform=tf)
    elif name == 'fashion':
        tf = transforms.Compose([transforms.ToTensor(), norm1])
        ds = datasets.FashionMNIST(ROOT / 'data', train=True, download=False, transform=tf)
    elif name == 'celeba64':
        # same preprocessing as theory/scaleup_64.py
        tf = transforms.Compose([transforms.Resize(64, antialias=True),
                                 transforms.CenterCrop(64),
                                 transforms.ToTensor(), norm3])
        ds = datasets.CelebA(str(ROOT / 'data'), split='train', download=False, transform=tf)
    elif name == 'cbsd68':
        from PIL import Image
        files = sorted((ROOT / 'data' / 'CBSD68').rglob('*.png'))
        if not files:
            raise FileNotFoundError('no PNGs under data/CBSD68')
        tf = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(), norm3])
        return torch.stack([tf(Image.open(fp).convert('RGB')) for fp in files[:n]])
    else:
        raise ValueError(f'unknown dataset spec: {name}')
    n = min(n, len(ds))
    return torch.stack([ds[i][0] for i in range(n)])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(spectra, fits, path_stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for name, (f, P) in spectra.items():
        bh, r2 = fits[name]
        ax.loglog(f, P / P[0], lw=1.6,
                  label=f"{name}  " + (f"$\\hat\\beta$={bh:.2f} (R²={r2:.2f})"
                                       if np.isfinite(bh) else "(fit failed)"))
    ax.set_xlabel('spatial frequency $f$ (cycles/pixel)')
    ax.set_ylabel('radially-averaged power (normalized)')
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title('Data power spectra and fitted decay exponents')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{path_stem}.{ext}", dpi=180)
    plt.close(fig)
    print(f"[fig] {path_stem}.pdf/.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['grf:1.0', 'grf:2.0', 'grf:3.0', 'cifar10', 'mnist', 'fashion'])
    p.add_argument('-n', type=int, default=2000, help='images per dataset (default 2000)')
    p.add_argument('--fmin', type=float, default=0.05, help='fit band lower edge (cycles/pixel)')
    p.add_argument('--fmax', type=float, default=0.40, help='fit band upper edge (cycles/pixel)')
    p.add_argument('--alphas', default='',
                   help='optional measured alphas, e.g. "cifar10=1.376,celeba64=1.262" — '
                        'prints the implied gamma = alpha*beta_hat/2 per dataset')
    args = p.parse_args()

    spectra, fits, results = {}, {}, {}
    for name in args.datasets:
        try:
            imgs = load_images(name, args.n)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        f, P = radial_spectrum(imgs)
        bh, r2 = fit_beta(f, P, args.fmin, args.fmax)
        spectra[name], fits[name] = (f, P), (bh, r2)
        results[name] = {'beta_hat': bh, 'r2': r2, 'n_images': int(imgs.shape[0]),
                         'size': int(imgs.shape[-1]), 'band': [args.fmin, args.fmax]}
        print(f"{name:>12}:  beta_hat = {bh:6.3f}   R^2 = {r2:.4f}   "
              f"({imgs.shape[0]} imgs @ {imgs.shape[-1]}px)")

    if args.alphas:
        print("\nimplied gamma = alpha * beta_hat / 2  (theory: constant across datasets "
              "at fixed architecture family):")
        for pair in args.alphas.split(','):
            ds, a = pair.split('=')
            ds, a = ds.strip(), float(a)
            if ds in results and np.isfinite(results[ds]['beta_hat']):
                g = a * results[ds]['beta_hat'] / 2.0
                results[ds]['alpha'] = a
                results[ds]['gamma_implied'] = g
                print(f"  {ds:>12}: alpha={a:.3f}  beta_hat={results[ds]['beta_hat']:.3f}"
                      f"  ->  gamma = {g:.3f}")

    (OUT / 'data_beta.json').write_text(json.dumps(results, indent=2))
    print(f"\n[json] {OUT / 'data_beta.json'}")
    if spectra:
        make_figure(spectra, fits, str(OUT / 'data_beta_spectra'))


if __name__ == '__main__':
    main()
