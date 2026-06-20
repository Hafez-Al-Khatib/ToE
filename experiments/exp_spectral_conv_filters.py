"""
exp_spectral_conv_filters.py
============================
Spectral analysis of KAN-EBM conv filter bank.

Hypothesis (brain-motivated): the wavelet-cortex analogy targets the EARLY
sensory filtering stage (V1: Gabor receptive fields), not the readout. The
KAN-EBM's first stage is a learned 5x5 conv filter bank that operates per
channel. If this stage is doing what V1 does, the learned filters should be
Gabor-like (oriented, band-pass, localized in both space and frequency).

This script:
  1. Loads the KAN-EBM 110K checkpoint, extracts filters[F, C, k, k].
  2. For each filter (F * C of them), measures:
        - dominant spatial frequency (peak of 2D power spectrum, cycles/pixel)
        - orientation selectivity (concentration of FFT energy in angular bins)
        - Gabor fit: brute-force best-fit Gabor (sigma, freq, theta, phase, phase_y)
          and resulting cosine similarity
        - DC bias (mean of filter, should be small for band-pass)
  3. Compares against a random-initialized null filter bank.
  4. Renders the trained filter bank as a grid + spectra.

Outputs:
  outputs/spectral_kan/conv_filter_spectra.json
  outputs/spectral_kan/conv_filter_bank.pdf
  outputs/spectral_kan/conv_filter_spectra.pdf
  outputs/spectral_kan/conv_summary.txt
"""
import json
import sys
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel  # noqa: E402

CKPT = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
OUT_DIR = ROOT / 'outputs' / 'spectral_kan'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gabor_2d(k, sigma, freq, theta, phase, cx=0.0, cy=0.0):
    """Generate a kxk Gabor patch centered on (cx, cy) in [-1, 1] coordinates.

    sigma:  gaussian envelope width
    freq:   cycles per unit (the kernel spans [-1, 1] so freq in cycles/(half-kernel))
    theta:  orientation (radians)
    phase:  carrier phase
    """
    half = (k - 1) / 2
    xs = (np.arange(k) - half) / half  # [-1, 1]
    ys = (np.arange(k) - half) / half
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Xc, Yc = X - cx, Y - cy
    Xr = Xc * np.cos(theta) + Yc * np.sin(theta)
    Yr = -Xc * np.sin(theta) + Yc * np.cos(theta)
    env = np.exp(-(Xr ** 2 + Yr ** 2) / (2 * sigma ** 2))
    carrier = np.cos(2 * np.pi * freq * Xr + phase)
    g = env * carrier
    # zero-mean (a proper wavelet has m0 = 0)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return g / (n + 1e-12)


def best_gabor_fit(filt, k):
    """Brute-force best-fit Gabor for a single kxk filter. Returns max
    cosine similarity (|<filter, gabor>| / ||filter||) and best params."""
    f = filt - filt.mean()
    n = np.linalg.norm(f)
    if n < 1e-12:
        return 0.0, None
    f_u = f / n

    best = 0.0
    best_params = None
    for sigma in [0.30, 0.45, 0.60, 0.85, 1.20]:
        for freq in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for theta in np.linspace(0, np.pi, 12, endpoint=False):
                for phase in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                    g = gabor_2d(k, sigma, freq, theta, phase)
                    c = abs(float(f_u.ravel() @ g.ravel()))
                    if c > best:
                        best = c
                        best_params = dict(sigma=sigma, freq=freq, theta=float(theta),
                                           phase=phase)
    return best, best_params


def filter_metrics_single(filt):
    """Per-filter metrics for one kxk filter."""
    filt = np.asarray(filt, dtype=np.float64)
    k = filt.shape[0]
    # DC component
    dc = float(filt.mean())
    dc_rel = float(abs(dc) / (np.linalg.norm(filt) + 1e-12))
    # 2D FFT
    f_zm = filt - filt.mean()
    F = np.fft.fftshift(np.fft.fft2(f_zm, s=(32, 32)))  # upsample for resolution
    P = np.abs(F) ** 2
    H = P.shape[0]; W = P.shape[1]
    cy_, cx_ = H // 2, W // 2
    # Distance/angle of each frequency bin from center
    yy, xx = np.meshgrid(np.arange(H) - cy_, np.arange(W) - cx_, indexing='ij')
    r = np.sqrt(yy ** 2 + xx ** 2)
    ang = np.arctan2(yy, xx)
    # Peak (skip DC)
    P_masked = P.copy(); P_masked[cy_, cx_] = 0
    idx = np.unravel_index(P_masked.argmax(), P.shape)
    peak_r = float(r[idx])
    peak_theta = float(ang[idx])
    # Orientation selectivity: bin P by angle (modulo pi) and measure concentration
    n_bins = 18
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    ang_mod = ang % np.pi
    bin_energy = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (ang_mod >= bin_edges[b]) & (ang_mod < bin_edges[b + 1]) & (r > 1)
        bin_energy[b] = P[mask].sum()
    p = bin_energy / (bin_energy.sum() + 1e-12)
    orient_entropy = float(-np.sum(p * np.log(p + 1e-12)))  # low = more oriented
    # Best Gabor fit
    gabor_sim, gabor_params = best_gabor_fit(filt, k)
    return {
        'dc_rel': float(dc_rel),
        'peak_freq_bins': float(peak_r),
        'peak_orientation_rad': float(peak_theta),
        'orient_entropy': float(orient_entropy),
        'gabor_similarity': float(gabor_sim),
        'gabor_params': gabor_params,
    }


def filter_bank_metrics(filters):
    """filters: (F, C, k, k) numpy. Returns per-filter list and aggregates."""
    F, C, k, _ = filters.shape
    rows = []
    for fi in range(F):
        for ci in range(C):
            m = filter_metrics_single(filters[fi, ci])
            m['filter_idx'] = fi
            m['channel_idx'] = ci
            rows.append(m)
    return rows


def aggregate_rows(rows):
    keys = ['dc_rel', 'peak_freq_bins', 'orient_entropy', 'gabor_similarity']
    agg = {}
    for kkey in keys:
        vals = np.array([r[kkey] for r in rows])
        agg[kkey] = {'mean': float(vals.mean()), 'median': float(np.median(vals)),
                     'std': float(vals.std()), 'min': float(vals.min()),
                     'max': float(vals.max())}
    return agg


def plot_filter_bank(filters, out_pdf, title='KAN-EBM 5x5 filters per channel'):
    """Render the F x C filter grid."""
    F, C, k, _ = filters.shape
    fig, axes = plt.subplots(F, C, figsize=(2 * C, 2 * F / 2))
    if F == 1:
        axes = axes[None, :]
    if C == 1:
        axes = axes[:, None]
    for fi in range(F):
        for ci in range(C):
            ax = axes[fi, ci]
            f = filters[fi, ci]
            vmax = max(abs(f.min()), abs(f.max()))
            ax.imshow(f, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if fi == 0:
                ax.set_title(f'ch{ci}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'f{fi}', fontsize=7, rotation=0, ha='right')
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_spectra(filters, out_pdf, title='Filter power spectra'):
    F, C, k, _ = filters.shape
    fig, axes = plt.subplots(F, C, figsize=(2 * C, 2 * F / 2))
    if F == 1:
        axes = axes[None, :]
    if C == 1:
        axes = axes[:, None]
    for fi in range(F):
        for ci in range(C):
            ax = axes[fi, ci]
            f = filters[fi, ci]
            f = f - f.mean()
            P = np.abs(np.fft.fftshift(np.fft.fft2(f, s=(32, 32)))) ** 2
            ax.imshow(np.log1p(P), cmap='magma', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if fi == 0:
                ax.set_title(f'ch{ci}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'f{fi}', fontsize=7, rotation=0, ha='right')
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    plt.close(fig)


def random_filter_bank_like(filters_t):
    """Random Gaussian with same std as trained filters."""
    return torch.randn_like(filters_t) * filters_t.std()


def main():
    device = torch.device('cpu')
    print(f'[ckpt] {CKPT}')
    model = KANEnergyModel(n_filters=32, kan_hidden=[96, 16]).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    filters_t = model.filters.detach().cpu()  # (F, C, k, k)
    F, C, k, _ = filters_t.shape
    print(f'[filters] shape=({F},{C},{k},{k})  '
          f'mean={filters_t.mean().item():+.4f}  std={filters_t.std().item():.4f}')

    filters = filters_t.numpy()
    rows_trained = filter_bank_metrics(filters)
    agg_trained = aggregate_rows(rows_trained)

    filters_null_t = random_filter_bank_like(filters_t)
    filters_null = filters_null_t.numpy()
    rows_null = filter_bank_metrics(filters_null)
    agg_null = aggregate_rows(rows_null)

    print('\n=== TRAINED filter bank ===')
    for k_, v in agg_trained.items():
        print(f'  {k_:20s} mean={v["mean"]:+.4f}  median={v["median"]:+.4f}  '
              f'min={v["min"]:+.4f}  max={v["max"]:+.4f}')
    print('\n=== RANDOM (null) filter bank ===')
    for k_, v in agg_null.items():
        print(f'  {k_:20s} mean={v["mean"]:+.4f}  median={v["median"]:+.4f}  '
              f'min={v["min"]:+.4f}  max={v["max"]:+.4f}')

    # Strip gabor_params for JSON (keep top-K most Gabor-like per channel)
    rows_clean = []
    for r in rows_trained:
        rc = {kk: vv for kk, vv in r.items() if kk != 'gabor_params'}
        rows_clean.append(rc)
    top10 = sorted(rows_trained, key=lambda r: -r['gabor_similarity'])[:10]
    top10_clean = [{kk: vv for kk, vv in r.items()} for r in top10]
    for r in top10_clean:
        if r['gabor_params']:
            r['gabor_params'] = {pk: float(pv) for pk, pv in r['gabor_params'].items()}
    record = {
        'checkpoint': str(CKPT),
        'filter_shape': list(filters.shape),
        'trained': {'aggregate': agg_trained, 'per_filter': rows_clean},
        'random_null': {'aggregate': agg_null},
        'top10_gabor_examples': top10_clean,
    }
    (OUT_DIR / 'conv_filter_spectra.json').write_text(json.dumps(record, indent=2))
    print(f'\n[save] {OUT_DIR / "conv_filter_spectra.json"}')

    plot_filter_bank(filters, OUT_DIR / 'conv_filter_bank.pdf',
                     title=f'Trained 5x5 filters ({F} filters x {C} channels)')
    plot_spectra(filters, OUT_DIR / 'conv_filter_spectra.pdf',
                 title='Trained filter power spectra (log scale)')
    print(f'[save] {OUT_DIR / "conv_filter_bank.pdf"}')
    print(f'[save] {OUT_DIR / "conv_filter_spectra.pdf"}')

    # Summary text
    lines = ['Conv filter spectral analysis (KAN-EBM 110K)',
             '=' * 60,
             f'Filter bank shape: {filters.shape}',
             '',
             'TRAINED vs RANDOM (Gabor similarity is the key metric):',
             f'  DC bias (rel):       trained={agg_trained["dc_rel"]["mean"]:.4f}   '
             f'random={agg_null["dc_rel"]["mean"]:.4f}',
             f'  Orient. entropy:     trained={agg_trained["orient_entropy"]["mean"]:.4f}   '
             f'random={agg_null["orient_entropy"]["mean"]:.4f}   '
             f'(lower=more oriented; max=ln(18)={math.log(18):.2f})',
             f'  Peak freq (FFT pts): trained={agg_trained["peak_freq_bins"]["mean"]:.2f}   '
             f'random={agg_null["peak_freq_bins"]["mean"]:.2f}',
             f'  Gabor similarity:    trained={agg_trained["gabor_similarity"]["mean"]:.4f}   '
             f'random={agg_null["gabor_similarity"]["mean"]:.4f}',
             '',
             f'Top-10 Gabor-like trained filters (cosine similarity to best Gabor):']
    for r in top10:
        gp = r['gabor_params'] or {}
        lines.append(
            f"  filter {r['filter_idx']:2d} ch {r['channel_idx']}: "
            f"sim={r['gabor_similarity']:.3f}  sigma={gp.get('sigma','?'):.2f}  "
            f"freq={gp.get('freq','?'):.2f}  theta={gp.get('theta',0)*180/math.pi:.0f} deg  "
            f"phase={gp.get('phase',0)/(math.pi):.2f}*pi")
    out_txt = OUT_DIR / 'conv_summary.txt'
    out_txt.write_text('\n'.join(lines))
    print(f'[save] {out_txt}')
    print()
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
