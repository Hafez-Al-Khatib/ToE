"""
exp_spectral_kan_edges.py
=========================
Spectral analysis of learned KAN-EBM edge functions.

Hypothesis (brain-motivated): if intelligence learns by decomposing sensory
input into its natural frequencies, then a KAN-EBM trained on natural images
should spontaneously discover wavelet-like edge functions even though its
parameterization is generic B-splines with no wavelet inductive bias.

This script:
  1. Loads the trained KAN-EBM 110K checkpoint.
  2. Reconstructs each edge function phi_ij(x) on a dense x-grid:
        phi_ij(x) = base_weight[j,i] * SiLU(x)
                  + sum_k (spline_weight[j,i,k] * spline_scaler[j,i]) * B_k(x)
  3. For each edge, measures:
        - effective compact support (fraction of L2 mass in central window)
        - vanishing moments m_n = | int x^n phi(x) dx |, n=0,1,2
        - dominant frequency from FFT
        - spectral concentration (entropy of normalized power spectrum)
        - wavelet match: best (scale, shift) fit to Mexican-hat / Morlet
  4. Compares against random B-spline baselines (same parameterization,
     random coefficients) to factor out parameterization bias.

Outputs:
  outputs/spectral_kan/edge_spectra.json
  outputs/spectral_kan/edge_examples.pdf (representative edges + spectra)
  outputs/spectral_kan/summary.txt
"""
import json
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel  # noqa: E402

CKPT = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
OUT_DIR = ROOT / 'outputs' / 'spectral_kan'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_X = 512
X_RANGE = (-1.0, 1.0)
CENTRAL_FRACTION = 0.5  # window for compact-support metric


def b_splines_eval(grid, spline_order, x):
    """Vectorized B-spline basis evaluation on a 1-D x grid.

    grid:   (in_features, grid_size + 2*order + 1) -- knot positions
    Returns bases of shape (x_len, in_features, grid_size + order).
    """
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
        )
    return bases.contiguous()


def reconstruct_edges(layer, x):
    """For one KANLinear layer, return (out, in, N_x) tensor of phi_ij(x)."""
    x_in = x.unsqueeze(1).expand(-1, layer.in_features).contiguous()
    bases = b_splines_eval(layer.grid, layer.spline_order, x_in)
    # bases: (N_x, in, G+order)
    silu_base = F.silu(x)  # (N_x,)
    # base contribution: base_weight[j, i] * silu(x) -> (out, in, N_x)
    base_part = layer.base_weight.unsqueeze(-1) * silu_base.view(1, 1, -1)
    # spline contribution: scaler[j,i] * sum_k spline_weight[j,i,k] * bases[N_x, i, k]
    scaled_w = layer.spline_weight * layer.spline_scaler.unsqueeze(-1)  # (out, in, G+order)
    # einsum: spline_part[j, i, n] = sum_k scaled_w[j, i, k] * bases[n, i, k]
    spline_part = torch.einsum('jik,nik->jin', scaled_w, bases)
    return base_part + spline_part  # (out, in, N_x)


def edge_metrics(phi, x):
    """Per-edge spectral and shape descriptors. phi: (N_edges, N_x).
    Returns dict of np.ndarrays of length N_edges.
    """
    N_x = x.shape[0]
    dx = (x[-1] - x[0]) / (N_x - 1)
    phi_np = phi.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    # L2 normalize each edge (zero-mean-aware: we keep DC for moment computation)
    energy = np.sum(phi_np ** 2, axis=1) * float(dx)
    norm = np.sqrt(np.maximum(energy, 1e-12))
    phi_n = phi_np / norm[:, None]  # unit-L2 edges

    # Compact-support: fraction of L2 mass inside |x| < CENTRAL_FRACTION
    mask_central = (np.abs(x_np) < CENTRAL_FRACTION)
    central_energy = np.sum(phi_n[:, mask_central] ** 2, axis=1) * float(dx)
    # Effective support: width of smallest contiguous window containing 95% of energy
    cumulative = np.cumsum(phi_n ** 2 * float(dx), axis=1)
    eff_width = np.zeros(phi_n.shape[0])
    for i in range(phi_n.shape[0]):
        c = cumulative[i]
        # find smallest window [a,b] s.t. c[b] - c[a] >= 0.95
        best = N_x
        j = 0
        for k in range(N_x):
            while c[k] - c[j] > 0.95:
                j += 1
            if c[k] - c[max(j - 1, 0)] >= 0.95:
                best = min(best, k - max(j - 1, 0))
        eff_width[i] = best * float(dx)

    # Vanishing moments (on un-normalized, with the actual amplitude)
    m0 = np.abs(np.sum(phi_np, axis=1) * float(dx))
    m1 = np.abs(np.sum(phi_np * x_np[None, :], axis=1) * float(dx))
    m2 = np.abs(np.sum(phi_np * (x_np[None, :] ** 2), axis=1) * float(dx))
    # Normalize moments by L2 norm so we compare shape, not amplitude
    m0_rel = m0 / np.sqrt(np.maximum(energy, 1e-12))
    m1_rel = m1 / np.sqrt(np.maximum(energy, 1e-12))

    # FFT
    spec = np.fft.rfft(phi_np, axis=1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(N_x, d=float(dx))  # cycles per unit x
    # Dominant frequency: peak of power (skip DC)
    if power.shape[1] > 1:
        peak_idx = 1 + np.argmax(power[:, 1:], axis=1)
    else:
        peak_idx = np.zeros(power.shape[0], dtype=int)
    dom_freq = freqs[peak_idx]

    # Spectral entropy (lower = more peaked, more wavelet-like)
    p_total = power.sum(axis=1, keepdims=True) + 1e-12
    p_norm = power / p_total
    spec_entropy = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1)

    # Spectral concentration: power within +/- 1 bin of peak / total
    peak_power = np.take_along_axis(power, peak_idx[:, None], axis=1).squeeze(1)
    spec_conc = peak_power / (power.sum(axis=1) + 1e-12)

    return {
        'central_energy_frac': central_energy,
        'effective_width': eff_width,
        'm0_normalized': m0_rel,
        'm1_normalized': m1_rel,
        'm2': m2,
        'dom_freq_cycles_per_unit_x': dom_freq,
        'spectral_entropy': spec_entropy,
        'spectral_concentration': spec_conc,
        'l2_energy': energy,
    }


def wavelet_match_scores(phi, x):
    """Best cross-correlation between each learned edge and parametric
    wavelet families (Mexican hat, Morlet), sweeping (scale, shift).
    Returns max correlation per edge for each family.
    """
    phi_np = phi.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    N = phi_np.shape[0]

    # Normalize each edge to unit L2 (zero-mean for fair comparison with wavelets)
    phi_zm = phi_np - phi_np.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(phi_zm, axis=1, keepdims=True) + 1e-12
    phi_u = phi_zm / norm

    scales = np.linspace(0.05, 0.6, 12)
    shifts = np.linspace(-0.6, 0.6, 13)

    def mexhat(t):
        return (1 - t ** 2) * np.exp(-t ** 2 / 2)

    def morlet(t, w0=5.0):
        return np.exp(-t ** 2 / 2) * np.cos(w0 * t)

    best_mex = np.zeros(N)
    best_morlet = np.zeros(N)
    for s in scales:
        for b in shifts:
            t = (x_np - b) / s
            mh = mexhat(t); mh -= mh.mean(); mh /= (np.linalg.norm(mh) + 1e-12)
            mo = morlet(t); mo -= mo.mean(); mo /= (np.linalg.norm(mo) + 1e-12)
            c_mex = np.abs(phi_u @ mh)
            c_mor = np.abs(phi_u @ mo)
            best_mex = np.maximum(best_mex, c_mex)
            best_morlet = np.maximum(best_morlet, c_mor)
    return {'mexhat_corr': best_mex, 'morlet_corr': best_morlet}


def random_baseline_layer(layer, x):
    """Build a clone of `layer` with re-randomized spline weights to act as a
    null model: same B-spline parameterization, no training signal. Returns
    the reconstructed phi tensor."""
    with torch.no_grad():
        spline_w = torch.randn_like(layer.spline_weight) * layer.spline_weight.std()
        scaler = torch.randn_like(layer.spline_scaler) * layer.spline_scaler.std()
        base_w = torch.randn_like(layer.base_weight) * layer.base_weight.std()

        # Temporarily swap
        orig = (layer.spline_weight.data.clone(),
                layer.spline_scaler.data.clone(),
                layer.base_weight.data.clone())
        layer.spline_weight.data.copy_(spline_w)
        layer.spline_scaler.data.copy_(scaler)
        layer.base_weight.data.copy_(base_w)
        phi = reconstruct_edges(layer, x)
        layer.spline_weight.data.copy_(orig[0])
        layer.spline_scaler.data.copy_(orig[1])
        layer.base_weight.data.copy_(orig[2])
    return phi


def aggregate(metrics_dict):
    """Compute mean/median across edges for each metric."""
    agg = {}
    for k, v in metrics_dict.items():
        agg[k] = {'mean': float(np.mean(v)), 'median': float(np.median(v)),
                  'std': float(np.std(v)), 'min': float(np.min(v)),
                  'max': float(np.max(v))}
    return agg


def plot_examples(phi_by_layer, x, out_pdf, n_examples=6, sort_key='mexhat_corr'):
    """Plot most wavelet-like and least wavelet-like edges per layer."""
    n_layers = len(phi_by_layer)
    fig, axes = plt.subplots(n_layers, 2, figsize=(10, 2.8 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]
    x_np = x.detach().cpu().numpy()
    for li, (name, phi, scores) in enumerate(phi_by_layer):
        phi_np = phi.detach().cpu().numpy()
        order = np.argsort(-scores[sort_key])  # descending wavelet-likeness
        top_idx = order[:n_examples]
        bot_idx = order[-n_examples:]
        for idx, color in zip(top_idx, plt.cm.Reds(np.linspace(0.4, 0.9, n_examples))):
            axes[li, 0].plot(x_np, phi_np[idx] / (np.abs(phi_np[idx]).max() + 1e-12),
                             color=color, alpha=0.7, linewidth=1.2)
        for idx, color in zip(bot_idx, plt.cm.Blues(np.linspace(0.4, 0.9, n_examples))):
            axes[li, 1].plot(x_np, phi_np[idx] / (np.abs(phi_np[idx]).max() + 1e-12),
                             color=color, alpha=0.7, linewidth=1.2)
        axes[li, 0].set_title(f'{name}: top-{n_examples} wavelet-like (Mexican hat)')
        axes[li, 1].set_title(f'{name}: bottom-{n_examples} wavelet-like')
        for ax in axes[li]:
            ax.axhline(0, color='k', alpha=0.3, linewidth=0.5)
            ax.set_xlabel('x'); ax.set_ylabel(r'$\phi_{ij}(x)$ (normalized)')
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    device = torch.device('cpu')
    print(f'[device] {device}')
    print(f'[ckpt]   {CKPT}')

    model = KANEnergyModel(n_filters=32, kan_hidden=[96, 16]).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model]  KAN-EBM ({n_params:,} params), kan layers={len(model.kan.layers)}')

    x = torch.linspace(X_RANGE[0], X_RANGE[1], N_X, dtype=torch.float32)

    record = {'checkpoint': str(CKPT), 'n_params': n_params,
              'n_x_grid': N_X, 'x_range': list(X_RANGE),
              'central_fraction': CENTRAL_FRACTION,
              'layers': {}}
    phi_by_layer = []

    for li, layer in enumerate(model.kan.layers):
        name = f'layer{li}_{layer.in_features}to{layer.out_features}'
        print(f'\n[layer {li}] {name}: {layer.in_features} -> {layer.out_features} '
              f'(grid={layer.grid_size}, order={layer.spline_order})')

        with torch.no_grad():
            phi = reconstruct_edges(layer, x)  # (out, in, N_x)
        N_edges = phi.shape[0] * phi.shape[1]
        phi_flat = phi.view(N_edges, N_X)

        mets = edge_metrics(phi_flat, x)
        wm = wavelet_match_scores(phi_flat, x)
        mets.update(wm)
        print(f'  edges={N_edges}')
        print(f'  mean |m0|/||phi|| = {mets["m0_normalized"].mean():.4f}   '
              f'(wavelet target: ~0)')
        print(f'  mean spectral entropy = {mets["spectral_entropy"].mean():.3f}   '
              f'(lower = more peaked)')
        print(f'  mean central energy frac (|x|<{CENTRAL_FRACTION}) = '
              f'{mets["central_energy_frac"].mean():.3f}')
        print(f'  mean effective 95% width = {mets["effective_width"].mean():.3f}   '
              f'(of total range {X_RANGE[1]-X_RANGE[0]:.1f})')
        print(f'  mean Mexican-hat corr = {mets["mexhat_corr"].mean():.3f}   '
              f'max = {mets["mexhat_corr"].max():.3f}')
        print(f'  mean Morlet corr      = {mets["morlet_corr"].mean():.3f}   '
              f'max = {mets["morlet_corr"].max():.3f}')

        # Null baseline: random B-splines, same parameterization
        with torch.no_grad():
            phi_null = random_baseline_layer(layer, x).view(N_edges, N_X)
        null_mets = edge_metrics(phi_null, x)
        null_wm = wavelet_match_scores(phi_null, x)
        null_mets.update(null_wm)
        print(f'  [null random splines] m0/||phi|| mean={null_mets["m0_normalized"].mean():.4f}, '
              f'mex_corr mean={null_mets["mexhat_corr"].mean():.3f}, '
              f'morlet_corr mean={null_mets["morlet_corr"].mean():.3f}')

        record['layers'][name] = {
            'in_features': layer.in_features,
            'out_features': layer.out_features,
            'n_edges': N_edges,
            'grid_size': layer.grid_size,
            'spline_order': layer.spline_order,
            'trained': aggregate(mets),
            'null_random': aggregate(null_mets),
        }
        phi_by_layer.append((name, phi_flat, mets))

    # Save JSON
    out_json = OUT_DIR / 'edge_spectra.json'
    out_json.write_text(json.dumps(record, indent=2))
    print(f'\n[save] {out_json}')

    # Plot
    plot_examples(phi_by_layer, x, OUT_DIR / 'edge_examples.pdf')
    print(f'[save] {OUT_DIR / "edge_examples.pdf"}')

    # Summary text
    lines = ['Spectral analysis of learned KAN-EBM edges',
             '=' * 60,
             f'Checkpoint: {CKPT.name} ({n_params:,} params)',
             f'Grid: {N_X} x-samples in {X_RANGE}',
             '']
    for name, phi, mets in phi_by_layer:
        agg_t = aggregate(mets)
        null = record['layers'][name]['null_random']
        lines.append(f'{name}: {record["layers"][name]["n_edges"]} edges')
        lines.append(f'  TRAINED  m0/||phi||={agg_t["m0_normalized"]["mean"]:.4f}  '
                     f'spec_ent={agg_t["spectral_entropy"]["mean"]:.3f}  '
                     f'central={agg_t["central_energy_frac"]["mean"]:.3f}  '
                     f'mex_corr={agg_t["mexhat_corr"]["mean"]:.3f}  '
                     f'morlet_corr={agg_t["morlet_corr"]["mean"]:.3f}')
        lines.append(f'  RANDOM   m0/||phi||={null["m0_normalized"]["mean"]:.4f}  '
                     f'spec_ent={null["spectral_entropy"]["mean"]:.3f}  '
                     f'central={null["central_energy_frac"]["mean"]:.3f}  '
                     f'mex_corr={null["mexhat_corr"]["mean"]:.3f}  '
                     f'morlet_corr={null["morlet_corr"]["mean"]:.3f}')
        d_mex = agg_t["mexhat_corr"]["mean"] - null["mexhat_corr"]["mean"]
        d_mor = agg_t["morlet_corr"]["mean"] - null["morlet_corr"]["mean"]
        lines.append(f'  delta wavelet match (trained-random): '
                     f'mexhat={d_mex:+.3f}  morlet={d_mor:+.3f}')
        lines.append('')
    out_txt = OUT_DIR / 'summary.txt'
    out_txt.write_text('\n'.join(lines))
    print(f'[save] {out_txt}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
