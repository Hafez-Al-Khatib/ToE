"""Render top-N Gabor-fit trained filters alongside their best-fit Gabors."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel  # noqa: E402
from exp_spectral_conv_filters import gabor_2d  # noqa: E402

CKPT = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
SPEC_JSON = ROOT / 'outputs' / 'spectral_kan' / 'conv_filter_spectra.json'
OUT = ROOT / 'outputs' / 'spectral_kan' / 'gabor_match_grid.pdf'

device = torch.device('cpu')
model = KANEnergyModel(n_filters=32, kan_hidden=[96, 16]).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False))
model.eval()
filters = model.filters.detach().cpu().numpy()

spec = json.loads(SPEC_JSON.read_text())
top = spec['top10_gabor_examples']

N = 8
fig, axes = plt.subplots(N, 3, figsize=(6.5, 2.0 * N))
for row, r in enumerate(top[:N]):
    fi, ci = r['filter_idx'], r['channel_idx']
    sim = r['gabor_similarity']
    gp = r['gabor_params']
    f = filters[fi, ci]
    f_zm = f - f.mean()
    f_n = f_zm / (np.linalg.norm(f_zm) + 1e-12)
    g = gabor_2d(5, gp['sigma'], gp['freq'], gp['theta'], gp['phase'])
    # Render
    vmax = max(abs(f_n.min()), abs(f_n.max()))
    axes[row, 0].imshow(f_n, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[row, 0].set_title(f'Learned filter [f{fi}, ch{ci}]', fontsize=10)
    vmax_g = max(abs(g.min()), abs(g.max()))
    axes[row, 1].imshow(g, cmap='RdBu_r', vmin=-vmax_g, vmax=vmax_g, interpolation='nearest')
    axes[row, 1].set_title(
        f'Best Gabor: theta={gp["theta"]*180/math.pi:.0f}deg, '
        f'freq={gp["freq"]:.1f}, sig={gp["sigma"]:.2f}', fontsize=9)
    diff = f_n - g / (np.linalg.norm(g) + 1e-12)
    vmax_d = max(abs(diff.min()), abs(diff.max()))
    axes[row, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, interpolation='nearest')
    axes[row, 2].set_title(f'Residual (cos sim = {sim:.3f})', fontsize=9)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Trained KAN-EBM filters spontaneously learn Gabor (V1-like) structure',
             fontsize=12, y=1.005)
plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'[save] {OUT}')
