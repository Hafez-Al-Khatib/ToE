"""
Generate fig10_natural_psnr_vs_k.pdf
PSNR vs K for KAN-EBM on CBSD68 at σ ∈ {15, 25, 50}
Sources: outputs/natural_images/natural_results.json
Output:  results/interpretability/fig10_natural_psnr_vs_k.pdf
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open(ROOT / 'outputs' / 'natural_images' / 'natural_results.json') as f:
    data = json.load(f)

K_vals   = [1, 5, 10, 20]
sigmas   = [15, 25, 50]
colors   = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
markers  = ['o', 's', '^']

# ── Figure 1: PSNR-vs-K for KAN-EBM on CBSD68 (all σ) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

for ax_idx, (dataset_key, dataset_name) in enumerate([('CBSD68', 'CBSD68'), ('Set12', 'Set12')]):
    ax = axes[ax_idx]

    for col_i, sigma in enumerate(sigmas):
        key = f'sigma_{sigma}'
        kan = data[dataset_key][key]['KAN-EBM']
        ffn = data[dataset_key][key]['FFN']

        psnrs_kan = [float(kan[str(k)]['psnr']) for k in K_vals]
        psnr_ffn  = float(ffn['1']['psnr'])   # FFN is flat — use K=1

        ax.plot(K_vals, psnrs_kan,
                color=colors[col_i], marker=markers[col_i],
                linewidth=2, markersize=6,
                label=f'KAN-EBM σ={sigma}')
        ax.axhline(psnr_ffn, color=colors[col_i], linestyle='--',
                   linewidth=1.2, alpha=0.55,
                   label=f'FFN σ={sigma} (flat)')

    ax.set_xlabel('Inference steps $K$', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(f'{dataset_name} (in-distribution)', fontsize=11, fontweight='bold')
    ax.set_xticks(K_vals)
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.legend(fontsize=8, loc='lower right', ncol=2)

fig.suptitle('KAN-EBM test-time K-scaling on natural images\n(trained on CBSD68 patches)',
             fontsize=11, fontweight='bold')
fig.tight_layout()

out = ROOT / 'results' / 'interpretability' / 'fig10_natural_psnr_vs_k.pdf'
fig.savefig(out, bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'Saved: {out}')

# ── Figure 2: PSNR-vs-K bar chart for σ=15, K=1/10/20, KAN vs FFN vs MLP ──
fig2, ax2 = plt.subplots(figsize=(8, 3.8))

dataset_key = 'CBSD68'
sigma = 15
key   = f'sigma_{sigma}'
k_plot = [1, 10, 20]
x     = np.arange(len(k_plot))
width = 0.22

kan_vals = [float(data[dataset_key][key]['KAN-EBM'][str(k)]['psnr']) for k in k_plot]
mlp_vals = [float(data[dataset_key][key]['MLP-EBM'][str(k)]['psnr']) for k in k_plot]
ffn_vals = [float(data[dataset_key][key]['FFN'][str(k)]['psnr'])     for k in k_plot]

ax2.bar(x - width, kan_vals, width, label='KAN-EBM (ours)', color='#1f77b4', edgecolor='black', linewidth=0.5)
ax2.bar(x,         mlp_vals, width, label='MLP-EBM',         color='#d62728', edgecolor='black', linewidth=0.5)
ax2.bar(x + width, ffn_vals, width, label='FFN (flat)',      color='#7f7f7f', edgecolor='black', linewidth=0.5, alpha=0.7)

# Published baselines as horizontal lines
ax2.axhline(33.52, color='green',  linestyle=':', linewidth=1.5, label='BM3D (33.52 dB)')
ax2.axhline(33.90, color='purple', linestyle=':', linewidth=1.5, label='DnCNN (33.90 dB)')

ax2.set_xticks(x)
ax2.set_xticklabels([f'K={k}' for k in k_plot], fontsize=11)
ax2.set_ylabel('PSNR (dB)', fontsize=11)
ax2.set_title(f'CBSD68 denoising (σ=15/255, in-distribution training)\nKAN-EBM monotonically improves; MLP-EBM stagnates',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim(20, 36)

fig2.tight_layout()
out2 = ROOT / 'results' / 'interpretability' / 'fig11_natural_bar.pdf'
fig2.savefig(out2, bbox_inches='tight', dpi=200)
plt.close(fig2)
print(f'Saved: {out2}')

# ── Figure 3: Fashion-MNIST generalization strip (training on MNIST, test on Fashion) ──
# Already exists: results/denoise/denoise_kan_fashion_noise0.3.png
print('Fashion-MNIST strip already at results/denoise/denoise_kan_fashion_noise0.3.png')
print('Done.')
