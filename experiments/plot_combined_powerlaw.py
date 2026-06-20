import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

kan = json.load(open(ROOT / 'outputs/fine_grid_kstar/fine_grid_kstar.json'))
convmlp_gelu = json.load(open(ROOT / 'outputs/fine_grid_kstar/fine_grid_convmlp_gelu.json'))
convmlp_silu = json.load(open(ROOT / 'outputs/fine_grid_kstar/fine_grid_convmlp_silu.json'))
convmlp_tanh = json.load(open(ROOT / 'outputs/fine_grid_kstar/fine_grid_convmlp_tanh.json'))
convmlp_relu = json.load(open(ROOT / 'outputs/fine_grid_kstar/fine_grid_convmlp_relu.json'))
unet = json.load(open(ROOT / 'outputs/fine_grid_kstar/fine_grid_unet.json'))

fig, ax = plt.subplots(figsize=(6, 4.5))
sigmas = [0.05, 0.1, 0.15, 0.2, 0.3]

def plot_arch(data, label, color, marker, zorder, linestyle='-'):
    K = [data['per_sigma'][f'{s:.3f}']['kstar_mean'] for s in sigmas]
    pl = data['power_law']
    if 'fine_grid' in pl:
        pl = pl['fine_grid']
    alpha = pl['alpha']
    r2 = pl['R2']
    C = pl['C']
    fit = [C * (s ** alpha) for s in sigmas]
    ax.plot(sigmas, K, marker=marker, color=color, linestyle=linestyle,
            label=f"{label}  ($\\alpha$={alpha:.3f}, $R^2$={r2:.3f})",
            markersize=6, zorder=zorder, linewidth=1.8)
    ax.plot(sigmas, fit, '--', color=color, alpha=0.35, linewidth=1.2, zorder=zorder-1)

plot_arch(kan, 'KAN-EBM', '#d62728', 'o', 10)
plot_arch(unet, 'U-Net EBM', '#2ca02c', 's', 9)
plot_arch(convmlp_relu, 'ConvMLP-ReLU', '#9467bd', 'D', 8)
plot_arch(convmlp_tanh, 'ConvMLP-Tanh', '#8c564b', 'v', 7)
plot_arch(convmlp_gelu, 'ConvMLP-GELU', '#1f77b4', '^', 6)
plot_arch(convmlp_silu, 'ConvMLP-SiLU', '#17becf', 'p', 5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Noise level $\\sigma$', fontsize=12)
ax.set_ylabel('Optimal steps $K^*(\\sigma)$', fontsize=12)
ax.set_title('Architecture-dependent scaling exponents (fine-grid, 500 images)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out_dir = ROOT / 'outputs/fine_grid_kstar/figures'
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'combined_powerlaw_all6.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out_dir / 'combined_powerlaw_all6.png', dpi=300, bbox_inches='tight')
print('Saved combined plot with all 6 architectures')
