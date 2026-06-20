"""
Generate fig12_fashionmnist_generalization.pdf
Side-by-side visual comparison: MNIST-trained KAN-EBM and MLP-EBM
applied zero-shot to Fashion-MNIST at sigma=0.3
Sources: results/denoise/denoise_kan_fashion_noise0.3.png
         results/denoise/denoise_mlp_fashion_noise0.3.png
Output:  results/interpretability/fig12_fashionmnist_generalization.pdf
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

kan_path = ROOT / 'results' / 'denoise' / 'denoise_kan_fashion_noise0.3.png'
mlp_path = ROOT / 'results' / 'denoise' / 'denoise_mlp_fashion_noise0.3.png'

if not kan_path.exists() or not mlp_path.exists():
    print('Source images not found - skipping')
    sys.exit(0)

kan_img = mpimg.imread(str(kan_path))
mlp_img = mpimg.imread(str(mlp_path))

fig, axes = plt.subplots(2, 1, figsize=(10, 5.5))
fig.suptitle(
    'Fashion-MNIST zero-shot generalization (model trained only on MNIST, $\\sigma=0.3$)\n'
    'KAN-EBM (top) vs MLP-EBM (bottom)',
    fontsize=11, fontweight='bold'
)

for ax, img, label in zip(axes, [kan_img, mlp_img], ['KAN-EBM', 'MLP-EBM']):
    ax.imshow(img, aspect='auto')
    ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=6, va='center')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.tight_layout()
out = ROOT / 'results' / 'interpretability' / 'fig12_fashionmnist_generalization.pdf'
fig.savefig(str(out), bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'Saved: {out}')
