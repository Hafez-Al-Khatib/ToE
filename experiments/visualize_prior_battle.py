"""
Visualize the "Geometric Prior" Battle: KAN-EBM vs. Large SiLU-MLP.
Extracts the zero-shot slowness field n(x) from denoising models.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel as KANEBM
from exp_steel_man import SmoothMLPEBM
from exp_triple_role_steel import get_slowness
from train_phase3 import generate_maze_with_path

def visualize_prior_battle():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = ROOT / 'outputs' / 'prior_battle'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Models
    kan = KANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    kan_ckpt = ROOT / 'results' / 'kan_ebm_multitask.pt'
    if kan_ckpt.exists(): kan.load_state_dict(torch.load(kan_ckpt, map_location=device))
    
    mlp = SmoothMLPEBM(n_channels=1, n_filters=16, filter_size=5, hidden_dim=128).to(device)
    # We use a trained MLP from previous runs if possible, or just look at structural bias
    
    # 2. Generate Test Maze
    torch.manual_seed(42)
    maze, _, _, _ = generate_maze_with_path(size=32, device=device)
    
    # 3. Extract n(x)
    n_kan = get_slowness(kan, maze)
    n_mlp = get_slowness(mlp, maze)
    
    # 4. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(maze[0, 0].cpu(), cmap='gray')
    axes[0].set_title("Input Maze (Perception)")
    
    im1 = axes[1].imshow(n_kan[0].cpu(), cmap='hot')
    axes[1].set_title("KAN-EBM Zero-Shot n(x)\n(Learned 'Strokiness' Prior)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(n_mlp[0].cpu(), cmap='hot')
    axes[2].set_title("SiLU-MLP Zero-Shot n(x)\n(Global Interference)")
    plt.colorbar(im2, ax=axes[2])
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'prior_comparison.png', dpi=200)
    plt.close()
    print(f"Comparison saved to {out_dir}/prior_comparison.png")

if __name__ == '__main__':
    visualize_prior_battle()
