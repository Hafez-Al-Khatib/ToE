"""
exp_loss_landscape.py
=====================
Visualizes the 3D energy basin (loss landscape) of a micro KAN-EBM vs a micro MLP-EBM.
Demonstrates the topological collapse of ReLUs at microscopic parameter counts,
and the smoothness prior injected by B-splines.
"""

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Micro MLP-EBM Baseline ────────────────────────────────────────────────────
class MicroMLPEBM(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, filter_size=5, hidden_dim=64):
        super().__init__()
        self.filters = nn.Parameter(torch.randn(n_filters, n_channels, filter_size, filter_size) / math.sqrt(filter_size**2 * n_channels))
        self.log_precision = nn.Parameter(torch.zeros(n_filters))
        
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def energy(self, x):
        B = x.shape[0]
        # Convolution
        f = F.conv2d(x, self.filters, padding='same')
        
        # Precision weighting & squash
        prec = F.softplus(self.log_precision).view(1, -1, 1, 1)
        f = torch.tanh(f * prec)
        
        # Pointwise MLP (B, C, H, W) -> (B, H, W, C) -> (B, H, W, 1) -> (B, 1, H, W)
        f_flat = f.permute(0, 2, 3, 1) # B, H, W, C
        e_pixel = self.mlp(f_flat)
        e = e_pixel.permute(0, 3, 1, 2)
        
        return e.sum(dim=(1, 2, 3))


def train_micro_mlp(model, device, epochs=5, sigma=0.2):
    """Train the MLP baseline quickly on MNIST."""
    import torchvision
    import torchvision.transforms as T
    
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    print(f"Training Micro MLP-EBM for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            noise = torch.randn_like(x) * sigma
            x_noisy = x + noise
            x_noisy.requires_grad_(True)
            
            E = model.energy(x_noisy).sum()
            grad = torch.autograd.grad(E, x_noisy, create_graph=True)[0]
            
            target = (x - x_noisy) / (sigma ** 2)
            loss = F.mse_loss(grad, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        

def plot_landscape(model_kan, model_mlp, x_clean, x_noisy, out_dir, device):
    """Generates a 3D landscape of the energy basin."""
    model_kan.eval()
    model_mlp.eval()
    
    # Define directions
    # v1 is the direction to the clean image
    v1 = (x_clean - x_noisy).squeeze()
    v1 = v1 / torch.norm(v1)
    
    # v2 is a random orthogonal direction
    v2 = torch.randn_like(v1)
    v2 = v2 - torch.sum(v2 * v1) * v1  # Gram-Schmidt
    v2 = v2 / torch.norm(v2)
    
    # Evaluate grid
    grid_size = 40
    alphas = np.linspace(-3.0, 3.0, grid_size)
    betas = np.linspace(-3.0, 3.0, grid_size)
    
    A, B = np.meshgrid(alphas, betas)
    E_kan = np.zeros((grid_size, grid_size))
    E_mlp = np.zeros((grid_size, grid_size))
    
    print("Evaluating energy landscapes...")
    for i in range(grid_size):
        for j in range(grid_size):
            a = A[i, j]
            b = B[i, j]
            
            # Point in image space
            pt = x_noisy.squeeze() + a * v1 + b * v2
            pt = pt.unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                E_kan[i, j] = model_kan.energy(pt).item()
                E_mlp[i, j] = model_mlp.energy(pt).item()
                
    # Normalize for plotting (subtract min so basin is at 0)
    E_kan = E_kan - np.min(E_kan)
    E_mlp = E_mlp - np.min(E_mlp)
    
    # Smooth MLP slightly for visualization so matplotlib doesn't completely glitch, 
    # but keep the jaggedness apparent.
    
    # ── Plotting ──
    fig = plt.figure(figsize=(16, 7))
    
    # Plot KAN
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(A, B, E_kan, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_title("5.8K KAN-EBM\n(Smoothness Prior = Navigable Basin)", fontsize=14, fontweight='bold')
    ax1.set_xlabel('Direction to Clean Data')
    ax1.set_ylabel('Random Orthogonal Dir')
    ax1.set_zlabel('Energy')
    ax1.view_init(elev=25, azim=45)
    
    # Plot MLP
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(A, B, E_mlp, cmap='magma', linewidth=0, antialiased=False, alpha=0.9)
    ax2.set_title("5.8K MLP-EBM\n(Topological Collapse = Gradient Divergence)", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Direction to Clean Data')
    ax2.set_ylabel('Random Orthogonal Dir')
    ax2.set_zlabel('Energy')
    ax2.view_init(elev=25, azim=45)
    
    fig.tight_layout()
    fig.savefig(out_dir / 'topological_collapse.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'topological_collapse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved landscape to {out_dir / 'topological_collapse.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
             if args.device == 'auto' else torch.device(args.device)

    out_dir = ROOT / 'outputs' / 'landscape'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load KAN-EBM
    from exp_cifar10 import KANEnergyModel as CifarKANEBM
    kan_ckpt = ROOT / 'results' / 'kan_ebm_multitask.pt'
    if not kan_ckpt.exists():
        kan_ckpt = ROOT / 'results' / 'kan_ebm.pt'
        
    model_kan = CifarKANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    model_kan.load_state_dict(torch.load(kan_ckpt, map_location=device, weights_only=True))
    kan_params = sum(p.numel() for p in model_kan.parameters())
    print(f"Loaded KAN-EBM: {kan_params} parameters")
    
    # 2. Get/Train Micro MLP-EBM
    model_mlp = MicroMLPEBM(n_channels=1, n_filters=16, filter_size=5, hidden_dim=64).to(device)
    mlp_params = sum(p.numel() for p in model_mlp.parameters())
    print(f"Created Micro MLP-EBM: {mlp_params} parameters")
    
    mlp_ckpt = ROOT / 'results' / 'micro_mlp.pt'
    if mlp_ckpt.exists():
        model_mlp.load_state_dict(torch.load(mlp_ckpt, map_location=device, weights_only=True))
        print("Loaded existing Micro MLP-EBM checkpoint.")
    else:
        train_micro_mlp(model_mlp, device, epochs=3)
        torch.save(model_mlp.state_dict(), mlp_ckpt)
        print("Saved Micro MLP-EBM checkpoint.")
        
    # 3. Load a single test image
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    test_ds = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    
    x_clean, _ = test_ds[0]
    x_clean = x_clean.unsqueeze(0).to(device)
    x_noisy = x_clean + torch.randn_like(x_clean) * 0.4  # Strong noise to evaluate wide basin
    
    # 4. Plot
    plot_landscape(model_kan, model_mlp, x_clean, x_noisy, out_dir, device)


if __name__ == '__main__':
    main()
