"""
Proof of Theory: Thermodynamic Neural Field (v3)
================================================

Trains and evaluates a Thermodynamic Neural Field (TNF) — NOT an MLP,
NOT a ConvNet. The TNF is a physical system:

- Data = physical field on a 2D grid
- Energy = generalized Ginzburg-Landau functional with LEARNED spatial filters
- Inference = energy gradient descent (no separate PDE)
- Training = 1-step denoising: L = ‖x - η∇E(x+noise) - x_clean‖²

All 1690 parameters are fully differentiable. No KAN indexing.

Ablation Mode (New for Phase 4):
- Supports disabling diffusion (α=0)
- Supports disabling potentials (V(u)=u²)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from thermodynamic_field import ThermodynamicField, one_step_denoising_loss


class AblatedThermodynamicField(ThermodynamicField):
    """
    ThermodynamicField with specific components disabled for ablation studies.
    
    Modes:
    - 'no_diffusion': Force α = 0 (no spatial smoothing/diffusion)
    - 'no_potential': Force V(u) = u² (standard L2 attractor, no complex wells)
    """
    def __init__(self, ablation_mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_mode = ablation_mode
        print(f"!!! ABLATION MODE: {ablation_mode} !!!")
    
    def compute_energy(self, fields: torch.Tensor) -> torch.Tensor:
        """Overridden energy functional for ablation."""
        B, C, H, W = fields.shape
        energy = torch.zeros(B, device=fields.device)
        
        # Original parameters
        alphas = self.alphas
        pad = self.spatial_filters[0].shape[-1] // 2
        
        # ABLATION: Zero out diffusion if requested
        if self.ablation_mode == 'no_diffusion':
            alphas = torch.zeros_like(alphas)
        
        for i in range(C):
            u_i = fields[:, i:i+1, :, :]
            
            # 1. Spatial filter energy (unchanged)
            filtered = F.conv2d(u_i, self.spatial_filters[i], padding=pad)
            filter_energy = (self.filter_weights[i].abs().view(1, -1, 1, 1) *
                           filtered.pow(2)).sum(dim=(1, 2, 3))
            energy = energy + filter_energy
            
            # 2. Gradient energy (Diffusion) - ABLATED if no_diffusion
            if self.ablation_mode != 'no_diffusion':
                grad_x = u_i[:, :, :, 1:] - u_i[:, :, :, :-1]
                grad_y = u_i[:, :, 1:, :] - u_i[:, :, :-1, :]
                grad_energy = alphas[i] * (grad_x.pow(2).sum(dim=(1,2,3)) +
                                           grad_y.pow(2).sum(dim=(1,2,3)))
                energy = energy + grad_energy
            
            # 3. Polynomial potential
            if self.ablation_mode == 'no_potential':
                # ABLATION: Standard L2 Potential V(u) = 0.5 * u^2 (simple attractor to 0)
                # Effectively just weight decay on the field
                pot = 0.5 * u_i.pow(2)
            else:
                # Full learnable potential
                pot = self.potential_a[i] * u_i.pow(2) + self.potential_b[i] * u_i.pow(4)
            
            energy = energy + pot.sum(dim=(1, 2, 3))
        
        # 4. Bilinear coupling (unchanged)
        idx = 0
        for i in range(C):
            for j in range(i + 1, C):
                coupling_e = self.coupling_weights[idx] * (
                    fields[:, i] * fields[:, j]
                ).sum(dim=(1, 2))
                energy = energy + coupling_e
                idx += 1
        
        # Normalize
        energy = energy / (H * W)
        
        return energy


class TrainingVisualizer:
    """Tracks and visualizes training dynamics."""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.history = {
            "epoch": [],
            "energy_clean": [],
            "energy_noisy": [],
            "energy_gap": [],
            "psnr": [],
            "loss": [],
        }
        self.energy_landscapes = []
    
    def log(self, epoch: int, metrics: Dict):
        """Log metrics for one epoch."""
        self.history["epoch"].append(epoch)
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)
    
    def capture_energy_landscape(
        self, 
        model: ThermodynamicField,
        sample: torch.Tensor,
        epoch: int
    ):
        """Capture 2D slice of energy landscape for visualization."""
        device = next(model.parameters()).device
        
        with torch.no_grad():
            x_range = torch.linspace(-2, 2, 50, device=device)
            y_range = torch.linspace(-2, 2, 50, device=device)
            
            energies = torch.zeros(50, 50)
            # Use a real sample reshaped to image
            base = sample.mean(dim=0)
            if base.dim() == 1:
                base = base.view(1, 28, 28)
            
            for i, dx in enumerate(x_range):
                for j, dy in enumerate(y_range):
                    perturbed = base.clone()
                    perturbed[0, 0, 0] = dx.item()
                    perturbed[0, 0, 1] = dy.item()
                    energies[i, j] = model(perturbed.unsqueeze(0).view(1, -1)).item()
        
        self.energy_landscapes.append({
            "epoch": epoch,
            "energies": energies.cpu().numpy(),
            "x_range": x_range.cpu().numpy(),
            "y_range": y_range.cpu().numpy()
        })
    
    def plot_training_dynamics(self, dataset_name: str):
        """Plot how energy gap evolves - shows 'intelligence emerging'."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = self.history["epoch"]
        
        # Plot 1: Energy separation
        ax = axes[0, 0]
        ax.plot(epochs, self.history["energy_clean"], 'g-', label="E(clean)")
        ax.plot(epochs, self.history["energy_noisy"], 'r--', label="E(noisy)")
        ax.set_title("The Energy Gap (Signal vs Noise)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy Gap
        ax = axes[0, 1]
        ax.plot(epochs, self.history["energy_gap"], 'b-', linewidth=2)
        ax.set_title("Energy Gap Magnitude")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ΔE")
        ax.fill_between(epochs, self.history["energy_gap"], alpha=0.2)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: PSNR
        ax = axes[1, 0]
        ax.plot(epochs, self.history["psnr"], 'k-', label="Restoration Quality")
        ax.set_title("Perception Quality (PSNR)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("dB")
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss
        ax = axes[1, 1]
        ax.plot(epochs, self.history["loss"], 'purple')
        ax.set_title("Denoising Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"{dataset_name}: Emergence of Order from Chaos", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{dataset_name}_dynamics.png", dpi=150)
        plt.close()
        
    def plot_energy_landscape_evolution(self, dataset_name: str):
        """Show how energy landscape settles over training."""
        if not self.energy_landscapes:
            return
        
        n_snapshots = min(len(self.energy_landscapes), 6)
        indices = np.linspace(0, len(self.energy_landscapes)-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            snapshot = self.energy_landscapes[idx]
            ax = axes[i]
            
            im = ax.contourf(snapshot["x_range"], snapshot["y_range"], 
                            snapshot["energies"].T, levels=30, cmap='viridis')
            ax.set_title(f'Epoch {snapshot["epoch"]}')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            plt.colorbar(im, ax=ax, label='Energy')
        
        plt.suptitle(f'{dataset_name}: Energy Landscape Evolution\n(Should develop clear valleys for clean data)', 
                    fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / f"{dataset_name}_landscape_evolution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def get_mnist_data(train: bool = True, fashion: bool = False):
    """Get MNIST or Fashion-MNIST data as 2D images (not flattened)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] → [-1, 1]
    ])
    
    DatasetClass = datasets.FashionMNIST if fashion else datasets.MNIST
    dataset = DatasetClass('./data', train=train, download=True, transform=transform)
    return dataset


def add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add Gaussian noise."""
    return x + torch.randn_like(x) * noise_std


def compute_psnr(clean: torch.Tensor, denoised: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = (clean - denoised).pow(2).mean()
    if mse < 1e-10:
        return float('inf')
    return (10 * torch.log10(4.0 / mse)).item()


def train_and_visualize(
    dataset_name: str,
    model: ThermodynamicField,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    noise_std: float = 0.3,
    lr: float = 1e-3
):
    print(f"\nTraining on {dataset_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vis = TrainingVisualizer()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            # Create noise
            loss, metrics = one_step_denoising_loss(model, data, noise_std=noise_std)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # End of epoch stats
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation stats on one batch
        model.eval()
        with torch.no_grad():
            sample = next(iter(train_loader))[0][:32].to(device)
            noisy = sample + torch.randn_like(sample) * noise_std
            
            # Energy gap check
            E_clean = model.compute_energy_from_image(sample).mean().item()
            E_noisy = model.compute_energy_from_image(noisy).mean().item()
            
            # PSNR check
            with torch.enable_grad():
                denoised = model.one_step_denoise(noisy)
            mse = F.mse_loss(denoised, sample)
            psnr = -10 * torch.log10(mse).item()
            
            vis.log(epoch, {
                "energy_clean": E_clean,
                "energy_noisy": E_noisy,
                "energy_gap": E_noisy - E_clean,
                "psnr": psnr,
                "loss": avg_loss
            })
            
            if epoch % 5 == 0:
                vis.capture_energy_landscape(model, sample, epoch)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={psnr:.1f}dB, Gap={E_noisy - E_clean:.2f}")
    
    vis.plot_training_dynamics(dataset_name)
    vis.plot_energy_landscape_evolution(dataset_name)
    return model, vis


def evaluate_denoising(
    model: ThermodynamicField,
    test_loader: DataLoader,
    device: torch.device,
    noise_std: float,
    dataset_name: str,
    n_steps: int = 50
):
    print(f"\nEvaluating {dataset_name} (Inferring with {n_steps} steps)...")
    model.eval()
    
    # Visualize one batch
    data, _ = next(iter(test_loader))
    data = data.to(device)[:8]  # Take 8 images
    noisy = data + torch.randn_like(data) * noise_std
    
    # In-place evolution
    current = noisy.clone()
    for _ in range(n_steps):
        current = model.one_step_denoise(current).detach()
    
    # Calculate PSNR
    mse = F.mse_loss(current, data)
    psnr = -10 * torch.log10(mse).item()
    print(f"Final Test PSNR: {psnr:.2f} dB")
    
    # Save visualization
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        # Original
        axes[0, i].imshow(data[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")
        
        # Noisy
        axes[1, i].imshow(noisy[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"Noisy (σ={noise_std})")
        
        # Denoised
        axes[2, i].imshow(current[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(f"Restored ({n_steps} steps)")
    
    plt.tight_layout()
    plt.savefig(f"./outputs/{dataset_name}_evaluation.png")
    plt.close()
    
    return psnr


def main():
    parser = argparse.ArgumentParser(description="Prove Natural Intelligence Theory with TNF")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--n-channels", type=int, default=4, help="Number of field channels")
    parser.add_argument("--n-filters", type=int, default=16, help="Spatial filters per channel")
    parser.add_argument("--evolve-steps", type=int, default=50, help="Gradient descent steps for denoising")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_diffusion", "no_potential"],
                        help="Ablation study mode")
    parser.add_argument("--skip-mnist", action="store_true")
    parser.add_argument("--skip-fashion", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Architecture: Thermodynamic Neural Field ({args.n_channels}-channel, {args.n_filters} filters)")
    print(f"Training method: 1-Step Denoising (end-to-end)")
    print(f"Ablation Mode: {args.ablation}")
    
    os.makedirs("./outputs", exist_ok=True)
    
    results = {}
    
    # helper to create model
    def create_model():
        if args.ablation == "none":
            return ThermodynamicField(
                n_channels=args.n_channels, n_filters=args.n_filters
            ).to(device)
        else:
            return AblatedThermodynamicField(
                ablation_mode=args.ablation,
                n_channels=args.n_channels, n_filters=args.n_filters
            ).to(device)

    # =========================================================================
    # Dataset 1: MNIST
    # =========================================================================
    if not args.skip_mnist:
        train_data = get_mnist_data(train=True, fashion=False)
        test_data = get_mnist_data(train=False, fashion=False)
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        
        model = create_model()
        
        model, _ = train_and_visualize(
            f"MNIST_{args.ablation}", model, train_loader, device,
            epochs=args.epochs, noise_std=args.noise_std, lr=args.lr
        )
        
        psnr = evaluate_denoising(
            model, test_loader, device, args.noise_std, f"MNIST_{args.ablation}",
            n_steps=args.evolve_steps
        )
        results["MNIST"] = psnr
        
        torch.save(model.state_dict(), f"./outputs/mnist_tnf_{args.ablation}.pt")
    
    # =========================================================================
    # Dataset 2: Fashion-MNIST
    # =========================================================================
    if not args.skip_fashion:
        train_data = get_mnist_data(train=True, fashion=True)
        test_data = get_mnist_data(train=False, fashion=True)
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        
        model = create_model()
        
        model, _ = train_and_visualize(
            f"FashionMNIST_{args.ablation}", model, train_loader, device,
            epochs=args.epochs, noise_std=args.noise_std, lr=args.lr
        )
        
        psnr = evaluate_denoising(
            model, test_loader, device, args.noise_std, f"FashionMNIST_{args.ablation}",
            n_steps=args.evolve_steps
        )
        results["FashionMNIST"] = psnr
        
        torch.save(model.state_dict(), f"./outputs/fashion_tnf_{args.ablation}.pt")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*60)
    print(f"PROOF OF THEORY: THERMODYNAMIC NEURAL FIELD (Ablation: {args.ablation})")
    print("="*60)
    
    print("\nThe Natural Intelligence theory states:")
    print("1. Data is a physical field")
    print("2. Intelligence is energy minimization")
    print("3. Inference is relaxation")
    
    print("\nResults supporting this:")
    for name, psnr in results.items():
        print(f"  {name}: {psnr:.2f} dB (Restored via Energy Gradient Descent)")
        
    print(f"\nSaved models to ./outputs/*_{args.ablation}.pt")
    print("Visualizations saved to ./outputs/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
