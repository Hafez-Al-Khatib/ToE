"""
Visualization Tools for Energy-Based Models
============================================

This module provides tools to visualize:
1. Energy landscapes (2D slices through high-dimensional space)
2. Langevin trajectories (how samples evolve over time)
3. Denoising comparisons (before/after)
4. Energy histograms (distribution of energies)

Visualization is critical for understanding energy-based models.
Unlike feedforward networks where we analyze logits or attention,
EBMs learned a "landscape" — best understood visually.

Key Insights to Look For
------------------------
1. **Clean data in valleys**: Energy should be lowest at real data points
2. **Gradients toward data**: Arrows/trajectories should point to valleys
3. **Separable modes**: Each digit class should have its own basin
4. **Smooth landscape**: No extreme spikes (indicates training issues)
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from energy_model import EnergyMLP, langevin_dynamics


def plot_energy_landscape_2d(
    model: nn.Module,
    center: torch.Tensor,
    direction1: torch.Tensor,
    direction2: torch.Tensor,
    range_: float = 2.0,
    n_points: int = 50,
    device: str = "cpu",
    title: str = "Energy Landscape",
    save_path: Optional[str] = None
):
    """
    Plot a 2D slice of the energy landscape.
    
    The energy function E(x) lives in 784-dimensional space (for MNIST).
    We can't visualize this directly. Instead, we pick a plane through
    this space and plot E on that plane.
    
    Mathematical Setup
    ------------------
    Given:
    - Center point c ∈ ℝ⁷⁸⁴ (typically a real image)
    - Two orthogonal directions v₁, v₂ ∈ ℝ⁷⁸⁴
    
    We parameterize the plane:
        x(α, β) = c + α·v₁ + β·v₂
    
    And plot E(x(α, β)) as a heatmap or surface.
    
    Choosing Directions
    -------------------
    Good choices:
    1. Random orthogonal vectors (generic view)
    2. Principal components of data (variance directions)
    3. Gradient direction + orthogonal (shows steepest descent)
    4. Difference between two classes (class boundary)
    
    Parameters
    ----------
    model : nn.Module
        Energy network E(x).
    
    center : torch.Tensor
        Center point of the plane, shape (784,).
        Typically a real image.
    
    direction1, direction2 : torch.Tensor
        Basis vectors for the plane, shape (784,).
        Should be normalized.
    
    range_ : float
        How far to extend in each direction.
        Larger = wider view, may miss local structure.
    
    n_points : int
        Grid resolution. Higher = smoother but slower.
    
    device : str
        Computation device.
    
    title : str
        Plot title.
    
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    model.eval()
    model = model.to(device)
    
    center = center.to(device)
    direction1 = direction1.to(device)
    direction2 = direction2.to(device)
    
    # Normalize directions
    direction1 = direction1 / direction1.norm()
    direction2 = direction2 / direction2.norm()
    
    # Make direction2 orthogonal to direction1 (Gram-Schmidt)
    direction2 = direction2 - (direction2 @ direction1) * direction1
    direction2 = direction2 / direction2.norm()
    
    # Create grid
    alphas = torch.linspace(-range_, range_, n_points, device=device)
    betas = torch.linspace(-range_, range_, n_points, device=device)
    
    # Compute energy on grid
    energies = torch.zeros(n_points, n_points)
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            x = center + alpha * direction1 + beta * direction2
            with torch.no_grad():
                energies[i, j] = model(x.unsqueeze(0)).item()
    
    # Convert to numpy
    alphas_np = alphas.cpu().numpy()
    betas_np = betas.cpu().numpy()
    energies_np = energies.cpu().numpy()
    
    # Create figure with two subplots: heatmap and 3D surface
    fig = plt.figure(figsize=(14, 5))
    
    # Subplot 1: 2D heatmap
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(
        energies_np.T, extent=[alphas_np[0], alphas_np[-1], betas_np[0], betas_np[-1]],
        origin='lower', cmap='viridis', aspect='auto'
    )
    ax1.set_xlabel("Direction 1 (α)")
    ax1.set_ylabel("Direction 2 (β)")
    ax1.set_title(f"{title} (2D slice)")
    ax1.plot(0, 0, 'r*', markersize=15, label='Center')
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Energy E(x)")
    
    # Add gradient arrows
    # Compute gradient at center and plot as arrow
    x_center = center.unsqueeze(0).requires_grad_(True)
    E_center = model(x_center)
    grad = torch.autograd.grad(E_center, x_center)[0].squeeze()
    
    # Project gradient onto the plane
    grad_alpha = (grad @ direction1).item()
    grad_beta = (grad @ direction2).item()
    
    # Normalize for visibility
    grad_norm = np.sqrt(grad_alpha**2 + grad_beta**2)
    if grad_norm > 0.01:
        ax1.arrow(0, 0, -grad_alpha/grad_norm * 0.3, -grad_beta/grad_norm * 0.3,
                  head_width=0.1, head_length=0.05, fc='red', ec='red', label='−∇E')
    
    # Subplot 2: 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    A, B = np.meshgrid(alphas_np, betas_np)
    ax2.plot_surface(A, B, energies_np.T, cmap='viridis', alpha=0.8, linewidth=0)
    ax2.set_xlabel("α")
    ax2.set_ylabel("β")
    ax2.set_zlabel("E(x)")
    ax2.set_title(f"{title} (3D surface)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved landscape to {save_path}")
    
    return fig


def plot_langevin_trajectory(
    model: nn.Module,
    x_init: torch.Tensor,
    n_steps: int = 100,
    step_size: float = 10.0,
    device: str = "cpu",
    title: str = "Langevin Trajectory",
    save_path: Optional[str] = None
):
    """
    Visualize how a sample evolves during Langevin dynamics.
    
    This shows the "rolling ball" intuition:
    - Start from a noisy/random point
    - Follow the negative gradient of energy
    - Converge to a low-energy state (clean image)
    
    We show:
    1. The trajectory as images at different timesteps
    2. Energy over time (should monotonically decrease)
    3. Step sizes (gradient magnitudes)
    
    Parameters
    ----------
    model : nn.Module
        Energy network.
    
    x_init : torch.Tensor
        Starting point, shape (784,) or (1, 784).
    
    n_steps : int
        Number of Langevin steps.
    
    step_size : float
        Step size for dynamics.
    
    device : str
        Computation device.
    
    title : str
        Plot title.
    
    save_path : str, optional
        If provided, save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    model.eval()
    model = model.to(device)
    
    if x_init.dim() == 1:
        x_init = x_init.unsqueeze(0)
    x_init = x_init.to(device)
    
    # Run Langevin and record trajectory
    x = x_init.clone().detach().requires_grad_(True)
    trajectory = [x.detach().clone()]
    energies = []
    grad_norms = []
    
    for step in range(n_steps):
        E = model(x)
        energies.append(E.item())
        
        grad = torch.autograd.grad(E.sum(), x, create_graph=False)[0]
        grad_norms.append(grad.norm().item())
        
        # Gradient clipping
        grad = torch.clamp(grad, -1.0, 1.0)
        
        x = x - step_size * grad
        x = x.detach().requires_grad_(True)
        
        trajectory.append(x.detach().clone())
    
    # Final energy
    energies.append(model(x).item())
    
    # Select steps to visualize
    n_show = min(10, n_steps + 1)
    step_indices = np.linspace(0, n_steps, n_show, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=(15, 6))
    
    # Top row: Images at different timesteps
    for i, step_idx in enumerate(step_indices):
        ax = fig.add_subplot(2, n_show, i + 1)
        img = trajectory[step_idx].view(28, 28).cpu().numpy()
        img = (img + 1) / 2  # [-1,1] → [0,1]
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(f"t=0\nE={energies[0]:.1f}", fontsize=9)
        elif step_idx == n_steps:
            ax.set_title(f"t={step_idx}\nE={energies[-1]:.1f}", fontsize=9)
        else:
            ax.set_title(f"t={step_idx}", fontsize=9)
    
    # Bottom left: Energy over time
    ax_e = fig.add_subplot(2, 2, 3)
    ax_e.plot(energies, 'b-', linewidth=2)
    ax_e.set_xlabel("Langevin Step")
    ax_e.set_ylabel("Energy E(x)")
    ax_e.set_title("Energy vs Time")
    ax_e.grid(True, alpha=0.3)
    
    # Mark key points
    ax_e.axhline(y=energies[0], color='r', linestyle='--', alpha=0.5, label='Initial')
    ax_e.axhline(y=energies[-1], color='g', linestyle='--', alpha=0.5, label='Final')
    ax_e.legend()
    
    # Bottom right: Gradient norm over time
    ax_g = fig.add_subplot(2, 2, 4)
    ax_g.plot(grad_norms, 'orange', linewidth=2)
    ax_g.set_xlabel("Langevin Step")
    ax_g.set_ylabel("||∇E||")
    ax_g.set_title("Gradient Magnitude")
    ax_g.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory to {save_path}")
    
    return fig


def plot_energy_histogram(
    model: nn.Module,
    clean_samples: torch.Tensor,
    noisy_samples: torch.Tensor,
    random_samples: Optional[torch.Tensor] = None,
    device: str = "cpu",
    title: str = "Energy Distribution",
    save_path: Optional[str] = None
):
    """
    Compare energy distributions of different sample types.
    
    Purpose
    -------
    A well-trained EBM should have:
    - Clean data: low energy (leftmost peak)
    - Noisy data: medium energy (middle)
    - Random noise: high energy (rightmost)
    
    If these overlap significantly, the model hasn't learned to discriminate.
    
    Parameters
    ----------
    model : nn.Module
        Energy network.
    
    clean_samples : torch.Tensor
        Clean images, shape (N, 784).
    
    noisy_samples : torch.Tensor
        Noisy images (same shape).
    
    random_samples : torch.Tensor, optional
        Random noise samples (same shape).
    
    device, title, save_path : standard params
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        E_clean = model(clean_samples.to(device)).cpu().numpy()
        E_noisy = model(noisy_samples.to(device)).cpu().numpy()
        if random_samples is not None:
            E_random = model(random_samples.to(device)).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot histograms
    bins = 50
    alpha = 0.6
    
    ax.hist(E_clean, bins=bins, alpha=alpha, label=f'Clean (μ={E_clean.mean():.1f})', color='green')
    ax.hist(E_noisy, bins=bins, alpha=alpha, label=f'Noisy (μ={E_noisy.mean():.1f})', color='orange')
    if random_samples is not None:
        ax.hist(E_random, bins=bins, alpha=alpha, label=f'Random (μ={E_random.mean():.1f})', color='red')
    
    ax.set_xlabel("Energy E(x)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(
        "← Lower energy = Higher probability",
        xy=(0.02, 0.98), xycoords='axes fraction',
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")
    
    return fig


def plot_denoising_grid(
    model: nn.Module,
    clean_images: torch.Tensor,
    noise_std: float = 0.5,
    langevin_steps: int = 100,
    step_size: float = 10.0,
    device: str = "cpu",
    title: str = "Denoising Results",
    save_path: Optional[str] = None
):
    """
    Create a grid showing clean → noisy → denoised.
    
    Parameters
    ----------
    model : nn.Module
        Energy network.
    
    clean_images : torch.Tensor
        Clean images, shape (N, 784).
    
    noise_std : float
        Noise level.
    
    langevin_steps : int
        Steps for denoising.
    
    step_size : float
        Langevin step size.
    
    device, title, save_path : standard params
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    model.eval()
    model = model.to(device)
    
    n_samples = clean_images.size(0)
    clean = clean_images.to(device)
    noisy = clean + torch.randn_like(clean) * noise_std
    
    # Denoise
    with torch.no_grad():
        # Use gradient-based inference
        x = noisy.clone().requires_grad_(True)
        for _ in tqdm(range(langevin_steps), desc="Denoising"):
            E = model(x)
            grad = torch.autograd.grad(E.sum(), x)[0]
            grad = torch.clamp(grad, -1.0, 1.0)
            x = x - step_size * grad
            x = x.detach().requires_grad_(True)
        denoised = x.detach()
    
    # Convert to numpy
    clean_np = clean.cpu().view(-1, 28, 28).numpy()
    noisy_np = noisy.cpu().view(-1, 28, 28).numpy()
    denoised_np = denoised.cpu().view(-1, 28, 28).numpy()
    
    # Create figure
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 1.5, 4.5))
    
    row_labels = ["Clean", "Noisy", "Denoised"]
    data_rows = [clean_np, noisy_np, denoised_np]
    
    for row, (label, data) in enumerate(zip(row_labels, data_rows)):
        for col in range(n_samples):
            img = (data[col] + 1) / 2  # [-1,1] → [0,1]
            img = np.clip(img, 0, 1)
            
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved denoising grid to {save_path}")
    
    return fig


def create_full_visualization(
    model: nn.Module,
    test_loader,
    output_dir: str = "./outputs",
    device: str = "cpu",
    noise_std: float = 0.5
):
    """
    Create a complete visualization suite.
    
    Generates:
    1. Energy histogram
    2. Langevin trajectory for one sample
    3. Energy landscape around a sample
    4. Denoising grid
    
    Parameters
    ----------
    model : nn.Module
        Trained energy network.
    
    test_loader : DataLoader
        Test data.
    
    output_dir : str
        Output directory.
    
    device : str
        Computation device.
    
    noise_std : float
        Noise level.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get some test samples
    test_batch, _ = next(iter(test_loader))
    clean_samples = test_batch[:64].to(device)
    noisy_samples = clean_samples + torch.randn_like(clean_samples) * noise_std
    random_samples = torch.randn_like(clean_samples)
    
    print("Creating visualizations...")
    
    # 1. Energy histogram
    plot_energy_histogram(
        model, clean_samples, noisy_samples, random_samples,
        device=device,
        title="Energy Distribution by Sample Type",
        save_path=os.path.join(output_dir, "energy_histogram.png")
    )
    
    # 2. Langevin trajectory
    x_noisy = noisy_samples[0:1]
    plot_langevin_trajectory(
        model, x_noisy,
        n_steps=100, step_size=10.0,
        device=device,
        title="Denoising via Langevin Dynamics",
        save_path=os.path.join(output_dir, "langevin_trajectory.png")
    )
    
    # 3. Energy landscape
    center = clean_samples[0]
    # Use noise direction and a random orthogonal direction
    noise_dir = torch.randn(784, device=device)
    random_dir = torch.randn(784, device=device)
    
    plot_energy_landscape_2d(
        model, center, noise_dir, random_dir,
        range_=1.0, n_points=40,
        device=device,
        title="Energy Landscape around Clean Image",
        save_path=os.path.join(output_dir, "energy_landscape.png")
    )
    
    # 4. Denoising grid
    plot_denoising_grid(
        model, clean_samples[:8],
        noise_std=noise_std,
        langevin_steps=100, step_size=10.0,
        device=device,
        title="Energy-Based Denoising Results",
        save_path=os.path.join(output_dir, "denoising_grid.png")
    )
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    """
    Demo: Create visualizations with a random (untrained) model.
    Results will be meaningless but demonstrates the tools.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("Loading MNIST for demo...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    print("Creating untrained model for demo...")
    model = EnergyMLP(input_dim=784)**2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Generating visualizations (NOTE: model is untrained, results meaningless)...")
    create_full_visualization(
        model, test_loader,
        output_dir="./outputs/demo",
        device=device,
        noise_std=0.5
    )
    
    print("\nDemo complete!")
