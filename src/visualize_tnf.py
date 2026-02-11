"""
TNF Visualization Suite
========================

Generates publication-quality visualizations of the Thermodynamic Neural Field:

1. 3D Energy Surface with gradient descent trajectory
2. Learned spatial filter gallery
3. Step-by-step denoising evolution 
4. Energy landscape cross-sections
5. Training dynamics summary

Usage:
    py -3.12 src/visualize_tnf.py --dataset mnist
    py -3.12 src/visualize_tnf.py --dataset fashion
    py -3.12 src/visualize_tnf.py --dataset both
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from thermodynamic_field import ThermodynamicField


def load_model(checkpoint_path: str, device: str = "cuda") -> ThermodynamicField:
    """Load trained TNF model from checkpoint."""
    model = ThermodynamicField(n_channels=4, n_filters=16, filter_size=5).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_test_samples(fashion: bool = False, n_samples: int = 8, device: str = "cuda"):
    """Get a few test samples."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    DatasetClass = datasets.FashionMNIST if fashion else datasets.MNIST
    dataset = DatasetClass('./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    x, y = next(iter(loader))
    return x.to(device), y


# =============================================================================
# 1. 3D Energy Surface with Gradient Descent Trajectory
# =============================================================================
def plot_3d_energy_surface(model, sample, noise_std=0.3, save_path="outputs/3d_energy_surface.png"):
    """
    Create a 3D surface plot of the energy landscape.
    
    Takes a real sample, picks two principal perturbation directions,
    and maps E(sample + α·d1 + β·d2) as a 3D surface.
    Shows the gradient descent trajectory from noisy → denoised.
    """
    device = next(model.parameters()).device
    
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)
    
    # Create perturbation directions
    # d1 = noise direction (what noise adds)
    # d2 = orthogonal random direction
    torch.manual_seed(42)
    d1 = torch.randn_like(sample)
    d1 = d1 / d1.norm() * noise_std * np.sqrt(28 * 28)  # scale to noise magnitude
    
    d2 = torch.randn_like(sample)
    # Orthogonalize d2 w.r.t d1
    d2 = d2 - (d2 * d1).sum() / (d1 * d1).sum() * d1
    d2 = d2 / d2.norm() * d1.norm()
    
    # Grid for the energy surface
    n_grid = 60
    alphas = np.linspace(-1.5, 2.0, n_grid)
    betas = np.linspace(-1.5, 1.5, n_grid)
    
    energies = np.zeros((n_grid, n_grid))
    
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = sample + a * d1 + b * d2
                perturbed = torch.clamp(perturbed, -1.5, 1.5)
                E = model(perturbed.view(1, -1)).item()
                energies[i, j] = E
    
    # Get gradient descent trajectory
    noisy = sample + d1  # noise is exactly along d1 direction
    noisy = torch.clamp(noisy, -1.5, 1.5)
    trajectory = model.evolve(noisy, n_steps=30, return_trajectory=True)
    
    # Project trajectory onto (d1, d2) plane
    traj_alphas = []
    traj_betas = []
    traj_energies = []
    
    with torch.no_grad():
        for t in range(trajectory.shape[0]):
            step = trajectory[t]
            if step.dim() == 4:
                step = step[0:1]
            diff = step - sample
            a = (diff * d1).sum().item() / (d1 * d1).sum().item()
            b = (diff * d2).sum().item() / (d2 * d2).sum().item()
            E = model(step.view(1, -1)).item()
            traj_alphas.append(a)
            traj_betas.append(b)
            traj_energies.append(E)
    
    # Create the 3D figure
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    
    fig = plt.figure(figsize=(16, 6))
    
    # --- Panel 1: 3D Surface ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Normalize energies for coloring
    E_min, E_max = energies.min(), energies.max()
    
    surf = ax1.plot_surface(A, B, energies, cmap='viridis', alpha=0.7,
                            linewidth=0, antialiased=True,
                            vmin=E_min, vmax=E_max)
    
    # Plot trajectory as red line on surface
    ax1.plot(traj_alphas, traj_betas, traj_energies,
             'r.-', linewidth=2.5, markersize=4, zorder=10, label='∇E descent')
    
    # Mark start (noisy) and end (denoised)
    ax1.scatter([traj_alphas[0]], [traj_betas[0]], [traj_energies[0]],
                color='red', s=100, marker='*', zorder=11, label='Noisy (start)')
    ax1.scatter([traj_alphas[-1]], [traj_betas[-1]], [traj_energies[-1]],
                color='lime', s=100, marker='*', zorder=11, label='Denoised (end)')
    ax1.scatter([0], [0], [model(sample.view(1, -1)).item()],
                color='cyan', s=100, marker='o', zorder=11, label='Clean (target)')
    
    ax1.set_xlabel('\nNoise direction (α)', fontsize=10)
    ax1.set_ylabel('\nOrthogonal (β)', fontsize=10)
    ax1.set_zlabel('\nEnergy E(u)', fontsize=10)
    ax1.set_title('3D Energy Landscape\nwith Gradient Descent Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.view_init(elev=25, azim=-60)
    
    # --- Panel 2: Top-down contour with trajectory ---
    ax2 = fig.add_subplot(122)
    
    contour = ax2.contourf(A, B, energies, levels=40, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='Energy E(u)')
    
    # Gradient arrows along trajectory
    for k in range(len(traj_alphas) - 1):
        da = traj_alphas[k+1] - traj_alphas[k]
        db = traj_betas[k+1] - traj_betas[k]
        ax2.annotate('', xy=(traj_alphas[k+1], traj_betas[k+1]),
                     xytext=(traj_alphas[k], traj_betas[k]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax2.plot(traj_alphas, traj_betas, 'r.-', linewidth=2, markersize=5, label='∇E descent')
    ax2.scatter([traj_alphas[0]], [traj_betas[0]], color='red', s=120, marker='*', 
                zorder=10, label='Noisy', edgecolors='white')
    ax2.scatter([traj_alphas[-1]], [traj_betas[-1]], color='lime', s=120, marker='*',
                zorder=10, label='Denoised', edgecolors='white')
    ax2.scatter([0], [0], color='cyan', s=120, marker='o',
                zorder=10, label='Clean', edgecolors='white')
    
    ax2.set_xlabel('Noise direction (α)', fontsize=11)
    ax2.set_ylabel('Orthogonal direction (β)', fontsize=11)
    ax2.set_title('Energy Contour Map\n"Denoising = Rolling Downhill"', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 2. Learned Spatial Filter Gallery
# =============================================================================
def plot_learned_filters(model, save_path="outputs/learned_filters.png"):
    """
    Visualize the learned spatial filters as a gallery.
    These are the Kₖ in E = Σₖ wₖ|Kₖ*u|². 
    Shows what spatial patterns the field has learned to prefer.
    """
    n_channels = model.n_channels
    n_filters = model.n_filters
    
    fig, axes = plt.subplots(n_channels, n_filters, 
                              figsize=(n_filters * 1.2, n_channels * 1.4))
    
    for ch in range(n_channels):
        filters = model.spatial_filters[ch].detach().cpu().numpy()
        weights = model.filter_weights[ch].detach().cpu().abs().numpy()
        
        for f in range(n_filters):
            ax = axes[ch, f]
            kernel = filters[f, 0]
            
            vmax = max(abs(kernel.max()), abs(kernel.min()))
            ax.imshow(kernel, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Border color indicates weight magnitude
            w = weights[f]
            for spine in ax.spines.values():
                spine.set_color(plt.cm.hot(min(w / weights.max(), 1.0)))
                spine.set_linewidth(2)
            
            if ch == 0:
                ax.set_title(f'K{f}', fontsize=8)
        
        axes[ch, 0].set_ylabel(f'Ch {ch}\n(α={model.alphas[ch].item():.2f})', 
                                fontsize=9, rotation=0, labelpad=50)
    
    plt.suptitle('Learned Spatial Filters Kₖ\n'
                 'Red-Blue = filter weights, Border intensity = importance wₖ\n'
                 'These define what spatial patterns the field "prefers"',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 3. Step-by-Step Denoising Evolution
# =============================================================================
def plot_denoising_evolution(model, sample, noise_std=0.3, 
                              save_path="outputs/denoising_evolution.png"):
    """
    Show the field evolving from noisy → clean step by step.
    Displays the image at each gradient descent step and the energy curve.
    """
    device = next(model.parameters()).device
    
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)
    
    noisy = sample + torch.randn_like(sample) * noise_std
    
    # Get trajectory with 30 steps
    trajectory = model.evolve(noisy, n_steps=30, return_trajectory=True)
    
    # Compute energy at each step
    energies = []
    with torch.no_grad():
        for t in range(trajectory.shape[0]):
            E = model(trajectory[t].view(1, -1)).item()
            energies.append(E)
    
    # Select 8 snapshots to display
    n_snapshots = 8
    snapshot_indices = np.linspace(0, trajectory.shape[0] - 1, n_snapshots, dtype=int)
    
    fig = plt.figure(figsize=(18, 8))
    
    # Top row: image evolution
    for idx, snap_i in enumerate(snapshot_indices):
        ax = fig.add_subplot(2, n_snapshots, idx + 1)
        img = trajectory[snap_i, 0, 0].cpu().numpy()
        img = (img + 1) / 2  # [-1,1] → [0,1]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if snap_i == 0:
            ax.set_title(f'Step 0\n(Noisy)', fontsize=10, color='red', fontweight='bold')
        elif snap_i == trajectory.shape[0] - 1:
            ax.set_title(f'Step {snap_i}\n(Denoised)', fontsize=10, color='green', fontweight='bold')
        else:
            ax.set_title(f'Step {snap_i}', fontsize=10)
    
    # Also show clean target
    ax = fig.add_subplot(2, n_snapshots, n_snapshots)
    img_clean = sample[0, 0].cpu().numpy()
    img_clean = (img_clean + 1) / 2
    # This overlays the clean as the last panel already (replace)
    
    # Bottom row: energy curve spanning full width
    ax_energy = fig.add_subplot(2, 1, 2)
    
    steps = list(range(len(energies)))
    ax_energy.plot(steps, energies, 'b-', linewidth=2.5, label='Energy E(u)')
    ax_energy.fill_between(steps, energies, min(energies), alpha=0.15, color='blue')
    
    # Mark the snapshot points
    for snap_i in snapshot_indices:
        ax_energy.plot(snap_i, energies[snap_i], 'ko', markersize=8)
    
    # Mark start and end
    ax_energy.plot(0, energies[0], 'r*', markersize=15, label=f'Noisy (E={energies[0]:.1f})')
    ax_energy.plot(len(energies)-1, energies[-1], 'g*', markersize=15,
                   label=f'Denoised (E={energies[-1]:.1f})')
    
    # Add clean energy reference
    with torch.no_grad():
        E_clean = model(sample.view(1, -1)).item()
    ax_energy.axhline(E_clean, color='cyan', linestyle='--', linewidth=1.5,
                      label=f'Clean target (E={E_clean:.1f})')
    
    ax_energy.set_xlabel('Gradient Descent Step', fontsize=12)
    ax_energy.set_ylabel('Energy E(u)', fontsize=12)
    ax_energy.set_title('Energy Minimization During Denoising\n'
                        '"The universe doesn\'t compute. It relaxes."',
                        fontsize=12, fontweight='bold')
    ax_energy.legend(fontsize=10, loc='upper right')
    ax_energy.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 4. Multi-Sample 3D Energy Cross-Section
# =============================================================================
def plot_energy_cross_section(model, samples, noise_std=0.3,
                               save_path="outputs/energy_cross_section.png"):
    """
    For multiple samples, show 1D energy cross-sections along the noise direction.
    Demonstrates that E(noisy) > E(clean) consistently.
    """
    device = next(model.parameters()).device
    n_show = min(4, samples.shape[0])
    
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 4.5, 4))
    if n_show == 1:
        axes = [axes]
    
    for idx in range(n_show):
        ax = axes[idx]
        sample = samples[idx:idx+1].to(device)
        
        torch.manual_seed(idx)
        noise = torch.randn_like(sample)
        noise = noise / noise.norm() * noise_std * np.sqrt(28 * 28)
        
        # Sweep along noise direction
        t_values = np.linspace(-1.5, 2.5, 100)
        energies = []
        
        with torch.no_grad():
            for t in t_values:
                perturbed = sample + t * noise
                perturbed = torch.clamp(perturbed, -1.5, 1.5)
                E = model(perturbed.view(1, -1)).item()
                energies.append(E)
        
        ax.plot(t_values, energies, 'b-', linewidth=2)
        ax.fill_between(t_values, energies, min(energies), alpha=0.1, color='blue')
        
        # Mark clean (t=0) and noisy (t=1)
        E_clean = energies[np.argmin(np.abs(np.array(t_values) - 0))]
        E_noisy = energies[np.argmin(np.abs(np.array(t_values) - 1))]
        
        ax.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.scatter([0], [E_clean], color='green', s=100, zorder=5, label=f'Clean')
        ax.scatter([1], [E_noisy], color='red', s=100, zorder=5, label=f'Noisy')
        
        # Arrow showing gradient descent direction
        ax.annotate('', xy=(0.2, E_noisy * 0.95), xytext=(0.8, E_noisy * 0.95),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))
        ax.text(0.5, E_noisy * 0.92, '∇E descent', ha='center', fontsize=9,
                color='orange', fontweight='bold')
        
        ax.set_xlabel('Perturbation α', fontsize=10)
        ax.set_ylabel('Energy E(u)', fontsize=10)
        ax.set_title(f'Sample {idx+1}\nΔE = {E_noisy - E_clean:+.1f}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Energy Cross-Sections Along Noise Direction\n'
                 'Clean images sit in energy valleys — noise pushes uphill',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 5. Learned Potential & Coupling Visualization
# =============================================================================
def plot_learned_physics(model, save_path="outputs/learned_physics.png"):
    """
    Visualize the learned physical parameters:
    - Polynomial potentials V(u) = a·u² + b·u⁴ per channel
    - Diffusion rates α per channel
    - Channel coupling weights
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Potential V(u) for each channel
    ax = axes[0]
    u_range = np.linspace(-1.5, 1.5, 200)
    
    for ch in range(model.n_channels):
        a = model.potential_a[ch].item()
        b = model.potential_b[ch].item()
        V = a * u_range**2 + b * u_range**4
        
        label = f'Ch{ch}: {a:.3f}u² + {b:.3f}u⁴'
        ch_type = 'Activator' if ch % 2 == 0 else 'Inhibitor'
        ax.plot(u_range, V, linewidth=2.5, label=f'{ch_type} (Ch{ch})')
    
    ax.set_xlabel('Field value u', fontsize=11)
    ax.set_ylabel('Potential V(u)', fontsize=11)
    ax.set_title('Learned Potentials V(u)\n(What pixel values are "natural")', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    # Panel 2: Diffusion rates
    ax = axes[1]
    alphas = model.alphas.detach().cpu().numpy()
    colors = ['#2196F3' if i % 2 == 0 else '#FF5722' for i in range(len(alphas))]
    labels = ['Activator' if i % 2 == 0 else 'Inhibitor' for i in range(len(alphas))]
    
    bars = ax.bar(range(len(alphas)), alphas, color=colors, alpha=0.85, edgecolor='black')
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('Diffusion rate α', fontsize=11)
    ax.set_title('Turing Pattern Parameters\n(Slow activator + Fast inhibitor → Patterns)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'Ch{i}\n({labels[i]})' for i in range(len(alphas))], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Coupling weights
    ax = axes[2]
    coupling_w = model.coupling_weights.detach().cpu().numpy()
    n_ch = model.n_channels
    coupling_matrix = np.zeros((n_ch, n_ch))
    idx = 0
    labels_pairs = []
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            coupling_matrix[i, j] = coupling_w[idx]
            coupling_matrix[j, i] = coupling_w[idx]
            labels_pairs.append(f'({i},{j}): {coupling_w[idx]:.3f}')
            idx += 1
    
    vmax = max(abs(coupling_w.max()), abs(coupling_w.min()))
    im = ax.imshow(coupling_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Coupling strength c')
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('Channel', fontsize=11)
    ax.set_title('Inter-Channel Coupling\n(Turing activator-inhibitor interaction)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_ch))
    ax.set_yticks(range(n_ch))
    ax.set_xticklabels([f'Ch{i}' for i in range(n_ch)])
    ax.set_yticklabels([f'Ch{i}' for i in range(n_ch)])
    
    # Annotate values
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j:
                ax.text(j, i, f'{coupling_matrix[i,j]:.3f}', 
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(coupling_matrix[i,j]) > vmax*0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 6. Combined Summary Figure
# =============================================================================
def plot_summary(model, sample, noise_std=0.3, dataset_name="MNIST",
                 save_path="outputs/tnf_summary.png"):
    """Create a single combined summary figure."""
    device = next(model.parameters()).device
    
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)
    
    noisy = sample + torch.randn_like(sample) * noise_std
    
    # 1-step denoising
    x_input = noisy.requires_grad_(True)
    E = model(x_input.view(1, -1))
    grad = torch.autograd.grad(E.sum(), x_input, create_graph=False)[0]
    denoised_1step = (x_input - model.step_size * grad).detach()
    
    # Multi-step evolution
    trajectory = model.evolve(noisy, n_steps=30, return_trajectory=True)
    
    # Energy along trajectory
    traj_energies = []
    with torch.no_grad():
        for t in range(trajectory.shape[0]):
            traj_energies.append(model(trajectory[t].view(1, -1)).item())
    
    fig = plt.figure(figsize=(20, 10))
    
    # ---- Row 1: Clean → Noisy → 1-step → Multi-step ----
    titles = ['Clean', f'Noisy (σ={noise_std})', '1-Step Denoised', '30-Step Evolved']
    images = [sample, noisy, denoised_1step, trajectory[-1]]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 4, i + 1)
        show_img = img[0, 0].detach().cpu().numpy()
        show_img = (show_img + 1) / 2
        ax.imshow(show_img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add PSNR
        with torch.no_grad():
            mse = (img - sample).pow(2).mean().item()
            if mse > 1e-10:
                psnr = 10 * np.log10(4.0 / mse)
                ax.text(0.5, -0.05, f'PSNR: {psnr:.1f} dB', transform=ax.transAxes,
                       ha='center', fontsize=10, color='blue')
    
    # ---- Row 2 Left: Energy descent ----
    ax = fig.add_subplot(2, 4, 5)
    ax.plot(traj_energies, 'b-', linewidth=2.5)
    ax.fill_between(range(len(traj_energies)), traj_energies, min(traj_energies),
                    alpha=0.15, color='blue')
    ax.plot(0, traj_energies[0], 'r*', markersize=15, label='Noisy')
    ax.plot(len(traj_energies)-1, traj_energies[-1], 'g*', markersize=15, label='Denoised')
    with torch.no_grad():
        E_clean = model(sample.view(1, -1)).item()
    ax.axhline(E_clean, color='cyan', linestyle='--', label='Clean')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Descent', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ---- Row 2 Middle: Learned potentials ----
    ax = fig.add_subplot(2, 4, 6)
    u_range = np.linspace(-1.5, 1.5, 100)
    for ch in range(model.n_channels):
        a = model.potential_a[ch].item()
        b = model.potential_b[ch].item()
        V = a * u_range**2 + b * u_range**4
        ax.plot(u_range, V, linewidth=2, label=f'Ch{ch}')
    ax.set_xlabel('u')
    ax.set_ylabel('V(u)')
    ax.set_title('Learned Potentials', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ---- Row 2 Right: Diffusion rates ----
    ax = fig.add_subplot(2, 4, 7)
    alphas = model.alphas.detach().cpu().numpy()
    colors = ['#2196F3' if i % 2 == 0 else '#FF5722' for i in range(len(alphas))]
    ax.bar(range(len(alphas)), alphas, color=colors, edgecolor='black')
    ax.set_xlabel('Channel')
    ax.set_ylabel('α')
    ax.set_title('Diffusion Rates\n(Turing Pattern)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'Act' if i%2==0 else 'Inh' for i in range(len(alphas))])
    ax.grid(True, alpha=0.3, axis='y')
    
    # ---- Row 2 Far Right: A few filters ----
    ax = fig.add_subplot(2, 4, 8)
    # Show top 9 filters by weight for channel 0
    filters = model.spatial_filters[0].detach().cpu().numpy()
    weights = model.filter_weights[0].detach().cpu().abs().numpy()
    top_idx = np.argsort(weights)[::-1][:9]
    
    combined = np.zeros((15, 15))  # 3x3 grid of 5x5 filters
    for k, fi in enumerate(top_idx):
        row, col = k // 3, k % 3
        combined[row*5:(row+1)*5, col*5:(col+1)*5] = filters[fi, 0]
    
    vmax = max(abs(combined.max()), abs(combined.min()))
    ax.imshow(combined, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title('Top Spatial Filters\n(Learned edges/textures)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'Thermodynamic Neural Field — {dataset_name}\n'
                 f'1690 params, 4 channels, 16 filters/channel | '
                 f'η={model.step_size.item():.3f}, μ={model.data_weight.item():.3f}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="TNF Visualization Suite")
    parser.add_argument("--dataset", choices=["mnist", "fashion", "both"], default="both")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    datasets_to_viz = []
    if args.dataset in ("mnist", "both"):
        datasets_to_viz.append(("MNIST", "outputs/mnist_tnf.pt", False))
    if args.dataset in ("fashion", "both"):
        datasets_to_viz.append(("FashionMNIST", "outputs/fashion_tnf.pt", True))
    
    for name, ckpt, fashion in datasets_to_viz:
        print(f"\n{'='*60}")
        print(f"  Generating visualizations for {name}")
        print(f"{'='*60}")
        
        if not Path(ckpt).exists():
            print(f"  ⚠ Checkpoint not found: {ckpt}, skipping")
            continue
        
        model = load_model(ckpt, device)
        samples, labels = get_test_samples(fashion=fashion, n_samples=8, device=device)
        
        prefix = name.lower()
        
        print(f"\n  1. 3D Energy Surface...")
        plot_3d_energy_surface(model, samples[0], noise_std=0.3,
                               save_path=f"outputs/{prefix}_3d_energy.png")
        
        print(f"  2. Learned Spatial Filters...")
        plot_learned_filters(model, save_path=f"outputs/{prefix}_filters.png")
        
        print(f"  3. Denoising Evolution...")
        plot_denoising_evolution(model, samples[0], noise_std=0.3,
                                 save_path=f"outputs/{prefix}_evolution.png")
        
        print(f"  4. Energy Cross-Sections...")
        plot_energy_cross_section(model, samples[:4], noise_std=0.3,
                                   save_path=f"outputs/{prefix}_cross_sections.png")
        
        print(f"  5. Learned Physics...")
        plot_learned_physics(model, save_path=f"outputs/{prefix}_physics.png")
        
        print(f"  6. Summary Figure...")
        plot_summary(model, samples[0], noise_std=0.3, dataset_name=name,
                     save_path=f"outputs/{prefix}_summary.png")
    
    print(f"\n✓ All visualizations saved to ./outputs/")


if __name__ == "__main__":
    main()
