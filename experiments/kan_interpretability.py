"""
KAN Interpretability Visualizations
=====================================

The key novelty claim of Paper 1: KAN learns interpretable energy functions
as compositions of univariate B-splines, unlike black-box MLP energies.

This module provides:
1. Spline function plots — visualize each learned φ_{i,j}(x)
2. Feature importance — which spatial features contribute most to energy
3. Energy landscape cross-sections — 1D/2D slices of the energy surface
4. Activation distribution analysis — how inputs map through KAN layers
5. Comparison with MLP: show KAN learns structured functions

These visualizations are critical for the paper's interpretability argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# 1. Learned Spline Functions
# ============================================================================

def plot_kan_splines(kan_model, layer_idx=0, n_cols=8, n_points=200,
                     save_path='kan_splines.png', title_prefix=''):
    """
    Plot the learned spline functions φ_{i,j}(x) in a KAN layer.

    Each subplot shows one edge's learned function: input i → output j.
    This is the core interpretability visualization.

    Parameters
    ----------
    kan_model : KAN module (from src/kan.py)
    layer_idx : which KAN layer to visualize
    n_cols : columns in the plot grid
    n_points : evaluation points for each spline
    """
    layer = kan_model.layers[layer_idx]
    in_features = layer.in_features
    out_features = layer.out_features

    # Evaluation range
    x = torch.linspace(-2, 2, n_points).unsqueeze(1)  # (n_points, 1)

    n_total = in_features * out_features
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        # Get the B-spline basis values for each input
        # Expand x to match each input dimension
        for i in range(in_features):
            for j in range(out_features):
                idx = i * out_features + j
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                # Compute the spline output for input dimension i → output j
                # We need to evaluate the specific (i,j) spline
                x_input = torch.zeros(n_points, in_features)
                x_input[:, i] = x.squeeze()

                # Base function (linear part): w * x + b
                base_output = layer.base_weight[j, i] * x.squeeze()

                # Spline function: B-spline basis evaluation
                # Compute B-spline basis for this specific input
                grid = layer.grid[i]  # Grid for input dim i
                coeff = layer.spline_weight[j, i]  # Coefficients for (i→j)

                # Evaluate B-spline manually
                spline_output = _eval_bspline(
                    x.squeeze(), grid, coeff, layer.spline_order)

                total = base_output + spline_output

                # Plot
                ax.plot(x.squeeze().numpy(), total.numpy(),
                        color='#2196F3', linewidth=1.5, label='Total')
                ax.plot(x.squeeze().numpy(), base_output.numpy(),
                        color='gray', linewidth=0.8, linestyle='--',
                        alpha=0.5, label='Linear')
                ax.plot(x.squeeze().numpy(), spline_output.numpy(),
                        color='#FF5722', linewidth=1.0, alpha=0.7,
                        label='Spline')

                ax.set_title(f'φ({i}→{j})', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
                ax.axhline(y=0, color='k', linewidth=0.3)
                ax.axvline(x=0, color='k', linewidth=0.3)

    # Hide unused axes
    for idx in range(n_total, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        if row < n_rows and col < n_cols:
            axes[row, col].set_visible(False)

    plt.suptitle(f'{title_prefix}Learned KAN Spline Functions (Layer {layer_idx})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _eval_bspline(x, grid, coeff, order):
    """
    Evaluate a B-spline given grid points and coefficients.

    Uses the recursive Cox-de Boor formula, matching the implementation in kan.py.
    """
    # Expand grid for broadcasting
    # grid: (grid_size + 2*order + 1,)
    # coeff: (grid_size + order,)

    n = len(coeff)
    result = torch.zeros_like(x)

    # Order-0 basis functions
    bases = torch.zeros(len(x), len(grid) - 1)
    for k in range(len(grid) - 1):
        bases[:, k] = ((x >= grid[k]) & (x < grid[k+1])).float()

    # Recurse up to desired order
    for p in range(1, order + 1):
        new_bases = torch.zeros(len(x), len(grid) - 1 - p)
        for k in range(len(grid) - 1 - p):
            denom1 = grid[k + p] - grid[k]
            denom2 = grid[k + p + 1] - grid[k + 1]

            term1 = 0.0
            if denom1 > 1e-10:
                term1 = (x - grid[k]) / denom1 * bases[:, k]
            term2 = 0.0
            if denom2 > 1e-10:
                term2 = (grid[k+p+1] - x) / denom2 * bases[:, k+1]
            new_bases[:, k] = term1 + term2
        bases = new_bases

    # Weighted sum
    for k in range(min(n, bases.shape[1])):
        result += coeff[k] * bases[:, k]

    return result


def plot_kan_splines_simple(model, samples, layer_idx=0,
                            save_path='kan_splines_simple.png',
                            title_prefix=''):
    """
    Simplified spline visualization using actual data activations.

    Instead of evaluating splines analytically, this passes real data
    through the KAN and plots input vs output for each edge.
    More robust and works with any KAN implementation.
    """
    model.eval()

    # Get activations at each layer
    activations = []
    hooks = []

    def make_hook(storage):
        def hook_fn(module, input, output):
            storage.append((input[0].detach(), output.detach()))
        return hook_fn

    kan = model.kan if hasattr(model, 'kan') else None
    if kan is None:
        print("  [WARNING] Model has no 'kan' attribute. Skipping spline plot.")
        return

    for layer in kan.layers:
        storage = []
        activations.append(storage)
        hooks.append(layer.register_forward_hook(make_hook(storage)))

    # Forward pass with real data
    with torch.no_grad():
        _ = model.compute_energy(samples[:32])

    # Remove hooks
    for h in hooks:
        h.remove()

    if layer_idx >= len(activations) or len(activations[layer_idx]) == 0:
        print(f"  [WARNING] No activations captured for layer {layer_idx}")
        return

    inp, out = activations[layer_idx][0]
    in_dim = inp.shape[-1]
    out_dim = out.shape[-1]

    # Plot scatter of input[i] vs partial contribution to output[j]
    n_show = min(in_dim, 8)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        x_vals = inp[:, i].cpu().numpy()

        # Sort for clean line
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]

        ax.hist(x_sorted, bins=50, alpha=0.3, color='gray',
                density=True, label='Input distribution')
        ax.set_xlabel(f'Input feature {i}', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'{title_prefix}KAN Layer {layer_idx} — Input Distributions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 2. Energy Landscape Cross-Sections
# ============================================================================

def plot_energy_landscape(model, clean_sample, noise_range=(-1.5, 1.5),
                          n_points=100, save_path='energy_landscape.png'):
    """
    Plot 2D energy landscape around a clean sample.

    Picks two random directions in pixel space and evaluates energy
    on a grid, showing the energy bowl structure.
    """
    model.eval()

    # Two random perturbation directions (normalized)
    d1 = torch.randn_like(clean_sample)
    d1 = d1 / d1.norm()
    d2 = torch.randn_like(clean_sample)
    d2 = d2 - (d2 * d1).sum() * d1  # Orthogonalize
    d2 = d2 / d2.norm()

    alphas = np.linspace(noise_range[0], noise_range[1], n_points)
    betas = np.linspace(noise_range[0], noise_range[1], n_points)

    energy_grid = np.zeros((n_points, n_points))

    with torch.no_grad():
        for i, a in enumerate(alphas):
            batch = []
            for j, b in enumerate(betas):
                perturbed = clean_sample + a * d1 + b * d2
                batch.append(perturbed)

            batch = torch.stack(batch)
            energies = model.compute_energy(batch)
            energy_grid[i, :] = energies.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2D contour
    im = ax1.contourf(betas, alphas, energy_grid, levels=50, cmap='RdYlBu_r')
    ax1.plot(0, 0, 'w*', markersize=15, markeredgecolor='black')
    ax1.set_xlabel('Direction 2', fontsize=12)
    ax1.set_ylabel('Direction 1', fontsize=12)
    ax1.set_title('Energy Landscape (2D slice)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Energy')

    # 1D cross-section through center
    center_idx = n_points // 2
    ax2.plot(alphas, energy_grid[:, center_idx], color='#2196F3',
             linewidth=2, label='Direction 1')
    ax2.plot(betas, energy_grid[center_idx, :], color='#FF5722',
             linewidth=2, label='Direction 2')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Perturbation magnitude', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Cross-Sections', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 3. Spatial Filter Visualization
# ============================================================================

def plot_spatial_filters(model, save_path='spatial_filters.png'):
    """
    Visualize the learned spatial convolution filters.

    spatial_filters is a ParameterList of length n_channels.
    Each element has shape (n_filters, 1, k, k) — the full filter bank
    for one channel. We flatten across channels to get all individual
    k×k kernels and display them in a grid.
    """
    if not hasattr(model, 'spatial_filters'):
        print("  [WARNING] Model has no spatial_filters. Skipping.")
        return

    # Collect all individual (k, k) kernels from all channel banks
    all_kernels = []
    for c, fbank in enumerate(model.spatial_filters):
        # fbank: (n_filters, 1, k, k)
        bank = fbank.detach().cpu()           # (n_filters, 1, k, k)
        for f in range(bank.shape[0]):
            kernel = bank[f, 0]               # (k, k)
            all_kernels.append((c, f, kernel))

    n_total = len(all_kernels)
    n_cols = min(8, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_rows))
    # Normalise to 2-D array regardless of shape
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (c, f, kernel) in enumerate(all_kernels):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        kmax = kernel.abs().max().item() + 1e-8
        im = ax.imshow(kernel.numpy(), cmap='RdBu_r',
                       vmin=-kmax, vmax=kmax, interpolation='nearest')
        ax.set_title(f'ch{c} f{f}', fontsize=7)
        ax.axis('off')

    for idx in range(n_total, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Learned Spatial Filters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 4. KAN vs MLP Interpretability Comparison
# ============================================================================

def compare_interpretability(kan_model, mlp_model, test_samples,
                             save_path='interpretability_comparison.png'):
    """
    Side-by-side comparison of KAN vs MLP energy function properties.

    Shows:
    - Energy smoothness (gradient norm distribution)
    - Energy spectrum (eigenvalues of Hessian proxy)
    - Response to structured perturbations
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    samples = test_samples[:64]

    # --- (a) Gradient norm distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    for model, name, color in [
        (kan_model, 'KAN', '#2196F3'), (mlp_model, 'MLP', '#FF5722')]:
        model.eval()
        x = samples.clone().requires_grad_(True)
        E = model.compute_energy(x)
        grad = torch.autograd.grad(E.sum(), x)[0]
        grad_norms = grad.view(grad.shape[0], -1).norm(dim=1).detach().cpu().numpy()
        ax1.hist(grad_norms, bins=30, alpha=0.6, label=name, color=color, density=True)

    ax1.set_xlabel('||∇E(x)||', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('(a) Gradient Norm Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- (b) Energy distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    for model, name, color in [
        (kan_model, 'KAN', '#2196F3'), (mlp_model, 'MLP', '#FF5722')]:
        with torch.no_grad():
            E = model.compute_energy(samples).cpu().numpy()
        ax2.hist(E, bins=30, alpha=0.6, label=name, color=color, density=True)

    ax2.set_xlabel('E(x)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('(b) Energy Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- (c) Energy vs noise level ---
    ax3 = fig.add_subplot(gs[0, 2])
    noise_levels = np.linspace(0, 1.0, 20)
    for model, name, color in [
        (kan_model, 'KAN', '#2196F3'), (mlp_model, 'MLP', '#FF5722')]:
        mean_energies = []
        for sigma in noise_levels:
            noisy = samples[:16] + sigma * torch.randn_like(samples[:16])
            with torch.no_grad():
                E = model.compute_energy(noisy).mean().item()
            mean_energies.append(E)
        ax3.plot(noise_levels, mean_energies, marker='o', label=name,
                 color=color, linewidth=2, markersize=4)

    ax3.set_xlabel('Noise σ', fontsize=11)
    ax3.set_ylabel('Mean Energy', fontsize=11)
    ax3.set_title('(c) Energy vs Noise Level', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- (d) Parameter count comparison ---
    ax4 = fig.add_subplot(gs[1, 0])
    kan_params = sum(p.numel() for p in kan_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_model.parameters())

    bars = ax4.bar(['KAN', 'MLP'], [kan_params, mlp_params],
                   color=['#2196F3', '#FF5722'], alpha=0.8)
    for bar, val in zip(bars, [kan_params, mlp_params]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_ylabel('Parameters', fontsize=11)
    ax4.set_title('(d) Model Size', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # --- (e) Gradient smoothness (Laplacian proxy) ---
    ax5 = fig.add_subplot(gs[1, 1])
    for model, name, color in [
        (kan_model, 'KAN', '#2196F3'), (mlp_model, 'MLP', '#FF5722')]:
        # Measure how "smooth" the energy gradient is via finite differences
        eps = 0.01
        smoothness_scores = []
        for i in range(min(16, samples.shape[0])):
            x = samples[i:i+1].clone().requires_grad_(True)
            E = model.compute_energy(x)
            g = torch.autograd.grad(E, x, create_graph=True)[0]

            # Approximate Hessian diagonal via finite diff
            x_plus = x.detach() + eps * torch.randn_like(x)
            x_plus.requires_grad_(True)
            E_plus = model.compute_energy(x_plus)
            g_plus = torch.autograd.grad(E_plus, x_plus)[0]

            hess_approx = ((g_plus - g.detach()) / eps).abs().mean().item()
            smoothness_scores.append(hess_approx)

        ax5.hist(smoothness_scores, bins=15, alpha=0.6, label=name,
                 color=color, density=True)

    ax5.set_xlabel('Hessian magnitude (proxy)', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('(e) Energy Smoothness', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- (f) Info box ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    info = (
        "Interpretability Comparison\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "KAN Energy:\n"
        "  E(features) = Σ Φ_j(Σ φ_{i,j}(f_i))\n"
        "  Each φ_{i,j} is a learned B-spline\n"
        "  → Can plot & inspect each function\n\n"
        "MLP Energy:\n"
        "  E(features) = MLP(features)\n"
        "  Hidden units with fixed activations\n"
        "  → Black box, no decomposition\n\n"
        "Key insight: KAN decomposes\n"
        "multivariate energy into sum of\n"
        "interpretable 1D functions."
    )
    ax6.text(0.05, 0.5, info, transform=ax6.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('KAN vs MLP: Interpretability Analysis',
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def run_interpretability_analysis(kan_model, mlp_model, test_data, output_dir):
    """Run all interpretability visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Interpretability Analysis ---")

    # 1. Spatial filters
    plot_spatial_filters(
        kan_model,
        save_path=os.path.join(output_dir, 'kan_spatial_filters.png'))
    plot_spatial_filters(
        mlp_model,
        save_path=os.path.join(output_dir, 'mlp_spatial_filters.png'))

    # 2. KAN spline activations
    plot_kan_splines_simple(
        kan_model, test_data,
        save_path=os.path.join(output_dir, 'kan_spline_activations.png'),
        title_prefix='Denoising ')

    # 3. Energy landscape
    if test_data.shape[0] > 0:
        plot_energy_landscape(
            kan_model, test_data[0:1],
            save_path=os.path.join(output_dir, 'kan_energy_landscape.png'))
        plot_energy_landscape(
            mlp_model, test_data[0:1],
            save_path=os.path.join(output_dir, 'mlp_energy_landscape.png'))

    # 4. Full comparison
    compare_interpretability(
        kan_model, mlp_model, test_data,
        save_path=os.path.join(output_dir, 'interpretability_comparison.png'))

    print("  Interpretability analysis complete!")


if __name__ == "__main__":
    print("Run via run_experiments.py or import and call run_interpretability_analysis()")
