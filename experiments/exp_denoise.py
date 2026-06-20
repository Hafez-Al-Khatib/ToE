"""
Paper 1 — Experiment 1: Image Denoising
=========================================

Compares KAN-Hamiltonian vs MLP-Baseline on image denoising.

Tasks:
  1. Synthetic shapes (circles/squares, 28x28) — always available
  2. MNIST (if torchvision available)
  3. FashionMNIST (if torchvision available)

Metrics: PSNR, SSIM, MSE
Ablations: noise level σ ∈ {0.1, 0.3, 0.5, 0.7}

Usage:
    python experiments/exp_denoise.py --dataset synthetic --epochs 30
    python experiments/exp_denoise.py --dataset mnist --epochs 30
    python experiments/exp_denoise.py --all  # Run full ablation suite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
from collections import defaultdict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hamiltonian_field import HamiltonianField
from mlp_field import MLPField
from metrics import (compute_psnr, compute_ssim, compute_noisy_psnr,
                     compute_energy_statistics, compute_mass_conservation,
                     ExperimentLogger, print_comparison_table)


# ============================================================================
# Datasets
# ============================================================================

class SyntheticShapesDataset(torch.utils.data.Dataset):
    """Circles and squares on 28x28 grid. Values in [-1, 1]."""

    def __init__(self, length=2000, size=28, seed=42):
        self.length = length
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.data = self._generate_all()

    def _generate_all(self):
        images = []
        for i in range(self.length):
            img = np.full((self.size, self.size), -1.0, dtype=np.float32)
            shape_type = self.rng.choice(['circle', 'square'])
            cx, cy = self.rng.randint(6, self.size - 6, size=2)
            r = self.rng.randint(3, 8)

            if shape_type == 'circle':
                y, x = np.ogrid[:self.size, :self.size]
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                img[mask] = 1.0
            else:
                x0, x1 = max(0, cx - r), min(self.size, cx + r)
                y0, y1 = max(0, cy - r), min(self.size, cy + r)
                img[y0:y1, x0:x1] = 1.0

            images.append(torch.from_numpy(img).unsqueeze(0))  # (1, H, W)
        return torch.stack(images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataset(name, train=True, size=28):
    """Load dataset by name. Falls back to synthetic if torchvision unavailable."""
    if name == 'synthetic':
        seed = 42 if train else 123
        length = 2000 if train else 500
        return SyntheticShapesDataset(length=length, size=size, seed=seed)

    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
        ])

        if name == 'mnist':
            ds = torchvision.datasets.MNIST(
                root='./data', train=train, download=True, transform=transform)
        elif name == 'fashion':
            ds = torchvision.datasets.FashionMNIST(
                root='./data', train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Wrap to return only images (no labels)
        class ImageOnly(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                img, _ = self.ds[idx]
                return img

        return ImageOnly(ds)

    except ImportError:
        print(f"  [WARNING] torchvision not available. Falling back to synthetic shapes.")
        seed = 42 if train else 123
        length = 2000 if train else 500
        return SyntheticShapesDataset(length=length, size=size, seed=seed)


# ============================================================================
# Training
# ============================================================================

def train_denoising(model, dataset, config):
    """
    Train a field model for 1-step denoising.

    Loss: ||x - (x_noisy - η∇E(x_noisy))||²

    Parameters
    ----------
    model : HamiltonianField or MLPField
    dataset : image dataset
    config : dict with epochs, lr, noise_std, batch_size, etc.
    """
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 30)
    lr = config.get('lr', 1e-3)
    noise_std = config.get('noise_std', 0.3)
    batch_size = config.get('batch_size', 32)
    grad_reg = config.get('grad_reg', 0.001)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = defaultdict(list)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            clean = batch.to(device)
            noise = torch.randn_like(clean) * noise_std
            noisy = clean + noise

            optimizer.zero_grad()

            # 1-step denoising: x_denoised = x_noisy - step_size * grad(E)
            noisy_input = noisy.clone().requires_grad_(True)
            energy = model.compute_energy(noisy_input)
            grad = torch.autograd.grad(energy.sum(), noisy_input, create_graph=True)[0]
            denoised = noisy_input - model.step_size * grad

            # Reconstruction loss
            loss = F.mse_loss(denoised, clean)

            # Gradient regularization (prevents energy landscape from being too steep)
            if grad_reg > 0:
                loss = loss + grad_reg * (grad ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: loss={avg_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return history


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_denoising(model, dataset, noise_std=0.3, n_steps_list=[1, 5, 10, 20],
                       device='cpu', n_eval=200):
    """
    Evaluate denoising across different evolution step counts.

    Returns dict of metrics for each n_steps setting.
    """
    model = model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    results = {}

    for n_steps in n_steps_list:
        psnr_list = []
        ssim_list = []
        noisy_psnr_list = []
        n_seen = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            clean = batch.to(device)
            noise = torch.randn_like(clean) * noise_std
            noisy = clean + noise

            # Multi-step denoising via evolve
            with torch.enable_grad():
                denoised = model.evolve(noisy, n_steps=n_steps, anchor=noisy)

            # Per-sample metrics
            for i in range(clean.shape[0]):
                psnr_list.append(compute_psnr(clean[i], denoised[i]))
                ssim_list.append(compute_ssim(clean[i], denoised[i]))
                noisy_psnr_list.append(compute_psnr(clean[i], noisy[i]))

            n_seen += clean.shape[0]
            if n_seen >= n_eval:
                break

        results[f'steps_{n_steps}'] = {
            'psnr_mean': np.mean(psnr_list),
            'psnr_std': np.std(psnr_list),
            'ssim_mean': np.mean(ssim_list),
            'ssim_std': np.std(ssim_list),
            'noisy_psnr': np.mean(noisy_psnr_list),
        }

    return results


# ============================================================================
# Visualization
# ============================================================================

def visualize_denoising(model, dataset, noise_std=0.3, n_steps=10,
                        device='cpu', save_path='denoising_results.png'):
    """Generate denoising visualization grid."""
    model.eval()

    # Get 6 samples
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    clean = next(iter(loader))
    if isinstance(clean, (list, tuple)):
        clean = clean[0]
    clean = clean.to(device)
    noise = torch.randn_like(clean) * noise_std
    noisy = clean + noise

    with torch.enable_grad():
        denoised = model.evolve(noisy, n_steps=n_steps, anchor=noisy)

    # Plot: 3 rows (clean, noisy, denoised) x 6 columns
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    titles = ['Clean', 'Noisy', 'Denoised']

    for i in range(6):
        for row, (img, title) in enumerate(zip(
            [clean[i], noisy[i], denoised[i]], titles)):
            ax = axes[row, i]
            im = img.squeeze().cpu().numpy()
            ax.imshow(im, cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(title, fontsize=14, fontweight='bold')

            # Add PSNR to denoised
            if row == 2:
                psnr = compute_psnr(clean[i], denoised[i])
                ax.set_title(f'PSNR: {psnr:.1f} dB', fontsize=10)

    plt.suptitle(f'Denoising Results (σ={noise_std}, steps={n_steps})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(histories: dict, save_path: str):
    """Plot training loss curves for multiple models."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'KAN-Hamiltonian': '#2196F3', 'MLP-Baseline': '#FF5722'}
    for name, history in histories.items():
        ax.plot(history['loss'], label=name, color=colors.get(name, None),
                linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss: KAN vs MLP Energy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_noise_ablation(results_by_noise: dict, save_path: str):
    """Plot PSNR/SSIM vs noise level for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'KAN-Hamiltonian': '#2196F3', 'MLP-Baseline': '#FF5722'}

    for model_name, noise_results in results_by_noise.items():
        noise_levels = sorted(noise_results.keys())
        psnrs = [noise_results[n]['psnr_mean'] for n in noise_levels]
        ssims = [noise_results[n]['ssim_mean'] for n in noise_levels]
        psnr_stds = [noise_results[n].get('psnr_std', 0) for n in noise_levels]
        ssim_stds = [noise_results[n].get('ssim_std', 0) for n in noise_levels]

        color = colors.get(model_name, None)
        ax1.errorbar(noise_levels, psnrs, yerr=psnr_stds, marker='o',
                     label=model_name, color=color, linewidth=2, capsize=4)
        ax2.errorbar(noise_levels, ssims, yerr=ssim_stds, marker='s',
                     label=model_name, color=color, linewidth=2, capsize=4)

    ax1.set_xlabel('Noise σ', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Noise Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Noise σ', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM vs Noise Level', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Noise Level Ablation: KAN vs MLP Energy',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_experiment(dataset_name, noise_std, epochs, device, output_dir):
    """Run one KAN vs MLP comparison at a specific noise level."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Noise: σ={noise_std}  |  Epochs: {epochs}")
    print(f"{'='*60}")

    # Load data
    train_ds = get_dataset(dataset_name, train=True)
    test_ds = get_dataset(dataset_name, train=False)
    sample = train_ds[0]
    C, H, W = sample.shape
    print(f"  Data shape: ({C}, {H}, {W})")

    results = {}
    histories = {}

    # ---- KAN-Hamiltonian ----
    print(f"\n--- Training KAN-Hamiltonian ---")
    kan_model = HamiltonianField(
        n_channels=C, height=H, width=W,
        n_filters=16, kan_hidden=[32, 1], kan_grid_size=5
    )
    n_params_kan = sum(p.numel() for p in kan_model.parameters())
    print(f"  Parameters: {n_params_kan:,}")

    # Honest Verification Fix: Initialize KAN grid with a batch of features
    print("  Initializing KAN grid...")
    kan_model.to(device)
    with torch.no_grad():
        batch = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=64)))
        if isinstance(batch, (list, tuple)): batch = batch[0]
        noisy_batch = (batch + torch.randn_like(batch) * noise_std).to(device)
        # We need to compute features to update the grid
        pad = kan_model.spatial_filters[0].shape[-1] // 2
        feature_maps = []
        for i in range(C):
            u_i = noisy_batch[:, i:i+1, :, :]
            f_i = F.conv2d(u_i, kan_model.spatial_filters[i], padding=pad)
            feature_maps.append(f_i)
        features = torch.cat(feature_maps, dim=1)
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        # Multi-layer KAN needs sequential updates or a dedicated method.
        # Simple fix: call forward with update_grid=True
        kan_model.kan(features_flat, update_grid=True)

    config = {
        'device': device, 'epochs': epochs, 'lr': 1e-3,
        'noise_std': noise_std, 'batch_size': 32, 'grad_reg': 0.001
    }

    t0 = time.time()
    histories['KAN-Hamiltonian'] = train_denoising(kan_model, train_ds, config)
    kan_train_time = time.time() - t0

    kan_eval = evaluate_denoising(
        kan_model, test_ds, noise_std=noise_std,
        n_steps_list=[1, 5, 10, 20], device=device)
    results['KAN-Hamiltonian'] = {
        **kan_eval.get('steps_10', {}),
        'params': n_params_kan,
        'train_time_s': kan_train_time,
    }

    # ---- MLP-Baseline ----
    print(f"\n--- Training MLP-Baseline ---")
    mlp_model = MLPField(
        n_channels=C, height=H, width=W,
        n_filters=16, mlp_hidden=[32]
    )
    n_params_mlp = sum(p.numel() for p in mlp_model.parameters())
    print(f"  Parameters: {n_params_mlp:,}")

    t0 = time.time()
    histories['MLP-Baseline'] = train_denoising(mlp_model, train_ds, config)
    mlp_train_time = time.time() - t0

    mlp_eval = evaluate_denoising(
        mlp_model, test_ds, noise_std=noise_std,
        n_steps_list=[1, 5, 10, 20], device=device)
    results['MLP-Baseline'] = {
        **mlp_eval.get('steps_10', {}),
        'params': n_params_mlp,
        'train_time_s': mlp_train_time,
    }

    # ---- Print Comparison ----
    print_comparison_table(results, f"Denoising: {dataset_name} (σ={noise_std})")

    # ---- Visualizations ----
    tag = f"{dataset_name}_noise{noise_std}"
    visualize_denoising(
        kan_model, test_ds, noise_std=noise_std, n_steps=10,
        device=device,
        save_path=os.path.join(output_dir, f'denoise_kan_{tag}.png'))
    visualize_denoising(
        mlp_model, test_ds, noise_std=noise_std, n_steps=10,
        device=device,
        save_path=os.path.join(output_dir, f'denoise_mlp_{tag}.png'))
    plot_training_curves(
        histories,
        save_path=os.path.join(output_dir, f'training_curves_{tag}.png'))

    # Save models
    torch.save(kan_model.state_dict(),
               os.path.join(output_dir, f'kan_denoise_{tag}.pth'))
    torch.save(mlp_model.state_dict(),
               os.path.join(output_dir, f'mlp_denoise_{tag}.pth'))

    return results, kan_eval, mlp_eval, histories


def run_full_ablation(dataset_name, epochs, device, output_dir):
    """Run noise level ablation: σ ∈ {0.1, 0.3, 0.5, 0.7}."""
    noise_levels = [0.1, 0.3, 0.5, 0.7]
    all_results_kan = {}
    all_results_mlp = {}

    for sigma in noise_levels:
        results, kan_eval, mlp_eval, _ = run_single_experiment(
            dataset_name, sigma, epochs, device, output_dir)
        all_results_kan[sigma] = kan_eval.get('steps_10', {})
        all_results_mlp[sigma] = mlp_eval.get('steps_10', {})

    # Plot ablation
    plot_noise_ablation(
        {'KAN-Hamiltonian': all_results_kan, 'MLP-Baseline': all_results_mlp},
        save_path=os.path.join(output_dir, f'noise_ablation_{dataset_name}.png'))

    return all_results_kan, all_results_mlp


def main():
    parser = argparse.ArgumentParser(description="Paper 1: Denoising Experiments")
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'mnist', 'fashion'])
    parser.add_argument('--noise-std', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='./results/denoise')
    parser.add_argument('--all', action='store_true',
                        help='Run full noise ablation suite')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        # Full ablation across noise levels
        run_full_ablation(args.dataset, args.epochs, device, args.output_dir)
    else:
        # Single experiment
        run_single_experiment(
            args.dataset, args.noise_std, args.epochs, device, args.output_dir)

    print("\n✓ All denoising experiments complete!")
    print(f"  Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
