"""
Phase 1 Training: Energy-Based MNIST Denoising
===============================================

This script trains an energy-based model to denoise MNIST digits.

The Training Process
--------------------
We train the energy function E(x) such that:
- Clean digits → Low energy (stable states)
- Noisy/corrupted images → High energy (unstable states)

Training uses Persistent Contrastive Divergence (PCD):
1. Sample negative particles from a persistent buffer
2. Refine them with Langevin dynamics
3. Push down energy of real data, push up energy of negatives
4. Store refined particles back in buffer

After training, denoising is just gradient descent on E:
    x_{t+1} = x_t - ε∇E(x_t)

The ball rolls from noisy hilltops to clean valleys.

Usage
-----
Basic training:
    python src/train_phase1.py

With options:
    python src/train_phase1.py --epochs 30 --batch-size 128 --noise-std 0.5

Evaluation only (loads checkpoint):
    python src/train_phase1.py --eval --checkpoint outputs/model.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from energy_model import (
    EnergyMLP,
    EnergyBasedDenoiser,
    langevin_dynamics,
    annealed_langevin_dynamics,
    SampleBuffer,
    contrastive_divergence_loss
)


def get_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0
):
    """
    Load MNIST dataset with appropriate transforms.
    
    We flatten images to 784-dim vectors and normalize to [-1, 1].
    
    Why [-1, 1]?
    -----------
    Langevin dynamics works better when data is centered at 0.
    Otherwise, the energy gradient has a constant bias.
    
    Parameters
    ----------
    data_dir : str
        Directory to store/load MNIST data.
    
    batch_size : int
        Batch size for training.
    
    num_workers : int
        DataLoader workers. Set to 0 on Windows.
    
    Returns
    -------
    train_loader, test_loader : DataLoader
        PyTorch data loaders.
    """
    # Transform: Tensor → Flatten → Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # 28*28 = 784
        transforms.Lambda(lambda x: x * 2 - 1),   # [0,1] → [-1,1]
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    The noise model:
        x̃ = x + σ·ε, where ε ~ N(0, I)
    
    Parameters
    ----------
    x : torch.Tensor
        Clean images.
    
    noise_std : float
        Standard deviation of noise (relative to data scale).
    
    Returns
    -------
    torch.Tensor
        Noisy images.
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise


def compute_psnr(clean: torch.Tensor, denoised: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(MAX² / MSE)
    
    For data in [-1, 1], MAX = 2.
    
    Higher PSNR = Better reconstruction
    - < 20 dB: Poor
    - 20-25 dB: Acceptable
    - 25-30 dB: Good
    - > 30 dB: Excellent
    
    Parameters
    ----------
    clean : torch.Tensor
        Ground truth images.
    
    denoised : torch.Tensor
        Reconstructed images.
    
    Returns
    -------
    float
        PSNR in dB.
    """
    mse = (clean - denoised).pow(2).mean()
    if mse < 1e-10:
        return float('inf')
    
    max_val = 2.0  # Range of [-1, 1]
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def train_epoch(
    model: nn.Module,
    buffer: SampleBuffer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    langevin_steps: int = 20,
    langevin_lr: float = 10.0
) -> dict:
    """
    Train for one epoch using Persistent Contrastive Divergence.
    
    PCD Algorithm
    -------------
    For each batch of real data x_pos:
    
    1. Sample x_neg from persistent buffer
    2. Refine x_neg with k Langevin steps:
       x_neg ← x_neg - ε∇E(x_neg) + noise
    3. Compute CD loss: E(x_pos) - E(x_neg)
    4. Update model parameters
    5. Store refined x_neg back in buffer
    
    The key insight: the buffer particles gradually converge to the
    model's distribution, providing high-quality negative samples
    without running MCMC from scratch.
    
    Parameters
    ----------
    model : nn.Module
        Energy network E(x).
    
    buffer : SampleBuffer
        Persistent buffer for negative samples.
    
    train_loader : DataLoader
        Training data loader.
    
    optimizer : Optimizer
        Model optimizer.
    
    device : torch.device
        Device for computation.
    
    epoch : int
        Current epoch number (for logging).
    
    langevin_steps : int
        Number of Langevin steps for negative refinement.
    
    langevin_lr : float
        Step size for Langevin dynamics.
    
    Returns
    -------
    dict
        Epoch metrics (average energy_pos, energy_neg, etc.)
    """
    model.train()
    
    metrics_sum = {
        "energy_pos": 0.0,
        "energy_neg": 0.0,
        "energy_diff": 0.0,
        "cd_loss": 0.0,
    }
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    
    for x_pos, _ in pbar:
        x_pos = x_pos.to(device)
        batch_size = x_pos.size(0)
        
        # === Step 1: Sample from buffer ===
        x_neg, indices = buffer.sample(batch_size)
        x_neg = x_neg.to(device)
        
        # === Step 2: Refine with Langevin dynamics ===
        # Note: langevin_dynamics needs gradients internally to compute ∇E
        # but returns detached tensors, so no backprop through this step
        model.eval()
        x_neg = langevin_dynamics(
            model, x_neg,
            n_steps=langevin_steps,
            step_size=langevin_lr,
            noise_scale=0.01,
            clip_grad=1.0
        )
        model.train()
        
        # === Step 3: Update buffer ===
        buffer.update(indices, x_neg)
        
        # === Step 4: Compute loss and update ===
        optimizer.zero_grad()
        
        loss, batch_metrics = contrastive_divergence_loss(
            model, x_pos, x_neg,
            alpha=1.0,
            reg_weight=0.01
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics.get(k, 0.0)
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "E+": f"{batch_metrics['energy_pos']:.2f}",
            "E-": f"{batch_metrics['energy_neg']:.2f}",
            "diff": f"{batch_metrics['energy_diff']:.2f}"
        })
    
    # Average metrics
    return {k: v / n_batches for k, v in metrics_sum.items()}


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    noise_std: float = 0.5,
    langevin_steps: int = 100,
    langevin_lr: float = 10.0,
    n_samples: int = 16
) -> dict:
    """
    Evaluate denoising performance on test set.
    
    Note: We cannot use @torch.no_grad() here because annealed_langevin_dynamics
    needs gradients internally to compute ∇E. However, the results are detached.
    
    Evaluation Metrics
    ------------------
    1. **Energy ratio**: E(noisy) / E(clean)
       Should be > 1 if model learned correctly.
    
    2. **PSNR (noisy)**: How bad is the input?
    
    3. **PSNR (denoised)**: How good is our reconstruction?
       The improvement is what matters.
    
    4. **Visual quality**: Save sample images.
    
    Parameters
    ----------
    model : nn.Module
        Trained energy network.
    
    test_loader : DataLoader
        Test data loader.
    
    device : torch.device
        Computation device.
    
    noise_std : float
        Noise level for corruption.
    
    langevin_steps : int
        Steps for Langevin denoising.
    
    langevin_lr : float
        Step size for denoising.
    
    n_samples : int
        Number of sample images to save.
    
    Returns
    -------
    dict
        Evaluation metrics.
    """
    model.eval()
    
    total_psnr_noisy = 0.0
    total_psnr_denoised = 0.0
    total_energy_clean = 0.0
    total_energy_noisy = 0.0
    n_samples_eval = 0
    
    # Store samples for visualization
    samples_clean = []
    samples_noisy = []
    samples_denoised = []
    
    for x_clean, _ in tqdm(test_loader, desc="Evaluating"):
        x_clean = x_clean.to(device)
        batch_size = x_clean.size(0)
        
        # Add noise
        x_noisy = add_noise(x_clean, noise_std)
        
        # Denoise
        x_denoised = annealed_langevin_dynamics(
            model, x_noisy,
            n_steps=langevin_steps,
            step_size_init=langevin_lr * 5,
            step_size_final=langevin_lr * 0.1
        )
        
        # Compute metrics
        psnr_noisy = compute_psnr(x_clean, x_noisy)
        psnr_denoised = compute_psnr(x_clean, x_denoised)
        
        total_psnr_noisy += psnr_noisy * batch_size
        total_psnr_denoised += psnr_denoised * batch_size
        
        total_energy_clean += model(x_clean).sum().item()
        total_energy_noisy += model(x_noisy).sum().item()
        
        n_samples_eval += batch_size
        
        # Store samples for visualization
        if len(samples_clean) < n_samples:
            n_take = min(n_samples - len(samples_clean), batch_size)
            samples_clean.append(x_clean[:n_take].cpu())
            samples_noisy.append(x_noisy[:n_take].cpu())
            samples_denoised.append(x_denoised[:n_take].cpu())
    
    # Aggregate metrics
    metrics = {
        "psnr_noisy": total_psnr_noisy / n_samples_eval,
        "psnr_denoised": total_psnr_denoised / n_samples_eval,
        "psnr_improvement": (total_psnr_denoised - total_psnr_noisy) / n_samples_eval,
        "energy_clean": total_energy_clean / n_samples_eval,
        "energy_noisy": total_energy_noisy / n_samples_eval,
        "energy_ratio": total_energy_noisy / (total_energy_clean + 1e-8)
    }
    
    # Stack samples
    samples = {
        "clean": torch.cat(samples_clean, dim=0)[:n_samples],
        "noisy": torch.cat(samples_noisy, dim=0)[:n_samples],
        "denoised": torch.cat(samples_denoised, dim=0)[:n_samples],
    }
    
    return metrics, samples


def save_samples(samples: dict, output_path: str):
    """
    Save before/after denoising comparison image.
    
    Creates a grid with three rows:
    - Row 1: Clean (ground truth)
    - Row 2: Noisy (input)
    - Row 3: Denoised (output)
    """
    n_samples = samples["clean"].size(0)
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 1.5, 4.5))
    
    titles = ["Clean", "Noisy", "Denoised"]
    keys = ["clean", "noisy", "denoised"]
    
    for row, (title, key) in enumerate(zip(titles, keys)):
        for col in range(n_samples):
            img = samples[key][col].view(28, 28).numpy()
            # Convert from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            
            if col == 0:
                ax.set_ylabel(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train energy-based denoiser on MNIST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train_phase1.py --epochs 20
  python src/train_phase1.py --noise-std 0.3 --batch-size 256
  python src/train_phase1.py --eval --checkpoint outputs/model.pt
        """
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    
    # Energy model parameters
    parser.add_argument("--hidden-dims", type=str, default="512,256,128",
                        help="Hidden layer sizes, comma-separated (default: 512,256,128)")
    
    # Langevin parameters
    parser.add_argument("--langevin-steps-train", type=int, default=20,
                        help="Langevin steps during training (default: 20)")
    parser.add_argument("--langevin-steps-eval", type=int, default=100,
                        help="Langevin steps during evaluation (default: 100)")
    parser.add_argument("--langevin-lr", type=float, default=10.0,
                        help="Langevin step size (default: 10.0)")
    
    # Noise parameters
    parser.add_argument("--noise-std", type=float, default=0.5,
                        help="Noise std for corruption (default: 0.5)")
    
    # I/O
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory for MNIST data (default: ./data)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory for outputs (default: ./outputs)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for resuming/evaluation")
    
    # Mode
    parser.add_argument("--eval", action="store_true",
                        help="Evaluation only (requires --checkpoint)")
    
    args = parser.parse_args()
    
    # Parse hidden dims
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0  # Windows compatibility
    )
    print(f"Loaded MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    
    # Model
    model = EnergyMLP(
        input_dim=784,
        hidden_dims=hidden_dims,
        use_spectral_norm=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Buffer for PCD
    buffer = SampleBuffer(
        buffer_size=10000,
        sample_dim=784,
        reinit_prob=0.05
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluation only mode
    if args.eval:
        print("\n" + "=" * 50)
        print("EVALUATION MODE")
        print("=" * 50)
        
        metrics, samples = evaluate(
            model, test_loader, device,
            noise_std=args.noise_std,
            langevin_steps=args.langevin_steps_eval,
            langevin_lr=args.langevin_lr
        )
        
        print("\nResults:")
        print(f"  PSNR (noisy input):  {metrics['psnr_noisy']:.2f} dB")
        print(f"  PSNR (denoised):     {metrics['psnr_denoised']:.2f} dB")
        print(f"  PSNR improvement:    {metrics['psnr_improvement']:+.2f} dB")
        print(f"  Energy (clean):      {metrics['energy_clean']:.2f}")
        print(f"  Energy (noisy):      {metrics['energy_noisy']:.2f}")
        print(f"  Energy ratio:        {metrics['energy_ratio']:.2f}x")
        
        save_samples(samples, os.path.join(args.output_dir, "denoised_samples.png"))
        return
    
    # Training loop
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Noise std: {args.noise_std}")
    print(f"Langevin steps (train): {args.langevin_steps_train}")
    print()
    
    best_psnr = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, buffer, train_loader, optimizer, device, epoch,
            langevin_steps=args.langevin_steps_train,
            langevin_lr=args.langevin_lr
        )
        
        print(f"\nEpoch {epoch} - Train:")
        print(f"  E(clean): {train_metrics['energy_pos']:.3f}")
        print(f"  E(noise): {train_metrics['energy_neg']:.3f}")
        print(f"  Diff: {train_metrics['energy_diff']:.3f}")
        
        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics, samples = evaluate(
                model, test_loader, device,
                noise_std=args.noise_std,
                langevin_steps=args.langevin_steps_eval,
                langevin_lr=args.langevin_lr
            )
            
            print(f"\nEpoch {epoch} - Eval:")
            print(f"  PSNR (noisy): {metrics['psnr_noisy']:.2f} dB")
            print(f"  PSNR (denoised): {metrics['psnr_denoised']:.2f} dB")
            print(f"  Improvement: {metrics['psnr_improvement']:+.2f} dB")
            print(f"  Energy ratio: {metrics['energy_ratio']:.2f}x")
            
            # Save best model
            if metrics['psnr_denoised'] > best_psnr:
                best_psnr = metrics['psnr_denoised']
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "psnr": best_psnr,
                    "metrics": metrics
                }, os.path.join(args.output_dir, "best_model.pt"))
                print(f"  ✓ Saved best model (PSNR: {best_psnr:.2f} dB)")
            
            # Save samples
            save_samples(
                samples,
                os.path.join(args.output_dir, f"samples_epoch_{epoch}.png")
            )
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs - 1,
    }, os.path.join(args.output_dir, "final_model.pt"))
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
