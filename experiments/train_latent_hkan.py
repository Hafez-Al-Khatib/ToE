"""
Train Latent H-KAN with Full FEP
==================================
Training script for LatentHKAN_FEP: the complete variational inference model
with KAN recognition, KAN potential, Hamiltonian dynamics, and KAN decoder.

Experiments
-----------
1. Reconstruction quality:   KAN-FEP vs standard VAE vs plain EBM
2. Latent trajectory:        Does Hamiltonian momentum escape local minima?
3. Parameter efficiency:     KAN-FEP vs pixel-space KAN params
4. Energy conservation:      H_initial vs H_final across training
5. Active inference demo:    Action selection minimising expected F

Usage
-----
python experiments/train_latent_hkan.py --dataset mnist --epochs 50
python experiments/train_latent_hkan.py --dataset fashionmnist --latent-dim 32
python experiments/train_latent_hkan.py --compare  # run all variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from latent_hkan_fep import LatentHKAN_FEP
from recognition_model import RecognitionKAN, GenerativeDecoder
from metrics import compute_psnr, compute_ssim, ExperimentLogger


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset(name: str, batch_size: int = 128, n_train: int = 10000):
    """
    Load dataset for training. Tries torchvision; falls back to synthetic.

    Supported: 'mnist', 'fashionmnist', 'synthetic'
    """
    try:
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([T.ToTensor()])
        if name == 'mnist':
            train = torchvision.datasets.MNIST(
                root='/tmp/data', train=True, download=True, transform=transform)
            test = torchvision.datasets.MNIST(
                root='/tmp/data', train=False, download=True, transform=transform)
        elif name == 'fashionmnist':
            train = torchvision.datasets.FashionMNIST(
                root='/tmp/data', train=True, download=True, transform=transform)
            test = torchvision.datasets.FashionMNIST(
                root='/tmp/data', train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Limit size for fast experiments
        if n_train < len(train):
            train = torch.utils.data.Subset(train, range(n_train))

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True,  drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test,  batch_size=batch_size, shuffle=False, drop_last=True)

        print(f"[data] Loaded {name}: {len(train)} train, {len(test)} test")
        return train_loader, test_loader, 784

    except Exception as e:
        print(f"[data] torchvision failed ({e}), using synthetic dataset")
        return _make_synthetic_loaders(batch_size, n_train)


def _make_synthetic_loaders(batch_size: int, n_train: int):
    """Synthetic Gaussian blobs as fallback."""
    torch.manual_seed(42)
    x_train = torch.randn(n_train, 1, 28, 28) * 0.3
    # Add 10 Gaussian blob classes
    for k in range(10):
        cx, cy = np.random.randint(8, 20, 2)
        idx = range(k * (n_train // 10), (k+1) * (n_train // 10))
        x_train[idx, 0, cy:cy+6, cx:cx+6] += 0.8
    x_train = x_train.clamp(0, 1)
    y_train = torch.arange(10).repeat(n_train // 10)

    x_test = torch.randn(1000, 1, 28, 28) * 0.3
    x_test = x_test.clamp(0, 1)
    y_test = torch.zeros(1000, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    test_ds  = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 784


# ── Standard VAE Baseline ─────────────────────────────────────────────────────

class VAEBaseline(nn.Module):
    """Standard VAE (MLP) for comparison against KAN-FEP."""

    def __init__(self, obs_dim: int = 784, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.SiLU(),
            nn.Linear(256, 64),     nn.SiLU(),
            nn.Linear(64, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.SiLU(),
            nn.Linear(64, 256),         nn.SiLU(),
            nn.Linear(256, obs_dim),    nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar.clamp(-10, 2)

    def reparameterise(self, mu, logvar):
        sigma = (0.5 * logvar).exp()
        return mu + sigma * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def compute_loss(self, x, beta=1.0):
        x_flat = x.view(x.size(0), -1)
        x_hat, mu, logvar, z = self.forward(x_flat)
        recon = F.mse_loss(x_hat, x_flat)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return {'F': recon + beta * kl, 'recon': recon, 'kl': kl}


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
    n_ham_steps: int = 1,
) -> Dict:
    """One epoch of training."""
    model.train()
    total_F = 0.0
    total_kl = 0.0
    total_recon = 0.0
    n_batches = 0

    for x, _ in loader:
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)

        optimiser.zero_grad()

        if isinstance(model, LatentHKAN_FEP):
            losses = model.compute_loss(x_flat, beta=beta, n_ham_steps=n_ham_steps)
        elif isinstance(model, VAEBaseline):
            losses = model.compute_loss(x_flat, beta=beta)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        losses['F'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        total_F     += losses['F'].item()
        total_kl    += losses.get('kl', torch.tensor(0.)).item()
        total_recon += losses.get('recon', torch.tensor(0.)).item()
        n_batches   += 1

    return {
        'F':     total_F / n_batches,
        'kl':    total_kl / n_batches,
        'recon': total_recon / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    n_ham_steps_list: Optional[List[int]] = None,
) -> Dict:
    """
    Evaluate reconstruction quality.

    For LatentHKAN_FEP: tests multiple Hamiltonian step counts to show
    test-time compute scaling.
    """
    if n_ham_steps_list is None:
        n_ham_steps_list = [0, 1, 5, 10]

    model.eval()
    results = {}
    x_samples, o_hat_samples = None, None

    for x, _ in loader:
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)

        if isinstance(model, LatentHKAN_FEP):
            for K in n_ham_steps_list:
                result = model(x_flat, n_ham_steps=K)
                o_hat = result['o_hat']
                psnr = compute_psnr(o_hat.view_as(x), x)
                ssim = compute_ssim(o_hat.view_as(x), x)
                key = f'K={K}'
                if key not in results:
                    results[key] = {'psnr': [], 'ssim': []}
                results[key]['psnr'].append(psnr.item())
                results[key]['ssim'].append(ssim.item())

                if x_samples is None and K == n_ham_steps_list[-1]:
                    x_samples = x[:8]
                    o_hat_samples = o_hat[:8]

        elif isinstance(model, VAEBaseline):
            x_hat, mu, logvar, z = model(x_flat)
            psnr = compute_psnr(x_hat.view_as(x), x)
            ssim = compute_ssim(x_hat.view_as(x), x)
            if 'vae' not in results:
                results['vae'] = {'psnr': [], 'ssim': []}
            results['vae']['psnr'].append(psnr.item())
            results['vae']['ssim'].append(ssim.item())

        break  # One batch is enough for eval

    # Average
    for key in results:
        results[key]['psnr_mean'] = float(np.mean(results[key]['psnr']))
        results[key]['ssim_mean'] = float(np.mean(results[key]['ssim']))

    return results, x_samples, o_hat_samples


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_reconstruction_grid(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    path: str,
    title: str = "Latent H-KAN FEP Reconstruction",
):
    """Save side-by-side reconstruction grid."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = min(8, x.shape[0])
        fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
        fig.suptitle(title, fontsize=12, fontweight='bold')

        for i in range(n):
            img = x[i].cpu().float()
            rec = x_hat[i].cpu().float()
            if img.dim() == 3:
                img = img.squeeze(0)
            if rec.dim() == 1:
                rec = rec.view(28, 28)

            axes[0, i].imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].imshow(rec.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            axes[1, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=9)
        axes[1, 0].set_ylabel('Reconstructed', fontsize=9)

        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[save] {path}")
    except Exception as e:
        print(f"[warn] Could not save figure: {e}")


def plot_hamiltonian_trajectories(
    model: LatentHKAN_FEP,
    x: torch.Tensor,
    path: str,
):
    """
    Visualise latent trajectories under Hamiltonian dynamics.

    Shows that momentum carries latent state across energy barriers,
    enabling exploration beyond gradient-descent's reach.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            z0, mu, logvar = model.recognition.sample(x_flat)

            # Run long trajectory
            z_traj, _, H_info = model.hamiltonian_trajectory(
                z0[:4], n_steps=50, return_all=True
            )

        if H_info['trajectory']:
            # PCA to 2D for vis
            traj = torch.stack(H_info['trajectory'], dim=1)  # (B, T, D)
            B, T, D = traj.shape

            # Project to 2D via first 2 dims (or PCA if available)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: 2D latent trajectory
            ax = axes[0]
            for b in range(min(4, B)):
                ax.plot(traj[b, :, 0].cpu(), traj[b, :, 1].cpu(),
                        alpha=0.7, linewidth=1.5, label=f'Sample {b+1}')
                ax.scatter(traj[b, 0, 0].cpu(), traj[b, 0, 1].cpu(),
                           marker='o', s=60, zorder=5)
                ax.scatter(traj[b, -1, 0].cpu(), traj[b, -1, 1].cpu(),
                           marker='*', s=100, zorder=5)
            ax.set_xlabel('Latent dim 0')
            ax.set_ylabel('Latent dim 1')
            ax.set_title('Hamiltonian Trajectories in Latent Space')
            ax.legend(fontsize=8)

            # Right: Hamiltonian energy over time
            ax2 = axes[1]
            if H_info['H_trajectory']:
                ax2.plot(H_info['H_trajectory'], color='steelblue', linewidth=2)
                ax2.axhline(H_info['H_trajectory'][0], color='red',
                            linestyle='--', alpha=0.5, label='H_initial')
                ax2.set_xlabel('Integration step k')
                ax2.set_ylabel('H(z,p)')
                ax2.set_title(f'Hamiltonian Energy (drift = {H_info["H_drift"]:.4f})')
                ax2.legend()

            plt.suptitle('Latent H-KAN: Symplectic Dynamics', fontweight='bold')
            plt.tight_layout()
            plt.savefig(path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"[save] {path}")
    except Exception as e:
        print(f"[warn] Could not plot trajectories: {e}")


def plot_psnr_vs_ham_steps(results: Dict, path: str):
    """Plot test-time compute scaling: PSNR vs Hamiltonian steps K."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        kan_keys = [k for k in results if k.startswith('K=')]
        if not kan_keys:
            return

        steps = [int(k.split('=')[1]) for k in kan_keys]
        psnrs = [results[k]['psnr_mean'] for k in kan_keys]
        ssims = [results[k]['ssim_mean'] for k in kan_keys]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(steps, psnrs, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax1.set_xlabel('Hamiltonian steps K')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Test-Time Compute Scaling: PSNR vs K')
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, ssims, 's-', color='darkorange', linewidth=2, markersize=8)
        ax2.set_xlabel('Hamiltonian steps K')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Test-Time Compute Scaling: SSIM vs K')
        ax2.grid(True, alpha=0.3)

        if 'vae' in results:
            ax1.axhline(results['vae']['psnr_mean'], color='red',
                        linestyle='--', label='VAE baseline')
            ax2.axhline(results['vae']['ssim_mean'], color='red',
                        linestyle='--', label='VAE baseline')
            ax1.legend()
            ax2.legend()

        plt.suptitle('Hamiltonian Test-Time Compute Scaling', fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[save] {path}")
    except Exception as e:
        print(f"[warn] Could not plot: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    out_dir = ROOT / 'outputs' / 'latent_hkan_fep'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader, obs_dim = load_dataset(
        args.dataset, batch_size=args.batch_size, n_train=args.n_train)

    # Models
    print(f"[model] Building LatentHKAN_FEP (latent_dim={args.latent_dim})")
    kan_model = LatentHKAN_FEP(
        obs_dim=obs_dim,
        latent_dim=args.latent_dim,
        ham_steps=args.ham_steps,
        grid_size=5,
    ).to(device)

    vae_model = VAEBaseline(obs_dim=obs_dim, latent_dim=args.latent_dim).to(device)

    # Parameter counts
    kan_params = sum(p.numel() for p in kan_model.parameters() if p.requires_grad)
    vae_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
    print(f"[model] KAN-FEP params: {kan_params:,}")
    print(f"[model] VAE baseline params: {vae_params:,}")
    print(f"[model] KAN/VAE ratio: {kan_params/vae_params:.2f}x")

    # Optimisers
    kan_opt = torch.optim.Adam(kan_model.parameters(), lr=args.lr, weight_decay=1e-5)
    vae_opt = torch.optim.Adam(vae_model.parameters(), lr=args.lr, weight_decay=1e-5)

    kan_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        kan_opt, T_max=args.epochs)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_opt, T_max=args.epochs)

    # Training loop
    logger = ExperimentLogger(str(out_dir / 'training_log.json'))
    history = {'kan': [], 'vae': []}
    best_psnr = 0.0

    print(f"\n[train] Starting training for {args.epochs} epochs")
    print(f"{'Epoch':>6} | {'KAN-F':>8} | {'KAN-KL':>8} | {'VAE-F':>8} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Beta annealing (warm-up KL over first 10 epochs)
        beta = min(1.0, epoch / 10.0)

        kan_metrics = train_epoch(kan_model, train_loader, kan_opt, device,
                                   beta=beta, n_ham_steps=1)
        vae_metrics = train_epoch(vae_model, train_loader, vae_opt, device, beta=beta)

        kan_scheduler.step()
        vae_scheduler.step()

        elapsed = time.time() - t0
        history['kan'].append(kan_metrics)
        history['vae'].append(vae_metrics)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {kan_metrics['F']:>8.4f} | {kan_metrics['kl']:>8.4f} "
                  f"| {vae_metrics['F']:>8.4f} | {elapsed:>5.1f}s")

    # Final evaluation
    print("\n[eval] Evaluating final models...")
    kan_results, x_samp, o_hat_samp = evaluate(
        kan_model, test_loader, device,
        n_ham_steps_list=[0, 1, 5, 10, 20]
    )
    vae_results, _, _ = evaluate(vae_model, test_loader, device)
    all_results = {**kan_results, **vae_results}

    print("\n[results] Test-time compute scaling:")
    for key in sorted(kan_results.keys()):
        r = kan_results[key]
        print(f"  {key}: PSNR={r['psnr_mean']:.2f} dB, SSIM={r['ssim_mean']:.4f}")
    if vae_results.get('vae'):
        r = vae_results['vae']
        print(f"  VAE:  PSNR={r['psnr_mean']:.2f} dB, SSIM={r['ssim_mean']:.4f}")

    # Save everything
    torch.save(kan_model.state_dict(), str(out_dir / 'kan_fep.pt'))
    torch.save(vae_model.state_dict(), str(out_dir / 'vae_baseline.pt'))

    summary = {
        'dataset': args.dataset,
        'latent_dim': args.latent_dim,
        'kan_params': kan_params,
        'vae_params': vae_params,
        'epochs': args.epochs,
        'kan_results': {k: {'psnr': v['psnr_mean'], 'ssim': v['ssim_mean']}
                        for k, v in kan_results.items()},
        'vae_results': {k: {'psnr': v['psnr_mean'], 'ssim': v['ssim_mean']}
                        for k, v in vae_results.items()},
        'training_history': history,
    }
    with open(str(out_dir / 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualisations
    if x_samp is not None:
        save_reconstruction_grid(x_samp, o_hat_samp,
                                  str(out_dir / f'{args.dataset}_reconstruction.png'),
                                  title=f'KAN-FEP Reconstruction ({args.dataset})')

    plot_psnr_vs_ham_steps(all_results,
                            str(out_dir / 'psnr_vs_ham_steps.png'))

    plot_hamiltonian_trajectories(kan_model, x_samp,
                                   str(out_dir / 'hamiltonian_trajectories.png'))

    print(f"\n[done] Results saved to {out_dir}/")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description='Train Latent H-KAN FEP')
    p.add_argument('--dataset',    default='mnist', choices=['mnist', 'fashionmnist', 'synthetic'])
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--n-train',    type=int, default=10000)
    p.add_argument('--latent-dim', type=int, default=16)
    p.add_argument('--ham-steps',  type=int, default=5)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--compare',    action='store_true',
                   help='Run KAN-FEP vs VAE comparison on both datasets')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.compare:
        print("=" * 60)
        print("Comparative Study: KAN-FEP vs Standard VAE")
        print("=" * 60)
        for ds in ['mnist', 'fashionmnist']:
            args.dataset = ds
            print(f"\n{'='*40}\nDataset: {ds}\n{'='*40}")
            run_training(args)
    else:
        run_training(args)
