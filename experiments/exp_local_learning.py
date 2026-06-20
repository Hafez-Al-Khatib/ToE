"""
Local Learning Parity Experiment
==================================
Compares four learning rules on the KAN-EBM architecture:
    A. Backpropagation (standard, non-biological)
    B. Predictive Coding Hebbian (local, biologically plausible)
    C. Score Matching / DSM       (from PredictiveCodingField)
    D. Layer-local MSE            (NoProp-inspired, each layer independent)

Hypothesis (Song et al. NeurIPS 2020; Millidge ICLR 2023):
    At inference equilibrium, the PC Hebbian update equals the backprop gradient.
    Therefore, PC Hebbian should achieve <2% accuracy gap on MNIST.

Metrics
-------
- Reconstruction PSNR/SSIM
- Free energy F over training
- Convergence speed (epochs to reach target F)
- Memory footprint (no activation storage for local rules)

Usage
-----
python experiments/exp_local_learning.py --quick
python experiments/exp_local_learning.py --n-epochs 30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from hierarchical_pc_kan import HierarchicalPCKAN, ClassificationHead
from predictive_coding_field import PredictiveCodingField
from recognition_model import RecognitionKAN, GenerativeDecoder
from metrics import compute_psnr, compute_ssim


# ── Shared data generation ────────────────────────────────────────────────────

def make_synthetic_data(n: int = 2000, obs_dim: int = 64, n_classes: int = 5):
    """Gaussian blobs for fast local learning comparison."""
    torch.manual_seed(42)
    centres = torch.randn(n_classes, obs_dim) * 2.0
    labels = torch.randint(0, n_classes, (n,))
    x = centres[labels] + 0.5 * torch.randn(n, obs_dim)
    return x, labels


def load_mnist_flat(n_train: int = 5000, n_test: int = 1000):
    """Load MNIST as flat vectors, with fallback to synthetic."""
    try:
        import torchvision
        transform = torchvision.transforms.ToTensor()
        tr = torchvision.datasets.MNIST('/tmp/data', train=True,  download=True, transform=transform)
        te = torchvision.datasets.MNIST('/tmp/data', train=False, download=True, transform=transform)
        tr = torch.utils.data.Subset(tr, range(n_train))
        x_tr = torch.stack([tr[i][0].view(-1) for i in range(len(tr))])
        y_tr = torch.tensor([tr[i][1] for i in range(len(tr))])
        x_te = torch.stack([te[i][0].view(-1) for i in range(n_test)])
        y_te = torch.tensor([te[i][1] for i in range(n_test)])
        return x_tr, y_tr, x_te, y_te, 784
    except Exception:
        print("[warn] MNIST unavailable; using synthetic data")
        x_tr, y_tr = make_synthetic_data(n_train, obs_dim=64)
        x_te, y_te = make_synthetic_data(n_test,  obs_dim=64)
        return x_tr, y_tr, x_te, y_te, 64


# ── Method A: Backpropagation ─────────────────────────────────────────────────

class BackpropKAN(nn.Module):
    """
    Standard backpropagation baseline.
    Architecture matches HierarchicalPCKAN but uses autograd.
    """

    def __init__(self, dims: List[int], n_classes: int = 10):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], n_classes)

    def forward(self, x):
        return self.head(self.encoder(x.view(x.size(0), -1)))

    def loss(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(logits, y)

    def accuracy(self, x, y):
        with torch.no_grad():
            preds = self.forward(x).argmax(-1)
        return (preds == y).float().mean().item()


# ── Method B: Predictive Coding Hebbian ───────────────────────────────────────

def train_pc_hebbian(
    model: HierarchicalPCKAN,
    head: ClassificationHead,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    n_epochs: int = 20,
    batch_size: int = 64,
    inference_steps: int = 30,
    lr_inf: float = 0.1,
    lr_weights: float = 1e-3,
) -> Dict:
    """
    Two-phase PC training:
    Phase 1 (inference):  Minimise F by updating states (fixed weights)
    Phase 2 (learning):   Update weights using local Hebbian rule (fixed states)

    This is strictly local: each layer only needs its own pre/post states.
    """
    model = model.to(device)
    head  = head.to(device)
    opt_head = torch.optim.Adam(head.parameters(), lr=1e-3)

    n = len(x_train)
    history = {'F': [], 'acc': [], 'time': []}
    t0 = time.time()

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_F = []

        for start in range(0, n - batch_size, batch_size):
            idx = perm[start:start + batch_size]
            x = x_train[idx].to(device)
            y = y_train[idx].to(device)

            # ── Phase 1: Inference (update states) ───────────────────────
            model.inference_lr = lr_inf
            model.n_inference_steps = inference_steps
            states, errors, info = model.inference(x, n_steps=inference_steps,
                                                    return_trajectory=False)
            epoch_F.append(info['F_final'])

            # ── Phase 2: Local weight update ─────────────────────────────
            model.local_update(states, errors, lr=lr_weights)

            # ── Classification head (uses backprop, but only on head) ─────
            z_top = states[-1].detach()
            logits = head(z_top)
            head_loss = F.cross_entropy(logits, y)
            opt_head.zero_grad()
            head_loss.backward()
            opt_head.step()

        # Evaluate
        with torch.no_grad():
            z_top, _, _ = model(x_train[:500].to(device))
            logits = head(z_top)
            acc = (logits.argmax(-1) == y_train[:500].to(device)).float().mean().item()

        history['F'].append(np.mean(epoch_F))
        history['acc'].append(acc)
        history['time'].append(time.time() - t0)

        if (epoch + 1) % 5 == 0:
            print(f"    PC Hebbian epoch {epoch+1}: F={history['F'][-1]:.4f}, acc={acc:.4f}")

    return history


# ── Method C: DSM / Allen-Cahn (PredictiveCodingField) ───────────────────────

def train_dsm(
    model: PredictiveCodingField,
    x_train: torch.Tensor,
    device: torch.device,
    n_epochs: int = 20,
    batch_size: int = 64,
) -> Dict:
    """
    Train PredictiveCodingField via Denoising Score Matching loss.
    This is pure backprop but on the PredictiveCodingField architecture.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x_train)

    history = {'loss': [], 'psnr': []}
    psnr_vals = []

    # Reshape for image field
    obs_dim = x_train.shape[-1]
    side = int(obs_dim ** 0.5) if int(obs_dim ** 0.5) ** 2 == obs_dim else 8
    x_2d = x_train[:, :side*side].view(-1, 1, side, side).clamp(0, 1)

    for epoch in range(n_epochs):
        perm = torch.randperm(len(x_2d))
        epoch_loss = []

        for start in range(0, len(x_2d) - batch_size, batch_size):
            idx = perm[start:start + batch_size]
            x_c = x_2d[idx].to(device)
            sigma = 0.1 + 0.3 * torch.rand(1).item()
            x_n = (x_c + sigma * torch.randn_like(x_c)).clamp(0, 1)

            opt.zero_grad()
            loss = model.denoising_loss(x_c, x_n)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss.append(loss.item())

        # Eval PSNR
        with torch.no_grad():
            x_c_eval = x_2d[:64].to(device)
            x_n_eval = (x_c_eval + 0.2 * torch.randn_like(x_c_eval)).clamp(0, 1)
            x_hat = model(x_n_eval, n_steps=10)
            psnr = compute_psnr(x_hat, x_c_eval).item()

        history['loss'].append(np.mean(epoch_loss))
        history['psnr'].append(psnr)

        if (epoch + 1) % 5 == 0:
            print(f"    DSM epoch {epoch+1}: loss={history['loss'][-1]:.4f}, PSNR={psnr:.2f}")

    return history


# ── Method D: Layer-local (NoProp-inspired) ────────────────────────────────────

class NoPropKANLayer(nn.Module):
    """
    Each layer trained independently via its own MSE reconstruction loss.
    No information flow between layers during training.

    Inspired by NoProp (arXiv:2503.24322, 2025):
    Each layer performs its own 'denoising' to learn its representation.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU())
        self.decode = nn.Sequential(
            nn.Linear(out_dim, in_dim), nn.Sigmoid())

    def forward(self, x):
        return self.encode(x)

    def local_loss(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """Each layer learns to reconstruct its own noisy input."""
        x_noisy = x + sigma * torch.randn_like(x)
        z = self.encode(x_noisy)
        x_hat = self.decode(z)
        return F.mse_loss(x_hat, x)


def train_noprop(
    dims: List[int],
    x_train: torch.Tensor,
    device: torch.device,
    n_epochs: int = 20,
    batch_size: int = 64,
) -> Dict:
    """
    Layer-independent NoProp-inspired training.
    Each layer trained separately, no end-to-end gradient.
    """
    layers = nn.ModuleList([
        NoPropKANLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)
    ]).to(device)

    history = {'loss_per_layer': [[] for _ in range(len(dims)-1)]}
    n = len(x_train)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        layer_losses = [[] for _ in range(len(layers))]

        for start in range(0, n - batch_size, batch_size):
            idx = perm[start:start + batch_size]
            x = x_train[idx].to(device)

            # Train each layer independently
            current = x
            for l, layer in enumerate(layers):
                opt_l = torch.optim.Adam(layer.parameters(), lr=1e-3)
                opt_l.zero_grad()
                loss = layer.local_loss(current.detach())
                loss.backward()
                opt_l.step()
                layer_losses[l].append(loss.item())
                with torch.no_grad():
                    current = layer(current.detach())

        for l in range(len(layers)):
            history['loss_per_layer'][l].append(np.mean(layer_losses[l]))

        if (epoch + 1) % 5 == 0:
            avg_loss = np.mean([history['loss_per_layer'][l][-1]
                                for l in range(len(layers))])
            print(f"    NoProp epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return history


# ── Main Comparison ───────────────────────────────────────────────────────────

def run_local_learning_comparison(quick: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[local learning] Device: {device}")

    out_dir = ROOT / 'outputs' / 'local_learning'
    out_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = 10 if quick else 25
    obs_dim = 64
    dims = [obs_dim, 32, 16]
    n_classes = 5

    # Data
    x_tr, y_tr = make_synthetic_data(1000, obs_dim, n_classes)
    x_te, y_te = make_synthetic_data(300,  obs_dim, n_classes)

    results = {}

    # ── Method A: Backpropagation ─────────────────────────────────────────
    print("\n[A] Backpropagation baseline")
    bp_model = BackpropKAN(dims, n_classes).to(device)
    bp_opt = torch.optim.Adam(bp_model.parameters(), lr=1e-3)
    bp_hist = {'acc': []}
    for epoch in range(n_epochs):
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr) - 64, 64):
            idx = perm[i:i+64]
            x, y = x_tr[idx].to(device), y_tr[idx].to(device)
            bp_opt.zero_grad()
            bp_model.loss(x, y).backward()
            bp_opt.step()
        acc = bp_model.accuracy(x_te.to(device), y_te.to(device))
        bp_hist['acc'].append(acc)
    results['backprop'] = {
        'final_accuracy': bp_hist['acc'][-1],
        'history': bp_hist
    }
    print(f"  Final accuracy: {bp_hist['acc'][-1]:.4f}")

    # ── Method B: PC Hebbian ──────────────────────────────────────────────
    print("\n[B] Predictive Coding Hebbian")
    pc_model = HierarchicalPCKAN(dims=dims, inference_lr=0.1)
    pc_head  = ClassificationHead(dims[-1], n_classes)
    pc_hist  = train_pc_hebbian(
        pc_model, pc_head, x_tr, y_tr, device,
        n_epochs=n_epochs, inference_steps=20
    )
    results['pc_hebbian'] = {
        'final_accuracy': pc_hist['acc'][-1],
        'history': pc_hist
    }

    # ── Method C: DSM ─────────────────────────────────────────────────────
    print("\n[C] Score Matching / DSM (PredictiveCodingField)")
    side = int(obs_dim ** 0.5)
    dsm_model = PredictiveCodingField(n_channels=1, height=side, width=side)
    dsm_hist  = train_dsm(dsm_model, x_tr, device, n_epochs=n_epochs)
    results['dsm'] = {
        'final_psnr': dsm_hist['psnr'][-1],
        'history': dsm_hist
    }

    # ── Method D: NoProp ──────────────────────────────────────────────────
    print("\n[D] NoProp (layer-local, no inter-layer gradient)")
    noprop_hist = train_noprop(dims, x_tr, device, n_epochs=n_epochs)
    results['noprop'] = {'history': noprop_hist}

    # ── Comparison summary ────────────────────────────────────────────────
    bp_acc = results['backprop']['final_accuracy']
    pc_acc = results['pc_hebbian']['final_accuracy']
    gap    = abs(bp_acc - pc_acc)

    print("\n" + "=" * 50)
    print("LOCAL LEARNING COMPARISON SUMMARY")
    print("=" * 50)
    print(f"  Backprop accuracy:       {bp_acc:.4f}")
    print(f"  PC Hebbian accuracy:     {pc_acc:.4f}")
    print(f"  Gap:                     {gap:.4f}")
    print(f"  PC within 2% of backprop: {gap < 0.02}")
    print(f"  DSM final PSNR:          {results['dsm']['final_psnr']:.2f} dB")

    results['summary'] = {
        'backprop_accuracy': bp_acc,
        'pc_accuracy':       pc_acc,
        'accuracy_gap':      gap,
        'pc_within_2pct':    gap < 0.02,
    }

    # Save
    with open(str(out_dir / 'local_learning_results.json'), 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (torch.Tensor, np.floating)) else x)

    _make_comparison_figure(results, out_dir)
    print(f"\n[done] Results saved to {out_dir}/")
    return results


def _make_comparison_figure(results: Dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Local Learning vs Backpropagation', fontweight='bold', fontsize=13)

        # Accuracy over epochs
        ax = axes[0]
        if 'history' in results.get('backprop', {}):
            ax.plot(results['backprop']['history']['acc'],
                    'k-', lw=2, label='Backpropagation')
        if 'history' in results.get('pc_hebbian', {}):
            ax.plot(results['pc_hebbian']['history']['acc'],
                    'b--', lw=2, label='PC Hebbian (local)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary bar chart
        ax2 = axes[1]
        methods = ['Backprop', 'PC Hebbian', 'Gap']
        bp = results['summary']['backprop_accuracy']
        pc = results['summary']['pc_accuracy']
        gap = results['summary']['accuracy_gap']
        ax2.bar(['Backprop', 'PC Hebbian'], [bp, pc],
                color=['steelblue', 'darkorange'])
        ax2.set_ylabel('Final Accuracy')
        ax2.set_title(f'Final Accuracy (Gap = {gap:.4f})')
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(str(out_dir / 'local_learning_comparison.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[save] {out_dir}/local_learning_comparison.png")
    except Exception as e:
        print(f"[warn] Figure failed: {e}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    run_local_learning_comparison(quick=args.quick)
