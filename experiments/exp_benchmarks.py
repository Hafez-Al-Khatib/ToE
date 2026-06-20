"""
Benchmark Experiments for Paper 1
=====================================
Standard denoising benchmarks to position the KAN-EBM against known baselines.

Benchmarks
----------
Dataset   | Noise | Baselines           | Our Model
----------|-------|---------------------|--------------------
MNIST     | σ=0.1 | BM3D*, FFN, MLP-EBM | KAN-PC-Field
MNIST     | σ=0.2 | BM3D*, FFN, MLP-EBM | KAN-PC-Field
MNIST     | σ=0.3 | BM3D*, FFN, MLP-EBM | KAN-PC-Field
FashionM. | σ=0.2 | BM3D*, FFN, MLP-EBM | KAN-PC-Field

(* BM3D is non-learning; best classical denoising baseline)

Metrics: PSNR (dB), SSIM, inference time (ms)

Interpretability Bonus
-----------------------
In addition to PSNR/SSIM, we show:
- KAN spline visualisations (what each B-spline learned)
- Energy landscape cross-sections (where clean images sit)
- Filter bank visualisations (learned spatial patterns)

Usage
-----
python experiments/exp_benchmarks.py --quick
python experiments/exp_benchmarks.py --full --dataset mnist fashionmnist
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

from predictive_coding_field import PredictiveCodingField
from metrics import compute_psnr, compute_ssim, ExperimentLogger, print_comparison_table


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_benchmark_data(name: str, n_test: int = 500) -> torch.utils.data.DataLoader:
    """Load test data for benchmarking."""
    try:
        import torchvision
        transform = torchvision.transforms.ToTensor()
        if name == 'mnist':
            ds = torchvision.datasets.MNIST('/tmp/data', train=False,
                                             download=True, transform=transform)
        elif name == 'fashionmnist':
            ds = torchvision.datasets.FashionMNIST('/tmp/data', train=False,
                                                    download=True, transform=transform)
        else:
            raise ValueError(f"Unknown: {name}")
        ds = torch.utils.data.Subset(ds, range(min(n_test, len(ds))))
        return torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False)
    except Exception:
        return None   # Will use synthetic


def load_benchmark_train(name: str, n_train: int = 5000) -> torch.utils.data.DataLoader:
    try:
        import torchvision
        transform = torchvision.transforms.ToTensor()
        if name == 'mnist':
            ds = torchvision.datasets.MNIST('/tmp/data', train=True,
                                             download=True, transform=transform)
        elif name == 'fashionmnist':
            ds = torchvision.datasets.FashionMNIST('/tmp/data', train=True,
                                                    download=True, transform=transform)
        else:
            raise ValueError(f"Unknown: {name}")
        ds = torch.utils.data.Subset(ds, range(n_train))
        return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
    except Exception:
        return None


def synthetic_test_loader(n: int = 200, side: int = 28) -> torch.utils.data.DataLoader:
    torch.manual_seed(42)
    x = torch.zeros(n, 1, side, side)
    for i in range(n):
        cx, cy = np.random.randint(8, side-8, 2)
        r = np.random.randint(3, 6)
        for a in range(side):
            for b in range(side):
                if (a-cy)**2 + (b-cx)**2 < r**2:
                    x[i, 0, a, b] = 0.9
    x = x.clamp(0, 1)
    y = torch.zeros(n, dtype=torch.long)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=100, shuffle=False)


# ── Model Definitions ─────────────────────────────────────────────────────────

def build_kan_model(side: int = 28, quick: bool = False) -> PredictiveCodingField:
    """Our primary model: KAN-parameterised predictive coding field."""
    return PredictiveCodingField(
        n_channels=1,
        height=side,
        width=side,
        n_filters=16 if not quick else 8,
        filter_size=5,
        kan_hidden=[32, 16] if not quick else [16, 8],
    )


class MLPEBMBaseline(nn.Module):
    """
    MLP-EBM baseline: Same spatial filter bank as PredictiveCodingField,
    but MLP energy density instead of KAN.
    This is the fair comparison (only energy function differs).
    """

    def __init__(self, n_channels=1, n_filters=16, filter_size=5, height=28, width=28):
        super().__init__()
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.height = height
        self.width = width

        # Identical spatial filter bank to PredictiveCodingField
        self.spatial_filters = nn.ParameterList([
            nn.Parameter(torch.randn(n_filters, 1, filter_size, filter_size) * 0.05)
            for _ in range(n_channels)
        ])

        # MLP energy density (replaces KAN)
        self.mlp_energy = nn.Sequential(
            nn.Linear(n_channels * n_filters, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )

        self.log_step_size = nn.Parameter(torch.tensor(-1.0))

    @property
    def step_size(self):
        return torch.exp(self.log_step_size)

    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = self.spatial_filters[0].shape[-1] // 2
        all_features = []
        for c in range(C):
            u_c = x[:, c:c+1, :, :]
            f_c = F.conv2d(u_c, self.spatial_filters[c], padding=pad)
            all_features.append(f_c)
        features = torch.cat(all_features, dim=1)
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        energy_density = self.mlp_energy(features_flat)
        return energy_density.sum()

    def denoising_loss(self, x_clean, x_noisy):
        u = x_noisy.requires_grad_(True) if not x_noisy.requires_grad else x_noisy
        u = x_noisy.detach().requires_grad_(True)
        E = self.compute_energy(u)
        grad = torch.autograd.grad(E, u, create_graph=True)[0]
        x_pred = u - self.step_size * grad
        loss = F.mse_loss(x_pred[:, :1], x_clean[:, :1])
        return loss + 0.01 * (grad ** 2).mean()

    def forward(self, x_noisy, n_steps=10):
        u = x_noisy.clone()
        for _ in range(n_steps):
            u = u.detach().requires_grad_(True)
            E = self.compute_energy(u)
            grad = torch.autograd.grad(E, u)[0]
            u = u.detach() - self.step_size.detach() * grad
        return u[:, :1]


class FFNBaseline(nn.Module):
    """Plain feedforward denoiser (no energy, no iterative refinement)."""

    def __init__(self, obs_dim=784, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, obs_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1)).view_as(x)

    def denoising_loss(self, x_clean, x_noisy):
        return F.mse_loss(self.forward(x_noisy), x_clean)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader,
    device: torch.device,
    n_epochs: int,
    sigma: float = 0.2,
    name: str = 'model',
):
    """Generic training loop for denoising models."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = []
        for x, _ in train_loader:
            x_c = x.to(device)
            x_n = (x_c + sigma * torch.randn_like(x_c)).clamp(0, 1)
            opt.zero_grad()
            if isinstance(model, FFNBaseline):
                loss = model.denoising_loss(x_c, x_n)
            else:
                loss = model.denoising_loss(x_c, x_n)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss.append(loss.item())
        sched.step()

        if (epoch + 1) % max(1, n_epochs // 4) == 0:
            print(f"    [{name}] epoch {epoch+1}/{n_epochs}: loss={np.mean(epoch_loss):.4f}")

    model.eval()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    sigma: float = 0.2,
    n_steps_list: Optional[List[int]] = None,
    model_name: str = 'model',
) -> Dict:
    """Evaluate denoising quality at different inference step counts."""
    if n_steps_list is None:
        n_steps_list = [1, 5, 10, 20]

    results = {}
    t_start = time.time()
    n_batches = 0

    for x, _ in (test_loader or []):
        x_c = x.to(device)
        x_n = (x_c + sigma * torch.randn_like(x_c)).clamp(0, 1)

        if isinstance(model, FFNBaseline):
            x_hat = model(x_n)
            psnr = compute_psnr(x_hat, x_c).item()
            ssim = compute_ssim(x_hat, x_c).item()
            key = 'K=1'
            if key not in results:
                results[key] = {'psnr': [], 'ssim': []}
            results[key]['psnr'].append(psnr)
            results[key]['ssim'].append(ssim)
        else:
            for K in n_steps_list:
                x_hat = model(x_n, n_steps=K)
                psnr = compute_psnr(x_hat, x_c).item()
                ssim = compute_ssim(x_hat, x_c).item()
                key = f'K={K}'
                if key not in results:
                    results[key] = {'psnr': [], 'ssim': []}
                results[key]['psnr'].append(psnr)
                results[key]['ssim'].append(ssim)

        n_batches += 1
        if n_batches >= 5:
            break   # Limit for speed

    elapsed = (time.time() - t_start) / n_batches if n_batches > 0 else 0

    for key in results:
        results[key]['psnr_mean'] = float(np.mean(results[key]['psnr']))
        results[key]['ssim_mean'] = float(np.mean(results[key]['ssim']))
    results['inference_time_per_batch_ms'] = elapsed * 1000

    return results


# ── Main Benchmark Runner ─────────────────────────────────────────────────────

def run_benchmarks(
    datasets: List[str] = None,
    sigma_list: List[float] = None,
    quick: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[benchmarks] Device: {device}")

    if datasets is None:
        datasets = ['mnist']
    if sigma_list is None:
        sigma_list = [0.2] if quick else [0.1, 0.2, 0.3]

    n_epochs = 5 if quick else 20
    n_steps = [1, 5, 10] if quick else [1, 5, 10, 20]

    out_dir = ROOT / 'outputs' / 'benchmarks'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*50}\nDataset: {dataset_name}\n{'='*50}")

        train_loader = load_benchmark_train(dataset_name, n_train=5000 if not quick else 1000)
        test_loader  = load_benchmark_data(dataset_name, n_test=500 if not quick else 100)

        if train_loader is None:
            print(f"[warn] {dataset_name} unavailable, using synthetic")
            train_loader = synthetic_test_loader(n=800)
            test_loader  = synthetic_test_loader(n=200)

        ds_results = {}

        for sigma in sigma_list:
            print(f"\nNoise σ={sigma}")
            sigma_results = {}

            # ── Build + train models ──────────────────────────────────────
            print("  Training KAN-PC-Field...")
            kan_model = build_kan_model(quick=quick)
            train_model(kan_model, train_loader, device, n_epochs, sigma=sigma, name='KAN')
            kan_params = sum(p.numel() for p in kan_model.parameters())

            print("  Training MLP-EBM baseline...")
            mlp_model = MLPEBMBaseline(n_filters=16 if not quick else 8)
            train_model(mlp_model, train_loader, device, n_epochs, sigma=sigma, name='MLP-EBM')
            mlp_params = sum(p.numel() for p in mlp_model.parameters())

            print("  Training FFN baseline...")
            ffn_model = FFNBaseline()
            train_model(ffn_model, train_loader, device, n_epochs, sigma=sigma, name='FFN')
            ffn_params = sum(p.numel() for p in ffn_model.parameters())

            # ── Evaluate ──────────────────────────────────────────────────
            kan_eval = evaluate_model(kan_model, test_loader, device, sigma,
                                       n_steps_list=n_steps, model_name='KAN')
            mlp_eval = evaluate_model(mlp_model, test_loader, device, sigma,
                                       n_steps_list=n_steps, model_name='MLP-EBM')
            ffn_eval = evaluate_model(ffn_model, test_loader, device, sigma,
                                       model_name='FFN')

            # ── Print comparison table ────────────────────────────────────
            print(f"\n  Results at σ={sigma}:")
            print(f"  {'Model':<20} {'PSNR (K=best)':<16} {'SSIM (K=best)':<16} {'Params':<12}")
            print(f"  {'-'*64}")

            kan_best_psnr = max(v['psnr_mean'] for v in kan_eval.values() if isinstance(v, dict) and 'psnr_mean' in v)
            mlp_best_psnr = max(v['psnr_mean'] for v in mlp_eval.values() if isinstance(v, dict) and 'psnr_mean' in v)
            ffn_best_psnr = max(v['psnr_mean'] for v in ffn_eval.values() if isinstance(v, dict) and 'psnr_mean' in v)

            kan_best_ssim = max(v['ssim_mean'] for v in kan_eval.values() if isinstance(v, dict) and 'ssim_mean' in v)
            mlp_best_ssim = max(v['ssim_mean'] for v in mlp_eval.values() if isinstance(v, dict) and 'ssim_mean' in v)
            ffn_best_ssim = max(v['ssim_mean'] for v in ffn_eval.values() if isinstance(v, dict) and 'ssim_mean' in v)

            for name, best_psnr, best_ssim, params in [
                ('KAN-PC-Field (ours)', kan_best_psnr, kan_best_ssim, kan_params),
                ('MLP-EBM (baseline)',  mlp_best_psnr, mlp_best_ssim, mlp_params),
                ('FFN (feedforward)',   ffn_best_psnr, ffn_best_ssim, ffn_params),
            ]:
                print(f"  {name:<20} {best_psnr:<16.2f} {best_ssim:<16.4f} {params:<12,}")

            sigma_results = {
                'kan':     kan_eval,
                'mlp_ebm': mlp_eval,
                'ffn':     ffn_eval,
                'params':  {'kan': kan_params, 'mlp_ebm': mlp_params, 'ffn': ffn_params},
            }
            ds_results[f'sigma={sigma}'] = sigma_results

            # Save model weights
            torch.save(kan_model.state_dict(),
                       str(out_dir / f'{dataset_name}_kan_sigma{sigma}.pt'))

        all_results[dataset_name] = ds_results

        # Visualise best model
        _save_benchmark_figure(
            kan_model, test_loader, device,
            sigma=sigma_list[len(sigma_list)//2],
            path=str(out_dir / f'{dataset_name}_denoising_comparison.png'),
            title=f'{dataset_name.upper()} Denoising: KAN-PC-Field',
        )
        _save_psnr_table_figure(ds_results, sigma_list,
                                 str(out_dir / f'{dataset_name}_psnr_table.png'))

    # Save JSON
    with open(str(out_dir / 'benchmark_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (torch.Tensor, np.floating)) else x)

    print(f"\n[done] Benchmark results saved to {out_dir}/")
    return all_results


# ── Visualisation ─────────────────────────────────────────────────────────────

def _save_benchmark_figure(
    model, test_loader, device, sigma, path, title,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for x, _ in (test_loader or []):
            x_c = x[:8].to(device)
            x_n = (x_c + sigma * torch.randn_like(x_c)).clamp(0, 1)
            with torch.no_grad():
                x_hat_1  = model(x_n, n_steps=1)
                x_hat_10 = model(x_n, n_steps=10)
            break

        n = 8
        fig, axes = plt.subplots(4, n, figsize=(2*n, 8))
        fig.suptitle(title, fontweight='bold', fontsize=12)
        rows = [x_c, x_n, x_hat_1, x_hat_10]
        labels = ['Clean', f'Noisy (σ={sigma})', 'Denoised K=1', 'Denoised K=10']

        for r, (imgs, lbl) in enumerate(zip(rows, labels)):
            for i in range(n):
                img = imgs[i].squeeze().cpu().float().numpy()
                axes[r, i].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[r, i].axis('off')
            axes[r, 0].set_ylabel(lbl, fontsize=9, rotation=90, va='center')

        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[save] {path}")
    except Exception as e:
        print(f"[warn] Figure failed: {e}")


def _save_psnr_table_figure(ds_results: Dict, sigma_list: List[float], path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(sigma_list))
        width = 0.25

        kan_psnrs, mlp_psnrs, ffn_psnrs = [], [], []
        for sigma in sigma_list:
            key = f'sigma={sigma}'
            if key in ds_results:
                r = ds_results[key]
                kan_vals = r.get('kan', {})
                mlp_vals = r.get('mlp_ebm', {})
                ffn_vals = r.get('ffn', {})
                kan_best = max((v['psnr_mean'] for v in kan_vals.values()
                                if isinstance(v, dict) and 'psnr_mean' in v), default=0)
                mlp_best = max((v['psnr_mean'] for v in mlp_vals.values()
                                if isinstance(v, dict) and 'psnr_mean' in v), default=0)
                ffn_best = max((v['psnr_mean'] for v in ffn_vals.values()
                                if isinstance(v, dict) and 'psnr_mean' in v), default=0)
                kan_psnrs.append(kan_best)
                mlp_psnrs.append(mlp_best)
                ffn_psnrs.append(ffn_best)

        ax.bar(x - width, kan_psnrs, width, label='KAN-PC-Field (ours)', color='steelblue')
        ax.bar(x,         mlp_psnrs, width, label='MLP-EBM (baseline)',  color='darkorange')
        ax.bar(x + width, ffn_psnrs, width, label='FFN (feedforward)',   color='green')

        ax.set_xlabel('Noise Level σ')
        ax.set_ylabel('Best PSNR (dB)')
        ax.set_title('Denoising PSNR: KAN-PC-Field vs Baselines')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sigma_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[save] {path}")
    except Exception as e:
        print(f"[warn] Figure failed: {e}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Run denoising benchmarks')
    p.add_argument('--quick',    action='store_true')
    p.add_argument('--dataset',  nargs='+', default=['mnist'])
    p.add_argument('--sigma',    nargs='+', type=float, default=[0.2])
    args = p.parse_args()
    run_benchmarks(datasets=args.dataset, sigma_list=args.sigma, quick=args.quick)
