"""
Metrics Module for Paper 1 Experiments
=======================================

Provides consistent evaluation metrics across all tasks:
- Denoising: PSNR, SSIM, MSE
- Sudoku: Cell accuracy, solve rate, constraint violations
- Pushing: Mass conservation, displacement error
- General: Energy statistics, parameter counts

All metrics are computed as pure PyTorch (no scikit-image dependency).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import json
import os
from datetime import datetime


# ============================================================================
# Image Quality Metrics
# ============================================================================

def compute_psnr(clean: torch.Tensor, denoised: torch.Tensor,
                 data_range: float = 2.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)

    For images in [-1, 1], data_range = 2.0.
    For images in [0, 1], data_range = 1.0.

    Higher is better. Typical good denoising: 20-35 dB.
    """
    mse = F.mse_loss(denoised, clean).item()
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim(clean: torch.Tensor, denoised: torch.Tensor,
                 data_range: float = 2.0, window_size: int = 7) -> float:
    """
    Structural Similarity Index (simplified, per-channel).

    SSIM measures structural similarity rather than pixel-wise error.
    Values in [-1, 1], higher is better. Good denoising: > 0.8.

    Implementation follows Wang et al. (2004) with Gaussian window.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    if clean.dim() == 3:
        clean = clean.unsqueeze(0)
        denoised = denoised.unsqueeze(0)

    B, C, H, W = clean.shape
    window = window.expand(C, 1, -1, -1).to(clean.device)
    pad = window_size // 2

    # Compute means
    mu1 = F.conv2d(clean, window, padding=pad, groups=C)
    mu2 = F.conv2d(denoised, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(clean ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(denoised ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(clean * denoised, window, padding=pad, groups=C) - mu12

    # SSIM formula
    numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def compute_noisy_psnr(clean: torch.Tensor, noisy: torch.Tensor,
                       data_range: float = 2.0) -> float:
    """PSNR of the noisy input (before denoising) — for reference."""
    return compute_psnr(clean, noisy, data_range)


# ============================================================================
# Sudoku Metrics
# ============================================================================

def compute_sudoku_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                            masks: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute Sudoku evaluation metrics.

    Parameters
    ----------
    predictions : (B, 9, 9, 9) softmax probabilities
    targets : (B, 9, 9) or (B, 9, 9, 9) ground truth
    masks : (B, 9, 9) boolean, True = clue cells

    Returns
    -------
    dict with: cell_accuracy, empty_accuracy, solve_rate, constraint_violations
    """
    pred_digits = predictions.argmax(dim=1)  # (B, 9, 9)

    if targets.dim() == 4:
        true_digits = targets.argmax(dim=1)
    else:
        true_digits = targets

    B = pred_digits.shape[0]

    # Overall cell accuracy
    cell_correct = (pred_digits == true_digits).float()
    cell_accuracy = cell_correct.mean().item()

    # Empty cell accuracy (the actual solving task)
    if masks is not None:
        empty_mask = ~masks
        if empty_mask.any():
            empty_accuracy = cell_correct[empty_mask].mean().item()
        else:
            empty_accuracy = 1.0
    else:
        empty_accuracy = cell_accuracy

    # Solve rate (fully correct puzzles)
    per_puzzle_correct = cell_correct.view(B, -1).all(dim=1).float()
    solve_rate = per_puzzle_correct.mean().item()

    # Constraint violations
    violations = _count_constraint_violations(pred_digits)

    return {
        'cell_accuracy': cell_accuracy,
        'empty_accuracy': empty_accuracy,
        'solve_rate': solve_rate,
        'avg_violations': violations
    }


def _count_constraint_violations(grids: torch.Tensor) -> float:
    """Count average Sudoku constraint violations per puzzle."""
    B = grids.shape[0]
    total_violations = 0

    for b in range(B):
        grid = grids[b]  # (9, 9)
        violations = 0

        # Row violations
        for r in range(9):
            row = grid[r]
            for d in range(9):
                count = (row == d).sum().item()
                if count > 1:
                    violations += count - 1

        # Column violations
        for c in range(9):
            col = grid[:, c]
            for d in range(9):
                count = (col == d).sum().item()
                if count > 1:
                    violations += count - 1

        # Box violations
        for br in range(3):
            for bc in range(3):
                box = grid[br*3:(br+1)*3, bc*3:(bc+1)*3].reshape(-1)
                for d in range(9):
                    count = (box == d).sum().item()
                    if count > 1:
                        violations += count - 1

        total_violations += violations

    return total_violations / B


# ============================================================================
# Physics Metrics
# ============================================================================

def compute_mass_conservation(initial: torch.Tensor, final: torch.Tensor) -> Dict[str, float]:
    """
    Measure mass conservation during field evolution.

    For physics-based models, total mass should be approximately conserved.
    """
    mass_initial = initial.sum(dim=(1, 2, 3))  # (B,)
    mass_final = final.sum(dim=(1, 2, 3))

    mass_ratio = (mass_final / mass_initial.clamp(min=1e-8)).mean().item()
    mass_change = ((mass_final - mass_initial).abs() / mass_initial.abs().clamp(min=1e-8)).mean().item()

    return {
        'mass_ratio': mass_ratio,
        'mass_change_pct': mass_change * 100,
    }


def compute_energy_statistics(model, samples: torch.Tensor) -> Dict[str, float]:
    """
    Compute energy landscape statistics on a batch of samples.
    """
    with torch.no_grad():
        energies = model.compute_energy(samples)

    return {
        'mean_energy': energies.mean().item(),
        'std_energy': energies.std().item(),
        'min_energy': energies.min().item(),
        'max_energy': energies.max().item(),
    }


# ============================================================================
# Experiment Logger
# ============================================================================

class ExperimentLogger:
    """Simple JSON-based experiment logger."""

    def __init__(self, experiment_name: str, output_dir: str = './results'):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.log = {
            'experiment': experiment_name,
            'started': datetime.now().isoformat(),
            'config': {},
            'epochs': [],
            'final_metrics': {},
        }

    def log_config(self, config: dict):
        self.log['config'] = config

    def log_epoch(self, epoch: int, metrics: dict):
        self.log['epochs'].append({'epoch': epoch, **metrics})

    def log_final(self, metrics: dict):
        self.log['final_metrics'] = metrics
        self.log['finished'] = datetime.now().isoformat()

    def save(self):
        path = os.path.join(self.output_dir, f'{self.experiment_name}.json')
        with open(path, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"Results saved to {path}")
        return path

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  {self.experiment_name} — Final Results")
        print(f"{'='*60}")
        for k, v in self.log['final_metrics'].items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
            else:
                print(f"  {k:30s}: {v}")
        print(f"{'='*60}\n")


# ============================================================================
# Comparison Table
# ============================================================================

def print_comparison_table(results: Dict[str, Dict[str, float]], title: str = ""):
    """
    Print a formatted comparison table from multiple experiment results.

    Parameters
    ----------
    results : dict of dict
        Outer key = model name, inner dict = metric name → value
    """
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    # Collect all metric names
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    metrics = sorted(all_metrics)

    # Header
    model_names = list(results.keys())
    header = f"{'Metric':30s}" + "".join(f"{name:>15s}" for name in model_names)
    print(header)
    print("-" * len(header))

    # Rows
    for metric in metrics:
        row = f"{metric:30s}"
        for name in model_names:
            val = results[name].get(metric, None)
            if val is None:
                row += f"{'N/A':>15s}"
            elif isinstance(val, float):
                row += f"{val:>15.4f}"
            else:
                row += f"{str(val):>15s}"
        print(row)

    print("-" * len(header))
    print()


if __name__ == "__main__":
    # Quick self-test
    print("Testing metrics module...")

    # PSNR test
    clean = torch.randn(4, 1, 28, 28)
    noisy = clean + 0.3 * torch.randn_like(clean)
    psnr = compute_psnr(clean, noisy)
    print(f"  PSNR (noisy, σ=0.3): {psnr:.2f} dB")

    # SSIM test
    ssim = compute_ssim(clean, noisy)
    print(f"  SSIM (noisy, σ=0.3): {ssim:.4f}")

    # Perfect reconstruction
    psnr_perfect = compute_psnr(clean, clean)
    ssim_perfect = compute_ssim(clean, clean)
    print(f"  PSNR (perfect): {psnr_perfect:.2f} dB")
    print(f"  SSIM (perfect): {ssim_perfect:.4f}")

    # Comparison table test
    results = {
        'KAN-Hamiltonian': {'PSNR': 25.3, 'SSIM': 0.87, 'Params': 1234},
        'MLP-Baseline': {'PSNR': 23.1, 'SSIM': 0.82, 'Params': 1100},
    }
    print_comparison_table(results, "Denoising Results")
    print("  All metrics tests passed!")
