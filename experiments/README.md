# Paper 1 Experiments: KAN-Hamiltonian Energy Fields

## Quick Start

```bash
# From the project root directory:
cd /path/to/ToE

# Quick test (synthetic data, 10 epochs, ~5 min on CPU):
python experiments/run_experiments.py --quick

# Full suite (all datasets, 30 epochs, ~30-60 min on GPU):
python experiments/run_experiments.py --full

# With GPU:
python experiments/run_experiments.py --full --device cuda
```

## Requirements

```bash
pip install torch matplotlib numpy
pip install torchvision  # Optional: enables MNIST/FashionMNIST experiments
```

## Experiment Structure

### 1. Denoising (exp_denoise.py)
Compares KAN vs MLP energy density on image denoising via gradient-based inference.

```bash
python experiments/exp_denoise.py --dataset synthetic --epochs 30
python experiments/exp_denoise.py --dataset mnist --epochs 30
python experiments/exp_denoise.py --dataset synthetic --all  # noise ablation
```

**Metrics:** PSNR, SSIM, MSE
**Ablations:** noise σ ∈ {0.1, 0.3, 0.5, 0.7}

### 2. Sudoku (exp_sudoku.py)
Compares KAN vs MLP on constraint satisfaction (Sudoku as 9-channel energy minimization).

```bash
python experiments/exp_sudoku.py --epochs 30 --n-clues 35
```

**Metrics:** cell accuracy (empty cells), solve rate, constraint violations

### 3. Interpretability (kan_interpretability.py)
Generates visualizations showing KAN's learned spline functions vs MLP black box.

Runs automatically as part of the full suite, or:
```bash
python experiments/run_experiments.py --interpret-only
```

### 4. Ablations
- **KAN grid size:** {3, 5, 8, 12} — measures quality vs model complexity
- **Evolution steps:** {1, 2, 5, 10, 20, 50} — convergence analysis

```bash
python experiments/run_experiments.py --ablation-only
```

## Output Structure

```
results/
├── summary.json              # All metrics in one file
├── denoise/
│   ├── denoise_kan_*.png     # KAN denoising visualizations
│   ├── denoise_mlp_*.png     # MLP denoising visualizations
│   ├── training_curves_*.png # Loss comparison plots
│   ├── noise_ablation_*.png  # PSNR/SSIM vs noise level
│   ├── kan_denoise_*.pth     # Trained KAN models
│   └── mlp_denoise_*.pth     # Trained MLP models
├── sudoku/
│   ├── sudoku_kan.png        # KAN solving visualization
│   ├── sudoku_mlp.png        # MLP solving visualization
│   ├── sudoku_training.png   # Training curves
│   └── *.pth                 # Trained models
├── interpretability/
│   ├── kan_spatial_filters.png
│   ├── kan_spline_activations.png
│   ├── kan_energy_landscape.png
│   ├── mlp_energy_landscape.png
│   └── interpretability_comparison.png
└── ablations/
    ├── grid_size_ablation.png
    └── steps_ablation.png
```

## Key Files

| File | Purpose |
|------|---------|
| `run_experiments.py` | Master script — runs everything |
| `exp_denoise.py` | Denoising experiment (KAN vs MLP) |
| `exp_sudoku.py` | Sudoku experiment (KAN vs MLP) |
| `mlp_field.py` | MLP baseline (matches HamiltonianField architecture) |
| `metrics.py` | PSNR, SSIM, Sudoku accuracy, logging |
| `kan_interpretability.py` | KAN spline visualization & analysis |

## Architecture Comparison

Both models share the SAME infrastructure:
- Spatial filter bank (learned 5×5 conv kernels)
- Diffusion coefficients
- Gradient-based inference (Langevin dynamics)

The ONLY difference is the energy density function:
- **KAN-Hamiltonian:** `E(features) = KAN(features)` — B-spline basis, interpretable
- **MLP-Baseline:** `E(features) = MLP(features)` — ReLU/SiLU basis, black box

This controlled comparison isolates the effect of using KAN for the energy function.
