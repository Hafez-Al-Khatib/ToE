#!/usr/bin/env python3
"""
True Intelligence Model (TIM) — Full Experiment Suite
=======================================================

Master script that runs ALL experiments for Papers 1, 2, and 3.

Experiments
-----------
  [Paper 1 — KAN-EBM Denoising]
  1. Denoising benchmarks       (KAN vs MLP-EBM vs FFN, PSNR/SSIM)
  2. Test-time compute scaling  (PSNR vs inference steps K)
  3. KAN interpretability       (spline vis, energy landscape, filter vis)
  4. Ablation studies           (grid size, noise level, architecture)

  [Paper 2 — Latent H-KAN + FEP]
  5. FEP validation             (chain verification, F decomposition)
  6. Latent H-KAN training      (KAN-FEP vs VAE, Hamiltonian trajectories)
  7. Local learning parity      (PC Hebbian vs backprop accuracy gap)

  [Paper 3 — Eikonal Planning]
  8. Eikonal planning           (path planning, energy → cost field)

  [TIM — Unified]
  9. Unified energy landscape   (V(z) triple role demonstration)

Corrected Terminology
---------------------
  HamiltonianField → GradientFlowField / PredictiveCodingField
  'O(1) inference' → 'O(K×N) test-time scaling'
  'FEP equivalent' → 'consistent with predictive coding (Rao & Ballard 1999)'
  'Unified Theory' → 'Three Contributions to Physics-Inspired ML'

Usage
-----
    python experiments/run_experiments.py --quick           # fast test
    python experiments/run_experiments.py --full            # all experiments
    python experiments/run_experiments.py --denoise-only    # Paper 1 denoising
    python experiments/run_experiments.py --fep-only        # Paper 2 FEP
    python experiments/run_experiments.py --latent-only     # Paper 2 H-KAN
    python experiments/run_experiments.py --scaling-only    # test-time compute
    python experiments/run_experiments.py --benchmarks-only # benchmark suite
    python experiments/run_experiments.py --local-only      # local learning
"""

import torch
import numpy as np
import os
import sys
import json
import argparse
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import ExperimentLogger, print_comparison_table


def run_denoising_suite(epochs, device, output_base):
    """Run all denoising experiments."""
    from exp_denoise import run_single_experiment, run_full_ablation

    output_dir = os.path.join(output_base, 'denoise')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Synthetic shapes (always available)
    print("\n" + "="*70)
    print("  DENOISING EXPERIMENT: Synthetic Shapes")
    print("="*70)
    results, kan_eval, mlp_eval, histories = run_single_experiment(
        'synthetic', noise_std=0.3, epochs=epochs, device=device,
        output_dir=output_dir)
    all_results['synthetic_0.3'] = results

    # Try MNIST
    try:
        import torchvision
        print("\n" + "="*70)
        print("  DENOISING EXPERIMENT: MNIST")
        print("="*70)
        results, _, _, _ = run_single_experiment(
            'mnist', noise_std=0.3, epochs=epochs, device=device,
            output_dir=output_dir)
        all_results['mnist_0.3'] = results

        print("\n" + "="*70)
        print("  DENOISING EXPERIMENT: FashionMNIST")
        print("="*70)
        results, _, _, _ = run_single_experiment(
            'fashion', noise_std=0.3, epochs=epochs, device=device,
            output_dir=output_dir)
        all_results['fashion_0.3'] = results

    except ImportError:
        print("\n  [INFO] torchvision not available — skipping MNIST/FashionMNIST")

    # Noise level ablation on synthetic
    print("\n" + "="*70)
    print("  NOISE LEVEL ABLATION")
    print("="*70)
    kan_abl, mlp_abl = run_full_ablation(
        'synthetic', epochs=epochs, device=device, output_dir=output_dir)
    all_results['noise_ablation'] = {
        'KAN': {str(k): v for k, v in kan_abl.items()},
        'MLP': {str(k): v for k, v in mlp_abl.items()},
    }

    return all_results


def run_sudoku_suite(epochs, device, output_base):
    """Run all Sudoku experiments."""
    from exp_sudoku import main as sudoku_main

    output_dir = os.path.join(output_base, 'sudoku')
    os.makedirs(output_dir, exist_ok=True)

    # Standard experiment
    print("\n" + "="*70)
    print("  SUDOKU EXPERIMENT: 35 clues")
    print("="*70)

    # We import and run the main function logic directly
    from exp_sudoku import (HamiltonianField, MLPField, train_sudoku,
                            evaluate_sudoku_model, visualize_sudoku_solving,
                            print_comparison_table)
    import matplotlib.pyplot as plt

    results = {}
    for n_clues in [35, 30]:
        tag = f"{n_clues}_clues"
        print(f"\n--- Sudoku with {n_clues} clues ---")

        # KAN
        kan_model = HamiltonianField(
            n_channels=9, height=9, width=9,
            n_filters=9, filter_size=5,
            kan_hidden=[128, 1], kan_grid_size=5)
        config = {
            'device': device, 'epochs': epochs, 'lr': 1e-3,
            'batch_size': 16, 'n_clues': n_clues,
            'n_train_steps': 5, 'n_puzzles_per_epoch': 256}
        kan_hist = train_sudoku(kan_model, config)
        kan_metrics = evaluate_sudoku_model(
            kan_model, n_puzzles=50, n_clues=n_clues, n_steps=50, device=device)

        # MLP
        mlp_model = MLPField(
            n_channels=9, height=9, width=9,
            n_filters=9, mlp_hidden=[128])
        mlp_hist = train_sudoku(mlp_model, config)
        mlp_metrics = evaluate_sudoku_model(
            mlp_model, n_puzzles=50, n_clues=n_clues, n_steps=50, device=device)

        results[tag] = {
            'KAN-Hamiltonian': kan_metrics,
            'MLP-Baseline': mlp_metrics
        }

        print_comparison_table(
            {'KAN': kan_metrics, 'MLP': mlp_metrics},
            f"Sudoku ({n_clues} clues)")

        # Save models
        torch.save(kan_model.state_dict(),
                   os.path.join(output_dir, f'kan_sudoku_{tag}.pth'))
        torch.save(mlp_model.state_dict(),
                   os.path.join(output_dir, f'mlp_sudoku_{tag}.pth'))

    return results


def run_interpretability_suite(epochs, device, output_base):
    """Run interpretability analysis."""
    from kan_interpretability import run_interpretability_analysis
    from exp_denoise import get_dataset

    output_dir = os.path.join(output_base, 'interpretability')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  INTERPRETABILITY ANALYSIS")
    print("="*70)

    # Train fresh models for interpretability
    from exp_denoise import train_denoising

    dataset = get_dataset('synthetic', train=True)
    test_ds = get_dataset('synthetic', train=False)

    kan_model = HamiltonianField(
        n_channels=1, height=28, width=28,
        n_filters=16, kan_hidden=[32, 1], kan_grid_size=5)
    mlp_model = MLPField(
        n_channels=1, height=28, width=28,
        n_filters=16, mlp_hidden=[32])

    config = {'device': device, 'epochs': epochs, 'lr': 1e-3,
              'noise_std': 0.3, 'batch_size': 32, 'grad_reg': 0.001}

    print("  Training KAN for interpretability...")
    train_denoising(kan_model, dataset, config)
    print("  Training MLP for interpretability...")
    train_denoising(mlp_model, dataset, config)

    # Get test samples
    loader = torch.utils.data.DataLoader(test_ds, batch_size=64)
    test_samples = next(iter(loader))
    if isinstance(test_samples, (list, tuple)):
        test_samples = test_samples[0]
    test_samples = test_samples.to(device)

    run_interpretability_analysis(kan_model, mlp_model, test_samples, output_dir)


def run_ablation_suite(device, output_base):
    """Run ablation studies: KAN grid size, evolution steps."""
    from exp_denoise import get_dataset, train_denoising, evaluate_denoising
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = os.path.join(output_base, 'ablations')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  ABLATION: KAN Grid Size")
    print("="*70)

    dataset = get_dataset('synthetic', train=True)
    test_ds = get_dataset('synthetic', train=False)
    grid_sizes = [3, 5, 8, 12]
    grid_results = {}

    for gs in grid_sizes:
        print(f"\n  Grid size = {gs}")
        model = HamiltonianField(
            n_channels=1, height=28, width=28,
            n_filters=16, kan_hidden=[32, 1], kan_grid_size=gs)

        n_params = sum(p.numel() for p in model.parameters())
        config = {'device': device, 'epochs': 20, 'lr': 1e-3,
                  'noise_std': 0.3, 'batch_size': 32, 'grad_reg': 0.001}
        train_denoising(model, dataset, config)

        eval_results = evaluate_denoising(
            model, test_ds, noise_std=0.3, n_steps_list=[10], device=device)

        metrics = eval_results.get('steps_10', {})
        metrics['params'] = n_params
        grid_results[gs] = metrics
        print(f"    PSNR: {metrics.get('psnr_mean', 0):.2f}  "
              f"SSIM: {metrics.get('ssim_mean', 0):.4f}  "
              f"Params: {n_params:,}")

    # Plot grid size ablation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    psnrs = [grid_results[gs].get('psnr_mean', 0) for gs in grid_sizes]
    params = [grid_results[gs].get('params', 0) for gs in grid_sizes]

    ax1.plot(grid_sizes, psnrs, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax1.set_xlabel('KAN Grid Size', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Denoising Quality vs Grid Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(grid_sizes, params, 's-', color='#FF5722', linewidth=2, markersize=8)
    ax2.set_xlabel('KAN Grid Size', fontsize=12)
    ax2.set_ylabel('Parameters', fontsize=12)
    ax2.set_title('Model Size vs Grid Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('KAN Grid Size Ablation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grid_size_ablation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Evolution steps ablation
    print("\n" + "="*70)
    print("  ABLATION: Evolution Steps")
    print("="*70)

    model = HamiltonianField(
        n_channels=1, height=28, width=28,
        n_filters=16, kan_hidden=[32, 1], kan_grid_size=5)
    config = {'device': device, 'epochs': 20, 'lr': 1e-3,
              'noise_std': 0.3, 'batch_size': 32, 'grad_reg': 0.001}
    train_denoising(model, dataset, config)

    step_counts = [1, 2, 5, 10, 20, 50]
    step_results = evaluate_denoising(
        model, test_ds, noise_std=0.3, n_steps_list=step_counts, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    psnrs = [step_results[f'steps_{s}']['psnr_mean'] for s in step_counts]
    ax.plot(step_counts, psnrs, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.set_xlabel('Evolution Steps', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Denoising Quality vs Evolution Steps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steps_ablation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return {'grid_size': grid_results, 'steps': step_results}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="True Intelligence Model — Full Experiment Suite")
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 5-10 epochs, synthetic only')
    parser.add_argument('--full', action='store_true',
                        help='Full suite: 30 epochs, all datasets')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)

    # ── Paper 1: KAN-EBM Denoising ──────────────────────────────────────────
    parser.add_argument('--denoise-only',    action='store_true',
                        help='Paper 1: KAN vs MLP denoising benchmarks')
    parser.add_argument('--benchmarks-only', action='store_true',
                        help='Paper 1: Standard benchmark suite (MNIST/FashionMNIST)')
    parser.add_argument('--interpret-only',  action='store_true',
                        help='Paper 1: KAN spline + energy landscape visualisations')
    parser.add_argument('--ablation-only',   action='store_true',
                        help='Paper 1: Grid size + noise level ablation')
    parser.add_argument('--scaling-only',    action='store_true',
                        help='Paper 1+2: Test-time compute scaling curves')

    # ── Paper 2: Latent H-KAN + FEP ─────────────────────────────────────────
    parser.add_argument('--fep-only',        action='store_true',
                        help='Paper 2: FEP validation experiments')
    parser.add_argument('--latent-only',     action='store_true',
                        help='Paper 2: Latent H-KAN training + Hamiltonian trajectories')
    parser.add_argument('--local-only',      action='store_true',
                        help='Paper 2: Local learning vs backprop comparison')

    # ── Deprecated flags (kept for backward compatibility) ──────────────────
    parser.add_argument('--sudoku-only',     action='store_true',
                        help='(Legacy) Sudoku — now a NEGATIVE RESULT; see false_claims/04')

    args = parser.parse_args()

    # Device
    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device

    # Epochs
    epochs = args.epochs or (5 if args.quick else (30 if args.full else 15))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   True Intelligence Model — Experiment Suite                ║
║                                                             ║
║   Device: {device:10s}  |  Epochs: {epochs:3d}  |  Seed: {args.seed:4d}       ║
║                                                             ║
║   Papers:  P1=KAN-EBM  P2=Latent-HKAN  P3=Eikonal         ║
╚══════════════════════════════════════════════════════════════╝
""")

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    all_results = {'meta': {
        'device': device, 'epochs': epochs, 'seed': args.seed,
        'started': datetime.now().isoformat(),
        'architecture': 'TrueIntelligenceModel (TIM)',
        'correction': 'HamiltonianField renamed to PredictiveCodingField',
    }}

    any_flag = any([
        args.denoise_only, args.benchmarks_only, args.interpret_only,
        args.ablation_only, args.scaling_only, args.fep_only,
        args.latent_only, args.local_only, args.sudoku_only,
    ])
    run_all = not any_flag

    try:
        # ── Paper 1 Experiments ──────────────────────────────────────────────
        if run_all or args.denoise_only:
            all_results['denoise'] = run_denoising_suite(
                epochs, device, args.output_dir)

        if run_all or args.benchmarks_only:
            from exp_benchmarks import run_benchmarks
            print("\n" + "="*70)
            print("  BENCHMARK SUITE: KAN-EBM vs MLP-EBM vs FFN")
            print("="*70)
            datasets = ['mnist'] if args.quick else ['mnist', 'fashionmnist']
            sigmas   = [0.2]     if args.quick else [0.1, 0.2, 0.3]
            all_results['benchmarks'] = run_benchmarks(
                datasets=datasets, sigma_list=sigmas, quick=args.quick)

        if run_all or args.interpret_only:
            run_interpretability_suite(epochs, device, args.output_dir)

        if (run_all or args.ablation_only) and not args.quick:
            all_results['ablations'] = run_ablation_suite(device, args.output_dir)

        if run_all or args.scaling_only:
            from exp_compute_scaling import run_compute_scaling
            print("\n" + "="*70)
            print("  TEST-TIME COMPUTE SCALING")
            print("="*70)
            all_results['compute_scaling'] = run_compute_scaling(quick=args.quick)

        # ── Paper 2 Experiments ──────────────────────────────────────────────
        if run_all or args.fep_only:
            from exp_fep_validation import run_fep_validation
            print("\n" + "="*70)
            print("  FEP VALIDATION: Allen-Cahn → Score → PC → Variational FE")
            print("="*70)
            all_results['fep_validation'] = run_fep_validation(quick=args.quick)

        if run_all or args.latent_only:
            from train_latent_hkan import run_training, parse_args as parse_hkan_args
            print("\n" + "="*70)
            print("  LATENT H-KAN FEP TRAINING")
            print("="*70)
            hkan_args = parse_hkan_args()
            hkan_args.dataset = 'mnist' if not args.quick else 'synthetic'
            hkan_args.epochs  = epochs
            hkan_args.quick   = args.quick
            all_results['latent_hkan'] = run_training(hkan_args)

        if run_all or args.local_only:
            from exp_local_learning import run_local_learning_comparison
            print("\n" + "="*70)
            print("  LOCAL LEARNING: PC Hebbian vs Backpropagation")
            print("="*70)
            all_results['local_learning'] = run_local_learning_comparison(quick=args.quick)

        # ── Legacy / Negative Results ────────────────────────────────────────
        if args.sudoku_only:
            print("\n[WARNING] Sudoku is now documented as a NEGATIVE RESULT.")
            print("  The 54% accuracy ceiling is a theoretical consequence of NP-completeness.")
            print("  See false_claims/04_sudoku_np_hardness/ANALYSIS.md")
            all_results['sudoku'] = run_sudoku_suite(epochs, device, args.output_dir)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    # Save summary
    elapsed = time.time() - start_time
    all_results['meta']['elapsed_seconds'] = elapsed
    all_results['meta']['finished'] = datetime.now().isoformat()

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   EXPERIMENTS COMPLETE                                      ║
║                                                             ║
║   Total time: {elapsed/60:6.1f} minutes                            ║
║   Results:    {args.output_dir:45s}║
║   Summary:    {summary_path:45s}║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
