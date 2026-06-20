"""
run_reviewer_experiments.py
============================
Master script that runs all three reviewer-demanded experiments in sequence.

Experiments:
  1. exp_extended_scaling    — K up to 200 on MNIST (reviewer #1: "does scaling plateau?")
  2. exp_natural_diffusion_comparison — matched-param diffusion baseline on CBSD68
                                        (reviewer #2: "natural image comparison required")
  3. exp_cifar10             — color images, 3-channel (reviewer #3: "beyond MNIST")

Usage:
  # Full run (GPU strongly recommended, takes ~2–6h depending on hardware):
  python experiments/run_reviewer_experiments.py --device cuda

  # Quick smoke test (~5-10 min, reduced data/epochs):
  python experiments/run_reviewer_experiments.py --device cuda --quick

  # Run only specific experiments:
  python experiments/run_reviewer_experiments.py --device cuda --only extended
  python experiments/run_reviewer_experiments.py --device cuda --only natural
  python experiments/run_reviewer_experiments.py --device cuda --only cifar10

  # Custom epoch counts:
  python experiments/run_reviewer_experiments.py --device cuda \\
      --epochs_extended 100 --epochs_natural 80 --epochs_cifar10 60

After this script completes, update paper.tex with:
  - Table from outputs/extended_scaling/table_extended_scaling.tex
  - Table from outputs/natural_diffusion_comparison/table.tex
  - Table from outputs/cifar10/table_cifar10.tex
  - Figures from outputs/*/psnr_vs_k.pdf

Expected runtimes (NVIDIA A100 / RTX 3090):
  Extended scaling (50 epochs MNIST):       ~20 min
  Natural diffusion comparison (50 epochs): ~45 min (3 models × CBSD68 patches)
  CIFAR-10 (50 epochs):                     ~60 min (3 models × 32×32 color)
  Total:                                    ~2.5 hours
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
EXPS  = Path(__file__).resolve().parent


def run(cmd, tag):
    print(f'\n{"="*65}')
    print(f'  Running: {tag}')
    print(f'  Command: {" ".join(cmd)}')
    print(f'{"="*65}\n')
    t0  = time.time()
    ret = subprocess.run(cmd, check=False)
    dt  = time.time() - t0
    status = 'SUCCESS' if ret.returncode == 0 else f'FAILED (code {ret.returncode})'
    print(f'\n  {tag}: {status}  ({dt/60:.1f} min)')
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',           default='auto')
    parser.add_argument('--quick',            action='store_true',
                        help='Quick smoke test: 5 epochs, reduced data')
    parser.add_argument('--only',             default=None,
                        choices=['extended', 'natural', 'cifar10'],
                        help='Run only one experiment')
    parser.add_argument('--epochs_extended',  type=int, default=50,
                        help='Epochs for extended scaling (default: 50)')
    parser.add_argument('--k_max',            type=int, default=200,
                        help='Max K for extended scaling (default: 200)')
    parser.add_argument('--epochs_natural',   type=int, default=50,
                        help='Epochs for natural diffusion comparison (default: 50)')
    parser.add_argument('--sigma_natural',    type=int, default=25,
                        help='Noise level for natural images [0-255] (default: 25)')
    parser.add_argument('--full_eval',        action='store_true',
                        help='Run sliding-window full-image evaluation (slow)')
    parser.add_argument('--epochs_cifar10',   type=int, default=50,
                        help='Epochs for CIFAR-10 (default: 50)')
    parser.add_argument('--sigma_cifar10',    type=float, default=0.2,
                        help='Noise sigma for CIFAR-10 in [-1,1] space (default: 0.2)')
    args = parser.parse_args()

    py = sys.executable
    device_flag = ['--device', args.device]
    quick_flag  = ['--quick'] if args.quick else []
    statuses    = {}
    t0_total    = time.time()

    # ── Experiment 1: Extended K scaling ──────────────────────────────────────
    if args.only in (None, 'extended'):
        cmd = [py, str(EXPS / 'exp_extended_scaling.py')] + device_flag + quick_flag + [
            '--epochs', str(args.epochs_extended if not args.quick else 5),
            '--k_max',  str(50 if args.quick else args.k_max),
            '--n_runs', '3',
        ]
        statuses['extended'] = run(cmd, 'Extended K Scaling (K up to 200, MNIST)')

    # ── Experiment 2: Natural image diffusion comparison ──────────────────────
    if args.only in (None, 'natural'):
        cmd = [py, str(EXPS / 'exp_natural_diffusion_comparison.py')] + device_flag + quick_flag + [
            '--epochs', str(args.epochs_natural if not args.quick else 5),
            '--sigma',  str(args.sigma_natural),
        ]
        if args.full_eval:
            cmd.append('--full_eval')
        statuses['natural'] = run(cmd, 'Natural Image Diffusion Comparison (CBSD68)')

    # ── Experiment 3: CIFAR-10 ────────────────────────────────────────────────
    if args.only in (None, 'cifar10'):
        cmd = [py, str(EXPS / 'exp_cifar10.py')] + device_flag + quick_flag + [
            '--epochs', str(args.epochs_cifar10 if not args.quick else 5),
            '--sigma',  str(args.sigma_cifar10),
        ]
        statuses['cifar10'] = run(cmd, 'CIFAR-10 Color Denoising')

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.time() - t0_total
    print(f'\n{"="*65}')
    print(f'  ALL EXPERIMENTS DONE  ({total/60:.1f} min total)')
    print(f'{"="*65}')
    for name, ok in statuses.items():
        icon = '✓' if ok else '✗'
        print(f'  {icon}  {name}')

    print('\n  Output locations:')
    for d in ['extended_scaling', 'natural_diffusion_comparison', 'cifar10']:
        out = ROOT / 'outputs' / d
        if out.exists():
            files = list(out.iterdir())
            for f in files:
                print(f'    {f}')

    print('\n  Next steps:')
    print('  1. Paste LaTeX table snippets into paper.tex')
    print('  2. Add psnr_vs_k.pdf figures to paper figures directory')
    print('  3. Update abstract & contributions with new results')
    print('  4. Re-run python experiments/exp_interpretability.py to regenerate figs')

    any_failed = any(not v for v in statuses.values())
    sys.exit(1 if any_failed else 0)


if __name__ == '__main__':
    main()
