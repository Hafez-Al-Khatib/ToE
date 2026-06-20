"""
compile_results.py
==================
Aggregate all experimental results into updated paper tables and figures.

Reads:
  - outputs/scaled_eval/cifar10_scaled_500.json
  - outputs/dncnn_baseline/dncnn_cifar10_results.json
  - outputs/diffusion_cifar10/diffusion_cifar10_results.json
  - outputs/finalization/rebuttal/oracle_recovery.json
  - outputs/finalization/rebuttal/smooth_mlp_stats_independent.json

Generates:
  - outputs/scaled_eval/table_scaled_cifar10.tex
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'scaled_eval'


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def make_psnr_table(data, sigmas, models, K_list):
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering\small')
    header = r'\begin{tabular}{lr|' + 'r' * len(K_list) + '}'
    lines.append(header)
    lines.append(r'\toprule')
    lines.append('Model & Params & ' + ' & '.join('$K={}$'.format(K) for K in K_list) + r' \\')
    lines.append(r'\midrule')

    for model_name in models:
        if model_name not in data.get(str(sigmas[0]), {}):
            continue
        n_params = 'TBD'
        row = [model_name, n_params]
        for K in K_list:
            vals = []
            for sigma in sigmas:
                if str(sigma) in data and model_name in data[str(sigma)] and str(K) in data[str(sigma)][model_name]:
                    vals.append(data[str(sigma)][model_name][str(K)]['psnr'])
            if vals:
                row.append('{:.2f}'.format(np.mean(vals)))
            else:
                row.append('--')
        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{PSNR (dB) on CIFAR-10 test set (500 images).}')
    lines.append(r'\label{tab:scaled}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def main():
    print("Compiling results...")

    scaled = load_json(OUT / 'cifar10_scaled_500.json')
    dncnn = load_json(ROOT / 'outputs' / 'dncnn_baseline' / 'dncnn_cifar10_results.json')
    diffusion = load_json(ROOT / 'outputs' / 'diffusion_cifar10' / 'diffusion_cifar10_results.json')
    oracle = load_json(ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'oracle_recovery.json')

    print("\n=== Oracle Recovery ===")
    if oracle:
        print(f"Average recovery: {oracle['average_recovery_percent']:.2f}%")

    print("\n=== Scaled Evaluation Summary ===")
    if scaled:
        sigmas = scaled['sigmas']
        models = list(scaled['results'][str(sigmas[0])].keys())
        for sigma in sigmas:
            print(f"\nsigma={sigma}")
            for model in models:
                best_k = max(scaled['results'][str(sigma)][model], key=lambda k: scaled['results'][str(sigma)][model][k]['psnr'])
                best_psnr = scaled['results'][str(sigma)][model][best_k]['psnr']
                best_ssim = scaled['results'][str(sigma)][model][best_k]['ssim']
                print(f"  {model:18s}: K={best_k:2s}  PSNR={best_psnr:.2f}  SSIM={best_ssim:.3f}")

    print("\n=== DnCNN Baseline ===")
    if dncnn:
        print(f"Params: {dncnn['n_params']:,}")
        for sigma, r in dncnn['results'].items():
            print(f"  sigma={sigma}: PSNR={r['psnr_mean']:.2f}  SSIM={r['ssim_mean']:.3f}")

    print("\n=== Diffusion Baseline ===")
    if diffusion:
        print(f"Params: {diffusion['n_params']:,}")
        for sigma in diffusion['results']:
            best_k = max(diffusion['results'][sigma], key=lambda k: diffusion['results'][sigma][k]['psnr_mean'])
            r = diffusion['results'][sigma][best_k]
            print(f"  sigma={sigma}: K={best_k}  PSNR={r['psnr_mean']:.2f}  SSIM={r['ssim_mean']:.3f}")

    print("\nCompilation complete.")


if __name__ == '__main__':
    main()
