"""
plot_scaling_laws.py
====================
Generates the final NeurIPS figures for the parameter scaling ablation.
Reads the results from the overnight run and plots the K-scaling curves
for the Nano (16K), Base (64K), and Large (256K) KAN-EBM models on a single graph.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'outputs' / 'cifar10'

def main():
    if not OUT_DIR.exists():
        print(f"Error: {OUT_DIR} not found.")
        return

    # Load results
    results = {}
    for f in [8, 32, 128]:
        res_file = OUT_DIR / f'results_f{f}.json'
        if res_file.exists():
            with open(res_file, 'r') as file:
                results[f] = json.load(file)
        else:
            print(f"Warning: {res_file} not found. Did the overnight run finish?")

    if not results:
        print("No results found. Run run_overnight.ps1 first.")
        return

    # 1. Plot K-Scaling across parameter scales
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {8: '#4C6EF5', 32: '#FA5252', 128: '#12B886'}
    labels = {8: 'KAN-EBM (Nano ~16K)', 32: 'KAN-EBM (Base ~64K)', 128: 'KAN-EBM (Large ~256K)'}
    markers = {8: 'o-', 32: 's-', 128: '^-'}

    for f, res in results.items():
        if 'KAN-EBM' not in res['results']: continue
        
        kan_res = res['results']['KAN-EBM']
        K_LIST = [int(k) for k in kan_res.keys()]
        psnrs = [float(v) for v in kan_res.values()]
        
        ax.plot(K_LIST, psnrs, markers[f], color=colors[f], linewidth=2.5,
                markersize=8, label=labels[f])

    ax.set_xlabel('Test-Time Inference Steps (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CIFAR-10 Denoising PSNR (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Test-Time Compute Scaling Across Parameter Scales', fontsize=16, fontweight='bold')
    
    if max(K_LIST) >= 20:
        ax.set_xscale('log')
        ax.set_xticks(K_LIST)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')
    
    fig.tight_layout()
    out_path = OUT_DIR / 'paper_scaling_laws.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'paper_scaling_laws.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Successfully generated scaling laws figure: {out_path}")

if __name__ == '__main__':
    main()
