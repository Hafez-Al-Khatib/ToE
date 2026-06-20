import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    res_path = Path('outputs/cleanup/robustness_and_transfer.json')
    if not res_path.exists():
        print("No Cleanup results found.")
        return
    with open(res_path, 'r') as f:
        res = json.load(f)
    
    # ── Robustness Plot ──
    rob = res['robustness']
    eps_list = rob['eps_list']
    # Use K=10 results
    kan_rob = rob['kan']['refined']['10']
    mlp_rob = rob['mlp']['refined']['10']
    
    plt.figure(figsize=(6, 5))
    plt.plot(eps_list, kan_rob, 'o-', label='KAN-EBM (K=10)', color='blue')
    plt.plot(eps_list, mlp_rob, 's--', label='MLP-EBM (K=10)', color='red')
    plt.xlabel('Adversarial Epsilon (FGSM)')
    plt.ylabel('Refined PSNR (dB)')
    plt.title('Adversarial Robustness (MNIST)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_rob = Path('outputs/cleanup/robustness_plot.pdf')
    plt.savefig(out_rob, bbox_inches='tight')
    plt.savefig(out_rob.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved plot: {out_rob}")
    
    # ── Transfer Table/Bar ──
    tr = res['transfer']
    K_list = tr['K_list']
    kan_tr = [tr['kan_mean_per_K'][str(k)] for k in K_list]
    mlp_tr = [tr['mlp_mean_per_K'][str(k)] for k in K_list]
    
    plt.figure(figsize=(6, 5))
    plt.plot(K_list, kan_tr, 'o-', label='KAN-EBM', color='blue')
    plt.plot(K_list, mlp_tr, 's--', label='MLP-EBM', color='red')
    plt.xlabel('Inference Steps K')
    plt.ylabel('Fashion-MNIST PSNR (dB)')
    plt.title('Zero-Shot Domain Transfer (MNIST -> Fashion)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_tr = Path('outputs/cleanup/transfer_plot.pdf')
    plt.savefig(out_tr, bbox_inches='tight')
    plt.savefig(out_tr.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved plot: {out_tr}")

if __name__ == '__main__':
    main()
