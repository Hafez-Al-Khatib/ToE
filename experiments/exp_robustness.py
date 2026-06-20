"""
exp_robustness.py
=================
Quantifies the "Smoothness Proof" via Adversarial Robustness.
Compares KAN-EBM vs MLP-EBM under FGSM attacks.

The Hypothesis:
1. KAN-EBM (B-splines) has a smoother energy landscape than MLP-EBM (ReLU).
2. Consequently, KAN-EBM is more resilient to adversarial perturbations.
3. Langevin refinement (test-time compute) can "pull back" adversarial 
   perturbations into the energy basin more effectively in KANs than in MLPs.

Metrics:
- PSNR vs. Epsilon (Attack Strength)
- Recovery PSNR (after K steps of Langevin refinement)
"""

import sys
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions (Standardized for robustness testing)
# ─────────────────────────────────────────────────────────────────────────────

from exp_steel_man import SmoothMLPEBM as MLPEBM
from exp_cifar10 import KANEnergyModel as KANEBM

def load_models(device):
    kan_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
    
    # KAN-EBM (~5.8K params)
    kan = KANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    if kan_path.exists():
        kan.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
        print(f"Loaded KAN-EBM from {kan_path}")

    # Smooth MLP (Steel-Man Baseline, SiLU)
    mlp = MLPEBM(n_channels=1, n_filters=16, filter_size=5, hidden_dim=128).to(device)
    # Train it quickly to ensure it's a valid prior
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    
    print("Pre-training Steel-Man MLP baseline (SiLU) for robustness test...")
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
    mlp.train()
    for _ in range(20):
        for x, _ in loader:
            x = x.to(device)
            loss = mlp.loss(x, 0.3)
            opt.zero_grad(); loss.backward(); opt.step()
            
    kan.eval()
    mlp.eval()
    return kan, mlp

# ─────────────────────────────────────────────────────────────────────────────
# FGSM Attack Implementation
# ─────────────────────────────────────────────────────────────────────────────

def fgsm_attack(image, epsilon, data_grad):
    # Collect the elements of the gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [-1,1] range (MNIST-style normalization)
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def get_adversarial_examples(model, x, epsilon, device):
    """
    Generate adversarial examples using FGSM.
    Since we don't have a 'label' for energy, we maximize the energy 
    to move away from the clean data manifold.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    energy = model.energy(x_adv).sum()
    model.zero_grad()
    energy.backward()
    data_grad = x_adv.grad.data
    return fgsm_attack(x, epsilon, data_grad)

# ─────────────────────────────────────────────────────────────────────────────
# Refinement (Langevin)
# ─────────────────────────────────────────────────────────────────────────────

def refine(model, x_noisy, n_steps=10, dt=0.05):
    u = x_noisy.clone()
    for i in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            # Minimize energy
            g = torch.autograd.grad(model.energy(ui).sum(), ui)[0].detach()
        u = (u.detach() - dt * (0.97**i) * g.clamp(-1, 1))
        u = torch.clamp(u, -1, 1)
    return u

# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness_test(quick=False, device_str='auto'):
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
             if device_str == 'auto' else torch.device(device_str)
    
    kan, mlp = load_models(device)
    
    # Load MNIST test images
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    test_ds = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    
    n_test = 50 if quick else 200
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.3] if not quick else [0, 0.1, 0.3]
    K_steps = [10, 20] if not quick else [10]
    
    results = {
        'epsilons': epsilons,
        'kan': {'attacked': [], 'refined': {k: [] for k in K_steps}},
        'mlp': {'attacked': [], 'refined': {k: [] for k in K_steps}}
    }

    from metrics import compute_psnr
    
    for eps in epsilons:
        print(f"Testing epsilon: {eps}")
        kan_psnr_batch = []
        mlp_psnr_batch = []
        
        kan_refine_batches = {k: [] for k in K_steps}
        mlp_refine_batches = {k: [] for k in K_steps}
        
        processed = 0
        for x, _ in test_loader:
            if processed >= n_test: break
            x = x.to(device)
            
            # 1. Attack
            x_adv_kan = get_adversarial_examples(kan, x, eps, device)
            x_adv_mlp = get_adversarial_examples(mlp, x, eps, device)
            
            kan_psnr_batch.append(compute_psnr(x_adv_kan, x))
            mlp_psnr_batch.append(compute_psnr(x_adv_mlp, x))
            
            # 2. Refine
            for k in K_steps:
                x_ref_kan = refine(kan, x_adv_kan, n_steps=k)
                x_ref_mlp = refine(mlp, x_adv_mlp, n_steps=k)
                
                kan_refine_batches[k].append(compute_psnr(x_ref_kan, x))
                mlp_refine_batches[k].append(compute_psnr(x_ref_mlp, x))
            
            processed += x.shape[0]

        results['kan']['attacked'].append(np.mean(kan_psnr_batch))
        results['mlp']['attacked'].append(np.mean(mlp_psnr_batch))
        
        for k in K_steps:
            results['kan']['refined'][k].append(np.mean(kan_refine_batches[k]))
            results['mlp']['refined'][k].append(np.mean(mlp_refine_batches[k]))

    # ── Saving and Plotting ──────────────────────────────────────────────────
    out_dir = ROOT / 'outputs' / 'robustness'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, results['kan']['attacked'], 'b--', label='KAN-EBM (Attacked)')
    plt.plot(epsilons, results['mlp']['attacked'], 'r--', label='MLP-EBM (Attacked)')
    
    for k in K_steps:
        plt.plot(epsilons, results['kan']['refined'][k], 'b-', label=f'KAN-EBM (K={k} Refined)')
        plt.plot(epsilons, results['mlp']['refined'][k], 'r-', label=f'MLP-EBM (K={k} Refined)')
    
    plt.xlabel('Epsilon (Attack Strength)')
    plt.ylabel('PSNR (dB)')
    plt.title('Adversarial Robustness: KAN-EBM vs MLP-EBM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'robustness_plot.png', dpi=300)
    plt.close()

    print(f"Results saved to {out_dir}")
    
    # Conclusion text for the paper
    print("\nEmpirical Robustness Conclusion:")
    kan_gain = np.mean(np.array(results['kan']['refined'][K_steps[-1]]) - np.array(results['kan']['attacked']))
    mlp_gain = np.mean(np.array(results['mlp']['refined'][K_steps[-1]]) - np.array(results['mlp']['attacked']))
    
    print(f"KAN-EBM Average Refinement Gain: {kan_gain:.2f} dB")
    print(f"MLP-EBM Average Refinement Gain: {mlp_gain:.2f} dB")
    if kan_gain > mlp_gain:
        print(f"RESULT: KAN-EBM is {kan_gain - mlp_gain:.2f} dB more resilient to adversarial attacks.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    run_robustness_test(quick=args.quick, device_str=args.device)
