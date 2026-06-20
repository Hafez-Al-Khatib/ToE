"""
test_monotonicity.py
====================
Verifies the In-Distribution Inference Scaling Law.
Checks if PSNR increases monotonically with K on MNIST.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel as KANEBM
from exp_loss_landscape import MicroMLPEBM as MLPEBM
from metrics import compute_psnr

def refine(model, x_noisy, n_steps=10, dt=0.05):
    u = x_noisy.clone()
    for i in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            g = torch.autograd.grad(model.energy(ui).sum(), ui)[0].detach()
        u = (u.detach() - dt * (0.97**i) * g.clamp(-1, 1))
        u = torch.clamp(u, -1, 1)
    return u

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load MNIST-trained KAN-EBM
    kan_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
    model = KANEBM(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    if kan_path.exists():
        model.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
        print(f"Loaded KAN-EBM from {kan_path}")
    model.eval()

    # Load MNIST test set
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    test_ds = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    
    # Tiny subset for quick verification, small batch size to save memory
    loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False)
    
    sigma = 0.3
    K_values = [0, 1, 5, 10, 20, 50, 100]
    
    print(f"\nEvaluating In-Distribution Monotonicity: MNIST -> MNIST (sigma={sigma})")
    
    results = []
    for K in K_values:
        psnrs = []
        processed = 0
        for x, _ in loader:
            if processed >= 40: break # Just test 40 images
            x = x.to(device)
            xn = x + torch.randn_like(x) * sigma
            xn = torch.clamp(xn, -1, 1)
            
            with torch.no_grad():
                if K == 0:
                    xp = xn
                else:
                    xp = refine(model, xn, n_steps=K)
                
                psnrs.append(compute_psnr(xp, x))
            processed += x.shape[0]
            
        avg_psnr = np.mean(psnrs)
        results.append(avg_psnr)
        print(f"  K={K:>3} | PSNR: {avg_psnr:.4f} dB")

    # Check for monotonicity
    is_monotonic = all(x <= y + 0.01 for x, y in zip(results, results[1:]))
    print(f"\nMonotonicity Check: {'PASSED' if is_monotonic else 'FAILED'}")
    if not is_monotonic:
        print("Note: Minor fluctuations (<0.01 dB) are ignored.")

if __name__ == '__main__':
    run_test()
