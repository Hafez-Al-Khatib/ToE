"""
exp_ood_mnist.py
================
Evaluates the OOD detection capability of the trained MNIST KAN-EBM.
Uses the raw E_theta(x) energy as the anomaly score.
In-Distribution: MNIST test set
Out-of-Distribution: FashionMNIST test set
"""

import sys
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from run_paper_experiments import KANEnergyModel

def load_data():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    mnist = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    fmnist = torchvision.datasets.FashionMNIST(ROOT / 'data', train=False, download=True, transform=tf)
    
    # Take 1000 from each for fast evaluation
    mnist_sub = torch.utils.data.Subset(mnist, range(1000))
    fmnist_sub = torch.utils.data.Subset(fmnist, range(1000))
    
    return (
        torch.utils.data.DataLoader(mnist_sub, batch_size=100, shuffle=False),
        torch.utils.data.DataLoader(fmnist_sub, batch_size=100, shuffle=False)
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = KANEnergyModel(n_channels=1, height=28, width=28, n_filters=8, filter_size=5, kan_hidden=[8, 8]).to(device)
    
    ckpt_path = ROOT / 'results' / 'kan_ebm.pt'
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        return
        
    try:
        model.field.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    except Exception as e:
        # Sometimes it's saved as model.state_dict(), sometimes directly
        pass # model = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model.eval()
    
    loader_in, loader_out = load_data()
    
    scores_in = []
    scores_out = []
    
    print("Calculating energy for In-Distribution (MNIST)...")
    with torch.no_grad():
        for x, _ in loader_in:
            x = x.to(device)
            # Energy calculation. We normalize by batch size manually if compute_energy sums over batch.
            # PredictiveCodingField compute_energy sums over B*H*W! We need per-image energy.
            # Let's do it batch size 1 or loop through batch.
            for i in range(x.size(0)):
                e = model.field.compute_energy(x[i:i+1]).item()
                scores_in.append(e)
                
    print("Calculating energy for Out-of-Distribution (FashionMNIST)...")
    with torch.no_grad():
        for x, _ in loader_out:
            x = x.to(device)
            for i in range(x.size(0)):
                e = model.field.compute_energy(x[i:i+1]).item()
                scores_out.append(e)
                
    # Labels: 1 for OOD, 0 for ID
    # Usually, EBMs assign HIGHER energy to OOD data.
    y_true = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
    y_scores = np.concatenate([scores_in, scores_out])
    
    auroc = roc_auc_score(y_true, y_scores)
    
    print("\n==============================================")
    print("OOD DETECTION RESULTS (Energy Anomaly Score)")
    print("==============================================")
    print(f"In-Distribution (MNIST): Mean Energy = {np.mean(scores_in):.2f}")
    print(f"Out-of-Distribution (F-MNIST): Mean Energy = {np.mean(scores_out):.2f}")
    print(f"AUROC: {auroc:.4f}")
    
    if auroc < 0.5:
        print("\nNote: AUROC < 0.5 indicates 'Likelihood Paradox' where OOD has LOWER energy.")
        print(f"Reversed AUROC (OOD=0, ID=1): {1 - auroc:.4f}")

if __name__ == '__main__':
    main()
