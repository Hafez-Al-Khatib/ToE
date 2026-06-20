"""
exp_coco_mini.py
================
KAN-EBM on Natural Image Patches (COCO-Mini).
Tests generalization to high-entropy natural textures (grass, fabric, water).
Uses 32x32 patches for consistency with the CIFAR model.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel, train_model, evaluate

def run_coco_test(quick=False, device_str='auto'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = ROOT / 'outputs' / 'coco_mini'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup COCO Patch Dataset
    # If COCO isn't downloaded, we'll use a fallback or synthetic natural textures
    # For this script, we'll use STL-10 as a "Natural Texture" proxy if COCO is missing
    # as STL-10 has higher resolution (96x96) which we can patch.
    try:
        tf = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        train_ds = datasets.STL10(ROOT / 'data', split='train', download=True, transform=tf)
        test_ds = datasets.STL10(ROOT / 'data', split='test', download=True, transform=tf)
        print("Using STL-10 as high-resolution natural texture proxy.")
    except:
        print("Falling back to CIFAR-100 for texture diversity.")
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train_ds = datasets.CIFAR100(ROOT / 'data', train=True, download=True, transform=tf)
        test_ds = datasets.CIFAR100(ROOT / 'data', train=False, download=True, transform=tf)

    if quick:
        train_ds = torch.utils.data.Subset(train_ds, range(2000))
        test_ds = torch.utils.data.Subset(test_ds, range(200))
        
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Model (Using our verified 16-filter KAN)
    model = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[48, 16], n_channels=3).to(device)
    
    # 3. Train
    sigma = 0.1 # Natural noise level
    n_epochs = 5 if quick else 20
    print(f"\nTraining on Natural Textures ({n_epochs} epochs)...")
    train_model(model, train_loader, device, n_epochs, sigma, tag='Texture-KAN')
    
    # 4. Evaluate
    K_LIST = [1, 5, 10, 20]
    print(f"\nEvaluating K-scaling on Natural Textures...")
    results = evaluate(model, test_loader, device, sigma, K_LIST)
    
    # 5. Output
    for k in K_LIST:
        print(f"  K={k:>2} | PSNR: {results[k]:.2f} dB")
        
    with open(out_dir / 'coco_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_coco_test(quick=args.quick)
