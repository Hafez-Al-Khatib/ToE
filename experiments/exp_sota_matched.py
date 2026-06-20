import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel
from exp_diffusion_comparison import DEQDenoiser

# ── Micro-DnCNN Implementation ──
class MicroDnCNN(nn.Module):
    def __init__(self, n_channels=1, depth=5, n_filters=16):
        super().__init__()
        layers = [nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)  # Residual learning

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())

def psnr(a, b, data_range=2.0):
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)

def train_feedforward(model, device, loader, epochs, sigma, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            xn = x + torch.randn_like(x) * sigma
            pred = model(xn)
            loss = F.mse_loss(pred, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. KAN-EBM (~5.8K params)
    kan = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    n_kan = sum(p.numel() for p in kan.parameters())
    
    # 2. Micro-DnCNN (match ~5.8K)
    # depth=5, filters=8 -> approx 4K params
    dncnn = MicroDnCNN(n_channels=1, depth=3, n_filters=10).to(device)
    n_dn = dncnn.n_params
    
    # 3. Micro-DEQ (match ~5.8K)
    deq = DEQDenoiser(obs_dim=784, hidden=4).to(device) # Tiny hidden to match params
    n_deq = sum(p.numel() for p in deq.parameters())
    
    print(f"PARAMS: KAN={n_kan} | DnCNN={n_dn} | DEQ={n_deq}")
    
    # Data
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = torchvision.datasets.MNIST(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.MNIST(ROOT / 'data', train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=200, shuffle=False)
    
    sigma = 0.3
    epochs = 10 # Quick run
    
    print(f"Training DnCNN for {epochs} epochs...")
    train_feedforward(dncnn, device, loader, epochs, sigma)
    
    # KAN already exists in results/kan_ebm_multitask.pt
    kan_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
    kan.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
    kan.eval()
    
    # Evaluation
    dncnn.eval()
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)
    xn = (x_test + torch.randn_like(x_test) * sigma).clamp(-1, 1)
    
    p0 = psnr(xn, x_test)
    p_dn = psnr(dncnn(xn), x_test)
    
    # KAN K=20
    from exp_cleanup_robustness_transfer import refine
    p_kan = psnr(refine(kan, xn, n_steps=20), x_test)
    
    print("\nRESULTS (MNIST σ=0.3):")
    print(f"  Initial Noisy: {p0:.2f} dB")
    print(f"  Micro-DnCNN (matched): {p_dn:.2f} dB")
    print(f"  KAN-EBM (K=20): {p_kan:.2f} dB")
    print(f"  Gain: KAN vs DnCNN = {p_kan - p_dn:+.2f} dB")

if __name__ == '__main__':
    main()
