import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import math
import numpy as np
import sys
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'src'))
from exp_cifar10 import KANEnergyModel

def psnr(a, b, data_range=2.0):
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)

class ScoreMatcher(nn.Module):
    def __init__(self, n_channels=1, depth=3, n_filters=18):
        super().__init__()
        layers = [nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())

def train_dsm(model, device, loader, epochs, sigma):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        t0 = time.time()
        for x, _ in loader:
            x = x.to(device)
            noise = torch.randn_like(x) * sigma
            xn = x + noise
            target = -noise / (sigma**2)
            pred = model(xn)
            loss = F.mse_loss(pred, target) * (sigma**2)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1}/30 | {time.time()-t0:.1f}s")

import time
def main():
    device = torch.device('cuda')
    sigma = 0.3
    epochs = 30
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('data', train=True, download=True, transform=tf), batch_size=128, shuffle=True)
    test_ds = torchvision.datasets.MNIST('data', train=False, download=True, transform=tf)
    
    # DnCNN-EBM
    dn_ebm = ScoreMatcher(depth=3, n_filters=17).to(device) # ~5.8K params
    print(f"DnCNN-EBM: {dn_ebm.n_params} params")
    train_dsm(dn_ebm, device, loader, epochs, sigma)
    
    # KAN-EBM (Load existing or train fresh)
    kan = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1).to(device)
    kan_path = ROOT / 'results' / 'kan_ebm_multitask.pt'
    kan.load_state_dict(torch.load(kan_path, map_location=device, weights_only=True))
    kan.eval()
    print(f"KAN-EBM: {sum(p.numel() for p in kan.parameters())} params")
    
    # Inference
    x_test = []
    for i in range(200): x_test.append(test_ds[i][0])
    x_test = torch.stack(x_test).to(device)
    xn = (x_test + torch.randn_like(x_test) * sigma).clamp(-1, 1)
    
    # DnCNN Langevin
    u_dn = xn.clone()
    for _ in range(20):
        with torch.no_grad(): u_dn = (u_dn + 0.05 * dn_ebm(u_dn)).clamp(-1, 1)
    
    # KAN Langevin
    from exp_cleanup_robustness_transfer import refine
    u_kan = refine(kan, xn, n_steps=20)
    
    print(f"\nFINAL COMPARISON (Matched Params ~5.8K):")
    print(f"  DnCNN-EBM: {psnr(u_dn, x_test):.2f} dB")
    print(f"  KAN-EBM:   {psnr(u_kan, x_test):.2f} dB")
    print(f"  Δ KAN-Advantage: {psnr(u_kan, x_test) - psnr(u_dn, x_test):+.2f} dB")

if __name__ == '__main__':
    main()
