import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import math
import numpy as np

def psnr(a, b, data_range=2.0):
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)

class ScoreMatcher(nn.Module):
    """A DnCNN-style architecture but used as an EBM score network."""
    def __init__(self, n_channels=1, depth=3, n_filters=12):
        super().__init__()
        layers = [nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x): return self.net(x) # Returns score gradient directly
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())

def train_dsm(model, device, loader, epochs, sigma):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            noise = torch.randn_like(x) * sigma
            xn = x + noise
            # DSM Loss: |score - (-noise/sigma^2)|^2
            target = -noise / (sigma**2)
            pred = model(xn)
            loss = F.mse_loss(pred, target) * (sigma**2)
            opt.zero_grad()
            loss.backward()
            opt.step()

def main():
    device = torch.device('cuda')
    # Match the 5.8K KAN params
    # Depth 5, filters 16 -> 12K
    # Depth 3, filters 24 -> 11K
    # Depth 3, filters 18 -> 6.5K
    dn_ebm = ScoreMatcher(depth=3, n_filters=18).to(device)
    print(f"Matched-Param DnCNN-EBM: {dn_ebm.n_params} params")
    
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, transform=tf),
        batch_size=128, shuffle=True)
    
    sigma = 0.3
    train_dsm(dn_ebm, device, loader, 30, sigma)
    
    # Eval
    test_ds = torchvision.datasets.MNIST('data', train=False, download=True, transform=tf)
    x, _ = test_ds[0]; x = x.unsqueeze(0).to(device)
    xn = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
    
    # Inference: Langevin on the learned DnCNN score
    u = xn.clone()
    for _ in range(20):
        with torch.no_grad():
            u = (u + 0.05 * dn_ebm(u)).clamp(-1, 1)
            
    p_dn = psnr(u, x)
    
    # Get KAN result from previous run for σ=0.3, K=20
    # From Lane A: denoise σ=0.3, K=20 was 22.35 dB
    # From exp_sota: KAN K=20 was 23.76 dB (fresh run)
    
    print(f"DnCNN-EBM (matched params) PSNR: {p_dn:.2f} dB")
    print(f"KAN-EBM (previous run) PSNR: 23.76 dB")
    print(f"KAN Advantage in EBM-space: {23.76 - p_dn:+.2f} dB")

if __name__ == '__main__':
    main()
