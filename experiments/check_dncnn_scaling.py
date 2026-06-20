import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import math
import numpy as np
import time

def psnr(a, b, data_range=2.0):
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)

class ScoreMatcher(nn.Module):
    def __init__(self, n_channels=1, depth=3, n_filters=17):
        super().__init__()
        layers = [nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def main():
    device = torch.device('cuda')
    sigma = 0.3
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('data', train=True, download=True, transform=tf), batch_size=128, shuffle=True)
    test_ds = torchvision.datasets.MNIST('data', train=False, download=True, transform=tf)
    
    model = ScoreMatcher(depth=3, n_filters=17).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Training DnCNN-EBM (30 epochs)...")
    for ep in range(30):
        for x, _ in loader:
            x = x.to(device)
            noise = torch.randn_like(x) * sigma
            xn = x + noise
            target = -noise / (sigma**2)
            pred = model(xn)
            loss = F.mse_loss(pred, target) * (sigma**2)
            opt.zero_grad(); loss.backward(); opt.step()
    
    model.eval()
    x_test = torch.stack([test_ds[i][0] for i in range(200)]).to(device)
    xn = (x_test + torch.randn_like(x_test) * sigma).clamp(-1, 1)
    
    K_LIST = [0, 1, 2, 5, 10, 20, 50]
    results = {}
    
    u = xn.clone()
    results[0] = psnr(u, x_test)
    
    # Langevin with small step to be fair
    dt = 0.05
    for k in range(1, 51):
        with torch.no_grad():
            u = (u + dt * model(u)).clamp(-1, 1)
        if k in K_LIST:
            results[k] = psnr(u, x_test)
    
    print("\nDnCNN-EBM K-SCALING CURVE:")
    print("K   | PSNR (dB)")
    print("----|----------")
    for k in K_LIST:
        print(f"{k:>2}  | {results[k]:.2f}")

if __name__ == '__main__':
    main()
