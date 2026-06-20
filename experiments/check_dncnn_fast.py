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
    def __init__(self, n_channels=1, depth=3, n_filters=17):
        super().__init__()
        layers = [nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

device = torch.device('cuda')
sigma = 0.3
tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
train_ds = torchvision.datasets.MNIST('data', train=True, download=True, transform=tf)
# Subset for faster training
train_ds = torch.utils.data.Subset(train_ds, range(10000))
loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_ds = torchvision.datasets.MNIST('data', train=False, download=True, transform=tf)

model = ScoreMatcher(depth=3, n_filters=17).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for ep in range(15):
    for x, _ in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * sigma
        xn = x + noise
        target = -noise / (sigma**2)
        loss = F.mse_loss(model(xn), target) * (sigma**2)
        opt.zero_grad(); loss.backward(); opt.step()

model.eval()
x_test = torch.stack([test_ds[i][0] for i in range(100)]).to(device)
xn = (x_test + torch.randn_like(x_test) * sigma).clamp(-1, 1)

K_LIST = [0, 1, 5, 10, 20, 50]
u = xn.clone()
print("K | PSNR")
for k in range(51):
    if k in K_LIST: print(f"{k} | {psnr(u, x_test):.2f}")
    with torch.no_grad(): u = (u + 0.05 * model(u)).clamp(-1, 1)
