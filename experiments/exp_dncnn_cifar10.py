"""
exp_dncnn_cifar10.py
====================
Train a tiny DnCNN baseline on CIFAR-10 for denoising.

This provides a modern CNN denoiser at matched parameter count (~30-40K)
to compare against KAN-EBM.

Architecture: Micro-DnCNN with residual learning.
  - 5 convolutional layers, 16 filters each
  - BatchNorm + ReLU
  - Residual: output = input - network(input)
  - ~35K parameters

Run: python experiments/exp_dncnn_cifar10.py --epochs 20 --device cpu
"""
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

from metrics import compute_ssim


class MicroDnCNN(nn.Module):
    """Tiny DnCNN with residual learning."""
    def __init__(self, n_channels=3, depth=5, n_filters=16):
        super().__init__()
        layers = [nn.Conv2d(n_channels, n_filters, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(n_filters, n_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


def psnr(a, b, data_range=2.0):
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)


def train(model, device, loader, epochs, sigma, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        losses = []
        for x, _ in loader:
            x = x.to(device)
            xn = x + torch.randn_like(x) * sigma
            pred = model(xn)
            loss = F.mse_loss(pred, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={np.mean(losses):.5f}")
    return model


def evaluate(model, device, test_loader, sigmas, K_list):
    """FFN has no K-scaling; evaluate at single pass."""
    model.eval()
    results = {}
    for sigma in sigmas:
        psnrs = []
        ssims = []
        for x, _ in test_loader:
            x = x.to(device)
            xn = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
            with torch.no_grad():
                pred = model(xn).clamp(-1, 1)
            mse = F.mse_loss(pred, x, reduction='none').mean(dim=[1,2,3])
            psnr_batch = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
            psnrs.extend(psnr_batch.tolist())
            for i in range(pred.shape[0]):
                s = compute_ssim(x[i:i+1], pred[i:i+1], data_range=2.0)
                ssims.append(s)
        results[sigma] = {
            'psnr_mean': float(np.mean(psnrs)),
            'psnr_std': float(np.std(psnrs)),
            'ssim_mean': float(np.mean(ssims)),
            'ssim_std': float(np.std(ssims)),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--sigma_train', type=float, default=0.15)
    parser.add_argument('--n_filters', type=int, default=16)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_eval', type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=0)

    # Model
    model = MicroDnCNN(n_channels=3, depth=args.depth, n_filters=args.n_filters).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Micro-DnCNN: {n_params:,} params | depth={args.depth} filters={args.n_filters}")

    # Train
    print(f"Training for {args.epochs} epochs at sigma={args.sigma_train}...")
    t0 = time.time()
    model = train(model, device, train_loader, args.epochs, args.sigma_train, lr=args.lr)
    train_time = time.time() - t0
    print(f"Training complete in {train_time/60:.1f} minutes")

    # Evaluate
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.3]
    print(f"Evaluating on {args.n_eval} test images...")
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, range(args.n_eval)),
        batch_size=100, shuffle=False)
    results = evaluate(model, device, eval_loader, sigmas, K_list=[1])

    # Print summary
    print("\n" + "="*60)
    print("Micro-DnCNN Results")
    print("="*60)
    for sigma in sigmas:
        r = results[sigma]
        print(f"  sigma={sigma:.2f}: PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f} dB  "
              f"SSIM={r['ssim_mean']:.3f}±{r['ssim_std']:.3f}")

    # Save
    out_dir = ROOT / 'outputs' / 'dncnn_baseline'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'dncnn_cifar10_results.json'
    out_path.write_text(json.dumps({
        'n_params': n_params,
        'train_time_s': train_time,
        'epochs': args.epochs,
        'sigma_train': args.sigma_train,
        'results': results,
    }, indent=2))
    print(f"Saved to {out_path}")

    # Save checkpoint
    ckpt_path = out_dir / 'dncnn_cifar10.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()
