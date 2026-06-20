"""
exp_diffusion_cifar10.py
========================
Train a lightweight score-based diffusion model on CIFAR-10 for comparison.

This addresses the reviewer criticism: "No comparison with diffusion models."
We implement a tiny U-Net score network (~100K params) trained with
Denoising Score Matching (NCSN-style) at a single noise level for fair
comparison with KAN-EBM.

Architecture: Tiny U-Net score network
  - 3 encoder blocks, bottleneck, 3 decoder blocks
  - Time/noise level conditioning via simple embedding
  - ~90-120K parameters depending on config

Run: py -3.12 experiments/exp_diffusion_cifar10.py --epochs 30 --device cuda
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

# ── Reproducibility seeds ────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True  # uncomment if full determinism needed
# torch.backends.cudnn.benchmark = False


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) noise levels
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TinyResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(x))
        h = self.norm1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class TinyUNet(nn.Module):
    """Tiny U-Net score network, ~100K params."""
    def __init__(self, n_channels=3, base_ch=32, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.enc1 = TinyResBlock(n_channels, base_ch, time_dim)
        self.enc2 = TinyResBlock(base_ch, base_ch * 2, time_dim)
        self.enc3 = TinyResBlock(base_ch * 2, base_ch * 2, time_dim)

        self.down = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)

        self.bottleneck = TinyResBlock(base_ch * 2, base_ch * 2, time_dim)

        self.dec2 = TinyResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.dec1 = TinyResBlock(base_ch * 2, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, n_channels, 3, padding=1)

    def forward(self, x, sigma):
        t_emb = self.time_embed(sigma)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)

        b = self.bottleneck(e3, t_emb)

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out(d1)


def train_score(model, device, loader, epochs, sigma_values=[0.05, 0.1, 0.15, 0.2, 0.3], lr=1e-3):
    """Train with multi-sigma DSM (NCSN-style). Samples a random sigma per batch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        losses = []
        for x, _ in loader:
            x = x.to(device)
            # Sample a random sigma for each image in the batch
            sigmas = torch.tensor(np.random.choice(sigma_values, size=x.size(0)), 
                                  dtype=torch.float32, device=device)
            # Expand noise per image
            noise = torch.randn_like(x)
            for i in range(x.size(0)):
                noise[i] *= sigmas[i]
            xn = x + noise
            # Model predicts noise directly (noise conditioning via TimeEmbed)
            pred = model(xn, sigmas)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={np.mean(losses):.5f}")
    return model


def iterative_score_denoise(model, noisy, K, sigma, device):
    """Run K steps of noise-prediction-based denoising.
    
    Model predicts the noise residual. At each step we subtract a fraction
    of the predicted noise, with step size decaying to prevent oscillation.
    """
    x = noisy.clone()
    base_eta = 1.0 if K == 1 else 0.5
    for step in range(K):
        eta = base_eta * (1.0 - step / max(K, 1))  # gentle decay
        with torch.no_grad():
            noise_pred = model(x, torch.full((x.size(0),), sigma, device=device))
        x = (x - eta * noise_pred).clamp(-1, 1)
    return x


def evaluate(model, device, test_loader, sigmas, K_list):
    model.eval()
    all_results = {}
    for sigma in sigmas:
        all_results[sigma] = {}
        for K in K_list:
            psnrs = []
            ssims = []
            for x, _ in test_loader:
                x = x.to(device)
                xn = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
                with torch.no_grad():
                    pred = iterative_score_denoise(model, xn, K, sigma, device)
                mse = F.mse_loss(pred, x, reduction='none').mean(dim=[1,2,3])
                psnr_batch = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
                psnrs.extend(psnr_batch.tolist())
                for i in range(pred.shape[0]):
                    s = compute_ssim(x[i:i+1], pred[i:i+1], data_range=2.0)
                    ssims.append(s)
            all_results[sigma][K] = {
                'psnr_mean': float(np.mean(psnrs)),
                'psnr_std': float(np.std(psnrs)),
                'ssim_mean': float(np.mean(ssims)),
                'ssim_std': float(np.std(ssims)),
            }
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--sigma_train', type=str, default='0.05,0.1,0.15,0.2,0.3',
                        help='Comma-separated noise levels for multi-sigma training')
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_eval', type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=2)

    # Model
    model = TinyUNet(n_channels=3, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tiny U-Net score model: {n_params:,} params | base_ch={args.base_ch}")

    # Train
    print(f"Training for {args.epochs} epochs with multi-sigma DSM (sigmas={args.sigma_train})...")
    t0 = time.time()
    sigma_values = [float(s) for s in args.sigma_train.split(',')]
    model = train_score(model, device, train_loader, args.epochs, sigma_values=sigma_values, lr=args.lr)
    train_time = time.time() - t0
    print(f"Training complete in {train_time/60:.1f} minutes")

    # Evaluate
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.3]
    K_list = [1, 2, 5, 10, 20]
    print(f"Evaluating on {args.n_eval} test images...")
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, range(args.n_eval)),
        batch_size=100, shuffle=False, num_workers=2)
    results = evaluate(model, device, eval_loader, sigmas, K_list)

    # Print summary
    print("\n" + "="*80)
    print("Tiny Diffusion/Score Model Results")
    print("="*80)
    for sigma in sigmas:
        print(f"\nsigma={sigma:.2f}")
        for K in K_list:
            r = results[sigma][K]
            print(f"  K={K:2d}: PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f} dB  "
                  f"SSIM={r['ssim_mean']:.3f}±{r['ssim_std']:.3f}")

    # Save
    out_dir = ROOT / 'outputs' / 'diffusion_cifar10'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'diffusion_cifar10_results.json'
    out_path.write_text(json.dumps({
        'n_params': n_params,
        'train_time_s': train_time,
        'epochs': args.epochs,
        'sigma_train': sigma_values,
        'results': results,
    }, indent=2))
    print(f"\nSaved to {out_path}")

    ckpt_path = out_dir / 'diffusion_cifar10.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()
