"""
exp_unet_ebm.py
===============
Small U-Net EBM ablation for CIFAR-10.

Tests whether the K* scaling law is specific to the conv-backbone + per-pixel
formulation, or generalizes to other local architectures (U-Net with skip
connections).

Architecture:
  - 3-level U-Net encoder-decoder
  - Skip connections at each level
  - Final 1x1 conv outputs scalar energy per pixel
  - ~35K parameters (matched to KAN-EBM-small)

Training:
  - Same DSM objective as KAN-EBM
  - Same hyperparameters (200 epochs, Adam, lr=1e-3)

Evaluation:
  - Fine-grid K sweep (K=1..30) on 100 test images
  - Power-law fit K*(sigma) ~ C sigma^alpha
  - Compare alpha, R^2 to KAN-EBM and ConvMLP

Run:
  py -3.12 experiments/exp_unet_ebm.py --device cuda --train
  py -3.12 experiments/exp_unet_ebm.py --device cuda --eval --n_images 100
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

SEED = 42
OUT_DIR = ROOT / 'outputs' / 'unet_ebm'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Small U-Net EBM
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.conv(x)


class UNetEBM(nn.Module):
    """Small U-Net energy function for CIFAR-10."""
    def __init__(self, base_ch=16, n_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(n_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = ConvBlock(base_ch * 4, base_ch * 4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Output: scalar energy per pixel
        self.out = nn.Conv2d(base_ch, 1, 1)

    def energy(self, x):
        """Scalar energy (sum over spatial locations)."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Bottleneck
        b = self.bot(self.pool(e3))
        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out(d1)
        return out.sum()

    def energy_grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.energy(xi), xi)[0].detach().clamp(-1., 1.)

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        u = x_noisy.clone(); step = dt
        for _ in range(n_steps):
            u = (u - step * self.energy_grad(u)).detach()
            step *= dt_decay
        return u

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi = xn.detach().requires_grad_(True)
            g = torch.autograd.grad(self.energy(xi), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma ** 2 * g, x_clean)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_unet(device, epochs=100, batch_size=128, lr=1e-3):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = UNetEBM(base_ch=8).to(device)
    print(f"[model] U-Net EBM: {model.n_params:,} params")

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    sigmas = [0.05, 0.1, 0.15, 0.2, 0.3]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            sigma = np.random.choice(sigmas)
            loss = model.loss(xb, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.4f}")
        
        if (epoch + 1) % 25 == 0:
            ckpt_path = OUT_DIR / f'unet_ebm_epoch{epoch+1}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [save] Checkpoint saved to {ckpt_path}")

    # Save checkpoint
    ckpt_path = OUT_DIR / 'unet_ebm.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"[save] Checkpoint saved to {ckpt_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Fine-grid evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_fine_grid(model, images_clean, sigma, K_max=30, dt=0.05, dt_decay=0.97):
    noisy = (images_clean + torch.randn_like(images_clean) * sigma).clamp(-1, 1)
    per_step_psnr = []
    u = noisy.clone()
    step = dt
    for k in range(K_max):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            g = model.energy_grad(ui)
        u = (u - step * g).detach()
        mse = F.mse_loss(u.clamp(-1, 1), images_clean, reduction='none').mean(dim=[1, 2, 3])
        psnr = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
        per_step_psnr.append(psnr)
    return np.stack(per_step_psnr, axis=0)


def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_images', type=int, default=100)
    parser.add_argument('--K_max', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.05, 0.1, 0.15, 0.2, 0.3])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")

    if args.train:
        print("[mode] Training U-Net EBM...")
        train_unet(device)

    if args.eval:
        ckpt = OUT_DIR / 'unet_ebm.pt'
        if not ckpt.exists():
            print(f"[error] Checkpoint not found: {ckpt}")
            print("        Run with --train first.")
            return

        model = UNetEBM(base_ch=8).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        print(f"[model] Loaded U-Net EBM ({model.n_params:,} params)")

        tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
        images_all = torch.stack([test_ds[i][0] for i in range(min(args.n_images, len(test_ds)))])
        print(f"[data] Loaded {len(images_all)} test images")

        sigmas = args.sigmas
        K_max = args.K_max
        n_seeds = 3
        all_results = {}

        for seed_idx in range(n_seeds):
            torch.manual_seed(SEED + seed_idx * 100)
            np.random.seed(SEED + seed_idx * 100)
            print(f"\n=== Seed {seed_idx} ===")
            all_results[seed_idx] = {}

            for sigma in sigmas:
                print(f"  sigma={sigma:.2f} ...", end='', flush=True)
                t0 = time.time()
                per_step_psnr = []
                for b in range(0, len(images_all), args.batch_size):
                    batch = images_all[b:b + args.batch_size].to(device)
                    psnr = evaluate_fine_grid(model, batch, sigma, K_max)
                    per_step_psnr.append(psnr)
                per_step_psnr = np.concatenate(per_step_psnr, axis=1)
                all_results[seed_idx][sigma] = per_step_psnr
                kstar_mean = int(np.argmax(per_step_psnr.mean(axis=1))) + 1
                print(f" done in {time.time()-t0:.1f}s | K*={kstar_mean}")

        # Aggregate
        summary = {'per_sigma': {}}
        for sigma in sigmas:
            stacked = np.stack([all_results[s][sigma] for s in range(n_seeds)], axis=0)
            mean_psnr = stacked.mean(axis=0)
            kstar_per_image = np.argmax(mean_psnr, axis=0) + 1
            kstar_mean = float(np.mean(kstar_per_image))
            kstar_std = float(np.std(kstar_per_image, ddof=1))
            print(f"\nsigma={sigma:.2f}: K*={kstar_mean:.2f} ± {kstar_std:.2f}")
            summary['per_sigma'][f'{sigma:.3f}'] = {
                'kstar_mean': kstar_mean,
                'kstar_std': kstar_std,
                'mean_psnr_per_K': mean_psnr.mean(axis=1).tolist(),
            }

        # Power-law fit
        K_vals = [summary['per_sigma'][f'{s:.3f}']['kstar_mean'] for s in sigmas]
        a, alpha, r2 = fit_power_law(sigmas, K_vals)
        C = math.exp(a)
        print(f"\nPower-law fit: alpha={alpha:.3f}, C={C:.1f}, R^2={r2:.3f}")
        summary['power_law'] = {'alpha': alpha, 'C': C, 'R2': r2}

        out_path = OUT_DIR / 'unet_evaluation.json'
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[save] Results saved to {out_path}")


if __name__ == '__main__':
    main()
