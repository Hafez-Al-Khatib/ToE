"""
exp_scaled_eval_cifar10.py
==========================
Scaled evaluation of existing CIFAR-10 checkpoints on 500 test images.

Computes PSNR and SSIM for KAN-EBM, FFN-DSM, and ConvMLP baselines
at multiple noise levels and inference steps.

Run: python experiments/exp_scaled_eval_cifar10.py --n_images 500 --device cpu
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

# Import model definitions
from exp_cifar10 import KANEnergyModel, MLPEnergyModel, FFNDenoiser
from rebuttal_smooth_mlp import ConvSmoothMLPEBM


def load_kan(checkpoint_path, device, n_filters=32, kan_hidden=None):
    if kan_hidden is None:
        kan_hidden = [96, 16]
    model = KANEnergyModel(n_filters=n_filters, kan_hidden=kan_hidden).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_conv_mlp(checkpoint_path, device, activation='gelu'):
    model = ConvSmoothMLPEBM(n_filters=16, mlp_hidden=160, activation=activation).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_ffn(checkpoint_path, device, hidden=512):
    model = FFNDenoiser(hidden=hidden).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def iterative_denoise(model, noisy, K, eta0=0.05, delta=0.97):
    """Gradient descent denoising matching the original exp_cifar10 protocol."""
    # Use built-in denoise if available (KAN, MLP)
    if hasattr(model, 'denoise'):
        return model.denoise(noisy, n_steps=K, dt=eta0, dt_decay=delta)
    # Fallback for ConvMLP: match the KAN-EBM protocol
    x = noisy.clone()
    step = eta0
    for _ in range(K):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            grad = torch.autograd.grad(model.energy(xi), xi)[0].detach().clamp(-1., 1.)
        x = (x - step * grad).detach()
        step *= delta
    return x


def evaluate_ebm(model, images_clean, sigma, K_list, device):
    """Evaluate an EBM model (KAN or ConvMLP) at multiple K values."""
    noisy = (images_clean + torch.randn_like(images_clean) * sigma).clamp(-1, 1)
    results = {}
    for K in K_list:
        denoised = iterative_denoise(model, noisy, K)
        mse = F.mse_loss(denoised, images_clean, reduction='none').mean(dim=[1, 2, 3])
        psnr = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
        ssim_vals = []
        for i in range(denoised.shape[0]):
            s = compute_ssim(images_clean[i:i+1], denoised[i:i+1], data_range=2.0)
            ssim_vals.append(s)
        results[K] = {
            'psnr_mean': float(np.mean(psnr)),
            'psnr_std': float(np.std(psnr)),
            'ssim_mean': float(np.mean(ssim_vals)),
            'ssim_std': float(np.std(ssim_vals)),
        }
    return results


def evaluate_ffn(model, images_clean, sigma, device):
    """Evaluate feedforward denoiser (single K)."""
    noisy = (images_clean + torch.randn_like(images_clean) * sigma).clamp(-1, 1)
    with torch.no_grad():
        denoised = model(noisy).clamp(-1, 1)
    mse = F.mse_loss(denoised, images_clean, reduction='none').mean(dim=[1, 2, 3])
    psnr = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
    ssim_vals = []
    for i in range(denoised.shape[0]):
        s = compute_ssim(images_clean[i:i+1], denoised[i:i+1], data_range=2.0)
        ssim_vals.append(s)
    return {
        'psnr_mean': float(np.mean(psnr)),
        'psnr_std': float(np.std(psnr)),
        'ssim_mean': float(np.mean(ssim_vals)),
        'ssim_std': float(np.std(ssim_vals)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument('--K_list', type=int, nargs='+', default=[1, 2, 5, 10, 20, 50])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | Evaluating on {args.n_images} images")

    # Load CIFAR-10 test set
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)

    # Select first n_images
    images = []
    for i in range(min(args.n_images, len(test_ds))):
        img, _ = test_ds[i]
        images.append(img)
    images = torch.stack(images).to(device)

    # Load models
    models = {}
    models['KAN-EBM-110K'] = ('ebm', load_kan(ROOT / 'outputs/cifar10/kan_ebm_f32.pt', device, n_filters=32, kan_hidden=[96, 16]))
    models['KAN-EBM-32K'] = ('ebm', load_kan(ROOT / 'outputs/finalization/kstar_param_scaling/kan_small.pt', device, n_filters=16, kan_hidden=[48, 16]))
    models['ConvMLP-GELU'] = ('ebm', load_conv_mlp(ROOT / 'outputs/finalization/rebuttal/conv_mlp_gelu.pt', device, 'gelu'))
    models['FFN-DSM'] = ('ffn', load_ffn(ROOT / 'outputs/cifar10/ffn_dsm_f32.pt', device, hidden=512))

    all_results = {}
    total_start = time.time()

    for sigma in args.sigmas:
        print(f"\n=== sigma={sigma} ===")
        all_results[sigma] = {}
        for model_name, (mtype, model) in models.items():
            print(f"  {model_name} ...", end='', flush=True)
            t0 = time.time()

            # Process in batches
            batch_results = []
            for b in range(0, len(images), args.batch_size):
                batch = images[b:b + args.batch_size]
                if mtype == 'ebm':
                    res = evaluate_ebm(model, batch, sigma, args.K_list, device)
                else:
                    res = {K: evaluate_ffn(model, batch, sigma, device) for K in args.K_list}
                batch_results.append(res)

            # Aggregate across batches
            agg = {}
            for K in args.K_list:
                psnr_vals = [r[K]['psnr_mean'] for r in batch_results]
                ssim_vals = [r[K]['ssim_mean'] for r in batch_results]
                agg[K] = {
                    'psnr': float(np.mean(psnr_vals)),
                    'psnr_std': float(np.mean([r[K]['psnr_std'] for r in batch_results])),
                    'ssim': float(np.mean(ssim_vals)),
                    'ssim_std': float(np.mean([r[K]['ssim_std'] for r in batch_results])),
                }

            all_results[sigma][model_name] = agg
            elapsed = time.time() - t0
            print(f" done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal evaluation time: {total_elapsed / 60:.1f} minutes")

    # Save results
    out_dir = ROOT / 'outputs' / 'scaled_eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'cifar10_scaled_{args.n_images}.json'
    out_path.write_text(json.dumps({
        'n_images': args.n_images,
        'sigmas': args.sigmas,
        'K_list': args.K_list,
        'total_time_s': total_elapsed,
        'results': all_results,
    }, indent=2))
    print(f"Saved results to {out_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("PSNR (dB) summary")
    print("=" * 100)
    for sigma in args.sigmas:
        print(f"\nsigma={sigma}")
        for model_name in models:
            row = f"  {model_name:18s}:"
            for K in args.K_list:
                if K in all_results[sigma][model_name]:
                    row += f"  K={K:2d}: {all_results[sigma][model_name][K]['psnr']:6.2f}"
            print(row)

    print("\n" + "=" * 100)
    print("SSIM summary")
    print("=" * 100)
    for sigma in args.sigmas:
        print(f"\nsigma={sigma}")
        for model_name in models:
            row = f"  {model_name:18s}:"
            for K in args.K_list:
                if K in all_results[sigma][model_name]:
                    row += f"  K={K:2d}: {all_results[sigma][model_name][K]['ssim']:6.3f}"
            print(row)


if __name__ == '__main__':
    main()
