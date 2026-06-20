"""
exp_scaled_eval_cifar10_full.py
===============================
FULL CIFAR-10 test set evaluation (10,000 images) for ALL model variants.

Extends exp_scaled_eval_cifar10.py to include:
- KAN-EBM 110K and 32K
- ConvMLP-GELU, SiLU, Tanh, ReLU
- U-Net EBM
- FFN-DSM
- Micro-DnCNN (single-pass baseline)

Run overnight:
  python experiments/exp_scaled_eval_cifar10_full.py --n_images 10000 --device cuda --batch_size 128 --K_list 1 2 5 10 20
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
from exp_unet_ebm import UNetEBM
from exp_dncnn_cifar10 import MicroDnCNN


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


def load_unet_ebm(checkpoint_path, device):
    model = UNetEBM().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_dncnn(checkpoint_path, device):
    model = MicroDnCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def iterative_denoise(model, noisy, K, eta0=0.05, delta=0.97):
    """Gradient descent denoising matching the original exp_cifar10 protocol."""
    # Use built-in denoise if available (KAN, MLP, UNet)
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
    """Evaluate an EBM model (KAN, ConvMLP, or UNet) at multiple K values."""
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
    parser.add_argument('--n_images', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument('--K_list', type=int, nargs='+', default=[1, 2, 5, 10, 20])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | Evaluating on {args.n_images} images")
    print(f"Batch size: {args.batch_size} | K values: {args.K_list}")
    print(f"Sigmas: {args.sigmas}")
    print(f"Models: KAN-EBM 110K, KAN-EBM 32K, ConvMLP-GELU, ConvMLP-SiLU, ConvMLP-Tanh, ConvMLP-ReLU, U-Net-EBM, FFN-DSM, DnCNN")
    print("=" * 70)

    # Load CIFAR-10 test set
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)

    # Select first n_images (full test set = 10,000)
    images = []
    for i in range(min(args.n_images, len(test_ds))):
        img, _ = test_ds[i]
        images.append(img)
    images = torch.stack(images).to(device)
    print(f"Loaded {len(images)} images to {device}")

    # Load all models
    models = {}
    print("\nLoading models...")
    
    models['KAN-EBM-110K'] = ('ebm', load_kan(
        ROOT / 'outputs/cifar10/kan_ebm_f32.pt', device, n_filters=32, kan_hidden=[96, 16]))
    print("  KAN-EBM-110K loaded")
    
    models['KAN-EBM-32K'] = ('ebm', load_kan(
        ROOT / 'outputs/finalization/kstar_param_scaling/kan_small.pt', device, n_filters=16, kan_hidden=[48, 16]))
    print("  KAN-EBM-32K loaded")
    
    models['ConvMLP-GELU'] = ('ebm', load_conv_mlp(
        ROOT / 'outputs/finalization/rebuttal/conv_mlp_gelu.pt', device, 'gelu'))
    print("  ConvMLP-GELU loaded")
    
    models['ConvMLP-SiLU'] = ('ebm', load_conv_mlp(
        ROOT / 'outputs/finalization/rebuttal/conv_mlp_silu.pt', device, 'silu'))
    print("  ConvMLP-SiLU loaded")
    
    models['ConvMLP-Tanh'] = ('ebm', load_conv_mlp(
        ROOT / 'outputs/finalization/rebuttal/conv_mlp_tanh.pt', device, 'tanh'))
    print("  ConvMLP-Tanh loaded")
    
    models['ConvMLP-ReLU'] = ('ebm', load_conv_mlp(
        ROOT / 'outputs/finalization/rebuttal/conv_mlp_relu.pt', device, 'relu'))
    print("  ConvMLP-ReLU loaded")
    
    models['U-Net-EBM'] = ('ebm', load_unet_ebm(
        ROOT / 'outputs/unet_ebm/unet_ebm.pt', device))
    print("  U-Net-EBM loaded")
    
    models['FFN-DSM'] = ('ffn', load_ffn(
        ROOT / 'outputs/cifar10/ffn_dsm_f32.pt', device, hidden=512))
    print("  FFN-DSM loaded")
    
    models['DnCNN'] = ('ffn', load_dncnn(
        ROOT / 'outputs/dncnn_baseline/dncnn_cifar10.pt', device))
    print("  DnCNN loaded")
    
    print(f"\nTotal models loaded: {len(models)}")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    for sigma in args.sigmas:
        print(f"\n{'='*70}")
        print(f"SIGMA = {sigma}")
        print(f"{'='*70}")
        all_results[sigma] = {}
        
        for model_name, (mtype, model) in models.items():
            print(f"\n  [{model_name}] ...", end='', flush=True)
            t0 = time.time()

            # Process in batches
            batch_results = []
            n_batches = math.ceil(len(images) / args.batch_size)
            
            for b_idx in range(n_batches):
                b_start = b_idx * args.batch_size
                b_end = min((b_idx + 1) * args.batch_size, len(images))
                batch = images[b_start:b_end]
                
                if mtype == 'ebm':
                    res = evaluate_ebm(model, batch, sigma, args.K_list, device)
                else:
                    # FFN/DnCNN: single-pass, same result for all K
                    res = {K: evaluate_ffn(model, batch, sigma, device) for K in args.K_list}
                batch_results.append(res)
                
                # Progress indicator every 10 batches
                if (b_idx + 1) % 10 == 0 or b_idx == n_batches - 1:
                    print(f" {(b_idx+1)/n_batches*100:.0f}%", end='', flush=True)

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
            print(f" done in {elapsed/60:.1f} min")
            
            # Print key results for this sigma
            peak_k = max(agg.keys(), key=lambda k: agg[k]['psnr'])
            print(f"    Peak PSNR: {agg[peak_k]['psnr']:.2f} dB at K={peak_k} (SSIM: {agg[peak_k]['ssim']:.3f})")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Total evaluation time: {total_elapsed / 3600:.2f} hours")
    print(f"{'='*70}")

    # Save results
    out_dir = ROOT / 'outputs' / 'scaled_eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'cifar10_full_{args.n_images}_{int(time.time())}.json'
    
    result_dict = {
        'n_images': args.n_images,
        'sigmas': args.sigmas,
        'K_list': args.K_list,
        'batch_size': args.batch_size,
        'device': str(device),
        'total_time_s': total_elapsed,
        'models': list(models.keys()),
        'results': all_results,
    }
    
    out_path.write_text(json.dumps(result_dict, indent=2))
    print(f"\nResults saved to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
