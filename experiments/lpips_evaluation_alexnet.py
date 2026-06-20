"""
lpips_evaluation.py
=====================
Compute LPIPS perceptual metric for all denoising models at sigma=0.10
on 100 CIFAR-10 test images. Evaluates each model at K=1 and at its
peak K (from existing results).

Output: outputs/landscape_geometry/lpips_evaluation.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

import lpips

# ---------------------------------------------------------------------------
# Imports for model classes
# ---------------------------------------------------------------------------
from exp_cifar10 import KANEnergyModel, FFNDenoiser
from rebuttal_smooth_mlp import ConvSmoothMLPEBM
from exp_dncnn_cifar10 import MicroDnCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA = 0.10
N_IMAGES = 100
BATCH_SIZE = 16
SEED = 42
OUT_PATH = ROOT / 'outputs' / 'landscape_geometry' / 'lpips_evaluation_alexnet.json'

MODELS = {
    'kan_f32': {
        'class': KANEnergyModel,
        'kwargs': {'n_filters': 32, 'filter_size': 5, 'kan_hidden': [96, 16], 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt',
        'peak_K': 2,
    },
    'kan_small': {
        'class': KANEnergyModel,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'kan_hidden': [48, 16], 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'kstar_param_scaling' / 'kan_small.pt',
        'peak_K': 2,
    },
    'conv_mlp_gelu': {
        'class': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'gelu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_gelu.pt',
        'peak_K': 2,
    },
    'conv_mlp_relu': {
        'class': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'relu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_relu.pt',
        'peak_K': 2,
    },
    'ffn_dsm': {
        'class': FFNDenoiser,
        'kwargs': {'img_size': 32, 'n_channels': 3, 'hidden': 512},
        'ckpt': ROOT / 'outputs' / 'cifar10' / 'ffn_dsm_f32.pt',
        'peak_K': 1,
    },
    'micro_dncnn': {
        'class': MicroDnCNN,
        'kwargs': {'n_channels': 3, 'depth': 5, 'n_filters': 16},
        'ckpt': ROOT / 'outputs' / 'dncnn_baseline' / 'dncnn_cifar10.pt',
        'peak_K': 1,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_images(n=100):
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    images = torch.stack([ds[i][0] for i in range(n)])
    return images  # (N, 3, 32, 32) in [-1, 1]

# ---------------------------------------------------------------------------
# Denoising helpers
# ---------------------------------------------------------------------------
def denoise_ebm(model, x_noisy, n_steps, dt=0.05, dt_decay=0.97):
    """Standard gradient-descent denoising for EBM models."""
    u = x_noisy.clone()
    step = dt
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            E = model.energy(ui)
            E = E.sum() if E.ndim > 0 else E
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach()
        step *= dt_decay
    return u


def denoise_kan_ebm(model, x_noisy, n_steps, dt=0.05, dt_decay=0.97):
    """KANEnergyModel has its own denoise() with clamped energy_grad."""
    # Use the model's built-in denoise to stay consistent with training
    return model.denoise(x_noisy, n_steps=n_steps, dt=dt, dt_decay=dt_decay)


def denoise_feedforward(model, x_noisy, n_steps=1):
    """Feed-forward models: just call forward (ignore n_steps)."""
    with torch.no_grad():
        return model(x_noisy)

# ---------------------------------------------------------------------------
# LPIPS evaluation
# ---------------------------------------------------------------------------
def compute_lpips_for_model(model, clean_images, sigma, K, loss_fn, model_key):
    """
    Compute LPIPS for a model at a given K.
    Returns array of shape (N_IMAGES,).
    """
    model.eval()
    all_lpips = []

    # Use appropriate denoising function
    if isinstance(model, KANEnergyModel):
        denoise_fn = lambda x, k: denoise_kan_ebm(model, x, n_steps=k)
    elif isinstance(model, ConvSmoothMLPEBM):
        denoise_fn = lambda x, k: denoise_ebm(model, x, n_steps=k)
    else:
        denoise_fn = lambda x, k: denoise_feedforward(model, x, n_steps=k)

    n_batches = math.ceil(len(clean_images) / BATCH_SIZE)
    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(clean_images))
        xc = clean_images[start:end].to(DEVICE)

        torch.manual_seed(SEED + b)  # deterministic noise per batch
        xn = xc + torch.randn_like(xc) * sigma

        with torch.no_grad() if not isinstance(model, (KANEnergyModel, ConvSmoothMLPEBM)) else torch.enable_grad():
            if isinstance(model, (KANEnergyModel, ConvSmoothMLPEBM)):
                # EBM inference requires gradients
                u = denoise_fn(xn, K)
            else:
                u = denoise_fn(xn, K)

        u = u.clamp(-1, 1)
        with torch.no_grad():
            dist = loss_fn(u, xc)  # shape depends on lpips version
        dist = dist.view(-1).detach().cpu()
        all_lpips.append(dist)

    return torch.cat(all_lpips).numpy()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading {N_IMAGES} CIFAR-10 test images...")
    clean_images = load_test_images(N_IMAGES)
    print(f"Clean images shape: {clean_images.shape}, range: [{clean_images.min():.3f}, {clean_images.max():.3f}]")

    # Initialize LPIPS (SqueezeNet backbone - lightweight, valid alternative)
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
    loss_fn.eval()

    results = {
        'sigma': SIGMA,
        'n_images': N_IMAGES,
        'models': {},
    }

    print("\n" + "=" * 70)
    print("LPIPS Evaluation @ sigma=0.10")
    print("=" * 70)

    for model_key, cfg in MODELS.items():
        print(f"\n[{model_key}] Loading checkpoint...")
        try:
            model = cfg['class'](**cfg['kwargs']).to(DEVICE)
            state = torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=False)
            model.load_state_dict(state)
            print(f"  -> Loaded {cfg['ckpt'].name}")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  -> Parameters: {n_params:,}")
        except Exception as e:
            print(f"  -> FAILED to load: {e}. Skipping.")
            continue

        # Evaluate at K=1
        print(f"  Computing LPIPS @ K=1...")
        lpips_k1 = compute_lpips_for_model(model, clean_images, SIGMA, K=1, loss_fn=loss_fn, model_key=model_key)
        lpips_k1_mean = float(np.mean(lpips_k1))
        lpips_k1_std = float(np.std(lpips_k1, ddof=1))

        # Evaluate at peak K
        peak_K = cfg['peak_K']
        print(f"  Computing LPIPS @ K={peak_K} (peak)...")
        lpips_peak = compute_lpips_for_model(model, clean_images, SIGMA, K=peak_K, loss_fn=loss_fn, model_key=model_key)
        lpips_peak_mean = float(np.mean(lpips_peak))
        lpips_peak_std = float(np.std(lpips_peak, ddof=1))

        results['models'][model_key] = {
            'lpips_at_K1': {'mean': round(lpips_k1_mean, 6), 'std': round(lpips_k1_std, 6)},
            'lpips_at_peak_K': {'mean': round(lpips_peak_mean, 6), 'std': round(lpips_peak_std, 6), 'K': peak_K},
            'n_params': n_params,
        }

        print(f"  LPIPS @ K=1   : {lpips_k1_mean:.4f} ± {lpips_k1_std:.4f}")
        print(f"  LPIPS @ K={peak_K}   : {lpips_peak_mean:.4f} ± {lpips_peak_std:.4f}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Save JSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {OUT_PATH}")

    # Print formatted table
    print("\n" + "=" * 70)
    print("LPIPS Results Summary (lower is better)")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>10} {'LPIPS@K=1':>14} {'LPIPS@Peak K':>16}")
    print("-" * 70)
    for model_key, r in results['models'].items():
        k1 = r['lpips_at_K1']['mean']
        pk = r['lpips_at_peak_K']['mean']
        K = r['lpips_at_peak_K']['K']
        npar = r['n_params']
        print(f"{model_key:<20} {npar:>10,} {k1:>14.4f} {pk:>12.4f} (K={K})")
    print("=" * 70)


if __name__ == '__main__':
    main()
