"""
landscape_geometry.py
=====================
Subagent 3 – Landscape Geometry Swarm

Measures basin-of-attraction radii and energy-barrier heights for
four EBM variants on CIFAR-10.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_cifar10 import KANEnergyModel          # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM  # noqa: E402

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)

# ── constants ────────────────────────────────────────────────────────────────
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_IMAGES_BASIN = 16
N_TRIALS_BASIN = 4
K_STEPS = 20
DT_INIT = 0.05
DT_DECAY = 0.97
PSNR_THRESHOLD = 20.0
N_IMAGES_BARRIER = 8
BARRIER_SIGMA = 0.20
BARRIER_ALPHAS = np.linspace(0, 1, 50).tolist()

OUT_DIR = ROOT / 'outputs' / 'landscape_geometry'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def psnr(clean, recon):
    mse = F.mse_loss(recon, clean).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(4.0 / mse)


def load_cifar10():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = datasets.CIFAR10(str(ROOT / 'data'), train=False, download=True, transform=tf)
    return ds


def denoise_fixed_batch(model, x_noisy, n_steps=K_STEPS, dt=DT_INIT, dt_decay=DT_DECAY):
    """Batched denoising; x_noisy may be (B, C, H, W)."""
    u = x_noisy.clone()
    step = dt
    for _ in range(n_steps):
        ui = u.detach().requires_grad_(True)
        E = model.energy(ui)
        if E.ndim > 0:
            E = E.sum()
        g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach()
        step *= dt_decay
    return u.clamp(-1, 1)


def denoise_fixed(model, x_noisy, n_steps=K_STEPS, dt=DT_INIT, dt_decay=DT_DECAY):
    return denoise_fixed_batch(model, x_noisy, n_steps, dt, dt_decay)


# ── basin radius ─────────────────────────────────────────────────────────────

def measure_basin_radius(model, images):
    mean_psnr_per_sigma = []
    for sigma in SIGMAS:
        psnrs = []
        for img_idx in range(N_IMAGES_BASIN):
            x_clean = images[img_idx].unsqueeze(0).cuda()
            # batch 4 trials together
            batch_clean = x_clean.repeat(N_TRIALS_BASIN, 1, 1, 1)
            for trial in range(N_TRIALS_BASIN):
                seed = 42 + img_idx * 7 + trial
                torch.manual_seed(seed)
                np.random.seed(seed)
            # generate noise for the whole batch with a single RNG call after last seed
            # (seeds per trial were set sequentially, but we need independent noise)
            # Instead: generate each trial's noise separately under its own seed
            noises = []
            for trial in range(N_TRIALS_BASIN):
                seed = 42 + img_idx * 7 + trial
                torch.manual_seed(seed)
                np.random.seed(seed)
                noises.append(torch.randn_like(x_clean))
            noise = torch.cat(noises, dim=0) * sigma
            x_noisy = (batch_clean + noise).clamp(-1, 1)
            x_denoised = denoise_fixed_batch(model, x_noisy)
            for b in range(N_TRIALS_BASIN):
                psnrs.append(psnr(x_clean, x_denoised[b:b+1]))
        mean_psnr = float(np.mean(psnrs))
        mean_psnr_per_sigma.append(mean_psnr)
        print(f"  sigma={sigma:.2f}  mean PSNR={mean_psnr:.2f} dB")

    basin_radius = 0.0
    for s, p in zip(SIGMAS, mean_psnr_per_sigma):
        if p > PSNR_THRESHOLD:
            basin_radius = s
    return {
        "sigma_init_list": SIGMAS,
        "mean_psnr_per_sigma": mean_psnr_per_sigma,
        "basin_radius_threshold": basin_radius,
    }


# ── energy barrier ───────────────────────────────────────────────────────────

def measure_barrier(model, images):
    barrier_heights = []
    energy_profiles = []
    energy_at_clean = []
    energy_at_noisy = []

    for img_idx in range(N_IMAGES_BARRIER):
        x_clean = images[img_idx].unsqueeze(0).cuda()
        torch.manual_seed(42 + img_idx)
        np.random.seed(42 + img_idx)
        noise = torch.randn_like(x_clean) * BARRIER_SIGMA
        x_noisy = (x_clean + noise).clamp(-1, 1)

        profile = []
        with torch.no_grad():
            for alpha in BARRIER_ALPHAS:
                x_interp = (1 - alpha) * x_noisy + alpha * x_clean
                E = model.energy(x_interp)
                if E.ndim > 0:
                    E = E.sum()
                profile.append(E.item())

        e_noisy = profile[0]
        e_clean = profile[-1]
        barrier = max(profile) - min(e_noisy, e_clean)

        barrier_heights.append(barrier)
        energy_profiles.append(profile)
        energy_at_noisy.append(e_noisy)
        energy_at_clean.append(e_clean)

    return {
        "barrier_heights": barrier_heights,
        "energy_profiles": energy_profiles,
        "energy_at_clean": energy_at_clean,
        "energy_at_noisy": energy_at_noisy,
    }


# ── model wrappers ───────────────────────────────────────────────────────────

MODEL_SPECS = [
    {
        "key": "kan_f32",
        "cls": KANEnergyModel,
        "kwargs": {"n_filters": 32, "filter_size": 5, "kan_hidden": [96, 16], "n_channels": 3},
        "ckpt": ROOT / "outputs" / "cifar10" / "kan_ebm_f32.pt",
    },
    {
        "key": "kan_small",
        "cls": KANEnergyModel,
        "kwargs": {"n_filters": 16, "filter_size": 5, "kan_hidden": [48, 16], "n_channels": 3},
        "ckpt": ROOT / "outputs" / "finalization" / "kstar_param_scaling" / "kan_small.pt",
    },
    {
        "key": "conv_mlp_gelu",
        "cls": ConvSmoothMLPEBM,
        "kwargs": {"n_filters": 16, "filter_size": 5, "mlp_hidden": 160, "n_channels": 3, "activation": "gelu"},
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_gelu.pt",
    },
    {
        "key": "conv_mlp_relu",
        "cls": ConvSmoothMLPEBM,
        "kwargs": {"n_filters": 16, "filter_size": 5, "mlp_hidden": 160, "n_channels": 3, "activation": "relu"},
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_relu.pt",
    },
]


def run_all():
    ds = load_cifar10()
    images = [ds[i][0] for i in range(max(N_IMAGES_BASIN, N_IMAGES_BARRIER))]

    results = {}

    for spec in MODEL_SPECS:
        key = spec["key"]
        print(f"\n{'='*60}")
        print(f"Model: {key}")
        print(f"{'='*60}")

        if not spec["ckpt"].exists():
            print(f"  Checkpoint not found: {spec['ckpt']} – skipping.")
            continue

        try:
            model = spec["cls"](**spec["kwargs"]).cuda()
            state = torch.load(spec["ckpt"], map_location='cuda', weights_only=False)
            model.load_state_dict(state)
            model.eval()
            print(f"  Loaded {sum(p.numel() for p in model.parameters()):,} params")
        except Exception as e:
            print(f"  Failed to load model: {e} – skipping.")
            continue

        print("  -- Basin radius --")
        basin = measure_basin_radius(model, images)
        print("  -- Energy barrier --")
        with torch.no_grad():
            barrier = measure_barrier(model, images)

        results[key] = {
            "basin_radius": basin,
            "barrier": barrier,
        }

        del model
        torch.cuda.empty_cache()

    # ── save JSON ──
    json_path = OUT_DIR / "basin_geometry.json"
    with open(json_path, 'w') as f:
        json.dump({"models": results}, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    # ── figure ──
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Top: PSNR vs σ_init
    ax_top = axes[0]
    colors = {'kan_f32': 'C0', 'kan_small': 'C1', 'conv_mlp_gelu': 'C2', 'conv_mlp_relu': 'C3'}
    markers = {'kan_f32': 'o', 'kan_small': 's', 'conv_mlp_gelu': '^', 'conv_mlp_relu': 'v'}
    labels = {'kan_f32': 'KAN-EBM f32', 'kan_small': 'KAN-EBM small',
              'conv_mlp_gelu': 'ConvMLP-GELU', 'conv_mlp_relu': 'ConvMLP-ReLU'}

    for key, data in results.items():
        basin = data["basin_radius"]
        ax_top.plot(basin["sigma_init_list"], basin["mean_psnr_per_sigma"],
                    color=colors[key], marker=markers[key], label=labels[key], linewidth=2)
    ax_top.axhline(PSNR_THRESHOLD, color='gray', linestyle='--', linewidth=1)
    ax_top.set_xlabel(r'Initial noise $\sigma_{\rm init}$')
    ax_top.set_ylabel('Mean PSNR (dB)')
    ax_top.set_title('Basin of Attraction: PSNR Recovery vs Noise Level')
    ax_top.legend(loc='best')
    ax_top.grid(True, alpha=0.3)

    # Bottom: example energy profiles (image 0)
    ax_bot = axes[1]
    for key, data in results.items():
        prof = data["barrier"]["energy_profiles"][0]
        ax_bot.plot(BARRIER_ALPHAS, prof,
                    color=colors[key], marker=markers[key], markevery=5,
                    label=labels[key], linewidth=2, alpha=0.8)
    ax_bot.set_xlabel(r'Interpolation parameter $\alpha$')
    ax_bot.set_ylabel('Energy')
    ax_bot.set_title('Energy Barrier: Profile Along Clean–Noisy Interpolation (Image 0)')
    ax_bot.legend(loc='best')
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = OUT_DIR / "basin_figures.png"
    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")


if __name__ == '__main__':
    run_all()
