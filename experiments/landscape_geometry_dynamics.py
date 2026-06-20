"""
landscape_geometry_dynamics.py
==============================
Subagent 4: Test whether KAN's smoother landscape enables better dynamics
with momentum and varying step sizes.

Compares:
  1. KAN-EBM f32  (112,608 params)
  2. ConvMLP-GELU (34,977 params)
  3. ConvMLP-ReLU (34,977 params)

Experiments:
  Part A: Momentum Benefit
  Part B: Step-Size Robustness
  Part C: Convergence Trajectory

Outputs:
  outputs/landscape_geometry/dynamics_analysis.json
  outputs/landscape_geometry/dynamics_figures.png
"""

import math
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel
from rebuttal_smooth_mlp import ConvSmoothMLPEBM

OUT_DIR = ROOT / 'outputs' / 'landscape_geometry'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {device}")
if device.type == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

from torchvision import datasets, transforms
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)

# Collect images
def get_images(indices):
    return torch.stack([test_ds[i][0] for i in indices])

N_IMAGES_MOMENTUM = 16
N_IMAGES_STEP = 16
N_IMAGES_CONV = 4

indices_momentum = list(range(N_IMAGES_MOMENTUM))
indices_step = list(range(N_IMAGES_STEP))
indices_conv = list(range(N_IMAGES_CONV))

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def psnr(clean, pred, data_range=2.0):
    pred = pred.clamp(-1, 1)
    mse = F.mse_loss(pred, clean).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(data_range ** 2 / mse)

# ──────────────────────────────────────────────────────────────────────────────
# Inference routines
# ──────────────────────────────────────────────────────────────────────────────

def vanilla_gd(model, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
    u = x_noisy.clone()
    step = dt
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            E = model.energy(ui)
            if E.ndim > 0:
                E = E.sum()
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach().clamp(-1, 1)
        step *= dt_decay
    return u


def heavy_ball_gd(model, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97, beta=0.9):
    u = x_noisy.clone()
    v = torch.zeros_like(u)
    step = dt
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            E = model.energy(ui)
            if E.ndim > 0:
                E = E.sum()
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        v = beta * v - step * g
        u = (u + v).detach().clamp(-1, 1)
        step *= dt_decay
    return u


def nesterov_gd(model, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97, beta=0.9):
    u = x_noisy.clone()
    v = torch.zeros_like(u)
    step = dt
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = (u + beta * v).detach().requires_grad_(True)
            E = model.energy(ui)
            if E.ndim > 0:
                E = E.sum()
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        v = beta * v - step * g
        u = (u + v).detach().clamp(-1, 1)
        step *= dt_decay
    return u


def adam_gd(model, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97,
            beta1=0.9, beta2=0.999, eps=1e-8):
    u = x_noisy.clone()
    m = torch.zeros_like(u)
    s = torch.zeros_like(u)
    step = dt
    for t in range(1, n_steps + 1):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            E = model.energy(ui)
            if E.ndim > 0:
                E = E.sum()
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        s_hat = s / (1 - beta2 ** t)
        u = (u - step * m_hat / (s_hat.sqrt() + eps)).detach().clamp(-1, 1)
        step *= dt_decay
    return u


def vanilla_gd_with_trajectory(model, x_noisy, n_steps=50, dt=0.05, dt_decay=0.97):
    u = x_noisy.clone()
    step = dt
    psnrs = []
    energies = []
    for _ in range(n_steps):
        ui = u.detach().requires_grad_(True)
        E = model.energy(ui)
        if E.ndim > 0:
            E = E.sum()
        g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach().clamp(-1, 1)
        step *= dt_decay
    return u


def convergence_trajectory(model, clean, x_noisy, n_steps=50, dt=0.05, dt_decay=0.97):
    u = x_noisy.clone()
    step = dt
    psnrs = []
    energies = []
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            E = model.energy(ui)
            if E.ndim > 0:
                E = E.sum()
            g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach().clamp(-1, 1)
        step *= dt_decay
        # Record metrics
        psnrs.append(psnr(clean, u))
        with torch.no_grad():
            e_val = model.energy(u)
            if e_val.ndim > 0:
                e_val = e_val.sum()
            energies.append(e_val.item())
    return psnrs, energies

# ──────────────────────────────────────────────────────────────────────────────
# Part A: Momentum Benefit
# ──────────────────────────────────────────────────────────────────────────────

def run_momentum_experiment(model, clean_images, sigma=0.15):
    model.eval()
    vanilla_psnr = []
    heavy_ball_psnr = []
    nesterov_psnr = []
    adam_psnr = []

    for idx, clean in enumerate(clean_images):
        torch.manual_seed(42 + idx)
        clean = clean.unsqueeze(0)  # (1, C, H, W)
        noise = torch.randn_like(clean) * sigma
        x_noisy = (clean + noise).clamp(-1, 1)

        with torch.no_grad():
            u_v = vanilla_gd(model, x_noisy)
            u_h = heavy_ball_gd(model, x_noisy)
            u_n = nesterov_gd(model, x_noisy)
            u_a = adam_gd(model, x_noisy)

        vanilla_psnr.append(psnr(clean, u_v))
        heavy_ball_psnr.append(psnr(clean, u_h))
        nesterov_psnr.append(psnr(clean, u_n))
        adam_psnr.append(psnr(clean, u_a))

    heavy_ball_gain = [h - v for h, v in zip(heavy_ball_psnr, vanilla_psnr)]
    nesterov_gain = [n - v for n, v in zip(nesterov_psnr, vanilla_psnr)]
    adam_gain = [a - v for a, v in zip(adam_psnr, vanilla_psnr)]

    return {
        'vanilla_psnr': vanilla_psnr,
        'heavy_ball_psnr': heavy_ball_psnr,
        'nesterov_psnr': nesterov_psnr,
        'adam_psnr': adam_psnr,
        'heavy_ball_gain': heavy_ball_gain,
        'nesterov_gain': nesterov_gain,
        'adam_gain': adam_gain,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Part B: Step-Size Robustness
# ──────────────────────────────────────────────────────────────────────────────

def run_step_size_experiment(model, clean_images, dt_values, sigma=0.15):
    model.eval()
    psnr_per_dt = []

    for dt in dt_values:
        psnrs = []
        for idx, clean in enumerate(clean_images):
            torch.manual_seed(42 + idx)
            clean = clean.unsqueeze(0)
            noise = torch.randn_like(clean) * sigma
            x_noisy = (clean + noise).clamp(-1, 1)
            with torch.no_grad():
                u = vanilla_gd(model, x_noisy, n_steps=10, dt=dt, dt_decay=0.97)
            psnrs.append(psnr(clean, u))
        psnr_per_dt.append(float(np.mean(psnrs)))

    best_psnr = max(psnr_per_dt)
    stable_dts = [dt for dt, p in zip(dt_values, psnr_per_dt) if p >= best_psnr - 1.0]
    stable_range = max(stable_dts) - min(stable_dts) if stable_dts else 0.0

    return {
        'dt_values': dt_values,
        'psnr_per_dt': psnr_per_dt,
        'stable_range': stable_range,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Part C: Convergence Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def run_convergence_experiment(model, clean_images, sigma=0.2):
    model.eval()
    psnr_trajectories = []
    energy_trajectories = []

    for idx, clean in enumerate(clean_images):
        torch.manual_seed(42 + idx)
        clean = clean.unsqueeze(0)
        noise = torch.randn_like(clean) * sigma
        x_noisy = (clean + noise).clamp(-1, 1)
        psnrs, energies = convergence_trajectory(model, clean, x_noisy)
        psnr_trajectories.append(psnrs)
        energy_trajectories.append(energies)

    return {
        'psnr_trajectories': psnr_trajectories,
        'energy_trajectories': energy_trajectories,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(name, config, checkpoint_path):
    print(f"\n[Loading] {name}")
    try:
        model = config['cls'](**config['kwargs']).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")
        print(f"  Checkpoint: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    model_configs = {
        'kan_f32': {
            'cls': KANEnergyModel,
            'kwargs': {
                'n_filters': 32,
                'filter_size': 5,
                'kan_hidden': [96, 16],
                'n_channels': 3,
            },
            'checkpoint': ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt',
        },
        'conv_mlp_gelu': {
            'cls': ConvSmoothMLPEBM,
            'kwargs': {
                'n_filters': 16,
                'filter_size': 5,
                'mlp_hidden': 160,
                'n_channels': 3,
                'activation': 'gelu',
            },
            'checkpoint': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_gelu.pt',
        },
        'conv_mlp_relu': {
            'cls': ConvSmoothMLPEBM,
            'kwargs': {
                'n_filters': 16,
                'filter_size': 5,
                'mlp_hidden': 160,
                'n_channels': 3,
                'activation': 'relu',
            },
            'checkpoint': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_relu.pt',
        },
    }

    # Prepare data on CPU first
    clean_momentum = get_images(indices_momentum).to(device)
    clean_step = get_images(indices_step).to(device)
    clean_conv = get_images(indices_conv).to(device)

    results = {}

    for model_name, config in model_configs.items():
        model = load_model(model_name, config, config['checkpoint'])
        if model is None:
            print(f"  Skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Running experiments for {model_name}")
        print(f"{'='*60}")

        # Part A
        print("\n[Part A] Momentum Benefit")
        momentum_res = run_momentum_experiment(model, clean_momentum)
        print(f"  Vanilla mean PSNR: {np.mean(momentum_res['vanilla_psnr']):.3f} dB")
        print(f"  Heavy-ball gain:   {np.mean(momentum_res['heavy_ball_gain']):+.3f} dB")
        print(f"  Nesterov gain:     {np.mean(momentum_res['nesterov_gain']):+.3f} dB")
        print(f"  Adam gain:         {np.mean(momentum_res['adam_gain']):+.3f} dB")

        # Part B
        print("\n[Part B] Step-Size Robustness")
        dt_values = [0.01, 0.02, 0.05, 0.10, 0.20]
        step_res = run_step_size_experiment(model, clean_step, dt_values)
        print(f"  dt_values: {dt_values}")
        print(f"  PSNR per dt: {[f'{p:.3f}' for p in step_res['psnr_per_dt']]}")
        print(f"  Stable range: {step_res['stable_range']:.3f}")

        # Part C
        print("\n[Part C] Convergence Trajectory")
        conv_res = run_convergence_experiment(model, clean_conv)
        print(f"  Trajectories recorded for {len(conv_res['psnr_trajectories'])} images x 50 steps")

        results[model_name] = {
            'momentum': momentum_res,
            'step_size_robustness': step_res,
            'convergence': conv_res,
        }

        # Clean up
        del model
        torch.cuda.empty_cache()
        print(f"\n[Cleared GPU cache for {model_name}]")

    # Save JSON
    json_path = OUT_DIR / 'dynamics_analysis.json'
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n[Saved] {json_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Plotting] Generating figure...")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Colors
    colors = {
        'kan_f32': '#3B5BDB',
        'conv_mlp_gelu': '#0CA678',
        'conv_mlp_relu': '#E64980',
    }
    labels = {
        'kan_f32': 'KAN-EBM f32',
        'conv_mlp_gelu': 'ConvMLP-GELU',
        'conv_mlp_relu': 'ConvMLP-ReLU',
    }

    # Top: Bar chart of mean momentum gain
    ax = axes[0]
    models_in_results = list(results.keys())
    x = np.arange(3)  # heavy_ball, nesterov, adam
    width = 0.25
    for i, model_name in enumerate(models_in_results):
        gains = [
            np.mean(results[model_name]['momentum']['heavy_ball_gain']),
            np.mean(results[model_name]['momentum']['nesterov_gain']),
            np.mean(results[model_name]['momentum']['adam_gain']),
        ]
        ax.bar(x + i * width, gains, width, label=labels[model_name], color=colors[model_name])
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Heavy-ball', 'Nesterov', 'Adam'])
    ax.set_ylabel('Mean PSNR Gain (dB)')
    ax.set_title('Momentum Gain vs Vanilla GD')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Middle: Step-size robustness curves
    ax = axes[1]
    for model_name in models_in_results:
        dt_values = results[model_name]['step_size_robustness']['dt_values']
        psnrs = results[model_name]['step_size_robustness']['psnr_per_dt']
        ax.plot(dt_values, psnrs, 'o-', label=labels[model_name], color=colors[model_name], linewidth=2)
    ax.set_xlabel('Step size dt')
    ax.set_ylabel('Mean PSNR (dB)')
    ax.set_title('Step-Size Robustness (K=10)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: Convergence trajectories (representative image index 0)
    ax = axes[2]
    for model_name in models_in_results:
        psnr_traj = results[model_name]['convergence']['psnr_trajectories'][0]
        ax.plot(range(1, len(psnr_traj) + 1), psnr_traj, '-', label=labels[model_name],
                color=colors[model_name], linewidth=2)
    ax.set_xlabel('Inference Step')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Convergence Trajectory (Image 0, σ=0.2)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = OUT_DIR / 'dynamics_figures.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {fig_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
