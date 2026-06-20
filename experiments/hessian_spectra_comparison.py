"""
hessian_spectra_comparison.py
=============================
Subagent 1 of the Landscape Geometry Swarm.

Compute and compare Hessian eigenvalue spectra across EBM models using
Lanczos iteration on CIFAR-10 test images at multiple noise levels.
"""

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

# Must import after path setup
from exp_cifar10 import KANEnergyModel, KAN  # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM  # noqa: E402

OUT_DIR = ROOT / 'outputs' / 'landscape_geometry'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Lanczos implementation (exact from project) ──────────────────────────────

def lanczos_extremes(hvp_fn, dim, device, n_iters=40, dtype=torch.float32):
    """Lanczos on a Hermitian operator given by hvp_fn(v) -> H v.
    Returns (lambda_min_estimate, lambda_max_estimate, all_eigs)."""
    v = torch.randn(dim, device=device, dtype=dtype)
    v = v / v.norm()
    alphas, betas = [], []
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0
    for k in range(n_iters):
        w = hvp_fn(v)
        alpha = torch.dot(w, v).item()
        w = w - alpha * v - beta_prev * v_prev
        w = w - torch.dot(w, v) * v  # re-orthogonalize
        beta = w.norm().item()
        alphas.append(alpha)
        if beta < 1e-10:
            break
        v_prev = v.clone()
        v = w / beta
        beta_prev = beta
        betas.append(beta)
    m = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:m-1], 1) + np.diag(betas[:m-1], -1)
    eigs = np.linalg.eigvalsh(T)
    eig_pos = eigs[eigs > 1e-6 * abs(eigs).max()]
    lam_min = float(eig_pos.min()) if eig_pos.size else float(eigs[eigs > 0].min() if (eigs > 0).any() else abs(eigs).min())
    lam_max = float(eigs.max())
    return lam_min, lam_max, eigs


def make_hvp(model, x_anchor, device):
    """Hessian-vector product of E_theta at x_anchor."""
    x_anchor = x_anchor.detach().to(device)
    shape = x_anchor.shape
    def hvp(v_flat):
        v = v_flat.view(shape)
        with torch.enable_grad():
            xi = x_anchor.detach().requires_grad_(True)
            E = model.energy(xi)
            if E.ndim > 0:
                E = E.sum()
            g = torch.autograd.grad(E, xi, create_graph=True)[0]
            gv = (g * v).sum()
            Hv = torch.autograd.grad(gv, xi)[0]
        return Hv.view(-1).detach()
    return hvp


# ── Data loading ─────────────────────────────────────────────────────────────

def load_cifar10_test():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_ds = datasets.CIFAR10(str(ROOT / 'data'), train=False, download=True, transform=tf)
    return test_ds


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    'kan_f32': {
        'cls': KANEnergyModel,
        'kwargs': {'n_filters': 32, 'filter_size': 5, 'kan_hidden': [96, 16], 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt',
    },
    'kan_small': {
        'cls': KANEnergyModel,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'kan_hidden': [48, 16], 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'kstar_param_scaling' / 'kan_small.pt',
    },
    'conv_mlp_gelu': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'gelu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_gelu.pt',
    },
    'conv_mlp_relu': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'relu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_relu.pt',
    },
    'conv_mlp_silu': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'silu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_silu.pt',
    },
}


SIGMAS = [0.05, 0.10, 0.20]
N_IMAGES_NOISY = 8
N_IMAGES_CLEAN = 4
LANCZOS_ITERS = 40


def compute_spectrum(model, x, device):
    """Compute Hessian spectrum at x. Returns (lam_min, lam_max, eigs)."""
    dim = math.prod(x.shape)
    hvp_fn = make_hvp(model, x, device)
    lam_min, lam_max, eigs = lanczos_extremes(
        lambda v: hvp_fn(v), dim, device, n_iters=LANCZOS_ITERS
    )
    return lam_min, lam_max, eigs


def main():
    print(f"Device: {DEVICE}")
    test_ds = load_cifar10_test()

    results = {"models": {}}

    for model_name, cfg in MODELS.items():
        ckpt_path = cfg['ckpt']
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n=== Loading {model_name} ===")
        model = cfg['cls'](**cfg['kwargs']).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        model_results = {}

        # Noisy images
        for sigma in SIGMAS:
            key = f"sigma_{sigma:.2f}"
            lam_mins, lam_maxs, kappas = [], [], []
            for idx in range(N_IMAGES_NOISY):
                x_clean, _ = test_ds[idx]
                x_clean = x_clean.unsqueeze(0).to(DEVICE)
                x_noisy = x_clean + torch.randn_like(x_clean) * sigma

                lam_min, lam_max, _ = compute_spectrum(model, x_noisy, DEVICE)
                kappa = lam_max / lam_min if lam_min > 0 else float('inf')
                lam_mins.append(lam_min)
                lam_maxs.append(lam_max)
                kappas.append(kappa)
                print(f"  {model_name}, {key}, image={idx}: lam_min={lam_min:.6f}, lam_max={lam_max:.6f}, kappa={kappa:.2f}")
            model_results[key] = {
                "lambda_min": lam_mins,
                "lambda_max": lam_maxs,
                "condition_number": kappas,
            }

        # Clean images
        lam_mins, lam_maxs, kappas = [], [], []
        for idx in range(N_IMAGES_CLEAN):
            x_clean, _ = test_ds[idx]
            x_clean = x_clean.unsqueeze(0).to(DEVICE)

            lam_min, lam_max, _ = compute_spectrum(model, x_clean, DEVICE)
            kappa = lam_max / lam_min if lam_min > 0 else float('inf')
            lam_mins.append(lam_min)
            lam_maxs.append(lam_max)
            kappas.append(kappa)
            print(f"  {model_name}, clean, image={idx}: lam_min={lam_min:.6f}, lam_max={lam_max:.6f}, kappa={kappa:.2f}")
        model_results["clean"] = {
            "lambda_min": lam_mins,
            "lambda_max": lam_maxs,
            "condition_number": kappas,
        }

        results["models"][model_name] = model_results

        # Clean up before next model
        del model
        torch.cuda.empty_cache()

    # Save JSON
    json_path = OUT_DIR / 'hessian_spectra.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gather pooled data per model
    model_names = []
    cond_data = []
    lmax_data = []

    for mname, mres in results["models"].items():
        model_names.append(mname)
        cond_vals = []
        lmax_vals = []
        for key, vals in mres.items():
            cond_vals.extend(vals["condition_number"])
            lmax_vals.extend(vals["lambda_max"])
        cond_data.append(cond_vals)
        lmax_data.append(lmax_vals)

    axes[0].boxplot(cond_data, labels=model_names)
    axes[0].set_title("Condition Number")
    axes[0].set_ylabel(r"$\kappa = \lambda_{\max} / \lambda_{\min}$")
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].boxplot(lmax_data, labels=model_names)
    axes[1].set_title("Max Eigenvalue")
    axes[1].set_ylabel(r"$\lambda_{\max}$")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fig_path = OUT_DIR / 'hessian_figures.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    print(f"Saved figure: {fig_path}")


if __name__ == '__main__':
    main()
