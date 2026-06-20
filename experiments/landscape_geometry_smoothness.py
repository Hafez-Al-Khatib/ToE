"""
landscape_geometry_smoothness.py
================================
Subagent 2: Measure gradient smoothness (local Lipschitz constants and
Hessian proxy) across EBM models.

Outputs:
  outputs/landscape_geometry/gradient_smoothness.json
  outputs/landscape_geometry/smoothness_figures.png
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

from exp_cifar10 import KANEnergyModel, MLPEnergyModel  # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = ROOT / 'outputs' / 'landscape_geometry'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

# ── Load CIFAR-10 test images ─────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_ds = datasets.CIFAR10('data', train=False, download=True, transform=TF)


def get_images(indices):
    """Return a batch tensor (B, C, H, W) on DEVICE."""
    imgs = torch.stack([test_ds[i][0] for i in indices])
    return imgs.to(DEVICE)


CLEAN_IMGS_32 = get_images(list(range(32)))
CLEAN_IMGS_16 = get_images(list(range(16)))


def energy_scalar(model, x):
    """Call model.energy(x) and return a scalar."""
    e = model.energy(x)
    if e.numel() > 1:
        return e.sum()
    return e


def compute_gradient(model, x, create_graph=False):
    """Compute ∇E(x) for a batch x. Returns tensor same shape as x."""
    xi = x.detach().requires_grad_(True)
    e = energy_scalar(model, xi)
    grad = torch.autograd.grad(e, xi, create_graph=create_graph)[0]
    return grad


def gradient_norm(grad):
    """||∇E(x)||_2 for each sample in a batch. grad: (B, ...)."""
    return grad.view(grad.shape[0], -1).norm(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Gradient Lipschitz Constant
# ═══════════════════════════════════════════════════════════════════════════════
def measure_lipschitz_and_grad_norms(model, clean_imgs, n_pert=16, eps=0.01, upsample_to=None):
    """
    Returns dict with:
      - lipschitz_proxy: list of L values (len = B * n_pert)
      - gradient_norm_clean: list of ||g||_2 at clean images
      - gradient_norm_noisy_0.1: list of ||g||_2 at noisy images (σ=0.1)
      - gradient_norm_noisy_0.2: list of ||g||_2 at noisy images (σ=0.2)
    """
    model.eval()
    if upsample_to is not None:
        clean_imgs = F.interpolate(clean_imgs, size=upsample_to, mode='bilinear', align_corners=False)
    B = clean_imgs.shape[0]
    lipschitz = []

    with torch.no_grad():
        noisy_01 = clean_imgs + torch.randn_like(clean_imgs) * 0.1
        noisy_02 = clean_imgs + torch.randn_like(clean_imgs) * 0.2

    # Clean gradients
    g_clean = compute_gradient(model, clean_imgs)
    grad_norm_clean = gradient_norm(g_clean).cpu().tolist()

    # Noisy gradients
    g_01 = compute_gradient(model, noisy_01)
    grad_norm_noisy_01 = gradient_norm(g_01).cpu().tolist()
    g_02 = compute_gradient(model, noisy_02)
    grad_norm_noisy_02 = gradient_norm(g_02).cpu().tolist()

    # Lipschitz: for each image, for n_pert random perturbations
    for i in range(B):
        x = clean_imgs[i:i + 1]
        g_x = compute_gradient(model, x)
        for _ in range(n_pert):
            delta = torch.randn_like(x) * eps
            x_pert = x + delta
            g_pert = compute_gradient(model, x_pert)

            num = (g_pert - g_x).view(-1).norm().item()
            den = delta.view(-1).norm().item()
            L = num / (den + 1e-12)
            lipschitz.append(L)

    return {
        "lipschitz_proxy": lipschitz,
        "gradient_norm_clean": grad_norm_clean,
        "gradient_norm_noisy_0.1": grad_norm_noisy_01,
        "gradient_norm_noisy_0.2": grad_norm_noisy_02,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: Finite-Difference Hessian Proxy
# ═══════════════════════════════════════════════════════════════════════════════
def measure_hessian_proxy(model, clean_imgs, eps=0.01, upsample_to=None):
    """
    Returns list of H_diag_proxy values (one per image).
    H_diag_proxy = ||g(x+ε·randn) - g(x)|| / ε
    """
    model.eval()
    if upsample_to is not None:
        clean_imgs = F.interpolate(clean_imgs, size=upsample_to, mode='bilinear', align_corners=False)
    B = clean_imgs.shape[0]
    hessian_proxy = []

    for i in range(B):
        x = clean_imgs[i:i + 1]
        g_x = compute_gradient(model, x)
        delta = torch.randn_like(x) * eps
        x_pert = x + delta
        g_pert = compute_gradient(model, x_pert)

        diff_norm = (g_pert - g_x).view(-1).norm().item()
        H = diff_norm / eps
        hessian_proxy.append(H)

    return hessian_proxy


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions & checkpoints
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_SPECS = [
    {
        "name": "kan_f32",
        "class": KANEnergyModel,
        "kwargs": {"n_filters": 32, "filter_size": 5, "kan_hidden": [96, 16], "n_channels": 3},
        "ckpt": ROOT / "outputs" / "cifar10" / "kan_ebm_f32.pt",
    },
    {
        "name": "kan_small",
        "class": KANEnergyModel,
        "kwargs": {"n_filters": 16, "filter_size": 5, "kan_hidden": [48, 16], "n_channels": 3},
        "ckpt": ROOT / "outputs" / "finalization" / "kstar_param_scaling" / "kan_small.pt",
    },
    {
        "name": "conv_mlp_gelu",
        "class": ConvSmoothMLPEBM,
        "kwargs": {"n_filters": 16, "filter_size": 5, "mlp_hidden": 160, "n_channels": 3, "activation": "gelu"},
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_gelu.pt",
    },
    {
        "name": "conv_mlp_relu",
        "class": ConvSmoothMLPEBM,
        "kwargs": {"n_filters": 16, "filter_size": 5, "mlp_hidden": 160, "n_channels": 3, "activation": "relu"},
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_relu.pt",
    },
    {
        "name": "conv_mlp_silu",
        "class": ConvSmoothMLPEBM,
        "kwargs": {"n_filters": 16, "filter_size": 5, "mlp_hidden": 160, "n_channels": 3, "activation": "silu"},
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_silu.pt",
    },
    {
        "name": "mlp_ebm",
        "class": MLPEnergyModel,
        "kwargs": {"img_size": 64, "n_channels": 3, "hidden": 256},
        "ckpt": ROOT / "outputs" / "finalization" / "celeba_fair" / "mlp_ebm_fair.pt",
        "upsample_to": 64,
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════
json_path = OUT_DIR / "gradient_smoothness.json"
if json_path.exists():
    with open(json_path, 'r') as f:
        results = json.load(f)
    print(f"Loaded existing results from {json_path}")
else:
    results = {"models": {}}

for spec in MODEL_SPECS:
    name = spec["name"]
    if name in results["models"]:
        print(f"\n=== Skipping {name} (already in results) ===")
        continue
    print(f"\n=== Processing {name} ===")

    # Load model
    model = spec["class"](**spec["kwargs"]).to(DEVICE)
    try:
        state = torch.load(spec["ckpt"], map_location=DEVICE)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {spec['ckpt']}")
    except Exception as e:
        print(f"  ERROR loading checkpoint for {name}: {e}")
        continue

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    upsample = spec.get("upsample_to")
    # Part A
    print("  Computing Lipschitz proxy and gradient norms ...")
    part_a = measure_lipschitz_and_grad_norms(model, CLEAN_IMGS_32, n_pert=16, eps=0.01, upsample_to=upsample)

    # Part B
    print("  Computing Hessian proxy ...")
    hessian = measure_hessian_proxy(model, CLEAN_IMGS_16, eps=0.01, upsample_to=upsample)

    results["models"][name] = {
        "lipschitz_proxy": part_a["lipschitz_proxy"],
        "gradient_norm_clean": part_a["gradient_norm_clean"],
        "gradient_norm_noisy_0.1": part_a["gradient_norm_noisy_0.1"],
        "gradient_norm_noisy_0.2": part_a["gradient_norm_noisy_0.2"],
        "hessian_proxy": hessian,
    }

    # Cleanup
    del model, state
    torch.cuda.empty_cache()
    print(f"  Done. Lipschitz values: {len(part_a['lipschitz_proxy'])}, Hessian values: {len(hessian)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save JSON
# ═══════════════════════════════════════════════════════════════════════════════
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {json_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(results["models"].keys())
labels = [n.replace('_', '\n') for n in model_names]

# Left: Lipschitz proxy violin plot
lipschitz_data = [results["models"][n]["lipschitz_proxy"] for n in model_names]
parts1 = axes[0].violinplot(lipschitz_data, positions=range(len(model_names)),
                             showmeans=True, showmedians=True)
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(labels, fontsize=8)
axes[0].set_ylabel("Local Lipschitz constant L")
axes[0].set_title("Gradient Lipschitz Proxy")
axes[0].set_yscale('log')
axes[0].grid(True, ls='--', alpha=0.3)

# Right: Hessian proxy violin plot
hessian_data = [results["models"][n]["hessian_proxy"] for n in model_names]
parts2 = axes[1].violinplot(hessian_data, positions=range(len(model_names)),
                             showmeans=True, showmedians=True)
axes[1].set_xticks(range(len(model_names)))
axes[1].set_xticklabels(labels, fontsize=8)
axes[1].set_ylabel("Hessian diagonal proxy H")
axes[1].set_title("Finite-Difference Hessian Proxy")
axes[1].set_yscale('log')
axes[1].grid(True, ls='--', alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "smoothness_figures.png"
fig.savefig(fig_path, bbox_inches='tight', dpi=150)
print(f"Saved figure to {fig_path}")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name in model_names:
    lip = np.array(results["models"][name]["lipschitz_proxy"])
    hes = np.array(results["models"][name]["hessian_proxy"])
    print(f"{name:20s}  Lipschitz  median={np.median(lip):.4f}  mean={lip.mean():.4f}")
    print(f"{'':20s}  Hessian    median={np.median(hes):.4f}  mean={hes.mean():.4f}")
