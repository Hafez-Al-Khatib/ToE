"""
GO/NO-GO test for the spectral-shrinkage theory  (theory/THEORY_NOTE.md).
========================================================================

Prediction:  alpha = 2*gamma/beta = 2*kappa, where in the DATA-COVARIANCE
eigenbasis {v_k, s2_k} the learned curvature couples to signal power as
    lambda_k  =  v_k^T H v_k  ~  (s2_k)^(-kappa).
We measure kappa per architecture (curvature vs signal-power slope, in the data
basis, via Hessian-vector products at clean minima) and compare 2*kappa to the
inference-scaling exponent alpha measured in the paper.

Assumption-light: we measure the DIAGONAL of H in the data basis (the per-mode
curvature the GD dynamics actually feel), not eigenvalues -- so we do NOT assume
H diagonalizes in the data basis.
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from torchvision import datasets, transforms          # noqa: E402
from exp_cifar10 import KANEnergyModel                 # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM        # noqa: E402
from exp_unet_ebm import UNetEBM                         # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paper_v4 measured inference-scaling exponents
ALPHA_MEASURED = {
    'unet':          1.434,
    'kan_f32':       1.365,
    'kan_small':     1.358,
    'conv_mlp_gelu': 1.132,
    'conv_mlp_silu': 1.126,
    'conv_mlp_tanh': 1.136,
    'conv_mlp_relu': 1.166,   # piecewise-linear: ~zero curvature (expected outlier)
}

MODELS = {
    'unet': dict(cls=UNetEBM, kw=dict(base_ch=16),
                 ckpt=ROOT/'outputs'/'unet_ebm'/'unet_ebm.pt'),
    'kan_f32': dict(cls=KANEnergyModel,
                    kw=dict(n_filters=32, filter_size=5, kan_hidden=[96, 16], n_channels=3),
                    ckpt=ROOT/'outputs'/'cifar10'/'kan_ebm_f32.pt'),
    'kan_small': dict(cls=KANEnergyModel,
                      kw=dict(n_filters=16, filter_size=5, kan_hidden=[48, 16], n_channels=3),
                      ckpt=ROOT/'outputs'/'finalization'/'kstar_param_scaling'/'kan_small.pt'),
    'conv_mlp_gelu': dict(cls=ConvSmoothMLPEBM,
                          kw=dict(n_filters=16, filter_size=5, mlp_hidden=160, n_channels=3, activation='gelu'),
                          ckpt=ROOT/'outputs'/'finalization'/'rebuttal'/'conv_mlp_gelu.pt'),
    'conv_mlp_silu': dict(cls=ConvSmoothMLPEBM,
                          kw=dict(n_filters=16, filter_size=5, mlp_hidden=160, n_channels=3, activation='silu'),
                          ckpt=ROOT/'outputs'/'finalization'/'rebuttal'/'conv_mlp_silu.pt'),
    'conv_mlp_tanh': dict(cls=ConvSmoothMLPEBM,
                          kw=dict(n_filters=16, filter_size=5, mlp_hidden=160, n_channels=3, activation='tanh'),
                          ckpt=ROOT/'outputs'/'finalization'/'rebuttal'/'conv_mlp_tanh.pt'),
    'conv_mlp_relu': dict(cls=ConvSmoothMLPEBM,
                          kw=dict(n_filters=16, filter_size=5, mlp_hidden=160, n_channels=3, activation='relu'),
                          ckpt=ROOT/'outputs'/'finalization'/'rebuttal'/'conv_mlp_relu.pt'),
}

N_COV_IMAGES = 4000   # images to estimate the data covariance
N_ANCHORS    = 8      # clean test images used as curvature anchors
N_MODES      = 80     # log-spaced data eigenmodes probed


def load_cifar_test():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return datasets.CIFAR10(str(ROOT / 'data'), train=False, download=True, transform=tf)


def data_eigenbasis(test_ds, n=N_COV_IMAGES):
    X = torch.stack([test_ds[i][0] for i in range(n)]).view(n, -1)  # (n, 3072)
    X = X - X.mean(0, keepdim=True)
    cov = (X.T @ X) / (n - 1)                       # (3072, 3072)
    s2, V = torch.linalg.eigh(cov)                  # ascending
    idx = torch.argsort(s2, descending=True)        # rank 1 = most signal
    return s2[idx].clamp(min=1e-12), V[:, idx]      # eigenvalues, eigenvectors (cols)


def fit_beta(s2):
    """Signal power-law exponent: s2_k ~ k^-beta over a clean rank range."""
    k = np.arange(1, len(s2) + 1)
    lo, hi = 2, min(2000, len(s2) - 50)             # avoid DC mode and numerical tail
    x, y = np.log(k[lo:hi]), np.log(s2.cpu().numpy()[lo:hi])
    slope = np.polyfit(x, y, 1)[0]
    return -slope


def grad_E(model, x):
    xi = x.detach().requires_grad_(True)
    E = model.energy(xi)
    if E.ndim > 0:
        E = E.sum()
    return torch.autograd.grad(E, xi)[0].detach()


def operating_point(model, clean, sigma=0.15, K=10, dt=0.05, decay=0.97):
    """Denoise from a noisy init to the mid-descent operating point (anchor)."""
    u = clean + torch.randn_like(clean) * sigma
    step = dt
    for _ in range(K):
        u = (u - step * grad_E(model, u)).detach()
        step *= decay
    return u.detach()


def curvature_along(model, x_batch, v):
    """Per-image Rayleigh curvature lambda = v^T H v at anchors x_batch (||v||=1)."""
    vb = v.expand_as(x_batch)
    xi = x_batch.detach().requires_grad_(True)
    E = model.energy(xi)
    if E.ndim > 0:
        E = E.sum()
    g = torch.autograd.grad(E, xi, create_graph=True)[0]
    Hv = torch.autograd.grad((g * vb).sum(), xi)[0]
    return (vb * Hv).sum(dim=[1, 2, 3]).detach()      # (B,)


def measure_kappa(model, anchors, V, mode_idx):
    """lambda_k averaged over anchors for each probed data-mode k; fit kappa."""
    lam = []
    for k in mode_idx:
        v = V[:, k].view(1, *anchors.shape[1:]).to(DEVICE)
        lam_k = curvature_along(model, anchors, v).cpu().numpy()
        lam.append(np.median(lam_k))                  # robust over anchors
    return np.array(lam)


def main():
    print(f"Device: {DEVICE}")
    test_ds = load_cifar_test()

    print("Building data covariance eigenbasis ...")
    s2, V = data_eigenbasis(test_ds)
    beta = fit_beta(s2)
    print(f"  data signal exponent  beta = {beta:.3f}   (s2_k ~ k^-beta)")

    D = s2.numel()
    mode_idx = np.unique(np.geomspace(1, min(2900, D - 1), N_MODES).astype(int))
    s2_modes = s2.cpu().numpy()[mode_idx]

    clean_anchors = torch.stack([test_ds[i][0] for i in range(N_ANCHORS)]).to(DEVICE)

    def fit_k(lam):
        pos = lam > 0
        if pos.sum() < 5:
            return float('nan'), float('nan'), float(np.mean(~pos))
        x, y = np.log(s2_modes[pos]), np.log(lam[pos])
        slope, intercept = np.polyfit(x, y, 1)
        yh = intercept + slope * x
        r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
        return -slope, r2, float(np.mean(~pos))

    print(f"\n{'model':>14} | {'k_clean':>7} {'k_traj':>7} | "
          f"{'2k_traj':>7} | {'alpha_meas':>10} | {'R2_traj':>7}")
    print("  " + "-" * 64)

    rows = []
    for name, cfg in MODELS.items():
        if not cfg['ckpt'].exists():
            print(f"{name:>14} | checkpoint missing")
            continue
        try:
            model = cfg['cls'](**cfg['kw']).to(DEVICE)
            state = torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            print(f"{name:>14} | LOAD FAILED: {str(e)[:50]}")
            continue

        k_clean, _, _ = fit_k(measure_kappa(model, clean_anchors, V, mode_idx))
        traj = operating_point(model, clean_anchors)
        k_traj, r2_traj, _ = fit_k(measure_kappa(model, traj, V, mode_idx))

        a_meas = ALPHA_MEASURED[name]
        rows.append((name, k_clean, k_traj, 2 * k_traj, a_meas, r2_traj))
        print(f"{name:>14} | {k_clean:>7.3f} {k_traj:>7.3f} | "
              f"{2*k_traj:>7.3f} | {a_meas:>10.3f} | {r2_traj:>7.3f}")
        del model

    smooth = [r for r in rows if r[0] != 'conv_mlp_relu' and np.isfinite(r[2])]
    if len(smooth) >= 3:
        ap = np.array([r[3] for r in smooth])   # 2*k_traj
        am = np.array([r[4] for r in smooth])   # alpha_meas
        print("\n  Trajectory operating point, smooth architectures (ReLU excluded):")
        print(f"    corr(2*k_traj, alpha_meas) = {np.corrcoef(ap, am)[0, 1]:.3f}")
        print(f"    mean |2*k_traj - alpha_meas| = {np.mean(np.abs(ap - am)):.3f}")
        order_pred = [r[0] for r in sorted(smooth, key=lambda r: -r[2])]
        order_meas = [r[0] for r in sorted(smooth, key=lambda r: -r[4])]
        print(f"    ranking by k_traj:     {order_pred}")
        print(f"    ranking by alpha_meas: {order_meas}")
        print(f"    rankings match: {order_pred == order_meas}")


if __name__ == '__main__':
    main()
