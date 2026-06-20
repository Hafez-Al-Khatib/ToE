"""
tier1_trajectory_geometry.py
=============================
Tier 1.2 + 1.3 (combined): Hessian-along-trajectory + radial-p fit.

For each architecture, on N CIFAR-10 test images at sigma=0.20:
  1. Run gradient descent for K=30 steps, saving u(t).
  2. (Tier 1.2) At K in {1, 5, 10, 15, 20, 30}, compute lambda_max via Lanczos
     at the trajectory point.
  3. (Tier 1.3) Take u* = argmin_t E(u(t)) (the lowest-energy point on the
     trajectory). Along the trajectory, define r_t = ||u(t) - u*|| and
     DeltaE_t = E(u(t)) - E(u*). Fit log(DeltaE) = log A + p log r.
     Predicted alpha_pred = 2 - p. Compare to observed alpha.

Outputs:
  outputs/tier1_trajectory_geometry/trajectory_geometry.json
  outputs/tier1_trajectory_geometry/trajectory_geometry.txt
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel  # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM  # noqa: E402
from exp_unet_ebm import UNetEBM  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = ROOT / 'outputs' / 'tier1_trajectory_geometry'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
SIGMA = 0.20
K_MAX = 30
DT_INIT = 0.05
DT_DECAY = 0.97
N_IMAGES = 8           # number of CIFAR-10 test images (reduced for CPU)
HESSIAN_K_GRID = [1, 5, 10, 20]  # trajectory points for Lanczos
LANCZOS_ITERS = 20     # fewer for speed on CPU

MODELS = {
    'kan_f32': {
        'cls': KANEnergyModel,
        'kwargs': {'n_filters': 32, 'filter_size': 5, 'kan_hidden': [96, 16], 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt',
        'observed_alpha': 1.3651,
    },
    'kan_small': {
        'cls': KANEnergyModel,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'kan_hidden': [48, 16], 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'kstar_param_scaling' / 'kan_small.pt',
        'observed_alpha': 1.3651,  # same family; we'll rerun fits below
    },
    'conv_mlp_gelu': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'gelu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_gelu.pt',
        'observed_alpha': 1.132,
    },
    'conv_mlp_silu': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'silu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_silu.pt',
        'observed_alpha': 1.126,
    },
    'unet_ebm': {
        'cls': UNetEBM,
        'kwargs': {'base_ch': 16, 'n_channels': 3},
        'ckpt': ROOT / 'outputs' / 'unet_ebm' / 'unet_ebm.pt',
        'observed_alpha': 1.434,
    },
}


# ─── Lanczos (exact-form HVP) ─────────────────────────────────────────────────
def make_hvp(model, x_anchor):
    x_anchor = x_anchor.detach().to(DEVICE)
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


def lanczos_lam_max(hvp_fn, dim, n_iters=LANCZOS_ITERS, dtype=torch.float32):
    v = torch.randn(dim, device=DEVICE, dtype=dtype)
    v = v / v.norm()
    alphas, betas = [], []
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0
    for _ in range(n_iters):
        w = hvp_fn(v)
        alpha = torch.dot(w, v).item()
        w = w - alpha * v - beta_prev * v_prev
        w = w - torch.dot(w, v) * v
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
    return float(np.linalg.eigvalsh(T).max())


# ─── Descent trajectory ───────────────────────────────────────────────────────
def descend_recording(model, u0, K_max=K_MAX, dt=DT_INIT, dt_decay=DT_DECAY):
    """Returns list of trajectory states (length K_max+1: u0, u1, ..., uK_max)
    and per-step energies."""
    states = [u0.detach().clone()]
    energies = [float(model.energy(u0).sum().item())]
    u = u0.clone()
    step = dt
    for _ in range(K_max):
        ui = u.detach().requires_grad_(True)
        E = model.energy(ui)
        Esum = E.sum() if E.ndim > 0 else E
        g = torch.autograd.grad(Esum, ui)[0].detach().clamp(-1.0, 1.0)
        u = (u - step * g).detach()
        states.append(u.clone())
        with torch.no_grad():
            energies.append(float(model.energy(u).sum().item()))
        step *= dt_decay
    return states, energies


def fit_loglog(xs, ys):
    """Fit y = A x^p in log-log."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    mask = (xs > 1e-8) & (ys > 1e-8) & np.isfinite(np.log(xs)) & np.isfinite(np.log(ys))
    if mask.sum() < 3:
        return None, None, None
    lx, ly = np.log(xs[mask]), np.log(ys[mask])
    p, log_A = np.polyfit(lx, ly, 1)
    yh = log_A + p * lx
    ss_res = float(np.sum((ly - yh) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(p), float(math.exp(log_A)), float(r2)


# ─── Per-model analysis ───────────────────────────────────────────────────────
def analyze_model(model, name, images_clean):
    print(f"\n=== {name} ===")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    per_image_p = []
    per_image_p_r2 = []
    per_image_lam_max = {k: [] for k in HESSIAN_K_GRID}

    for img_idx, x_clean in enumerate(images_clean):
        x_clean = x_clean.unsqueeze(0).to(DEVICE)
        torch.manual_seed(1000 + img_idx)
        x_noisy = (x_clean + torch.randn_like(x_clean) * SIGMA).clamp(-1, 1)
        states, energies = descend_recording(model, x_noisy)

        # ── Tier 1.3: pick u* as min-energy state along trajectory ──
        e_arr = np.array(energies)
        t_min = int(np.argmin(e_arr))
        u_star = states[t_min]
        E_star = energies[t_min]

        # Compute r_t and DeltaE_t along trajectory (excluding u*)
        rs, dEs = [], []
        for t, u_t in enumerate(states):
            if t == t_min:
                continue
            r = float((u_t - u_star).flatten().norm().item())
            dE = energies[t] - E_star
            if dE > 1e-7 and r > 1e-7:
                rs.append(r)
                dEs.append(dE)

        if len(rs) >= 4:
            p, A, r2 = fit_loglog(rs, dEs)
            if p is not None and 0.05 < p < 4.0:
                per_image_p.append(p)
                per_image_p_r2.append(r2)

        # ── Tier 1.2: Hessian lambda_max along trajectory ──
        for k in HESSIAN_K_GRID:
            if k >= len(states):
                continue
            u_k = states[k]
            dim = int(np.prod(u_k.shape))
            try:
                hvp_fn = make_hvp(model, u_k)
                lam = lanczos_lam_max(hvp_fn, dim)
                per_image_lam_max[k].append(lam)
            except Exception as e:
                print(f"  [warn] Lanczos failed at img {img_idx} K={k}: {e}")

        if (img_idx + 1) % 4 == 0:
            print(f"  [{img_idx + 1}/{len(images_clean)}] images processed")

    # ── Aggregate ──
    p_mean = float(np.mean(per_image_p)) if per_image_p else float('nan')
    p_std = float(np.std(per_image_p, ddof=1)) if len(per_image_p) > 1 else 0.0
    p_median = float(np.median(per_image_p)) if per_image_p else float('nan')
    alpha_predicted = 2.0 - p_mean
    alpha_predicted_median = 2.0 - p_median

    lam_max_summary = {}
    for k in HESSIAN_K_GRID:
        lst = per_image_lam_max[k]
        if lst:
            lam_max_summary[k] = {
                'mean': float(np.mean(lst)),
                'std': float(np.std(lst, ddof=1)) if len(lst) > 1 else 0.0,
                'n': len(lst),
            }
        else:
            lam_max_summary[k] = {'mean': float('nan'), 'std': 0.0, 'n': 0}

    print(f"  p (mean over {len(per_image_p)} imgs): {p_mean:.3f} +/- {p_std:.3f}")
    print(f"  p (median): {p_median:.3f}")
    print(f"  alpha_predicted = 2 - p_mean   = {alpha_predicted:.3f}")
    print(f"  alpha_predicted = 2 - p_median = {alpha_predicted_median:.3f}")
    print(f"  Hessian lambda_max along trajectory:")
    for k in HESSIAN_K_GRID:
        s = lam_max_summary[k]
        print(f"    K={k:>3d}:  lam_max = {s['mean']:.2f}  (std {s['std']:.2f}, n={s['n']})")

    return {
        'n_params': n_params,
        'p_fit': {
            'p_mean': p_mean, 'p_std': p_std, 'p_median': p_median,
            'p_per_image': per_image_p,
            'r2_per_image': per_image_p_r2,
            'r2_mean': float(np.mean(per_image_p_r2)) if per_image_p_r2 else float('nan'),
        },
        'alpha_predicted_from_p_mean': alpha_predicted,
        'alpha_predicted_from_p_median': alpha_predicted_median,
        'hessian_along_trajectory': lam_max_summary,
    }


def main():
    # Load CIFAR-10 test images
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = datasets.CIFAR10(str(ROOT / 'data'), train=False, download=True, transform=tf)
    images = [ds[i][0] for i in range(N_IMAGES)]

    record = {
        'config': {
            'sigma': SIGMA, 'K_max': K_MAX, 'n_images': N_IMAGES,
            'hessian_K_grid': HESSIAN_K_GRID, 'lanczos_iters': LANCZOS_ITERS,
            'dt_init': DT_INIT, 'dt_decay': DT_DECAY,
        },
        'models': {},
    }

    for name, cfg in MODELS.items():
        if not cfg['ckpt'].exists():
            print(f"[skip] checkpoint missing: {cfg['ckpt']}")
            continue
        model = cfg['cls'](**cfg['kwargs']).to(DEVICE)
        state = torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        result = analyze_model(model, name, images)
        result['observed_alpha'] = cfg['observed_alpha']
        result['delta_alpha_pred_minus_obs'] = result['alpha_predicted_from_p_mean'] - cfg['observed_alpha']
        record['models'][name] = result
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ─── Save ──────────────────────────────────────────────────────────────
    out_json = OUT_DIR / 'trajectory_geometry.json'
    out_json.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {out_json}")

    # ─── Headline summary table ────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"{'Model':<16} {'params':>9} {'p_mean':>8} {'p_med':>7} {'alpha_pred(mean)':>18} {'alpha_obs':>11} {'|delta|':>9}")
    print("-" * 92)
    lines = [
        "Tier 1.3: Empirical p-fit and toy-model prediction of alpha = 2 - p",
        "=" * 92,
        f"{'Model':<16} {'params':>9} {'p_mean':>8} {'p_med':>7} {'alpha_pred(mean)':>18} {'alpha_obs':>11} {'|delta|':>9}",
        "-" * 92,
    ]
    for name, r in record['models'].items():
        pm = r['p_fit']['p_mean']
        pmed = r['p_fit']['p_median']
        ap = r['alpha_predicted_from_p_mean']
        ao = r['observed_alpha']
        d = abs(ap - ao)
        line = f"{name:<16} {r['n_params']:>9,} {pm:>8.3f} {pmed:>7.3f} {ap:>18.3f} {ao:>11.3f} {d:>9.3f}"
        print(line)
        lines.append(line)
    lines.append("")
    lines.append("Tier 1.2: Hessian lambda_max along descent trajectory (sigma=0.20)")
    lines.append("=" * 92)
    header = f"{'Model':<16}" + "".join(f"  K={k:<3d}" for k in HESSIAN_K_GRID)
    print("\n" + header)
    print("-" * 92)
    lines.append(header)
    lines.append("-" * 92)
    for name, r in record['models'].items():
        row = f"{name:<16}"
        for k in HESSIAN_K_GRID:
            v = r['hessian_along_trajectory'][str(k)]['mean'] if str(k) in r['hessian_along_trajectory'] else r['hessian_along_trajectory'][k]['mean']
            row += f"  {v:>5.1f}"
        print(row)
        lines.append(row)

    out_txt = OUT_DIR / 'trajectory_geometry.txt'
    out_txt.write_text("\n".join(lines))
    print(f"\nSaved {out_txt}")


if __name__ == "__main__":
    main()
