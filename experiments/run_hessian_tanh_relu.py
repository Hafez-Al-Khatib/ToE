"""
Run Hessian trajectory for Tanh and ReLU only, and append to existing JSON.
"""
import json
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

from exp_cifar10 import KANEnergyModel
from rebuttal_smooth_mlp import ConvSmoothMLPEBM
from exp_unet_ebm import UNetEBM

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_PATH = ROOT / 'outputs' / 'tier1_trajectory_geometry' / 'trajectory_geometry.json'

SIGMA = 0.20
K_MAX = 30
DT_INIT = 0.05
DT_DECAY = 0.97
N_IMAGES = 8
HESSIAN_K_GRID = [1, 5, 10, 20]
LANCZOS_ITERS = 20

MODELS = {
    'conv_mlp_tanh': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'tanh'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_tanh.pt',
        'observed_alpha': 1.136,
    },
    'conv_mlp_relu': {
        'cls': ConvSmoothMLPEBM,
        'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'relu'},
        'ckpt': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_relu.pt',
        'observed_alpha': 1.166,
    },
}


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


def descend_recording(model, u0, K_max=K_MAX, dt=DT_INIT, dt_decay=DT_DECAY):
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
    xs = np.log(np.asarray(xs, dtype=float))
    ys = np.log(np.asarray(ys, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def radial_p_fit(model, states, energies):
    u_star_idx = int(np.argmin(energies))
    u_star = states[u_star_idx]
    ps = []
    for i in range(len(states)):
        if i == u_star_idx:
            continue
        r = float(torch.norm(states[i] - u_star).item())
        dE = float((energies[i] - energies[u_star_idx]))
        if r > 1e-6 and dE > 1e-6:
            p, _ = fit_loglog([r], [dE])
            ps.append(p)
    if not ps:
        return None, None
    ps = np.array(ps)
    p_mean = float(np.mean(ps))
    p_std = float(np.std(ps, ddof=1))
    p_median = float(np.median(ps))
    return p_mean, p_std


def run_one_model(name, cfg):
    print(f"\n=== {name} ===")
    model = cfg['cls'](**cfg['kwargs']).to(DEVICE)
    ckpt = torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Load test images
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    images = torch.stack([ds[i][0] for i in range(N_IMAGES)])

    hessian_results = {str(k): [] for k in HESSIAN_K_GRID}
    p_values = []

    for img_idx in range(N_IMAGES):
        x_clean = images[img_idx].unsqueeze(0).to(DEVICE)
        x_noisy = (x_clean + torch.randn_like(x_clean) * SIGMA).clamp(-1, 1)
        states, energies = descend_recording(model, x_noisy, K_max=K_MAX)

        # Hessian at trajectory points
        for k in HESSIAN_K_GRID:
            x_k = states[k]
            hvp = make_hvp(model, x_k)
            dim = int(np.prod(x_k.shape))
            lam = lanczos_lam_max(hvp, dim, n_iters=LANCZOS_ITERS)
            hessian_results[str(k)].append(lam)
            print(f"  img{img_idx} K={k}: lambda_max={lam:.2f}")

        # Radial p fit
        p_mean, p_std = radial_p_fit(model, states, energies)
        if p_mean is not None:
            p_values.append(p_mean)
            print(f"  img{img_idx}: p_mean={p_mean:.3f}")

    # Aggregate
    hessian_agg = {}
    for k in HESSIAN_K_GRID:
        vals = hessian_results[str(k)]
        hessian_agg[str(k)] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'n': len(vals),
        }

    p_mean = float(np.mean(p_values)) if p_values else None
    p_std = float(np.std(p_values, ddof=1)) if p_values else None
    alpha_pred = 2.0 - p_mean if p_mean else None

    return {
        'n_params': n_params,
        'p_fit': {
            'p_mean': p_mean,
            'p_std': p_std,
            'p_per_image': [float(v) for v in p_values],
        } if p_values else {},
        'alpha_predicted_from_p_mean': alpha_pred,
        'hessian_along_trajectory': hessian_agg,
        'observed_alpha': cfg['observed_alpha'],
    }


def main():
    # Load existing JSON
    if OUT_PATH.exists():
        data = json.load(open(OUT_PATH))
    else:
        data = {'config': {}, 'models': {}}

    # Run new models
    for name, cfg in MODELS.items():
        result = run_one_model(name, cfg)
        data['models'][name] = result

    # Save
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n[save] Updated {OUT_PATH}")

    # Print summary
    print("\n=== Hessian Summary ===")
    for name, result in data['models'].items():
        hess = result.get('hessian_along_trajectory', {})
        print(f"{name:20s} alpha={result.get('observed_alpha', '?'):.3f}  "
              f"λ@K=1={hess.get('1', {}).get('mean', 0):.1f}  "
              f"λ@K=20={hess.get('20', {}).get('mean', 0):.1f}")


if __name__ == '__main__':
    main()
