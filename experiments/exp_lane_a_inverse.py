"""
exp_lane_a_inverse.py
=====================
Lane A: "One Energy, Many Inverse Problems" on CIFAR-10.

Single CIFAR-10 KAN-EBM checkpoint (no retraining, no fine-tuning) is used to
solve five distinct inverse problems via gradient descent on E_theta(u) +
optional data fidelity.

Tasks:
  1. Denoise          -- additive Gaussian, sigma=0.1
  2. Inpaint (random) -- 50% random pixel mask, hard projection
  3. Inpaint (box)    -- centered 16x16 box mask (25% area), hard projection
  4. Super-Res 4x     -- bicubic LR -> HR, soft data fidelity to LR
  5. Deblur           -- Gaussian blur sigma=1.0 forward op, soft fidelity
  6. JPEG-AR          -- JPEG QF=10 artifacts, prior-only refinement

Inference dynamics (matched across all tasks for clean attribution):
  - Decaying step size: eta_t = 0.05 * 0.97^t
  - Gradient clamp [-1, 1]
  - K in {0, 1, 2, 5, 10, 20}
  - No TV regularization (only the learned energy + data fidelity)

Output: outputs/lane_a/results.json with per-task PSNR-vs-K curves.
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_cifar10 import KANEnergyModel  # noqa: E402

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Forward operators
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_kernel_2d(ksize: int, sigma: float, n_channels: int = 3) -> torch.Tensor:
    """Build a per-channel separable Gaussian kernel of shape (C, 1, ksize, ksize)."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    return k2.expand(n_channels, 1, ksize, ksize).contiguous()


def gaussian_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur with per-channel kernel; same padding."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


def jpeg_compress(x_minus1_to_1: torch.Tensor, qf: int = 10) -> torch.Tensor:
    """
    Round-trip through real JPEG at quality qf.
    Input: tensor in [-1, 1]; output: same shape, JPEG-compressed values in [-1, 1].
    Non-differentiable (we only need this as a fixed forward op for evaluation).
    """
    out = []
    for img in x_minus1_to_1:
        # (C, H, W) in [-1, 1] -> uint8 [0, 255]
        arr = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()  # H, W, C
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
            mode = 'L'
        else:
            mode = 'RGB'
        buf = io.BytesIO()
        Image.fromarray(arr, mode=mode).save(buf, format='JPEG', quality=int(qf))
        buf.seek(0)
        comp = np.array(Image.open(buf).convert(mode))
        if comp.ndim == 2:
            comp = comp[:, :, None]
        comp = torch.from_numpy(comp).permute(2, 0, 1).float() / 127.5 - 1.0
        out.append(comp)
    return torch.stack(out, dim=0).to(x_minus1_to_1.device)


# ──────────────────────────────────────────────────────────────────────────────
# Unified inverse-problem refinement
# ──────────────────────────────────────────────────────────────────────────────

def refine(model, u_init, n_steps, *,
           dt=0.05, dt_decay=0.97, grad_clamp=1.0,
           anchor_mask=None, anchor_vals=None,
           data_op=None, data_target=None, data_weight=0.0,
           bounds=(-1.0, 1.0)):
    """
    Generic gradient descent on E_theta(u) + (data_weight/2) * || data_op(u) - data_target ||^2.

    The energy gradient is clamped to [-grad_clamp, grad_clamp] for stability
    (matching the main paper's inference). The data-fidelity gradient is NOT
    clamped: it expresses a hard physical constraint and must be allowed to dominate.
    """
    u = u_init.clone().detach()
    step = dt
    for t in range(n_steps):
        # Energy gradient (clamped, as in the main paper)
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            eg = torch.autograd.grad(model.energy(ui), ui)[0].detach().clamp(-grad_clamp, grad_clamp)
        # Data-fidelity gradient (unclamped; mean reduction so weight is scale-free)
        dg = 0.0
        if data_op is not None and data_target is not None and data_weight > 0:
            with torch.enable_grad():
                ud = u.detach().requires_grad_(True)
                pred = data_op(ud)
                if pred.shape != data_target.shape:
                    raise RuntimeError(
                        f"data_op output {pred.shape} != target {data_target.shape}")
                # Sum reduction so the data-fidelity gradient is independent of
                # measurement-tensor size; data_weight then has the same scale
                # across all forward operators.
                dl = 0.5 * F.mse_loss(pred, data_target, reduction='sum')
                dg = data_weight * torch.autograd.grad(dl, ud)[0].detach()
        u = (u - step * (eg + dg)).detach()
        # Hard projection (inpainting)
        if anchor_mask is not None and anchor_vals is not None:
            u = torch.where(anchor_mask, anchor_vals, u)
        u = u.clamp(*bounds)
        step *= dt_decay
    return u


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(a.clamp(-1, 1), b).item()
    return 100.0 if mse < 1e-12 else 10.0 * math.log10(data_range ** 2 / mse)


# ──────────────────────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────────────────────

def task_denoise(model, x, K_list, sigma=0.1, **_):
    xn = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
    psnrs = {k: psnr(refine(model, xn, k), x) if k > 0 else psnr(xn, x) for k in K_list}
    return psnrs, psnr(xn, x)


def task_inpaint_random(model, x, K_list, mask_rate=0.5, **_):
    # Random pixel mask (broadcast over channels: same mask all channels)
    mask = (torch.rand_like(x[:, :1]) > mask_rate).float()
    mask = mask.expand_as(x)
    y = x * mask  # masked input
    u0 = torch.where(mask.bool(), x, torch.zeros_like(x))
    psnrs = {}
    for k in K_list:
        if k == 0:
            psnrs[k] = psnr(u0, x); continue
        u_hat = refine(model, u0, k, anchor_mask=mask.bool(), anchor_vals=x)
        psnrs[k] = psnr(u_hat, x)
    return psnrs, psnr(y, x)


def task_inpaint_box(model, x, K_list, box=8, **_):
    # Center box hole (default 8x8 = 6.25% area on 32x32; harder than random
    # mask because the missing region has no in-pixel anchors nearby).
    mask = torch.ones_like(x)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    half = box // 2
    mask[..., cy - half:cy + half, cx - half:cx + half] = 0
    y = x * mask
    u0 = torch.where(mask.bool(), x, torch.zeros_like(x))
    psnrs = {}
    for k in K_list:
        if k == 0:
            psnrs[k] = psnr(u0, x); continue
        u_hat = refine(model, u0, k, anchor_mask=mask.bool(), anchor_vals=x)
        psnrs[k] = psnr(u_hat, x)
    return psnrs, psnr(y, x)


def task_superres(model, x, K_list, scale=4, lambda_data=500.0, **_):
    # Forward: bicubic downsample by `scale`, then bicubic upsample for u_init
    lr = F.avg_pool2d(x, scale)
    u0 = F.interpolate(lr, scale_factor=scale, mode='bicubic', align_corners=False)
    # Data op: avg-pool downsample at the same scale
    def op(u):
        return F.avg_pool2d(u, scale)
    psnrs = {}
    for k in K_list:
        if k == 0:
            psnrs[k] = psnr(u0, x); continue
        u_hat = refine(model, u0, k, data_op=op, data_target=lr, data_weight=lambda_data)
        psnrs[k] = psnr(u_hat, x)
    return psnrs, psnr(u0, x)


def task_deblur(model, x, K_list, blur_sigma=1.0, ksize=7, lambda_data=50.0, **_):
    kernel = gaussian_kernel_2d(ksize, blur_sigma, n_channels=x.shape[1]).to(x.device)
    y = gaussian_blur(x, kernel)
    u0 = y.clone()
    def op(u):
        return gaussian_blur(u, kernel)
    psnrs = {}
    for k in K_list:
        if k == 0:
            psnrs[k] = psnr(u0, x); continue
        u_hat = refine(model, u0, k, data_op=op, data_target=y, data_weight=lambda_data)
        psnrs[k] = psnr(u_hat, x)
    return psnrs, psnr(u0, x)


def task_jpeg_ar(model, x, K_list, qf=10, **_):
    # Real JPEG roundtrip as the corruption; refine with prior only (no data term).
    y = jpeg_compress(x, qf=qf)
    u0 = y.clone()
    psnrs = {}
    for k in K_list:
        if k == 0:
            psnrs[k] = psnr(u0, x); continue
        u_hat = refine(model, u0, k)
        psnrs[k] = psnr(u_hat, x)
    return psnrs, psnr(u0, x)


TASKS = {
    'denoise':       task_denoise,
    'inpaint_random': task_inpaint_random,
    'inpaint_box':   task_inpaint_box,
    'superres_4x':   task_superres,
    'deblur_g1':     task_deblur,
    'jpeg_q10':      task_jpeg_ar,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_model(device):
    f32 = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
    f16 = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f16.pt'
    if f32.exists():
        m = KANEnergyModel(n_filters=32, filter_size=5, kan_hidden=[96, 16], n_channels=3).to(device)
        m.load_state_dict(torch.load(f32, map_location=device, weights_only=True))
        tag = 'f32 (~110K params)'
    elif f16.exists():
        m = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[48, 16], n_channels=3).to(device)
        m.load_state_dict(torch.load(f16, map_location=device, weights_only=True))
        tag = 'f16 (~32K params)'
    else:
        raise FileNotFoundError("No CIFAR-10 KAN-EBM checkpoint found.")
    m.eval()
    return m, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_test', type=int, default=50,
                    help='Number of CIFAR-10 test images (default: 50)')
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()),
                    help='Subset of tasks to run')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = torch.device('cuda') if (args.device == 'auto' and torch.cuda.is_available()) else torch.device('cpu')
    print(f"Device: {device}")

    model, tag = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded KAN-EBM {tag} = {n_params:,} parameters")

    K_list = [0, 1, 2, 5, 10, 20]
    print(f"K_list: {K_list}")
    print(f"Tasks:  {args.tasks}")
    print(f"n_test: {args.n_test}")

    # Data
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    results = {t: {str(k): [] for k in K_list} for t in args.tasks}
    corrupted = {t: [] for t in args.tasks}

    n_done = 0
    t0_total = time.time()
    for i, (x, _) in enumerate(loader):
        if n_done >= args.n_test:
            break
        x = x.to(device)
        for t_name in args.tasks:
            t0 = time.time()
            psnrs, corr = TASKS[t_name](model, x, K_list)
            for k, v in psnrs.items():
                results[t_name][str(k)].append(v)
            corrupted[t_name].append(corr)
            if i == 0:
                print(f"  [{t_name:14s}] sample 0: corrupted={corr:.2f}, "
                      f"K={K_list[-1]} -> {psnrs[K_list[-1]]:.2f} dB  ({time.time()-t0:.1f}s)")
        n_done += 1
        if n_done % 10 == 0:
            print(f"  Processed {n_done}/{args.n_test}  "
                  f"({(time.time()-t0_total)/60:.1f} min elapsed)")

    # Aggregate
    summary = {
        'n_params': n_params,
        'checkpoint_tag': tag,
        'n_test': args.n_test,
        'K_list': K_list,
        'tasks': {},
        'inference': {'dt': 0.05, 'dt_decay': 0.97, 'grad_clamp': 1.0,
                      'tv_weight': 0.0, 'note': 'energy-only (no TV) for clean attribution'},
        'wall_clock_s': round(time.time() - t0_total, 1),
    }
    for t_name in args.tasks:
        per_k = {str(k): float(np.mean(results[t_name][str(k)])) for k in K_list}
        per_k_std = {str(k): float(np.std(results[t_name][str(k)])) for k in K_list}
        peak_k = max(per_k, key=lambda k: per_k[k])
        summary['tasks'][t_name] = {
            'corrupted_mean': float(np.mean(corrupted[t_name])),
            'mean_psnr_per_K': per_k,
            'std_psnr_per_K': per_k_std,
            'peak_K': int(peak_k),
            'peak_psnr': per_k[peak_k],
            'gain_K0_to_peak': per_k[peak_k] - per_k['0'],
        }

    out_dir = ROOT / 'outputs' / 'lane_a'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'results.json'
    out_path.write_text(json.dumps(summary, indent=2))

    # Console table
    print("\n" + "=" * 80)
    print(f"LANE A: One Energy, Many Inverse Problems (CIFAR-10, {tag})")
    print("=" * 80)
    print(f"{'Task':<16s}  {'Corr':>6s}  " + "  ".join(f'K={k:>2d}' for k in K_list)
          + f"  {'Peak':>5s}  {'Gain':>6s}")
    print("-" * 80)
    for t_name in args.tasks:
        s = summary['tasks'][t_name]
        ks = '  '.join(f'{s["mean_psnr_per_K"][str(k)]:>5.2f}' for k in K_list)
        print(f"{t_name:<16s}  {s['corrupted_mean']:>6.2f}  {ks}  "
              f"K={s['peak_K']:>2d}  {s['gain_K0_to_peak']:+5.2f}")
    print("=" * 80)
    print(f"Wall-clock: {summary['wall_clock_s']:.1f}s = {summary['wall_clock_s']/60:.1f} min")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
