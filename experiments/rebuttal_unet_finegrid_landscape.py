"""
rebuttal_unet_finegrid_landscape.py
====================================
Two GPU jobs in one script for U-Net EBM:
  1. Fine-grid K* on 500 CIFAR-10 test images (sample-size parity with the
     ConvMLP fine-grid runs).
  2. Landscape geometry (basin radius + barrier) on the U-Net checkpoint, so
     Table 3 in the paper is complete.

Outputs:
  outputs/fine_grid_kstar/fine_grid_unet.json
  outputs/landscape_geometry/basin_geometry.json   (in-place merge of unet entry)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_unet_ebm import UNetEBM  # noqa: E402
import landscape_geometry as lg  # noqa: E402

OUT_KSTAR = ROOT / 'outputs' / 'fine_grid_kstar' / 'fine_grid_unet.json'
OUT_LAND = ROOT / 'outputs' / 'landscape_geometry' / 'basin_geometry.json'
CKPT = ROOT / 'outputs' / 'unet_ebm' / 'unet_ebm.pt'

SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]
K_MAX = 30
N_IMAGES = 500
DT_INIT = 0.05
DT_DECAY = 0.97
SEED = 42


def psnr(clean, recon):
    mse = F.mse_loss(recon, clean).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(4.0 / mse)


def kstar_per_image(model, batch_clean, sigma, device, K_max=K_MAX, dt=DT_INIT,
                    dt_decay=DT_DECAY):
    """Return list of K* (argmax_K PSNR_at_K) per image."""
    n = batch_clean.size(0)
    torch.manual_seed(SEED)
    xn = batch_clean + torch.randn_like(batch_clean) * sigma
    psnr_at_K = np.zeros((n, K_max), dtype=np.float32)
    u = xn.clone()
    step = dt
    for k in range(1, K_max + 1):
        u = u.detach().requires_grad_(True)
        E = model.energy(u)
        E_scalar = E.sum() if E.ndim > 0 else E
        g = torch.autograd.grad(E_scalar, u)[0].detach().clamp(-1, 1)
        u = (u - step * g).detach()
        step *= dt_decay
        # Compute PSNR per image
        diff = (u.clamp(-1, 1) - batch_clean) ** 2
        mse = diff.view(n, -1).mean(dim=1)
        psnr_at_K[:, k - 1] = (10.0 * torch.log10(4.0 / mse.clamp(min=1e-12))
                                ).cpu().numpy()
    # K* per image (argmax over k)
    kstar = (psnr_at_K.argmax(axis=1) + 1).astype(int).tolist()
    mean_psnr = psnr_at_K.mean(axis=0).tolist()
    return kstar, mean_psnr


def fit_powerlaw(sigmas, Ks):
    xs = np.log(np.asarray(sigmas, float))
    ys = np.log(np.asarray(Ks, float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    if not CKPT.exists():
        raise SystemExit(f"U-Net checkpoint missing: {CKPT}")

    # ─── Load U-Net ───────────────────────────────────────────────────────────
    model = UNetEBM(base_ch=16, n_channels=3).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[unet] loaded {n_params:,} params from {CKPT.name}")

    # ─── Load CIFAR-10 test set ────────────────────────────────────────────────
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    eval_batch = torch.stack([test_ds[i][0] for i in range(N_IMAGES)]).to(device)
    print(f"[data] {N_IMAGES} CIFAR-10 test images")

    # ─── Fine-grid K* sweep ────────────────────────────────────────────────────
    print(f"\n[1/2] Fine-grid K* sweep ({N_IMAGES} images, K_max={K_MAX}, "
          f"sigmas={SIGMAS})")
    record = {
        'n_images': N_IMAGES, 'K_max': K_MAX, 'sigmas': SIGMAS,
        'n_params': n_params, 'per_sigma': {},
    }
    Ks_mean = []
    for sigma in SIGMAS:
        print(f"  sigma={sigma}", flush=True)
        kstar_list, psnr_per_K = kstar_per_image(
            model, eval_batch, sigma, device, K_max=K_MAX, dt=DT_INIT,
            dt_decay=DT_DECAY,
        )
        kmean = float(np.mean(kstar_list))
        kstd = float(np.std(kstar_list, ddof=1))
        Ks_mean.append(kmean)
        record['per_sigma'][f"{sigma:.3f}"] = {
            'kstar_mean': kmean,
            'kstar_std': kstd,
            'kstar_per_image': kstar_list,
            'mean_psnr_per_K': psnr_per_K,
        }
        print(f"    K* per image: mean={kmean:.3f}  std={kstd:.3f}  "
              f"max PSNR={max(psnr_per_K):.2f} at K={int(np.argmax(psnr_per_K))+1}")
    a, b, r2 = fit_powerlaw(SIGMAS, Ks_mean)
    record['power_law'] = {'alpha': b, 'C': math.exp(a), 'R2': r2}
    print(f"\n  U-Net power law: alpha={b:.3f}  C={math.exp(a):.2f}  R^2={r2:.4f}")

    OUT_KSTAR.parent.mkdir(parents=True, exist_ok=True)
    OUT_KSTAR.write_text(json.dumps(record, indent=2))
    print(f"  Saved {OUT_KSTAR}")

    # ─── Landscape geometry on U-Net ──────────────────────────────────────────
    print("\n[2/2] Landscape geometry: basin radius + barrier")
    ds_full = lg.load_cifar10()
    n_imgs = max(lg.N_IMAGES_BASIN, lg.N_IMAGES_BARRIER)
    images = [ds_full[i][0] for i in range(n_imgs)]

    print("  -- Basin radius --")
    basin = lg.measure_basin_radius(model, images)
    print("  -- Energy barrier --")
    with torch.no_grad():
        barrier = lg.measure_barrier(model, images)

    # Merge into existing basin_geometry.json
    OUT_LAND.parent.mkdir(parents=True, exist_ok=True)
    if OUT_LAND.exists():
        existing = json.loads(OUT_LAND.read_text())
    else:
        existing = {'models': {}}
    existing['models']['unet_ebm'] = {
        'basin_radius': basin,
        'barrier': barrier,
    }
    OUT_LAND.write_text(json.dumps(existing, indent=2))
    print(f"  basin_radius_threshold = {basin['basin_radius_threshold']}")
    print(f"  mean barrier height = {float(np.mean(barrier['barrier_heights'])):.1f}")
    print(f"  Updated {OUT_LAND}")
    print(f"\n  Variants now in landscape table: {sorted(existing['models'].keys())}")


if __name__ == "__main__":
    main()
