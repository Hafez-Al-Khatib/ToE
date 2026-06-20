"""
tier2_stl10_zeroshot.py
=======================
Tier 2: Zero-shot cross-dataset transfer to Tiny-ImageNet (no retraining).

Loads existing CIFAR-10 KAN-EBM checkpoint, evaluates fine-grid K* on
Tiny-ImageNet val images (64x64) downsampled to 32x32. If alpha matches
CIFAR's (~1.365), this strengthens the cross-dataset claim from 2 to 3
datasets without any new training -- evidence that alpha is an
architectural invariant of the model class, not an artefact of training
data distribution.

Outputs:
  outputs/tier2_stl10/tinyimagenet_zeroshot.json
  outputs/tier2_stl10/tinyimagenet_zeroshot.txt
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_scaled_eval_cifar10 import load_kan  # noqa: E402

OUT_DIR = ROOT / 'outputs' / 'tier2_stl10'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / 'tinyimagenet_zeroshot.json'
OUT_TXT = OUT_DIR / 'tinyimagenet_zeroshot.txt'
TINYIN_VAL = ROOT / 'data' / 'tiny-imagenet-200' / 'val'

CIFAR_CKPT = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'

SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]
K_MAX = 30
N_IMAGES = 200
DT_INIT = 0.05
DT_DECAY = 0.97
SEED = 42
B_BOOT = 10000


def kstar_per_image(model, batch_clean, sigma, K_max=K_MAX, dt=DT_INIT,
                    dt_decay=DT_DECAY):
    n = batch_clean.size(0)
    torch.manual_seed(SEED)
    xn = (batch_clean + torch.randn_like(batch_clean) * sigma).clamp(-1, 1)
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
        diff = (u.clamp(-1, 1) - batch_clean) ** 2
        mse = diff.view(n, -1).mean(dim=1)
        psnr_at_K[:, k - 1] = (10.0 * torch.log10(4.0 / mse.clamp(min=1e-12))
                                ).cpu().numpy()
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


def parametric_bootstrap_ci(sigmas, Ks, B=B_BOOT, rng_seed=20260506):
    a, b, _ = fit_powerlaw(sigmas, Ks)
    xs = np.log(np.asarray(sigmas, float))
    ys = np.log(np.asarray(Ks, float))
    yh = a + b * xs
    resid = ys - yh
    resid_c = resid - resid.mean()
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(B)
    for i in range(B):
        e = rng.choice(resid_c, size=len(xs), replace=True)
        s, _ = np.polyfit(xs, yh + e, 1)
        boot[i] = s
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    if not CIFAR_CKPT.exists():
        raise SystemExit(f"CIFAR KAN-EBM checkpoint missing: {CIFAR_CKPT}")

    # ─── Load CIFAR-trained KAN-EBM ──────────────────────────────────────────
    model = load_kan(CIFAR_CKPT, device, n_filters=32, kan_hidden=[96, 16])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CIFAR KAN-EBM ({n_params:,} params) loaded -- ZERO-SHOT to STL-10")

    # ─── Load Tiny-ImageNet val (64x64), downsample to 32x32 ─────────────────
    tf = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.Lambda(lambda img: img.convert('RGB')),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print(f"[data] Loading Tiny-ImageNet val from {TINYIN_VAL}")
    test_ds = torchvision.datasets.ImageFolder(str(TINYIN_VAL), transform=tf)
    rng_idx = np.random.default_rng(SEED).permutation(len(test_ds))[:N_IMAGES]
    images = torch.stack([test_ds[int(i)][0] for i in rng_idx]).to(device)
    print(f"[data] {N_IMAGES} Tiny-ImageNet val images at 32x32 (from {len(test_ds)} total)")

    # ─── Fine-grid K* sweep ──────────────────────────────────────────────────
    print(f"\nFine-grid K* sweep on Tiny-ImageNet (K_max={K_MAX}, sigmas={SIGMAS})")
    record = {
        'dataset': 'Tiny-ImageNet val (resize 64->32)',
        'protocol': 'ZERO-SHOT (CIFAR ckpt, no retraining)',
        'n_images': N_IMAGES, 'K_max': K_MAX, 'sigmas': SIGMAS,
        'n_params': n_params, 'per_sigma': {},
    }
    Ks_mean = []
    for sigma in SIGMAS:
        t0 = time.time()
        print(f"  sigma={sigma}", flush=True)
        kstar_list, psnr_per_K = kstar_per_image(model, images, sigma)
        kmean = float(np.mean(kstar_list))
        kstd = float(np.std(kstar_list, ddof=1))
        Ks_mean.append(kmean)
        record['per_sigma'][f"{sigma:.3f}"] = {
            'kstar_mean': kmean, 'kstar_std': kstd,
            'kstar_per_image': kstar_list,
            'mean_psnr_per_K': psnr_per_K,
        }
        print(f"    K* mean={kmean:.3f}  std={kstd:.3f}  "
              f"max PSNR={max(psnr_per_K):.2f} at K={int(np.argmax(psnr_per_K))+1}  "
              f"[{time.time()-t0:.1f}s]")

    a, b, r2 = fit_powerlaw(SIGMAS, Ks_mean)
    ci_lo, ci_hi, _ = parametric_bootstrap_ci(SIGMAS, Ks_mean)
    record['power_law'] = {
        'alpha': b, 'C': math.exp(a), 'R2': r2,
        'ci95_alpha': [ci_lo, ci_hi],
    }
    print(f"\nTiny-ImageNet zero-shot fit:")
    print(f"  alpha = {b:.4f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  C     = {math.exp(a):.2f}")
    print(f"  R^2   = {r2:.4f}")

    # ─── Compare to CIFAR / CelebA ──────────────────────────────────────────
    cifar_path = ROOT / 'outputs' / 'fine_grid_kstar' / 'fine_grid_kstar.json'
    celeba_path = ROOT / 'outputs' / 'finalization' / 'kstar_validation' / 'kstar_validation.json'

    cmp = {}
    if cifar_path.exists():
        cd = json.loads(cifar_path.read_text())
        cifar_alpha = cd['power_law']['fine_grid']['alpha']
        cifar_ci = cd['power_law']['fine_grid']['ci95_alpha']
        cmp['cifar10'] = {'alpha': cifar_alpha, 'ci95_alpha': cifar_ci,
                          'delta_vs_stl10': abs(b - cifar_alpha),
                          'in_cifar_ci': cifar_ci[0] <= b <= cifar_ci[1]}
        print(f"\n  CIFAR-10  alpha = {cifar_alpha:.4f}  CI [{cifar_ci[0]:.3f}, {cifar_ci[1]:.3f}]")
        print(f"  |Delta alpha (CIFAR vs STL-10)| = {abs(b - cifar_alpha):.4f}")
        print(f"  STL-10 alpha in CIFAR CI? {cmp['cifar10']['in_cifar_ci']}")

    if celeba_path.exists():
        cd = json.loads(celeba_path.read_text())
        cel_alpha = None
        try:
            cel_alpha = cd['experiments']['celeba_kan_seeded']['fit']['alpha']
        except (KeyError, TypeError):
            cel_alpha = (cd.get('power_law') or {}).get('alpha') or cd.get('alpha')
        if cel_alpha is not None:
            cmp['celeba'] = {'alpha': cel_alpha,
                             'delta_vs_tinyimagenet': abs(b - cel_alpha)}
            print(f"  CelebA-64 alpha = {cel_alpha:.4f}")
            print(f"  |Delta alpha (CelebA vs Tiny-ImageNet)| = {abs(b - cel_alpha):.4f}")
    record['cross_dataset_comparison'] = cmp

    OUT_JSON.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {OUT_JSON}")

    # ─── Plain-text summary ─────────────────────────────────────────────────
    lines = [
        "Tier 2: Tiny-ImageNet zero-shot K* law (CIFAR ckpt, no retraining)",
        "=" * 70,
        f"  Dataset:    Tiny-ImageNet val (64x64 -> 32x32 resize+crop)",
        f"  Model:      CIFAR-trained KAN-EBM ({n_params:,} params)",
        f"  N images:   {N_IMAGES}",
        f"  K_max:      {K_MAX}",
        f"  sigmas:     {SIGMAS}",
        "",
        f"  alpha    = {b:.4f}   95% bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        f"  C        = {math.exp(a):.3f}",
        f"  R^2      = {r2:.4f}",
        "",
        "  Per-sigma K* (zero-shot):",
    ]
    for s in SIGMAS:
        d = record['per_sigma'][f"{s:.3f}"]
        lines.append(f"    sigma={s:.2f}: K* mean={d['kstar_mean']:.2f}  std={d['kstar_std']:.2f}")
    lines.append("")
    if cmp:
        lines.append("  Cross-dataset alpha comparison:")
        for k, v in cmp.items():
            delta = v.get('delta_vs_tinyimagenet') or v.get('delta_vs_stl10')
            lines.append(f"    {k}: alpha={v['alpha']:.4f}  |Delta vs Tiny-ImageNet|={delta:.4f}")
    OUT_TXT.write_text("\n".join(lines))
    print(f"Saved {OUT_TXT}")


if __name__ == '__main__':
    main()
