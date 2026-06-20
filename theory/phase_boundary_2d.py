"""
2D Phase Boundary: R² of K*(σ) ~ σ^α as a function of (blur, noise)
====================================================================
For each blur level, sweep noise level, find K* per (blur, noise) pair,
then fit the power law across noise levels.  The resulting R²(blur) reveals
the phase boundary where the universal scaling law breaks down.

Run:  python theory/phase_boundary_2d.py
Time: ~15 hours on RTX 4090 (6 blur × 5 noise × 3 models × 40 images × 31 K steps)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'theory'))
from measure_kappa import MODELS, grad_E, load_cifar_test, DEVICE  # noqa: E402

OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

N_IMAGES = 40
K_MAX = 30
USE_MODELS = ['kan_f32', 'conv_mlp_gelu', 'unet']
BLUR_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
OUT_JSON = OUT / 'phase_boundary_2d.json'


def psnr(u, clean):
    """Mean PSNR over a batch of images (both in [-1, 1])."""
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).mean().item()


def ksweep_kstar(model, corrupted, clean, K_max=K_MAX, dt=0.05, decay=0.97):
    """Return K* = argmax mean-PSNR over K in 0..K_max (0 = no iteration)."""
    u = corrupted.clone()
    step = dt
    series = [psnr(u, clean)]
    for _ in range(K_max):
        u = (u - step * grad_E(model, u).clamp(-1., 1.)).detach()
        step *= decay
        series.append(psnr(u, clean))
    return int(np.argmax(series)), series[0], max(series)


def fit_law(sev, kstar):
    """Fit power law log(K*) = alpha * log(sev) + const; return alpha, R²."""
    sev = np.asarray(sev, float)
    kstar = np.asarray(kstar, float)
    ok = kstar > 0
    if ok.sum() < 4 or np.std(kstar[ok]) < 1e-6:
        return 0.0, 0.0   # flat / no productive iteration => no law
    x, y = np.log(sev[ok]), np.log(kstar[ok])
    slope, intercept = np.polyfit(x, y, 1)
    yh = intercept + slope * x
    r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(r2)


def c_blur_then_noise(x, blur_sig, noise_sig):
    """Apply Gaussian blur then additive Gaussian noise (both in [-1, 1])."""
    if blur_sig > 0.0:
        x = gaussian_blur(x, kernel_size=[5, 5], sigma=[float(blur_sig), float(blur_sig)])
    x = x + noise_sig * torch.randn_like(x)
    return x


def main():
    print(f"Device: {DEVICE}  | images={N_IMAGES}  K_max={K_MAX}")
    print(f"Models: {USE_MODELS}")
    print(f"Blur levels: {BLUR_LEVELS}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Output: {OUT_JSON}")
    print("=" * 60)

    # ── resume-safe JSON loading ──────────────────────────────────────────
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text())
        print(f"[resume] Loaded existing results from {OUT_JSON}")
    else:
        results = {}

    test_ds = load_cifar_test()
    clean = torch.stack([test_ds[i][0] for i in range(N_IMAGES)]).to(DEVICE)

    # ── outer loop: model → blur → noise ────────────────────────────────
    for mname in USE_MODELS:
        cfg = MODELS[mname]
        if not cfg['ckpt'].exists():
            print(f"\n[skip] {mname}: checkpoint missing at {cfg['ckpt']}")
            continue

        if mname not in results:
            results[mname] = {}

        # Load model once per architecture
        model = cfg['cls'](**cfg['kw']).to(DEVICE)
        model.load_state_dict(torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"\n{'='*60}")
        print(f"MODEL: {mname}")
        print(f"{'='*60}")

        for blur_lvl in BLUR_LEVELS:
            blur_key = f"blur_{blur_lvl:.1f}"

            # Resume skip
            if blur_key in results[mname]:
                print(f"\n[skip] {blur_key} already completed")
                continue

            print(f"\n--- blur_level = {blur_lvl} ---")
            kstars = []
            in_psnrs = []
            best_psnrs = []

            for noise_lvl in NOISE_LEVELS:
                torch.manual_seed(0)  # deterministic corruption
                corrupted = c_blur_then_noise(clean, blur_lvl, noise_lvl).to(DEVICE)
                ks, p_in, p_best = ksweep_kstar(model, corrupted, clean)
                kstars.append(ks)
                in_psnrs.append(p_in)
                best_psnrs.append(p_best)
                print(f"  noise={noise_lvl:.2f}  K*={ks:2d}  "
                      f"PSNR_in={p_in:6.2f}  PSNR_best={p_best:6.2f}")

            # Fit power law across noise levels (fixed blur)
            alpha, r2 = fit_law(NOISE_LEVELS, kstars)
            verdict = 'LAW' if (r2 >= 0.9 and alpha > 0.3) else ('weak' if r2 >= 0.7 else 'BREAKS')

            results[mname][blur_key] = {
                'alpha': alpha,
                'r2': r2,
                'kstars': kstars,
                'noise_levels': NOISE_LEVELS,
                'in_psnr': in_psnrs,
                'best_psnr': best_psnrs,
                'verdict': verdict,
            }

            print(f"  --> alpha={alpha:.3f}  R^2={r2:.3f}  [{verdict}]")

            # Save after each blur level (resume-safe)
            OUT_JSON.write_text(json.dumps(results, indent=2))
            print(f"  [saved] {OUT_JSON}")

        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: 2D Phase Boundary (R² of power-law fit)")
    print("=" * 60)
    print(f"{'model':>14} | {'blur':>8} | {'alpha':>7} | {'R^2':>6} | verdict")
    print("-" * 60)
    for mname in results:
        for blur_key in sorted(results[mname].keys(), key=lambda s: float(s.split('_')[1])):
            r = results[mname][blur_key]
            print(f"{mname:>14} | {blur_key:>8} | {r['alpha']:>7.3f} | {r['r2']:>6.3f} | {r['verdict']}")
    print(f"\nSaved to: {OUT_JSON}")


if __name__ == '__main__':
    main()
