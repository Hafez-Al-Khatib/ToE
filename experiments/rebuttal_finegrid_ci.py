"""
rebuttal_finegrid_ci.py
========================
Wave 5: Bootstrap CIs on the fine-grid alpha for KAN + 4 ConvMLP variants,
plus per-image K* variability statistics.

Addresses:
  - Minor 1: ReLU outlier in cluster -- is ReLU's alpha statistically distinct
    from GELU/SiLU/Tanh, or within sampling noise?
  - Minor 3: Per-image K* variability (the law is population-level; how much
    do individual images deviate?)

Reads:
  outputs/fine_grid_kstar/fine_grid_kstar.json           (KAN-EBM)
  outputs/fine_grid_kstar/fine_grid_convmlp_{gelu,silu,tanh,relu}.json

Writes:
  outputs/fine_grid_kstar/finegrid_ci_summary.json
  outputs/fine_grid_kstar/finegrid_ci_summary.txt
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'outputs' / 'fine_grid_kstar'
OUT = SRC / 'finegrid_ci_summary.json'
OUT_TXT = SRC / 'finegrid_ci_summary.txt'

VARIANTS = {
    'KAN-EBM':       SRC / 'fine_grid_kstar.json',
    'ConvMLP-GELU':  SRC / 'fine_grid_convmlp_gelu.json',
    'ConvMLP-SiLU':  SRC / 'fine_grid_convmlp_silu.json',
    'ConvMLP-Tanh':  SRC / 'fine_grid_convmlp_tanh.json',
    'ConvMLP-ReLU':  SRC / 'fine_grid_convmlp_relu.json',
}

RNG = np.random.default_rng(20260502)
B_BOOT = 10000


def fit_powerlaw(sigmas, Ks):
    xs = np.log(np.asarray(sigmas, float))
    ys = np.log(np.asarray(Ks, float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def parametric_bootstrap(sigmas, Ks, B=B_BOOT):
    a, b, _ = fit_powerlaw(sigmas, Ks)
    xs = np.log(np.asarray(sigmas, float))
    ys = np.log(np.asarray(Ks, float))
    yh = a + b * xs
    resid = ys - yh
    resid_c = resid - resid.mean()
    boot = np.empty(B)
    for i in range(B):
        e = RNG.choice(resid_c, size=len(xs), replace=True)
        s, _ = np.polyfit(xs, yh + e, 1)
        boot[i] = s
    return boot, b


def load_data(path):
    """Returns (sigmas, kstar_means, per_image_kstars)."""
    d = json.loads(path.read_text())
    per = d['per_sigma']
    sigmas = sorted(float(k) for k in per.keys())
    means = [per[f"{s:.3f}"]['kstar_mean'] for s in sigmas]
    per_image = {}
    for s in sigmas:
        rec = per[f"{s:.3f}"]
        if 'kstar_per_image' in rec:
            per_image[s] = rec['kstar_per_image']
        else:
            per_image[s] = None
    return sigmas, means, per_image


def main():
    print("\n" + "=" * 90)
    print("Fine-grid alpha bootstrap CIs (B = 10,000)")
    print("=" * 90)
    print(f"{'Variant':<14} {'alpha':>7} {'95% CI alpha':<22} {'C':>7} {'R^2':>7} {'mean K* std/img':>16}")
    print("-" * 90)

    record = {'B': B_BOOT, 'variants': {}, 'pairwise': {}}
    boots = {}

    for name, path in VARIANTS.items():
        sigmas, Ks, per_image = load_data(path)
        a, b, r2 = fit_powerlaw(sigmas, Ks)
        C = math.exp(a)
        boot, _ = parametric_bootstrap(sigmas, Ks)
        boots[name] = boot
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        # Per-image K* variability
        per_img_stds = []
        for s in sigmas:
            arr = per_image.get(s)
            if arr is not None and len(arr) > 1:
                per_img_stds.append(float(np.std(arr, ddof=1)))
        mean_per_img_std = float(np.mean(per_img_stds)) if per_img_stds else float('nan')

        print(f"{name:<14} {b:>7.3f} [{ci_lo:>+5.3f}, {ci_hi:>+5.3f}] {C:>7.2f} {r2:>7.4f} {mean_per_img_std:>16.3f}")
        record['variants'][name] = {
            'alpha': b, 'ci95_alpha': [ci_lo, ci_hi], 'C': C, 'r_squared': r2,
            'mean_per_image_kstar_std': mean_per_img_std,
            'sigmas': sigmas, 'kstar_means': Ks,
        }

    # ─── Pairwise comparisons: KAN vs each ConvMLP, ReLU vs (GELU,SiLU,Tanh) ────
    print("\n" + "=" * 90)
    print("Pairwise alpha differences (parametric bootstrap on residuals)")
    print("=" * 90)
    print(f"{'A':<14} {'B':<14} {'A-B':>8} {'95% CI of diff':<22} {'P(A>B)':>8}  {'Verdict'}")
    print("-" * 90)

    pairs_to_test = [
        ('KAN-EBM', 'ConvMLP-GELU'),
        ('KAN-EBM', 'ConvMLP-SiLU'),
        ('KAN-EBM', 'ConvMLP-Tanh'),
        ('KAN-EBM', 'ConvMLP-ReLU'),
        ('ConvMLP-ReLU', 'ConvMLP-GELU'),
        ('ConvMLP-ReLU', 'ConvMLP-SiLU'),
        ('ConvMLP-ReLU', 'ConvMLP-Tanh'),
        ('ConvMLP-Tanh', 'ConvMLP-GELU'),
    ]
    for A, B in pairs_to_test:
        diff = boots[A] - boots[B]
        d_lo, d_hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        p_a_gt_b = float(np.mean(diff > 0))
        if d_lo > 0:
            verdict = "** A statistically larger"
        elif d_hi < 0:
            verdict = "** B statistically larger"
        else:
            verdict = "(CI overlaps zero -- not distinguishable)"
        print(f"{A:<14} {B:<14} {float(np.mean(diff)):>+8.3f} "
              f"[{d_lo:>+6.3f}, {d_hi:>+6.3f}] {p_a_gt_b:>8.3f}  {verdict}")
        record['pairwise'][f"{A}_vs_{B}"] = {
            'mean_diff': float(np.mean(diff)),
            'ci95_diff': [d_lo, d_hi],
            'p_A_gt_B': p_a_gt_b,
            'verdict': verdict,
        }

    OUT.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {OUT}")

    # Plain-text summary
    lines = [
        "Fine-grid alpha bootstrap CIs (B = 10,000)",
        "=" * 75,
        "",
        "Per-variant fits + 95% bootstrap CIs:",
    ]
    for name, v in record['variants'].items():
        a, lo, hi = v['alpha'], v['ci95_alpha'][0], v['ci95_alpha'][1]
        std = v['mean_per_image_kstar_std']
        lines.append(f"  {name:<14}  alpha = {a:.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                     f"C = {v['C']:.2f}  R^2 = {v['r_squared']:.4f}  per-image K* std (mean across sigma) = {std:.2f}")
    lines.append("")
    lines.append("Pairwise tests (CI excludes 0 means statistically distinct):")
    for k, p in record['pairwise'].items():
        lines.append(f"  {k}:  diff = {p['mean_diff']:+.3f}  "
                     f"95% CI [{p['ci95_diff'][0]:+.3f}, {p['ci95_diff'][1]:+.3f}]  "
                     f"P(A>B) = {p['p_A_gt_B']:.3f}  {p['verdict']}")
    OUT_TXT.write_text("\n".join(lines))
    print(f"Saved {OUT_TXT}")


if __name__ == "__main__":
    main()
