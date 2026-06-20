"""
rebuttal_smooth_mlp_analysis.py
================================
Post-hoc statistical analysis on the matched-architecture activation ablation.
Run after rebuttal_smooth_mlp.py completes.

Computes:
  1. Per-variant 95% CI on alpha (parametric bootstrap on log-log residuals)
  2. Peak PSNR per variant per sigma (apples-to-apples quality comparison)
  3. Effect-size summary: KAN-vs-each-variant in (alpha, peak PSNR)
  4. Pairwise alpha-difference test (does KAN's alpha differ from any
     smooth-MLP variant outside the bootstrap CI?)

Outputs:
  outputs/finalization/rebuttal/smooth_mlp_stats.json
  outputs/finalization/rebuttal/smooth_mlp_stats_summary.txt
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'finalization' / 'rebuttal'
ABLATION = OUT / 'smooth_mlp_ablation.json'
KSTAR_VAL = ROOT / 'outputs' / 'finalization' / 'kstar_validation' / 'kstar_validation.json'


def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def bootstrap_alpha(sigmas, K_means, B=10000, seed=20260501):
    a, b, r2 = fit_power_law(sigmas, K_means)
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    yh = a + b * xs
    resid = ys - yh
    resid_c = resid - resid.mean()
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for i in range(B):
        e = rng.choice(resid_c, size=len(xs), replace=True)
        ys_b = yh + e
        s, _ = np.polyfit(xs, ys_b, 1)
        boot[i] = s
    return boot, b


def main():
    if not ABLATION.exists():
        raise SystemExit(f"Run rebuttal_smooth_mlp.py first: {ABLATION}")

    abl = json.loads(ABLATION.read_text())
    kstar = json.loads(KSTAR_VAL.read_text())
    sigmas = abl['config']['sigmas']
    K_list = abl['config']['K_list']

    out = {'sigmas': sigmas, 'variants': {}}

    # KAN reference (110K, large) from kstar_validation
    kan = kstar['experiments']['cifar_kan_seeded']
    kan_K = [kan['mean'][str(s)] for s in sigmas]
    kan_boot, kan_alpha = bootstrap_alpha(sigmas, kan_K)
    kan_intercept, _, kan_r2 = fit_power_law(sigmas, kan_K)
    kan_C = math.exp(kan_intercept)
    print(f"\n{'Variant':<14} {'Params':>7} {'alpha':>7} {'95% CI alpha':<20} "
          f"{'C':>7} {'R^2':>6} {'Peak PSNR per sigma'}")
    print("-" * 110)

    # Add KAN reference row -- use peak PSNR from kstar_spectrum.json psnr_table if available
    kstar_spec = json.loads((ROOT / 'outputs' / 'finalization' / 'kstar' /
                              'kstar_spectrum.json').read_text())
    kan_peak_psnr = {}
    for s in sigmas:
        kan_peak_psnr[s] = max(kstar_spec['psnr_table'][str(s)].values())
    psnr_str = " ".join(f"{kan_peak_psnr[s]:5.2f}" for s in sigmas)
    a_lo, a_hi = float(np.percentile(kan_boot, 2.5)), float(np.percentile(kan_boot, 97.5))
    print(f"{'KAN (ref)':<14} {32096:>7} {kan_alpha:>7.3f} "
          f"[{a_lo:>+5.3f}, {a_hi:>+5.3f}] {kan_C:>7.2f} "
          f"{kan_r2:>6.3f}   {psnr_str}")
    out['variants']['KAN_ref'] = {
        'n_params': 32096,
        'alpha': kan_alpha,
        'ci95_alpha': [a_lo, a_hi],
        'C': kan_C,
        'r_squared': kan_r2,
        'peak_psnr_per_sigma': {str(s): kan_peak_psnr[s] for s in sigmas},
    }

    # Each smooth-MLP variant
    for variant, r in abl['variants'].items():
        K_means = [r['kstar_means'][str(s)] for s in sigmas]
        if len(set(K_means)) < 2:
            print(f"{variant:<14} {r['n_params']:>7} {'n/a':>7} {'(degenerate)':<20} "
                  f"{'n/a':>7} {'n/a':>6}")
            continue
        boot, alpha = bootstrap_alpha(sigmas, K_means)
        intercept, _, r2 = fit_power_law(sigmas, K_means)
        C = math.exp(intercept)
        a_lo, a_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        # Peak PSNR per sigma (max over K)
        peak_psnr = {}
        for s in sigmas:
            psnrs_at_s = r['psnr_per_K_avg'][str(s)]
            peak_psnr[s] = max(psnrs_at_s.values())
        psnr_str = " ".join(f"{peak_psnr[s]:5.2f}" for s in sigmas)
        print(f"{variant:<14} {r['n_params']:>7} {alpha:>7.3f} "
              f"[{a_lo:>+5.3f}, {a_hi:>+5.3f}] {C:>7.2f} "
              f"{r2:>6.3f}   {psnr_str}")

        # Pairwise: how often does KAN's bootstrap alpha exceed this variant's?
        diff_dist = kan_boot - boot
        p_kan_larger = float(np.mean(diff_dist > 0))
        diff_lo, diff_hi = float(np.percentile(diff_dist, 2.5)), float(np.percentile(diff_dist, 97.5))

        out['variants'][variant] = {
            'n_params': r['n_params'],
            'alpha': alpha,
            'ci95_alpha': [a_lo, a_hi],
            'C': C,
            'r_squared': r2,
            'peak_psnr_per_sigma': {str(s): peak_psnr[s] for s in sigmas},
            'kan_minus_variant_alpha': {
                'point_estimate': kan_alpha - alpha,
                'ci95': [diff_lo, diff_hi],
                'p_kan_larger': p_kan_larger,
            },
        }

    # ─── Pairwise table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Pairwise alpha differences (KAN vs each smooth-MLP variant)")
    print("=" * 80)
    print(f"{'Variant':<14} {'KAN-variant':<14} {'95% CI of difference':<25} "
          f"{'P(KAN > variant)'}")
    print("-" * 80)
    for variant, v in out['variants'].items():
        if variant == 'KAN_ref' or 'kan_minus_variant_alpha' not in v:
            continue
        d = v['kan_minus_variant_alpha']
        sig = "** distinct" if d['ci95'][0] > 0 else "(overlap)"
        print(f"{variant:<14} {d['point_estimate']:>+8.3f}      "
              f"[{d['ci95'][0]:>+6.3f}, {d['ci95'][1]:>+6.3f}]   "
              f"{d['p_kan_larger']:>6.3f}  {sig}")

    # ─── Peak PSNR comparison at sigma=0.15 (KAN-EBM training sigma) ──────────
    print("\n" + "=" * 80)
    print("Peak PSNR head-to-head at sigma=0.15 (the training noise level)")
    print("=" * 80)
    for variant, v in out['variants'].items():
        peak = v['peak_psnr_per_sigma']['0.15']
        delta = peak - out['variants']['KAN_ref']['peak_psnr_per_sigma']['0.15']
        print(f"  {variant:<14}  peak PSNR (sigma=0.15) = {peak:.3f} dB  "
              f"(vs KAN: {delta:+.2f} dB)")

    note = ("NOTE: K* is selected from a discrete grid {1,2,3,5,7,10,15,20,30,50}. "
            "Identical K* sequences across variants therefore yield mathematically identical "
            "power-law fits; this is expected and does not indicate methodological error.")
    out['note'] = note
    print("\n" + note)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / 'smooth_mlp_stats_independent.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
