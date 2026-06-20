"""
rebuttal_bootstrap_ci.py
========================
Wave 1a: Bootstrap + jackknife confidence intervals on the K*(sigma) power-law
exponent alpha.  Addresses Reviewer 1 R3 and Reviewer 2 minor weaknesses
(no CIs reported, "fit could be coincidental from 5 points").

Reads existing outputs/finalization/kstar_validation/kstar_validation.json
and produces:
  - outputs/finalization/rebuttal/bootstrap_ci.json
  - outputs/finalization/rebuttal/bootstrap_ci_summary.txt

Methods:
  1. Parametric bootstrap on log-log residuals (B=10000)
  2. Jackknife (leave-one-out) on the 5 sigma anchors
  3. Pivotal 95% CI from each
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / 'outputs' / 'finalization' / 'kstar_validation' / 'kstar_validation.json'
OUT_DIR = ROOT / 'outputs' / 'finalization' / 'rebuttal'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260501)
B = 10000  # bootstrap reps


def fit_power_law(sigmas, K_means):
    """log K = a + b log sigma, returns (intercept_a, slope_b, R2)."""
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2), yh


def jackknife_alpha(sigmas, K_means):
    """Leave-one-out: refit alpha with each sigma removed."""
    n = len(sigmas)
    alphas = []
    for i in range(n):
        s = [sigmas[j] for j in range(n) if j != i]
        k = [K_means[j] for j in range(n) if j != i]
        _, b, _, _ = fit_power_law(s, k)
        alphas.append(b)
    alphas = np.array(alphas)
    mean_a = alphas.mean()
    se_a = math.sqrt((n - 1) / n * float(np.sum((alphas - mean_a) ** 2)))
    return alphas, mean_a, se_a


def parametric_bootstrap(sigmas, K_means, B=10000):
    """Resample residuals from the nominal log-log fit."""
    a, b, r2, yh = fit_power_law(sigmas, K_means)
    ys = np.log(np.asarray(K_means, dtype=float))
    resid = ys - yh
    # Center residuals (recommended for bootstrap of regression residuals)
    resid_c = resid - resid.mean()
    xs = np.log(np.asarray(sigmas, dtype=float))
    n = len(sigmas)
    boot_alphas = np.empty(B)
    boot_intercepts = np.empty(B)
    boot_C = np.empty(B)
    for b_i in range(B):
        # Resample residuals with replacement
        e = RNG.choice(resid_c, size=n, replace=True)
        ys_boot = yh + e
        slope, intercept = np.polyfit(xs, ys_boot, 1)
        boot_alphas[b_i] = slope
        boot_intercepts[b_i] = intercept
        boot_C[b_i] = math.exp(intercept)
    return boot_alphas, boot_intercepts, boot_C, (a, b, r2)


def case_resampling_bootstrap(sigmas, K_means, B=10000):
    """Resample (sigma, K*) pairs with replacement."""
    sigmas = np.asarray(sigmas, dtype=float)
    K_means = np.asarray(K_means, dtype=float)
    n = len(sigmas)
    boot_alphas = []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        s = sigmas[idx]
        k = K_means[idx]
        # Need at least 2 distinct sigmas for a fit
        if len(np.unique(s)) < 2:
            continue
        _, slope, _, _ = fit_power_law(s.tolist(), k.tolist())
        boot_alphas.append(slope)
    return np.array(boot_alphas)


def percentile_ci(samples, level=0.95):
    lo = (1.0 - level) / 2.0 * 100
    hi = (1.0 + level) / 2.0 * 100
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def analyse(label, sigmas, K_means, out):
    print(f"\n=== {label}  (n={len(sigmas)}) ===")
    a, b, r2, _ = fit_power_law(sigmas, K_means)
    C = math.exp(a)
    print(f"  Nominal fit:  alpha = {b:.4f}   C = {C:.3f}   R^2 = {r2:.4f}")

    # Jackknife
    jack_alphas, jack_mean, jack_se = jackknife_alpha(sigmas, K_means)
    jack_lo = b - 1.96 * jack_se
    jack_hi = b + 1.96 * jack_se
    print(f"  Jackknife SE: {jack_se:.4f}    95% CI: [{jack_lo:.3f}, {jack_hi:.3f}]")
    print(f"  Leave-one-out alphas: {[f'{x:.3f}' for x in jack_alphas]}")

    # Parametric bootstrap on residuals
    pb_alphas, pb_ints, pb_C, _ = parametric_bootstrap(sigmas, K_means, B=B)
    pb_lo, pb_hi = percentile_ci(pb_alphas)
    pbC_lo, pbC_hi = percentile_ci(pb_C)
    print(f"  Parametric bootstrap (B={B}):")
    print(f"    alpha 95% CI: [{pb_lo:.3f}, {pb_hi:.3f}]   median {np.median(pb_alphas):.3f}")
    print(f"    C     95% CI: [{pbC_lo:.2f}, {pbC_hi:.2f}]   median {np.median(pb_C):.2f}")

    # Case-resampling bootstrap (more conservative)
    case_alphas = case_resampling_bootstrap(sigmas, K_means, B=B)
    if len(case_alphas) > 0:
        case_lo, case_hi = percentile_ci(case_alphas)
        print(f"  Case bootstrap     ({len(case_alphas)} valid reps):")
        print(f"    alpha 95% CI: [{case_lo:.3f}, {case_hi:.3f}]   median {np.median(case_alphas):.3f}")
    else:
        case_lo, case_hi = float('nan'), float('nan')

    out[label] = {
        'n_anchors': len(sigmas),
        'sigmas': sigmas,
        'K_means': K_means,
        'nominal': {'alpha': b, 'C': C, 'r_squared': r2},
        'jackknife': {
            'leave_one_out_alphas': jack_alphas.tolist(),
            'se_alpha': jack_se,
            'ci95_alpha': [jack_lo, jack_hi],
        },
        'parametric_bootstrap': {
            'B': B,
            'ci95_alpha': [pb_lo, pb_hi],
            'median_alpha': float(np.median(pb_alphas)),
            'std_alpha': float(np.std(pb_alphas)),
            'ci95_C': [pbC_lo, pbC_hi],
            'median_C': float(np.median(pb_C)),
        },
        'case_bootstrap': {
            'B_valid': int(len(case_alphas)),
            'ci95_alpha': [case_lo, case_hi],
            'median_alpha': float(np.median(case_alphas)) if len(case_alphas) > 0 else None,
        },
    }


def main():
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    rec = json.loads(IN_PATH.read_text())
    sigmas_str = rec['sigmas']
    sigmas = [float(s) for s in sigmas_str]

    out = {}

    for tag, exp in rec['experiments'].items():
        K_means = [exp['mean'][str(s)] for s in sigmas_str]
        # Fair MLP has K*=1 across all sigmas except 0.30 -> degenerate, skip CI
        if len(set(K_means)) < 2:
            print(f"\n[skip] {tag}: K* is constant ({K_means}); no power-law CI possible.")
            continue
        analyse(tag, sigmas, K_means, out)

    # Write JSON record
    out_json = OUT_DIR / 'bootstrap_ci.json'
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_json}")

    # Human-readable summary table
    lines = ["Bootstrap & jackknife CIs on K*(sigma) power-law exponent",
             "=" * 70,
             f"Methods: parametric bootstrap on log-log residuals (B={B}),",
             "         jackknife (leave-one-out), case resampling.",
             "",
             f"{'Experiment':<26} {'n':>3} {'alpha':>8} {'jack 95% CI':<22} {'param-boot 95% CI':<22}"]
    lines.append("-" * 90)
    for tag, r in out.items():
        n = r['n_anchors']
        a = r['nominal']['alpha']
        jlo, jhi = r['jackknife']['ci95_alpha']
        plo, phi = r['parametric_bootstrap']['ci95_alpha']
        lines.append(f"{tag:<26} {n:>3} {a:>8.3f} [{jlo:>+6.3f}, {jhi:>+6.3f}]  "
                     f"[{plo:>+6.3f}, {phi:>+6.3f}]")
    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / 'bootstrap_ci_summary.txt').write_text(summary)
    print(f"\nSaved {OUT_DIR / 'bootstrap_ci_summary.txt'}")


if __name__ == "__main__":
    main()
