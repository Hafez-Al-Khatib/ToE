"""
One-shot calibration validation.
================================
Since alpha ~= 1.37 is near-universal, a practitioner can:
  1. Fix alpha = 1.37 (no fitting needed)
  2. Measure K* at ONE noise level (sigma_cal)
  3. Solve C = K* / sigma_cal^alpha
  4. Predict K*(sigma) = C * sigma^1.37 for any sigma

This script validates the one-shot rule against the oracle (per-sigma K*)
using existing tier0 + cross-dataset results. Reports:
  - K* prediction error at each sigma (%)
  - Comparison to 5-point fit (all sigmas)
  - Comparison to FIXED-K baseline

No GPU needed — pure numpy analysis of existing JSON results.

Run:  py -3.12 theory/oneshot_calibration.py
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SIGMAS = np.array([0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30])
ALPHA_UNIVERSAL = 1.37

DATASETS = {
    'cifar10': {
        'file': ROOT / 'outputs' / 'tier0_seeds' / 'tier0_results.json',
        'resolution': '32px',
    },
    'fashion': {
        'file': ROOT / 'outputs' / 'cross_dataset' / 'cross_dataset_results.json',
        'resolution': '28px',
    },
    'mnist': {
        'file': ROOT / 'outputs' / 'cross_dataset' / 'cross_dataset_results.json',
        'resolution': '28px',
    },
    'ddpm': {
        'file': ROOT / 'ddpm_baseline' / 'ddpm_results.json',
        'resolution': '32px',
    },
}


def fit_alpha(sigmas, kstars):
    ok = np.array(kstars) > 0
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    log_s = np.log(sigmas[ok])
    log_k = np.log(np.array(kstars)[ok])
    coeffs = np.polyfit(log_s, log_k, 1)
    alpha = coeffs[0]
    C = np.exp(coeffs[1])
    pred = C * sigmas[ok] ** alpha
    ss_res = np.sum((np.array(kstars)[ok] - pred) ** 2)
    ss_tot = np.sum((np.array(kstars)[ok] - np.mean(np.array(kstars)[ok])) ** 2)
    r2 = 1 - ss_res / ss_tot
    return alpha, C, r2


def oneshot_predict(kstar_cal, sigma_cal, alpha_fixed, sigmas):
    C = kstar_cal / sigma_cal ** alpha_fixed
    return C * sigmas ** alpha_fixed


def main():
    print("=" * 80)
    print("ONE-SHOT CALIBRATION VALIDATION")
    print(f"Universal alpha = {ALPHA_UNIVERSAL}")
    print("=" * 80)

    all_results = {}

    # --- CIFAR-10 ---
    with open(DATASETS['cifar10']['file']) as f:
        cifar_data = json.load(f)

    print("\n--- CIFAR-10 (32px) ---")
    for arch in sorted(cifar_data):
        for si, seed_kstars in enumerate(cifar_data[arch]['kstars']):
            kstars = np.array(seed_kstars)

            # Oracle: 5-point fit
            alpha_oracle, C_oracle, r2_oracle = fit_alpha(SIGMAS, kstars)

            # One-shot: fix alpha, calibrate C from sigma=0.16 (index 3, middle of range)
            sigma_cal_idx = 3  # sigma = 0.16
            C_oneshot = kstars[sigma_cal_idx] / SIGMAS[sigma_cal_idx] ** ALPHA_UNIVERSAL
            kstar_oneshot = C_oneshot * SIGMAS ** ALPHA_UNIVERSAL

            # Prediction errors
            pct_err = (kstar_oneshot - kstars) / kstars * 100

            key = f"cifar10_{arch}_s{si}"
            all_results[key] = {
                'alpha_oracle': alpha_oracle,
                'C_oracle': C_oracle,
                'C_oneshot': C_oneshot,
                'kstar_oracle': (C_oracle * SIGMAS ** alpha_oracle).tolist(),
                'kstar_oneshot': kstar_oneshot.tolist(),
                'kstar_true': kstars.tolist(),
                'pct_errors': pct_err.tolist(),
                'max_abs_pct_err': float(np.max(np.abs(pct_err))),
                'mean_abs_pct_err': float(np.mean(np.abs(pct_err))),
            }

    # --- Fashion + MNIST ---
    with open(DATASETS['fashion']['file']) as f:
        cross_data = json.load(f)

    for dataset_name in ['fashion', 'mnist']:
        if dataset_name not in cross_data:
            continue
        ds = cross_data[dataset_name]
        print(f"\n--- {dataset_name.upper()} ({DATASETS[dataset_name]['resolution']}) ---")
        for family in sorted(ds['families']):
            fam_data = ds['families'][family]
            for si, seed_kstars in enumerate(fam_data['kstars']):
                kstars = np.array(seed_kstars)
                alpha_oracle, C_oracle, r2_oracle = fit_alpha(SIGMAS, kstars)

                sigma_cal_idx = 3
                C_oneshot = kstars[sigma_cal_idx] / SIGMAS[sigma_cal_idx] ** ALPHA_UNIVERSAL
                kstar_oneshot = C_oneshot * SIGMAS ** ALPHA_UNIVERSAL

                pct_err = (kstar_oneshot - kstars) / kstars * 100

                key = f"{dataset_name}_{family}_s{si}"
                all_results[key] = {
                    'alpha_oracle': alpha_oracle,
                    'C_oracle': C_oracle,
                    'C_oneshot': C_oneshot,
                    'kstar_oracle': (C_oracle * SIGMAS ** alpha_oracle).tolist(),
                    'kstar_oneshot': kstar_oneshot.tolist(),
                    'kstar_true': kstars.tolist(),
                    'pct_errors': pct_err.tolist(),
                    'max_abs_pct_err': float(np.max(np.abs(pct_err))),
                    'mean_abs_pct_err': float(np.mean(np.abs(pct_err))),
                }

    # --- DDPM pretrained ---
    ddpm_path = DATASETS['ddpm']['file']
    if ddpm_path.exists():
        with open(ddpm_path) as f:
            ddpm_data = json.load(f)
        print(f"\n--- DDPM pretrained (32px, 35.7M params) ---")
        for version in ['V1_fixed_cond_0.15', 'V2_noise_matched']:
            if version not in ddpm_data:
                continue
            vdata = ddpm_data[version]
            kstars_list = []
            for s in SIGMAS:
                key_str = f"{s:.2f}" if f"{s:.2f}" in vdata else str(s)
                kstars_list.append(vdata[key_str]['kstar'])
            kstars = np.array(kstars_list)
            alpha_oracle, C_oracle, r2_oracle = fit_alpha(SIGMAS, kstars)

            sigma_cal_idx = 3
            C_oneshot = kstars[sigma_cal_idx] / SIGMAS[sigma_cal_idx] ** ALPHA_UNIVERSAL
            kstar_oneshot = C_oneshot * SIGMAS ** ALPHA_UNIVERSAL
            pct_err = (kstar_oneshot - kstars) / kstars * 100

            short = version.split('_')[0] + '_' + version.split('_')[1]
            key = f"ddpm_{short}"
            all_results[key] = {
                'alpha_oracle': alpha_oracle,
                'C_oracle': C_oracle,
                'C_oneshot': C_oneshot,
                'kstar_oracle': (C_oracle * SIGMAS ** alpha_oracle).tolist(),
                'kstar_oneshot': kstar_oneshot.tolist(),
                'kstar_true': kstars.tolist(),
                'pct_errors': pct_err.tolist(),
                'max_abs_pct_err': float(np.max(np.abs(pct_err))),
                'mean_abs_pct_err': float(np.mean(np.abs(pct_err))),
            }

    # --- Summary ---
    print("\n" + "=" * 80)
    print("RESULTS: One-shot (alpha=1.37, C from sigma=0.16) vs Oracle (5-point fit)")
    print("=" * 80)
    print(f"{'Run':<35} {'alpha_oracle':>12} {'max_err%':>10} {'mean_err%':>10}")
    print("-" * 70)

    max_errs = []
    mean_errs = []
    for key in sorted(all_results):
        r = all_results[key]
        print(f"{key:<35} {r['alpha_oracle']:>12.4f} {r['max_abs_pct_err']:>10.2f}% {r['mean_abs_pct_err']:>10.2f}%")
        max_errs.append(r['max_abs_pct_err'])
        mean_errs.append(r['mean_abs_pct_err'])

    print("-" * 70)
    print(f"{'OVERALL':<35} {'':>12} {np.max(max_errs):>10.2f}% {np.mean(mean_errs):>10.2f}%")

    # Per-sigma breakdown
    print(f"\n--- Per-sigma prediction error (one-shot, averaged across all runs) ---")
    print(f"{'sigma':>8} {'mean_err%':>12} {'max_err%':>12}")
    for si, sig in enumerate(SIGMAS):
        errs_at_sig = [all_results[k]['pct_errors'][si] for k in all_results]
        print(f"{sig:>8.2f} {np.mean(np.abs(errs_at_sig)):>12.2f}% {np.max(np.abs(errs_at_sig)):>12.2f}%")

    # Try different calibration sigmas
    print(f"\n--- Effect of calibration sigma choice ---")
    print(f"{'cal_sigma':>12} {'mean_max_err%':>15} {'worst_max_err%':>16}")

    best_cal = None
    best_err = 1e9
    for cal_idx in range(len(SIGMAS)):
        cal_max_errs = []
        for key in all_results:
            kstars = np.array(all_results[key]['kstar_true'])
            C_cal = kstars[cal_idx] / SIGMAS[cal_idx] ** ALPHA_UNIVERSAL
            pred = C_cal * SIGMAS ** ALPHA_UNIVERSAL
            pct = np.abs((pred - kstars) / kstars * 100)
            cal_max_errs.append(np.max(pct))
        mean_max = np.mean(cal_max_errs)
        worst_max = np.max(cal_max_errs)
        if mean_max < best_err:
            best_err = mean_max
            best_cal = cal_idx
        print(f"{SIGMAS[cal_idx]:>12.2f} {mean_max:>15.2f}% {worst_max:>16.2f}%")

    print(f"\nBest calibration sigma: {SIGMAS[best_cal]:.2f} (mean max error = {best_err:.2f}%)")

    # --- Multi-point calibration comparison ---
    print(f"\n{'='*80}")
    print("CALIBRATION BUDGET COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<35} {'#sigmas':>8} {'mean_err%':>10} {'max_err%':>10}")
    print("-" * 70)

    # 1-point (one-shot): fix alpha=1.37, fit C from 1 sigma
    oneshot_max = np.max(max_errs)
    oneshot_mean = np.mean(mean_errs)
    print(f"{'1-pt (fix a=1.37, fit C)':<35} {'1':>8} {oneshot_mean:>10.2f}% {oneshot_max:>10.2f}%")

    # 2-point: fit both alpha and C from 2 sigmas
    from itertools import combinations
    best_2pt_err = 1e9
    best_2pt_pair = None
    for i, j in combinations(range(len(SIGMAS)), 2):
        pair_errs = []
        for key in all_results:
            kstars = np.array(all_results[key]['kstar_true'])
            # Fit alpha, C from 2 points
            log_s = np.log(SIGMAS[[i, j]])
            log_k = np.log(kstars[[i, j]])
            alpha_2 = (log_k[1] - log_k[0]) / (log_s[1] - log_s[0])
            C_2 = np.exp(log_k[0] - alpha_2 * log_s[0])
            pred = C_2 * SIGMAS ** alpha_2
            pct = np.abs((pred - kstars) / kstars * 100)
            pair_errs.append(np.max(pct))
        mean_max = np.mean(pair_errs)
        if mean_max < best_2pt_err:
            best_2pt_err = mean_max
            best_2pt_pair = (i, j)
            best_2pt_worst = np.max(pair_errs)
            best_2pt_mean_mean = np.mean([np.mean(np.abs((
                np.exp(np.log(np.array(all_results[k]['kstar_true'])[[i,j]])[0] -
                ((np.log(np.array(all_results[k]['kstar_true'])[[i,j]])[1] - np.log(np.array(all_results[k]['kstar_true'])[[i,j]])[0]) /
                (np.log(SIGMAS[[i,j]])[1] - np.log(SIGMAS[[i,j]])[0])) *
                np.log(SIGMAS[[i,j]])[0]) *
                SIGMAS ** ((np.log(np.array(all_results[k]['kstar_true'])[[i,j]])[1] - np.log(np.array(all_results[k]['kstar_true'])[[i,j]])[0]) /
                (np.log(SIGMAS[[i,j]])[1] - np.log(SIGMAS[[i,j]])[0])) -
                np.array(all_results[k]['kstar_true'])) /
                np.array(all_results[k]['kstar_true']) * 100)) for k in all_results])

    i2, j2 = best_2pt_pair
    print(f"{'2-pt (fit a,C from best pair)':<35} {'2':>8} {best_2pt_mean_mean:>10.2f}% {best_2pt_worst:>10.2f}%")
    print(f"  Best pair: sigma = [{SIGMAS[i2]:.2f}, {SIGMAS[j2]:.2f}]")

    # 5-point (oracle): fit alpha and C from all 5+ sigmas
    oracle_max_errs = []
    oracle_mean_errs = []
    for key in all_results:
        kstars = np.array(all_results[key]['kstar_true'])
        alpha_o, C_o, _ = fit_alpha(SIGMAS, kstars)
        pred = C_o * SIGMAS ** alpha_o
        pct = np.abs((pred - kstars) / kstars * 100)
        oracle_max_errs.append(np.max(pct))
        oracle_mean_errs.append(np.mean(pct))
    print(f"{'7-pt oracle (fit a,C from all)':<35} {'7':>8} {np.mean(oracle_mean_errs):>10.2f}% {np.max(oracle_max_errs):>10.2f}%")

    # Fixed-K baseline (K=5 for all sigmas)
    fixed_k = 5.0
    fk_max_errs = []
    fk_mean_errs = []
    for key in all_results:
        kstars = np.array(all_results[key]['kstar_true'])
        pct = np.abs((fixed_k - kstars) / kstars * 100)
        fk_max_errs.append(np.max(pct))
        fk_mean_errs.append(np.mean(pct))
    print(f"{'FIXED K=5 (no calibration)':<35} {'0':>8} {np.mean(fk_mean_errs):>10.2f}% {np.max(fk_max_errs):>10.2f}%")

    # Per-dataset summary
    print(f"\n--- Per-dataset one-shot error ---")
    for ds_prefix in ['cifar10', 'fashion', 'mnist', 'ddpm']:
        ds_keys = [k for k in all_results if k.startswith(ds_prefix)]
        if not ds_keys:
            continue
        ds_max = max(all_results[k]['max_abs_pct_err'] for k in ds_keys)
        ds_mean = np.mean([all_results[k]['mean_abs_pct_err'] for k in ds_keys])
        print(f"  {ds_prefix:<15} mean={ds_mean:.2f}%  max={ds_max:.2f}%  (n={len(ds_keys)} runs)")

    # Save
    out_path = ROOT / 'outputs' / 'theory' / 'oneshot_calibration.json'
    summary = {
        'alpha_universal': ALPHA_UNIVERSAL,
        'calibration_sigma': float(SIGMAS[3]),
        'n_runs': len(all_results),
        'overall_max_pct_error': float(np.max(max_errs)),
        'overall_mean_pct_error': float(np.mean(mean_errs)),
        'best_cal_sigma': float(SIGMAS[best_cal]),
        'best_cal_mean_max_err': float(best_err),
        'best_2pt_pair': [float(SIGMAS[i2]), float(SIGMAS[j2])],
        'best_2pt_mean_max_err': float(best_2pt_err),
        'oracle_7pt_mean_max_err': float(np.mean(oracle_max_errs)),
        'per_run': all_results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == '__main__':
    main()
