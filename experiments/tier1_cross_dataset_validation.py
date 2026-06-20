"""
tier1_cross_dataset_validation.py
=================================
Cross-dataset predictive validation for the K* scaling law.

Question: Can the law fit on dataset A (CIFAR-10) predict K*(sigma) on
dataset B (CelebA-64) without retraining?

Three protocols:
  (1) Full transfer:  use (C_A, alpha_A) directly on B
  (2) One-anchor:     use alpha_A, fit C from a single sigma on B
  (3) Native:         baseline -- fit (C, alpha) on B itself

For each, we measure:
  - K* prediction error per sigma (predicted K* vs observed K*)
  - On CIFAR (where PSNR-per-K is available), PSNR recovery vs oracle K*

Inputs:
  outputs/fine_grid_kstar/fine_grid_kstar.json     (CIFAR KAN fine-grid)
  outputs/finalization/kstar_validation/kstar_validation.json  (CelebA KAN)

Output:
  outputs/tier1_cross_dataset/cross_dataset_validation.json
  outputs/tier1_cross_dataset/cross_dataset_validation.txt
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'outputs' / 'tier1_cross_dataset'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CIFAR_FINE = ROOT / 'outputs' / 'fine_grid_kstar' / 'fine_grid_kstar.json'
KSTAR_VAL = ROOT / 'outputs' / 'finalization' / 'kstar_validation' / 'kstar_validation.json'


def fit_powerlaw(sigmas, Ks):
    xs = np.log(np.asarray(sigmas, float))
    ys = np.log(np.asarray(Ks, float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(math.exp(intercept)), float(slope), float(r2)


def predict_K(sigma, C, alpha):
    return C * (sigma ** alpha)


def round_to_grid(K_continuous, K_grid):
    """Round predicted K to nearest available step on the evaluation grid."""
    return int(min(K_grid, key=lambda k: abs(k - K_continuous)))


def psnr_recovery(predicted_K, oracle_K, psnr_per_K):
    """
    PSNR at predicted K divided by PSNR at oracle K, as percentage.
    psnr_per_K is 1-indexed list (psnr_per_K[k-1] is PSNR after step k).
    """
    pK = max(1, min(predicted_K, len(psnr_per_K)))
    oK = max(1, min(oracle_K, len(psnr_per_K)))
    return 100.0 * psnr_per_K[pK - 1] / psnr_per_K[oK - 1]


def main():
    # ─── Load CIFAR-10 KAN fine-grid (K=1..30, 500 imgs) ─────────────────────
    cifar = json.loads(CIFAR_FINE.read_text())
    cifar_sigmas = cifar['sigmas']
    cifar_K_obs = []  # observed K* per sigma (rounded mean)
    cifar_K_grid = list(range(1, cifar['K_max'] + 1))
    cifar_psnr = {}  # sigma -> psnr_per_K list
    for s in cifar_sigmas:
        rec = cifar['per_sigma'][f'{s:.3f}']
        cifar_K_obs.append(round(rec['kstar_mean']))
        cifar_psnr[s] = rec['mean_psnr_per_K']
    cifar_C, cifar_alpha, cifar_r2 = fit_powerlaw(cifar_sigmas, cifar_K_obs)
    cifar_alpha_paper = cifar['power_law']['fine_grid']['alpha']
    cifar_C_paper = cifar['power_law']['fine_grid']['C']
    print(f"CIFAR-10 KAN  (110K, fine-grid): alpha={cifar_alpha_paper:.4f}  C={cifar_C_paper:.3f}  R^2={cifar['power_law']['fine_grid']['R2']:.4f}")

    # ─── Load CelebA-64 KAN ──────────────────────────────────────────────────
    val = json.loads(KSTAR_VAL.read_text())
    celeba = val['experiments']['celeba_kan_seeded']
    celeba_sigmas = sorted(float(k) for k in celeba['mean'].keys())
    celeba_K_obs = [int(celeba['mean'][f'{s}'.rstrip('0').rstrip('.') or '0']) for s in celeba_sigmas]
    # Some sigmas are like 0.05, 0.1, etc. Use the actual key:
    celeba_K_obs = []
    for s in celeba_sigmas:
        # Find matching key (string formatting may differ)
        matched = None
        for k in celeba['mean']:
            if abs(float(k) - s) < 1e-6:
                matched = k
                break
        celeba_K_obs.append(int(celeba['mean'][matched]))
    celeba_C_native = celeba['fit']['C']
    celeba_alpha_native = celeba['fit']['alpha']
    celeba_r2_native = celeba['fit']['r_squared']
    celeba_K_grid = val['K_list']  # [1,2,3,5,7,10,15,20,30,50]
    print(f"CelebA-64 KAN (32K, coarse): alpha={celeba_alpha_native:.4f}  C={celeba_C_native:.3f}  R^2={celeba_r2_native:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-dataset prediction (use cifar paper-quality fit)
    # ═══════════════════════════════════════════════════════════════════════
    record = {
        'cifar_native': {
            'sigmas': cifar_sigmas,
            'K_observed': cifar_K_obs,
            'alpha': cifar_alpha_paper,
            'C': cifar_C_paper,
            'r2': cifar['power_law']['fine_grid']['R2'],
        },
        'celeba_native': {
            'sigmas': celeba_sigmas,
            'K_observed': celeba_K_obs,
            'alpha': celeba_alpha_native,
            'C': celeba_C_native,
            'r2': celeba_r2_native,
        },
        'predictions': {},
    }

    # ── (A) CIFAR-fit -> predict CelebA ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("(A) CIFAR-trained fit -> predict CelebA K*")
    print("=" * 80)
    print(f"{'sigma':>7} {'K_obs':>6} {'K_pred_full':>11} {'K_pred_1anchor':>15} {'K_pred_alpha_only':>17}")

    a_results = {'full_transfer': [], 'one_anchor': [], 'alpha_only_native_C': []}
    # One-anchor: re-fit C using only celeba sigma=0.05 anchor
    anchor_sigma = celeba_sigmas[0]  # 0.05
    anchor_K = celeba_K_obs[0]
    C_oneanchor = anchor_K / (anchor_sigma ** cifar_alpha_paper)

    for s, K_obs in zip(celeba_sigmas, celeba_K_obs):
        K_full = predict_K(s, cifar_C_paper, cifar_alpha_paper)
        K_one = predict_K(s, C_oneanchor, cifar_alpha_paper)
        K_alpha = predict_K(s, celeba_C_native, cifar_alpha_paper)  # alpha from cifar, C from celeba native
        a_results['full_transfer'].append({
            'sigma': s, 'K_obs': K_obs,
            'K_pred': K_full, 'K_pred_rounded': round_to_grid(K_full, celeba_K_grid),
            'abs_error': abs(K_full - K_obs),
            'rel_error_pct': 100.0 * abs(K_full - K_obs) / K_obs,
        })
        a_results['one_anchor'].append({
            'sigma': s, 'K_obs': K_obs,
            'K_pred': K_one, 'K_pred_rounded': round_to_grid(K_one, celeba_K_grid),
            'abs_error': abs(K_one - K_obs),
            'rel_error_pct': 100.0 * abs(K_one - K_obs) / K_obs,
        })
        a_results['alpha_only_native_C'].append({
            'sigma': s, 'K_obs': K_obs,
            'K_pred': K_alpha, 'K_pred_rounded': round_to_grid(K_alpha, celeba_K_grid),
            'abs_error': abs(K_alpha - K_obs),
            'rel_error_pct': 100.0 * abs(K_alpha - K_obs) / K_obs,
        })
        print(f"{s:>7.3f} {K_obs:>6d} {K_full:>11.2f} {K_one:>15.2f} {K_alpha:>17.2f}")
    # Aggregates
    for k, lst in a_results.items():
        mae = float(np.mean([d['abs_error'] for d in lst]))
        mre = float(np.mean([d['rel_error_pct'] for d in lst]))
        print(f"  [{k}]  MAE={mae:.3f} steps   MRE={mre:.2f}%")
    record['predictions']['cifar_to_celeba'] = a_results

    # ── (B) CelebA-fit -> predict CIFAR (with PSNR recovery) ────────────────
    print("\n" + "=" * 80)
    print("(B) CelebA-trained fit -> predict CIFAR K*  (PSNR recovery available)")
    print("=" * 80)
    print(f"{'sigma':>7} {'K_obs':>6} {'K_pred_full':>11} {'K_pred_1anchor':>15} {'PSNR rec full':>14} {'PSNR rec 1anchor':>17}")

    b_results = {'full_transfer': [], 'one_anchor': [], 'alpha_only_native_C': []}
    anchor_sigma = cifar_sigmas[0]
    anchor_K = cifar_K_obs[0]
    C_oneanchor_b = anchor_K / (anchor_sigma ** celeba_alpha_native)

    for s, K_obs in zip(cifar_sigmas, cifar_K_obs):
        psnr_curve = cifar_psnr[s]
        oracle_K = int(np.argmax(psnr_curve)) + 1

        K_full = predict_K(s, celeba_C_native, celeba_alpha_native)
        K_one = predict_K(s, C_oneanchor_b, celeba_alpha_native)
        K_alpha = predict_K(s, cifar_C_paper, celeba_alpha_native)

        K_full_r = round_to_grid(K_full, cifar_K_grid)
        K_one_r = round_to_grid(K_one, cifar_K_grid)
        K_alpha_r = round_to_grid(K_alpha, cifar_K_grid)
        rec_full = psnr_recovery(K_full_r, oracle_K, psnr_curve)
        rec_one = psnr_recovery(K_one_r, oracle_K, psnr_curve)
        rec_alpha = psnr_recovery(K_alpha_r, oracle_K, psnr_curve)

        b_results['full_transfer'].append({
            'sigma': s, 'K_obs': K_obs, 'oracle_K': oracle_K,
            'K_pred': K_full, 'K_pred_rounded': K_full_r,
            'psnr_recovery_pct': rec_full,
        })
        b_results['one_anchor'].append({
            'sigma': s, 'K_obs': K_obs, 'oracle_K': oracle_K,
            'K_pred': K_one, 'K_pred_rounded': K_one_r,
            'psnr_recovery_pct': rec_one,
        })
        b_results['alpha_only_native_C'].append({
            'sigma': s, 'K_obs': K_obs, 'oracle_K': oracle_K,
            'K_pred': K_alpha, 'K_pred_rounded': K_alpha_r,
            'psnr_recovery_pct': rec_alpha,
        })
        print(f"{s:>7.3f} {K_obs:>6d} {K_full:>11.2f} {K_one:>15.2f} {rec_full:>13.2f}% {rec_one:>16.2f}%")
    for k, lst in b_results.items():
        mean_rec = float(np.mean([d['psnr_recovery_pct'] for d in lst]))
        worst_rec = float(np.min([d['psnr_recovery_pct'] for d in lst]))
        print(f"  [{k}]  mean PSNR recovery = {mean_rec:.2f}%   worst = {worst_rec:.2f}%")
    record['predictions']['celeba_to_cifar'] = b_results

    # ── Summary stats (the headline numbers for the paper) ─────────────────
    print("\n" + "=" * 80)
    print("HEADLINE NUMBERS")
    print("=" * 80)
    headline = {}

    # CIFAR alpha vs CelebA alpha (architectural class transfer):
    diff_alpha = abs(cifar_alpha_paper - celeba_alpha_native)
    headline['alpha_cifar_vs_celeba'] = {
        'cifar_alpha': cifar_alpha_paper,
        'celeba_alpha': celeba_alpha_native,
        'absolute_difference': diff_alpha,
        'relative_difference_pct': 100.0 * diff_alpha / cifar_alpha_paper,
    }
    print(f"alpha(CIFAR-KAN, 110K) = {cifar_alpha_paper:.4f}")
    print(f"alpha(CelebA-KAN, 32K) = {celeba_alpha_native:.4f}")
    print(f"Difference: {diff_alpha:.4f} ({100*diff_alpha/cifar_alpha_paper:.2f}% of CIFAR alpha)")

    # CelebA alpha is inside CIFAR's bootstrap CI?
    cifar_ci = cifar['power_law']['fine_grid']['ci95_alpha']
    inside_ci = cifar_ci[0] <= celeba_alpha_native <= cifar_ci[1]
    headline['celeba_alpha_inside_cifar_ci'] = {
        'cifar_ci95': cifar_ci,
        'celeba_alpha': celeba_alpha_native,
        'inside': inside_ci,
    }
    print(f"CIFAR 95% CI for alpha: [{cifar_ci[0]:.4f}, {cifar_ci[1]:.4f}]")
    print(f"CelebA alpha {'IS' if inside_ci else 'is NOT'} inside CIFAR's CI")

    # Best protocol (one-anchor) PSNR recovery on CIFAR:
    one_anchor_recoveries = [d['psnr_recovery_pct'] for d in b_results['one_anchor']]
    headline['celeba_to_cifar_one_anchor'] = {
        'mean_psnr_recovery_pct': float(np.mean(one_anchor_recoveries)),
        'worst_psnr_recovery_pct': float(np.min(one_anchor_recoveries)),
    }
    print(f"CelebA->CIFAR (one-anchor): mean PSNR recovery = "
          f"{np.mean(one_anchor_recoveries):.2f}%, worst = {np.min(one_anchor_recoveries):.2f}%")

    # CelebA prediction K* MAE
    one_anchor_K_errors = [d['abs_error'] for d in a_results['one_anchor']]
    headline['cifar_to_celeba_one_anchor'] = {
        'mean_K_error': float(np.mean(one_anchor_K_errors)),
        'worst_K_error': float(np.max(one_anchor_K_errors)),
    }
    print(f"CIFAR->CelebA (one-anchor): mean |K* error| = "
          f"{np.mean(one_anchor_K_errors):.2f} steps, worst = {np.max(one_anchor_K_errors):.2f}")

    record['headline'] = headline

    # ─── Save ──────────────────────────────────────────────────────────────
    out_json = OUT_DIR / 'cross_dataset_validation.json'
    out_json.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {out_json}")

    # Plain-text summary for the paper
    lines = [
        "Cross-dataset Predictive Validation",
        "=" * 75,
        "",
        f"  CIFAR-10 KAN  (110K, fine-grid):  alpha = {cifar_alpha_paper:.4f}  C = {cifar_C_paper:.2f}  R^2 = {cifar['power_law']['fine_grid']['R2']:.4f}",
        f"  CelebA-64 KAN (32K,  coarse):     alpha = {celeba_alpha_native:.4f}  C = {celeba_C_native:.2f}  R^2 = {celeba_r2_native:.4f}",
        "",
        f"  |Delta alpha| = {diff_alpha:.4f}  ({100*diff_alpha/cifar_alpha_paper:.2f}% relative)",
        f"  CelebA alpha INSIDE CIFAR 95% CI [{cifar_ci[0]:.3f}, {cifar_ci[1]:.3f}]: {inside_ci}",
        "",
        "Cross-dataset prediction protocols:",
        "",
        "  (A) CIFAR fit -> CelebA K* prediction:",
    ]
    for proto, lst in a_results.items():
        mae = float(np.mean([d['abs_error'] for d in lst]))
        mre = float(np.mean([d['rel_error_pct'] for d in lst]))
        lines.append(f"    [{proto:>20}] MAE = {mae:.2f} steps,  MRE = {mre:.2f}%")
    lines.append("")
    lines.append("  (B) CelebA fit -> CIFAR K* prediction (with PSNR recovery):")
    for proto, lst in b_results.items():
        mean_rec = float(np.mean([d['psnr_recovery_pct'] for d in lst]))
        worst_rec = float(np.min([d['psnr_recovery_pct'] for d in lst]))
        lines.append(f"    [{proto:>20}] mean PSNR recovery = {mean_rec:.2f}%   worst = {worst_rec:.2f}%")
    out_txt = OUT_DIR / 'cross_dataset_validation.txt'
    out_txt.write_text("\n".join(lines))
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
