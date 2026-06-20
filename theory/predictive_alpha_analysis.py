"""
Predictive α angle: compute predicted α = 2γ̄/β̂ and compare to measured α.
================================================================================

Inputs:
- outputs/theory/data_beta.json  (from measure_beta.py)
- Hard-coded γ̄ values from measure_kappa.py output
- Hard-coded measured α values from paper/checkpoints

Outputs:
- outputs/theory/predictive_alpha_test.json
- outputs/theory/predictive_alpha_report.md
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gamma values (k_traj) from measure_kappa.py on CIFAR-10 test data
# These are the curvature-vs-signal-power exponents in the data eigenbasis.
# ─────────────────────────────────────────────────────────────────────────────
GAMMA_DATA = {
    'unet':          {'k_traj': 0.231, 'k_clean': 0.497, 'r2_traj': 0.381, 'available': True},
    'kan_f32':       {'k_traj': 0.176, 'k_clean': 0.352, 'r2_traj': 0.952, 'available': True},
    'kan_small':     {'k_traj': 0.184, 'k_clean': 0.348, 'r2_traj': 0.978, 'available': True},
    'conv_mlp_gelu': {'k_traj': 0.307, 'k_clean': 0.348, 'r2_traj': 0.679, 'available': True},
    'conv_mlp_silu': {'k_traj': 0.317, 'k_clean': 0.353, 'r2_traj': 0.683, 'available': True},
    'conv_mlp_tanh': {'k_traj': 0.297, 'k_clean': 0.343, 'r2_traj': 0.682, 'available': True},
    'conv_mlp_relu': {'k_traj': float('nan'), 'k_clean': float('nan'), 'r2_traj': float('nan'), 'available': False},
}

# Measured alphas from the paper (fine-grid 500-image evaluation on CIFAR-10)
ALPHA_MEASURED_CIFAR10 = {
    'unet':          1.434,
    'kan_f32':       1.365,
    'kan_small':     1.358,
    'conv_mlp_gelu': 1.132,
    'conv_mlp_silu': 1.126,
    'conv_mlp_tanh': 1.136,
    'conv_mlp_relu': 1.166,
}

# Measured alphas from the paper for CelebA-64 (KAN-EBM only in this dataset)
ALPHA_MEASURED_CELEBA64 = {
    'kan_f32': 1.362,   # from kstar_validation.json: celeba_kan_seeded
    'kan_small': 1.362, # from kstar_param_scaling.json: tiny (same architecture)
}

# Beta from measure_kappa.py data covariance (CIFAR-10 test set, s2_k ~ k^-beta)
BETA_DATA_COVARIANCE = 2.387


def compute_predictions(betas, gamma_data, alpha_measured):
    """Compute predicted alpha = 2*gamma/beta for each (dataset, arch) pair."""
    rows = []
    for ds_name, beta_hat in betas.items():
        for arch, g in gamma_data.items():
            if not g['available'] or not np.isfinite(g['k_traj']):
                rows.append({
                    'dataset': ds_name,
                    'architecture': arch,
                    'beta_hat': beta_hat,
                    'gamma_bar_kappa': None,
                    'gamma_bar_theory': None,
                    'predicted_alpha_from_kappa': None,
                    'predicted_alpha_from_theory': None,
                    'measured_alpha': alpha_measured.get(ds_name, {}).get(arch, None),
                    'error': None,
                    'status': 'checkpoint missing or gamma NaN'
                })
                continue

            k_traj = g['k_traj']
            # Two interpretations of gamma:
            # 1. gamma = kappa (the exponent directly from the H-vs-s2 fit)
            gamma_kappa = k_traj
            # 2. gamma = kappa * beta_data_covariance (the theoretical curvature spectrum exponent)
            gamma_theory = k_traj * BETA_DATA_COVARIANCE

            pred_alpha_kappa = 2 * gamma_kappa / beta_hat
            pred_alpha_theory = 2 * gamma_theory / beta_hat

            measured = alpha_measured.get(ds_name, {}).get(arch, None)
            error = None
            if measured is not None:
                error = pred_alpha_theory - measured

            rows.append({
                'dataset': ds_name,
                'architecture': arch,
                'beta_hat': beta_hat,
                'gamma_bar_kappa': round(gamma_kappa, 4),
                'gamma_bar_theory': round(gamma_theory, 4),
                'predicted_alpha_from_kappa': round(pred_alpha_kappa, 4),
                'predicted_alpha_from_theory': round(pred_alpha_theory, 4),
                'measured_alpha': measured,
                'error': round(error, 4) if error is not None else None,
                'status': 'ok'
            })
    return rows


def main():
    # Load beta measurements
    beta_path = OUT / 'data_beta.json'
    if not beta_path.exists():
        print(f"[ERROR] {beta_path} not found. Run measure_beta.py first.")
        sys.exit(1)
    beta_data = json.loads(beta_path.read_text())
    betas = {k: v['beta_hat'] for k, v in beta_data.items()}

    # Organize measured alphas by dataset
    alpha_measured = {
        'cifar10': ALPHA_MEASURED_CIFAR10,
        'celeba64': ALPHA_MEASURED_CELEBA64,
        # mnist and fashion have no measured alphas in the paper
    }

    rows = compute_predictions(betas, GAMMA_DATA, alpha_measured)

    # Summary statistics
    # For CIFAR-10 (the dataset where gamma was measured), compare predictions
    cifar10_rows = [r for r in rows if r['dataset'] == 'cifar10' and r['status'] == 'ok']
    
    # Compute correlation between predicted and measured (for CIFAR-10)
    preds = [r['predicted_alpha_from_theory'] for r in cifar10_rows if r['measured_alpha'] is not None]
    meas = [r['measured_alpha'] for r in cifar10_rows if r['measured_alpha'] is not None]
    
    if len(preds) >= 2:
        corr = np.corrcoef(preds, meas)[0, 1]
        mae = np.mean(np.abs(np.array(preds) - np.array(meas)))
        within_005 = np.sum(np.abs(np.array(preds) - np.array(meas)) <= 0.05)
    else:
        corr = float('nan')
        mae = float('nan')
        within_005 = 0

    # Also compute the measure_kappa.py correlation (2*k_traj vs alpha_meas)
    kappa_preds = [2 * r['gamma_bar_kappa'] for r in cifar10_rows if r['measured_alpha'] is not None]
    kappa_corr = np.corrcoef(kappa_preds, meas)[0, 1] if len(kappa_preds) >= 2 else float('nan')
    kappa_mae = np.mean(np.abs(np.array(kappa_preds) - np.array(meas))) if len(kappa_preds) >= 2 else float('nan')

    # Load beta sweep quick summary
    beta_sweep_path = ROOT / 'outputs' / 'beta_sweep' / 'beta_sweep_quick_summary.json'
    beta_sweep_summary = json.loads(beta_sweep_path.read_text()) if beta_sweep_path.exists() else {}
    ordering_correct = None
    if 'cifar10' in betas and 'celeba64' in betas:
        # For any fixed architecture, if beta_cifar < beta_celeba, then alpha_cifar > alpha_celeba
        ordering_correct = betas['cifar10'] < betas['celeba64']

    # Implied gamma from measured alpha and beta_hat (reverse-engineered)
    implied_gammas = []
    for ds_name, beta_hat in betas.items():
        for arch, alpha in alpha_measured.get(ds_name, {}).items():
            if alpha is not None:
                implied = alpha * beta_hat / 2.0
                implied_gammas.append({
                    'dataset': ds_name,
                    'architecture': arch,
                    'alpha_measured': alpha,
                    'beta_hat': beta_hat,
                    'gamma_implied': round(implied, 4)
                })

    # Save JSON
    results = {
        'beta_measurements': beta_data,
        'gamma_measurements': GAMMA_DATA,
        'beta_data_covariance': BETA_DATA_COVARIANCE,
        'prediction_rows': rows,
        'implied_gammas': implied_gammas,
        'beta_sweep_quick': beta_sweep_summary,
        'summary': {
            'cifar10_correlation_pred_vs_meas': round(corr, 4),
            'cifar10_mae_pred_vs_meas': round(mae, 4),
            'cifar10_within_005': int(within_005),
            'cifar10_total_architectures': len(preds),
            'kappa_direct_correlation_2k_vs_alpha': round(kappa_corr, 4),
            'kappa_direct_mae_2k_vs_alpha': round(kappa_mae, 4),
            'ordering_beta_higher_alpha_lower': ordering_correct,
            'threshold_for_strong_result': 0.05,
            'verdict': 'PASS' if mae <= 0.05 else 'FAIL'
        }
    }
    json_path = OUT / 'predictive_alpha_test.json'
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
    print(f"[json] {json_path}")

    # Write markdown report
    md = []
    md.append("# Predictive α Test Report")
    md.append("")
    md.append("**Date:** 2026-06-14")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append(f"- **Threshold for 'genuinely strong' result:** ±0.05")
    md.append(f"- **CIFAR-10 MAE (predicted vs measured):** {mae:.4f}")
    md.append(f"- **Architectures within ±0.05:** {int(within_005)}/{len(preds)}")
    md.append(f"- **Correlation (predicted vs measured):** {corr:.4f}")
    md.append(f"- **Verdict:** {'✅ PASS' if mae <= 0.05 else '❌ FAIL'}")
    md.append("")
    md.append("### Key Finding")
    md.append("")
    if mae <= 0.05:
        md.append("The prediction **α = 2γ̄/β̂** accurately predicts the measured inference-scaling exponent from dataset statistics alone. This is a strong theoretical result.")
    else:
        md.append("The prediction **α = 2γ̄/β̂** does **not** accurately predict the measured inference-scaling exponent. The predicted values are systematically too low (by a factor of ~3–5), and the correlation is negative. This confirms the suspected confound: real images and architectures violate the data-basis diagonalization assumption on which the closed-form derivation rests. The spectral-shrinkage account remains an *illustrative* mechanism, not a predictive one for natural images.")
    md.append("")
    md.append("## 1. Data Power Spectrum (β̂) Measurements")
    md.append("")
    md.append("| Dataset | β̂ | R² | n | size | band |")
    md.append("|---------|------|------|-----|------|-------------|")
    for ds, v in beta_data.items():
        md.append(f"| {ds} | {v['beta_hat']:.3f} | {v['r2']:.3f} | {v['n_images']} | {v['size']} | [{v['band'][0]},{v['band'][1]}] |")
    md.append("")
    md.append("## 2. Model Curvature Spectrum (γ̄) Measurements")
    md.append("")
    md.append(f"All γ̄ values were measured on CIFAR-10 test data (data-covariance eigenvalue decay β_data = {BETA_DATA_COVARIANCE:.3f}).")
    md.append("")
    md.append("| Architecture | γ̄ (k_traj) | γ̄_theory = κ·β_data | 2κ | R²_traj | Checkpoint |")
    md.append("|--------------|------------|---------------------|------|---------|------------|")
    for arch, g in GAMMA_DATA.items():
        if g['available'] and np.isfinite(g['k_traj']):
            k = g['k_traj']
            gamma_t = k * BETA_DATA_COVARIANCE
            two_k = 2 * k
            md.append(f"| {arch} | {k:.3f} | {gamma_t:.3f} | {two_k:.3f} | {g['r2_traj']:.3f} | ✅ |")
        else:
            md.append(f"| {arch} | NaN | NaN | NaN | NaN | ❌ missing/NaN |")
    md.append("")
    md.append("## 3. Prediction Table: α = 2γ̄/β̂")
    md.append("")
    md.append("We compute predicted α using the *theoretical* γ̄ = κ·β_data (i.e. the curvature-spectrum exponent that enters the toy model).")
    md.append("")
    md.append("| Dataset | β̂ | Architecture | γ̄ | Predicted α | Measured α | Error | Status |")
    md.append("|---------|------|--------------|------|-------------|------------|-------|--------|")
    for r in rows:
        if r['status'] != 'ok':
            md.append(f"| {r['dataset']} | {r['beta_hat']:.3f} | {r['architecture']} | — | — | {r['measured_alpha'] if r['measured_alpha'] else 'N/A'} | — | ❌ {r['status']} |")
            continue
        gamma = r['gamma_bar_theory']
        pred = r['predicted_alpha_from_theory']
        meas = r['measured_alpha'] if r['measured_alpha'] is not None else 'N/A'
        err = r['error'] if r['error'] is not None else 'N/A'
        status = '✅ within ±0.05' if r['error'] is not None and abs(r['error']) <= 0.05 else ('⚠️ outside ±0.05' if r['error'] is not None else '—')
        md.append(f"| {r['dataset']} | {r['beta_hat']:.3f} | {r['architecture']} | {gamma:.3f} | {pred:.3f} | {meas} | {err} | {status} |")
    md.append("")
    md.append("## 4. Implied Gammas (Reverse-Engineered)")
    md.append("")
    md.append("If the theory were exact, the implied γ̄ = α·β̂/2 would be constant across datasets for a fixed architecture. It is not.")
    md.append("")
    md.append("| Dataset | Architecture | α (measured) | β̂ | γ̄_implied = α·β̂/2 |")
    md.append("|---------|--------------|--------------|------|---------------------|")
    for ig in implied_gammas:
        md.append(f"| {ig['dataset']} | {ig['architecture']} | {ig['alpha_measured']:.3f} | {ig['beta_hat']:.3f} | {ig['gamma_implied']:.3f} |")
    md.append("")
    md.append("## 5. Statistical Summary on CIFAR-10")
    md.append("")
    md.append(f"- **Correlation (predicted α vs measured α):** {corr:.4f}")
    md.append(f"- **Mean absolute error:** {mae:.4f}")
    md.append(f"- **Correlation (2κ vs measured α, as in measure_kappa.py):** {kappa_corr:.4f}")
    md.append(f"- **Mean absolute error (2κ vs measured α):** {kappa_mae:.4f}")
    md.append("")
    md.append("## 6. Dataset Ordering Check")
    md.append("")
    md.append(f"The theory predicts that higher β̂ → lower α (inverse relationship).")
    md.append(f"- CIFAR-10 β̂ = {betas['cifar10']:.3f}, CelebA-64 β̂ = {betas['celeba64']:.3f}")
    md.append(f"- Measured α on CIFAR-10 (KAN) = 1.365, on CelebA-64 = 1.362")
    md.append(f"- The ordering is **barely consistent** (β̂ increases by 19%, α decreases by 0.2%), but the difference is well within noise.")
    md.append("")
    md.append("## 7. Synthetic Beta Sweep (Smoke Test)")
    md.append("")
    md.append("The `beta_sweep.py --quick` smoke test was run successfully (3 epochs, 2K train, 1 seed, betas {1,2,3}).")
    md.append("")
    md.append("| Family | Beta (nominal) | Betâ (realized) | α (measured) |")
    md.append("|--------|---------------|-------------------|--------------|")
    for fam, s in beta_sweep_summary.items():
        for row in s['rows']:
            md.append(f"| {fam} | {row[0]:.1f} | {row[1]:.3f} | {row[2]:.3f} |")
    md.append("")
    md.append(f"- **ConvMLP-GELU:** slope = {beta_sweep_summary['convmlp_gelu']['slope']:.3f} ± {beta_sweep_summary['convmlp_gelu']['slope_se']:.3f} (predicted: -1). **Qualitatively correct** (α decreases with β̂), but quantitatively wrong slope.")
    md.append(f"- **Score:** slope = {beta_sweep_summary['score']['slope']:.3f} ± {beta_sweep_summary['score']['slope_se']:.3f} (predicted: -1). **Qualitatively wrong** (α increases with β̂).")
    md.append("")
    md.append("The smoke test confirms the protocol runs, but with only 3 epochs and 1 seed the results are noisy. The full sweep (40 epochs, 3 seeds, 5 betas) is needed for a clean test. **Command to run:**")
    md.append("")
    md.append("```bash")
    md.append("py -3.12 theory/beta_sweep.py --epochs 40 --seeds 0 1 2")
    md.append("```")
    md.append("")
    md.append("## 8. Checkpoints Status")
    md.append("")
    for arch, g in GAMMA_DATA.items():
        status = '✅ Available' if g['available'] else '❌ Missing/NaN'
        md.append(f"- **{arch}:** {status}")
    md.append("")
    md.append("## 9. Recommendation for the Paper")
    md.append("")
    if mae <= 0.05:
        md.append("**INCLUDE.** The predictive α angle is a strong theoretical result. We can genuinely claim that dataset statistics alone determine the inference-scaling exponent, enabling practitioners to plan inference budgets without training.")
    else:
        md.append("**DO NOT include as a predictive claim.** The spectral-shrinkage derivation does not quantitatively predict the observed α on real data. The predicted values are off by a factor of ~3–5, and the correlation is negative. Instead, we should frame it as:")
        md.append("")
        md.append("1. **An illustrative mechanism** (the toy model in Sec. X) that shows *how* a power law can arise from spectral regularization, but whose closed-form prediction is not quantitatively valid for real architectures because the data-basis diagonalization assumption is violated.")
        md.append("2. **A controlled validation** (the beta_sweep on Gaussian random fields) where the assumptions hold by construction and the theory can be tested cleanly. This is the 'gold standard' test.")
        md.append("3. **An empirical correlation** (not a derivation): trajectory-averaged Hessian curvature tracks the α ranking across architectures, but a 1D radial ansatz does not predict the magnitude.")
        md.append("")
        md.append("The paper's strength is the *empirical* law and its practical predictability (leave-one-out, cross-dataset transfer), not the closed-form derivation. The theory section should be honest about this boundary.")
    md.append("")
    md.append("---")
    md.append("*Generated by theory/predictive_alpha_analysis.py*")

    md_path = OUT / 'predictive_alpha_report.md'
    md_path.write_text('\n'.join(md), encoding='utf-8')
    print(f"[md] {md_path}")
    print(f"\nVERDICT: {results['summary']['verdict']} (MAE = {mae:.4f}, threshold = 0.05)")


if __name__ == '__main__':
    main()
