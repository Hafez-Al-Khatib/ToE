"""
Predictive-α analysis: can we predict the inference-depth exponent from
the data's power spectrum BEFORE training any model?
================================================================================
The theory says α = 2γ/β, where β is the data power-spectrum exponent and
γ is the model's learned curvature spectrum exponent. If γ is roughly
constant across datasets (at fixed architecture/recipe), then:

    α_predicted = 2 γ̄ / β̂

where β̂ is measured cheaply via FFT and γ̄ is calibrated once from a single
reference dataset.

This analysis:
  1. Measures β̂ for all available datasets (CIFAR-10, CelebA-64, Fashion, MNIST, CBSD68).
  2. Uses the existing measured α values (from tier0/scaleup64) to compute implied γ.
  3. Tests constancy of γ across datasets (theory prediction).
  4. Calibrates γ̄ from ONE reference dataset (CIFAR-10), predicts α for the others.
  5. Reports prediction error.

If prediction error is < 10%, the paper gains: "measure the FFT, predict
the inference budget, no training required." That is a DESIGN TOOL, not
just an observation.

Run:  py -3.12 theory/predictive_alpha.py

Inputs:
  - outputs/theory/data_beta.json            (from measure_beta.py)
  - outputs/tier0_seeds/tier0_results.json   (measured α on CIFAR-10)
  - outputs/theory/scaleup64_results.json    (measured α on CelebA-64)

Outputs:
  - outputs/theory/predictive_alpha.json
  - outputs/theory/predictive_alpha.pdf/.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'theory'


def load_measured_alphas():
    """Collect measured α values from all completed experiments."""
    alphas = {}

    # CIFAR-10 32px: from tier0_seeds (matched recipe, 3 seeds × 4 heads + score)
    tier0 = ROOT / 'outputs' / 'tier0_seeds' / 'tier0_results.json'
    if tier0.exists():
        t = json.loads(tier0.read_text())
        all_a = []
        for arch, r in t.items():
            for a in r.get('alpha', []):
                if np.isfinite(a):
                    all_a.append(a)
        if all_a:
            alphas['cifar10'] = {
                'alpha_mean': float(np.mean(all_a)),
                'alpha_std':  float(np.std(all_a, ddof=1)),
                'n_runs':     len(all_a),
                'source':     'tier0_seeds (4 EBM heads × 3 seeds)',
                'resolution': '32px',
            }

    # CelebA-64: from scaleup64
    sc64 = ROOT / 'outputs' / 'theory' / 'scaleup64_results.json'
    if sc64.exists():
        s = json.loads(sc64.read_text())
        all_a = []
        for arch, r in s.items():
            for a in r.get('alpha', []):
                if np.isfinite(a):
                    all_a.append(a)
        if all_a:
            alphas['celeba64'] = {
                'alpha_mean': float(np.mean(all_a)),
                'alpha_std':  float(np.std(all_a, ddof=1)),
                'n_runs':     len(all_a),
                'source':     'scaleup64 (score + ConvMLP + KAN)',
                'resolution': '64px',
            }

    # FashionMNIST 28px
    fash = ROOT / 'outputs' / 'theory' / 'fashion_results.json'
    if fash.exists():
        f = json.loads(fash.read_text())
        all_a = []
        for arch, r in f.items():
            for a in r.get('alpha', []):
                if np.isfinite(a):
                    all_a.append(a)
        if all_a:
            alphas['fashion'] = {
                'alpha_mean': float(np.mean(all_a)),
                'alpha_std':  float(np.std(all_a, ddof=1)),
                'n_runs':     len(all_a),
                'source':     'fashion_mnist_kstar (ConvMLP + Score x 3 seeds)',
                'resolution': '28px',
            }

    # MNIST 28px
    mn = ROOT / 'outputs' / 'theory' / 'mnist_results.json'
    if mn.exists():
        m = json.loads(mn.read_text())
        all_a = []
        for arch, r in m.items():
            for a in r.get('alpha', []):
                if np.isfinite(a):
                    all_a.append(a)
        if all_a:
            alphas['mnist'] = {
                'alpha_mean': float(np.mean(all_a)),
                'alpha_std':  float(np.std(all_a, ddof=1)),
                'n_runs':     len(all_a),
                'source':     'fashion_mnist_kstar (ConvMLP + Score x 3 seeds)',
                'resolution': '28px',
            }

    return alphas


def load_betas():
    """Load β̂ values from measure_beta.py output."""
    p = OUT / 'data_beta.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found — run measure_beta.py first')
    raw = json.loads(p.read_text())
    betas = {}
    for name, r in raw.items():
        if name.startswith('grf:'):
            continue
        if np.isfinite(r.get('beta_hat', float('nan'))):
            betas[name] = r
    return betas


def main():
    print("=" * 70)
    print("PREDICTIVE-ALPHA ANALYSIS")
    print("Can we predict the inference-depth exponent from data statistics alone?")
    print("=" * 70)

    alphas = load_measured_alphas()
    betas = load_betas()

    print(f"\n--- Measured alpha values ---")
    for ds, r in alphas.items():
        print(f"  {ds:>12}: alpha = {r['alpha_mean']:.3f} +/- {r['alpha_std']:.3f}"
              f"  ({r['n_runs']} runs, {r['resolution']})")

    print(f"\n--- Measured beta_hat (data power spectrum) ---")
    for ds, r in betas.items():
        print(f"  {ds:>12}: beta_hat = {r['beta_hat']:.3f}  (R2={r['r2']:.4f},"
              f" {r['n_images']} imgs @ {r['size']}px)")

    # Step 1: compute implied gamma = alpha * beta_hat / 2
    print(f"\n--- Implied gamma = alpha * beta_hat / 2  (theory: constant across datasets) ---")
    gammas = {}
    for ds in sorted(set(alphas) & set(betas)):
        a = alphas[ds]['alpha_mean']
        b = betas[ds]['beta_hat']
        g = a * b / 2.0
        gammas[ds] = g
        print(f"  {ds:>12}: alpha={a:.3f} * beta_hat={b:.3f} / 2 = gamma={g:.3f}")

    if len(gammas) < 2:
        print("\nNeed measured alpha for at least 2 datasets to test gamma-constancy.")
        print("Run scaleup_64.py or train on another dataset first.")
        return

    g_vals = list(gammas.values())
    g_mean = float(np.mean(g_vals))
    g_std = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else 0
    g_cv = g_std / g_mean if g_mean > 0 else float('nan')
    print(f"\n  gamma_bar = {g_mean:.3f} +/- {g_std:.3f}  (CV = {g_cv:.1%})")
    print(f"  gamma constancy: {'GOOD' if g_cv < 0.15 else 'MARGINAL' if g_cv < 0.25 else 'POOR'}"
          f" (CV < 15% is good)")

    # Step 2: leave-one-out prediction
    print(f"\n--- Leave-one-out alpha prediction ---")
    print(f"  For each dataset: calibrate gamma from all OTHER datasets, predict alpha.")
    ds_with_both = sorted(set(alphas) & set(betas))
    predictions = {}
    for held_out in ds_with_both:
        train_ds = [d for d in ds_with_both if d != held_out]
        gamma_cal = float(np.mean([gammas[d] for d in train_ds]))
        beta_held = betas[held_out]['beta_hat']
        alpha_pred = 2 * gamma_cal / beta_held
        alpha_true = alphas[held_out]['alpha_mean']
        err = alpha_pred - alpha_true
        rel_err = err / alpha_true
        predictions[held_out] = {
            'alpha_true': alpha_true, 'alpha_pred': alpha_pred,
            'error': err, 'rel_error': rel_err,
            'beta_hat': beta_held, 'gamma_cal': gamma_cal,
            'calibrated_on': train_ds,
        }
        ok = 'OK' if abs(rel_err) < 0.10 else '~' if abs(rel_err) < 0.20 else 'X'
        print(f"  {ok} {held_out:>12}: alpha_pred={alpha_pred:.3f}  alpha_true={alpha_true:.3f}"
              f"  error={err:+.3f} ({rel_err:+.1%})"
              f"  [calibrated on {', '.join(train_ds)}, gamma_cal={gamma_cal:.3f}]")

    # Step 3: predict alpha for datasets WITHOUT measured alpha
    print(f"\n--- Predicted alpha for unmeasured datasets (using gamma_bar={g_mean:.3f}) ---")
    unmeasured_preds = {}
    for ds in sorted(betas):
        if ds in alphas:
            continue
        b = betas[ds]['beta_hat']
        a_pred = 2 * g_mean / b
        unmeasured_preds[ds] = {
            'alpha_pred': a_pred, 'beta_hat': b, 'gamma_used': g_mean,
            'status': 'UNTESTED - needs K-sweep on this dataset to validate',
        }
        print(f"  ? {ds:>12}: beta_hat={b:.3f} -> alpha_pred = 2*{g_mean:.3f}/{b:.3f} = {a_pred:.3f}"
              f"  (UNTESTED)")

    # Step 4: summary assessment
    if predictions:
        errs = [abs(p['rel_error']) for p in predictions.values()]
        max_err = max(errs)
        mean_err = float(np.mean(errs))
        print(f"\n{'=' * 70}")
        print(f"SUMMARY")
        print(f"  LOO prediction: mean |error| = {mean_err:.1%}, max = {max_err:.1%}")
        print(f"  gamma constancy: CV = {g_cv:.1%}")
        if max_err < 0.10:
            verdict = ("STRONG: alpha is predictable from beta_hat alone within 10%.\n"
                       "  -> Paper upgrade: 'measure the FFT, predict the inference budget,\n"
                       "    no training required.'")
        elif max_err < 0.20:
            verdict = ("MODERATE: alpha is approximately predictable from beta_hat (~10-20% error).\n"
                       "  -> Paper upgrade: 'data spectrum provides a useful prior on alpha,\n"
                       "    5-point calibration refines it.'")
        else:
            verdict = ("WEAK: alpha is not reliably predictable from beta_hat alone (>20% error).\n"
                       "  -> The beta_hat relationship is directionally correct but not a design tool.\n"
                       "    The 5-point calibration remains the practical contribution.")
        print(f"  Verdict: {verdict}")
        print(f"{'=' * 70}")

    # Save
    out = {
        'gammas': gammas, 'gamma_mean': g_mean, 'gamma_std': g_std, 'gamma_cv': g_cv,
        'loo_predictions': {k: {kk: (vv if not isinstance(vv, list) else vv)
                                for kk, vv in v.items()}
                            for k, v in predictions.items()},
        'unmeasured_predictions': unmeasured_preds,
        'measured_alphas': {k: {kk: vv for kk, vv in v.items()}
                           for k, v in alphas.items()},
        'measured_betas': {k: {kk: vv for kk, vv in v.items() if kk != 'note'}
                          for k, v in betas.items()},
    }
    (OUT / 'predictive_alpha.json').write_text(json.dumps(out, indent=2))
    print(f"\n[json] {OUT / 'predictive_alpha.json'}")

    make_figure(out)


def make_figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Panel A: α vs 1/β̂ — should be linear through origin
    ax = axes[0]
    for ds, g in out['gammas'].items():
        b = out['measured_betas'][ds]['beta_hat']
        a = out['measured_alphas'][ds]['alpha_mean']
        ax.plot(1/b, a, 'o', ms=7, label=f"{ds} (measured)")
    for ds, p in out.get('unmeasured_predictions', {}).items():
        ax.plot(1/p['beta_hat'], p['alpha_pred'], 's', ms=6, alpha=0.5,
                label=f"{ds} (predicted)")
    # theory line
    brange = np.linspace(0.1, 0.55, 50)
    ax.plot(brange, 2 * out['gamma_mean'] * brange, 'k--', lw=1,
            label=f"α = 2γ̄/β̂  (γ̄={out['gamma_mean']:.2f})")
    ax.set_xlabel('1/β̂  (inverse data spectral exponent)')
    ax.set_ylabel('α  (inference-depth exponent)')
    ax.set_title('α vs 1/β̂')
    ax.legend(fontsize=7, frameon=False)

    # Panel B: implied γ bar chart — should be constant
    ax = axes[1]
    ds_names = list(out['gammas'].keys())
    g_vals = [out['gammas'][d] for d in ds_names]
    colors = ['#00b4d8' if d == 'cifar10' else '#2dd48a' for d in ds_names]
    ax.bar(ds_names, g_vals, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(out['gamma_mean'], color='white', ls='--', lw=1,
               label=f"γ̄ = {out['gamma_mean']:.3f}")
    ax.fill_between([-0.5, len(ds_names)-0.5],
                    out['gamma_mean'] - out['gamma_std'],
                    out['gamma_mean'] + out['gamma_std'],
                    alpha=0.15, color='white')
    ax.set_ylabel('implied γ = α·β̂/2')
    ax.set_title(f"γ constancy (CV={out['gamma_cv']:.1%})")
    ax.legend(fontsize=8, frameon=False)

    # Panel C: prediction accuracy
    ax = axes[2]
    preds = out.get('loo_predictions', {})
    if preds:
        ds_names_p = list(preds.keys())
        true_vals = [preds[d]['alpha_true'] for d in ds_names_p]
        pred_vals = [preds[d]['alpha_pred'] for d in ds_names_p]
        ax.plot([1.0, 1.6], [1.0, 1.6], 'k--', lw=0.8, alpha=0.4, label='perfect')
        for i, d in enumerate(ds_names_p):
            rel = preds[d]['rel_error']
            ax.plot(true_vals[i], pred_vals[i], 'o', ms=8,
                    label=f"{d} ({rel:+.1%})")
            ax.annotate(d, (true_vals[i], pred_vals[i]),
                       textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel('α measured')
    ax.set_ylabel('α predicted (LOO)')
    ax.set_title('Leave-one-out prediction')
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_aspect('equal')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'predictive_alpha.{ext}', dpi=180)
    plt.close(fig)
    print(f"[fig] {OUT / 'predictive_alpha.pdf'}")


if __name__ == '__main__':
    main()
