"""
rebuttal_make_smooth_mlp_table.py
=================================
Reads outputs/finalization/rebuttal/smooth_mlp_ablation.json after the
4-variant training is complete and emits:

  1. A LaTeX table (paper-ready) summarising K* fits per activation,
     comparable to the KAN reference row.
  2. A figure overlaying K*(sigma) for KAN vs each smooth-MLP variant.

Outputs:
  outputs/finalization/rebuttal/smooth_mlp_table.tex
  outputs/finalization/rebuttal/smooth_mlp_overlay.pdf
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'outputs' / 'finalization' / 'rebuttal'
ABLATION = OUT_DIR / 'smooth_mlp_ablation.json'
KSTAR_VAL = ROOT / 'outputs' / 'finalization' / 'kstar_validation' / 'kstar_validation.json'

if not ABLATION.exists():
    raise SystemExit(f"Run rebuttal_smooth_mlp.py first: {ABLATION}")

ablation = json.loads(ABLATION.read_text())
kstar = json.loads(KSTAR_VAL.read_text())
sigmas = ablation['config']['sigmas']

kan_ref = kstar['experiments']['cifar_kan_seeded']

# ─── LaTeX table ─────────────────────────────────────────────────────────────
lines = [
    r"\begin{table}[h]",
    r"\centering\small",
    r"\begin{tabular}{lr|rrr|c}",
    r"\toprule",
    r"Variant & Params & $\alpha$ & $C$ & $R^2$ & K* values per $\sigma\in\{0.05,0.1,0.15,0.2,0.3\}$ \\",
    r"\midrule",
]
# KAN reference row
ks_str = ", ".join(f"{int(round(kan_ref['mean'][str(s)]))}" for s in sigmas)
lines.append(
    rf"\textbf{{KAN-EBM (ref)}} & 32K & "
    rf"\textbf{{{kan_ref['fit']['alpha']:.2f}}} & "
    rf"\textbf{{{kan_ref['fit']['C']:.1f}}} & "
    rf"\textbf{{{kan_ref['fit']['r_squared']:.2f}}} & "
    rf"\{{{ks_str}\}} \\"
)
lines.append(r"\midrule")
# Each smooth-MLP variant
for variant, r in ablation['variants'].items():
    np_ = r['n_params']
    fit = r['fit']
    a, C, r2 = fit['alpha'], fit['C'], fit['r_squared']
    means = [r['kstar_means'][str(s)] for s in sigmas]
    ks_str = ", ".join(f"{int(round(m))}" for m in means)
    a_s = f"{a:.2f}" if not math.isnan(a) else "n/a"
    C_s = f"{C:.1f}" if not math.isnan(C) else "n/a"
    r2_s = f"{r2:.2f}" if not math.isnan(r2) else "n/a"
    lines.append(
        rf"ConvMLP-{variant.upper()} & {np_/1000:.1f}K & "
        rf"{a_s} & {C_s} & {r2_s} & "
        rf"\{{{ks_str}\}} \\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{\textbf{Matched-architecture activation ablation (CIFAR-10).} "
    r"All ConvMLP variants share the KAN-EBM conv backbone and are sized to "
    r"match KAN-EBM-small ($\sim$35K params); only the head activation differs. "
    r"$K^*(\sigma)$ values are means over 3 seeds, 24 evaluation images per $\sigma$. "
    r"Power-law fits use the protocol of \cref{sec:kstar}.}",
    r"\label{tab:smooth-mlp}",
    r"\end{table}",
]

tex = "\n".join(lines)
out_tex = OUT_DIR / 'smooth_mlp_table.tex'
out_tex.write_text(tex)
print(tex)
print(f"\nSaved {out_tex}")

# ─── Overlay plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sigs = np.array(sigmas)

# KAN reference
kan_means = np.array([kan_ref['mean'][str(s)] for s in sigmas])
ax.plot(sigs, kan_means, 'o-', color='C3', linewidth=2.5, markersize=9,
        label=f"KAN-EBM (ref): $\\alpha$={kan_ref['fit']['alpha']:.2f}, "
              f"$R^2$={kan_ref['fit']['r_squared']:.2f}")

variant_styles = {'gelu': ('s--', 'C0'), 'silu': ('^:', 'C2'),
                  'tanh': ('D-.', 'C4'), 'relu': ('v--', 'C5')}
for variant, r in ablation['variants'].items():
    means = np.array([r['kstar_means'][str(s)] for s in sigmas])
    style, color = variant_styles.get(variant, ('o-', 'gray'))
    fit = r['fit']
    a, r2 = fit['alpha'], fit['r_squared']
    label = f"ConvMLP-{variant.upper()}: "
    if math.isnan(a):
        label += "no fit (degenerate)"
    else:
        label += f"$\\alpha$={a:.2f}, $R^2$={r2:.2f}"
    ax.plot(sigs, np.maximum(means, 1.0), style, color=color, linewidth=1.8,
            markersize=7, alpha=0.9, label=label)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'noise level $\sigma$', fontsize=12)
ax.set_ylabel(r'observed $K^*$', fontsize=12)
ax.set_title('Matched-architecture activation ablation: $K^*(\\sigma)$ per head type',
             fontsize=12)
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=10)
fig.tight_layout()
out_pdf = OUT_DIR / 'smooth_mlp_overlay.pdf'
out_png = OUT_DIR / 'smooth_mlp_overlay.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")

# ─── One-line headline summary ───────────────────────────────────────────────
print("\nHeadline numbers:")
print(f"  KAN-EBM (ref):       alpha={kan_ref['fit']['alpha']:.3f}  "
      f"C={kan_ref['fit']['C']:.1f}  R^2={kan_ref['fit']['r_squared']:.3f}")
for v, r in ablation['variants'].items():
    fit = r['fit']
    print(f"  ConvMLP-{v.upper():<5} ({r['n_params']:>6,}p): "
          f"alpha={fit['alpha']:+.3f}  C={fit['C']:>6.1f}  R^2={fit['r_squared']:.3f}  "
          f"-> {r['verdict']}")
