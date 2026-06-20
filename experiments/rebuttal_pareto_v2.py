"""
rebuttal_pareto_v2.py
=====================
Wave 1c: FLOP-normalized Pareto re-render.

Reviewer R1: "The paper reports parameter counts but buries the FLOP penalty
-- KAN-EBM costs ~40x more compute per inference than FFN-DSM. Every headline
result looks different the moment you plot FLOP budget on the x-axis. Claiming
Pareto optimality without a FLOP-normalized frontier is simply not defensible."

This script produces three panels making the trade-off honest:
  Panel A: PSNR vs PARAMETERS   (KAN-EBM wins -- the memory Pareto)
  Panel B: PSNR vs cumulative FLOPs   (FFN-DSM wins per FLOP -- honest)
  Panel C: PSNR vs K   (KAN-EBM's iterative-quality property; the K* law domain)

Outputs:
  outputs/finalization/rebuttal/pareto_honest.pdf
  outputs/finalization/rebuttal/pareto_honest.png
  outputs/finalization/rebuttal/pareto_honest.json
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data sources
CIFAR = json.loads((ROOT / 'outputs' / 'cifar10' / 'results_f32.json').read_text())
FLOP  = json.loads((ROOT / 'outputs' / 'flop_benchmark' / 'flop_results.json').read_text())
CELEBA_FAIR = json.loads((ROOT / 'outputs' / 'finalization' / 'celeba_fair' /
                          'results_fair.json').read_text())
CELEBA = json.loads((ROOT / 'outputs' / 'celeba' / 'results.json').read_text())

# Pull KAN/MLP/FFN per-step FLOPs for CIFAR-10
cifar_flop = FLOP['CIFAR-10 (32x32, 3ch)']
KAN_STEP   = cifar_flop['kan_step_flops']           # 204.87 M
FFN_FWD    = cifar_flop['ffn_fwd_flops']            # 6.82 M

# Estimate MLP-EBM per-step FLOPs (matches run_paper_finalization.py formula)
def mlp_step_flops_cifar(hidden=256, img=32, ch=3):
    D = img * img * ch
    fwd = D * hidden + hidden * hidden + hidden * 1
    return 3 * fwd  # ~3x for HVP backward
MLP_STEP_CIFAR = mlp_step_flops_cifar()

# CIFAR PSNR results
res = CIFAR['results']
n_par = CIFAR['n_params']
K_LIST = sorted(int(k) for k in res['KAN-EBM'].keys())

# Build curves
def kan_curve():
    flops = [k * KAN_STEP for k in K_LIST]
    psnrs = [res['KAN-EBM'][str(k)] for k in K_LIST]
    return flops, psnrs, K_LIST

def mlp_curve():
    flops = [k * MLP_STEP_CIFAR for k in K_LIST]
    psnrs = [res['MLP-EBM'][str(k)] for k in K_LIST]
    return flops, psnrs, K_LIST

def ffn_point():
    return [FFN_FWD], [res['FFN-DSM']['1']]

# ─── Compute headline numbers ────────────────────────────────────────────────
KAN_BEST_K   = max(res['KAN-EBM'].items(), key=lambda kv: kv[1])
MLP_BEST_K   = max(res['MLP-EBM'].items(), key=lambda kv: kv[1])
FFN_PSNR     = res['FFN-DSM']['1']
KAN_BEST_FLOPS = int(KAN_BEST_K[0]) * KAN_STEP
MLP_BEST_FLOPS = int(MLP_BEST_K[0]) * MLP_STEP_CIFAR

print(f"CIFAR-10 (sigma={CIFAR['sigma']}):")
print(f"  KAN-EBM   best  K={KAN_BEST_K[0]:>2}  PSNR={KAN_BEST_K[1]:.2f}  "
      f"params={n_par['KAN-EBM']:>7,}  FLOPs={KAN_BEST_FLOPS/1e6:.1f} M")
print(f"  MLP-EBM   best  K={MLP_BEST_K[0]:>2}  PSNR={MLP_BEST_K[1]:.2f}  "
      f"params={n_par['MLP-EBM']:>7,}  FLOPs={MLP_BEST_FLOPS/1e6:.1f} M")
print(f"  FFN-DSM            PSNR={FFN_PSNR:.2f}  "
      f"params={n_par['FFN-DSM']:>7,}  FLOPs={FFN_FWD/1e6:.2f} M")
print()

# Honest framing numbers
print("HONEST FRAMING:")
kan_per_param = KAN_BEST_K[1] / n_par['KAN-EBM']
ffn_per_param = FFN_PSNR / n_par['FFN-DSM']
print(f"  PSNR per parameter:   KAN={kan_per_param*1e6:.2f}  FFN={ffn_per_param*1e6:.2f}  "
      f"-> KAN wins by {kan_per_param/ffn_per_param:.1f}x")
print(f"  PSNR per FLOP (peak): KAN={KAN_BEST_K[1]/KAN_BEST_FLOPS*1e9:.2f}  "
      f"FFN={FFN_PSNR/FFN_FWD*1e9:.2f}  -> FFN wins by "
      f"{(FFN_PSNR/FFN_FWD)/(KAN_BEST_K[1]/KAN_BEST_FLOPS):.1f}x")
print(f"  PSNR delta:           KAN-FFN = {KAN_BEST_K[1]-FFN_PSNR:+.2f} dB at "
      f"{KAN_BEST_FLOPS/FFN_FWD:.0f}x more FLOPs")

# ─── Build 3-panel figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

# Panel A: PSNR vs Parameters
axA = axes[0]
methods = [('KAN-EBM', 'C3', 'o'),
           ('MLP-EBM', 'C0', 's'),
           ('FFN-DSM', 'C2', '*')]
for name, color, marker in methods:
    if name == 'FFN-DSM':
        psnr = res[name]['1']
    else:
        # Use peak PSNR for comparison
        psnr = max(res[name].values())
    axA.scatter([n_par[name]], [psnr], s=200 if marker == '*' else 100,
                c=color, marker=marker, label=f"{name} ({n_par[name]/1000:.0f}K)",
                edgecolors='black', linewidths=1.2, zorder=3)
axA.set_xscale('log')
axA.set_xlabel('Parameters', fontsize=11)
axA.set_ylabel('Peak PSNR (dB)', fontsize=11)
axA.set_title('A. Memory Pareto (peak PSNR vs params)\nKAN-EBM dominates',
              fontsize=11, weight='bold')
axA.grid(True, alpha=0.3, which='both')
axA.legend(loc='lower right', fontsize=9)

# Panel B: PSNR vs cumulative FLOPs
axB = axes[1]
kf, kp, kk = kan_curve()
mf, mp, mk = mlp_curve()
ff, fp = ffn_point()
axB.plot(kf, kp, 'o-', color='C3', linewidth=2.0, markersize=7,
         label=f"KAN-EBM ({n_par['KAN-EBM']/1000:.0f}K params)")
axB.plot(mf, mp, 's--', color='C0', linewidth=1.6, markersize=6, alpha=0.85,
         label=f"MLP-EBM ({n_par['MLP-EBM']/1000:.0f}K params)")
axB.scatter(ff, fp, s=240, c='C2', marker='*', edgecolors='black',
            linewidths=1.2, label=f"FFN-DSM ({n_par['FFN-DSM']/1e6:.1f}M params)",
            zorder=3)
# Annotate K* on KAN curve
kstar_idx = int(np.argmax(kp))
axB.annotate(f"K*={kk[kstar_idx]}",
             xy=(kf[kstar_idx], kp[kstar_idx]), xytext=(8, 8),
             textcoords='offset points', fontsize=10, color='C3', weight='bold')
# Vertical lines marking equal-FLOP budget
axB.axvline(FFN_FWD, color='C2', linestyle=':', alpha=0.5)
axB.set_xscale('log')
axB.set_xlabel('Cumulative inference FLOPs', fontsize=11)
axB.set_ylabel('PSNR (dB)', fontsize=11)
axB.set_title('B. Compute Pareto (PSNR vs FLOPs)\nFFN-DSM is best per FLOP',
              fontsize=11, weight='bold')
axB.grid(True, alpha=0.3, which='both')
axB.legend(loc='lower right', fontsize=9)

# Panel C: PSNR vs K (the K* law domain)
axC = axes[2]
axC.plot(K_LIST, [res['KAN-EBM'][str(k)] for k in K_LIST], 'o-', color='C3',
         linewidth=2.0, markersize=7, label='KAN-EBM (iterative law)')
axC.plot(K_LIST, [res['MLP-EBM'][str(k)] for k in K_LIST], 's--', color='C0',
         linewidth=1.6, markersize=6, alpha=0.85, label='MLP-EBM (no scaling)')
axC.axhline(res['FFN-DSM']['1'], color='C2', linestyle=':', alpha=0.7,
            label='FFN-DSM (single pass)')
# Mark K*
axC.axvline(int(KAN_BEST_K[0]), color='C3', linestyle=':', alpha=0.6)
axC.annotate(f"K*={KAN_BEST_K[0]}", xy=(int(KAN_BEST_K[0]), KAN_BEST_K[1]),
             xytext=(5, -15), textcoords='offset points', fontsize=10,
             color='C3', weight='bold')
axC.set_xscale('log')
axC.set_xlabel('Inference steps K', fontsize=11)
axC.set_ylabel('PSNR (dB)', fontsize=11)
axC.set_title('C. Iterative-quality property\n(the K* law domain)',
              fontsize=11, weight='bold')
axC.grid(True, alpha=0.3, which='both')
axC.legend(loc='upper right', fontsize=9)

fig.suptitle(rf"CIFAR-10 ($\sigma$={CIFAR['sigma']}): KAN-EBM is parameter-efficient, "
             "not FLOP-efficient. The contribution is the iterative-quality scaling.",
             y=1.02, fontsize=12, weight='bold')
fig.tight_layout()

out_pdf = OUT_DIR / 'pareto_honest.pdf'
out_png = OUT_DIR / 'pareto_honest.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out_pdf}")
print(f"Saved {out_png}")

# Also dump JSON record for citation in paper
record = {
    'cifar_sigma': CIFAR['sigma'],
    'method_summary': {
        m: {
            'n_params': n_par[m],
            'peak_PSNR': res[m]['1'] if m == 'FFN-DSM' else max(res[m].values()),
            'best_K': '1' if m == 'FFN-DSM' else
                       max(res[m].items(), key=lambda kv: kv[1])[0],
            'flops_at_best_K': (FFN_FWD if m == 'FFN-DSM' else
                                int(max(res[m].items(), key=lambda kv: kv[1])[0]) *
                                (KAN_STEP if m == 'KAN-EBM' else MLP_STEP_CIFAR)),
        } for m in ['KAN-EBM', 'MLP-EBM', 'FFN-DSM']
    },
    'flop_ratios': {
        'KAN_step_over_FFN_fwd': KAN_STEP / FFN_FWD,
        'MLP_step_over_FFN_fwd': MLP_STEP_CIFAR / FFN_FWD,
        'KAN_total_at_best_K_over_FFN_fwd': KAN_BEST_FLOPS / FFN_FWD,
    },
    'honest_framing': {
        'parameter_dominance': f"KAN-EBM peak PSNR per parameter is "
                                f"{kan_per_param/ffn_per_param:.1f}x FFN-DSM",
        'flop_loss': f"KAN-EBM at peak K uses {KAN_BEST_FLOPS/FFN_FWD:.0f}x more "
                     f"FLOPs than FFN-DSM for {KAN_BEST_K[1]-FFN_PSNR:+.2f} dB",
        'contribution_statement': "KAN-EBM occupies the high-quality, low-parameter, "
                                  "high-FLOP corner of the trade-off space. The "
                                  "scientific contribution is the K*(sigma) scaling law "
                                  "that makes this corner predictable."
    }
}
out_json = OUT_DIR / 'pareto_honest.json'
out_json.write_text(json.dumps(record, indent=2))
print(f"Saved {out_json}")
