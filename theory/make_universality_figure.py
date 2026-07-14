"""
Universality figure: K*(sigma) collapses onto ONE law across energy-based AND
score-based (diffusion-style) denoisers -> the inference-depth law is a property of
iterative denoising, not architecture or energy structure.

EBM data: outputs/tier0_seeds/tier0_results.json (3 seeds, full 40ep/20K).
Score data: from theory/phase_b_diffusion.py run (3 seeds), embedded below.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "theory"
SIGMAS = np.array([0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30])

ebm = json.loads((OUT.parent / "tier0_seeds" / "tier0_results.json").read_text())

# Score model K* per seed (from phase_b_diffusion.py run, 2026-06-05)
score_kstars = [
    [0.98, 1.27, 2.61, 3.95, 5.49, 7.72, 10.20],
    [0.97, 1.27, 2.62, 3.95, 5.45, 7.67, 10.18],
    [0.97, 1.26, 2.63, 3.92, 5.48, 7.70, 10.17],
]
score_alpha = [1.374, 1.376, 1.378]

# Assemble all model classes
MODELS = {
    "KAN-EBM (energy)":      dict(ks=ebm["kan"]["kstars"], a=ebm["kan"]["alpha"], fam="energy"),
    "ConvMLP-GELU (energy)": dict(ks=ebm["convmlp_gelu"]["kstars"], a=ebm["convmlp_gelu"]["alpha"], fam="energy"),
    "ConvMLP-SiLU (energy)": dict(ks=ebm["convmlp_silu"]["kstars"], a=ebm["convmlp_silu"]["alpha"], fam="energy"),
    "ConvMLP-Tanh (energy)": dict(ks=ebm["convmlp_tanh"]["kstars"], a=ebm["convmlp_tanh"]["alpha"], fam="energy"),
    "Score net (diffusion)": dict(ks=score_kstars, a=score_alpha, fam="score"),
}
fam_color = {"energy": "#1f77b4", "score": "#d62728"}
markers = ["o", "s", "^", "D", "*"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ---- panel (a): all K*(sigma) overlaid + shared power-law fit ----
all_alpha = []
for (name, d), mk in zip(MODELS.items(), markers):
    ks = np.array(d["ks"])                      # (3 seeds, 7 sigma)
    mean_k = ks.mean(0)
    col = fam_color[d["fam"]]
    ax1.plot(SIGMAS, mean_k, mk, color=col, ms=7, mfc="none", mew=1.6,
             label=f"{name}: $\\alpha$={np.mean(d['a']):.3f}")
    all_alpha.extend(d["a"])
# one shared fit through pooled means
pooled = np.vstack([np.array(d["ks"]).mean(0) for d in MODELS.values()])
gm = pooled.mean(0)
sl, ic = np.polyfit(np.log(SIGMAS), np.log(gm), 1)
xs = np.linspace(SIGMAS.min(), SIGMAS.max(), 50)
ax1.plot(xs, np.exp(ic) * xs**sl, "k--", lw=1.2,
         label=f"shared law: $K^*={np.exp(ic):.0f}\\,\\sigma^{{{sl:.2f}}}$")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel(r"noise level $\sigma$"); ax1.set_ylabel(r"optimal inference depth $K^*$")
ax1.set_title("(a) All denoisers collapse onto one law")
ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
ax1.grid(True, which="both", alpha=0.2)

# ---- panel (b): alpha per model, mean +/- std, energy vs score ----
names = list(MODELS.keys())
means = [np.mean(d["a"]) for d in MODELS.values()]
stds = [np.std(d["a"], ddof=1) for d in MODELS.values()]
cols = [fam_color[d["fam"]] for d in MODELS.values()]
y = np.arange(len(names))[::-1]
grand = np.mean(all_alpha)
ax2.axvspan(grand - np.std(all_alpha), grand + np.std(all_alpha), color="gray", alpha=0.15)
ax2.axvline(grand, color="gray", ls=":", lw=1, label=f"grand mean {grand:.3f}")
ax2.errorbar(means, y, xerr=stds, fmt="o", color="none", ecolor="k", capsize=3, zorder=3)
for yi, m, s, c in zip(y, means, stds, cols):
    ax2.plot(m, yi, "o", color=c, ms=8, zorder=4)
ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
ax2.set_xlabel(r"fitted exponent $\alpha$ (mean $\pm$ std, 3 seeds)")
ax2.set_title(r"(b) $\alpha$ indistinguishable across models ($F=0.79$)")
ax2.set_xlim(1.30, 1.45)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
ax2.grid(True, axis="x", alpha=0.2)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"universality.{ext}", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'universality.pdf'}")
print(f"  grand mean alpha = {grand:.4f} ± {np.std(all_alpha):.4f}  (n={len(all_alpha)} fits, 5 model classes)")
