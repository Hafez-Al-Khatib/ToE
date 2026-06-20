"""Figure for the corruption phase-diagram (reads outputs/theory/corruption_phase_diagram.json)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "theory"
data = json.loads((OUT / "corruption_phase_diagram.json").read_text())

corrs = ["gaussian", "speckle", "poisson", "blur", "jpeg"]
models = list(data.keys())
colors = {"gaussian": "#1b9e77", "speckle": "#d95f02", "poisson": "#7570b3",
          "blur": "#e7298a", "jpeg": "#66a61e"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---- panel (a): K* vs severity for KAN-EBM (log-log, K* floored for display) ----
m = "kan_f32" if "kan_f32" in data else models[0]
for c in corrs:
    d = data[m][c]
    sev = np.array(d["sev"], float)
    ks = np.array(d["kstars"], float)
    disp = np.clip(ks, 0.7, None)              # floor zeros for log display
    holds = d["r2"] >= 0.9
    ax1.plot(sev, disp, "o-" if holds else "x--", color=colors[c], ms=6,
             lw=2 if holds else 1.2, alpha=0.95 if holds else 0.6,
             label=fr"{c}: $\alpha$={d['alpha']:.2f}, $R^2$={d['r2']:.2f}"
                   + ("" if holds else "  (breaks)"))
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("corruption severity")
ax1.set_ylabel(r"optimal inference depth $K^*$")
ax1.set_title(f"(a) {m}: law holds for additive noise only")
ax1.legend(fontsize=7.5, frameon=False)
ax1.grid(True, which="both", alpha=0.2)

# ---- panel (b): R^2 heatmap (models x corruptions) ----
R = np.array([[data[mm][c]["r2"] for c in corrs] for mm in models])
im = ax2.imshow(R, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(corrs))); ax2.set_xticklabels(corrs, rotation=30, ha="right")
ax2.set_yticks(range(len(models))); ax2.set_yticklabels(models)
for i in range(len(models)):
    for j in range(len(corrs)):
        ax2.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                 color="black", fontsize=9)
ax2.set_title(r"(b) power-law $R^2$: where the law holds")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label=r"$R^2$")

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"corruption_phase_diagram.{ext}", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'corruption_phase_diagram.pdf'}")
