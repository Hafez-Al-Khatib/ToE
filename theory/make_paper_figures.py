"""
Assemble the paper_v8 figure set into figures/ (PDF+PNG, AAAI single-column).

Sources (all archived, CPU-only):
  fig_ood_decay        <- outputs/theory/gate_b_kcheck.json
  fig_semiconvergence  <- outputs/cifar10/results.json          (sigma=0.2)
  fig_universality     <- outputs/theory/universality.{pdf,png}  (copied; regenerate
                          via theory/make_universality_figure.py first)
  fig_phase_diagram    <- outputs/theory/corruption_phase_diagram.{pdf,png} (copied)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GRAY = "#1f77b4", "#e07b39", "#6b6b6b"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved figures/{name}.pdf/.png")


# ---- 1. OOD detectability vs descent depth (Sec. 6.2) ----
d = json.loads((ROOT / "outputs/theory/gate_b_kcheck.json").read_text())
ks = sorted(d["rows"], key=int)
K = np.array([int(k) for k in ks])
svhn = np.array([d["rows"][k]["svhn"] for k in ks])
c100 = np.array([d["rows"][k]["cifar100"] for k in ks])

fig, ax = plt.subplots(figsize=(4.4, 3.1))
ax.plot(K, svhn, "-o", color=BLUE, lw=1.8, ms=3.5, markevery=3)
ax.plot(K, c100, "-s", color=ORANGE, lw=1.8, ms=3.5, markevery=3)
ax.axhline(0.5, color=GRAY, ls=":", lw=1)
ax.annotate("SVHN (far OOD)", (K[10], svhn[10]), xytext=(0, 8),
            textcoords="offset points", fontsize=8, color=BLUE)
ax.annotate("CIFAR-100 (near OOD)", (K[8], c100[8]), xytext=(0, 8),
            textcoords="offset points", fontsize=8, color="#a35520")
ax.annotate("chance", (K[-1], 0.5), xytext=(-30, -11),
            textcoords="offset points", fontsize=7.5, color=GRAY)
ax.set_xlabel("descent depth $K$", fontsize=9)
ax.set_ylabel("OOD AUROC (CIFAR-10 in-dist.)", fontsize=9)
ax.set_ylim(0.45, 1.0)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.2)
save(fig, "fig_ood_decay")
print(f"  svhn: K=0 {svhn[0]:.3f} -> K={K[-1]} {svhn[-1]:.3f}; "
      f"c100 range [{c100.min():.3f},{c100.max():.3f}]")

# ---- 2. Semiconvergence vs flat feedforward at sigma=0.2 (Sec. 5) ----
r = json.loads((ROOT / "outputs/cifar10/results.json").read_text())
assert r["sigma"] == 0.2
K_LIST = r["K_LIST"]
res, npar = r["results"], r["n_params"]

fig, ax = plt.subplots(figsize=(4.4, 3.1))
series = [
    ("KAN-EBM", BLUE, "-", "o"),
    ("FFN-DSM", ORANGE, "--", "^"),
    ("MLP-EBM", GRAY, ":", "s"),
]
for name, col, ls, mk in series:
    y = [res[name][str(k)] for k in K_LIST]
    lab = f"{name} ({npar[name]/1e3:.0f}K, iterative)" if name != "FFN-DSM" \
        else f"{name} ({npar[name]/1e6:.1f}M, feedforward)"
    ax.plot(K_LIST, y, ls, marker=mk, color=col, lw=1.8, ms=5, label=lab)
kan = [res["KAN-EBM"][str(k)] for k in K_LIST]
kbest = K_LIST[int(np.argmax(kan))]
ax.annotate(f"$K^*={kbest}$", (kbest, max(kan)), xytext=(6, 4),
            textcoords="offset points", fontsize=8, color=BLUE)
ax.set_ylim(top=max(kan) + 1.0)
ax.set_xscale("log")
ax.set_xlabel("inference steps $K$", fontsize=9)
ax.set_ylabel("PSNR (dB)", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.grid(True, which="both", alpha=0.2)
save(fig, "fig_semiconvergence")

# ---- Frontier figure (from remote sweep parts; skipped if absent) ----
import sys
sys.path.insert(0, str(ROOT / "experiments"))
try:
    from frontier_analysis import curve_from_psnr
    PARTS = ROOT / "outputs/inference_frontier/parts"
    SIGMAS = ["0.05", "0.10", "0.15", "0.20", "0.30"]
    MODELS = [  # (tag, label, color, marker) in ascending FLOPs/step
        ("group_kan_8k", "GroupKAN-8K", "#2ca02c", "v"),
        ("unet", "U-Net-1.1M", "#8c564b", "D"),
        ("kan_32k", "KAN-32K", "#1f77b4", "o"),
        ("group_kan_32k", "GroupKAN-32K", "#e07b39", "^"),
        ("conv_mlp_gelu", "ConvMLP-560K", "#7f7f7f", "s"),
        ("kan_110k", "KAN-110K", "#d62728", "*"),
    ]
    if PARTS.exists() and len(list(PARTS.glob("*.json"))) >= 60:
        fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), sharex=True)
        for ax, sig in zip(axes, SIGMAS):
            for tag, label, col, mk in MODELS:
                curves = []
                for seed in (0, 1):
                    d = json.loads((PARTS / f"{tag}_s{sig}_seed{seed}.json").read_text())
                    fl, ps = curve_from_psnr(d["psnr_per_step"], d["flops_step"], d["psnr_k0"])
                    curves.append(ps)
                ps = np.mean(curves, axis=0)
                ax.plot(fl[1:], ps[1:], "-", marker=mk, color=col, lw=1.3, ms=2.6,
                        label=label if sig == SIGMAS[0] else None)
            ax.set_xscale("log")
            ax.set_title(f"$\\sigma={sig}$", fontsize=9)
            ax.tick_params(labelsize=7.5)
            ax.grid(True, which="both", alpha=0.15)
        axes[0].set_ylabel("PSNR (dB)", fontsize=9)
        axes[2].set_xlabel("cumulative inference FLOPs / image", fontsize=9)
        axes[0].legend(fontsize=6.2, frameon=False, loc="lower left")
        plt.tight_layout()
        save(fig, "frontier")
except ImportError:
    print("frontier_analysis not importable; skipping frontier figure")

# ---- 3./4. Copy archived figures ----
for src, dst in [("outputs/theory/universality", "fig_universality"),
                 ("outputs/theory/corruption_phase_diagram", "fig_phase_diagram")]:
    for ext in ("pdf", "png"):
        shutil.copyfile(ROOT / f"{src}.{ext}", FIG / f"{dst}.{ext}")
    print(f"copied {src} -> figures/{dst}")
