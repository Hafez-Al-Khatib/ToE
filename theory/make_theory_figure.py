"""
Illustrative figure for the spectral-shrinkage account of K*(sigma) ~ C sigma^alpha.
Panel (a): the toy model produces clean power laws whose exponent is set by the
curvature-spectrum steepness gamma. Panel (b): the fitted exponent tracks the
closed-form prediction alpha = 2*gamma/beta, and natural-image beta=2 with a
sub-Wiener gamma~1.3 lands in the empirically observed band [1.1, 1.4].

This is presented as an ILLUSTRATIVE mechanism (existence proof), not a fitted law
for the trained networks (see THEORY_NOTE.md sec 6b).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "theory"))
from kstar_spectral_model import make_spectra, kstar_multimode, fit_powerlaw  # noqa: E402

OUT = ROOT / "outputs" / "theory"
OUT.mkdir(parents=True, exist_ok=True)

BETA = 2.0
sigmas = np.geomspace(0.03, 0.5, 14)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ---- panel (a): K*(sigma) power laws for three curvature spectra ----
colors = {0.8: "#1b9e77", 1.3: "#d95f02", 1.8: "#7570b3"}
for gamma, col in colors.items():
    s2, lam = make_spectra(N=256, beta=BETA, gamma=gamma)
    eta = 0.9 / lam.max()
    ks = np.array([kstar_multimode(s, s2, lam, eta) for s in sigmas])
    alpha, C, r2 = fit_powerlaw(sigmas, ks)
    ax1.loglog(sigmas, ks, "o", color=col, ms=5)
    ax1.loglog(sigmas, C * sigmas**alpha, "-", color=col,
               label=fr"$\gamma={gamma}$:  $\alpha={alpha:.2f}$  ($R^2={r2:.3f}$)")
ax1.set_xlabel(r"noise level $\sigma$")
ax1.set_ylabel(r"optimal inference depth $K^*$")
ax1.set_title("(a) Spectral shrinkage gives a power law")
ax1.legend(fontsize=8, frameon=False)
ax1.grid(True, which="both", alpha=0.2)

# ---- panel (b): fitted alpha vs prediction 2*gamma/beta ----
gammas = np.linspace(0.5, 2.0, 13)
fitted = []
for gamma in gammas:
    s2, lam = make_spectra(N=256, beta=BETA, gamma=gamma)
    eta = 0.9 / lam.max()
    ks = np.array([kstar_multimode(s, s2, lam, eta) for s in sigmas])
    a, _, _ = fit_powerlaw(sigmas, ks)
    fitted.append(a)
pred = 2 * gammas / BETA
ax2.plot(pred, pred, "k--", lw=1, label=r"$\alpha = 2\gamma/\beta$ (prediction)")
ax2.plot(pred, fitted, "o-", color="#d95f02", ms=4, label="toy-model fit")
ax2.axhspan(1.1, 1.4, color="gray", alpha=0.15)
ax2.text(0.55, 1.25, "observed\nband", fontsize=8, color="gray")
ax2.set_xlabel(r"prediction $2\gamma/\beta$")
ax2.set_ylabel(r"fitted exponent $\alpha$")
ax2.set_title(r"(b) Exponent set by curvature spectrum")
ax2.legend(fontsize=8, frameon=False, loc="upper left")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"spectral_shrinkage.{ext}", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'spectral_shrinkage.pdf'}")
print(f"  panel (b) fitted alpha range: [{min(fitted):.3f}, {max(fitted):.3f}] over gamma in [0.5, 2.0]")
