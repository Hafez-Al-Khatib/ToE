"""
Toy-model derivation of the inference-depth scaling law  K*(sigma) ~ C sigma^alpha
==================================================================================

Mechanism under test: optimal-stopping of gradient descent on a (locally) quadratic
learned energy is *spectral shrinkage*, i.e. early-stopped GD = iterative spectral
regularization (Yao-Rosasco 2007; Raskutti-Wainwright-Yu 2014). We ask whether this
mechanism reproduces:
  (1) a power law in sigma,
  (2) with the CORRECT SIGN (more noise -> MORE steps),
  (3) an exponent alpha in the empirically observed band [1.1, 1.4],
  (4) that MOVES with the curvature spectrum {lambda_i} (=> architecture dependence).

Model
-----
Clean signal x ~ N(0, diag(s2)), white noise:  x~ = x + sigma * eps, eps~N(0,I).
Learned energy  E(u) = 1/2 u^T A u,  A = diag(lambda).  Inference is constant-step GD:
    u_{t+1} = u_t - eta * A u_t  =>  (in eigenbasis)  u_K[i] = (1-eta*lambda_i)^K * x~[i].
Let r_i = 1 - eta*lambda_i in (0,1), and shrinkage a_i(K) = r_i^K in (0,1].

Per-mode expected reconstruction error (over noise and signal):
    E|u_K[i]-x[i]|^2 = (1-a_i)^2 s2_i + a_i^2 sigma^2
Global objective J(K) = sum_i [...]; K* = argmin_K J(K)  (continuous relaxation).
"""

import numpy as np


# ----------------------------------------------------------------------------- #
# 1. SINGLE MODE: closed form + brute-force check                               #
# ----------------------------------------------------------------------------- #
def kstar_single_closed(sigma, s2, lam, eta):
    """Closed-form optimal (continuous) stopping step for one mode.

    Optimal shrinkage is the Wiener/James-Stein factor a* = s2/(s2+sigma^2).
    Since a*(K) = r^K with r = 1-eta*lam < 1:  K* = ln(a*) / ln(r).
    """
    a_star = s2 / (s2 + sigma**2)
    r = 1.0 - eta * lam
    return np.log(a_star) / np.log(r)


def kstar_single_brute(sigma, s2, lam, eta, Kmax=5000):
    """Brute-force optimum on a fine continuous K grid (sanity check)."""
    r = 1.0 - eta * lam
    K = np.linspace(1e-4, Kmax, 400000)
    a = r**K
    err = (1 - a) ** 2 * s2 + a**2 * sigma**2
    return K[np.argmin(err)]


# ----------------------------------------------------------------------------- #
# 2. MULTI MODE: K*(sigma) for a spectrum                                        #
# ----------------------------------------------------------------------------- #
def kstar_multimode(sigma, s2, lam, eta, Kmax=200000):
    """Continuous K* minimizing the total error over all modes.

    Two-stage search: coarse geometric grid, then local parabolic refinement.
    """
    r = 1.0 - eta * lam  # (N,) all in (0,1)
    logr = np.log(r)

    def J(K):
        a = np.exp(K * logr)  # r^K, stable
        return np.sum((1 - a) ** 2 * s2 + a**2 * sigma**2)

    # coarse search over a wide geometric grid
    Kgrid = np.unique(np.concatenate([[0.0], np.geomspace(0.05, Kmax, 600)]))
    vals = np.array([J(k) for k in Kgrid])
    j = int(np.argmin(vals))

    # local refine around the coarse minimum
    lo = Kgrid[max(j - 1, 0)]
    hi = Kgrid[min(j + 1, len(Kgrid) - 1)]
    if hi <= lo:
        return Kgrid[j]
    Kfine = np.linspace(lo, hi, 4000)
    vfine = np.array([J(k) for k in Kfine])
    return Kfine[int(np.argmin(vfine))]


def kstar_multimode_decay(sigma, s2, lam, eta0, delta, Kmax=4000):
    """K* under the paper's DECAYING schedule eta_t = eta0 * delta^t.

    Shrinkage is a_i(K) = prod_{t<K} (1 - eta0 delta^t lam_i), evaluated on an
    integer step grid (the practical setting). Returns the integer argmin.
    """
    t = np.arange(Kmax)
    etas = eta0 * delta**t                      # (Kmax,)
    factors = 1.0 - np.outer(etas, lam)         # (Kmax, N)
    factors = np.clip(factors, 1e-12, None)
    a = np.cumprod(factors, axis=0)             # a[K-1] = shrinkage after K steps
    a = np.vstack([np.ones((1, lam.size)), a])  # prepend K=0 (no shrink)
    err = np.sum((1 - a) ** 2 * s2 + a**2 * sigma**2, axis=1)
    return float(np.argmin(err))


def fit_powerlaw(sigmas, kstars):
    """Fit log K* = log C + alpha log sigma  by OLS.  Returns alpha, C, R^2."""
    mask = kstars > 0
    x = np.log(sigmas[mask])
    y = np.log(kstars[mask])
    A = np.vstack([np.ones_like(x), x]).T
    (logC, alpha), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([logC, alpha])
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return alpha, np.exp(logC), r2


# ----------------------------------------------------------------------------- #
# 3. SPECTRA: natural-image signal power + tunable curvature spectrum            #
# ----------------------------------------------------------------------------- #
def make_spectra(N=256, beta=2.0, gamma=1.0):
    """Signal power s2_i ~ i^-beta (natural images); curvature lambda_i ~ i^+gamma.

    A good denoiser puts LARGE curvature on low-signal (high-index) modes so GD
    shrinks the noise there quickly, and SMALL curvature on high-signal (low-index)
    modes to preserve them -- the Wiener/precision coupling lambda_i ~ 1/s2_i.
    The exact Wiener filter is gamma = beta; real architectures under-fit the
    precision (gamma < beta), and gamma is the per-architecture knob.

    Prediction (crossover argument): the signal=noise mode is i* ~ sigma^(-2/beta),
    and the steps to shrink it scale as  K* ~ sigma^(2*gamma/beta), i.e. alpha=2*gamma/beta.
    """
    i = np.arange(1, N + 1)
    s2 = i.astype(float) ** (-beta)
    s2 /= s2[0]                      # top (lowest-freq) mode normalized to 1
    lam = i.astype(float) ** (gamma)  # curvature GROWS with mode index
    lam /= lam[0]                    # lowest-freq curvature = 1
    return s2, lam


def main():
    rng = np.random.default_rng(0)
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print("STEP 1  Single-mode: closed form vs brute force, and the sign/limits")
    print("=" * 78)
    eta = 0.4
    for sigma in [0.05, 0.1, 0.2, 0.4]:
        s2, lam = 1.0, 0.5
        kc = kstar_single_closed(sigma, s2, lam, eta)
        kb = kstar_single_brute(sigma, s2, lam, eta)
        print(f"  sigma={sigma:.2f}  K*_closed={kc:8.3f}  K*_brute={kb:8.3f}")
    print("  -> K* INCREASES with sigma (correct sign).")
    print("  Small-noise limit: K* ~ (sigma^2/s2)/(eta*lam)  => local slope alpha->2")
    print("  Large-noise limit: K* ~ 2 ln(sigma/s)/|ln r|    => logarithmic")
    print("  A single mode therefore CANNOT give a stable non-integer alpha.")

    print()
    print("=" * 78)
    print("STEP 2  Multi-mode spectral model: does a clean power law emerge?")
    print("=" * 78)
    sigmas = np.array([0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40])
    s2, lam = make_spectra(N=256, beta=2.0, gamma=1.0)
    eta = 0.9 / lam.max()
    kstars = np.array([kstar_multimode(s, s2, lam, eta) for s in sigmas])
    alpha, C, r2 = fit_powerlaw(sigmas, kstars)
    for s, k in zip(sigmas, kstars):
        print(f"  sigma={s:.2f}   K*={k:8.3f}")
    print(f"  FIT: K* ~ {C:.2f} * sigma^{alpha:.3f}   (R^2={r2:.4f})")
    print(f"  monotone increasing: {np.all(np.diff(kstars) > 0)}")

    print()
    print("=" * 78)
    print("STEP 3  Architecture dependence: alpha vs curvature-spectrum steepness")
    print("=" * 78)
    beta0 = 2.0
    print(f"  natural-image signal exponent beta = {beta0};  prediction alpha = 2*gamma/beta")
    print(f"  {'gamma (curv grow)':>18} | {'alpha_fit':>9} | {'alpha_pred':>10} | {'R^2':>7}")
    print("  " + "-" * 54)
    results = []
    for gamma in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]:
        s2, lam = make_spectra(N=256, beta=beta0, gamma=gamma)
        eta = 0.9 / lam.max()
        ks = np.array([kstar_multimode(s, s2, lam, eta) for s in sigmas])
        a, c, r2 = fit_powerlaw(sigmas, ks)
        results.append((gamma, a, c, r2))
        print(f"  {gamma:>18.2f} | {a:>9.3f} | {2*gamma/beta0:>10.3f} | {r2:>7.4f}")
    print()
    print("  Sweep the signal-power exponent beta (data statistics), gamma=1.3 fixed:")
    print(f"  {'beta (signal decay)':>18} | {'alpha_fit':>9} | {'alpha_pred':>10} | {'R^2':>7}")
    print("  " + "-" * 54)
    for beta in [1.5, 2.0, 2.5, 3.0]:
        s2, lam = make_spectra(N=256, beta=beta, gamma=1.3)
        eta = 0.9 / lam.max()
        ks = np.array([kstar_multimode(s, s2, lam, eta) for s in sigmas])
        a, c, r2 = fit_powerlaw(sigmas, ks)
        print(f"  {beta:>18.2f} | {a:>9.3f} | {2*1.3/beta:>10.3f} | {r2:>7.4f}")

    print()
    print("=" * 78)
    print("STEP 4  Does the exponent survive the paper's DECAYING step schedule?")
    print("=" * 78)
    s2, lam = make_spectra(N=256, beta=2.0, gamma=1.3)
    eta_max = 1.0 / lam.max()  # stability ceiling for the first (largest) step
    print(f"  {'schedule':>30} | {'alpha':>8} | {'R^2':>7}")
    print("  " + "-" * 52)
    # constant-step reference
    ks = np.array([kstar_multimode(s, s2, lam, 0.9 * eta_max) for s in sigmas])
    a, c, r2 = fit_powerlaw(sigmas, ks)
    print(f"  {'constant eta (reference)':>30} | {a:>8.3f} | {r2:>7.4f}")
    for name, (e0, d, kmax) in {
        "decaying delta=0.97 (ample)": (0.9 * eta_max, 0.97, 3000),
        "decaying delta=0.99 (ample)": (0.9 * eta_max, 0.99, 8000),
    }.items():
        ks = np.array([kstar_multimode_decay(s, s2, lam, e0, d, Kmax=kmax) for s in sigmas])
        a, c, r2 = fit_powerlaw(sigmas, ks)
        print(f"  {name:>30} | {a:>8.3f} | {r2:>7.4f}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    in_band = [g for g, a, c, r2 in results if 1.1 <= a <= 1.4]
    print(f"  curvature exponents giving alpha in [1.1,1.4]: {in_band}")
    print("  If non-empty AND alpha moves monotonically with the spectrum,")
    print("  the shrinkage mechanism reproduces the law with the right sign,")
    print("  band, and architecture dependence.")


if __name__ == "__main__":
    main()
