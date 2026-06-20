# A mechanistic derivation of the inference-depth law K*(σ) ≈ Cσ^α

**Status:** first-pass theory, numerically confirmed (`theory/kstar_spectral_model.py`).
**One-line result:** the law is **early-stopped-GD spectral shrinkage**, and the exponent
is **α = 2γ/β**, where β is the data power-spectrum exponent and γ is the architecture's
learned-curvature spectrum exponent. This gives the sign, the band, the architecture
ordering, *and* explains why C (not α) is dimension-dependent.

---

## 1. Setup (local quadratic / linear-GD model)

Clean signal `x ~ N(0, diag(s²))` in an eigenbasis, white noise `x̃ = x + σε`,
`ε~N(0,I)`. Near the basin the learned energy is quadratic, `E(u) = ½ uᵀA u`,
`A = diag(λ)`. Constant-step GD inference in the eigenbasis decouples:

```
u_K[i] = (1 - ηλ_i)^K · x̃[i] = r_i^K · x̃[i],   r_i = 1-ηλ_i ∈ (0,1).
```

So **step count K acts as a per-mode shrinkage knob**: `a_i(K) = r_i^K ∈ (0,1]`.
This is exactly the statement that early-stopped gradient descent is an iterative
spectral-regularization (Landweber-type) filter — the rigorous literature home of
the law (Yao–Rosasco–Caponnetto 2007; Raskutti–Wainwright–Yu 2014).

Per-mode expected reconstruction error:
```
E|u_K[i]-x[i]|² = (1-a_i)² s_i²  +  a_i² σ²
                  └ bias (signal lost) ┘   └ variance (noise kept) ┘
```

## 2. Single mode → mechanism (and why the radial ansatz failed)

Minimizing over the shrinkage `a` gives the **Wiener/James–Stein optimum**
`a* = s²/(s²+σ²)`, hence
```
K*(σ) = ln(a*) / ln(r) = ln(1 + σ²/s²) / |ln(1-ηλ)|.
```
- **Sign is correct:** K* increases with σ (more noise ⇒ more shrinkage steps).
- Small-noise limit: `K* ≈ (σ²/s²)/(ηλ)` ⇒ local slope **α→2**.
- Large-noise limit: `K* ≈ 2ln(σ/s)/|ln r|` ⇒ **logarithmic**.

A *single* mode therefore cannot produce a stable non-integer α. This is precisely
why the paper's radial ansatz `E∝r^p ⇒ α=2−p` failed (measured p∈[1.56,1.88]
predicted α∈[0.12,0.44]): the mechanism is **not** basin traversal of one coordinate,
it is **mode-wise shrinkage across a spectrum**.

## 3. Multi-mode → the law (crossover derivation)

Total error `J(K) = Σ_i [(1-r_i^K)² s_i² + r_i^{2K} σ²]`. Take the natural-image
signal spectrum `s_i² = i^{-β}` and a learned curvature that grows with mode index,
`λ_i = i^{γ}` (a good denoiser puts large curvature on low-signal modes to shrink
noise fast, small curvature on signal modes to preserve them — the Wiener coupling
`λ ∝ 1/s²` is the special case γ=β).

**Crossover argument.** A mode is noise-dominated when `s_i² < σ²`, i.e. above the
crossover index `i* = σ^{-2/β}`. The optimal stopping is set by the marginal
(crossover) mode, which needs `K ≈ ln2 / (η λ_{i*})` steps to reach `a*≈½`. Since
`λ_{i*} = (i*)^γ = σ^{-2γ/β}`:

```
        ┌─────────────────────────────────┐
        │   K*(σ)  ≈  (ln2/η) · σ^{2γ/β}   │   ⇒   α = 2γ/β
        └─────────────────────────────────┘
```

**Numerical confirmation** (`kstar_spectral_model.py`, R²=1.0000 power-law fits):

| γ (curvature growth) | α predicted = 2γ/β | α fitted |
|---|---|---|
| 0.5 | 0.50 | 0.73 |
| 1.0 | 1.00 | 1.06 |
| 1.3 | 1.30 | 1.32 |
| 1.6 | 1.60 | 1.61 |
| 2.0 | 2.00 | 2.00 |

(β=2 throughout; agreement is ~2% for γ≥1, with a small finite-mode upward bias at
very small γ.) Sweeping β at fixed γ=1.3 likewise tracks 2γ/β.

## 4. What this explains about the empirical paper

1. **Sign + band.** Natural images β≈2; a *sub-Wiener* learned curvature γ≈1.3 gives
   **α≈1.3**, squarely in the observed [1.1,1.4] band, with the right sign.
2. **Architecture dependence is α = 2γ/β.** Steeper learned curvature (larger γ) ⇒
   larger α. This reproduces the measured ordering **U-Net (1.43) > KAN (1.37) >
   ConvMLP (1.13)** as an ordering of curvature-spectrum steepness, *derived* rather
   than asserted.
3. **C is dimension-dependent, α is not.** Stability forces `η ∝ λ_max^{-1} ∝ N^{-γ}`,
   so `C ∝ N^{γ}` (N ≈ image dimension) while α is dimension-free. This matches the
   paper's finding that C varies with image size (CIFAR 58.2 vs CelebA 56.6) but α is
   stable across datasets.
4. **The falsifiable boundary falls out.** The crossover `i*=σ^{-2/β}` requires the
   corruption to displace the signal by an amount that scales as a power of σ — true
   for i.i.d. additive Gaussian noise, false for deblurring (a deterministic operator
   that does not move the manifold like additive noise), explaining R²=0.54 there.

## 5. The falsifiable test for the paper (this is the contribution)

The model is not just a fit — it makes a **parameter-free prediction** that can be
checked on the actual trained networks:

> **Measure** γ from the slope of each architecture's Hessian eigenvalue spectrum near
> the denoised minimum (you already run Lanczos for λ_max — extend to the full
> spectrum). **Measure** β from the dataset power spectrum (trivial). **Predict**
> α = 2γ/β and compare to the α fitted from the K* sweep.

If `α_measured ≈ 2γ/β` across U-Net / KAN / ConvMLP, the paper has a *derived,
falsifiable, mechanistic* law — exactly the "no theory" gap (reviewer M7) closed, and
a contribution neither KAEM nor the diffusion inference-scaling line possesses.

## 6. Honest limitations / open pieces

- **Local quadratic.** Valid near the basin; large-σ nonlinear regime not captured.
- **Basis alignment assumption.** Assumes the learned curvature A approximately
  diagonalizes in the data's power-spectrum (Fourier-like) basis. This is itself a
  testable claim about trained models, not a given.
- **Decaying schedule.** The clean α=2γ/β is for *constant* step size. A decaying
  schedule (the paper's default) perturbs the exponent — the toy model predicts the
  *direction* (slower decay / larger budget ⇒ steeper α), consistent with the paper's
  schedule-sensitivity finding, but does not reproduce its magnitude. Constant-step is
  the right theoretical object; reconciling the schedule precisely is open.
- γ must be measured, not assumed; if measured γ does **not** predict α via 2γ/β, the
  mechanism is wrong and we should know that before building the paper on it.

## 6b. GO/NO-GO RESULT (2026-06-04): the measured form does NOT confirm

Ran `theory/measure_kappa.py` on the 7 trained CIFAR-10 checkpoints (κ = slope of
curvature `v_kᵀHv_k` vs data-covariance signal power `s2_k`; β_data = 2.39):

- **Clean-minimum anchor:** corr(2κ, α_meas)=+0.62, but 2κ≈0.70 undershoots α_meas≈1.3
  by ~0.5, and κ does NOT separate KAN (α 1.36) from ConvMLP (α 1.13) — both κ≈0.35.
  Only the U-Net (κ=0.50, top α) and the ReLU exception (κ=nan, zero 2nd-deriv) land.
- **Operating-point anchor:** corr flips to **−0.78** (anti-correlated).

**Verdict: the quantitative prediction α=2κ, measured as diagonal curvature in the data
eigenbasis, is not confirmed.** Identified confounds (do not rescue it, but bound the
claim): (i) the derivation assumes H diagonalizes in the data basis — the measured
diagonal may not represent the effective per-mode dynamics; (ii) the operating-point
anchor uses a fixed (σ,K,dt) schedule, so energy-scale differences make the anchor
non-comparable across architectures; (iii) a single global κ-slope (fit R²≈0.6–0.9)
may miss the *local* slope at the crossover modes that actually sets α.

**What still stands (does not depend on the measurement):**
- The toy model remains a valid *existence proof / illustrative mechanism*: spectral
  shrinkage DOES produce a power law with correct sign, the observed band, and
  architecture-dependence via the curvature spectrum (§3, α=2γ/β confirmed numerically).
- The framing "early-stopped GD = iterative spectral (Landweber) regularization" is
  correct and citable, and closes the "no engagement with prior theory" gap regardless.

**Implication for the paper:** do NOT claim a measured, derived exponent law. Downgrade
to: (a) regularization-theory framing + (b) toy model as illustrative mechanism. Still a
real upgrade over "no theory", but honest about the open quantitative gap.

## 6c. PHASE-DIAGRAM RESULT (2026-06-04): the predicted boundary HOLDS

`theory/corruption_phase_diagram.py` (fig `outputs/theory/corruption_phase_diagram.pdf`)
sweeps K* vs severity for 5 corruptions x 3 architectures on the existing checkpoints.
The crossover argument (sec 3) predicts the law holds iff the corruption displaces the
signal by a power of the severity (stochastic additive) and breaks for a deterministic
forward operator. Confirmed cleanly and across all three architectures:

| corruption | type | KAN R^2 | ConvMLP R^2 | U-Net R^2 | verdict |
|---|---|---|---|---|---|
| Gaussian | additive | 0.994 | 0.953 | 0.993 | LAW (alpha 1.4-1.5) |
| speckle  | mult. additive | 0.981 | 0.976 | 0.984 | LAW (alpha 1.0-1.4) |
| Poisson  | shot noise | 0.991 | 0.988 | 0.969 | LAW (alpha 1.3-1.6) |
| blur     | deterministic op | 0.000 | 0.000 | 0.000 | BREAKS (K*=0, no iteration helps) |
| JPEG     | quantization | 0.311 | 0.000 | 0.000 | BREAKS |

This is independent QUALITATIVE support for the spectral-shrinkage mechanism (which the
direct kappa measurement, sec 6b, could not confirm quantitatively), and a paper-grade
result in its own right: it converts the one-line "law breaks on deblurring" footnote
into a phase diagram of exactly where the law applies. Harness validated against the
paper's KAN fine-grid K* (1,2,7,14 at sigma=0.05-0.3, matching 1.0/2.4/6.5/11.5).

## 7. Next actions

1. Extend the Hessian code to fit the full eigenvalue-spectrum slope γ per architecture.
2. Compute β for CIFAR-10 / CelebA / Tiny-ImageNet power spectra.
3. Test α_measured ≈ 2γ/β. **Go/no-go for the theory section.**
4. If it holds: this becomes §"A spectral-shrinkage account of the law", reframing the
   paper around iterative regularization + the derived exponent.
