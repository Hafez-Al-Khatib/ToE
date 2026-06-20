# Post-Meeting Empirical Overhaul — Execution Plan

**Trigger:** NeurIPS reviewer feedback identifies three empirical weaknesses:
(1) MNIST-only training, (2) BM3D as the only natural-image baseline, and
(3) no matched-parameter modern-baseline comparison on natural images.

**Goal:** move from a ~4/10 / borderline-reject position to a defensible 7–8/10.
10/10 is unrealistic on this timeline regardless of effort; 7–8 is achievable.

---

## Phase 0 — Pre-flight (1 hour, no compute)

Before any training runs, tighten the scaffolding so results are versioned and
reproducible. The largest hidden risk to the project is the current pattern of
running experiments offline and not committing outputs — that is what made the
CBSD68/Set12 and DDPM/DEQ numbers unverifiable in the current draft.

1. Create `results/logs/` with a `schema.json` defining the exact JSON fields
   every experiment script must write (model, dataset, σ, K, PSNR, SSIM, params,
   wall-clock, commit SHA, config hash).
2. Wrap `experiments/exp_diffusion_comparison.py` and `experiments/exp_natural_images.py`
   so that each run emits a `results/logs/{timestamp}_{expname}.json`. Make this
   the **only** way tables in the paper get populated.
3. Add a `scripts/build_paper_tables.py` that reads `results/logs/*.json` and
   emits the LaTeX for every table. This kills the "I ran it offline" drift
   permanently.
4. Commit the existing CBSD68/Set12 data and the current DDPM/DEQ logs once you
   find them on your machine — no experiments should depend on uncommitted data.

## Phase 1 — In-distribution training on natural images (2–3 days)

The central reviewer critique. This is non-negotiable for acceptance.

**Dataset choice.** Use **CIFAR-10** (32×32, 3-channel) as the primary in-distribution
training set, and **CBSD68-train** as a secondary run. CIFAR-10 is the pragmatic
choice because:
- It is small enough to train on a single local GPU in hours, not days.
- It has established conventions for gaussian-denoising benchmarks.
- The 32×32 resolution is a minimal extension of the current 28×28 pipeline.

**Architecture changes required:**
- Filter bank: `in_channels = 3` (RGB), filter count `F = 16` unchanged.
- `kernel_size = 5` unchanged; may tune to 3 if parameter budget is tight.
- Grid squash still `tanh` post-precision-weighting.
- KAN head: input dim = `3 · F = 48` (vs. 16 for grayscale MNIST). Keep
  `[48, 32, 1]` with grid `G=5`, order `p=3`. Resulting params ≈ 42K.
- Inference: identical decaying-step gradient descent.

**Training config:**
- 60 epochs, AdamW, `lr=3e-4`, `wd=1e-4`, cosine schedule.
- Noise levels σ ∈ {0.05, 0.1, 0.2} (normalized to [0,1]); corresponds roughly to
  σ ∈ {12.8, 25.5, 51.0} at 8-bit scale.
- Gradient clip 1.0, AMP on GPU.
- Augmentations: random horizontal flip. **Do not** use random crop — it breaks
  the patch-local energy assumption.

**Expected compute:** ≈4–6 hours per noise level on a single consumer GPU (e.g.,
RTX 3090 / 4090). All three σ levels: ~18 hours wall-clock.

**Deliverables:**
- `cifar10_kan_ebm_σ{0.05,0.1,0.2}.pt` checkpoints.
- A fresh `tab:natural_indist` table with K=1/5/10/20 PSNR + SSIM.
- Re-rendered `fig5_psnr_vs_k_cifar.pdf` showing monotonic K-scaling on CIFAR-10.

**Risk:** The current KAN-EBM pipeline was tuned for MNIST. It is possible that
K-scaling does not hold as cleanly on CIFAR-10 — the energy landscape may
develop oscillations or plateaus at later K. **If this happens**, two mitigations:
(a) revisit step-size decay δ; (b) increase KAN grid size G to 8 or 10.
Budget 1 extra day for mitigation.

## Phase 2 — Modern matched-parameter baseline (1–2 days)

The reviewer explicitly asked for a comparison against a modern EBM/diffusion /
flow-matching baseline at matched parameters. Recommended: a **small flow-matching
denoiser** rather than full DDPM, because:
- Flow matching is conceptually closest to "learn a gradient field" — apples to
  apples with DSM.
- Implementations are simpler than full DDPM with noise scheduling.
- Single-noise-level training is legitimate (no diffusion chain).

**Design:**
- Small U-Net encoder-decoder, parameter budget capped to **45K** to match
  KAN-EBM+CIFAR config.
- Conditional flow matching: train `v_θ(x_t, t)` where `x_t = (1-t)x̃ + t·x` and
  target is `(x - x̃)`.
- At inference: Euler-integrate the learned vector field from `x̃` for K steps.
  This gives a K-scaling curve directly comparable to KAN-EBM.

**Claim to defend:** KAN-EBM's K-scaling curve overtakes the flow-matching curve
past some crossover K*. You do not need to beat it at K=1.

**Deliverable:** Updated `tab:comparison` with flow-matching replacing (or adding
to) DDPM-score. Keep DEQ for the "fixed-point iteration saturates" story.

## Phase 2.5 — Task-breadth additions (2 days) — NEW, critical

**Motivation.** Subsequent reviewer-calibration with the 2023–2025 EBM / test-time
scaling literature (CDRL, CLEL, HEAT, ViSCALE workshop, inference-time-scaling
diffusion papers) confirms that modern strong EBM papers evaluate on **multiple
tasks**, not denoising alone. Denoising is a near-solved problem at the
absolute-PSNR level since DnCNN (2017); using it as a sole probe reads as weak to
reviewers in 2026. The fix: keep denoising, but show that K-scaling is a property
of the *architecture* by applying the same energy to more inverse problems, and
that the learned energy is meaningful in a distribution-structure sense (OOD).

All three additions below reuse the existing CIFAR-10 KAN-EBM checkpoint from
Phase 1 — **no new training required.**

### 2.5a — Inpainting (1 day)

Given observed pixels on mask $M$ and target $\hat u$, iteratively refine via
$$
u_{t+1} = M \odot \hat u + (1-M) \odot \bigl(u_t - \eta_t \nabla_u E_\theta(u_t)\bigr).
$$
Projection onto the observed set replaces the identity mask in the denoising
loop; the rest of the pipeline is unchanged. Evaluate on CIFAR-10 test set with
random box masks (25%, 50%, 75% missing) and report PSNR/SSIM vs. $K$.

**Expected signal:** monotonic improvement in PSNR with $K$ at every mask ratio.
At high mask ratio (75%), K-scaling should be substantially larger than denoising
because the problem is harder.

**Baseline:** same flow-matching model from Phase 2, run under the same
projection rule.

**Deliverable:** `tab:inpainting` with PSNR across mask ratios × K ∈ {1,5,10,20}.

### 2.5b — Super-resolution 2× / 4× (0.5 day)

Minimize $E_\theta(u) + \lambda \|Hu - y\|^2$ where $H$ is a bicubic downsampler.
Gradient descent on both terms, identical machinery. Evaluate on CIFAR-10 (upscale
16×16 → 32×32 for 2×; 8×8 → 32×32 for 4×).

**Deliverable:** `tab:sr` with PSNR at ×2 and ×4 vs. $K$.

### 2.5c — Out-of-distribution detection (0.5 day)

Use $E_\theta(x)$ directly as an OOD score. No inference loop needed. Evaluate
in-distribution (CIFAR-10 test) vs. out-of-distribution (SVHN, CIFAR-100,
Textures, LSUN). Report AUROC. This tests whether the DSM-learned energy
captures data-manifold structure, which is the *implicit* claim behind all of
denoising, inpainting, and SR working.

**Expected signal:** AUROC $> 0.85$ on SVHN vs. CIFAR-10 would place the method
in the competitive zone for energy-based OOD detection; $> 0.92$ would be strong.

**Deliverable:** `tab:ood` with AUROC across OOD corpora + a brief comparison
paragraph with JEM / HEAT published numbers.

### What NOT to add here

- **Image generation via Langevin.** Would require thousands of sampling
  steps + replay buffer + careful tempering; the K=20 regime cannot produce
  competitive FID. Framing K-scaling as an inverse-problem / MAP property is
  the correct positioning.
- **Classification via JEM.** Off-scope. JEM retrofits EBMs onto classifiers;
  our story is about inference-compute scaling, not classifier training.

## Phase 3 — Regenerate all figures from committed data (0.5 day)

Every figure currently in `results/interpretability/` was generated from
offline runs. Re-run the interpretability scripts on the CIFAR-10 checkpoint so
the figures show natural-image filter banks, spline activations, and energy
landscapes. Reviewers will check that figures match the headline training domain.

- `fig2_filter_bank` → new: 3-channel RGB filters, expect Gabor-like patterns
  with color selectivity (analogous to V1 simple cells with color opponency).
- `fig3_spline_functions` → regenerated from CIFAR checkpoint.
- `fig4_energy_landscape` → PCA slice through an image with a clear attractor.
- `fig6_inference_trajectory` → CIFAR denoising trajectory (much more visually
  striking than MNIST).
- `fig7_precision_weights` → per-channel precision over the 48-d feature space.

## Phase 4 — Text revision (0.5 day, after all experiments land)

- Replace MNIST-centric language in abstract/intro with CIFAR-10 as primary.
- Demote MNIST to an ablation ("controlled minimal setting that isolates the
  KAN vs. MLP effect").
- Keep the current narrowed theory claim (Proposition 1a / 1b — already done).
- Add crossover-K* discussion to Section 5.
- Write a limitations subsection that survives peer review: honest about
  resolution (32×32), channel count (3), noise model (additive Gaussian).

## Total timeline

| Phase | Work days | Wall-clock | Blocker risk |
|---|---|---|---|
| 0 | 1 | 1 day | low |
| 1 | 2–3 | 2–3 days (GPU hours inside) | **medium** (K-scaling may need tuning) |
| 2 | 1–2 | 1–2 days | low |
| 2.5 | 2 | 2 days | low — all reuse Phase 1 checkpoint |
| 3 | 0.5 | 0.5 day | low |
| 4 | 0.5 | 0.5 day | none |
| **total** | **7–9** | **7–9 calendar days** | — |

## What to do first, in order

1. **Verify existing CBSD68/Set12/DDPM/DEQ logs are actually committed.** Initial
   audit missed them; second pass found them under `data/CBSD68/`, `data/Set12/`,
   and `outputs/natural_images/`. Confirm `experiments/exp_natural_images.py`
   regenerates `outputs/natural_images/natural_results.json` identically.
2. Phase 0 scaffolding: standardize logging schema so future runs don't repeat
   the "offline, uncommitted" pattern.
3. Phase 1 CIFAR-10 training runs (start a σ=0.1 run overnight; it's the
   middle case and the one most likely to reveal K-scaling behavior).
4. Phase 2 (flow-matching baseline) in parallel once Phase 1 CIFAR runs produce
   monotonic K-scaling on at least one σ.
5. Phase 2.5 (inpainting + SR + OOD) as soon as Phase 1 checkpoint exists — these
   reuse the checkpoint, so they can start in parallel with Phase 2.
6. Phase 3 (regenerate figures), Phase 4 (text revision).

## What to explicitly **not** do

- Do not try to chase BM3D in absolute PSNR. The story is K-scaling, not raw dB.
- Do not add ImageNet, high-resolution denoising, or video — those expand the
  paper scope past what one overhaul can realistically support.
- Do not retain the original strong-form "Triple Equivalence" claim. The
  narrowed version (`prop:algorithmic` + `prop:quadratic`) is both stronger
  mathematically and safer reputationally.
- Do not publish numbers without a committed `results/logs/*.json` trail.
