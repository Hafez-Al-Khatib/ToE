# Rigorous Experimental Design — KAN-EBM Overhaul

This document specifies exactly what to run, in what order, with what hyperparameters,
what metrics, and what stop/decision criteria. Every experiment has explicit
falsification criteria — if a result fails to meet them, we stop, debug, or
reframe rather than continuing.

---

## Experiment 1 — CIFAR-10 In-Distribution Training (foundation)

**This is the gating experiment. Everything downstream assumes it works.**

### Hypothesis

H1: KAN-EBM trained end-to-end on CIFAR-10 produces monotonic PSNR improvement
with $K$ on CIFAR-10 test data. $\Delta$PSNR(K=1 → K=20) ≥ 2 dB at σ=25/255.

H2: At matched parameter counts, MLP-EBM (same filter bank, MLP head) does NOT
produce monotonic K-scaling on CIFAR-10.

### Setup

**Dataset.**
- CIFAR-10 50K train / 10K test, 32×32 RGB.
- Normalize to [0, 1] (consistent with MNIST pipeline).
- Augmentation: **random horizontal flip only**. No random crop — breaks the
  patch-local energy assumption and invalidates the filter bank's translation
  equivariance argument.

**Noise.** Additive Gaussian at **DnCNN-standard scales**:
- σ₁ = 15/255 = 0.0588 (light)
- σ₂ = 25/255 = 0.0980 (medium) **← primary noise level; run this first**
- σ₃ = 50/255 = 0.1961 (heavy)

Using these exact values makes results directly comparable to published DnCNN /
BM3D numbers without arithmetic gymnastics.

**Architecture (KAN-EBM-CIFAR).**
- Filter bank: `F=16, k=5, C=3`. Output per spatial location: 48-d vector.
- Precision Π: 48 learnable scalars, softplus-parametrized.
- Grid squash: `tanh` (unchanged).
- KAN head: `[48, 32, 1]` with `G=5, p=3`. 8 basis functions per edge.
- Total params: ≈42K.

**Baselines (all trained identically on CIFAR-10).**
| Baseline | Description | Params |
|---|---|---|
| FFN-DSM-small | Matched single-pass feedforward denoiser | ≈50K |
| MLP-EBM | Same filter bank + MLP head [48, 64, 32, 1] | ≈45K |
| DnCNN-small | Standard DnCNN, depth reduced to match params | ≈50K |

DnCNN-small is non-negotiable. It is the modern denoising baseline and its
exclusion was the reviewer's sharpest point.

**Training.**
- 60 epochs. Budget ~6 hours per run on a 24GB consumer GPU.
- AdamW, `lr=3e-4`, `wd=1e-4`, cosine annealing.
- Gradient clip 1.0 (same as MNIST).
- AMP (`torch.amp.autocast`).
- Batch size 128.

**Inference.**
- η₀ = 0.05, δ = 0.97 (same as MNIST).
- K ∈ {0, 1, 2, 5, 10, 20, 50, 100}. **Include K=50 and K=100** to detect late-stage
  divergence or saturation.
- Gradient clamp to [−1, 1].

### Metrics (all logged to `results/logs/{timestamp}_cifar_{σ}.json`)

- **PSNR** (primary), averaged over 10K test images.
- **SSIM**.
- **LPIPS** (critical for natural images — PSNR alone doesn't capture perceptual
  quality). Use AlexNet backbone for speed.
- **Energy trajectory**: log $E_\theta(u_t)$ at every K for 100 held-out images.
  Used to verify gradient descent is actually minimizing the learned energy.
- **Per-sample variance** across test set at each K. Report standard deviation
  alongside mean. A "monotonic mean" with huge per-sample variance is a weaker
  claim than "monotonic for every sample."
- **Wall-clock** per K step (for the compute-efficiency discussion).

### Falsification criteria (stop and debug, don't train more)

1. **K-scaling slope**: if $\Delta$PSNR($K{=}1 \to K{=}20$) < 1.0 dB at σ=25/255,
   the central claim does not transfer to natural images. STOP after σ=25/255.
2. **MLP-EBM control**: if MLP-EBM's K-scaling on CIFAR-10 is comparable to
   KAN-EBM's (within 0.5 dB), the architectural claim fails. STOP.
3. **FFN ceiling**: if FFN-DSM-small's static PSNR ≥ KAN-EBM's K=20 PSNR, the
   compute-for-quality trade-off has no win. STOP.
4. **DnCNN-small ceiling**: if DnCNN-small ≥ KAN-EBM K=20 by >1 dB, reframe the
   paper around compute-scaling curve shape, not absolute quality.

### Expected results (my best guess; used to calibrate surprise)

- KAN-EBM: ~28 dB at K=1, ~31 dB at K=20 (σ=25/255).
- DnCNN-small: ~31 dB (flat).
- FFN-DSM-small: ~29 dB (flat).
- MLP-EBM: ~27 dB at K=1, ~26 dB at K=20 (divergence but milder than MNIST).

If these land within ±1 dB I'll call it nominal. If KAN-EBM K=20 < 30 dB or
if MLP-EBM actually scales, we debug.

### Execution order

1. `σ=25/255` first (4 models in parallel if GPU has headroom, else serial).
2. If falsification criteria pass, run `σ=15/255` and `σ=50/255`.
3. Total wall-clock: ~18 hours for all three σ × 4 models. On a single GPU,
   this is 3 days of serial training.

---

## Experiment 2 — Flow-Matching Baseline (modern head-to-head)

**This is the risky one. Be mentally prepared for flow-matching to win at low K.**

### Hypothesis

H3: A small flow-matching denoiser at matched parameters (~45K) saturates
around some K* (expected ~5–10 steps), after which KAN-EBM's K-scaling curve
continues to climb. The crossover K* is the central empirical signal.

### Setup

**Model: FM-small.**
- Small U-Net: 3 downsample levels, channel widths [16, 32, 32], no attention.
- Time embedding: 16-dim sinusoidal, broadcast.
- Total params: ~45K (tune channel width to hit budget).

**Training.**
- Conditional flow matching. Given clean $x$ and noisy $\tilde x = x + \sigma\eps$:
  - Sample $t \sim \mathrm{Uniform}(0, 1)$.
  - $x_t = (1-t)\tilde x + t x$.
  - Target: $v^* = x - \tilde x$.
  - Loss: $\|v_\theta(x_t, t) - v^*\|^2$.
- 60 epochs, matched hyperparameters to Exp 1.

**Inference.**
- Euler integration. K steps with stepsize 1/K.
- $u_0 = \tilde x$; $u_{i+1} = u_i + (1/K) \cdot v_\theta(u_i, t_i)$ for $t_i = i/K$.
- Report PSNR/SSIM/LPIPS at K ∈ {1, 2, 5, 10, 20, 50, 100}.

### Falsification criteria

1. **FM dominance**: if FM-small beats KAN-EBM at EVERY K up to K=50, the
   "our curve overtakes theirs" claim fails. In that case, reframe: "KAN-EBM
   provides interpretable energy structure and comparable quality" (weaker
   claim, still publishable, but the headline changes).
2. **FM saturation not observed**: if FM-small continues to improve at K=50 and
   K=100 with no saturation, the K-scaling story becomes less distinctive. Need
   to reframe around different curve shapes rather than KAN uniquely scaling.

### Expected results

- FM-small: monotonic improvement to K≈10, saturates by K=20.
- KAN-EBM: monotonic improvement to K=50+.
- Crossover: between K=10 and K=20 for σ=25/255.

If instead FM saturates at K=5 at lower quality than KAN-EBM's K=20, that is the
cleanest possible outcome.

---

## Experiment 3 — Inpainting (same checkpoint, different task)

**Reuses Exp 1 CIFAR checkpoint. No retraining.**

### Hypothesis

H4: K-scaling is preserved under the inpainting projection. Specifically,
$\Delta$PSNR($K{=}1 \to K{=}20$) ≥ 3 dB for mask rates ≥ 50%.

### Setup

**Masks.**
- Random pixel masks: 25%, 50%, 75% of pixels hidden uniformly at random.
- Random block masks: 25%, 50%, 75% of pixels in contiguous 8×8 blocks.
- Center-hole mask: fixed 16×16 center block on 32×32.

**Inference.**
$$u_{t+1} = M \odot \hat y + (1-M) \odot \bigl(u_t - \eta_t \nabla_u E_\theta(u_t)\bigr)$$
- Initialize $(1-M)$ region to the mean of all CIFAR-10 training pixels (`u_0 ≈ 0.48`).
- Clamp to [0, 1] after each step.
- K ∈ {1, 5, 10, 20, 50}.

### Metrics

- PSNR/SSIM computed **only on the hidden region** $(1-M)$, not the whole image.
- Track per-mask-type results separately. Expect random-pixel (easier) to have
  better absolute PSNR than block (harder).

### Baselines

1. FM-small (Exp 2 checkpoint) under same projection.
2. Simple mean-fill (trivial baseline — every method should beat this).

### Falsification criteria

1. If $\Delta$PSNR < 1 dB at 50% masking, the K-scaling story doesn't extend to
   inpainting. Drop this experiment from the paper.
2. If mean-fill beats KAN-EBM at high K (regression to mean), the gradient
   descent has failed for inpainting and we need to investigate step size.

---

## Experiment 4 — Super-Resolution 2× / 4× (optional, high risk)

**Honest warning**: this may not work well. SR requires hallucinating
high-frequency detail, and KAN-EBM with K=20 has a tendency to smooth rather
than sharpen. Decide to include after Exp 3 results.

### Hypothesis

H5: KAN-EBM solves SR via $\arg\min_u E_\theta(u) + \lambda \|Hu - y\|^2$ with
PSNR improving monotonically in K at fixed λ.

### Setup

- Bicubic downsample $H$: 32→16 (2×) and 32→8 (4×).
- Initialize $u_0$ = bicubic upsample of $y$.
- $\lambda$ tuning: grid search $\lambda \in \{10, 50, 100, 500, 1000\}$ on 1K
  validation images. Pick $\lambda$ that gives best PSNR at K=20.
- K ∈ {1, 5, 10, 20}.

### Falsification criteria

1. If $\Delta$PSNR < 0.5 dB from K=1 to K=20 on 2×, the method isn't effective
   for SR. Drop this experiment.
2. If 2× works but 4× doesn't, include only 2× and note the limitation honestly.

### If it fails

Don't force it. The paper's story is stronger without a weak SR result.
Inpainting alone is sufficient to show "K-scaling extends beyond denoising."

---

## Experiment 5 — OOD Detection (uses energy as anomaly score)

**Honest warning about the "likelihood OOD paradox"**: Nalisnick et al. (ICLR
2019) showed that deep generative models often assign higher likelihood to OOD
data than in-distribution data. DSM-trained EBMs are not immune. Do not be
surprised if AUROC is low on some OOD corpora.

### Hypothesis

H6: $E_\theta(x)$ on CIFAR-10 trained KAN-EBM is lower on in-distribution
CIFAR-10 test than on OOD corpora. AUROC ≥ 0.85 on SVHN-vs-CIFAR-10.

### Setup

**In-distribution:** CIFAR-10 test (10K images).
**OOD corpora** (easy → hard):
1. SVHN (digits; most different from CIFAR; easiest OOD)
2. Textures / DTD (texture images)
3. LSUN-C (scenes)
4. CIFAR-100 (most similar; hardest OOD; HEAT paper AUROC ≈ 0.90 is SOTA)

**Score.** Just $E_\theta(x)$. No inference needed.

**Metrics.**
- AUROC (primary)
- AUPR
- FPR at 95% TPR

### Published comparisons

| Method | SVHN | CIFAR-100 | Textures |
|---|---|---|---|
| JEM | 0.96 | 0.67 | — |
| HEAT | 0.98 | 0.90 | 0.88 |

If KAN-EBM gets ≥0.90 on SVHN and ≥0.75 on CIFAR-100, we're competitive for a
single-task EBM with 42K params. If <0.80 on SVHN, something is structurally
wrong — likely the DSM-trained energy is a local denoising direction predictor
rather than a true density proxy. In that case, drop OOD from the paper.

### Falsification criteria

1. AUROC < 0.70 on SVHN: drop OOD table entirely. The energy isn't a density
   proxy at our param budget.
2. AUROC reverses (OOD has LOWER energy than in-distribution): this is the
   pathological case. Drop OOD and note the phenomenon in limitations.

### If OOD works well

This is the biggest potential upside of the overhaul. A 42K-param EBM that hits
competitive OOD numbers becomes a strong selling point — compute-efficient,
interpretable, AND distribution-aware. Worth 1 day to try.

---

## Execution sequence & decision tree

```
DAY 0 (today):  Phase 0 — commit existing CBSD68/Set12 logs;
                standardize logging schema;
                verify exp_natural_images.py regenerates Table 5 JSON.

DAY 1–2:        Exp 1 on σ=25/255 (KAN-EBM + MLP-EBM + FFN + DnCNN-small, 4 runs).
                → If falsification criteria pass, continue.
                → If any falsification triggers, STOP, regroup, reframe.

DAY 3:          Exp 1 on σ=15/255 and σ=50/255.

DAY 4:          Exp 2 (FM-small training + inference curves).
                Run Exp 3 (inpainting) in parallel — reuses Exp 1 checkpoint.

DAY 5:          Exp 4 (SR) if Exp 3 succeeded. Skip if Exp 3 marginal.
                Run Exp 5 (OOD) — 2 hours, parallel.

DAY 6:          Regenerate all figures from new results.
                Rebuild tables via scripts/build_paper_tables.py.

DAY 7:          Paper text revision. Submit to advisor for review.
```

**Critical decision points:**

1. **After Exp 1 σ=25/255**: does K-scaling transfer? If NO → stop, reframe
   paper around MNIST ablation-only with a clear limitations section.
2. **After Exp 2**: does FM-small saturate before KAN-EBM? If NO → reframe
   to emphasize interpretability and the energy-based characterization rather
   than compute-scaling dominance.
3. **After Exp 5**: does OOD work? If YES → OOD becomes a headline contribution.
   If NO → drop silently, don't mention.

---

## What this gets you, realistically

**If everything works (60% probability):**
- 4 experiments with consistent K-scaling story (denoise, inpaint, SR, OOD)
- Matched-parameter modern baseline (FM-small)
- Figures regenerated on CIFAR-10 (reviewer-defensible)
- Narrowed, defensible theory claim
- **Realistic venue:** TMLR, AISTATS, UAI, or NeurIPS workshop.
  NeurIPS main track: 15–25% chance.

**If Exp 2 reveals FM dominance (20% probability):**
- Reframe to "interpretable controllable inference with learned energy"
- Still publishable at workshop/TMLR. Main-conference chance drops to <10%.

**If Exp 1 reveals K-scaling doesn't transfer (15% probability):**
- Narrow scope: "KAN-EBMs enable K-scaling on structured-domain images (MNIST);
  extension to natural images requires architectural changes."
- Publishable as a honest negative result at workshop level.
- This is still a legitimate research outcome, not a failure.

**True failure mode (<5% probability):**
- Nothing works: no K-scaling, no OOD, no inpainting.
- In this case: pivot the thesis chapter to focus on the theoretical
  equivalence + MNIST as the positive result, and propose the natural-image
  extension as future work.

---

## What I recommend you do *this week*

1. Run `experiments/exp_natural_images.py` tonight; confirm the JSON regenerates.
2. Tomorrow: go to your advisor meeting with the revised `paper.tex`, the
   `MEETING_PREP.md`, and this `EXPERIMENT_DESIGN.md`. Ask specifically whether
   they endorse the CIFAR-10 overhaul plan. Advisor buy-in before 18 hours of
   GPU time is cheap insurance.
3. Day after meeting: kick off Exp 1 σ=25/255 overnight.
4. Decide on Exp 2–5 based on Exp 1 outcome.

---

## What will NOT fail your professor

- Showing up with a critiqued draft and a plan.
- Being honest about what hasn't been verified yet.
- Having a falsification-aware experimental plan with stop criteria.
- Delivering a correct paper with limitations clearly stated, even if the venue
  is TMLR and not NeurIPS.

## What might disappoint your professor

- Silently overclaiming (already fixed).
- Publishing numbers without reproducible logs (being fixed in Phase 0).
- Chasing NeurIPS when the paper is a workshop paper — padding results to look
  stronger than they are is always spotted by reviewers, and by advisors.

You are currently doing all the right things. Keep doing them.
