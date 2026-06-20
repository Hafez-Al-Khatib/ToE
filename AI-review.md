# Claude's Review:

I'll read the NeurIPS formatting instructions first, then carefully review the paper.Now let me carefully review the paper and render a detailed reviewer report.**Verdict: Weak Reject (Score 4/10)**

Here is a summary of my assessment:

**What the paper does well.** The core architectural insight — that B-spline KANs produce smooth C^p energy landscapes where MLP energies undergo topological collapse — is well-motivated and genuinely novel in the EBM context. The empirical scaling law K*(σ) ≈ 86.7σ^1.53 is the paper's crown jewel: it quantifies a phenomenon (peak-and-degrade) that the community has observed but never formalized, and the early-stopping rule with 99.4% oracle recovery is immediately deployable. The decision to explicitly demonstrate where the law fails (Gaussian-blur deblurring, R²=0.54) is exactly the kind of scientific rigor NeurIPS should reward.

**Why I cannot accept it as-is.**

The biggest problem is a structurally unfair comparison. The paper reports parameter counts but buries the FLOP penalty — KAN-EBM costs ~40× more compute per inference than FFN-DSM. Every headline result ("+6.83 dB over MLP-EBM with 100× more parameters") looks different the moment you plot FLOP budget on the x-axis. Claiming Pareto optimality without a FLOP-normalized frontier is simply not defensible.

The MLP-EBM baseline uses piecewise-linear activations, which is the worst possible choice for an iterative-inference energy. A GELU or SiLU MLP would be a far more informative ablation — the paper's central mechanistic claim (B-spline smoothness is the cause) requires ruling out activation smoothness as an equally effective substitute.

Proposition 1 is mathematically trivial: ∇ log p_E = −∇E follows in one line from the Boltzmann definition. Naming it a Proposition and giving it a section overstates the theory.

Finally, the scaling law fits 5 noise levels with a 2-parameter model (R²=0.98 over 5 points is expected from a well-chosen functional form) and never reports confidence intervals on the exponent α. This needs to be strengthened before the law can be called a regularity.

If the authors address R1–R3 (FLOP comparisons, smooth-activation baseline, and law confidence intervals), this could become a strong paper.

# GPT-5.5 Review:

Here is a rigorous NeurIPS-style review. I am intentionally critical and focus on whether the claims would survive top-tier scrutiny.

---

# 🔍 Summary

This paper proposes **KAN-EBM**, an energy-based model where the energy is parameterized by a **Kolmogorov–Arnold Network (KAN)** with B-spline activations, combined with a learned convolutional filter bank. The key claims are:

1. **Strong parameter–compute tradeoff**: small KAN-EBMs outperform much larger MLP-EBMs and feedforward denoisers when allowed iterative inference.
2. **Empirical scaling law**: optimal inference steps obey
   ( K^*(\sigma) \sim C \sigma^\alpha ) with high (R^2).
3. **KAN-specific phenomenon**: this scaling behavior does not appear in MLP-EBMs.
4. A unification of several inference frameworks (Allen–Cahn, score matching, predictive coding).

---

# 🧠 Overall Assessment

**Recommendation: ❌ Reject (borderline)**

The paper is **interesting and potentially impactful**, but **not yet at NeurIPS acceptance level** due to major issues in:

* experimental rigor and fairness
* strength of evidence for core claims
* novelty clarity
* theoretical contribution depth

That said, this is **close to a strong workshop / future conference paper** if fixed.

---

# ⭐ Strengths

### 1. Interesting idea with clear intuition

Using **KANs as smooth energy parameterizations** is a compelling hypothesis:

* Smoothness → better gradient flow → better iterative inference
  This is plausible and aligns with known EBM issues.

### 2. Test-time compute framing is timely

The paper taps into a **very active direction** (test-time scaling), which is a strong plus.

### 3. Empirical scaling law is intriguing

The (K^*(\sigma)) law is:

* clearly presented
* falsifiable (nice touch)
* potentially important if validated broadly

### 4. Clear narrative

The paper is well-written, structured, and easy to follow.

---

# 🚨 Major Weaknesses (Critical)

## 1. ❗ Core claim (KAN vs MLP) is not convincingly isolated

The paper claims:

> “test-time compute scaling is a property of B-spline-parameterised energies”

But this is **not causally established**.

### Problems:

* The architecture differs in **multiple ways simultaneously**:

  * KAN vs MLP
  * spline activations vs ReLU
  * parameterization style
  * implicit regularization differences
* No controlled ablation like:

  * smooth MLPs (e.g., GELU, SIREN, Fourier features)
  * higher-order smooth activations
  * spectral regularization

👉 Without these, the conclusion:

> “this is due to B-splines”
> is **overstated**.

---

## 2. ❗ Baselines are not competitive enough

This is the **biggest issue**.

### Problems:

* MLP-EBM is a weak baseline for vision:

  * No CNN-based EBMs compared
* Missing strong modern baselines:

  * diffusion models (properly tuned)
  * score-based models with iterative refinement
  * plug-and-play priors
* FFN-DSM baseline is simplistic and static

### Consequence:

The claim:

> “100× fewer parameters outperform baselines”

is **not meaningful without strong baselines**.

---

## 3. ❗ Compute fairness is questionable

The paper emphasizes:

* parameter efficiency

But **ignores actual compute cost**:

> “~40× more FLOPs at K=5”

This is buried in limitations but is **central**.

### Issue:

* Comparing:

  * 32K params + 50 iterations
  * vs 3M params single pass

is **not a fair comparison** unless:

* FLOPs
* wall-clock time
* energy usage

are matched.

👉 NeurIPS reviewers will penalize this heavily.

---

## 4. ❗ Scaling law is overclaimed

The (K^*(\sigma)) law is interesting but:

### Problems:

* Very limited scope:

  * only i.i.d Gaussian noise
  * only small datasets
* Small number of points (5 σ values)
* Fit could be coincidental

### Missing:

* Larger datasets (ImageNet)
* different corruption types
* theoretical explanation

👉 The claim:

> “clean empirical scaling law”

is **too strong for current evidence**.

---

## 5. ❗ Theoretical contribution is weak

Proposition 1:

> Allen-Cahn = score matching = predictive coding

This is **trivial and well-known**:

* All reduce to gradient descent on an energy

This will be viewed as:

* **not novel**
* **not insightful**

---

## 6. ❗ Overclaiming throughout

Examples:

* “Pareto-optimal” → not demonstrated rigorously
* “mechanistic cause” → not proven
* “specific property of B-spline energies” → not isolated

This hurts credibility.

---

# ⚠️ Minor Weaknesses

* No uncertainty/error bars in many key plots
* Small evaluation sets (e.g., 24 images for K*)
* PSNR-only evaluation (no perceptual metrics)
* Limited qualitative comparisons
* No real-world tasks beyond denoising/inverse problems

---

# 🔧 Required Changes for Acceptance

## MUST FIX (Major)

### 1. Stronger baselines

* CNN-based EBMs
* diffusion models (properly tuned)
* modern denoisers
* plug-and-play / RED methods

---

### 2. Proper compute matching

Report:

* FLOPs
* runtime
* memory

And compare under:

* equal compute budgets

---

### 3. Better ablations

Isolate the cause:

* KAN vs:

  * smooth MLP (GELU, SiLU)
  * SIREN / Fourier nets
  * spline vs non-spline

---

### 4. Strengthen scaling law

* More σ points
* more datasets
* different corruptions
* statistical confidence intervals

---

### 5. Tone down claims

Replace:

* “property of B-spline energies”

with:

* “observed in our experiments”

---

## SHOULD FIX

* Add perceptual metrics (LPIPS, FID)
* Improve qualitative results
* Larger-scale experiments
* Clarify theoretical contribution (or reduce emphasis)

---

# 🧾 Final Verdict

### ❌ Reject

### Confidence: High

---

# 💬 Meta-comment

This paper has **real potential**. The core intuition:

> smooth energy landscapes enable useful iterative inference

is promising and likely **directionally correct**.

However, in its current form, it reads as:

> “strong empirical observation + weak causal evidence + weak baselines”

Fixing that could realistically move this to:

👉 **NeurIPS accept or strong ICML paper**

---

# New reviews after paper_v3 latest edit

### **Overall Assessment: Borderline Reject / Major Revisions Needed**

Thank you for the opportunity to review your draft. The paper tackles a compelling question in the current landscape of deep learning: how to effectively leverage test-time compute for vision tasks. By investigating Energy-Based Models (EBMs) and isolating the architectural properties that enable beneficial inference-time iteration, the paper makes a distinct scientific contribution. 

However, while the empirical scaling law is interesting, the reliance on older, smaller-scale baselines and datasets makes it difficult to assess the true utility of KAN-EBMs against the modern state-of-the-art. Addressing the gaps below would significantly strengthen the submission and elevate it to a clear Accept.

---

### **Strengths**
*   **Architectural Ablation:** A major strength of this work is the matched-architecture activation ablation. Identifying that the scaling law is fundamentally a product of the convolutional backbone and per-pixel local energy—rather than simply attributing the entire phenomenon to Kolmogorov-Arnold Networks (KANs)—is a rigorous and intellectually honest finding[cite: 1].
*   **Formalization of Test-Time Scaling:** The paper establishes a clear, mathematically formalized empirical law, `$K^*(\sigma) \approx 86.7\sigma^{1.53}$`, for optimal inference depth[cite: 1]. The use of robust statistical methods, including parametric bootstrapping and jackknife confidence intervals, provides solid backing for the exponent[cite: 1].
*   **Transparency:** Your limitations section is candid and precise. You clearly acknowledge that the proposed KAN-EBM is 53× less FLOP-efficient at peak quality compared to a standard FFN baseline[cite: 1]. Furthermore, you explicitly define the falsifiable boundaries of the scaling law, noting its collapse on non-i.i.d. corruptions like Gaussian deblurring[cite: 1].

### **Weaknesses & Required Changes**
*   **Missing SOTA Baselines (Critical):** Comparing KAN-EBM strictly against global MLP-EBMs and FFNs limits the impact of the paper[cite: 1]. Continuous-space test-time compute is currently dominated by Diffusion Models, which share theoretical score-matching underpinnings with EBMs. Recent literature has heavily explored test-time scaling for diffusion (e.g., test-time augmentation, multi-trajectory sampling). Without comparing against a small-scale diffusion baseline, it is impossible for readers to evaluate if the parameter efficiency of KAN-EBM justifies its inference cost compared to modern standards.
*   **Dataset Scale (Major):** Evaluating primarily on CIFAR-10 and CelebA-64 is insufficient for a NeurIPS submission in generative modeling. These are widely considered toy datasets. While ImageNet-64 is briefly mentioned in the appendix, the empirical scaling law would be vastly more convincing if demonstrated on high-resolution data (e.g., 256x256). It is at these higher resolutions where the "per-pixel local-energy" parameterization might truly differentiate itself from global approaches.
*   **The Parameter vs. Compute Trade-off (Moderate):** The paper defends the high FLOP cost (using 60× more FLOPs than an FFN forward pass for a +3.57 dB PSNR advantage) by citing SRAM footprint advantages for edge deployments[cite: 1]. However, edge devices are often severely bottlenecked by energy consumption, which scales closely with FLOPs. To make this defense viable, you should include a brief profiling of actual latency or energy draw on a target device, rather than relying strictly on parameter counts to claim efficiency.
*   **Contextualizing Test-Time Compute (Minor):** The paper briefly mentions LLM test-time compute[cite: 1]. It would benefit from a broader discussion linking the proposed $K^*$ scaling to recent literature attempting to scale System 2 reasoning and inference compute in continuous visual domains, clearly positioning KAN-EBM within that wider movement.

---

### **Visualization Suggestion**
To help visualize the core empirical contribution of your work—the $K^*$ scaling law—I've drafted an interactive explorer below. It models the parameter scaling against the noise level based on the equations and specific parameters provided in your text. This type of dynamic visualization could be highly effective if included on a project page or in supplementary HTML materials to help practitioners immediately grasp the test-time scaling dynamics.

<GenerateWidget height="600px" component_placeholder_id="im_23530610cb4353af">
```json
{
  "widgetSpec": {
    "height": "600px",
    "prompt": "Create an interactive dashboard using D3.js to visualize the empirical test-time scaling law K*(sigma) = C * sigma^alpha. Initialize with the paper's data: KAN-EBM (C=86.7, alpha=1.53), ConvMLP-EBM (C=39.2, alpha=1.26), and Global MLP-EBM (C=10, alpha=0.29). Create a line chart with 'Corruption Noise (sigma)' on the x-axis (ranging from 0.05 to 0.35) and 'Optimal Inference Steps (K*)' on the y-axis (log scale). Visually distinguish the three lines. Include an interactive slider for 'sigma' that dynamically updates and displays the exact K* value required for each of the three architectures at the selected noise level. Add tooltips on the chart points and a clear legend."
  }
}
```
</GenerateWidget>


Here is a rigorous NeurIPS-style review of your paper.

---

# **Summary**

This paper proposes **KAN-EBM**, an energy-based model combining a convolutional backbone with a Kolmogorov–Arnold Network (KAN) head, and studies **test-time iterative inference scaling**. The central claim is an **empirical scaling law**:

[
K^*(\sigma) \approx C \sigma^\alpha
]

governing the optimal number of inference steps for denoising tasks. The paper argues:

* Iterative refinement depends on **architecture (conv + per-pixel energy)**, not head type.
* KAN yields **better scaling exponent and PSNR** than MLP heads.
* Strong **parameter efficiency** but poor **FLOP efficiency**.
* The scaling law is **robust across datasets and settings**, with a **clear failure boundary**.

---

# **Overall Recommendation: ❌ Reject (Borderline / Weak Reject)**

This is a **promising and interesting paper**, but it does **not yet meet NeurIPS acceptance standards** due to concerns about **novelty, rigor, and experimental depth**.

---

# **Strengths**

### 1. Clear empirical phenomenon

The identification of a **scaling law for inference steps** is interesting and potentially impactful:

* Well quantified (bootstrap + jackknife CIs)
* Tested across datasets (CIFAR-10, CelebA)
* Includes **negative results (deblurring failure)** → strong scientific practice

### 2. Strong ablation design

The **matched-architecture ablation (Section 3.3)** is a highlight:

* Clean isolation of variables
* Convincingly shows **architecture > activation/head**
* This significantly strengthens credibility

### 3. Honest reporting of trade-offs

The paper is commendably transparent:

* Explicit FLOP inefficiency
* Domain-of-validity discussion
* Non-claims clearly stated

This aligns well with good NeurIPS standards.

### 4. Potential importance

If validated further, the idea of:

> “test-time compute scaling laws for vision models”

could be **important and timely**, analogous to LLM inference scaling.

---

# **Weaknesses (Major Issues)**

## 1. Lack of theoretical grounding (critical)

The central contribution is **purely empirical**.

Problems:

* No derivation or mechanistic explanation of:

  * Why (K^* \propto \sigma^\alpha)?
  * Why (\alpha \approx 1.5)?
* The spectral analysis attempt (Fig. 3a) is inconclusive.

At NeurIPS level, for a **claimed “scaling law”**, reviewers expect:

* Either theory OR
* Much broader empirical universality

Currently, it feels like a **curve fit with limited explanation**.

---

## 2. Limited experimental scope

Despite strong internal validation, **external validation is weak**:

### Missing:

* Modern baselines:

  * No comparison with **diffusion models**
  * No comparison with **transformer-based denoisers** (e.g., Restormer, SwinIR)
* No evaluation on:

  * Real-world noise
  * Higher-resolution datasets
  * Non-i.i.d corruptions (except one failure case)

This makes it unclear whether:

> the scaling law is fundamental OR dataset-specific.

---

## 3. Novelty concerns

Key issue:

> The main finding (iterative inference improves EBMs) is not new.

The novelty is:

* The **specific scaling law**
* The **KAN improvement**

But:

* The law is **empirical and narrow**
* The KAN improvement is **incremental** (+0.77 dB)

Also, the paper itself shows:

> KAN is NOT necessary for the phenomenon

This weakens the central claim.

---

## 4. Weak evaluation metric (PSNR only)

The paper relies almost entirely on **PSNR**.

Missing:

* Perceptual metrics (LPIPS, SSIM)
* Visual comparisons
* Human perceptual quality

This is especially problematic in image restoration tasks.

---

## 5. Small-scale regime

All experiments are in **tiny-model regime**:

* 32K–110K parameters vs millions in real systems

This raises concern:

> Does the scaling law persist at realistic scales?

---

## 6. Statistical concerns

Although the authors attempt rigor:

* Only **5–9 σ points**
* Power-law fitting on small samples is fragile
* High R² is not strong evidence for scaling laws

This is a common pitfall in empirical scaling claims.

---

# **Minor Issues**

* Some claims are **overstated**:

  * “reproducible across datasets” → only 2 datasets
* Figures are dense and somewhat hard to interpret
* Writing is generally strong but occasionally **defensive/redundant**

---

# **What Would Make This Paper Acceptable**

To reach acceptance, the paper needs **at least two** of the following:

### 1. Stronger empirical validation

* Add:

  * Diffusion / transformer baselines
  * More datasets (e.g., ImageNet full, BSD, real noise)
  * More corruption types

### 2. Theoretical insight

Even partial:

* Link to:

  * Score matching theory
  * Gradient flow convergence rates
  * Energy landscape geometry

A convincing explanation of **why α ≈ 1.5** would be a major boost.

---

### 3. Scaling beyond toy regime

* Show:

  * Larger models (≥1M params)
  * Whether law holds at scale

---

### 4. Stronger evaluation

* Add perceptual metrics (LPIPS, SSIM)
* Include qualitative comparisons

---

### 5. Sharpen the contribution

Right now the message is split:

* Is it about **KAN superiority**?
* Or about **scaling law of conv-EBMs**?

You should:

> Center the paper on the scaling law, not KAN.

---

# **Final Verdict**

* **Recommendation:** Reject
* **Confidence:** Medium–High

### Reason:

The paper is **interesting, well-executed, and honest**, but:

* Lacks **depth (theory or breadth)**
* Has **limited novelty impact**
* Needs **stronger validation**

---

# **Meta-comment**

This is **close to a good NeurIPS paper**. With:

* broader experiments
* and some theoretical insight

it could become **accept-worthy in the next iteration**.

This is a substantially revised draft. Let me review it carefully against the previous version and my prior critique.---

## NeurIPS 2026 Re-Review — KAN-EBM (Revised Draft)

**Paper:** *KAN-Parameterized Energy-Based Models Enable Controllable Test-Time Compute*
**Recommendation: Accept (7/10)** — upgraded from weak accept. This revision is a meaningful improvement; the authors have addressed the most substantive criticisms squarely and honestly.

---

### What Changed and Why It Matters

This revision is not cosmetic. The authors made four structural changes that each resolve a specific criticism from the v1 review.

**1. The FLOPs accounting is now fully honest (W2 resolved).** Figure 2 is the single most important change in this draft. It presents three panels: (A) parameter Pareto, where KAN-EBM dominates; (B) compute Pareto, where FFN-DSM is clearly labeled "best per FLOP"; and (C) K-scaling, which is the actual contribution. The text explicitly states "We do not claim FLOP-Pareto-optimality." The 53× FLOP disadvantage at peak quality is not buried in a limitations section — it is in the abstract and Section 3.1. This is exactly what responsible framing looks like. Crucially, the authors go further and offer a legitimate defense of the FLOP cost for their target deployment context (edge/sensor, where weight-fetch energy dominates FLOP count), citing Horowitz 2014. This defense is reasonable, though it should ideally be tested empirically.

**2. The ConvMLP ablation (Section 3.3) is a genuine scientific contribution that substantially re-scopes the claim (W1 partially resolved, new strength added).** This is the most intellectually courageous change. The authors trained four ConvMLP-EBM variants with GELU, SiLU, Tanh, and ReLU heads at matched parameters and identical conv backbone. The finding — that all four reproduce K*(σ) ∼ Cσ^α with α=1.26 and R²=0.98, *including ReLU* — disproves the paper's own original hypothesis that B-spline smoothness is the necessary condition. The authors explicitly call this "informative and partly negative for our initial hypothesis" and restructure the contribution accordingly: the K* law is a property of the conv-backbone + per-pixel local-energy formulation; the KAN head's distinct contribution is a significantly larger exponent (Δα=+0.27, CI [+0.07, +0.46]) and +0.77 dB peak PSNR. This is more conservative, more accurate, and more credible than the original claim. NeurIPS reviewers should reward this kind of self-correction.

**3. Statistical rigor on the scaling law is now adequate (W3 resolved).** Bootstrap CIs (B=10,000) and jackknife CIs are reported on α. The sweep is extended from 5 to 9 σ anchors, and the 9-anchor refit (α=1.61, C=94.3, R²=0.97) is consistent with the 5-anchor fit. The CelebA CI is also reported ([1.18, 1.55]). This addresses the concern that a 2-parameter fit on 5 points could be coincidental.

**4. The trivial theoretical section has been correctly demoted (W5 resolved).** The Allen-Cahn/score-following/predictive-coding equivalence is now a single paragraph labeled "Algorithmic remark," explicitly stating "We do not claim this as a novel theoretical result." This is the right call and saves roughly a page of space for the ConvMLP ablation that replaced it.

---

### Remaining Weaknesses

**R1. Reproducibility is still the weakest dimension.** No code or weights are released. For a paper whose headline contribution is an empirical scaling law with a precise numerical constant (C=86.7, α=1.53), the inability to reproduce the law without reimplementing the full pipeline is a serious limitation. The training details appendix (Appendix A) gives only a single-line DSM loss equation. The Hessian eigenvalue estimation used for predictors A–D in Figure 3a is still not described with enough detail to replicate (how are the Lanczos iterations implemented? What software?). This is not a reason to reject, but it is the paper's main remaining vulnerability and should be addressed in a camera-ready with at minimum a detailed reproducibility checklist.

**R2. The constant C is dataset-specific with no guidance on its calibration.** The paper acknowledges in Section 4 that C must be re-fit on a 5-point calibration set per deployment. This is reasonable, but the concrete procedure is underspecified: how many images? Which σ values? How sensitive is the early-stopping rule to errors in the C estimate? The 99.4% oracle recovery figure is compelling, but it is reported only for the case where C is perfectly known from the training-set sweep. A practitioner who needs to calibrate C on a new dataset needs more guidance.

**R3. The significance of Δα=+0.27 deserves more discussion.** The CI [+0.07, +0.46] excludes zero, so the KAN advantage in exponent is statistically real. But the practical meaning of Δα=+0.27 is not spelled out. At σ=0.3, the difference in predicted K* is roughly: 86.7×0.3^1.53 ≈ 14 vs. 39.2×0.3^1.26 ≈ 7 — so the KAN head recommends roughly twice as many steps at high noise, which does translate to meaningfully higher peak PSNR (+0.77 dB is stated but only at σ=0.15). A table showing Δα's practical implication across the noise sweep would help practitioners understand when the KAN head is worth the additional architectural complexity.

**R4. The Checker puzzle failure (Section 3.4) is reported in one sentence and deserves a figure.** This is an important domain-of-validity finding — B-spline smoothness is an incorrect inductive bias for high-frequency signals. Reporting it without a figure or table makes it feel like a token acknowledgment. Even a two-row table (KAN-EBM vs. matched MLP on Checker, PSNR at K=1 and K=20) would sharpen the boundary.

**R5. The ImageNet-64 result in Appendix B is oddly presented.** The paper reports 25.05 dB for KAN-EBM versus 18.19 dB for FFN-DSM on ImageNet-64, but immediately notes the FFN result is single-pass. This is an apples-to-oranges comparison — a single-pass FFN at K=20 would presumably do better than 18.19 dB. Either match the comparison properly or remove it to avoid misleading readers.

---

### Minor Issues

- The abstract CI notation "95% CI on α: [1.35, 1.73]" should specify the method (bootstrap) to be complete.
- Section 3.3, point (iv) has a formatting error: "iv)the K*" is missing a space after the closing parenthesis.
- The reference list is now extremely sparse (4 references in the main paper). While this is a short-paper format, the conclusion references "Restormer, SwinIR" without citations — add them, or remove the mention.
- Figure 2's caption title "CIFAR-10 (σ=0.1): KAN-EBM is parameter-efficient, not FLOP-efficient" is good but the σ=0.1 qualification should also appear in the section body, since the trade-off axes presumably shift at different noise levels.

---

### Assessment of the Revision as a Whole

The revised paper is a significantly better piece of science than the original. The authors did not merely polish the writing — they ran new experiments (ConvMLP ablation, 9-anchor sweep, bootstrap CIs) that changed the scientific conclusions. The main claim is now more accurately scoped: the K* law is a property of conv-backbone EBMs generally, and the KAN head is the strongest member of that class rather than the unique cause. This is a harder message to sell but a more defensible one, and it actually opens a more interesting question for the community (why does local-energy formulation enable this scaling?).

The paper's contribution profile is now clear and honest: it is not a state-of-the-art denoising paper, it is not a FLOP-efficient paper, and it no longer overclaims a theoretical result. It is a careful empirical study of a scaling regularity in a specific architectural class, with a deployable early-stopping rule as the practical payoff. That is a legitimate NeurIPS contribution.

**Confidence in this review: 4/5.**