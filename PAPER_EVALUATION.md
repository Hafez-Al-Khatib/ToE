# Full Paper Evaluation: KAN-EBM NeurIPS Submission

## Executive Summary

The paper makes a genuine empirical discovery: conv-backbone EBMs exhibit a power-law scaling 
K*(σ) ~ Cσ^α for optimal inference depth. The core issue is that the narrative has evolved
through multiple rebuttal rounds and now contains internal contradictions, especially around
the coarse-grid vs fine-grid results and the "identical sequence" claim.

**Good news:** The new fine-grid data is stronger than the old narrative. Both KAN and ConvMLP
follow power laws with R²>0.997, but KAN has a steeper exponent (α=1.365 vs 1.132). This means
KAN's advantage is not just quality at each step, but also *scaling steepness*—it benefits more
from additional inference compute. This is a more defensible and more interesting claim.

---

## 1. Critical Narrative Issues (Must Fix)

### 1.1 "Identical Sequence" Claim is Misleading

**Current text (line 157):** "all conv-backbone variants---KAN, GELU, SiLU, Tanh, and ReLU---share
the identical optimal-step sequence {1,2,5,5,10}"

**Problem:** This is true on the coarse grid but hides the fact that fine-grid evaluation reveals
different exponents. The coarse grid's discrete steps {1,2,5,10,20} collapse genuinely different
continuous K* values to the same discrete values.

**New data:**
| σ    | KAN fine K* | ConvMLP fine K* | Coarse (both) |
|------|-------------|-----------------|---------------|
| 0.05 | 1.01        | 1.00            | 1             |
| 0.10 | 2.39        | 1.98            | 2             |
| 0.15 | 4.32        | 3.27            | 5             |
| 0.20 | 6.52        | 4.69            | 5             |
| 0.30 | 11.48       | 7.45            | 10            |

**Fix:** Reframe as: "The power-law *form* is universal across conv-backbone architectures, but
the *exponent* is architecture-dependent. KAN achieves the steepest scaling (α=1.365 vs 1.132
for ConvMLP-GELU). The coarse grid masked this difference by discretizing to the same step set."

### 1.2 "Measurement Artifact" Claim Needs Revision

**Current text:** "The 24-image α=1.53 vs. 1.26 difference was a measurement artifact of coarse
grid sampling that vanishes at 500 images."

**Problem:** The 24-image difference was partly artifact, but the fine-grid shows KAN genuinely
has a larger exponent than ConvMLP (1.365 vs 1.132, Δα=+0.233). The original +0.27 was close
to the true difference!

**Fix:** "On 24-image evaluation, sampling noise produced α=1.53 (KAN) vs. 1.26 (ConvMLP), a
difference of +0.27. Fine-grid evaluation at 500 images confirms KAN's exponent is indeed larger
(1.365 vs. 1.132, Δα=+0.233), though the 24-image values were individually inflated by small-sample
noise."

### 1.3 Abstract Overclaims

**Current:** "all four MLP heads reproduce it"

**Fix:** "all four MLP heads reproduce the power-law *form*, with KAN achieving the steepest
exponent."

---

## 2. Structural Issues

### 2.1 Section 3.2 is Overstuffed

The scaling law section currently contains:
- Empirical fit (original + 9-anchor + CelebA)
- 500-image coarse-grid results
- Fine-grid validation
- Toy model derivation
- σ-vs-Hessian explanation
- Step-size schedule robustness

That's 6 distinct claims in one subsection. The step-size robustness and fine-grid validation
should be their own subsubsections or at least clearly separated paragraphs.

### 2.2 Figure Placement

Figure 4 (fine-grid) is placed in §3.2 but references content from §3.5 (step-size). The figure
caption should be self-contained.

### 2.3 Section 3.5 (Scaled evaluation) is Misnamed

It contains:
- Scaled evaluation table (main)
- Predictive validation (subsubsection)
- Latency (subsubsection)
- Falsifiable boundary (separate subsection)

The predictive validation and latency should arguably be merged into the main text rather than
subsubsections, or the section should be renamed.

---

## 3. Content Gaps

### 3.1 Missing: ConvMLP Fine-Grid Results

The paper has fine-grid for KAN but not ConvMLP. The new data (α=1.132) is essential for showing
the power-law form generalizes while the exponent varies.

### 3.2 Missing: AlexNet LPIPS Validation

We have AlexNet results (identical to SqueezeNet). The paper should mention this explicitly to
address reviewer concerns about backbone choice.

### 3.3 Missing: Fine-Grid K* Uncertainty Intervals in Main Text

The paper mentions per-image K* std but doesn't report the bootstrap CIs for K* itself prominently.
These should be in a table.

---

## 4. Language and Tone Issues

### 4.1 Hedge Words

The paper uses "correlates with" heavily (limitation 4, conclusion). This is honest but makes
the claims sound weak. Consider: "The geometric advantages translate directly into sustained
empirical improvement" (line 191) is stronger than "correlate with quality but do not establish
causation" (line 240). Both are present. Pick one framing and stick to it.

### 4.2 Passive Voice

Too many sentences start with "We show that..." or "This paper characterises...". Vary the
sentence structure.

---

## 5. Recommendations

### Priority 1: Reframe the Core Claim
Change from "all architectures share the same sequence" to "all architectures follow power laws,
but KAN has the steepest exponent."

### Priority 2: Add Fine-Grid Comparison Table
A table showing KAN vs ConvMLP fine-grid K* values and exponents.

### Priority 3: Separate Overstuffed Sections
Split §3.2 into: (a) Empirical fit, (b) Fine-grid validation, (c) Toy model, (d) Schedule robustness.

### Priority 4: Update Abstract and Conclusion
Reflect the new nuanced claim.

### Priority 5: Add AlexNet LPIPS Mention
Brief sentence confirming AlexNet gives identical results.
