# Design: The Inference Frontier — Head Capacity vs Depth at Fixed FLOPs (+ GroupKAN head)

**Date:** 2026-07-02
**Status:** Approved by user (option: "Approve as designed")
**Target:** AAAI 2027 submission (`paper_v8.tex`); abstract deadline ~2026-07-21, full paper ~2026-07-28.

## 1. Decision record

The user proposed expanding/reconstructing the paper around "Implicit Chain-of-Thought (ICoT)
through GroupKANs" with three angles. Literature reality-check (2026-07-02) found:

- **Collision:** EBM-CoT (arXiv 2511.07124, Nov 2025) already refines latent chain-of-thought
  representations with an energy-based model. The "energy descent over latent thoughts" mechanism
  is taken; only architecture-level or law-level wedges remain.
- **Angle 1 (spline interpretability):** premise partially wrong — KAN splines are frozen at
  inference; only the operating point moves along fixed curves. KAN interpretability does not
  survive composition/depth (arXiv 2407.11075), and interpretable latent reasoning is already an
  active subfield (SPOT, arXiv 2603.06222). Demoted to an optional qualitative figure.
- **Angle 2 (curse of dimensionality):** premise unsound — the KST-bypasses-COD claim is refuted
  for trainable networks (Girosi & Poggio 1989; dimension-independent rates hold only for
  restricted function classes). REJECTED; do not build.
- **Angle 3 (fixed-FLOP inference frontier):** sound, falsifiable, cheap, and extends the existing
  paper's depth law with the missing width-vs-depth axis. ACCEPTED as the AAAI addition.

**Chosen path: two-paper split.**
- **Now (this spec):** de-risk + build Angle 3 in the paper's existing denoising domain; add a
  GroupKAN (GR-KAN-style, per KAT, Yang & Wang ICLR 2025) energy head as the strongest light-head
  point and a 6th architecture for the α table.
- **Later (captured in §7, not built):** Paper 2 = "a depth law for latent-thought refinement" —
  transplant K* to the EBM-CoT setting at GPT-2 scale. The LLM bridge to the supervisor's interests.

**Why the frontier experiment is win-win for `paper_v8`:** the paper's identity is "a depth law,
and what it does not buy." A positive result adds a practical substitution law (depth can replace
head capacity at fixed FLOPs); a negative result is a third honest negative (depth does NOT
substitute for capacity). Both outcomes get a section; the write-up is pre-drafted for both.

## 2. Component 1 — De-risk experiment: `experiments/exp_inference_frontier.py` (Jul 02–05)

**Question:** at matched *total inference FLOPs*, does a lighter energy head running more descent
steps reach higher denoising quality than a heavier head running fewer steps?

**Models (existing checkpoints only; no new training in this component):**
| Head | Params | Checkpoint |
|---|---|---|
| KAN-EBM 110K | 110K | `outputs/cifar10/kan_ebm_f32.pt` |
| KAN-EBM 32K | 32K | `outputs/finalization/kstar_param_scaling/kan_small.pt` (n_filters=16, kan_hidden=[48,16]) |
| ConvMLP-GELU | 35K | `outputs/finalization/rebuttal/conv_mlp_gelu.pt` (n_filters=16, mlp_hidden=160) |
| UNet-EBM | (as trained) | locate via `outputs/fine_grid_kstar/fine_grid_unet.json` provenance; if the checkpoint is missing, drop UNet — three heads suffice |

**Protocol:**
- σ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}; K = 0…30; 500 CIFAR-10 test images; 2 seeds.
- Per-step cost: analytic FLOPs for one energy forward + backward (backward ≈ 2× forward), counted
  per image. Wall-clock per step from `experiments/benchmark_latency.py` reported as a secondary
  axis (FLOPs are primary; wall-clock is hardware-confounded).
- Metric: PSNR at each K (primary); LPIPS at the per-model best-K point (secondary).
- Output: `outputs/inference_frontier/frontier.json` (all curves + per-step costs) and a
  Pareto figure `outputs/inference_frontier/frontier.png` — quality vs cumulative FLOPs, one curve
  per head, one panel per σ.

**Pre-registered pass criterion (decides the paper section's direction, not whether it exists):**
PASS (substitution holds) if a lighter head beats a heavier head by **≥0.3 dB PSNR at matched total
FLOPs over a contiguous budget range**, consistently across **≥3 of 5 σ values and both seeds**.
"Lighter/heavier" requires strictly fewer params AND per-step FLOPs. **Primary pair: KAN-32K vs
KAN-110K** (same family — isolates capacity from architecture); cross-family pairs (e.g. KAN-32K vs
ConvMLP-GELU) are secondary evidence. FAIL (no substitution) otherwise → the section reports the
negative.

**Environment:** RTX 4090 via SSH; RTX 3080 fallback (models are small). One torch process at a
time. Resumable: write partial JSON per (model, σ, seed) with a DONE sentinel.

## 3. Component 2 — GroupKAN energy head (Jul 06–08)

- `src/group_kan.py`: GR-KAN-style head — channels split into G groups, each group sharing one
  learnable univariate function (rational or spline), so the layer computes group-shared 1D
  transforms + a standard linear mix (matmul-friendly). Cite KAT (Yang & Wang, ICLR 2025) for the
  grouping idea; we claim only the *application as an EBM energy head*, not the mechanism.
- Sizes: two variants targeting ~8K and ~32K params (lighter than every existing head, and
  parameter-matched to KAN-32K respectively). G chosen so per-step FLOPs < KAN-32K's.
- Training: the *identical* DSM recipe from `experiments/exp_cifar10.py` (same conv backbone,
  same σ-sampling, same optimizer/epochs). Matched recipe is non-negotiable — it is what makes
  the frontier comparison fair.
- Measurements: (a) add GroupKAN curves to the frontier (rerun Component 1 protocol on it);
  (b) fine-grid K* sweep → α for the paper's architecture table (6th architecture; recompute the
  architecture-independence F-ratio with it included).
- Both measurements are valid regardless of PASS/FAIL: a new α row strengthens
  iteration-not-architecture even if the Pareto is negative.

## 4. Component 3 — Paper edits to `paper_v8.tex` (Jul 09–14, alongside GroupKAN frontier rerun)

- New results subsection after the corruption phase boundary: **"The inference frontier: trading
  head capacity for depth."** Contents: setup paragraph, the Pareto figure, the pass/fail-scoped
  claim, GroupKAN α row added to the architecture table, F-ratio updated.
- Two pre-drafted framings (pick by outcome):
  - PASS: "At fixed inference FLOPs, additional descent depth substitutes for energy-head capacity:
    a G-grouped KAN head at 8K params matches/exceeds a 110K head given its FLOP budget in steps.
    The depth law K*(σ) tells you how many steps that budget should buy."
  - FAIL: "Depth does not substitute for head capacity: at matched FLOPs, heavier heads dominate at
    every budget. Together with §6 (no per-instance adaptivity; refinement is dissipative), this
    bounds what iteration can buy."
- Abstract + contributions list updated with one sentence, scoped to the outcome.
- **Stretch goal (only if all above lands by ~Jul 14):** one qualitative figure — latent
  coordinates sliding along *fixed* GroupKAN splines during descent. Framed strictly as a
  visualization of the refinement trajectory; the words "interpretable"/"reading thoughts" do not
  appear.

## 5. Timeline & buffer

- Jul 02–05: Component 1 (de-risk) → decision gate.
- Jul 06–08: Component 2 (GroupKAN implement + train + α).
- Jul 09–11: GroupKAN frontier rerun + figures.
- Jul 12–14: Component 3 paper edits.
- Jul 15–21: buffer; citation verification (RaM author list, Christie 2015, Lieder 2020 — still
  open from the previous pass); Overleaf/AAAI-template port.

## 6. Risks

- **FLOP accounting disputes:** mitigate by publishing the counting convention (forward+backward,
  backward≈2×forward) in the section and reporting wall-clock as the secondary axis.
- **GroupKAN trains poorly under the matched recipe:** report it as-is (fair-recipe result); do
  not tune it beyond what other heads received. If it fails to reach baseline denoising quality at
  σ=0.10 (>2 dB below KAN-32K), drop it from the frontier and keep only the α row if the fit is
  clean (R²>0.95), else drop entirely.
- **UNet checkpoint missing:** drop UNet (contingency in §2).
- **GPU flakiness:** resumable JSON + DONE sentinels; 3080/CPU fallback; one torch process rule.
- **Timeline slip past Jul 14:** cut the stretch goal first, then cut the GroupKAN frontier rerun
  (keep the α row), never cut Component 1 — the section can stand on existing heads alone.

## 7. Paper 2 capture (NOT built now)

**Working title:** "How Many Latent Thoughts? A Depth Law for Latent-Space Reasoning Refinement."
**Wedge:** EBM-CoT (2511.07124) refines latent CoT with an EBM but gives no principled answer to
how many refinement steps to take. Our K*(difficulty) law is exactly that question. **Testbed:**
GPT-2-scale (124M) latent-CoT model (Coconut/CODI-style) + an EBM-CoT-style energy head over
latent thoughts; measure accuracy vs refinement depth K across task-difficulty strata; test for an
interior optimum K* and a law K*(difficulty). GroupKAN head is the efficiency vehicle if the
frontier result (this spec) is positive. **De-risk first (post-deadline):** reproduce an
EBM-CoT-lite at small scale and check that accuracy-vs-K even has an interior optimum before
claiming any law. 4090-feasible. This is the LLM-facing project for the supervisor.

## 8. Verification

- Every number in the new paper section regenerates from `outputs/inference_frontier/*.json` by
  rerunning the scripts; no hand-copied values.
- 2 seeds minimum for every frontier curve; the pass criterion requires consistency across both.
- The GroupKAN α row uses the same fine-grid protocol as every other row (KIMI_HANDOFF §1 protocol:
  K=1–30, five σ, 500 images, 3 seeds).
- Section text is checked against the PASS/FAIL verdict before submission (no leftover wrong-branch
  wording).

## 9. Explicitly out of scope

- Angle 2 (curse-of-dimensionality claims) — unsound premise, rejected.
- Any ICoT/LLM experiment before the AAAI deadline.
- Any claim of KAN "interpretability" — the stretch figure is a visualization only.
- Retuning training recipes per-architecture (breaks fairness).
