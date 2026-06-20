# Future Research Line: KAN-EBM as Iterative Inference Substrate

**Status:** Draft (2026-04-26). Companion to the NeurIPS 2026 submission.

## Premise

The current paper establishes a single, narrow claim: **a KAN-parameterized
energy function admits monotonic test-time-compute scaling at micro-parameter
budgets where MLP energies suffer topological collapse.** Lanes A and B extend
this with one-prior/many-tasks evidence. That is a wedge — not a "we changed
AI" moment, and packaging it as one would crater peer review.

The actual change-AI thesis lives in a **research line** of 3–5 papers, each
extending the wedge into a distinct axis. This document maps the line.

---

## Paper 1 (current submission): "KAN-EBM: Compositional Energy Landscapes for Test-Time-Compute Scaling"

**Venue target:** NeurIPS 2026 (main track) or, if reviewer signals are weak,
TMLR (no acceptance ceiling and rolling submission).

**Claim space (locked after Lane A/B + cleanup):**
- KAN parameterization of energy enables monotonic K-scaling at <100K params
  on grayscale and color images.
- The same energy handles five inverse problems (denoise / inpaint random /
  inpaint box / SR-4× / deblur / JPEG-AR) zero-shot via gradient descent,
  with peak-and-degrade observed at moderate K.
- The same architectural template handles abstract grid-puzzle constraint
  satisfaction (Latin-square completion) with K-scaling, demonstrating the
  inference operator transfers from perception to symbolic reasoning under
  the same energy-minimization mechanism.
- Algorithmic equivalence (Proposition 1) of Allen–Cahn / score / PC under
  shared E.

**What this paper does NOT claim:**
- Beat DnCNN/BM3D in absolute PSNR.
- Scale to ImageNet-64 / DIV2K / video (separate paper).
- Solve language-modeling tasks (out of architecture scope).
- Be a paradigm shift. It's a wedge.

---

## Paper 2: "Inference-Time Scaling Laws for Energy-Based Models"

**Venue target:** ICML 2027 / NeurIPS 2027.

**Premise:** Paper 1 shows K-scaling is real but peak-and-degrade is poorly
understood. There is no theory of *when* the peak occurs as a function of
data complexity, model capacity, or noise level — and no principled
early-stopping rule beyond "validation pick K*".

**Concrete contributions:**
1. **Empirical scaling law:** Across MNIST / CIFAR-10 / ImageNet-64 /
   CelebA-64 trained KAN-EBMs at parameter counts in {5K, 30K, 100K, 1M},
   measure peak K* vs. data manifold dimensionality (estimated via
   intrinsic-dim methods like TwoNN), noise level σ, and parameter count P.
   Hypothesis: log K* ∝ α log(P) − β log(σ) + γ log(d_intrinsic). Test against
   held-out datasets.
2. **Theoretical prediction:** Treat the energy landscape as a Gaussian
   random field around the data manifold; derive K* from the spectrum of
   the local Hessian (lowest non-zero eigenmode controls the timescale of
   manifold departure).
3. **Adaptive early stopping:** Stop when the per-step relative energy
   drop ΔE_t / E_t falls below an automatically-set threshold derived from
   the K* prediction. No validation required at inference.
4. **Projected gradient dynamics:** Bound the trajectory inside an
   ε-tube around training data via TV projection or learned manifold
   projection. Eliminates peak-and-degrade entirely if the projector is
   accurate.

**Why this is a strong second paper:** Paper 1 honestly admits peak-and-degrade
as a limitation. Paper 2 promotes it from limitation to *predictable
phenomenon* with a principled fix. Reviewers love a story arc.

**Pre-experiments needed (~2 months):**
- Train KAN-EBMs at 4 param scales × 4 datasets × 3 σ levels = 48 configs.
- Implement intrinsic-dim estimator.
- Implement learned manifold projector (autoencoder bottleneck).

---

## Paper 3: "Hierarchical KAN-EBMs: Predictive Coding for Compositional Inference"

**Venue target:** NeurIPS 2027 / ICLR 2028.

**Premise:** Paper 1's KAN-EBM is shallow (1–3 KAN layers). It captures local
spatial statistics through the filter bank but cannot represent compositional
structure across spatial scales. The Friston/PC connection in Section 4 is
only suggestive — the architecture is single-level.

**Concrete contributions:**
1. **Hierarchical architecture:** Stack KAN-EBMs at multiple spatial scales
   (e.g., 32×32 → 16×16 → 8×8 patches), each level predicting / refining
   the level below it. Inference becomes coarse-to-fine gradient descent on
   a sum of energies at each level, with bidirectional precision-weighted
   message passing.
2. **Predictive-coding interpretation:** Show that the architecture
   *literally* implements Friston's hierarchical free-energy minimization
   on a learned generative model g(z). The paper makes the PC connection
   concrete rather than analogy.
3. **Higher-resolution evaluation:** ImageNet-64, CelebA-64, DIV2K-128.
   Peak-and-degrade should be dramatically delayed because each level has
   its own attractor and the cross-level coupling stabilizes the trajectory.
4. **Latent-space variant (`latent_hkan.py` already in repo):** Place the
   KAN-EBM in the latent space of a pretrained VAE / VQ-GAN. This is the
   only legitimate path to *high-resolution* outputs without exploding
   parameter counts.

**Pre-experiments needed (~3-4 months):**
- Engineering: hierarchical training is fiddly (per-level learning rates,
  precision balancing).
- Baseline: hierarchical VAE / VQ-GAN at matched param counts.
- Latent variant: requires choosing and freezing a pretrained autoencoder.

**Why this matters for the line:** This is where "tiny models" stops being
a constraint and starts being a *feature* — small EBMs at each scale
compose into systems that handle real-world image complexity at total
parameter counts << any monolithic generative model.

---

## Paper 4: "Energy Minimization is Search: Iterative Inference for Grid-World Planning"

**Venue target:** ICLR 2028 / NeurIPS 2028.

**Premise:** Lane B in the current paper hints at a deeper connection
between energy minimization and search. The Eikonal/WaveBrain work in the
repo (P1–P6 in REGISTERED_CLAIMS) was paused but contains the right
machinery. Combine it with Paper 2's scaling theory.

**Concrete contributions:**
1. **Unified gridworld formulation:** Maze planning, Sudoku, ARC-AGI mini,
   bin-packing, graph coloring — all expressed as 2D-grid constraint
   satisfaction problems. Each gets a KAN-EBM trained on valid solutions.
2. **Test-time inference is search:** With Paper 2's scaling laws, predict
   the K* required for a given puzzle's complexity. Show that K* correlates
   with branching factor / search depth in classical solvers.
3. **Comparison to Neural-A* and SAT solvers:** Position KAN-EBM as a
   continuous-relaxation alternative that scales to puzzle sizes where
   discrete solvers struggle (or vice versa).
4. **Hybrid: discrete projection + continuous refinement.** After K
   continuous gradient steps, project to nearest valid discrete state
   and resume. This is the KAN analog of MCTS rollouts.

**Why this is strategically important:** This is the "System 2 reasoning"
paper. Done right, it positions the KAN-EBM line as a *cousin* to
chain-of-thought / o1-style reasoning — not in language but in continuous
problem-solving — which is a much more reviewable claim than "we have
LLM reasoning at 5K params."

---

## Paper 5: "Distillation: Iterative Energies as Training Curricula for Feedforward Models"

**Venue target:** ICML 2028 / NeurIPS 2028.

**Premise:** Paper 1's K=1 KAN-EBM is *worse* than a feedforward network of
the same parameter budget. The K=20 KAN-EBM is *better*. What if we use the
K=20 KAN-EBM as a teacher to train a feedforward student? Does the student
inherit the energy landscape's good behavior at single-pass cost?

**Concrete contributions:**
1. **Distillation protocol:** For each input x, compute u_K = K-step
   refinement of KAN-EBM. Train an FFN to map x → u_K. Single-forward-pass
   inference; teacher cost only at training time.
2. **Quality-vs-compute Pareto:** Sweep K_teacher ∈ {1, 5, 20, 50}.
   Map out FFN-student PSNR as a function of teacher compute budget.
3. **Surprising prediction:** The student should outperform a directly-trained
   FFN of the same parameter budget, because the K=20 targets are closer
   to the true clean image than the noisy targets seen in standard DSM.
4. **Iterated distillation (curriculum):** Train FFN on K=5 KAN targets,
   then use that FFN as a warm-start for a K=10 KAN, then re-distill.
   Tests whether the energy landscape continues to provide useful gradient
   even after distillation collapses some of its iterative refinement.

**Why this matters:** This converts the "KAN-EBM is slow at inference" problem
(B-spline FLOPs) into a feature: pay the cost once at training, get a fast
single-pass model that inherits the iterative-inference benefit. This is a
direct response to the most common reviewer critique of KAN-based methods
("but B-splines are slow on GPUs").

---

## Cross-cutting ablations across the line

These probes are useful for Papers 2–5 but don't fit any single one:

| Probe | Paper(s) it informs | Cost |
|---|---|---|
| **Spline order** $p \in \{2, 3, 5\}$ on K-scaling | 2, 3 | low |
| **Grid size** $G \in \{3, 5, 8, 16\}$ scaling laws | 2 | low |
| **Replace B-splines with rational basis (KAT-style)** for FLOP audit | 2, 5 | medium |
| **EBM density-as-density**: AUROC-OOD across all trained checkpoints | 2 | low |
| **Symplectic vs Euler vs RK4 integrator** for inference | 3, 4 | low |
| **Exact-vs-approximate Hessian in DSM training** | 2 | medium |
| **Replace filter bank with frozen pretrained encoder (CLIP, DINO)** | 3, 5 | medium |

---

## Anti-patterns to avoid (explicit don't-do list)

- **Do not pivot to NLP.** The architecture is image-shaped. Cross into
  language only if/when a Paper-3-style latent variant exists with a frozen
  language encoder, and even then only as a single section.
- **Do not chase ImageNet-1K classification.** Energy-as-density does not
  win FID/IS at large scale at micro-param budgets. We will lose to
  diffusion models and reviewers will (correctly) call us out.
- **Do not retire the "test-time compute" framing.** It is the line's
  spine; switching to "we have a new generative model" loses the wedge.
- **Do not silently rename old experiments to escape failed claims.** The
  Section E debacle (REGISTERED_CLAIMS v1.2 with mismatched JSON) is a
  cautionary tale. Every claim ships with its log file or it does not ship.

---

## Recommended sequencing

```
Year 1 (2026-2027):  Paper 1 (current submission cycle)
                     Paper 2 pre-experiments begin Q3
Year 2 (2027-2028):  Paper 2 (mid-2027)
                     Paper 3 + Paper 4 in parallel (different dimensions)
Year 3 (2028-2029):  Paper 3 + Paper 4 ship
                     Paper 5 (distillation) is the synthesis paper that
                     ties the line into a deployable system
```

If Paper 1 lands at NeurIPS, Papers 2–5 inherit the venue credibility and
each becomes individually reviewable on its own merits. If Paper 1 lands
at TMLR, the line still works but each subsequent paper needs to re-prove
the wedge in its abstract.

---

## What "changing AI" actually looks like for this line

Not: "we replaced transformers."

But: *if the line is fully realized*, the contribution is a principled,
parameter-efficient, interpretable mechanism for **iterative inference under
a learned scalar potential** that scales from low-level vision to abstract
constraint satisfaction. That is a mechanism the field currently lacks at
sub-LLM scales, and it would matter most in:

- Edge / on-device inference (where parameter count is binding)
- Scientific computing (where interpretability of the energy is binding)
- Hybrid neural-symbolic systems (where iterative refinement under
  constraints is the natural fit)

These are real markets, not "all of AI." But they are markets where
this line could plausibly become a default.
