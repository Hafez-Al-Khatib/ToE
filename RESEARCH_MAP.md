# Research Map: Natural Intelligence — What Is Real and What Is Not

This document is the master reference for the project's claims, organized by
their scientific validity. Read this before working on any component.

---

## ✅ Valid Research (Publishable)

These five claims are scientifically sound and each supports at least one paper.

### [1] KAN as Interpretable Energy Density
**Read:** `valid_research/01_kan_energy_fields/CLAIM.md`
**Code:** `src/kan.py`, `src/hamiltonian_field.py`
**Experiments:** `experiments/exp_denoise.py`

> KANs replace hardcoded potentials in EBMs, producing interpretable energy landscapes
> where each B-spline function is directly visualizable. Novel, well-grounded,
> experiments already partially done.

---

### [2] Thermodynamic Field Dynamics for Image Denoising
**Read:** `valid_research/02_ebm_denoising/CLAIM.md`
**Code:** `src/thermodynamic_field.py`, `src/energy_model.py`
**Results:** `outputs/MNIST_denoising_results.png`, `outputs/FashionMNIST_denoising_results.png`

> Multi-channel learned energy fields for image restoration, mathematically equivalent
> to Allen-Cahn gradient flow. Results on MNIST/FashionMNIST already exist.
> Most mature part of the codebase.

---

### [3] Latent-Space KAN: Separation of Perception and Reasoning
**Read:** `valid_research/03_latent_hkan/CLAIM.md`
**Code:** `src/latent_hkan.py`

> CNN encoder compresses high-dimensional input to compact latent z; KAN energy
> operates on z for interpretable, parameter-efficient reasoning. The ONLY place
> in the project where "Hamiltonian" is used correctly (latent dynamics with both z and p).
> Needs training script written.

---

### [4] Differentiable Eikonal Planning
**Read:** `valid_research/04_eikonal_planning/CLAIM.md`
**Code:** `src/wave_solver.py`, `src/unified_planner.py`
**Results:** `outputs/unified_planner/`, `outputs/wave_brain/`

> Fast Sweeping Method made differentiable via adjoint backprop, enabling learning
> of speed fields n(x) from path demonstrations. Valid approach, needs to correct
> all O(1) complexity claims to O(N).

---

### [5] Test-Time Iterative Refinement
**Read:** `valid_research/05_test_time_refinement/CLAIM.md`
**Code:** `src/thermodynamic_field.py:evolve`, `experiments/exp_denoise.py:evaluate_denoising`

> The reframing of the false "O(1)" claim into the true claim: quality improves
> monotonically with gradient steps K, providing a compute-quality tradeoff that
> feedforward networks cannot offer. Connects to 2024-2025 test-time scaling research.

---

## ❌ False Claims (Do Not Publish As-Is)

These six claims are either factually wrong, unsubstantiated, or unfalsifiable.
Each has a document explaining the problem and a recommended fix.

### [1] "Hamiltonian Dynamics" — Wrong Name
**Read:** `false_claims/01_hamiltonian_naming/ANALYSIS.md`
**Severity:** HIGH

The core inference loop is gradient descent (Allen-Cahn / gradient flow), which
DISSIPATES energy. Hamiltonian dynamics CONSERVES energy. They are opposites.
**Fix:** Rename `HamiltonianField` → `GradientFlowField` throughout.

---

### [2] "O(1) Inference" — False
**Read:** `false_claims/02_o1_inference/ANALYSIS.md`
**Severity:** HIGH

Inference is O(K × N). The code even has a comment: "Complexity Honesty: O(grid_size)
iterations, not O(1)." **Fix:** Replace all O(1) claims with correct complexity statement.

---

### [3] Free Energy Principle Equivalence — Unsubstantiated
**Read:** `false_claims/03_fep_connection/ANALYSIS.md`
**Severity:** MEDIUM

FEP requires explicit generative model p(x,o) and recognition model q(x|o).
Neither exists in the codebase. **Fix:** Replace with "consistent with predictive coding."

---

### [4] Sudoku via Continuous Relaxation — Structurally Limited
**Read:** `false_claims/04_sudoku_np_hardness/ANALYSIS.md`
**Severity:** HIGH

Sudoku is NP-complete. Continuous gradient descent cannot reliably solve NP-complete
problems. 54% accuracy is the theoretical ceiling for gradient descent, not a
training failure. **Fix:** Reframe as a negative result, or drop Sudoku entirely.

---

### [5] Mass Conservation Emerges from Learned Energy — False
**Read:** `false_claims/05_mass_conservation/ANALYSIS.md`
**Severity:** MEDIUM

Allen-Cahn dynamics (what the code uses) does NOT conserve mass. Only Cahn-Hilliard
dynamics conserves mass by construction. **Fix:** Add explicit conservation loss, or
switch to Cahn-Hilliard dynamics for the pushing task.

---

### [6] Unified Theory of Natural Intelligence — Unsubstantiated
**Read:** `false_claims/06_unified_theory/ANALYSIS.md`
**Severity:** HIGH for grand narrative; LOW for individual papers

Three separate methods (EBM, KAN-EBM, Eikonal) are not a unified theory without
mathematical derivation showing they are limiting cases of common first principles.
**Fix:** Publish three separate papers; thesis becomes "Three Contributions to
Physics-Inspired Machine Learning."

---

## Publication Strategy

Based on this analysis, here is the revised paper plan:

```
Paper 1: KAN-EBM (NeurIPS / ICLR)
  → Valid claims 1, 2, 5
  → Core: KAN energy density + thermodynamic field denoising + test-time refinement
  → Experiments: exp_denoise.py (already built)

Paper 2: Latent H-KAN (NeurIPS / UAI)
  → Valid claim 3
  → Core: perception-reasoning separation with latent Hamiltonian dynamics
  → Experiments: need to write train_latent_hkan.py

Paper 3: Differentiable Eikonal Planning (ICML / ICLR)
  → Valid claim 4
  → Core: differentiable FSM + operator splitting + learned n(x)
  → Experiments: maze-solving, comparison with A* / Value Iteration

PhD Thesis: "Three Contributions to Physics-Inspired Machine Learning"
  → Three papers + synthesis chapter
  → No unified theory claim
```

---

## Immediate Next Steps (Priority Order)

1. **Rename HamiltonianField → GradientFlowField** everywhere (30 min, zero risk)
2. **Find and replace all O(1) claims** in docs and comments (1 hour)
3. **Run exp_denoise.py --dataset mnist** to get quantitative Paper 1 results
4. **Write train_latent_hkan.py** to begin Paper 2 experiments
5. **Drop FEP equivalence claims** from all documents
6. **Reframe Sudoku** as negative result with spin glass explanation

---

## Workspace Structure

```
ToE/
├── RESEARCH_MAP.md           ← This file (start here)
├── valid_research/           ← Sound science, ready to develop
│   ├── 01_kan_energy_fields/
│   ├── 02_ebm_denoising/
│   ├── 03_latent_hkan/
│   ├── 04_eikonal_planning/
│   └── 05_test_time_refinement/
├── false_claims/             ← Known problems, do not publish as-is
│   ├── 01_hamiltonian_naming/
│   ├── 02_o1_inference/
│   ├── 03_fep_connection/
│   ├── 04_sudoku_np_hardness/
│   ├── 05_mass_conservation/
│   └── 06_unified_theory/
├── src/                      ← All source code
├── experiments/              ← Paper 1 experiment pipeline
├── tutorials/                ← Concept tutorial decks
├── outputs/                  ← All trained models and result plots
└── docs/                     ← Original documentation (some claims need revision)
```
