# True Intelligence Model (TIM)

### Three Contributions to Physics-Inspired Machine Learning

This repository implements the **True Intelligence Model** — a unified architecture that
applies variational physics principles (energy minimization, Hamiltonian mechanics,
predictive coding, and free energy) to machine learning. The work spans three published
contributions and two in-progress papers.

---

## Core Idea

A single KAN (Kolmogorov-Arnold Network) potential `V_KAN(z)` serves three simultaneous
roles in TIM:

```
V_KAN(z)
  ├── (A) Attractor landscape  →  Predictive coding inference target
  ├── (B) Hamiltonian potential →  Symplectic trajectory search in latent space
  └── (C) Refractive index      →  Eikonal path planning:  n(z) = 1/(α|V|+β)
```

This triple role is the central architectural contribution. All inference mechanisms are
special cases of Wasserstein gradient flow of the free energy functional F(ρ) = ∫ρ log ρ + ∫ρV
(Jordan–Kinderlehrer–Otto 1998).

---

## Quick Start

```bash
# Install
pip install -e .

# Run all experiments (full suite, Paper 1 + Paper 2 + TIM)
cd experiments
python run_experiments.py

# Run only benchmark experiments (MNIST/FashionMNIST denoising)
python run_experiments.py --benchmarks-only

# Run only FEP validation (mathematical proofs)
python run_experiments.py --fep-only

# Run only test-time compute scaling experiments
python run_experiments.py --scaling-only

# Quick smoke-test (reduced epochs)
python run_experiments.py --quick
```

Results are saved to `outputs/` with timestamps. Each experiment also saves a JSON summary.

---

## Architecture Overview

```
Observation o
  │
  ├─► [RecognitionKAN]          q_φ(z|o)  — amortised variational inference
  │     KAN encoder: obs_dim → latent_dim
  │
  ├─► [HierarchicalPCKAN]       ε_l = Π^½(x_l − KAN_l(x_{l+1}))
  │     Multi-layer predictive coding with local Hebbian learning
  │
  ├─► [LatentHKAN_FEP]          H(z,p) = ½‖p‖² + V_KAN(z)
  │     Symplectic Euler in latent space, full FEP loss
  │
  └─► [GenerativeDecoder]       p_θ(o|z)  — reconstruction
        KAN decoder: latent_dim → obs_dim

Free energy: F = KL[q_φ(z|o) || p(z)]  −  E_q[log p_θ(o|z)]
```

All four modules are integrated in `src/tim.py` as `TrueIntelligenceModel`.

---

## Repository Structure

```
src/
├── kan.py                     KAN: B-spline activations (Liu et al. 2024)
├── thermodynamic_field.py     Base class: spatial filter bank, step_size
├── predictive_coding_field.py Allen-Cahn gradient flow for image inference
├── recognition_model.py       RecognitionKAN + GenerativeDecoder + VFE
├── hierarchical_pc_kan.py     Multi-layer PC with local Hebbian updates
├── latent_hkan.py             Latent H-KAN (legacy, no recognition model)
├── latent_hkan_fep.py         Latent H-KAN with full FEP compliance
└── tim.py                     TrueIntelligenceModel unified class

experiments/
├── run_experiments.py         Master runner with per-paper flags
├── exp_benchmarks.py          Paper 1: MNIST/FashionMNIST vs MLP/FFN baselines
├── exp_compute_scaling.py     Paper 1: PSNR vs K, FLOPs, noise robustness
├── exp_fep_validation.py      Paper 2: 4 mathematical validation experiments
├── exp_local_learning.py      Paper 2: PC Hebbian vs backprop comparison
└── train_latent_hkan.py       Paper 2: Full LatentHKAN_FEP training pipeline

docs/
├── TIM_Architecture.md        Full architecture reference (equations + interfaces)
├── theory_foundations.md      JKO/Wasserstein mathematical foundations
├── phase1_explained.md        Phase 1 (EBM denoising) theory
├── phase2_explained.md        Phase 2 (H-KAN) theory
└── phase3_explained.md        Phase 3 (Eikonal) theory
```

---

## The Four Modules

### 1. `PredictiveCodingField` — Spatial Inference

Implements Allen-Cahn gradient flow on image fields:

```
∂u/∂t = −δE/δu
```

Spatial KAN energy `E(u)` computed via convolutional filter bank → KAN. K steps of
gradient descent improve PSNR monotonically (test-time compute scaling). At K→∞ this
converges to the MAP estimate.

See: `src/predictive_coding_field.py`, `experiments/exp_benchmarks.py`

### 2. `RecognitionKAN` + `GenerativeDecoder` — Variational Inference

Encodes observations to `(μ_z, log σ²_z)` and decodes latents to reconstructions.
Enables computation of the full variational free energy:

```
F = KL[q_φ(z|o) || p(z)]  −  E_{q_φ}[log p_θ(o|z)]
```

See: `src/recognition_model.py`

### 3. `HierarchicalPCKAN` — Local Learning

Multi-layer predictive coding hierarchy. Local Hebbian update at each layer:

```
ΔW_l = η · ε_l · ∂KAN_l/∂W_l
```

No global backpropagation required. Mathematically equivalent to backprop at equilibrium
(Song et al., NeurIPS 2020). Enables neuromorphic/on-chip learning.

See: `src/hierarchical_pc_kan.py`, `experiments/exp_local_learning.py`

### 4. `LatentHKAN_FEP` — Hamiltonian Latent Dynamics

Symplectic Euler integration in latent space for trajectory-based inference:

```
p_new = p − dt · ∇V(z)
z_new = z + dt · p_new         (symplectic: uses updated p)
```

Test-time quality improves with number of Hamiltonian steps K. Connects to Aitchison &
Lengyel (2016) neuromorphic interpretation (z = excitatory, p = inhibitory, E/I balance
= symplectic structure).

See: `src/latent_hkan_fep.py`, `experiments/train_latent_hkan.py`

---

## Benchmark Experiments

**Paper 1 (KAN-EBM):** MNIST and FashionMNIST denoising at σ ∈ {0.1, 0.2, 0.3}.

Models compared:
- `FFNBaseline` — feedforward denoiser (no inference steps)
- `MLPEBMBaseline` — same filter bank as PCField, MLP energy instead of KAN
- `PredictiveCodingField` — KAN energy, K-step gradient flow

Primary metric: PSNR at K = 1 and K = 10. Secondary: SSIM.

**Paper 2 (FEP + Latent H-KAN):** Four validation experiments to confirm mathematical
equivalences, then training of `LatentHKAN_FEP` on MNIST with comparison against a VAE
baseline at matched parameter count.

---

## Terminology Reference

Some terms were incorrect in earlier versions of this codebase. The corrected names:

| Old (incorrect) | New (correct) |
|---|---|
| `HamiltonianField` | `PredictiveCodingField` (Allen-Cahn is gradient flow, not Hamiltonian) |
| "O(1) inference" | "O(K×N) inference, K controllable" |
| "FEP equivalent" | "FEP-compliant with recognition model" |
| "Unified theory of intelligence" | "Three contributions to physics-inspired ML" |

For the full terminology table and mathematical derivations, see
`docs/TIM_Architecture.md`.

---

## Multi-Timescale Learning

TIM implements three nested learning timescales (Behrouz et al., NeurIPS 2025):

| Timescale | Neural analog | TIM variable | Rate |
|---|---|---|---|
| Fast γ | Gamma oscillations | PC state `x_l` | Every sample |
| Medium β | Beta / spindles | Precision `Π_l` | Every ~10 steps |
| Slow δ/θ | Theta / delta | Weights `W_l` | Every ~100 steps |

This mirrors hippocampal theta-gamma coupling and enables stable online learning without
catastrophic interference.

---

## Mathematical Chain

The four modules are related by a single theoretical chain:

```
Allen-Cahn (gradient flow)
    ≡ Score following (Vincent 2011, DSM)
    → Prediction error minimization (Rao & Ballard 1999)
    → Variational free energy (Bogacz 2017, Laplace approx)
    ⊂ Wasserstein gradient flow (JKO 1998)
```

This means all of the following are computing the same quantity under different
parameterisations: denoising score matching, predictive coding inference, ELBO gradient
descent, and Allen-Cahn field relaxation.

---

## References

- Jordan, Kinderlehrer, Otto (1998). "The variational formulation of the Fokker–Planck equation."
- Rao & Ballard (1999). "Predictive coding in the visual cortex." Nature Neuroscience.
- Vincent (2011). "A connection between score matching and denoising autoencoders." Neural Computation.
- Bogacz (2017). "A tutorial on the free-energy framework for modelling perception and learning." J. Math. Psychology.
- Aitchison & Lengyel (2016). "The Hamiltonian Brain." PLOS Computational Biology.
- Song et al. (2020). "Can the brain do backpropagation?" NeurIPS 2020.
- Liu et al. (2024). "KAN: Kolmogorov–Arnold Networks." arXiv 2404.19756.
- Snell et al. (2024). "Scaling LLM test-time compute optimally." arXiv 2408.03314.
- Behrouz et al. (2025). "Titans: Learning to memorize at test time." NeurIPS 2025.

---

*Research at AUB. Contact: hafezkhatib.2002@gmail.com*
