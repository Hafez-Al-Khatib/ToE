# True Intelligence Model (TIM): Architecture Reference

## Overview

The True Intelligence Model is a physics-inspired neural architecture that unifies four
principles under a single mathematical framework:

1. **Energy-Based Perception** — KAN energy landscapes for denoising and scene parsing
2. **Predictive Coding** — Hierarchical inference via local Hebbian learning
3. **Hamiltonian Latent Dynamics** — Symplectic trajectory search in latent space
4. **Variational Free Energy** — Full FEP compliance with recognition and generative models

The central design choice is that a single KAN potential `V_KAN(z)` serves three
simultaneous roles: (A) the attractor landscape for predictive coding, (B) the potential
energy for Hamiltonian inference, and (C) the refractive index for Eikonal path planning.
This triple role is the core novel contribution of TIM.

---

## Mathematical Foundations

### 1. The Unifying Chain (JKO 1998)

All four inference mechanisms are instances of Wasserstein gradient flow of the free-energy
functional:

```
F(ρ) = ∫ ρ(x) log ρ(x) dx  +  ∫ ρ(x) V(x) dx
         \_______entropy_______/  \_____potential_____/
```

The JKO (Jordan–Kinderlehrer–Otto 1998) theorem states that the Fokker–Planck equation is
the gradient flow of F(ρ) with respect to the 2-Wasserstein metric. This unifies:

| Module | Continuous dynamics | Discrete update |
|---|---|---|
| `PredictiveCodingField` | Allen-Cahn: `∂u/∂t = −δE/δu` | `u ← u − dt·∇E(u)` |
| `HierarchicalPCKAN` | Prediction error: `∂x_l/∂t = −ε_l + J_l^T ε_{l−1}` | Inference loop |
| `LatentHKAN_FEP` | Hamilton: `dz/dt = p`, `dp/dt = −∇V(z)` | Symplectic Euler |
| `RecognitionKAN` | Variational: `q*(z\|o) ∝ p(o\|z)p(z)` | ELBO gradient |

**Key insight:** Allen-Cahn dynamics (`∂u/∂t = −δE/δu`) is gradient flow, not Hamiltonian
dynamics. The previous codebase incorrectly named this module `HamiltonianField`. The
correct name is `PredictiveCodingField` (or `GradientFlowField` as alias). Hamiltonian
dynamics are used *only* in `latent_hkan.py` and `latent_hkan_fep.py`.

### 2. Variational Free Energy

The variational free energy for an observation `o` and latent variable `z` is:

```
F = KL[q_φ(z|o) || p(z)]  −  E_{q_φ}[log p_θ(o|z)]
    \______complexity______/  \________accuracy_______/
```

Minimising F tightens the bound on log p(o) (evidence lower bound / ELBO).

This requires **two** networks:
- `RecognitionKAN`: encodes `o → (μ_z, log σ²_z)` — the recognition density `q_φ(z|o)`
- `GenerativeDecoder`: decodes `z → o_hat` — the likelihood `p_θ(o|z)`

The previous codebase had only energy minimisation with no recognition model, making full
FEP computation impossible. `recognition_model.py` is the highest-leverage addition.

### 3. Predictive Coding

Layer-wise prediction errors at layer `l`:

```
ε_l = Π_l^{1/2} · (x_l − KAN_l(x_{l+1}))
```

where `Π_l` is the precision matrix (learnable diagonal, initialised to 1.0).

State dynamics (inference on fast timescale γ):

```
∂x_l/∂t = −ε_l  +  (∂KAN_{l−1}/∂x_l)^T · ε_{l−1}
```

At equilibrium, this is mathematically equivalent to backpropagation (Song et al.,
NeurIPS 2020). The local Hebbian weight update is:

```
ΔW_l = η · ε_l · (∂KAN_l/∂W_l)
```

No global gradient signal is required — each layer only needs its own prediction error.

### 4. Hamiltonian Dynamics

For latent variable `z` with conjugate momentum `p`:

```
H(z, p) = ½‖p‖²  +  V_KAN(z)
           \_kinetic_/  \__potential__/
```

Symplectic Euler integration (volume-preserving, conserves shadow Hamiltonian):

```
p_new = p − dt · ∇_z V(z)
z_new = z + dt · p_new          ← uses updated p, not old p
```

**Warning:** Naive Euler (`z_new = z + dt·p_old`) does not conserve energy and leads to
unbounded drift. Always use the symplectic form.

The neuromorphic interpretation (Aitchison & Lengyel 2016):
- `z` = excitatory firing rates
- `p` = inhibitory activity
- E/I balance = symplectic structure

### 5. Eikonal Path Planning

The Eikonal equation governs shortest paths through a refractive medium:

```
‖∇T(x)‖ = 1 / n(x)
```

where `n(x)` is the refractive index and `T(x)` is the travel time from a source.

In TIM, `V_KAN(z)` defines the cost landscape:

```
n(z) = 1 / (α · |V_KAN(z)| + β)
```

High-energy regions (large V) have low refractive index — paths are deflected away from
energy barriers, towards low-energy channels (attractors). This gives goal-directed
planning without reinforcement learning.

**Computational note:** The Fast Sweeping Method solves the Eikonal equation in O(N)
operations for a grid of N points. This is *not* O(1) — it is O(K×N) where K is the
number of inference steps. The false claim of O(1) inference from earlier papers has been
retracted.

---

## Module Reference

### `src/recognition_model.py`

**Class:** `RecognitionKAN(obs_dim, latent_dim, hidden_dims, use_conv, grid_size)`

KAN-based encoder implementing the recognition density `q_φ(z|o)`.

```python
recognition = RecognitionKAN(
    obs_dim=784,          # flattened MNIST
    latent_dim=16,
    hidden_dims=[256, 128],
    use_conv=False,
    grid_size=5
)

mu, logvar = recognition.encode(o)          # (B, latent_dim) each
z, mu, logvar = recognition.sample(o)       # reparameterization
kl = recognition.kl_divergence(mu, logvar)  # scalar
```

The KAN architecture: `obs_dim → hidden_dims → 2*latent_dim`. The output is split to give
`(μ, log σ²)`. A learnable `log_precision` parameter scales the KL term.

**Class:** `GenerativeDecoder(latent_dim, obs_dim, hidden_dims, output_activation, grid_size)`

KAN-based decoder implementing `p_θ(o|z)`.

```python
decoder = GenerativeDecoder(
    latent_dim=16,
    obs_dim=784,
    hidden_dims=[128, 256],
    output_activation='sigmoid'
)

o_hat = decoder(z)                           # (B, obs_dim)
ll = decoder.log_likelihood(o, o_hat)        # Gaussian log-likelihood
```

**Function:** `variational_free_energy(o, recognition, generative, n_samples, beta)`

Computes the full variational free energy:

```python
result = variational_free_energy(o, recognition, decoder, n_samples=1, beta=1.0)
# result['F']     — total free energy (loss to minimise)
# result['kl']    — complexity term
# result['recon'] — accuracy term (negative log-likelihood)
# result['z']     — sampled latent
# result['o_hat'] — reconstruction
```

---

### `src/predictive_coding_field.py`

**Class:** `PredictiveCodingField(ThermodynamicField)` — alias: `GradientFlowField`

Implements Allen-Cahn gradient flow on spatial fields. Inherits spatial filter bank from
`ThermodynamicField`.

```python
field = PredictiveCodingField(
    n_channels=1,
    n_filters=16,
    filter_size=5,
    kan_hidden=[32, 16],
    height=28, width=28
)

# Denoising inference (K steps of gradient descent on E)
u_clean, info = field.run_inference(
    u_init=x_noisy,
    mu=None,           # no external prior mean
    n_steps=10,
    dt=0.1
)

# Compute KAN energy
E = field.compute_energy(u)  # (B,) scalar per sample

# Prediction error (with precision weighting)
err = field.prediction_error(u, mu)  # (B, C, H, W)

# Local Hebbian update (no backprop)
field.local_weight_update(u, mu, lr=0.001)

# Denoising score matching loss
loss = field.denoising_loss(x_clean, x_noisy)
```

**Mathematical identity (DSM):**
The loss `‖x_clean − (x_noisy − η∇E(x_noisy))‖²` is equivalent to Denoising Score
Matching (Vincent 2011), which is equivalent to computing the score `∇log p(x)`.

---

### `src/hierarchical_pc_kan.py`

**Class:** `HierarchicalPCKAN(dims, hidden_dims, grid_size, inference_lr, n_inference_steps)`

Multi-layer predictive coding hierarchy with local learning.

```python
model = HierarchicalPCKAN(
    dims=[784, 256, 64, 16],   # obs → hidden → hidden → top
    hidden_dims=[64],
    grid_size=5,
    inference_lr=0.1,
    n_inference_steps=20
)

# Full inference (fast timescale γ)
states, errors, info = model.inference(x_obs, n_steps=20, return_trajectory=True)
# states: list of (B, dim_l) tensors, one per layer
# errors: list of (B, dim_l) prediction errors
# info['F_trajectory']: free energy over inference steps

# Local weight update (medium timescale β, no backprop)
loss_dict = model.local_update(states, errors, lr=0.001)

# Scalar free energy (for monitoring)
F = model.free_energy(x_obs)
```

**Three-timescale learning:**

| Timescale | Variable | Update rule | Period |
|---|---|---|---|
| Fast γ | State `x_l` | PC inference loop | Every sample |
| Medium β | Precision `Π_l` | Local Hebbian | Every N_medium steps |
| Slow δ/θ | Weights `W_l` | Local Hebbian | Every N_slow steps |

This schedule is managed by `TimescaleScheduler` in `tim.py`.

---

### `src/latent_hkan_fep.py`

**Class:** `LatentHKAN_FEP`

Full FEP-compliant latent model combining recognition, Hamiltonian dynamics, and decoding.

```python
model = LatentHKAN_FEP(
    obs_dim=784,
    latent_dim=16,
    kan_hidden=[64, 32],
    enc_hidden=[256, 128],
    dec_hidden=[128, 256],
    ham_steps=5,
    dt=0.05,
    damping=0.0,          # set > 0 for dissipative dynamics
    grid_size=5
)

# Forward pass
out = model.forward(o, n_ham_steps=5)
# out['o_hat']  — reconstruction
# out['z']      — latent after Hamiltonian trajectory
# out['z0']     — initial latent (from recognition)
# out['mu']     — recognition mean
# out['logvar'] — recognition log-variance

# Training loss
loss = model.compute_loss(o, beta=1.0, n_ham_steps=5)
# loss['F']       — total variational free energy
# loss['kl']      — complexity
# loss['recon']   — accuracy
# loss['H_drift'] — Hamiltonian conservation error (diagnostic)

# Active inference (action selection)
action = model.select_action(o, candidate_actions)
```

**Hamiltonian trajectory:** Starting from `z0 ~ q_φ(z|o)`, the model runs `ham_steps`
steps of symplectic Euler integration under `H(z,p) = ½‖p‖² + V_KAN(z)`. This allows
the latent to explore the KAN energy landscape before decoding, improving reconstruction
at test-time when `ham_steps` is increased.

---

### `src/tim.py`

**Class:** `TrueIntelligenceModel`

Unified architecture integrating all four modules.

```python
model = TrueIntelligenceModel(
    obs_dim=784,
    latent_dim=16,
    pc_dims=[784, 256, 64, 16],  # for HierarchicalPCKAN
    ham_steps=5,
    use_eikonal=True,
    use_hierarchy=True,          # use HierarchicalPCKAN (vs flat PCField)
    n_medium=10,
    n_slow=100,
    grid_size=5
)

# Forward pass
out = model.forward(o, n_ham_steps=5, n_pc_steps=20)
# out contains keys from both LatentHKAN_FEP and HierarchicalPCKAN

# Combined loss (FEP + PC)
loss = model.compute_loss(o_clean, o_noisy, beta=1.0, n_ham_steps=5)
# total = FEP_loss + 0.5 * PC_loss

# Local update (no backprop through PC layers)
model.local_update(inference_result, lr=0.001)

# Path planning via Eikonal
path = model.plan_path(z_start, z_goal)

# Diagnostics
n_params = model.parameter_count()
E = model.energy_at(z)            # V_KAN(z)
```

**Shared potential:** The `kan_potential` in `LatentHKAN_FEP` is referenced by all modules
as the shared energy landscape. This is the mechanism by which V_KAN(z) simultaneously
serves as (A) PC attractor, (B) Hamiltonian potential, and (C) Eikonal cost.

---

## TIM Inference–Learning Cycle

```
Observation o
    │
    ▼
[RecognitionKAN]  ──────────────────────────────────────────►  μ_z, σ_z
    │  q_φ(z|o) — amortised inference (fast, feed-forward)
    ▼
[HierarchicalPCKAN]  ────────────────────────────────────►  x_1 … x_L
    │  PC inference on observation (fast timescale γ)
    │  ∂x_l/∂t = −ε_l + J_l^T ε_{l−1}
    ▼
[LatentHKAN_FEP]  ──────────────────────────────────────►  z trajectory
    │  Symplectic Euler in latent space (medium timescale β)
    │  H(z,p) = ½‖p‖² + V_KAN(z)
    ▼
[GenerativeDecoder]  ───────────────────────────────────►  o_hat
    │  p_θ(o|z) — reconstruction
    │
    ├── Compute F = KL − E_q[log p(o|z)]
    │
    └── Backprop through recognition + decoder
        + local Hebbian on PC layers (slow timescale δ/θ)
```

Weight updates follow the three-timescale schedule:
- **Fast (γ):** State updates in PC inference — every sample, no weight change
- **Medium (β):** Precision updates `Π_l` — every `n_medium` steps
- **Slow (δ/θ):** Weight updates `W_l` via local Hebbian — every `n_slow` steps

---

## Experiment Pipeline

### Paper 1: KAN-EBM Benchmarks

```bash
cd experiments
python run_experiments.py --benchmarks-only     # MNIST/FashionMNIST denoising
python run_experiments.py --scaling-only        # PSNR vs inference steps K
python run_experiments.py --interpret-only      # KAN spline visualisation
python run_experiments.py --ablation-only       # filter size / KAN depth ablation
```

Expected results (σ=0.2 noise):

| Model | PSNR (K=1) | PSNR (K=10) | Params |
|---|---|---|---|
| FFN Baseline | ~26 dB | ~26 dB (flat) | ~200K |
| MLP-EBM Baseline | ~26.5 dB | ~27.5 dB | ~200K |
| KAN-EBM (PCField) | ~27 dB | ~29+ dB | ~200K |

Key claim: PSNR improves monotonically with K for EBM models; FFN is flat (no test-time
compute scaling). This is the compute-quality tradeoff of energy-based inference.

### Paper 2: FEP Validation and Latent H-KAN

```bash
python run_experiments.py --fep-only            # 4 mathematical validation experiments
python run_experiments.py --latent-only         # train LatentHKAN_FEP on MNIST
python run_experiments.py --local-only          # local Hebbian vs backprop comparison
```

The four FEP validation experiments (`exp_fep_validation.py`):
1. `exp_chain_verification` — Verifies cosine similarity between Allen-Cahn grad, score
   function, and prediction error is ≥ 0.99 for Gaussian energy
2. `exp_free_energy_decomposition` — Tracks KL + recon breakdown over training epochs
3. `exp_hamiltonian_conservation` — Confirms symplectic < naive Euler H-drift by ≥10×
4. `exp_pc_convergence` — Confirms free energy decreases monotonically during inference

### Running All Experiments

```bash
python run_experiments.py                       # full TIM suite
python run_experiments.py --quick               # reduced epochs for CI / dev check
```

Results are saved to `outputs/` with timestamp. JSON summary at
`outputs/experiment_results_TIMESTAMP.json`.

---

## Terminology Corrections

The following table summarises corrected terminology relative to earlier papers and code:

| Incorrect term (old) | Correct term (new) | Reason |
|---|---|---|
| `HamiltonianField` | `PredictiveCodingField` | Allen-Cahn is gradient flow, not Hamiltonian |
| "Hamiltonian dynamics" (spatial) | "Allen-Cahn gradient flow" | H requires kinetic+potential; AC has only potential |
| "O(1) inference" | "O(K×N) inference, K controllable" | Fast Sweeping is O(N); K steps cost O(K×N) |
| "FEP equivalent" | "FEP-compliant with recognition model" | FEP requires q(z\|o); previously absent |
| "Unified theory" | "Three contributions to physics-inspired ML" | Each contribution is valid separately |
| "Hamiltonian Brain energy" | "Latent Hamiltonian with KAN potential" | The brain analogy is one interpretive frame |

---

## Interfaces Between Modules

### Observation → Latent

```python
# recognition_model.py
z, mu, logvar = recognition.sample(o)      # (B, latent_dim)
```

### Latent → Energy

```python
# shared potential V_KAN in latent_hkan_fep.py / tim.py
V = model.potential(z)                     # (B,)
grad_V = torch.autograd.grad(V.sum(), z)[0]  # (B, latent_dim)
```

### Latent → Reconstruction

```python
# recognition_model.py
o_hat = decoder(z)                         # (B, obs_dim)
```

### Observation → PC States

```python
# hierarchical_pc_kan.py
states, errors, info = model.inference(x_obs)
# states[l].shape = (B, dims[l])
```

### Latent → Path

```python
# tim.py
path = model.plan_path(z_start, z_goal)   # list of waypoints
```

---

## Neuromorphic Design Patterns

TIM is designed to map naturally to neuromorphic hardware:

**Synaptic plasticity:** The local Hebbian rule `ΔW_l = η · ε_l · ∂KAN_l/∂W_l` requires
only locally available information at each synapse — matching spike-timing dependent
plasticity (STDP) in biological neurons.

**Excitatory/inhibitory balance:** The Hamiltonian latent space has `z` (excitatory) and
`p` (inhibitory) variables. The symplectic structure corresponds to E/I balance maintaining
oscillatory dynamics (Aitchison & Lengyel 2016).

**Theta-gamma coupling:** Slow `δ/θ` weight updates nested inside fast `γ` state
oscillations mirrors hippocampal theta-gamma coupling. The three-timescale scheduler
`TimescaleScheduler(n_medium, n_slow)` implements this structure.

**KAN as dendritic computation:** The spline activations in KAN nodes can be interpreted
as dendritic nonlinearities, where each synapse has its own learned transfer function
rather than a shared activation. This is more biologically plausible than MLPs with
shared activation functions.

---

## File Structure

```
src/
├── kan.py                    — KAN implementation (B-spline, KANLinear, KAN class)
├── thermodynamic_field.py    — Base class with spatial filter bank, step_size
├── predictive_coding_field.py — Allen-Cahn / PC field (replaces HamiltonianField)
├── recognition_model.py      — RecognitionKAN + GenerativeDecoder + VFE function
├── hierarchical_pc_kan.py    — Multi-layer PC with local Hebbian learning
├── latent_hkan.py            — Latent H-KAN (legacy, no recognition model)
├── latent_hkan_fep.py        — Latent H-KAN with full FEP compliance  ← NEW
└── tim.py                    — TrueIntelligenceModel unified class      ← NEW

experiments/
├── run_experiments.py        — Master runner (Paper 1 + Paper 2 + TIM flags)
├── exp_benchmarks.py         — MNIST/FashionMNIST denoising vs baselines
├── exp_compute_scaling.py    — PSNR vs K, FLOPs, noise robustness
├── exp_fep_validation.py     — 4 mathematical validation experiments
├── exp_local_learning.py     — PC Hebbian vs backprop comparison
├── train_latent_hkan.py      — Full training pipeline for LatentHKAN_FEP
└── exp_hamiltonian_dream.py  — (legacy) Eikonal path planning demo

docs/
├── TIM_Architecture.md       — This file
├── theory_foundations.md     — JKO/Wasserstein foundations
├── phase1_explained.md       — Phase 1 results context
├── phase2_explained.md       — Phase 2 results context
└── phase3_explained.md       — Phase 3 results context
```

---

## References

- Jordan, Kinderlehrer, Otto (1998). "The variational formulation of the Fokker–Planck
  equation." SIAM Journal on Mathematical Analysis.
- Rao & Ballard (1999). "Predictive coding in the visual cortex." Nature Neuroscience.
- Vincent (2011). "A connection between score matching and denoising autoencoders." Neural
  Computation. — proves DSM ≡ score matching
- Bogacz (2017). "A tutorial on the free-energy framework for modelling perception and
  learning." Journal of Mathematical Psychology.
- Aitchison & Lengyel (2016). "The Hamiltonian Brain: Efficient probabilistic inference
  with excitatory-inhibitory neural circuit dynamics." PLOS Computational Biology.
- Song et al. (2020). "Can the brain do backpropagation? Exact implementation of
  backpropagation in predictive coding networks." NeurIPS 2020.
- Snell et al. (2024). "Scaling LLM test-time compute optimally is more effective than
  scaling model parameters." arXiv 2408.03314.
- Behrouz et al. (2025). "Titans: Learning to memorize at test time." NeurIPS 2025.
- Liu et al. (2024). "KAN: Kolmogorov–Arnold Networks." arXiv 2404.19756.
