# Valid Claim 3: Latent-Space KAN Energy (Perception–Reasoning Separation)

**Status:** ✅ Publishable — Clean architecture with strong theoretical justification
**Target venue:** NeurIPS / UAI (reasoning / representation learning)
**Paper title candidate:** *"Separating Perception from Reasoning: Latent Hamiltonian-KAN Networks"*

---

## The Claim

Applying KAN directly to high-dimensional inputs (e.g., 784-dim MNIST pixels) causes
parameter explosion: a single KAN layer needs input_dim × output_dim × (grid_size + order)
parameters, which for MNIST gives ~4.4M parameters just for the energy function.

The solution is a two-stage architecture:

```
x ∈ ℝ^784  →[CNN/MLP Encoder]→  z ∈ ℝ^16  →[KAN Energy]→  E(z) ∈ ℝ
```

This separation is not just a computational trick — it is a principled distinction
between two cognitive functions:

1. **Encoder (perception):** maps messy, high-dimensional pixel space to a compact
   latent manifold. This is what the visual cortex does. Standard CNN/MLP is appropriate.

2. **KAN Energy (reasoning):** operates on the clean, low-dimensional latent space.
   This is where interpretable, structured reasoning happens. KAN is appropriate here
   because the functions it learns over z are directly interpretable.

---

## Why This Is True

### The parameter efficiency argument is provable

Direct KAN on pixels (784 → 512 → 1):
```
Parameters = 784 × 512 × (grid_size + order + 1) ≈ 784 × 512 × 12 ≈ 4.8M
```

Latent KAN (16 → 32 → 1):
```
Parameters = 16 × 32 × 12 + 32 × 1 × 12 ≈ 6,144 + 384 ≈ 6,500
```

That is ~740× fewer parameters in the KAN layer, while the encoder handles perception
efficiently with standard convolutional layers. This ratio is computable, exact, and
constitutes a real contribution.

### The biological analogy has structural support

The claim is not just metaphorical. In neuroscience:
- The visual hierarchy (V1 → V2 → V4 → IT) performs hierarchical feature extraction
  (equivalent to the encoder), reducing 10⁶ photoreceptor inputs to ~200 conceptual
  object features in inferotemporal cortex
- The prefrontal cortex operates on these compact representations to make decisions
  (equivalent to the KAN energy operating on latent z)
- The known dissociation between "what" (ventral stream = perception) and "where/how"
  (dorsal stream = action/reasoning) maps directly to Encoder → KAN Energy

This doesn't prove the architecture is correct, but it means the paper can engage
with neuroscience literature in a principled way, not just analogically.

### KAN interpretability is most meaningful in low-dimensional latent space

A KAN operating on 16-dimensional z can be fully visualized:
- 16 × 32 = 512 edges in the first layer, each a plotable 1D spline
- The splines over z dimensions represent what properties of the learned latent
  representation drive energy up or down
- This is far more interpretable than splines over raw pixels, which have no semantic meaning

---

## What Work Has Been Done

| Component | File | Status |
|-----------|------|--------|
| LatentHKAN architecture | `src/latent_hkan.py` | ✅ Complete (426 lines) |
| ConvEncoder + MLPEncoder | `src/latent_hkan.py` | ✅ Complete |
| KANNetwork (energy function) | `src/kan_layer.py` | ✅ Complete |
| Latent Hamiltonian dynamics | `src/latent_hkan.py:latent_dynamics_step` | ✅ Complete |
| Self-test script | `src/latent_hkan.py:__main__` | ✅ Complete |

The architecture exists and self-tests pass. It uses `kan_layer.py` (the older KAN
implementation) rather than `kan.py`. These should be unified.

---

## What Is Still Needed for Publication

1. **Training script** — no `train_latent_hkan.py` exists yet. Need to write experiments for:
   - Denoising: encode noisy x → z_noisy → minimize E(z) → decode → x_clean
   - Generation: sample z from E(z) via Langevin → decode → x
   - Reconstruction: compare encode-denoise-decode vs end-to-end denoising

2. **Unify KAN implementations** — `src/kan.py` and `src/kan_layer.py` are two
   separate KAN implementations. Consolidate to one (prefer `kan.py` which is cleaner).

3. **Energy conservation test** — the latent dynamics step uses symplectic Euler,
   which should approximately conserve H(z,p) = T(p) + V(z). Plot H(z_t, p_t) vs t
   to verify and show in the paper. (NOTE: this is the ONE place in the codebase
   where "Hamiltonian" is used correctly — with both z and p.)

4. **Latent manifold visualization** — use UMAP or t-SNE to show that the latent space
   z has structure (e.g., MNIST digits cluster), validating the encoder is doing
   meaningful compression.

5. **Reconstruction quality** — PSNR/SSIM for encode→denoise→decode vs other methods.

---

## Important Distinction: This is the Only True "Hamiltonian" in the Project

The rest of the project misuses the word "Hamiltonian" to mean "gradient flow."
This module is the ONLY place where a true Hamiltonian is implemented:

```python
def latent_hamiltonian(self, z, p):
    T = 0.5 * (p ** 2).sum(dim=-1)    # Kinetic energy
    V = self.energy(z)                  # Potential energy (KAN)
    return T + V                        # True Hamiltonian H(z, p)

def latent_dynamics_step(self, z, p, dt):
    dV_dz = grad(V, z)
    p_new = p - dt * dV_dz             # ṗ = -∂H/∂z (correct)
    z_new = z + dt * p_new             # ż = +∂H/∂p (correct)
    return z_new, p_new                # Symplectic Euler integrator
```

This IS Hamilton's equations. This module should be the one named "Hamiltonian."
The rename from `HamiltonianField` → `GradientFlowField` everywhere else would
reserve the word "Hamiltonian" for this architecturally correct usage.

---

## Key Figures for the Paper

1. **Figure 1:** Architecture diagram — encoder (CNN) + KAN energy + latent dynamics
2. **Figure 2:** Parameter count comparison — direct KAN vs Latent KAN (bar chart)
3. **Figure 3:** Learned latent energy landscape — 2D projection of E(z) over latent space
4. **Figure 4:** Hamiltonian conservation — H(z_t, p_t) vs t (should be nearly constant)
5. **Figure 5:** Reconstruction quality — noisy input → latent denoising → clean output

---

## Related Work to Cite

- Liu, Z. et al. (2024). *KAN: Kolmogorov-Arnold Networks.* arXiv:2404.19756
- Kingma, D. & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR
- Rezende, D. et al. (2014). *Stochastic Backpropagation and Approximate Inference.* ICML
- Cranmer, M. et al. (2020). *Discovering Symbolic Models from Deep Learning.* NeurIPS
  (uses symbolic regression on latent space — directly relevant)
- Greydanus, S. et al. (2019). *Hamiltonian Neural Networks.* NeurIPS
  (true Hamiltonian dynamics learned by neural networks — closest related work)
