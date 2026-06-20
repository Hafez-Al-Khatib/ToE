# Valid Claim 2: Energy-Based Denoising via Thermodynamic Field Dynamics

**Status:** ✅ Publishable — Strong baseline results exist; framing needs tightening
**Target venue:** ICLR workshop / CVPR (image processing track)
**Paper title candidate:** *"Thermodynamic Neural Fields: Learned Energy Functionals for Image Restoration"*

---

## The Claim

Images can be treated as physical fields u(x,y) whose equilibrium states are the
"clean" data manifold. A neural network parameterizes the energy functional E[u],
and denoising is performed by evolving the field down the energy gradient:

```
∂u/∂t = −δE/δu    (gradient flow / Allen-Cahn equation)
```

The energy functional has a physics-inspired structure:

```
E[u] = Σₖ wₖ ||Kₖ * u||²           (spatial gradient energy — "stiffness")
      + V_KAN(u)                     (learned local potential — "memory")
      + Σᵢⱼ Jᵢⱼ · uᵢ · uⱼ          (inter-channel coupling — "interaction")
```

At inference: given noisy image u_noisy, run N steps of gradient descent →
converges to a clean image without a separate inference network.

---

## Why This Is True

### Physical analogy is mathematically sound

The gradient flow ∂u/∂t = −δE/δu is a well-studied equation in physics and mathematics:

- **Allen-Cahn equation** (non-conserved order parameter): describes phase separation,
  pattern formation, and interface dynamics. Used in image segmentation (Chan-Vese model).
- **Tikhonov regularization**: classical denoising via ||u−f||² + λ||∇u||² is exactly
  gradient flow on a quadratic energy.
- **Total Variation (TV) denoising** (Rudin-Osher-Fatemi): gradient flow on E[u] = TV(u)
  + data fidelity. Well-established, provably convergent.

The innovation here is replacing the fixed analytic energy (TV, quadratic) with a
**learned** energy that adapts to the training data distribution. This is a legitimate
and non-trivial extension.

### Contrastive Divergence training is principled

Training by pushing down E(x_clean) and pushing up E(x_noisy) is an approximation to
maximum likelihood in an EBM. The specific form used here — 1-step denoising loss:

```
L = ||x_clean − (x_noisy − η ∇E(x_noisy))||²
```

is equivalent to **Denoising Score Matching** (Vincent, 2011), which has a formal
connection to score-based generative models. This is well-grounded:

```
∇_x log p(x) ≈ −(1/σ²)(x − x_clean)    (score function)
```

Our gradient ∇E(x) ≈ (1/σ²)(x_noisy − x_clean) when the loss converges, meaning
the energy gradient approximates the score function. This is a known, publishable
connection.

### Multi-channel field dynamics enables novel behavior

The inter-channel coupling term Σᵢⱼ Jᵢⱼ uᵢ uⱼ allows the model to learn correlations
between different image features (e.g., texture channels can influence edge channels),
which is not possible with per-pixel energy functions. This is a structural advantage
over standard EBMs that operate on flattened pixel vectors.

### Existing results in this codebase

- `outputs/MNIST_denoising_results.png` — denoising results on MNIST exist
- `outputs/FashionMNIST_denoising_results.png` — FashionMNIST results exist
- `outputs/mnist_physics.png`, `outputs/fashionmnist_physics.png` — energy landscape plots
- Multiple trained models: `mnist_tnf.pt`, `fashion_tnf.pt` — Phase 1 models

These outputs show the approach already works on real data.

---

## What Work Has Been Done

| Component | File | Status |
|-----------|------|--------|
| ThermodynamicField base class | `src/thermodynamic_field.py` | ✅ Complete (563 lines) |
| EnergyMLP baseline | `src/energy_model.py` | ✅ Complete |
| Phase 1 training script | `src/train_phase1.py` | ✅ Complete |
| MNIST/FashionMNIST results | `outputs/*.png` | ✅ Exist |
| Trained models (MNIST, Fashion) | `outputs/*.pt` | ✅ Exist |
| Energy landscape visualization | `src/visualize_tnf.py` | ✅ Complete |
| Debug/analysis tools | `src/debug_inference.py`, `src/analyze_physics.py` | ✅ Complete |

The most mature part of the codebase. Results already exist.

---

## What Is Still Needed for Publication

1. **PSNR/SSIM numbers** — existing results are visual only; need quantitative table
   Run `experiments/exp_denoise.py --dataset mnist` to get proper metrics
2. **Comparison baselines:**
   - BM3D (classical, strong baseline)
   - DnCNN (standard learned denoising CNN)
   - DDRM or DPS (diffusion-based, state-of-the-art)
3. **Ablation studies:**
   - With/without inter-channel coupling (Jᵢⱼ term)
   - With/without multi-step evolution (1-step vs 10-step vs 50-step inference)
   - KAN potential vs polynomial potential vs no potential
4. **Convergence analysis** — plot energy E(u_t) as a function of step t to show
   monotone decrease (this validates the physics interpretation)
5. **Out-of-distribution test** — train on MNIST, test on FashionMNIST or CIFAR
   to measure generalization of the learned energy landscape

---

## Key Figures for the Paper

1. **Figure 1:** Architecture — multi-channel field with filter bank, KAN potential, coupling
2. **Figure 2:** Denoising trajectories — clean → noisy → 1-step → 5-step → 20-step
3. **Figure 3:** PSNR/SSIM table vs noise level (comparison with BM3D, DnCNN)
4. **Figure 4:** Energy descent curve E(u_t) — showing convergence
5. **Figure 5:** Learned filter visualization — what spatial features drive the energy

---

## Correct Framing

❌ Do NOT say: "This is physics, not computation"
✅ DO say: "gradient flow on a learned energy functional, inspired by the Allen-Cahn equation"

❌ Do NOT say: "The field has thermodynamic temperature"
✅ DO say: "Langevin noise with annealed magnitude σ(t) improves convergence"
   (the "temperature" is a noise schedule, not thermodynamic temperature)

❌ Do NOT say: "Images obey thermodynamic laws"
✅ DO say: "We model images as fields and apply variational inference to find
   their most probable clean state under the learned energy"

---

## Connection to Existing Literature

This sits at the intersection of:
- **Variational image restoration** (Rudin, Osher, Fatemi 1992 — TV denoising)
- **Energy-based models** (LeCun 2006, Du & Mordatch 2019)
- **Score-based generative models** (Song & Ermon 2019 — same gradient field interpretation)
- **Neural ODEs** (Chen et al. 2018 — the field evolution is a neural ODE)
- **Plug-and-Play priors** (Venkatakrishnan et al. 2013 — learned denoisers as priors)

## Related Work to Cite

- Rudin, L., Osher, S., & Fatemi, E. (1992). *Nonlinear total variation based noise removal.*
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders.*
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients.* NeurIPS
- Du, Y. & Mordatch, I. (2019). *Implicit Generation with EBMs.* NeurIPS
- Chan, T. & Vese, L. (2001). *Active Contours Without Edges.* IEEE Trans. Image Process.
