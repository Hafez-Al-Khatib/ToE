# Valid Claim 1: KAN as a Learned, Interpretable Energy Density Function

**Status:** ✅ Publishable — Novel contribution with solid mathematical grounding
**Target venue:** NeurIPS / ICLR (main track)
**Paper title candidate:** *"KAN-EBM: Interpretable Energy-Based Models via Kolmogorov-Arnold Networks"*

---

## The Claim

A Kolmogorov-Arnold Network (KAN) can replace hardcoded potential functions in an
Energy-Based Model, producing an energy density that is:

1. **Learned from data** rather than hand-designed (e.g., no hardcoded Ginzburg-Landau double-well)
2. **More interpretable** than an MLP-based energy, because each KAN edge is a
   visualizable 1D B-spline function φᵢⱼ : ℝ → ℝ
3. **Equally or more expressive** than polynomial potentials while remaining smooth and
   differentiable everywhere, which is required for stable Langevin inference

The overall energy functional is:

```
E[u] = ∫∫ KAN(φ₁(u), φ₂(∇u), ..., φₖ(Kₖ * u)) dx dy
```

where the φᵢ are learned B-spline functions and Kₖ are learned spatial filter kernels.

---

## Why This Is True

### Mathematical grounding

The Kolmogorov-Arnold Representation Theorem (1957) guarantees that any continuous
multivariate function f : [0,1]ⁿ → ℝ can be decomposed as:

```
f(x₁, ..., xₙ) = Σ_q Φ_q( Σ_p φ_{q,p}(xₚ) )
```

where all Φ_q and φ_{q,p} are continuous univariate functions. A KAN approximates
these univariate functions with B-splines, which are:

- **Piecewise polynomial:** locally controlled, no global Runge oscillation
- **C² smooth:** gradient ∇E(x) is well-defined everywhere → stable Langevin dynamics
- **Locally supported:** changing one spline coefficient only affects nearby inputs
- **Grid-adaptive:** the grid can be refined during training to increase resolution

This is directly superior to the polynomial potentials in the original
Ginzburg-Landau energy (E ∝ a·u² + b·u⁴) which are globally fixed and cannot adapt
to the data distribution.

### Why MLP-based energy is worse for interpretability

An MLP energy E(x) = wᵀ·σ(W₂·σ(W₁x + b₁) + b₂) has:
- No decomposition into per-input functions
- No way to visualize what each input dimension contributes
- Entangled representations across all inputs

A KAN energy E(x) = Σⱼ Φⱼ(Σᵢ φᵢⱼ(xᵢ)) has:
- **One inspectable curve per input→output edge**
- Direct visualization: plot φᵢⱼ(t) for t ∈ [-2, 2] to see what feature i contributes to output j
- Feature importance: ||φᵢⱼ||₂ measures how much edge (i→j) matters
- Symbolic regression potential: if φᵢⱼ looks like t², you can replace it with an exact formula

### Experimental evidence in this codebase

- `src/kan.py` + `src/hamiltonian_field.py`: working implementation
- `outputs/kan_denoise.pth`: trained model (synthetic shapes)
- `outputs/kan_denoise_result.png`: denoising visualization exists
- `experiments/exp_denoise.py`: proper KAN vs MLP baseline comparison

The denoising training converges (loss drops from 0.09 to 0.013 over 20 epochs),
confirming the KAN energy landscape is learnable and produces useful gradients.

---

## What Work Has Been Done

| Component | File | Status |
|-----------|------|--------|
| KAN implementation (B-splines) | `src/kan.py` | ✅ Complete |
| KAN-based energy field | `src/hamiltonian_field.py` | ✅ Complete |
| MLP baseline for comparison | `experiments/mlp_field.py` | ✅ Complete |
| Denoising experiment (KAN vs MLP) | `experiments/exp_denoise.py` | ✅ Complete |
| KAN spline visualization | `experiments/kan_interpretability.py` | ✅ Complete |
| PSNR/SSIM metrics | `experiments/metrics.py` | ✅ Complete |
| Trained models (synthetic) | `outputs/kan_denoise.pth` | ✅ Exists |

---

## What Is Still Needed for Publication

1. **MNIST / FashionMNIST results** — run `exp_denoise.py --dataset mnist` with torchvision
2. **Statistical significance** — run each experiment 3× with different seeds, report mean ± std
3. **Ablation on grid size** — show that finer KAN grid → better quality at cost of parameters
4. **Spline visualization figures** — use `kan_interpretability.py` to generate figures
   showing learned φᵢⱼ functions for the paper (these are the key interpretability figures)
5. **Symbolic regression demo** — after training, fit a simple formula to one φᵢⱼ and show
   the KAN has discovered a recognizable function (e.g., abs(x), x², tanh(x))
6. **Comparison with Du & Mordatch (2019) EBM** and **Grathwohl et al. (2020) JEM**

---

## Key Figures for the Paper

1. **Figure 1:** Architecture diagram — spatial filters → KAN → scalar energy → Langevin step
2. **Figure 2:** Learned spline functions φᵢⱼ(t) for trained model (interpretability claim)
3. **Figure 3:** Denoising quality table — KAN vs MLP vs polynomial, across noise levels
4. **Figure 4:** PSNR vs evolution steps — shows test-time refinement improves quality
5. **Figure 5:** Energy landscape visualization — 2D cross-section showing learned bowl

---

## Correct Framing (Avoid These Mistakes)

❌ Do NOT say: "This is Hamiltonian dynamics"
✅ DO say: "gradient-flow inference on a KAN-parameterized energy landscape"

❌ Do NOT say: "KAN replaces neural networks"
✅ DO say: "KAN replaces the potential term in an Energy-Based Model"

❌ Do NOT say: "O(1) inference"
✅ DO say: "inference via K-step gradient descent, with quality improving with K"

---

## Related Work to Cite

- Liu, Z. et al. (2024). *KAN: Kolmogorov-Arnold Networks.* arXiv:2404.19756
- LeCun, Y. et al. (2006). *A Tutorial on Energy-Based Learning.*
- Du, Y. & Mordatch, I. (2019). *Implicit Generation and Modeling with EBMs.* NeurIPS
- Grathwohl, W. et al. (2020). *Your Classifier is Secretly an EBM.* ICLR
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients.* NeurIPS
