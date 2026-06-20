# False Claim 3: Equivalence with the Free Energy Principle (Friston)

**Verdict:** ❌ Unsubstantiated — the FEP requires mathematical structures absent from this project
**Severity:** Medium — the analogy is seductive but incomplete; claiming equivalence is wrong
**Fixable:** Partially — remove the equivalence claim; a weaker "consistent with" framing may survive

---

## What the Project Claims

`docs/theory_foundations.md` and various documents claim that energy minimization in
this project is equivalent to, or an instantiation of, Friston's Free Energy Principle.
The argument is: both minimize "free energy," therefore they are the same framework.

---

## What the Free Energy Principle Actually Requires

Friston's FEP (Friston, 2010; Friston et al., 2017) is a formal framework with
specific mathematical components:

### Required components

1. **A generative model** p(x, o) over hidden states x and observations o
   - Must be explicit and differentiable
   - Specifies how the world generates observations
   - Absent from this project entirely

2. **A recognition/variational distribution** q(x | o; μ)
   - The brain's posterior belief over causes x given observation o
   - Parameterized by variational parameters μ
   - Absent from this project (there is no q anywhere)

3. **Variational Free Energy**, defined as:
   ```
   F = E_q[log q(x)] − E_q[log p(x, o)]
     = KL[q(x) || p(x|o)] − log p(o)
     ≥ −log p(o)
   ```
   This is the ELBO (Evidence Lower Bound) with a sign flip. It requires both
   a generative model AND a recognition model.

4. **Active inference** — the agent takes actions to also minimize F, not just
   update beliefs. This requires an action space and policy.

5. **Markov blanket** — a statistical boundary separating internal states (brain)
   from external states (world) via sensory and active states

### What this project actually does

This project minimizes E(x) → ℝ, a scalar energy function. This overlaps with
FEP only in the sense that:

- Energy minimization ≈ MAP inference under p(x) ∝ exp(−E(x))
- This corresponds to the special case of FEP where: q(x) is a delta function
  (deterministic), and there is no generative model for observations

This special case strips out essentially all of what makes FEP interesting:
no uncertainty, no generative model, no recognition model, no active inference.

### The "energy" terminology overlap is misleading

The word "free energy" appears in both FEP and statistical physics, but they are
different quantities:

| Context | "Free Energy" meaning |
|---------|----------------------|
| Statistical physics | F = U − TS (Helmholtz free energy) |
| Friston's FEP | F = KL[q||p] − log p(o) (variational free energy / ELBO) |
| This project | E(x) (scalar energy function with no statistical interpretation) |

These three quantities are not the same. Conflating them gives the impression of
theoretical depth that is not actually present.

---

## The Specific Mathematical Gaps

To genuinely implement FEP, the project would need:

1. **An explicit generative model** p(x_clean, x_noisy):
   ```python
   # Not present anywhere in the codebase
   def generative_model(x_clean):
       return x_clean + noise   # Example: trivial Gaussian noise model
   ```

2. **A recognition model** q(x_clean | x_noisy):
   ```python
   # Also not present — the model directly minimizes E(x) without a posterior
   def recognition_model(x_noisy):
       return Normal(mu(x_noisy), sigma(x_noisy))  # Would need this
   ```

3. **The ELBO computation:**
   ```python
   F = KL_divergence(q, prior) - E_q[log_likelihood]
   # None of this computation occurs anywhere in the codebase
   ```

---

## What Is Salvageable

A weaker, honest claim is defensible:

> "Our energy-based inference is consistent with the predictive coding formulation
> of perception (Rao & Ballard, 1999), where inference corresponds to minimizing
> prediction error. When the energy E(x) is interpreted as prediction error between
> the observed state and the model's expectation, gradient-flow inference is
> equivalent to hierarchical predictive coding."

This connection to **predictive coding** (a specific, mechanistic implementation
of FEP) is defensible because:
- Predictive coding minimizes prediction error E(x) at each level of a hierarchy
- The gradient flow ∂u/∂t = −∇E is exactly the prediction error minimization step
- This does NOT require an explicit generative model in the FEP sense

**Use this instead of "FEP equivalence."**

---

## Why This Matters for Publication

The active inference / FEP community is small but vocal. If this paper is submitted
to a venue where Friston's work is taken seriously (e.g., PLOS Computational Biology,
Entropy, or neuroscience-adjacent ML venues), reviewers will:

1. Ask for the explicit generative model p(x, o)
2. Ask for the recognition model q(x | o)
3. Ask for the Markov blanket structure
4. Notice that none of these exist

The paper will be rejected on these grounds, or forced to substantially rewrite the
FEP framing.

The safer path: cite FEP as motivation and inspiration, but do not claim equivalence.
Claim "consistency with predictive coding" instead — a much more defensible position.

---

## What to Cite Instead

- Rao, R. & Ballard, D. (1999). *Predictive coding in the visual cortex.* Nature Neuroscience
  (predictive coding — more specific than FEP, directly maps to gradient flow)
- Clark, A. (2013). *Whatever next? Predictive brains, situated agents.* Brain & Behavioral Sciences
- Bogacz, R. (2017). *A tutorial on the free-energy framework for modelling perception and learning.*
  Journal of Mathematical Psychology (the most accessible FEP math tutorial)
