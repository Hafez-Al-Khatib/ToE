# False Claim 1: "Hamiltonian Dynamics" for the Core Inference Procedure

**Verdict:** ❌ Incorrect terminology — the core model uses gradient flow, not Hamiltonian dynamics
**Severity:** High — will cause immediate rejection if submitted to physics-aware venues
**Fixable:** Yes — rename and reframe without changing any code

---

## What the Project Claims

The file `src/hamiltonian_field.py` and all associated documentation call the inference
procedure "Hamiltonian dynamics." The `hamiltonian_kan_concepts.md` document describes
the model as implementing a "Hamiltonian field" where energy is minimized via physics.

The implicit claim is that the system evolves like a physical Hamiltonian system,
conserving energy and following Hamilton's equations.

---

## Why This Is Wrong

### What Hamiltonian dynamics actually requires

A Hamiltonian system has:
- **Phase space:** pairs (q, p) of position q and momentum p
- **Hamiltonian function:** H(q, p) = T(p) + V(q) = kinetic + potential energy
- **Hamilton's equations:**
  ```
  dq/dt = +∂H/∂p    (velocity = gradient of H w.r.t. momentum)
  dp/dt = −∂H/∂q    (force = −gradient of H w.r.t. position)
  ```
- **Energy conservation:** H(q(t), p(t)) = H(q(0), p(0)) for all t
- **Symplectic structure:** the flow preserves volume in phase space (Liouville's theorem)

### What `hamiltonian_field.py` actually does

The `evolve()` method in `hamiltonian_field.py`:

```python
# From hamiltonian_field.py (actual code):
energy = self.compute_energy(u)
grad = torch.autograd.grad(energy.sum(), u)[0]
u_new = u - self.step_size * grad    # ← THIS IS NOT HAMILTON'S EQUATIONS
```

This is **gradient descent** (equivalently: overdamped Langevin dynamics without noise):

```
u_{t+1} = u_t − η ∇_u E(u_t)
```

In physical terms, this is the **overdamped limit** of Newton's equation, which
is what you get when friction dominates inertia. It is also called:
- **Gradient flow** (mathematics)
- **Steepest descent** (optimization)
- **Allen-Cahn equation** (physics, non-conserved order parameter)
- **Model A dynamics** (condensed matter physics)

This is a well-studied and respectable equation. But it is NOT Hamiltonian dynamics.

### The fundamental difference

| Property | Hamiltonian Dynamics | Gradient Flow (what the code does) |
|----------|----------------------|-------------------------------------|
| Variables | (q, p) — position AND momentum | u — only position |
| Evolution | Hamilton's equations | u_{t+1} = u_t − η∇E(u_t) |
| Energy | **Conserved** — H is constant | **Decreasing** — E strictly decreases |
| Attractor | Orbits around minimum | Fixed point at minimum |
| Physical analogy | Frictionless pendulum | Overdamped pendulum in viscous fluid |
| Time-reversibility | Yes — reversible | No — irreversible |
| Phase space | 2N dimensional | N dimensional |

The code **dissipates** energy to zero. Hamiltonian dynamics **conserves** energy.
They are opposite behaviors.

---

## Where in the Codebase the Error Appears

| File | Incorrect usage |
|------|----------------|
| `src/hamiltonian_field.py` | Class name, all docstrings |
| `src/train_kan_push.py` | "Hamiltonian KAN" in comments |
| `src/train_kan_sudoku.py` | "HamiltonianField" import and use |
| `hamiltonian_kan_concepts.md` | Entire framing |
| `docs/phase2_explained.md` | Phase 2 documentation |
| `tutorials/05_Hamiltonian_Mechanics.pptx` | Tutorial slide deck |

---

## The One Place "Hamiltonian" IS Correct

`src/latent_hkan.py:latent_dynamics_step` correctly implements Hamiltonian dynamics:

```python
p_new = p - dt * dV_dz    # dp/dt = -∂V/∂z ✅ Hamilton's equation
z_new = z + dt * p_new    # dz/dt = +p      ✅ Hamilton's equation
H = 0.5 * p^2 + V(z)      # conserved quantity ✅
```

This is the only Hamiltonian structure in the entire project. It is in `latent_hkan.py`,
not in `hamiltonian_field.py`. The naming is backwards.

---

## The Fix

**Rename `HamiltonianField` → `GradientFlowField`** (or `VariationalField` or `EnergyField`)

No code logic needs to change. Only names and docstrings.

Updated correct framing:
```
"We parameterize the energy functional E[u] with a KAN and perform inference via
gradient flow: u_{t+1} = u_t − η∇E(u_t). This is equivalent to the Allen-Cahn
equation for a non-conserved order parameter, where the field evolves toward the
nearest energy minimum."
```

**Reserve "Hamiltonian" for `latent_hkan.py`**, which correctly implements Hamilton's
equations in the latent space (z, p).

---

## Why This Matters

If you submit to a venue with physics reviewers (NeurIPS, ICLR, ICML):

> Reviewer: "The authors claim Hamiltonian dynamics but the inference procedure is
> gradient descent, which dissipates energy rather than conserving it. This is a
> fundamental error that undermines the paper's theoretical claims."

This will cause rejection. Fixing the name costs zero research effort.
