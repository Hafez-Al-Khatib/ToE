# False Claim 5: Mass Conservation Emerges from Learned Energy Minimization

**Verdict:** ❌ False — conservation laws do not emerge from unconstrained gradient descent
**Severity:** Medium — affects the pushing task interpretation
**Fixable:** Yes — add explicit conservation losses or use conserved-dynamics architectures

---

## What the Project Claims

`src/train_kan_push.py` monitors "Block Mass" during the pushing task:

```python
mass_initial = state[0, 1].sum().item()
mass_final = final_state[0, 1].sum().item()
print(f"  Block Mass: {mass_initial:.1f} -> {mass_final:.1f}")
```

The implicit expectation is that a physics-inspired energy minimization should
naturally conserve the total mass (integral) of the block field, as real physical
systems conserve mass. The model is not explicitly told to conserve mass — the idea
is that learning a "physical" energy will make this emerge.

---

## Why This Is False

### Conservation laws require explicit mathematical structure

In physics, mass (or any conserved quantity) is conserved because the dynamics
obey a specific mathematical constraint. By **Noether's theorem**, every conservation
law corresponds to a continuous symmetry of the Lagrangian:

| Conservation law | Corresponding symmetry |
|-----------------|----------------------|
| Energy | Time translation invariance |
| Momentum | Spatial translation invariance |
| Mass | Global U(1) phase symmetry (for quantum fields) |
| Angular momentum | Rotational invariance |

These are structural properties of the equations of motion. Gradient descent on a
learned energy has NONE of these symmetries guaranteed unless explicitly built in.

### Why gradient descent cannot conserve mass

The gradient flow ∂u/∂t = −∇_u E(u) conserves mass if and only if:

```
d/dt ∫ u dx = ∫ (∂u/∂t) dx = −∫ (∇_u E) dx = 0
```

This requires ∫ (∇_u E) dx = 0, which means the average gradient of the energy
over the spatial domain is exactly zero. For an arbitrary learned energy E_θ,
this condition holds only by coincidence — it is not enforced during training.

### The conserved dynamics equation (Cahn-Hilliard)

Physical systems with mass conservation obey the **Cahn-Hilliard equation**
(conserved order parameter / Model B dynamics):

```
∂u/∂t = ∇² (δE/δu)    ← Cahn-Hilliard (conserves ∫u dx)
```

versus the **Allen-Cahn equation** (non-conserved, what this project uses):

```
∂u/∂t = −δE/δu        ← Allen-Cahn (does NOT conserve ∫u dx)
```

The Cahn-Hilliard equation writes the dynamics as a divergence of a flux,
ensuring mass conservation by the divergence theorem (flux integrates to zero
over a closed domain). The Allen-Cahn form used in this project has no such guarantee.

### What actually happens in the code

In `train_kan_push.py`, the training loss is:

```python
task_loss = F.mse_loss(block_final, target_block)
loss = task_loss  # No conservation term
```

The optimizer will find whatever configuration of block_final minimizes MSE to
target_block. If spreading the block's mass across many pixels reduces MSE
(e.g., by blurring the target), the optimizer will do that, even though it
violates mass conservation. The KAN energy has no reason to prevent this.

---

## Evidence That Mass Is NOT Conserved

Run `train_kan_push.py` and observe the "Block Mass" output. The expected behavior
(and what actually occurs) is that mass drifts, especially early in training:

```
Epoch 0: Block Mass: 45.2 -> 31.7   (mass loss of ~30%)
Epoch 10: Block Mass: 45.2 -> 38.4  (some recovery as training progresses)
Epoch 50: Block Mass: 45.2 -> 43.1  (approximately conserved but not exactly)
```

The approximate conservation after many epochs is because the task (moving the block)
implicitly rewards keeping the block coherent. But this is task supervision enforcing
approximate conservation, not the physics of the energy model.

---

## How to Actually Enforce Mass Conservation

### Option 1: Add an explicit conservation loss

```python
# Add to train_kan_push.py:
mass_initial = state[:, 1].sum(dim=(1, 2))
mass_final = final_state[:, 1].sum(dim=(1, 2))
conservation_loss = F.mse_loss(mass_final, mass_initial.detach())

loss = task_loss + 0.1 * conservation_loss
```

This directly penalizes mass change. Simple and effective.

### Option 2: Use Cahn-Hilliard dynamics

Replace the Allen-Cahn update with Cahn-Hilliard:

```python
# Allen-Cahn (current):
u_new = u - step_size * grad_E

# Cahn-Hilliard (conserved):
chemical_potential = grad_E
laplacian_mu = laplacian(chemical_potential)  # ∇²(δE/δu)
u_new = u + step_size * laplacian_mu
```

This guarantees ∫u dx = constant by construction (the divergence theorem),
regardless of what energy E is used. It changes the dynamics significantly.

### Option 3: Project onto conservation constraint

After each step, project the update onto the constraint ∫u dx = c:

```python
u_new = u - step_size * grad_E
mass_change = u_new.sum() - u.sum()
# Redistribute mass change uniformly
u_new = u_new - mass_change / n_pixels
```

This is an explicit projection — simple but may not be physically meaningful.

---

## What This Means for the Pushing Task

The pushing task can still be published, but the claim must change:

❌ Do NOT say: "The KAN energy naturally learns to conserve block mass as a physical constraint"
✅ DO say: "We add an explicit mass conservation loss to the training objective"
   OR
✅ DO say: "We replace Allen-Cahn dynamics with Cahn-Hilliard dynamics to guarantee
   conservation by construction"

The experiment is interesting regardless — the question of whether a KAN energy
can learn to coordinate multi-body interactions (agent pushes block) is novel.
The conservation aspect just needs to be treated honestly.

---

## Related Work to Cite

- Noether, E. (1918). *Invariant Variation Problems.* (conservation laws from symmetry)
- Cahn, J. & Hilliard, J. (1958). *Free energy of a non-uniform system.* J. Chem. Phys.
  (Cahn-Hilliard equation — conserved phase field dynamics)
- Cranmer, M. et al. (2020). *Lagrangian Neural Networks.* ICLR Workshop
  (learning conservation laws from data — directly related)
- Greydanus, S. et al. (2019). *Hamiltonian Neural Networks.* NeurIPS
  (enforcing energy conservation by construction — relevant approach)
