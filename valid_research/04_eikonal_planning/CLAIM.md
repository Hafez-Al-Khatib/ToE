# Valid Claim 4: Differentiable Eikonal Equation for Neural Path Planning

**Status:** ✅ Publishable — with corrected complexity claims
**Target venue:** ICML / ICLR (planning / robotics track)
**Paper title candidate:** *"Wave Brain: Differentiable Eikonal Solvers for Neural Planning via Adjoint Sensitivity"*

---

## The Claim (Corrected)

The Eikonal equation |∇u(x)|² = n(x)² describes wavefront propagation through a
medium with spatially varying speed 1/n(x). Its solution u(x) gives the minimum
travel time from a source to any point x — which is exactly the value function for
shortest-path planning.

The contribution here is:

1. **A differentiable Eikonal solver** — the Fast Sweeping Method is made
   differentiable via adjoint-method backpropagation, so the "speed field" n(x) can
   be learned end-to-end from planning demonstrations
2. **Operator splitting** — global wave propagation (Eikonal) is coupled with
   local field dynamics (thermodynamic field) via Strang splitting, allowing
   the two components to be trained jointly
3. **Unified planner** — the wave solution provides a global guidance field
   v = −∇u that advects the local thermodynamic field toward the goal

**Important:** This is O(N) per sweep iteration, NOT O(1). The Fast Sweeping Method
requires O(N) iterations over the grid. This must be stated correctly.

---

## Why This Is True

### The Eikonal equation is correct for planning

The connection between the Eikonal equation and shortest paths is mathematically exact:

```
|∇u(x)|² = n(x)²,   u(x_source) = 0
```

Solution u(x) = minimum-time path from x_source to x, through a medium where
velocity = 1/n(x). This is Fermat's principle of least time, and it is the
continuous-space analog of Dijkstra's algorithm.

For robot planning:
- n(x) = 1 everywhere except obstacles where n(x) → ∞
- u(x) = shortest path length from source to x
- ∇u(x) = direction of steepest ascent = direction AWAY from source
- v = −∇u(x) = direction TOWARD source = the optimal policy

This is rigorous, not metaphorical. The Eikonal equation IS the continuous-space
Hamilton-Jacobi equation for shortest-path planning, which is itself a well-known
equivalence in optimal control theory.

### Fast Sweeping Method is a valid, established solver

The Fast Sweeping Method (Zhao, 2005) solves the Eikonal equation via:
- Gauss-Seidel iterations with alternating sweep directions
- Convergence in O(N) total operations for N grid points (for smooth speed fields)
- Exact solution for piecewise-smooth n(x)

The code correctly implements this in `src/wave_solver.py`.

### Differentiability via adjoint is principled

Making the FSM differentiable allows backpropagation through the planning procedure.
The adjoint method (used in optimal control since the 1960s) computes ∂L/∂n(x)
without storing all intermediate states — O(N) memory instead of O(N × iterations).

This is used in PDE-constrained optimization and was recently applied to
differentiable physics (Hu et al., 2019; de Avila Belbute-Peres et al., 2018).
Applying it to the Eikonal equation for learning n(x) from demonstrations is a
legitimate contribution.

### Operator splitting is mathematically justified

Strang splitting for:
```
∂u/∂t = A(u) + B(u)
```
where A = advection (Eikonal guidance) and B = local field dynamics (thermodynamic),
is a second-order accurate method. The coupling is:

```
u^{n+1} = B(Δt/2) ∘ A(Δt) ∘ B(Δt/2) [u^n]
```

This is a standard numerical method (Strang, 1968) that produces second-order
accuracy if each sub-step is solved exactly. The approximation error is O(Δt²).

---

## What Work Has Been Done

| Component | File | Status |
|-----------|------|--------|
| Eikonal solver (Fast Sweeping) | `src/wave_solver.py` | ✅ Complete |
| WaveBrain module | `src/wave_solver.py` | ✅ Complete |
| Operator splitting planner | `src/unified_planner.py` | ✅ Complete |
| Semi-Lagrangian advection | `src/unified_planner.py:apply_advection` | ✅ Complete |
| Phase 3 training script | `src/train_phase3.py` | ✅ Complete |
| Planner output visualizations | `outputs/unified_planner/` | ✅ Exists |

---

## What Is Still Needed for Publication

1. **Correct complexity claims everywhere** — audit all docs and comments for "O(1)"
   and replace with "O(N) per sweep pass"

2. **Maze-solving experiments** — structured quantitative evaluation:
   - Path optimality ratio: (found path length) / (optimal path length)
   - Success rate on random mazes of varying sizes
   - Comparison with A* and Value Iteration baselines

3. **Learning n(x) from demonstrations** — the differentiable solver enables learning
   the speed field from expert paths. Implement and test this end-to-end pipeline.

4. **Scaling analysis** — show how planning quality and compute scale with grid size N

5. **Ablation: with/without operator splitting** — show that coupling local dynamics
   with global wave propagation outperforms each alone

---

## Critical Issue: "O(1)" Must Be Removed

The project documents and comments claim "O(1) inference" for the Eikonal solver.
The code itself refutes this (`wave_solver.py`: "Complexity Honesty: O(grid_size) iterations").

Correct complexity statement:
```
FSM solve time: O(N · d) where N = grid points, d = number of sweep directions (2^dim)
For 2D: 4 sweeps, each O(N) → O(4N) = O(N) total
Gradient computation (adjoint): O(N) additional
Total: O(N) — linear in grid size, matching Dijkstra on a discrete graph
```

This is still a useful algorithm — linear-time planning is good. The O(1) claim
was a misunderstanding that confused "no separate inference network" with "constant time."

---

## Key Figures for the Paper

1. **Figure 1:** Eikonal solution visualization — wavefronts emanating from goal
2. **Figure 2:** Guidance field v = −∇u overlaid on maze with obstacles
3. **Figure 3:** Unified planner — local thermodynamic field + global wave guidance
4. **Figure 4:** Planning success rate vs maze complexity (comparison with A*, VI)
5. **Figure 5:** Learned speed field n(x) recovered from demonstration trajectories

---

## Related Work to Cite

- Zhao, H. (2005). *A Fast Sweeping Method for Eikonal Equations.* Math. Comp.
- Sethian, J. (1999). *Fast Marching Methods.* SIAM Review (alternative solver)
- de Avila Belbute-Peres, F. et al. (2018). *End-to-End Differentiable Physics.* NeurIPS
- Tamar, A. et al. (2016). *Value Iteration Networks.* NeurIPS (planning with NNs)
- Lee, A. et al. (2018). *Gated Path Planning Networks.* ICML
- Strang, G. (1968). *On the construction of difference schemes.* SIAM J. Num. Anal.
