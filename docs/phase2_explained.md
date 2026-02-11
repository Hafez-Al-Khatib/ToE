# Phase 2: Hamiltonian-KAN — Shaping the Energy Landscape with Splines

This document explains the theoretical foundation of Phase 2: using Kolmogorov-Arnold Networks (KANs) to create highly expressive energy landscapes, combined with Hamiltonian dynamics for stable inference.

---

## Table of Contents
1. [Why KANs for Energy?](#1-why-kans-for-energy)
2. [The Kolmogorov-Arnold Theorem](#2-the-kolmogorov-arnold-theorem)
3. [KAN Architecture: Learnable Edges](#3-kan-architecture-learnable-edges)
4. [B-Spline Basis Functions](#4-b-spline-basis-functions)
5. [Hamiltonian Mechanics for AI](#5-hamiltonian-mechanics-for-ai)
6. [H-KAN: The Synthesis](#6-h-kan-the-synthesis)
7. [Implementation Details](#7-implementation-details)

---

## 1. Why KANs for Energy?

### The Problem with MLPs

In Phase 1, we used an MLP to parameterize the energy function:

$$E_\theta(x) = \text{MLP}(x)$$

MLPs are powerful but have limitations for energy landscapes:

1. **Smooth everywhere**: MLPs naturally produce smooth functions (due to smooth activations). But many physical systems have **sharp transitions** (phase boundaries, bifurcations).

2. **Fixed basis**: The "features" extracted by MLPs are fixed after training. They can't adapt to local data structure.

3. **Interpretability**: We can't easily see "what" the MLP learned. The energy landscape is a black box.

### The KAN Promise

KANs replace **learned weights** with **learned functions**:

| Aspect | MLP | KAN |
|--------|-----|-----|
| Where parameters live | Node weights | Edge functions |
| What's learned | Linear combinations | Nonlinear transforms |
| Flexibility | Fixed activation | Adaptive activation |
| Interpretability | Low | High (visualize edge functions) |

For energy landscapes, KANs let us learn:
- **Sharp valleys** (stable states)
- **Ridges and saddles** (transition states)
- **Multi-modal basins** (multiple solutions)

---

## 2. The Kolmogorov-Arnold Theorem

### The Theorem

**Kolmogorov-Arnold Representation Theorem (1957)**:

Any continuous function $f: [0,1]^n \to \mathbb{R}$ can be written as:

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

where:
- $\phi_{q,p}: [0,1] \to \mathbb{R}$ are **inner functions** (one per input dimension, per q)
- $\Phi_q: \mathbb{R} \to \mathbb{R}$ are **outer functions**
- All functions are continuous (but not necessarily smooth)

### Interpretation

This says: *any* multivariate function can be decomposed into:
1. **Univariate inner functions** applied to each input
2. **Sums** of these transformed inputs
3. **Univariate outer functions** applied to the sums

This is remarkable! We don't need to learn complex multi-dimensional interactions — just compositions of 1D functions.

### The Catch

The original theorem has issues for practical ML:
- The inner functions can be **highly non-smooth** (even fractal)
- The theorem is an existence result, not constructive
- $2n + 1$ terms may not suffice for learnable approximation

**KAN networks** relax the constraints, using smooth splines as the univariate functions.

---

## 3. KAN Architecture: Learnable Edges

### The Core Idea

In a standard MLP layer:

$$y = \sigma(Wx + b)$$

The parameters are the **weight matrix** $W$ and bias $b$. The activation $\sigma$ is fixed (e.g., ReLU, SiLU).

In a KAN layer:

$$y_j = \sum_{i=1}^{n_{\text{in}}} \phi_{i,j}(x_i)$$

where each $\phi_{i,j}$ is a **learnable 1D function** represented as a spline.

### Visual Comparison

```
MLP Layer:
    x₁ ──┬──[w₁₁]──┬──→ σ(·) → y₁
         │         │
    x₂ ──┼──[w₂₁]──┘
         │
         └──[w₁₂]──┬──→ σ(·) → y₂
                   │
    x₂ ──[w₂₂]──┘

KAN Layer:
    x₁ ──┬──[φ₁₁(·)]──┬──→  Σ  → y₁
         │            │
    x₂ ──┼──[φ₂₁(·)]──┘
         │
         └──[φ₁₂(·)]──┬──→  Σ  → y₂
                      │
    x₂ ──[φ₂₂(·)]──┘

    [wᵢⱼ] = scalar weight
    [φᵢⱼ(·)] = learnable B-spline function
```

### Parameters

For a KAN layer from $n_{\text{in}}$ to $n_{\text{out}}$:
- Each edge has a spline with $G$ grid points and degree $k$
- Total parameters: $n_{\text{in}} \times n_{\text{out}} \times (G + k)$

Compared to MLP: $n_{\text{in}} \times n_{\text{out}}$ (just the weights)

KAN has more parameters per connection, but the functions are **interpretable** — we can plot each $\phi_{i,j}$.

---

## 4. B-Spline Basis Functions

### What is a B-Spline?

A **B-spline of degree k** is a piecewise polynomial function defined by:
- A **knot vector** $\mathbf{t} = (t_0, t_1, \ldots, t_{G+k})$
- **Control points** $\mathbf{c} = (c_0, c_1, \ldots, c_{G+k-1})$

The spline is:

$$\phi(x) = \sum_{i=0}^{G+k-1} c_i \cdot B_{i,k}(x)$$

where $B_{i,k}(x)$ are the **B-spline basis functions**.

### B-Spline Recursion

The basis functions are defined recursively (Cox-de Boor):

$$B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \le x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)$$

### Properties

1. **Local support**: Each basis function is non-zero only in a small interval. This means modifying one control point only affects the function locally.

2. **Partition of unity**: At any point $x$, the basis functions sum to 1:
   $$\sum_i B_{i,k}(x) = 1$$

3. **Smoothness**: Degree-$k$ splines have $C^{k-1}$ continuity (smooth derivatives).

4. **Convex hull**: The spline lies within the convex hull of control points.

### Why Splines for Energy?

For energy landscapes, splines offer:
- **Flexibility**: Can approximate arbitrary smooth functions
- **Stability**: No extreme oscillations (unlike high-degree polynomials)
- **Locality**: Changes to the landscape are localized
- **Differentiability**: Smooth gradients for Langevin dynamics

---

## 5. Hamiltonian Mechanics for AI

### From Langevin to Hamilton

In Phase 1, we used Langevin dynamics:

$$x_{t+1} = x_t - \epsilon \nabla_x E(x_t) + \text{noise}$$

This is **first-order** dynamics — we only track position $x$, and velocity is implicitly $-\nabla E$.

**Hamiltonian dynamics** is **second-order** — we track both position and momentum:

$$\frac{dx}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial x}$$

where the Hamiltonian is:

$$H(x, p) = \underbrace{\frac{1}{2} p^T M^{-1} p}_{\text{Kinetic Energy}} + \underbrace{E(x)}_{\text{Potential Energy}}$$

### Why Hamiltonian?

1. **Energy conservation**: Without friction, $H$ is constant. The system orbits at a fixed energy level.

2. **Symplectic structure**: The dynamics preserve phase space volume (Liouville's theorem). No probability mass is gained or lost.

3. **Efficient exploration**: Momentum carries the system through low-gradient regions. It doesn't get stuck as easily as gradient descent.

4. **Physical interpretation**: The system behaves like a real physical particle. We can add friction, temperature, external forces.

### Hamiltonian Monte Carlo (HMC)

HMC is a sampling algorithm that uses Hamiltonian dynamics:

1. Sample momentum: $p \sim \mathcal{N}(0, M)$
2. Simulate Hamiltonian dynamics for $L$ steps (leapfrog integrator)
3. Accept/reject based on energy change

This produces samples from $p(x) \propto \exp(-E(x))$ with much better mixing than random walk.

---

## 6. H-KAN: The Synthesis

### The Architecture

**Hamiltonian-KAN** combines:
1. **KAN layers** as the potential energy function
2. **Hamiltonian dynamics** for inference

$$H(x, p) = \frac{1}{2} ||p||^2 + E_{\text{KAN}}(x)$$

where $E_{\text{KAN}}$ is a KAN network:

$$E_{\text{KAN}}(x) = \text{KAN}_L \circ \text{KAN}_{L-1} \circ \cdots \circ \text{KAN}_1(x)$$

### Inference

To find the minimum of $E(x)$, we can either:

**Option A: Gradient Descent** (as in Phase 1)
$$x_{t+1} = x_t - \epsilon \nabla E(x_t)$$

**Option B: Hamiltonian Dynamics** (more exploratory)
```python
# Initialize
p = torch.randn_like(x)  # Random momentum

# Leapfrog integration
for step in range(n_steps):
    p = p - 0.5 * step_size * grad_E(x)  # Half-step p
    x = x + step_size * p                # Full-step x
    p = p - 0.5 * step_size * grad_E(x)  # Half-step p
```

**Option C: Damped Hamiltonian** (the best of both)
$$\dot{x} = p, \quad \dot{p} = -\nabla E(x) - \gamma p$$

This has friction ($\gamma p$), so energy decreases and the system settles.

### Why H-KAN for Control?

For control tasks (inverted pendulum, robot arm), we need:
1. **Multiple equilibria**: The system may have several stable postures
2. **Smooth transitions**: Trajectories between states should be physically plausible
3. **Energy conservation**: (Approximately) for efficient motion

H-KAN provides:
- **KAN**: Learns complex, multi-modal energy landscapes
- **Hamiltonian**: Generates smooth, physically plausible trajectories

---

## 7. Implementation Details

### The KAN Layer

```python
class KANLayer(nn.Module):
    """
    A single KAN layer: y_j = Σ_i φ_{i,j}(x_i)
    
    Each edge (i,j) has a learnable B-spline φ_{i,j}.
    """
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Number of control points per spline
        n_ctrl = grid_size + spline_order
        
        # Learnable control points for all edges
        # Shape: (in_features, out_features, n_ctrl)
        self.ctrl_points = nn.Parameter(
            torch.randn(in_features, out_features, n_ctrl) * 0.1
        )
        
        # Fixed knot vector (uniform grid on [-1, 1])
        knots = torch.linspace(-1, 1, grid_size + 1)
        # Extend knots for boundary conditions
        knots = torch.cat([
            knots[0].repeat(spline_order),
            knots,
            knots[-1].repeat(spline_order)
        ])
        self.register_buffer('knots', knots)
```

### Spline Evaluation

```python
def eval_spline(self, x, ctrl_points):
    """
    Evaluate B-spline at x using control points.
    
    Uses the Cox-de Boor recursion efficiently.
    """
    # x: (batch, in_features)
    # ctrl_points: (in_features, out_features, n_ctrl)
    
    # Compute basis functions B_{i,k}(x) for all i
    bases = self.compute_bases(x)  # (batch, in_features, grid_size + spline_order)
    
    # Weighted sum: φ(x) = Σ_i c_i B_i(x)
    # (batch, in, out) = (batch, in, n_ctrl) @ (in, out, n_ctrl)^T
    return torch.einsum('bin,ion->bio', bases, ctrl_points)
```

### Hamiltonian Integrator

```python
def leapfrog_step(x, p, grad_E, step_size):
    """
    One step of the leapfrog (Störmer-Verlet) integrator.
    
    This is a symplectic integrator: it preserves the geometric
    structure of Hamiltonian dynamics, leading to stable long-term
    evolution.
    """
    # Half-step in momentum
    p = p - 0.5 * step_size * grad_E(x)
    
    # Full-step in position
    x = x + step_size * p
    
    # Half-step in momentum (with new position)
    p = p - 0.5 * step_size * grad_E(x)
    
    return x, p
```

### Training H-KAN

For control tasks, we can train with:

1. **Equilibrium Propagation** (as in Phase 1):
   - Let system relax to equilibrium
   - Compare to desired state
   - Backprop through dynamics

2. **Model Predictive Control**:
   - Learn $E(x)$ such that the desired trajectory is low-energy
   - Inference: Run Hamiltonian dynamics from current state

3. **Imitation Learning**:
   - Given expert trajectories $(x_t)$
   - Train $E$ to make these trajectories low-energy paths

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Grid size | 5-20 | More = finer detail, more parameters |
| Spline order | 3 (cubic) | 3 is standard, higher = smoother |
| Leapfrog steps | 10-50 | More = more exploration |
| Step size | 0.01-0.1 | Depends on energy scale |
| Damping (γ) | 0.1-0.5 | Higher = faster convergence |

---

## Summary

Phase 2 enhances the energy-based approach with:

1. **KAN layers**: Replace rigid MLPs with flexible spline-based functions. This allows learning complex, multi-modal energy landscapes with interpretable structure.

2. **Hamiltonian dynamics**: Replace first-order gradient descent with second-order momentum-based dynamics. This provides smoother trajectories and better exploration.

The combination is **H-KAN**: a physics-based architecture where:
- The learned energy landscape defines "what's good" (low energy)
- Hamiltonian dynamics defines "how to get there" (follow the physics)

> The energy landscape is the geometry of thought. KANs let us sculpt it precisely.

---

*Next: See [phase3_explained.md](phase3_explained.md) for wave-based reasoning with path integrals.*
