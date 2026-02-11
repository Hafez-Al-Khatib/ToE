# Phase 3: Wave Brain — Path Integral Reasoning

This document explains the theoretical foundation of Phase 3: using wave physics to solve reasoning problems at O(1) depth instead of iterative search.

---

## Table of Contents
1. [The Core Insight](#1-the-core-insight)
2. [Fermat's Principle & Light Propagation](#2-fermats-principle--light-propagation)
3. [Path Integrals & Quantum Mechanics](#3-path-integrals--quantum-mechanics)
4. [The Eikonal Equation](#4-the-eikonal-equation)
5. [Learning Refractive Indices](#5-learning-refractive-indices)
6. [Implementation: Differentiable Solver](#6-implementation-differentiable-solver)
7. [Applications](#7-applications)

---

## 1. The Core Insight

### The Problem with Search

Traditional pathfinding (A*, Dijkstra, BFS) explores the graph explicitly:
- Check node, add neighbors to queue, repeat
- Complexity: O(V + E) for graphs, O(N²) for N×N grids
- Each step is sequential — we can't parallelize the search

For reasoning chains (logical inference, planning):
- Each "node" is a state/premise
- Each "edge" is a logical step
- Depth-first search explores paths one at a time

### The Wave Solution

Light doesn't "search" for the shortest path. It **propagates as a wave**:
- The wavefront expands in all directions simultaneously
- Constructive interference amplifies the shortest path
- Destructive interference cancels longer paths

> **The optimal path emerges from the physics — no search required.**

This is O(1) reasoning depth: solve the wave equation once, read off the answer.

### Physical Analogy

Imagine a swimming pool with varying depth:
- Shallow regions = higher resistance (slow waves)
- Deep regions = lower resistance (fast waves)

Drop a stone at one end. The ripples will:
1. Spread outward
2. Bend around obstacles (diffraction)
3. Arrive at the target along the fastest route

**Key**: The "fastest route" through varying depths is exactly the problem of pathfinding in a weighted graph!

---

## 2. Fermat's Principle & Light Propagation

### Fermat's Principle of Least Time

> Light travels between two points along the path that takes the **least time**.

Mathematically:

$$t = \int_A^B \frac{ds}{v(x)} = \int_A^B \frac{n(x)}{c} \, ds$$

where:
- $v(x)$ is the local speed of light
- $n(x) = c/v(x)$ is the **refractive index**
- $c$ is the speed of light in vacuum
- $ds$ is the arc length element

Light minimizes the **optical path length**:

$$L = \int_A^B n(x) \, ds$$

### Snell's Law Derivation

Consider light crossing from medium 1 (index $n_1$) to medium 2 (index $n_2$):

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

This follows directly from Fermat's principle via calculus of variations.

### Connection to Pathfinding

If we interpret:
- **Refractive index** $n(x)$ = **cost** at position $x$
- **Optical path length** = **total cost of path**

Then Fermat's principle says: **light finds the minimum-cost path**.

This is exactly what we want for pathfinding!

---

## 3. Path Integrals & Quantum Mechanics

### Feynman's Path Integral

In quantum mechanics, a particle doesn't take a single path. It takes **all paths simultaneously**:

$$\langle x_f | x_i \rangle = \int \mathcal{D}[x(t)] \, e^{i S[x(t)] / \hbar}$$

where:
- The integral is over all possible paths from $x_i$ to $x_f$
- $S[x(t)]$ is the action of the path
- $\hbar$ is Planck's constant

### Interference & the Classical Limit

Each path contributes a complex phase $e^{iS/\hbar}$:
- Paths with similar action → phases align → **constructive interference**
- Paths with different action → phases cancel → **destructive interference**

In the classical limit ($\hbar \to 0$):
- Only paths with stationary action survive
- These are the classical trajectories (solutions to Newton's/Hamilton's equations)

### The AI Analogy

We can think of reasoning as:
1. **Question** = Source point
2. **Answer** = Target point
3. **Reasoning chains** = Paths through inference space
4. **Wave propagation** = Parallel exploration of all paths
5. **Interference** = Selecting the optimal chain

---

## 4. The Eikonal Equation

### From Waves to Rays

When the refractive index varies slowly (compared to wavelength), the wave equation simplifies to the **Eikonal equation**:

$$|\nabla u(x)|^2 = n(x)^2$$

where:
- $u(x)$ is the **travel time** (phase) from the source
- $n(x)$ is the refractive index
- The gradient $\nabla u$ points in the direction of wave propagation

### Physical Interpretation

The Eikonal equation says: the wavefront advances at speed $1/n(x)$.
- High $n(x)$ → slow propagation → high cost region
- Low $n(x)$ → fast propagation → low cost region

### Solving the Eikonal Equation

**Fast Marching Method (FMM)**:
1. Initialize: $u(\text{source}) = 0$, $u(\text{elsewhere}) = \infty$
2. Propagate: Update neighbors in order of increasing $u$
3. Uses Godunov upwind scheme for numerical derivative

**Fast Sweeping Method**:
1. Alternating directional sweeps (left-right, up-down, etc.)
2. Update each point using neighbors
3. Converges in O(N) sweeps for N×N grid

Both give the **travel time** $u(x)$ from source to every point.

### Extracting the Path

Once we have $u(x)$, the optimal path is found by:
1. Start at target
2. Follow $-\nabla u$ (steepest descent in travel time)
3. Continue until reaching source

This is just gradient descent on the travel time field!

---

## 5. Learning Refractive Indices

### The Key Insight

In pathfinding:
- The "maze" or "graph" defines costs
- We want to find the minimum-cost path

In our model:
- The **learned refractive index** $n_\theta(x)$ defines costs
- The **Eikonal solution** gives the optimal path

### Training Pipeline

```
Input: Problem description (e.g., maze image)
    ↓
Neural Network: n_θ(x) = predicted refractive index field
    ↓
Eikonal Solver: u(x) = travel time field
    ↓
Path Extraction: Follow -∇u from target to source
    ↓
Loss: Compare to ground truth path
    ↓
Backprop: Through solver → update θ
```

### Differentiable Solver

For end-to-end learning, we need gradients through the Eikonal solver:

$$\frac{\partial u}{\partial n} = ?$$

This requires:
1. **Implicit differentiation** of the Eikonal equation
2. **Adjoint method** for efficient backprop

We implement this using:
- Custom autograd function
- Iterative solver in forward pass
- Adjoint solve in backward pass

---

## 6. Implementation: Differentiable Solver

### Forward Pass: Solve Eikonal

We use an iterative solver based on Fast Sweeping:

```python
def solve_eikonal(n, source_pos, n_sweeps=4):
    """
    Solve |∇u|² = n² with boundary condition u(source) = 0.
    
    Parameters:
        n: Refractive index field (H, W)
        source_pos: (y, x) position of source
        n_sweeps: Number of sweeping passes
    
    Returns:
        u: Travel time field (H, W)
    """
    H, W = n.shape
    u = torch.full((H, W), float('inf'))
    u[source_pos] = 0.0
    
    # Godunov upwind discretization
    for sweep in range(n_sweeps):
        for y in sweep_orders_y:
            for x in sweep_orders_x:
                # Get neighbors
                u_left = u[y, x-1] if x > 0 else inf
                u_right = u[y, x+1] if x < W-1 else inf
                u_up = u[y-1, x] if y > 0 else inf
                u_down = u[y+1, x] if y < H-1 else inf
                
                # Godunov scheme
                u_x = min(u_left, u_right)
                u_y = min(u_up, u_down)
                
                # Solve quadratic for u
                u_new = solve_quadratic(u_x, u_y, n[y, x])
                u[y, x] = min(u[y, x], u_new)
    
    return u
```

### Godunov Quadratic

At each point, we solve:

$$\left(\frac{u - u_x}{\Delta x}\right)^2 + \left(\frac{u - u_y}{\Delta y}\right)^2 = n^2$$

For $\Delta x = \Delta y = 1$:

$$(u - u_x)^2 + (u - u_y)^2 = n^2$$

Solutions:
1. If $|u_x - u_y| \geq n$: $u = \min(u_x, u_y) + n$ (1D update)
2. Otherwise: $u = \frac{u_x + u_y + \sqrt{2n^2 - (u_x - u_y)^2}}{2}$ (2D update)

### Backward Pass: Adjoint Method

For backpropagation, we need $\partial L / \partial n$ given $\partial L / \partial u$.

Using the adjoint method:
1. **Forward**: Solve $|\nabla u| = n$ to get $u$
2. **Compute loss gradient**: $\bar{u} = \partial L / \partial u$
3. **Backward**: Solve adjoint equation for $\bar{n} = \partial L / \partial n$

The adjoint equation arises from implicit differentiation of the Eikonal equation.

---

## 7. Applications

### Maze Solving

**Setup**:
- Input: Maze image (white = passable, black = wall)
- Learn: $n_\theta(x) = $ encoder(maze image)
- High $n$ for walls, low $n$ for passages

**Training**:
- Given maze + optimal path pairs
- Loss: Did the wave find the correct path?

**Inference**:
- Feed new maze → get $n$ field → solve Eikonal → extract path
- Single forward pass, regardless of maze complexity

### Logic/Theorem Proving

**Setup**:
- States = propositions/formulas
- Edges = inference rules
- Cost = complexity/probability of each rule

**Implementation**:
- Embed logical state space into 2D grid (using learned embedding)
- Refractive index = inverse probability of transition
- Eikonal gives shortest proof

### Planning & Decision Making

**Setup**:
- States = world configurations
- Actions = transitions
- Cost = action cost + state uncertainty

**Benefit**: Plans entire trajectory at once, not step-by-step.

---

## Summary

Phase 3 replaces explicit search with implicit wave physics:

| Aspect | Traditional Search | Wave Brain |
|--------|-------------------|------------|
| Exploration | Sequential | Parallel (all paths at once) |
| Complexity | O(V + E) | O(solve Eikonal) ≈ O(N·sweeps) |
| Reasoning depth | Proportional to path length | O(1) forward passes |
| Parallelism | Limited | Highly parallel (GPU) |

The key insight:

> **Light doesn't search. It propagates. So should our reasoning.**

By learning the "refractive index" (cost structure) of a problem space, we transform discrete search into continuous wave physics — finding optimal solutions through interference rather than enumeration.

---

*This completes the theoretical foundations. See the implementation in [wave_solver.py](../src/wave_solver.py).*
