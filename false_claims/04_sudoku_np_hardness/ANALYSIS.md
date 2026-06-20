# False Claim 4: Sudoku Solving via Continuous Energy Minimization

**Verdict:** ❌ Structurally limited — NP-hardness prevents reliable continuous relaxation
**Severity:** High — 54% accuracy cannot be presented as a research success
**Fixable:** Partially — can be reframed as a negative result, or replaced with solvable tasks

---

## What the Project Claims

Sudoku solving is presented as a demonstration of the KAN-Hamiltonian energy field's
ability to perform constraint satisfaction via continuous energy minimization. The
system encodes Sudoku as a 9-channel field on a 9×9 grid, where the energy function
should learn to encode Sudoku rules (row/column/box uniqueness), and gradient descent
from a partial puzzle should converge to the complete solution.

The reported result from `hamiltonian_kan_concepts.md`:

> "Sudoku at 54% stable accuracy"

---

## Why This Approach Is Fundamentally Limited

### Sudoku is NP-complete

Determining whether a Sudoku puzzle has a solution is NP-complete (Yato & Seta, 2003).
This does not mean no solver can solve Sudoku — specialized solvers (constraint
propagation + backtracking) solve most human puzzles in milliseconds. It means:

**In the worst case, no polynomial-time algorithm can always solve Sudoku.**

Continuous relaxation is a polynomial-time algorithm. Therefore, it cannot always
solve Sudoku. This is not a weakness of your implementation — it is a theorem.

### The spin glass landscape explains 54% accuracy

The energy landscape for Sudoku in continuous space has the structure of a
**spin glass** — a disordered system with exponentially many local minima, most of
which correspond to *locally consistent but globally incorrect* partial assignments.

Concretely:
- At each cell, the 9-channel softmax is initially uniform (1/9 per digit)
- As evolution proceeds, the field sharpens toward discrete values
- BUT: once a cell commits to a wrong digit (a local minimum), propagation
  of that error through neighboring cells makes many other cells wrong too
- Gradient descent cannot escape these local minima without noise (and noise
  helps only probabilistically, not reliably)

The 54% accuracy means: on average, about half the empty cells are filled correctly,
the other half are stuck in local minima. This is consistent with spin glass behavior
and matches theoretical expectations.

### Why more training epochs will not fix this

The spin glass problem is in the **inference procedure at test time**, not in the
training. Training teaches the energy function to distinguish valid from invalid
configurations globally. But at test time:
- The field evolves via gradient descent
- Gradient descent finds *nearest local minimum*, not global minimum
- For NP-complete problems, nearest local minimum ≠ global minimum in general
- No energy function can fix this without exponential-time inference

Even if the KAN learns the perfect energy function (zero energy on valid solutions,
high energy on invalid ones), gradient descent from a random initialization has no
guarantee of finding the global minimum. It will find some local minimum, and for
Sudoku, most local minima are invalid.

---

## Empirical Evidence Within This Codebase

`hamiltonian_kan_concepts.md` documents the history of attempted fixes:

1. **"Gravity well" bias** — hardcoded potential pulling fields toward one-hot vectors;
   introduced wrong attractors, hurt accuracy → removed
2. **"Unbiased potential"** (potential_a=0.0) — removing the bias improved results
   from lower to ~54%, but hit a ceiling

The ceiling at ~54% is not coincidence. It corresponds to the regime where local
constraint propagation succeeds (cells that are uniquely determined by clues) but
globally frustrated cells (requiring backtracking) fail. This is exactly spin glass behavior.

---

## What Could Work Instead

### Option 1: Reframe as a negative result (publishable as a finding)

> "We show that continuous energy minimization, even with learned KAN energy functions,
> achieves ~54% accuracy on Sudoku (35 clues) due to spin glass frustration in the
> continuous relaxation. This provides empirical evidence that certain NP-complete
> constraint satisfaction problems have energy landscapes that resist gradient-based
> methods, consistent with theoretical predictions from spin glass physics."

This IS publishable — it's an interesting negative result with a principled explanation.

### Option 2: Replace Sudoku with solvable constraint problems

Constraint satisfaction problems that continuous relaxation CAN solve reliably:

- **Bipartite matching / assignment** — convex relaxation works exactly (Birkhoff-von Neumann)
- **Graph coloring on small graphs** (3-coloring of planar graphs)
- **Soft constraint satisfaction** (weighted MAX-CSP where violations are allowed)
- **Linear programming** — trivially solved by continuous relaxation
- **Easy Sudoku with 50+ clues** — when clues uniquely determine the solution,
  constraint propagation works and energy minimization succeeds

### Option 3: Use a different inference procedure for discrete problems

For discrete constraint satisfaction, continuous energy minimization should be
combined with:
- **Simulated annealing** (Langevin noise with decreasing temperature)
- **Parallel tempering** (multiple replicas at different temperatures)
- **Survey propagation** (message passing on the factor graph — state-of-the-art for SAT)

These add exponential complexity in the worst case, which is unavoidable given NP-completeness.

---

## What the 54% Tells You Scientifically

The 54% accuracy actually has an interesting interpretation:
- Sudoku cells with 1 or 2 possible digits (uniquely constrained by clues) are almost
  always solved correctly — energy minimization handles simple propagation well
- Cells that require global backtracking (typically 20-30% of cells in a 35-clue puzzle)
  are essentially random — energy minimization fails on these

So: **54% ≈ 70% local cells × 100% + 30% global cells × ~10%** ≈ 73% is actually
a reasonable rough expectation, and the observed 54% suggests even some locally
constrained cells are failing. This tells you the energy function has not fully
learned the row/column/box constraints — a separate training issue on top of the
fundamental NP-hardness problem.

---

## Summary

| Issue | Status |
|-------|--------|
| NP-hardness of Sudoku | Fundamental — cannot be fixed |
| Spin glass local minima | Fundamental — requires exponential-time escape |
| 54% accuracy ceiling | Predictable from theory |
| Training inadequacy | Secondary issue on top of fundamental problems |
| Fix via more epochs | Will not help beyond local propagation ceiling |
| Fix via better energy | Will not help if inference is still gradient descent |
| Reframe as negative result | ✅ Publishable and honest |

---

## Related Work to Cite

- Yato, T. & Seta, T. (2003). *Complexity and completeness of finding another solution.* IEICE
- Bian, Z. et al. (2020). *Solving Combinatorial Optimization Problems via Machine Learning.* JMLR
- Mezard, M. & Montanari, A. (2009). *Information, Physics, Computation.* (spin glass chapter)
- Mezard, M. et al. (2002). *Analytic and Algorithmic Solution of Random Satisfiability.* Science
  (survey propagation — state-of-the-art for random SAT, shows phase transitions)
