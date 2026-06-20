# False Claim 2: "O(1) Inference" via Physics

**Verdict:** ❌ False — inference is O(K × N), strictly slower than a feedforward pass
**Severity:** High — appears in the project's core motivation
**Fixable:** Yes — reframe as "test-time iterative refinement" (see valid_research/05)

---

## What the Project Claims

From `README.md` and various documentation: that energy-based inference constitutes
"O(1)" or constant-time inference because the "physics does the work" rather than
explicit computation. The implicit argument is that because physical systems find
equilibria without running a computation, the neural analogy inherits this property.

---

## Why This Is False

### The code itself refutes it

`src/wave_solver.py` contains this comment, written by the author:

```python
# Complexity Honesty:
# O(grid_size) iterations, not O(1)
```

The Fast Sweeping Method for the Eikonal equation requires iterating over all N grid
cells multiple times. For a 2D grid: 4 sweep passes, each O(N) → O(4N) = O(N) total.

### Gradient descent inference is O(K × N)

One step of energy gradient descent on a field of N pixels:
1. Forward pass through KAN: O(N × input_dim × KAN_params)
2. Backward pass (autograd): O(N × same)
3. Field update: O(N)

Total per step: O(N). For K steps: O(K × N).

A feedforward MLP with the same parameter count processes all N pixels in O(N × params)
in one pass. For typical K = 10-50, energy-based inference is 10-50× SLOWER than a
feedforward network of comparable size, not faster.

### Why physical intuition breaks down

The confusion comes from thinking about physical systems where:
- An actual ball rolls downhill in continuous time
- The physics "solves" the optimization for free
- There is no computational cost

But in a simulation (which is what a neural network running on a GPU is):
- Each time step requires explicit numerical computation
- The ball does not "actually" roll — the computer approximates the rolling
- Every step costs FLOPs, and FLOPs scale with problem size

There is no way to harness actual physical dynamics without specialized hardware
(e.g., analog computers, optical networks, or biological neurons). On digital hardware,
all physics simulations are O(N × steps).

---

## The Actual Complexity Comparison

For a 28×28 image (N = 784 pixels):

| Method | Inference cost |
|--------|---------------|
| MLP (784→512→256→1, EBM) | O(784 × 512 + 512 × 256) ≈ 530K ops, 1 pass |
| GradientFlowField, K=1 step | O(784 × same) + backward = ~1M ops |
| GradientFlowField, K=10 steps | ~10M ops |
| GradientFlowField, K=50 steps | ~50M ops |
| Fast Sweeping (28×28 grid) | O(4 × 784) = ~3K ops, but N=784 is tiny here |

Energy-based inference is always more expensive per unit quality than a forward pass,
because it requires computing the backward pass (gradient) at inference time.

---

## What Is True Instead

The advantage is not speed — it is **quality scaling with compute**:

```
K=1:   fast but low quality (roughly equivalent to feedforward)
K=10:  10× slower, noticeably better denoising
K=50:  50× slower, near-optimal for the learned energy landscape
```

This is a different kind of value proposition than O(1). It is:
- **Flexible quality-speed tradeoff** — not possible with feedforward networks
- **No retraining required** — to get better results, just run more steps
- **Adaptive stopping** — can halt when E(u_K) < threshold, before K_max

This reframing makes the paper stronger, not weaker, because it positions the work
correctly relative to modern test-time compute scaling research.

---

## Where the Claim Appears (Must Be Corrected Before Submission)

| Location | Incorrect text |
|----------|---------------|
| `README.md` | Any mention of "O(1)" or "physics-speed" |
| `docs/theory_foundations.md` | Complexity claims |
| `docs/phase2_explained.md` | Inference speed claims |
| `docs/phase3_explained.md` | Eikonal "O(1)" claims |
| Tutorial slides | Speed comparison slides |

**Action required:** Search the entire project for "O(1)" and review each occurrence.
Replace all instances with the correct O(K × N) framing.

---

## The Correct Statement

```
"Unlike feedforward networks where inference quality is fixed by a single forward
pass, energy-based inference allows trading additional compute for higher quality.
Running K gradient steps costs O(K × N) and produces quality that improves
monotonically with K, providing a controllable compute-quality tradeoff."
```
