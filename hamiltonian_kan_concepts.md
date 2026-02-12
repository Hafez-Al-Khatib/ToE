# Hamiltonian KAN: Conceptual Shifts

This document explains the architectural and philosophical changes introduced in the `feature/hamiltonian-kan` branch.

## 1. From "Hardcoded Physics" to "Learned Physics"

### Old Approach: Ginzburg-Landau (G-L)
Previously, `ThermodynamicField` used a hardcoded **Double-Well Potential**:
$$ V(u) = -a u^2 + b u^4 $$
This enforced a specific physical prior: "The system should be bistable (binary)."

### New Approach: Hamiltonian KAN
We introduced `HamiltonianField`, where the energy density is learned by a **Kolmogorov-Arnold Network (KAN)**:
$$ E[u] = \int \text{KAN}(u, \nabla u, \dots) dx $$

This allows the system to **discover** the physics required for the task:
-   **Pushing**: Learns conservation laws (Barrier Potentials).
-   **Denoising**: Learns "Smoothness" and "Edge" priors (convex/non-convex potentials).
-   **Sudoku**: Learns discrete constraints (Spin Glass landscape).

## 2. The "Unbiased Potential" Breakthrough

### The Problem: Gravity Wells
Initially, we initialized the KAN to mimic the G-L potential ($V = 0.1 u^2$).
-   **Effect**: This created a strong "Gravity Well" at $u=0$ (Empty).
-   **Symptom**: In Sudoku, the model would regress to "Random Guessing" because the physics (pull to 0) fought the logic (push to non-zero digits).

### The Solution: Zero-Initialization
We removed the bias:
-   **Old**: `potential_a = 0.1` (Bias toward 0)
-   **New**: `potential_a = 0.0` (Flat Landscape)

This forced the KAN to **learn the potential from scratch**.
-   **Result**: Sudoku accuracy stabilized at **54%** (stable) vs 63% (unstable). The model carved its own energy wells based purely on data.

## 3. Generalization Scope

This branch proves that a **Single Architecture** (`HamiltonianField`) handles diverse domains:

| Domain | Physics Type | Learned Dynamics | Status |
| :--- | :--- | :--- | :--- |
| **Pushing** | Continuous / Conservation | Mass Preservation | ✅ Solved |
| **Denoising** | Continuous / Geometry | Smoothness + Edges | ✅ Solved |
| **Sudoku** | Discrete / Logic | Local Constraints | ⚠️ Partial (54%) |

### Limitation
The Hamiltonian KAN excels at **Continuous Physics** but faces the **Spin Glass Frustration** problem in **Discrete Logic** (Sudoku). It learns local rules well but struggles to tunnel out of local minima to find the global "Perfect Solve."
