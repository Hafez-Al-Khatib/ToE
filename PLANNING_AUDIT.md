# 🛡️ Rigorous Audit: Paper 3 Planning & Zero-Shot Generalization

**Audit Date:** April 22, 2026  
**Status:** 🚩 CRITICAL FAILURE DETECTED (Implementation vs. Theory Mismatch)

## 1. Executive Summary of Failure
The background training runs for PID 31276 and 44724 were terminated due to **mathematical divergence** and **physical non-compliance**. The audit reveals that while the Eikonal theory is sound, the current training curriculum allows both models (WaveBrain and CNN) to "cheat" or "break" the underlying physics of the task.

---

## 2. Theoretical Divergence Audit

### ❌ Audit A: The CNN "Negative Time" Hallucination
*   **Observation:** CNN Loss reached `-2.24e12`.
*   **Root Cause:** The `CNNPlanner` output layer was linear (unbounded). The `path_distance_loss` rewards minimizing travel time. The network discovered a mathematical shortcut: by predicting negative trillions for travel time, it achieves a "super-optimal" loss that has no physical meaning.
*   **Verdict:** Theoretical violation. Planning requires a **strictly positive manifold**.
*   **Required Fix:** Apply `Softplus` or `Sigmoid` activation to all planning output heads to enforce $n(x) > 0$.

### ❌ Audit B: The WaveBrain "Flat-Field" Persistence
*   **Observation:** Solution images showed straight lines cutting through walls after 7 epochs.
*   **Root Cause:** The KAN-based `FieldEncoder` starts in a "Neutral Potential" state ($n \approx \text{const}$). Because the Eikonal solver finds a straight path in a flat field, the gradient $\partial L / \partial n$ is zero for all wall pixels that aren't on that straight line. The model is "blind" to the walls until it accidentally learns to see them.
*   **Verdict:** Implementation failure. The curriculum lacks **perceptual grounding**.
*   **Required Fix:** Implement a "Jumpstart Curriculum" where the KAN is pre-trained to reconstruct the maze geometry before the Eikonal loss is applied.

---

## 3. High-Tier Publication Pivot: The "Transfer" Roadmap

To move from "Bad Results" to "NeurIPS Best Paper" quality, we must stop treating Paper 3 as a standalone training task. The audit suggests a stronger architectural roadmap:

### Block 1: The Perceptual Foundation (Paper 1)
*   Train the KAN-EBM *only* on the **Perception Task** (Denoising).
*   Result: A KAN that has learned a sharp potential $V(u)$ where "Clean Image" = Minimum Energy.

### Block 2: Zero-Shot Transfer (The "Triple Role" Proof)
*   Inject the pre-trained KAN into the Eikonal Solver.
*   **Hypothesis:** The "Denoising Energy" is functionally equivalent to "Planning Slowness." 
*   **The Experiment:** Can the model solve a maze it has never "seen" for planning, just by using its "visual common sense"?

### Block 3: The Scaling Law Moat
*   Compare this **Zero-Shot WaveBrain** against a **Fully-Trained CNN**.
*   **Expected Result:** The CNN will still fail on 64x64 mazes (fixed receptive field), while the Zero-Shot WaveBrain will succeed (infinite physical scaling).

---

## 4. Verification Checkpoints for Next Execution

Before restarting any training, the following code audits must be passed:
- [ ] **Positivity Constraint:** Verify `Softplus` in `CNNPlanner`.
- [ ] **Loss Normalization:** Ensure `path_distance_loss` is scale-invariant.
- [ ] **Curriculum Logic:** Implement `pretrain_encoder_mse()` in `exp_planning_benchmarks.py`.
- [ ] **Collision Rigor:** Verify that the `is_valid` check in evaluation uses a 1-pixel tolerance (no corner-cutting allowed).

**Audit Signature:** *Gemini CLI Senior Auditor* 🛡️
