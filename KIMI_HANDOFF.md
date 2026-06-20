# Handoff: Remaining Experiments for KAN-EBM Paper

## Context
This is the NeurIPS 2026 paper submission for **KAN-EBM** (KAN-Parameterized Energy-Based Models). The paper is in `paper_v4.tex`. The main claim is an empirical inference scaling law K*(σ) ≈ Cσ^α for conv-backbone EBMs, where α is architecture-class-dependent.

All editorial fixes have been applied to `paper_v4.tex`. What remains are **three experiments** needed to address reviewer weaknesses.

---

## Experiment 1 (HIGH PRIORITY): KAN-EBM 32K Fine-Grid K* → Add to Table 1

### Why needed
The reviewer says the matched-parameter claim is unclear: Table 1 (ablation) shows KAN-EBM at 110K vs. ConvMLP at 35K — not matched. The paper already claims α=1.365 for KAN-EBM 32K (in the Hessian table) but this value comes from the Hessian JSON `observed_alpha`, not a dedicated fine-grid run.

### What to run
Run the fine-grid K* sweep for the 32K KAN-EBM checkpoint using the same protocol as the main experiment:
- K=1–30, σ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}, 500 CIFAR-10 test images, 3 seeds
- Checkpoint: `outputs/finalization/kstar_param_scaling/kan_small.pt`
- Reference script: `experiments/exp_fine_grid_kstar.py` (loads from `outputs/cifar10/kan_ebm_f32.pt`)
- You need to modify or copy the script to point to `kan_small.pt` and use `n_filters=16, kan_hidden=[48, 16]`

The `KANEnergyModel` constructor in `experiments/exp_cifar10.py` takes these args.

### Expected output
A JSON like `outputs/fine_grid_kstar/fine_grid_kstar_32k.json` with power_law.alpha ≈ 1.365 (based on existing trajectory data).

### Paper update needed
Add this row to **Table 1** (`tab:smooth-mlp`) in `paper_v4.tex` (currently after line ~156):
```latex
\textbf{KAN-EBM (32K)} & 32K & $1.3??$ & $0.???$ & $[?, ?]$ \\
```
Place it below "KAN-EBM (110K)" and above the `\midrule` before ConvMLP variants.

Also add a sentence to §4.3 text: "At matched parameters (KAN-EBM 32K vs. ConvMLP-GELU 35K), the exponent gap is Δα = +[X] (CI [lo, hi]), confirming the gap is not attributable to parameter count."

---

## Experiment 2 (MEDIUM PRIORITY): Hessian Trajectory for ConvMLP-Tanh and ConvMLP-ReLU

### Why needed
The paper currently says "five tested architectures" for the Hessian ranking claim (we already fixed "six" → "five" in the editorial pass). But the table (`tab:hessian-trajectory`) only has GELU and SiLU from the ConvMLP family. Adding Tanh and ReLU would:
1. Let us say "six architectures" again (consistent with fine-grid evaluation)
2. Strengthen the claim that the ranking is systematic across all ConvMLP variants

### What to run
Run Hessian-along-trajectory for `conv_mlp_tanh` and `conv_mlp_relu`:
- σ=0.20, K ∈ {1, 5, 10, 20}, 8 CIFAR-10 test images, Lanczos 20 iterations
- Reference script: look in `experiments/` for Hessian trajectory code, or check `outputs/tier1_trajectory_geometry/`
- Checkpoints: `outputs/finalization/rebuttal/conv_mlp_tanh.pt` and `conv_mlp_relu.pt`
  (same n_filters=16, mlp_hidden=160, activation='tanh'/'relu')

### Expected output
λ_max at K={1,5,10,20} for Tanh and ReLU. Expected values ≈ 55→71 (similar to GELU/SiLU since all 4 share the same conv backbone).

### Paper update needed
Add two rows to **Table 2** (`tab:hessian-trajectory`) in `paper_v4.tex` (currently lines ~178–183):
```latex
ConvMLP-Tanh    & $1.136$ &  $??$ &  $??$ &   $??$ &   $??$ & $?.??\times$ \\
ConvMLP-ReLU    & $1.166$ &  $??$ &  $??$ &   $??$ &   $??$ & $?.??\times$ \\
```
Then change "five tested architectures" back to "six architectures" in:
- Abstract (~line 51)
- Contribution item 4 (~line 78)
- §4.2 geometric correlate paragraph (~line 133)
- Conclusion (~line 253)
- Table 2 caption (~line 185)

---

## Experiment 3 (LOWER PRIORITY): AlexNet LPIPS Values

### Why needed
The reviewer says: "the paper says AlexNet-backbone LPIPS yields identical rankings, confirming backbone independence. But it does not show the AlexNet values."

### What to run
Re-run LPIPS evaluation using AlexNet backbone (lpips library) on the same 100 CIFAR-10 images at σ=0.10 for all models.

The SqueezeNet results are in `outputs/landscape_geometry/lpips_evaluation.json`. Existing values:
- KAN-EBM 110K: 0.021
- ConvMLP-GELU: 0.026
- ConvMLP-ReLU: 0.028
- Micro-DnCNN: 0.033

Run with `lpips.LPIPS(net='alex')` and record the same models.

### Paper update needed
Replace this sentence in §4.4 (~line 193):
> "AlexNet-backbone LPIPS yields identical rankings, confirming backbone independence."

With a small inline table or parenthetical like:
> "AlexNet-backbone LPIPS: KAN $0.0??$ vs. ConvMLP-GELU $0.0??$ vs. DnCNN $0.0??$ (identical ranking to SqueezeNet backbone)."

Or add a small 2-column table under Tab 3.

---

## File Map

| File | Role |
|------|------|
| `paper_v4.tex` | Main paper — all editorial fixes applied |
| `experiments/exp_fine_grid_kstar.py` | Fine-grid K* sweep (adapt for 32K) |
| `experiments/exp_cifar10.py` | KANEnergyModel, ConvSmoothMLPEBM definitions |
| `outputs/finalization/kstar_param_scaling/kan_small.pt` | 32K KAN checkpoint |
| `outputs/finalization/rebuttal/conv_mlp_gelu.pt` | ConvMLP-GELU checkpoint |
| `outputs/finalization/rebuttal/conv_mlp_tanh.pt` | ConvMLP-Tanh checkpoint (check exists) |
| `outputs/finalization/rebuttal/conv_mlp_relu.pt` | ConvMLP-ReLU checkpoint (check exists) |
| `outputs/tier1_trajectory_geometry/trajectory_geometry.json` | Existing Hessian trajectory data (5 models) |
| `outputs/landscape_geometry/lpips_evaluation.json` | Existing LPIPS (SqueezeNet) results |
| `outputs/fine_grid_kstar/fine_grid_kstar.json` | 110K KAN fine-grid canonical result |

---

## Editorial Changes Already Applied to paper_v4.tex

These are DONE — do not redo:
1. "Proposition" → "Empirical Law" (theorem environment renamed)
2. "iterative Langevin dynamics" → "iterative gradient descent"
3. DSM: added `\eps \sim \mathcal{N}(\mathbf{0}, I)` definition
4. Figure 2 caption: "only KAN-EBM" → "among the three models shown, only KAN-EBM"
5. Abstract Hessian claim: "six architectures" → "five tested architectures"
6. §4.2 Geometric correlate: "four ConvMLP heads" → "two representative ConvMLP variants (GELU, SiLU)"
7. Edge-deployment claim softened: no longer says "well suited to edge"
8. Cross-dataset sections: "establishing α as a property" → "suggesting α is relatively stable"
9. "Causal" language removed: "we report this as an empirical correlation, not a closed-form derivation; it does not establish causation"
10. Conclusion Hessian claim: "six architectures" → "five tested architectures"
11. Figure 3 (combined_powerlaw_all6.pdf): regenerated — now shows U-Net α=1.434 (was wrongly 1.198)
12. Cross-dataset section/table headers updated to "stability" rather than "transfer"

---

## Numeric Verification (all values correct in paper)

| Claim | JSON source | Value |
|-------|------------|-------|
| KAN-EBM 110K α=1.365 | `fine_grid_kstar/fine_grid_kstar.json` | 1.3651 ✓ |
| KAN-EBM 110K CI [1.325, 1.408] | same | ✓ |
| U-Net EBM α=1.434 | `fine_grid_kstar/fine_grid_unet.json` | 1.4343 ✓ |
| ConvMLP-GELU α=1.132 | `fine_grid_kstar/fine_grid_convmlp_gelu.json` | 1.1318 ✓ |
| CIFAR C=58.2 | `fine_grid_kstar/fine_grid_kstar.json` | 58.203 ✓ |
| Tiny-ImageNet α=1.3673 | `outputs/tier2_stl10/tinyimagenet_zeroshot.json` | 1.3673 ✓ |
| CelebA α=1.362 | `outputs/finalization/kstar_validation/kstar_validation.json` | 1.3620 ✓ |
