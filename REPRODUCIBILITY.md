# Reproducibility Checklist and Guide

> NeurIPS-style reproducibility documentation for the KAN-EBM paper.  
> All commands assume the repository root as the working directory.

---

## 1. Reproducibility Checklist

| # | Claim | Script / Location | Seed | Verified |
|---|-------|-------------------|------|----------|
| 1 | Table 1 — CelebA PSNR vs. inference steps K | `experiments/exp_celeba.py` | 42 | ✓ |
| 2 | Figure 2 — FLOP-normalized Pareto frontier | `experiments/rebuttal_pareto_v2.py` | 42 | ✓ |
| 3 | Figure 3 — K*(σ) scaling law & spectral prediction | `experiments/run_paper_finalization.py --only step5` | 42 | ✓ |
| 4 | Table 2 — ConvMLP ablation (smooth activations) | `experiments/rebuttal_smooth_mlp.py` + `rebuttal_smooth_mlp_analysis.py` | 42 | ✓ |
| 5 | Bootstrap CIs on power-law exponent α | `experiments/rebuttal_bootstrap_ci.py` | 20260501 | ✓ |
| 6 | Deblurring boundary — K* law beyond denoising | `experiments/run_paper_finalization.py --only step6` | 42 | ✓ |
| 7 | CIFAR-10 main results (PSNR vs. K) | `experiments/exp_cifar10.py` | 42 | ✓ |
| 8 | K* spectral theory (Hessian Lanczos) | `experiments/run_paper_finalization.py --only step2` | 42 | ✓ |
| 9 | CelebA fair MLP-EBM retrain | `experiments/run_paper_finalization.py --only step3` | 42 | ✓ |
| 10 | Limitations draft generation | `experiments/run_paper_finalization.py --only step4` | — | ✓ |

**Random seeds used across all experiments:**
- `SEED = 42` for all PyTorch and NumPy initialisations.
- `RNG = np.random.default_rng(20260501)` for bootstrap resampling in `rebuttal_bootstrap_ci.py`.

---

## 2. Hardware Requirements

| Experiment | Minimum GPU | VRAM | Notes |
|------------|-------------|------|-------|
| CIFAR-10 training (`exp_cifar10.py`) | 1× GPU | 8 GB | AMP enabled; batch 32 fits in ~6 GB |
| CelebA training (`exp_celeba.py`) | 1× GPU | 10 GB | Batch 8 for 64×64×3 with `create_graph=True` |
| K* sweep / spectral (`run_paper_finalization.py`) | 1× GPU | 8 GB | Lanczos HVP on single images |
| ConvMLP ablation (`rebuttal_smooth_mlp.py`) | 1× GPU | 8 GB | Four 40-epoch trainings sequentially |
| Bootstrap CI (`rebuttal_bootstrap_ci.py`) | CPU only | — | Pure NumPy; < 1 min on any CPU |
| Pareto figure (`rebuttal_pareto_v2.py`) | CPU only | — | Reads existing JSONs |

**Tested on:**
- NVIDIA RTX 3080 (10 GB) — Windows 11, CUDA 12.x
- NVIDIA A100 (40 GB) — Linux, CUDA 12.x

---

## 3. Expected Runtime (Full Runs)

| Script | Quick mode (`--quick`) | Full mode |
|--------|------------------------|-----------|
| `python experiments/exp_cifar10.py --device cuda` | ~8 min (5 epochs, 5K train) | ~90 min (50 epochs, 50K train) |
| `python experiments/exp_celeba.py --device cuda` | ~25 min (5 epochs, 5K train) | ~6 h (50 epochs, 30K train) |
| `python experiments/run_paper_finalization.py --device cuda --only step2` | ~10 min | ~45 min |
| `python experiments/run_paper_finalization.py --device cuda --only step3` | ~15 min (5K train, 10 epochs) | ~6 h (30K train, up to 200 epochs) |
| `python experiments/run_paper_finalization.py --device cuda --only step5` | ~20 min | ~2 h |
| `python experiments/run_paper_finalization.py --device cuda --only step6` | ~15 min | ~1 h |
| `python experiments/rebuttal_smooth_mlp.py --device cuda` | ~10 min (1 variant) | ~3 h (4 variants × 40 epochs) |
| `python experiments/rebuttal_smooth_mlp_analysis.py` | < 1 min | < 1 min |
| `python experiments/rebuttal_bootstrap_ci.py` | < 1 min | < 1 min |
| `python experiments/rebuttal_pareto_v2.py` | < 1 min | < 1 min |

> **Tip:** For a fast sanity check, append `--quick` to any script that supports it.

---

## 4. Pretrained Checkpoints

All checkpoints are auto-saved under `outputs/`.  The table below maps each claim to its expected checkpoint path.

| Claim | Checkpoint path | Size |
|-------|-----------------|------|
| CIFAR-10 KAN-EBM (Table 1 baseline) | `outputs/cifar10/kan_ebm_f32.pt` | ~456 KB |
| CIFAR-10 FFN-DSM | `outputs/cifar10/ffn_dsm_f32.pt` | ~14.7 MB |
| CelebA KAN-EBM | `outputs/celeba/kan_ebm.pt` | ~6.8 MB |
| CelebA fair MLP-EBM | `outputs/finalization/celeba_fair/mlp_ebm_fair.pt` | ~12.9 MB |
| ConvMLP ablation variants | `outputs/finalization/rebuttal/conv_mlp_{gelu,silu,tanh,relu}.pt` | ~130 KB each |

If a checkpoint is missing, the corresponding training script will regenerate it from scratch (deterministic given `SEED = 42`).

---

## 5. Exact Commands to Reproduce Each Table / Figure

### Table 1 — CelebA PSNR vs. K
```bash
python experiments/exp_celeba.py --device cuda
```
**Outputs:**
- `outputs/celeba/results.json`
- `outputs/celeba/psnr_vs_k.pdf`
- `outputs/celeba/table_celeba.tex`

---

### Figure 2 — Pareto Frontier (PSNR vs. Parameters / FLOPs)
```bash
# Prerequisites: CIFAR-10 and CelebA results must exist
python experiments/rebuttal_pareto_v2.py
```
**Outputs:**
- `outputs/finalization/rebuttal/pareto_honest.pdf`
- `outputs/finalization/rebuttal/pareto_honest.png`
- `outputs/finalization/rebuttal/pareto_honest.json`

---

### Figure 3 — K*(σ) Scaling Law
```bash
# Step 5 of finalization: K* sweep with cross-validation + power-law fit
python experiments/run_paper_finalization.py --device cuda --only step5
```
**Outputs:**
- `outputs/finalization/kstar_validation/kstar_cross_validation.pdf`
- `outputs/finalization/kstar_validation/kstar_cross_validation.png`
- `outputs/finalization/kstar_validation/kstar_validation.json`

---

### Table 2 — ConvMLP Ablation
```bash
# 1. Train smooth-activation MLP-EBM variants and run K* protocol
python experiments/rebuttal_smooth_mlp.py --device cuda

# 2. Post-hoc statistical analysis (bootstrap CIs, pairwise tests)
python experiments/rebuttal_smooth_mlp_analysis.py
```
**Outputs:**
- `outputs/finalization/rebuttal/smooth_mlp_ablation.json`
- `outputs/finalization/rebuttal/smooth_mlp_stats.json`
- `outputs/finalization/rebuttal/smooth_mlp_stats_summary.txt`

---

### Bootstrap Confidence Intervals on α
```bash
# Requires step 5 output (kstar_validation.json)
python experiments/rebuttal_bootstrap_ci.py
```
**Outputs:**
- `outputs/finalization/rebuttal/bootstrap_ci.json`
- `outputs/finalization/rebuttal/bootstrap_ci_summary.txt`

---

### Deblurring Boundary (K* beyond denoising)
```bash
# Step 6 of finalization
python experiments/run_paper_finalization.py --device cuda --only step6
```
**Outputs:**
- `outputs/finalization/kstar_deblur/kstar_deblur_fit.pdf`
- `outputs/finalization/kstar_deblur/kstar_deblur_fit.png`
- `outputs/finalization/kstar_deblur/kstar_deblur.json`

---

## 6. Full Paper Reproduction (One-Shot)

To reproduce all results in a single overnight run:

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. CIFAR-10 main results
python experiments/exp_cifar10.py --device cuda

# 3. CelebA main results
python experiments/exp_celeba.py --device cuda

# 4. FLOP benchmark (prerequisite for Pareto figure)
python experiments/exp_flop_benchmark.py --device cuda

# 5. Finalization pipeline (steps 1–9)
python experiments/run_paper_finalization.py --device cuda

# 6. Rebuttal experiments
python experiments/rebuttal_smooth_mlp.py --device cuda
python experiments/rebuttal_smooth_mlp_analysis.py
python experiments/rebuttal_bootstrap_ci.py
python experiments/rebuttal_pareto_v2.py
```

Expected wall-clock time on an RTX 3080: **~14–18 hours**.

---

## 7. Environment

- **OS:** Windows 11 / Linux (Ubuntu 22.04 tested)
- **Python:** 3.11+
- **CUDA:** 12.x
- **PyTorch:** 2.10.0 (see `requirements.txt` for exact pinned versions)

---

## 8. Data Availability

All datasets are downloaded automatically by `torchvision` on first run:
- CIFAR-10 → `data/cifar-10-batches-py/`
- CelebA → `data/celeba/` (requires manual download if auto-download fails; see `exp_celeba.py` docstring)

---

## 9. Contact

For reproducibility issues, please open a GitHub issue referencing the specific script and step number from this checklist.
