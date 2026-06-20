# CODE AUDIT REPORT — Theory of Everything (ToE) Project

**Auditor:** Code_Auditor  
**Date:** 2026-01-13  
**Scope:** 15 key Python files across `experiments/` and `theory/`  
**Objective:** Ensure reproducibility, correctness, and fairness for AAAI paper experiments.

---

## EXECUTIVE SUMMARY

| Severity | Count | Summary |
|----------|-------|---------|
| **CRITICAL** | 6 | Bugs that will crash, produce wrong results, or make experiments irreproducible |
| **HIGH** | 4 | Issues that significantly compromise fairness, correctness, or reproducibility |
| **MEDIUM** | 9 | Inefficiencies, inconsistencies, or missing robustness checks |
| **LOW** | 7 | Minor code quality issues, documentation mismatches, or style concerns |

**Top 3 Action Items:**
1. Fix `exp_multitask.py` `corrupt_inpaint` seed bug (identical masks every call).
2. Fix missing `dt_decay` in `exp_diffusion_comparison.py` KAN-EBM wrapper (unfair comparison).
3. Add `weights_only=True` and global seeds to `exp_diffusion_cifar10.py` and `exp_multitask.py`.

---

## CRITICAL ISSUES

### C1. `exp_multitask.py` — `corrupt_inpaint` resets global PyTorch seed, producing identical masks across all calls
**File:** `experiments/exp_multitask.py` (lines 229–244)  
**Severity:** CRITICAL

```python
def corrupt_inpaint(x_clean, mask_ratio=0.3, seed=0):
    ...
    torch.manual_seed(seed)   # <-- resets GLOBAL seed to 0
    mask = (torch.rand(B, 1, H, W, device=x_clean.device) > mask_ratio).float()
```

**Problem:** Every call to `corrupt_inpaint()` resets the global PyTorch RNG to seed 0, producing the **exact same mask pattern** for every image and every batch. This means the inpainting evaluation is not stochastic—it tests the same mask repeatedly. The `torch.manual_seed` inside a utility function is an anti-pattern that corrupts the global RNG state for all downstream code.

**Fix:** Remove the global seed reset. Use a local `Generator` or pass `torch.rand` with an explicit `generator` argument:
```python
g = torch.Generator(device=x_clean.device).manual_seed(seed)
mask = (torch.rand(B, 1, H, W, generator=g, device=x_clean.device) > mask_ratio).float()
```

---

### C2. `exp_multitask.py` — `KANEnergyModel` missing precision scaling and `tanh` squashing
**File:** `experiments/exp_multitask.py` (lines 119–144)  
**Severity:** CRITICAL

**Problem:** The `KANEnergyModel` in `exp_multitask.py` does **not** apply the learnable `precision` or `tanh` feature squashing that is present in the canonical `exp_cifar10.py` / `exp_celeba.py` versions:

```python
# exp_cifar10.py (correct)
prec  = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
feats = torch.tanh(feats * prec)

# exp_multitask.py (missing — returns raw conv features)
feats = torch.cat(feats, 1)
return feats.permute(0,2,3,1).contiguous().view(B*H*W, -1)
```

This means the multitask model is **architecturally different** from the CIFAR-10/CelebA models. The paper cannot claim "the same model on three tasks" if the model architecture is silently different. The missing `tanh` also means KAN B-spline inputs can fall outside the `[-1,1]` grid range, producing undefined spline behavior.

**Fix:** Add the precision and `tanh` block exactly as in `exp_cifar10.py`.

---

### C3. `exp_diffusion_comparison.py` — `KANEBMWrapper` uses FIXED step size, NO `dt_decay`
**File:** `experiments/exp_diffusion_comparison.py` (lines 234–242)  
**Severity:** CRITICAL

```python
def denoise(self, x_noisy, n_steps=10, dt=0.05, sigma=None):
    u = x_noisy.clone()
    for _ in range(n_steps):
        ...
        u = u.detach() - dt * g   # <-- dt is constant, no decay
    return u
```

**Problem:** The main EBM models (`exp_cifar10.py`, `exp_celeba.py`, `exp_unet_ebm.py`) all use the **exponentially decaying** schedule `dt *= 0.97` per step. The `KANEBMWrapper` in the diffusion comparison uses a **fixed** `dt=0.05`. This makes the KAN-EBM comparison unfair—at large K, the KAN-EBM in this script will overshoot because it lacks the step-size decay that stabilizes convergence in the canonical implementation.

**Fix:** Add `dt_decay=0.97` and shrink `dt` each iteration, exactly as in `exp_cifar10.py`.

---

### C4. `exp_multitask.py` — checkpoint loading missing `weights_only=True`
**File:** `experiments/exp_multitask.py` (line 540)  
**Severity:** CRITICAL

```python
kan.load_state_dict(torch.load(args.checkpoint, map_location=device))
```

**Problem:** Loading without `weights_only=True` is a **security vulnerability** (arbitrary code execution via pickled checkpoints) and can crash if the checkpoint was saved with different PyTorch / Python versions. PyTorch 2.0+ raises a `FutureWarning` about this.

**Fix:**
```python
kan.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
```

---

### C5. `exp_diffusion_cifar10.py` — NO stochastic seeds set anywhere
**File:** `experiments/exp_diffusion_cifar10.py`  
**Severity:** CRITICAL

**Problem:** The entire script contains **no** `torch.manual_seed()`, `np.random.seed()`, or `random.seed()`. Every training and evaluation run will produce **different** weights, noise realizations, and PSNR values. This makes the diffusion baseline results **non-reproducible** across runs.

**Fix:** Add at the top of `main()`:
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

### C6. `exp_diffusion_cifar10.py` — K=1 evaluation uses different step size than K>1
**File:** `experiments/exp_diffusion_cifar10.py` (lines 171–176)  
**Severity:** CRITICAL

```python
if K == 1:
    noise_pred = model(xn, ...)
    pred = (xn - noise_pred).clamp(-1, 1)          # step size = 1.0 implicitly
else:
    pred = iterative_score_denoise(model, xn, K, sigma, device)  # step size = 0.5 * (1 - step/K)
```

**Problem:** K=1 is treated as a special case that subtracts the **full** predicted noise (`step size = 1.0`), while the iterative path for K=1 would use `step size = 0.5`. This means K=1 and K=2+ are not on the same step-size schedule, making the K-scaling curve internally inconsistent.

**Fix:** Remove the K=1 special case and route all K through `iterative_score_denoise` with a consistent base step size.

---

## HIGH SEVERITY ISSUES

### H1. `exp_cifar10.py`, `exp_celeba.py`, `exp_diffusion_comparison.py` — `evaluate()` computes batch PSNR, not per-image PSNR
**Files:** `experiments/exp_cifar10.py` (line 331), `experiments/exp_celeba.py` (line 339), `experiments/exp_diffusion_comparison.py` (line 305)  
**Severity:** HIGH

```python
def psnr(clean, pred, data_range=2.0):
    pred = pred.clamp(-1, 1)
    mse  = F.mse_loss(pred, clean).item()   # <-- MSE over ENTIRE batch
    return 10.0 * math.log10(data_range ** 2 / mse)
```

**Problem:** `F.mse_loss(pred, clean)` computes the mean squared error over **all pixels in the batch**. Then PSNR is computed from this batch-level MSE. The `evaluate()` function averages these batch PSNRs:

```python
results[k].append(psnr(xc, xp))   # one PSNR per batch
return {k: float(np.mean(v)) for k, v in results.items()}  # average of batch PSNRs
```

Because PSNR is a **non-linear** function of MSE, `mean(PSNR_batch) ≠ PSNR(mean(MSE_batch))`. The correct method (used in `tier0_seeds.py`) is to compute per-image MSE, then per-image PSNR, then average. Additionally, if the last batch is smaller (e.g., 2000 test images with batch size 32 → last batch has 16 images), the smaller batch receives the same weight as a full batch, biasing the average.

**Impact:** Reported PSNR values are slightly off, and the K=1 vs K=50 "Gain" metric may be biased.

**Fix:** Use per-image MSE/PSNR as in `tier0_seeds.py`:
```python
def psnr_per_image(clean, pred, data_range=2.0):
    mse = F.mse_loss(pred.clamp(-1,1), clean, reduction='none').mean(dim=[1,2,3])
    return (10 * torch.log10(data_range**2 / mse)).cpu().numpy()
```

---

### H2. `exp_diffusion_cifar10.py` — diffusion baseline uses completely different step-size schedule than EBM
**File:** `experiments/exp_diffusion_cifar10.py` (lines 142–156)  
**Severity:** HIGH

```python
def iterative_score_denoise(model, noisy, K, sigma, device):
    x = noisy.clone()
    base_eta = 0.5
    for step in range(K):
        eta = base_eta * (1.0 - step / max(K, 1))   # <-- linear decay
        ...
        x = (x - eta * noise_pred).clamp(-1, 1)
```

**Problem:** The EBM baseline uses **exponential decay** (`dt *= 0.97` per step). The diffusion baseline uses **linear decay** (`eta = 0.5 * (1 - step/K)`). These are fundamentally different schedules. A fair comparison requires the same schedule family.

**Fix:** Change the diffusion baseline to use `dt=0.05, dt_decay=0.97` to match the EBM protocol, or document the schedule mismatch explicitly in the paper.

---

### H3. Most experiment files — missing full deterministic seeding for reproducibility
**Files:** `exp_cifar10.py`, `exp_celeba.py`, `exp_denoise.py`, `exp_diffusion_comparison.py`, `exp_multitask.py`, `exp_unet_ebm.py`, `tier0_seeds.py`, `phase_b_diffusion.py`, `beta_sweep.py`, `corruption_phase_diagram.py`  
**Severity:** HIGH

**Problem:** While `torch.manual_seed()` and `np.random.seed()` are set in most files, the following are **missing** across the board:
- `random.seed(SEED)` — needed for Python stdlib randomness (e.g., `np.random.choice` uses numpy's RNG, but some libraries use Python's `random`)
- `torch.cuda.manual_seed_all(SEED)` — needed for CUDA kernel randomness
- `torch.backends.cudnn.deterministic = True` — needed to disable non-deterministic cudnn algorithms
- `torch.backends.cudnn.benchmark = False` — needed to disable autotuning that can vary across runs

**Impact:** Even with `torch.manual_seed(42)`, different PyTorch / CUDA versions, GPU drivers, or hardware can produce different convolution backward passes, leading to non-reproducible training curves and final PSNR values. This is a known PyTorch issue documented at https://pytorch.org/docs/stable/notes/randomness.html.

**Fix:** Add the full deterministic seed block to every training script.

---

### H4. `exp_diffusion_comparison.py` — potential crash in `ScoreNetUNet._sigma_idx` due to `.expand(1)` on 0D tensor
**File:** `experiments/exp_diffusion_comparison.py` (line 132)  
**Severity:** HIGH

```python
def _sigma_idx(self, sigma: float) -> torch.Tensor:
    diffs = (self.sigma_values - sigma).abs()
    return diffs.argmin().expand(1)   # <-- argmin() returns 0D tensor; .expand(1) may crash
```

**Problem:** `diffs.argmin()` returns a 0-dimensional (scalar) tensor. In PyTorch, `.expand(1)` on a 0D tensor raises `RuntimeError: expand(...) called on a tensor with 0 dimensions`. The caller then does `.expand(x_noisy.shape[0])`, which would also fail on a 0D tensor. If this code has not been tested on a batch size of 1, it may crash unexpectedly.

**Fix:**
```python
return diffs.argmin().unsqueeze(0)   # or .view(1)
```

---

## MEDIUM SEVERITY ISSUES

### M1. `exp_multitask.py` — `corrupt_denoise` clamps to `[-2, 2]` instead of `[-1, 1]`
**File:** `experiments/exp_multitask.py` (line 211)  
**Severity:** MEDIUM

```python
return (x_clean + noise).clamp(-2., 2.), None, None
```

**Problem:** The data is normalized to `[-1, 1]`. Clamping to `[-2, 2]` is harmless (noise with σ=0.2 almost never exceeds ±1), but it is inconsistent with the data range convention and the clamping used in other files (`clamp(-1, 1)`).

**Fix:** Use `clamp(-1, 1)` for consistency.

---

### M2. `exp_denoise.py` — `HamiltonianField` and `MLPField` have mismatched regularization
**File:** `experiments/exp_denoise.py` (via `src/hamiltonian_field.py` and `experiments/mlp_field.py`)  
**Severity:** MEDIUM

**Problem:** `HamiltonianField.compute_energy` uses a hardcoded `reg = 0.01 * (fields ** 2).mean(...)`, while `MLPField.compute_energy` uses a learned `reg = self.reg_weight.abs() * (fields ** 2).sum(...)`. The regularization strength and type (`mean` vs `sum`) differ, making the energy landscapes structurally different in a way that is not controlled.

**Fix:** Use the same regularization formulation (learned weight + sum or mean) in both models for a fair ablation.

---

### M3. `exp_diffusion_comparison.py` — DDPM uses stochastic Langevin dynamics, EBM uses deterministic GD
**File:** `experiments/exp_diffusion_comparison.py` (lines 146–153)  
**Severity:** MEDIUM

**Problem:** `DDPMDenoiser.denoise` injects Gaussian noise at every step (`noise = torch.randn_like(u) * math.sqrt(step_size)`), while `KANEBMWrapper.denoise` is pure deterministic gradient descent. The comparison is between a stochastic sampler and a deterministic optimizer, which is a methodological mismatch.

**Fix:** Either (a) remove the Langevin noise from DDPM to make it deterministic score descent, or (b) add noise to the EBM denoiser to match Langevin dynamics, or (c) explicitly document the stochastic vs deterministic distinction in the paper.

---

### M4. `exp_cifar10.py` — output filename mismatch with docstring
**File:** `experiments/exp_cifar10.py` (lines 27–28, 437)  
**Severity:** MEDIUM

**Problem:** The docstring promises `outputs/cifar10/results.json`, but the actual code writes `outputs/cifar10/results_f{args.n_filters}.json`. This is a documentation bug that can confuse users and automated pipelines.

**Fix:** Update the docstring or write to the documented filename.

---

### M5. `rebuttal_smooth_mlp.py` and `exp_unet_ebm.py` — `fit_power_law` lacks numerical robustness
**Files:** `experiments/rebuttal_smooth_mlp.py` (lines 160–168), `experiments/exp_unet_ebm.py` (lines 203–211)  
**Severity:** MEDIUM

```python
def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
```

**Problem:** No check for `K_means <= 0` (which would make `np.log` return `-inf` or `nan`). No check for `len(sigmas) < 2` or all-same values. If the model fails to produce a valid K* curve, `np.polyfit` will crash or produce garbage.

**Fix:** Add guards as in `tier0_seeds.py`:
```python
ok = K_means > 0
if ok.sum() < 4 or np.std(K_means[ok]) < 1e-6:
    return float('nan'), float('nan'), float('nan')
xs = np.log(np.asarray(sigmas[ok], dtype=float))
ys = np.log(np.asarray(K_means[ok], dtype=float))
```

---

### M6. `exp_cifar10.py`, `exp_celeba.py` — `pin_memory=True` with `num_workers=0` is a no-op
**File:** `experiments/exp_cifar10.py` (lines 274–277), `experiments/exp_celeba.py` (lines 283–286)  
**Severity:** MEDIUM

**Problem:** `pin_memory=True` only has an effect when `num_workers > 0`. Setting `num_workers=0` means data loading is synchronous and `pin_memory` does nothing. This is a minor inefficiency.

**Fix:** Either set `num_workers=2` (standard) or remove `pin_memory=True` to reduce confusion.

---

### M7. `exp_diffusion_cifar10.py` — inefficient per-sample noise scaling loop
**File:** `experiments/exp_diffusion_cifar10.py` (lines 126–128)  
**Severity:** MEDIUM

```python
noise = torch.randn_like(x)
for i in range(x.size(0)):
    noise[i] *= sigmas[i]
```

**Problem:** This loop is O(B) and can be replaced with a single vectorized operation:
```python
noise = noise * sigmas.view(-1, 1, 1, 1)
```

---

### M8. `exp_diffusion_cifar10.py` — SSIM computed per-sample in Python loop
**File:** `experiments/exp_diffusion_cifar10.py` (lines 180–182)  
**Severity:** MEDIUM

```python
for i in range(pred.shape[0]):
    s = compute_ssim(x[i:i+1], pred[i:i+1], data_range=2.0)
    ssims.append(s)
```

**Problem:** `compute_ssim` is a pure PyTorch function that can operate on a batch. Looping over samples in Python is slow and unnecessary.

**Fix:** Call `compute_ssim` on the full batch directly, or use a vectorized SSIM implementation.

---

### M9. `kstar_spectral_model.py` — `fit_powerlaw` has no empty-mask guard
**File:** `theory/kstar_spectral_model.py` (lines 98–109)  
**Severity:** MEDIUM

```python
def fit_powerlaw(sigmas, kstars):
    mask = kstars > 0
    x = np.log(sigmas[mask])
    y = np.log(kstars[mask])
    ...
```

**Problem:** If all `kstars` are 0, `mask` is empty, and `np.log` on an empty array produces a warning. `np.linalg.lstsq` will then raise a `LinAlgError`.

**Fix:** Add an early return for `mask.sum() < 2` or `np.std(kstars[mask]) < 1e-6`.

---

## LOW SEVERITY ISSUES

### L1. `exp_multitask.py` — uses deprecated PyTorch AMP API
**File:** `experiments/exp_multitask.py` (line 268)  
**Severity:** LOW

```python
scaler  = torch.cuda.amp.GradScaler() if use_amp else None
```

**Problem:** `torch.cuda.amp.GradScaler()` is deprecated in PyTorch 2.0+ in favor of `torch.amp.GradScaler('cuda')`. This may generate a deprecation warning.

**Fix:** Use `torch.amp.GradScaler('cuda')` as done in `exp_cifar10.py`.

---

### L2. `exp_cifar10.py` — hardcoded `3` in `kan_hidden` computation
**File:** `experiments/exp_cifar10.py` (line 379)  
**Severity:** LOW

```python
kan_hidden = [args.n_filters * 3, 16]   # <-- magic number 3
```

**Problem:** The `3` is the number of RGB channels. If this code is ever adapted to grayscale or multi-spectral images, the hidden dimension will be wrong. It should use `n_channels` or a configurable parameter.

**Fix:** `kan_hidden = [args.n_filters * n_channels, 16]` where `n_channels` is passed explicitly.

---

### L3. `predictive_alpha.py` — `np.std` with `ddof=1` on single-element array returns `nan`
**File:** `theory/predictive_alpha.py` (line 140)  
**Severity:** LOW

```python
g_std = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else 0
```

**Problem:** The guard `if len(g_vals) > 1` handles the single-element case, but if `g_vals` has exactly 2 elements, `np.std` with `ddof=1` is well-defined. The code is actually correct. I withdraw this concern. However, the downstream `g_cv` computation:
```python
g_cv = g_std / g_mean if g_mean > 0 else float('nan')
```
This could be `inf` if `g_mean` is very small but positive. A more robust guard would be `abs(g_mean) > 1e-8`.

---

### L4. `measure_beta.py` — `load_images` may silently drop images for CBSD68
**File:** `theory/measure_beta.py` (line 156)  
**Severity:** LOW

```python
n = min(n, len(ds))
return torch.stack([ds[i][0] for i in range(n)])
```

**Problem:** For `cbsd68`, `n` defaults to 2000, but CBSD68 only has 68 images. The code silently returns only 68 images. This is fine for the function's purpose, but it could be confusing if the user expects 2000.

**Fix:** Add a warning or print when `n > len(ds)`.

---

### L5. `exp_cifar10.py`, `exp_celeba.py` — module-level seed setting affects imports
**File:** `experiments/exp_cifar10.py` (lines 51–53)  
**Severity:** LOW

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

**Problem:** Setting the global seed at module import time is risky. If another module imports `exp_cifar10`, the global RNG is silently reset. This can cause hard-to-debug reproducibility issues in interactive notebooks or complex pipelines.

**Fix:** Move seed setting inside the `main()` function or behind an `if __name__ == '__main__':` guard.

---

### L6. `corruption_phase_diagram.py` — JPEG corruption is extremely slow
**File:** `theory/corruption_phase_diagram.py` (lines 63–73)  
**Severity:** LOW

```python
def c_jpeg(x, q):
    for i in range(x01.shape[0]):
        arr = (x01[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=int(q))
        ...
```

**Problem:** JPEG encoding/decoding is done in a Python loop with PIL, which is very slow for large batches. This is fine for the intended use (N=40 images), but it could be a bottleneck if scaled up.

**Fix:** Vectorize with `torchjpeg` or `kornia` if scaling to larger N.

---

### L7. `exp_denoise.py` — `noise_std` is a float, but `np.random.choice` is used in `exp_unet_ebm.py`
**File:** `experiments/exp_denoise.py` (line 148) vs `experiments/exp_unet_ebm.py` (line 160)  
**Severity:** LOW

**Problem:** `exp_denoise.py` uses a fixed `noise_std` per training run, while `exp_unet_ebm.py` samples a random `sigma` per batch from a list. This inconsistency is documented but makes cross-file comparisons harder.

**Fix:** Document the difference clearly or unify the training protocol.

---

## CRITICAL CHECKS SUMMARY

### Check 1: K-sweep step-size schedule consistency (`dt=0.05`, `decay=0.97`)
| File | Schedule | Status |
|------|----------|--------|
| `exp_cifar10.py` | `dt=0.05, dt_decay=0.97` | ✅ |
| `exp_celeba.py` | `dt=0.05, dt_decay=0.97` | ✅ |
| `exp_multitask.py` | `dt=0.05, dt_decay=0.97` | ✅ |
| `exp_unet_ebm.py` | `dt=0.05, dt_decay=0.97` | ✅ |
| `tier0_seeds.py` | `dt=0.05, decay=0.97` | ✅ |
| `phase_b_diffusion.py` | `dt=0.05, decay=0.97` | ✅ |
| `corruption_phase_diagram.py` | `dt=0.05, decay=0.97` | ✅ |
| `measure_kappa.py` | `dt=0.05, decay=0.97` | ✅ |
| `exp_diffusion_cifar10.py` | `base_eta=0.5, linear decay` | ❌ **MISMATCH** |
| `exp_diffusion_comparison.py` (KAN-EBM) | `dt=0.05, NO decay` | ❌ **MISMATCH** |
| `exp_diffusion_comparison.py` (DDPM) | `sigma**2/10.0, fixed` | ❌ **MISMATCH** |

### Check 2: Seeds for all stochastic operations
| File | `torch` | `numpy` | `random` | `cuda` | `cudnn` | Status |
|------|---------|---------|----------|--------|---------|--------|
| `exp_cifar10.py` | ✅ | ✅ | ❌ | ❌ | ❌ | Partial |
| `exp_celeba.py` | ✅ | ✅ | ❌ | ❌ | ❌ | Partial |
| `exp_multitask.py` | ✅ | ✅ | ❌ | ❌ | ❌ | Partial |
| `exp_denoise.py` | ✅ | ✅ | ❌ | ❌ | ❌ | Partial |
| `exp_diffusion_cifar10.py` | ❌ | ❌ | ❌ | ❌ | ❌ | **MISSING** |
| `exp_diffusion_comparison.py` | ✅ | ✅ | ❌ | ❌ | ❌ | Partial |
| `exp_unet_ebm.py` | ✅ | ✅ | ❌ | ❌ | ❌ | Partial |
| `tier0_seeds.py` | ✅ (per run) | ✅ (per run) | ❌ | ❌ | ❌ | Partial |
| `phase_b_diffusion.py` | ✅ (per run) | ✅ (per run) | ❌ | ❌ | ❌ | Partial |

### Check 3: Checkpoint loading (`weights_only=True`, `map_location`)
| File | `map_location` | `weights_only` | Status |
|------|----------------|----------------|--------|
| `exp_multitask.py` | ✅ | ❌ | **UNSAFE** |
| `corruption_phase_diagram.py` | ✅ | ✅ | ✅ |
| `measure_kappa.py` | ✅ | ✅ | ✅ |
| `exp_unet_ebm.py` | ✅ | ✅ | ✅ |
| `exp_diffusion_cifar10.py` | N/A (saves, doesn't load) | N/A | N/A |

### Check 4: PSNR computation consistency (`data_range=2.0`, clamping, MSE formula)
| File | Clamping | MSE | PSNR Formula | Per-image? | Status |
|------|----------|-----|--------------|------------|--------|
| `exp_cifar10.py` | ✅ | `F.mse_loss` | `10*log10(4/mse)` | ❌ (batch) | **BIASED** |
| `exp_celeba.py` | ✅ | `F.mse_loss` | `10*log10(4/mse)` | ❌ (batch) | **BIASED** |
| `exp_multitask.py` | N/A | `((pred-target)**2).mean()` | `10*log10(4/mse)` | ✅ (single) | ✅ |
| `exp_denoise.py` | ✅ | `F.mse_loss` | `10*log10(4/mse)` | ✅ (per-image) | ✅ |
| `exp_diffusion_cifar10.py` | ✅ | `F.mse_loss(reduction='none')` | `10*log10(4/mse)` | ✅ (per-image) | ✅ |
| `exp_diffusion_comparison.py` | ✅ | `F.mse_loss` | `10*log10(4/mse)` | ❌ (batch) | **BIASED** |
| `tier0_seeds.py` | ✅ | `F.mse_loss(reduction='none')` | `10*log10(4/mse)` | ✅ (per-image) | ✅ |
| `metrics.py` | N/A | `F.mse_loss` | `10*log10(4/mse)` | N/A | ✅ |

### Check 5: Off-by-one errors in K* argmax
| File | K=0 included? | Argmax pattern | Status |
|------|-------------|----------------|--------|
| `tier0_seeds.py` | ✅ (series[0] = no step) | `arr.argmax(axis=0)` on `(K+1, N)` | ✅ |
| `phase_b_diffusion.py` | ✅ (series[0] = no step) | `arr.argmax(axis=0)` on `(K+1, N)` | ✅ |
| `corruption_phase_diagram.py` | ✅ (series[0] = no step) | `np.argmax(series)` on `(K+1,)` | ✅ |
| `kstar_spectral_model.py` | ✅ (a[0] = ones) | `np.argmin(err)` on `(K+1,)` | ✅ |
| `exp_unet_ebm.py` | ❌ (starts at K=1) | `np.argmax(...) + 1` on `(K_max,)` | ✅ (different convention, documented) |
| `rebuttal_smooth_mlp.py` | ❌ (K_list starts at 1) | `max(psnrs.items())` | ✅ (acceptable) |

### Check 6: Corruption functions mathematical correctness
| File | Gaussian | Speckle | Poisson | Blur | JPEG | Inpaint | Status |
|------|----------|---------|---------|------|------|---------|--------|
| `corruption_phase_diagram.py` | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | ✅ |
| `exp_multitask.py` | ✅ (but clamp to ±2) | N/A | N/A | N/A | N/A | ❌ (seed bug) | **BUG** |
| `exp_denoise.py` | ✅ | N/A | N/A | N/A | N/A | N/A | ✅ |

### Check 7: Power-law fit numerical stability
| File | `kstars > 0` check | `std > 0` check | `sigma > 0` check | Empty guard | Status |
|------|--------------------|-----------------|-------------------|-------------|--------|
| `tier0_seeds.py` | ✅ | ✅ | ✅ (implicit) | ✅ | ✅ |
| `corruption_phase_diagram.py` | ✅ | ✅ | ✅ (implicit) | ✅ | ✅ |
| `measure_kappa.py` | ✅ (lam > 0) | N/A | ✅ (s2 clamped) | ✅ | ✅ |
| `measure_beta.py` | N/A (P>0) | N/A | ✅ (f>0) | ✅ | ✅ |
| `rebuttal_smooth_mlp.py` | ❌ | ❌ | ❌ | ❌ (caller guards) | **WEAK** |
| `exp_unet_ebm.py` | ❌ | ❌ | ❌ | ❌ | **WEAK** |
| `kstar_spectral_model.py` | ✅ (mask > 0) | ❌ | ❌ | ❌ | **WEAK** |

### Check 8: Diffusion baseline training objective and denoising step
| File | Objective | Prediction target | Denoising step | Status |
|------|-----------|-------------------|----------------|--------|
| `exp_diffusion_cifar10.py` | `F.mse_loss(pred, noise)` | Epsilon (noise) | `x = x - eta * noise_pred` | ✅ (correct for epsilon-prediction) |
| `exp_diffusion_comparison.py` | `sigma^2 * F.mse_loss(pred_score, true_score)` | Score | `x = x + (step/2) * score + noise` | ✅ (correct for score-matching Langevin) |
| `exp_diffusion_cifar10.py` | K=1 step size | K=1 uses η=1.0 | K>1 uses η=0.5*(1-step/K) | ❌ **INCONSISTENT** |

---

## MODEL FILE AUDIT

### `src/hamiltonian_field.py`
- **Bugs:** `evolve()` calls `self.compute_energy_from_image(u_input)` which is inherited from `ThermodynamicField`. This is correct. No obvious crashes.
- **Issues:** `evolve()` does not clamp outputs (unlike `ThermodynamicField.evolve()` which clamps to `[-1.5, 1.5]`). For unbounded KAN energy, this could cause divergence during long inference. **MEDIUM**.
- **Inefficiencies:** The `compute_energy` method allocates an intermediate `energy` tensor of shape `(B,)` and accumulates in a loop. This is fine.

### `src/predictive_coding_field.py`
- **Bugs:** `compute_energy` returns a scalar `energy_density.sum()` regardless of batch size. This is correct for gradient computation but lacks per-sample energy tracking. **LOW**.
- **Issues:** `denoising_loss` computes `grad_reg = 0.01 * (energy_grad ** 2).mean()` with `create_graph=True` on the gradient. This is a double backward and may be memory-intensive for large batches. **LOW**.
- **Inefficiencies:** `local_weight_update` computes `delta_filt` using `F.conv2d` with a permuted tensor. The permutation is unnecessary if the tensor is already in the right format. **LOW**.

### `src/thermodynamic_field.py`
- **Bugs:** `initialize_fields` for multi-channel input uses `self.input_proj_weight[:, :min(n_channels, 9)]`. If `C_in > 9`, only the first 9 channels are used. This is a hardcoded limit that could silently drop information for high-channel inputs. **MEDIUM**.
- **Issues:** `compute_energy` normalizes by `energy = energy / (H * W)`. This is a mean energy per pixel, which is fine. But for different image sizes, the energy scale changes. **LOW**.

### `experiments/mlp_field.py`
- **Bugs:** `compute_energy` returns `energy + reg` where `energy` is `(B,)` and `reg` is also `(B,)`. Shapes match. No crash.
- **Issues:** `self.reg_weight.abs()` is redundant because `reg_weight` is initialized positive and never constrained to be negative. But `.abs()` ensures positivity. **LOW**.

### `experiments/rebuttal_smooth_mlp.py`
- **Bugs:** `extract_features` uses `self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)` but `self.precision` is `(n_filters,)`. This is correct for broadcasting. No crash.
- **Issues:** `sweep_kstar_seeded` computes `kstar` from `max(psnrs.items(), key=lambda kv: kv[1])[0]`. If two K values have the same PSNR, the smaller K wins (first occurrence). This is arbitrary but documented. **LOW**.

---

## RECOMMENDATIONS

1. **Immediate (before any AAAI submission):**
   - Fix C1 (`exp_multitask.py` seed bug) and C2 (missing precision/tanh).
   - Fix C3 (`dt_decay` missing in comparison wrapper) and C5 (missing seeds in diffusion baseline).
   - Fix H1 (per-image PSNR) in `exp_cifar10.py`, `exp_celeba.py`, and `exp_diffusion_comparison.py`.
   - Add `weights_only=True` to all `torch.load` calls.

2. **Before camera-ready:**
   - Add full deterministic seeding (`random`, `cuda`, `cudnn`) to all training scripts.
   - Fix the K=1 step-size inconsistency in `exp_diffusion_cifar10.py`.
   - Add robustness guards to all `fit_power_law` functions.
   - Unify `num_workers` and `pin_memory` settings across DataLoaders.

3. **Documentation:**
   - Explicitly document the step-size schedule for each baseline in the paper appendix.
   - Document the stochastic (Langevin) vs deterministic (GD) distinction in the DDPM comparison.

---

*End of Audit Report.*
