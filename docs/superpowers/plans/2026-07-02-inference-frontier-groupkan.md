# Inference Frontier + GroupKAN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a lighter energy head running more descent steps beats a heavier head running fewer steps at matched inference FLOPs (pre-registered criterion), add a GR-KAN-style GroupKAN energy head, and write the result into `paper_v8.tex` — per the approved spec `docs/superpowers/specs/2026-07-02-inference-frontier-groupkan-design.md`.

**Architecture:** Three new experiment scripts reuse the repo's existing EBM classes, checkpoint loaders, and fine-grid K* protocol. FLOPs are measured with `torch.utils.flop_counter.FlopCounterMode` (one energy forward + input-gradient backward = one descent step). Analysis (curve building, pass criterion) is pure-numpy and unit-tested. GroupKAN is a new self-contained model in `src/group_kan.py` mirroring `ConvSmoothMLPEBM`'s structure.

**Tech Stack:** PyTorch (≥2.1 for `flop_counter`), torchvision CIFAR-10, numpy, matplotlib (Agg), pytest. Python launcher: `py -3.12`.

## Global Constraints

- One torch process at a time. Heavy runs: RTX 4090 via SSH; fallback RTX 3080 local; smoke runs CPU-safe.
- Data domain `[-1, 1]`; PSNR = `10*log10(4.0/mse)` (data_range 2.0), matching `experiments/exp_fine_grid_kstar.py`.
- Descent protocol (never change): `dt=0.05`, `dt_decay=0.97`, gradient clamp `(-1, 1)`, exactly as `sequential_denoise_record` in `experiments/exp_fine_grid_kstar.py:50`.
- Frontier sweep: σ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}, K = 0…30, 500 CIFAR-10 test images, seeds {0, 1}, batch 100.
- Pre-registered pass criterion (spec §2, verbatim): PASS if a lighter head beats a heavier head by ≥0.3 dB PSNR at matched total FLOPs over a contiguous budget range, consistently across ≥3 of 5 σ values and both seeds. Primary pair: KAN-32K (light) vs KAN-110K (heavy). Lighter/heavier requires strictly fewer params AND per-step FLOPs.
- Matched training recipe for GroupKAN (= `experiments/rebuttal_smooth_mlp.py` recipe that produced `conv_mlp_gelu.pt`): AdamW lr=3e-4 wd=1e-4, CosineAnnealingLR T_max=40, 40 epochs, 20,000-image CIFAR-10 train subset, batch 128, `sigma_train=0.15`, grad-clip 1.0, AMP on CUDA.
- Resumability: every (model, σ, seed) cell writes `outputs/inference_frontier/parts/{model}_s{sigma}_seed{seed}.json`; a cell is skipped if its file exists and contains `"done": true`.
- Windows console: ASCII only in `print()` (no `σ` character); scripts runnable under `PYTHONIOENCODING=utf-8`.
- Never retune per-architecture recipes; report results as-is.

---

### Task 1: FLOP-per-step measurement utility

**Files:**
- Create: `experiments/frontier_flops.py`
- Test: `tests/test_frontier_flops.py`

**Interfaces:**
- Consumes: any model exposing `energy(x) -> scalar tensor` (all repo EBMs do).
- Produces: `flops_per_step(model, device, img_shape=(3, 32, 32)) -> int` — FLOPs for ONE descent step (energy forward + input-gradient backward) on ONE image. Used by Tasks 3, 8.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frontier_flops.py
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from frontier_flops import flops_per_step


class DummyLinearEBM(nn.Module):
    """Energy = w . x  (single Linear D->1, no bias interactions to count)."""
    def __init__(self, D=3 * 32 * 32):
        super().__init__()
        self.net = nn.Linear(D, 1, bias=False)

    def energy(self, x):
        return self.net(x.view(x.shape[0], -1)).sum()


def test_flops_matches_analytic_linear():
    D = 3 * 32 * 32
    model = DummyLinearEBM(D)
    got = flops_per_step(model, torch.device('cpu'))
    # forward mm: 2*D; backward (grad_input + grad_weight): 2 more mms, 2*D each.
    expected = 6 * D
    assert abs(got - expected) <= 2 * D, f"got {got}, expected ~{expected}"


def test_flops_positive_for_repo_model_shape():
    model = DummyLinearEBM()
    assert flops_per_step(model, torch.device('cpu')) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.12 -m pytest tests/test_frontier_flops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frontier_flops'`

- [ ] **Step 3: Write the implementation**

```python
# experiments/frontier_flops.py
"""Per-descent-step FLOP measurement for energy heads.

One descent step = one energy forward + one backward pass to get the input
gradient. Measured with torch.utils.flop_counter.FlopCounterMode, which
counts matmul/conv FLOPs in forward AND backward. Elementwise ops (tanh,
spline basis evaluation, piecewise-linear lookup) are NOT counted; matmul
and conv dominate every head in this study. This counting convention is
stated verbatim in the paper section (spec section 6, FLOP accounting risk).
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def flops_per_step(model, device, img_shape=(3, 32, 32)):
    """FLOPs for one energy forward + input-gradient backward on ONE image."""
    model = model.to(device)
    x = torch.randn(1, *img_shape, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        xi = x.detach().requires_grad_(True)
        E = model.energy(xi)
        E = E.sum() if E.ndim > 0 else E
        torch.autograd.grad(E, xi)
    return int(counter.get_total_flops())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.12 -m pytest tests/test_frontier_flops.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add experiments/frontier_flops.py tests/test_frontier_flops.py
git commit -m "feat: FLOP-per-descent-step measurement via FlopCounterMode"
```

---

### Task 2: Frontier analysis — curves and the pre-registered pass criterion

**Files:**
- Create: `experiments/frontier_analysis.py`
- Test: `tests/test_frontier_analysis.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure numpy).
- Produces (used by Tasks 3–5, 8):
  - `curve_from_psnr(psnr_per_step, flops_step, psnr_k0) -> (flops: np.ndarray, psnr: np.ndarray)` — arrays of length K_max+1 including the K=0 (0-FLOPs, noisy-input) point.
  - `advantage_region(flops_light, psnr_light, flops_heavy, psnr_heavy, n_grid=200, margin_db=0.3) -> list[(lo, hi)]` — contiguous FLOP-budget ranges where light ≥ heavy + margin.
  - `evaluate_pass(results, light, heavy) -> dict` with keys `verdict` ('PASS'|'FAIL'), `per_seed_pass_count`, `regions`, `pair`, `margin_db`. `results[model][sigma][seed]` must be a dict with keys `psnr_per_step` (list, len K_max), `psnr_k0` (float), `flops_step` (float).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frontier_analysis.py
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from frontier_analysis import curve_from_psnr, advantage_region, evaluate_pass


def _cell(psnr_per_step, flops_step, psnr_k0=20.0):
    return {'psnr_per_step': list(psnr_per_step), 'psnr_k0': psnr_k0,
            'flops_step': flops_step}


def test_curve_includes_k0_origin():
    f, q = curve_from_psnr([21.0, 22.0, 22.5], flops_step=10.0, psnr_k0=20.0)
    assert f[0] == 0.0 and q[0] == 20.0
    assert len(f) == 4 and f[-1] == 30.0 and q[-1] == 22.5


def test_advantage_region_found_when_light_dominates():
    # light: cheap steps, +1 dB everywhere over the shared budget range
    fl, ql = curve_from_psnr(np.linspace(22, 26, 30), 1.0, 20.0)
    fh, qh = curve_from_psnr(np.linspace(21, 25, 30), 3.0, 20.0)
    regions = advantage_region(fl, ql, fh, qh)
    assert len(regions) >= 1
    lo, hi = regions[0]
    assert hi > lo > 0


def test_no_advantage_region_when_heavy_dominates():
    fl, ql = curve_from_psnr(np.linspace(21, 23, 30), 1.0, 20.0)
    fh, qh = curve_from_psnr(np.linspace(24, 27, 30), 3.0, 20.0)
    assert advantage_region(fl, ql, fh, qh) == []


def test_evaluate_pass_requires_3_sigmas_both_seeds():
    win = list(np.linspace(23, 27, 30))    # strong light curve
    lose = list(np.linspace(21, 24, 30))   # weaker heavy curve at 3x step cost
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = {'light': {}, 'heavy': {}}
    for s in sigmas:
        results['light'][s] = {0: _cell(win, 1.0), 1: _cell(win, 1.0)}
        results['heavy'][s] = {0: _cell(lose, 3.0), 1: _cell(lose, 3.0)}
    out = evaluate_pass(results, light='light', heavy='heavy')
    assert out['verdict'] == 'PASS'
    # break seed 1 everywhere -> FAIL (criterion demands BOTH seeds)
    for s in sigmas:
        results['light'][s][1] = _cell(lose, 3.0)
        results['heavy'][s][1] = _cell(win, 1.0)
    out = evaluate_pass(results, light='light', heavy='heavy')
    assert out['verdict'] == 'FAIL'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.12 -m pytest tests/test_frontier_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frontier_analysis'`

- [ ] **Step 3: Write the implementation**

```python
# experiments/frontier_analysis.py
"""Quality-vs-FLOPs curves and the PRE-REGISTERED pass criterion.

Criterion (spec docs/superpowers/specs/2026-07-02-inference-frontier-
groupkan-design.md section 2): PASS iff the lighter head beats the heavier
head by >= 0.3 dB PSNR at matched total FLOPs over a contiguous budget
range, across >= 3 of 5 sigma values, in BOTH seeds. Do not change the
constants below after data collection has started.
"""
import numpy as np

MARGIN_DB = 0.3
MIN_SIGMAS_PASS = 3


def curve_from_psnr(psnr_per_step, flops_step, psnr_k0):
    """(cumulative FLOPs, PSNR) curve; K=0 point (noisy input, 0 FLOPs) first."""
    K_max = len(psnr_per_step)
    flops = np.arange(0, K_max + 1, dtype=float) * float(flops_step)
    psnr = np.concatenate([[float(psnr_k0)],
                           np.asarray(psnr_per_step, dtype=float)])
    return flops, psnr


def advantage_region(flops_light, psnr_light, flops_heavy, psnr_heavy,
                     n_grid=200, margin_db=MARGIN_DB):
    """Contiguous budget ranges where light - heavy >= margin_db.

    Both curves are interpolated onto a shared log-spaced FLOP grid spanning
    the overlap of their nonzero-FLOP supports. A region must span more than
    one grid point (a range, not a point).
    """
    lo = max(flops_light[1], flops_heavy[1])
    hi = min(flops_light[-1], flops_heavy[-1])
    if hi <= lo:
        return []
    grid = np.geomspace(lo, hi, n_grid)
    ql = np.interp(grid, flops_light, psnr_light)
    qh = np.interp(grid, flops_heavy, psnr_heavy)
    adv = (ql - qh) >= margin_db

    regions, start = [], None
    for i in range(len(adv)):
        if adv[i] and start is None:
            start = i
        closing = (not adv[i]) or (i == len(adv) - 1)
        if closing and start is not None:
            end = i if adv[i] else i - 1
            if end > start:
                regions.append((float(grid[start]), float(grid[end])))
            start = None
    return regions


def evaluate_pass(results, light, heavy):
    """Apply the pre-registered criterion to results[model][sigma][seed]."""
    sigmas = sorted(results[light].keys())
    seeds = sorted(next(iter(results[light].values())).keys())
    per_seed_pass_count, detail = {}, {}
    for seed in seeds:
        n_pass = 0
        for s in sigmas:
            rl, rh = results[light][s][seed], results[heavy][s][seed]
            fl, ql = curve_from_psnr(rl['psnr_per_step'], rl['flops_step'],
                                     rl['psnr_k0'])
            fh, qh = curve_from_psnr(rh['psnr_per_step'], rh['flops_step'],
                                     rh['psnr_k0'])
            regions = advantage_region(fl, ql, fh, qh)
            detail[f'sigma={s}/seed={seed}'] = regions
            if regions:
                n_pass += 1
        per_seed_pass_count[seed] = n_pass
    verdict = all(c >= MIN_SIGMAS_PASS for c in per_seed_pass_count.values())
    return {'verdict': 'PASS' if verdict else 'FAIL',
            'per_seed_pass_count': per_seed_pass_count,
            'regions': detail, 'pair': [light, heavy],
            'margin_db': MARGIN_DB}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.12 -m pytest tests/test_frontier_analysis.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add experiments/frontier_analysis.py tests/test_frontier_analysis.py
git commit -m "feat: frontier curves + pre-registered pass criterion (unit-tested)"
```

---

### Task 3: The frontier sweep script

**Files:**
- Create: `experiments/exp_inference_frontier.py`
- Reads (do not modify): `experiments/exp_fine_grid_kstar.py` (descent recorder), `experiments/exp_scaled_eval_cifar10.py` (`load_kan`, `load_conv_mlp`), `experiments/exp_unet_ebm.py` (`UNetEBM`), `experiments/exp_cifar10.py` (`KANEnergyModel`).

**Interfaces:**
- Consumes: `flops_per_step` (Task 1), `sequential_denoise_record(model, noisy, K_max, dt, dt_decay)` from `exp_fine_grid_kstar`.
- Produces: part files `outputs/inference_frontier/parts/{model}_s{sigma:.2f}_seed{seed}.json`, each `{"model", "sigma", "seed", "flops_step", "ms_per_step_per_image", "psnr_k0", "psnr_per_step" (len 30, mean over 500 images), "psnr_per_step_std", "kstar_per_image" (len 500), "done": true}`. Task 4/5 consume these via `--analyze`.

- [ ] **Step 1: Write the script**

```python
# experiments/exp_inference_frontier.py
"""Fixed-FLOP inference frontier: head capacity vs descent depth.

Spec: docs/superpowers/specs/2026-07-02-inference-frontier-groupkan-design.md
Sweep phase writes resumable part files; --analyze consolidates them into
outputs/inference_frontier/frontier.json + frontier.png and prints the
pre-registered verdict (primary pair: kan_32k vs kan_110k).

Run (sweep, GPU):   py -3.12 experiments/exp_inference_frontier.py --device cuda
Run (smoke, CPU):   py -3.12 experiments/exp_inference_frontier.py --quick --device cpu
Run (analysis):     py -3.12 experiments/exp_inference_frontier.py --analyze
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_fine_grid_kstar import sequential_denoise_record
from exp_scaled_eval_cifar10 import load_kan, load_conv_mlp
from exp_unet_ebm import UNetEBM
from frontier_flops import flops_per_step
from frontier_analysis import curve_from_psnr, evaluate_pass

OUT_DIR = ROOT / 'outputs' / 'inference_frontier'
PARTS = OUT_DIR / 'parts'
PARTS.mkdir(parents=True, exist_ok=True)

SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]
SEEDS = [0, 1]
K_MAX = 30
N_IMAGES = 500
BATCH = 100


def load_unet_ebm(path, device):
    model = UNetEBM().to(device)
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False))
    model.eval()
    return model


def model_registry(device):
    """name -> (loader lambda). Params strictly ordered for the primary pair."""
    return {
        'kan_110k': lambda: load_kan(ROOT / 'outputs/cifar10/kan_ebm_f32.pt',
                                     device, n_filters=32, kan_hidden=[96, 16]),
        'kan_32k': lambda: load_kan(
            ROOT / 'outputs/finalization/kstar_param_scaling/kan_small.pt',
            device, n_filters=16, kan_hidden=[48, 16]),
        'conv_mlp_gelu': lambda: load_conv_mlp(
            ROOT / 'outputs/finalization/rebuttal/conv_mlp_gelu.pt', device),
        'unet': lambda: load_unet_ebm(ROOT / 'outputs/unet_ebm/unet_ebm.pt',
                                      device),
    }


def load_test_images(n):
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False,
                                      download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)])


def noise_for(images, sigma, seed):
    """Deterministic noise, identical across models for a (sigma, seed) cell."""
    g = torch.Generator(device='cpu').manual_seed(10000 * seed
                                                  + int(round(sigma * 1000)))
    return torch.randn(images.shape, generator=g) * sigma


def psnr_batch(pred, clean):
    mse = torch.nn.functional.mse_loss(pred.clamp(-1, 1), clean,
                                       reduction='none').mean(dim=[1, 2, 3])
    return (10.0 * torch.log10(4.0 / mse)).cpu().numpy()


def run_cell(model, images, sigma, seed, k_max, device):
    """One (model, sigma, seed) sweep. Returns the part-file payload dict."""
    noisy_all = (images + noise_for(images, sigma, seed)).clamp(-1, 1)
    per_step, k0 = [], []
    t_steps, n_img_timed = 0.0, 0
    for b in range(0, len(images), BATCH):
        clean = images[b:b + BATCH].to(device)
        noisy = noisy_all[b:b + BATCH].to(device)
        k0.append(psnr_batch(noisy, clean))
        t0 = time.time()
        states = sequential_denoise_record(model, noisy, K_max=k_max)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_steps += time.time() - t0
        n_img_timed += len(clean)
        per_step.append(np.stack([psnr_batch(s, clean) for s in states]))
    per_step = np.concatenate(per_step, axis=1)     # (k_max, n_images)
    k0 = np.concatenate(k0)
    return {
        'sigma': sigma, 'seed': seed,
        'psnr_k0': float(k0.mean()),
        'psnr_per_step': per_step.mean(axis=1).tolist(),
        'psnr_per_step_std': per_step.std(axis=1, ddof=1).tolist(),
        'kstar_per_image': (np.argmax(per_step, axis=0) + 1).tolist(),
        'ms_per_step_per_image': 1000.0 * t_steps / (n_img_timed * k_max),
    }


def sweep(args, device):
    registry = model_registry(device)
    names = args.models or list(registry.keys())
    sigmas = args.sigmas
    seeds = list(range(args.n_seeds))
    for name in names:
        model = registry[name]()
        n_params = sum(p.numel() for p in model.parameters())
        fps = flops_per_step(model, device)
        print(f"[model] {name}: {n_params:,} params, "
              f"{fps / 1e6:.1f} MFLOPs/step/image")
        images = load_test_images(args.n_images)
        for seed in seeds:
            for sigma in sigmas:
                part = PARTS / f"{name}_s{sigma:.2f}_seed{seed}.json"
                if part.exists() and json.loads(part.read_text()).get('done'):
                    print(f"  [skip] {part.name}")
                    continue
                t0 = time.time()
                payload = run_cell(model, images, sigma, seed, args.k_max,
                                   device)
                payload.update({'model': name, 'n_params': n_params,
                                'flops_step': fps, 'done': True})
                part.write_text(json.dumps(payload))
                print(f"  [done] {part.name}  ({time.time() - t0:.0f}s, "
                      f"K*mean={np.mean(payload['kstar_per_image']):.1f})")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def analyze(args):
    results, meta = {}, {}
    for part in sorted(PARTS.glob('*.json')):
        d = json.loads(part.read_text())
        if not d.get('done'):
            continue
        m = d['model']
        results.setdefault(m, {}).setdefault(d['sigma'], {})[d['seed']] = d
        meta[m] = {'n_params': d['n_params'], 'flops_step': d['flops_step'],
                   'ms_per_step_per_image': d['ms_per_step_per_image']}
    if not results:
        print("[analyze] no completed parts found"); return

    verdicts = {}
    pairs = [('kan_32k', 'kan_110k')]                       # primary
    for light, heavy in [('kan_32k', 'conv_mlp_gelu'),
                         ('conv_mlp_gelu', 'kan_110k'),
                         ('kan_32k', 'unet')]:              # secondary
        if light in results and heavy in results:
            pairs.append((light, heavy))
    for light, heavy in pairs:
        if light in results and heavy in results:
            key = f"{light}_vs_{heavy}"
            verdicts[key] = evaluate_pass(results, light, heavy)
            tag = 'PRIMARY' if (light, heavy) == pairs[0] else 'secondary'
            print(f"[{tag}] {key}: {verdicts[key]['verdict']} "
                  f"(per-seed sigma-pass counts: "
                  f"{verdicts[key]['per_seed_pass_count']})")

    sigmas = sorted({s for m in results.values() for s in m.keys()})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 3.6),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sigmas):
        for m in sorted(results.keys()):
            if s not in results[m]:
                continue
            cells = list(results[m][s].values())
            psnr = np.mean([c['psnr_per_step'] for c in cells], axis=0)
            k0 = float(np.mean([c['psnr_k0'] for c in cells]))
            f, q = curve_from_psnr(psnr, meta[m]['flops_step'], k0)
            ax.plot(f[1:], q[1:], marker='.', ms=3, label=m)
        ax.set_xscale('log')
        ax.set_title(f"sigma={s:.2f}")
        ax.set_xlabel('cumulative FLOPs/image')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'frontier.png', dpi=180)

    (OUT_DIR / 'frontier.json').write_text(json.dumps(
        {'meta': meta, 'verdicts': verdicts,
         'protocol': {'sigmas': sigmas, 'k_max': K_MAX,
                      'n_images': N_IMAGES, 'seeds': SEEDS,
                      'dt': 0.05, 'dt_decay': 0.97,
                      'flop_convention':
                          'forward+backward per step, matmul/conv only, '
                          'FlopCounterMode'},
         'results': results}, indent=1))
    print(f"[analyze] wrote {OUT_DIR / 'frontier.json'} and frontier.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--models', nargs='+', default=None)
    p.add_argument('--sigmas', type=float, nargs='+', default=SIGMAS)
    p.add_argument('--n_seeds', type=int, default=len(SEEDS))
    p.add_argument('--n_images', type=int, default=N_IMAGES)
    p.add_argument('--k_max', type=int, default=K_MAX)
    p.add_argument('--quick', action='store_true',
                   help='smoke: 20 images, sigmas 0.10/0.30, K=10, 1 seed')
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.n_images, args.sigmas = 20, [0.10, 0.30]
        args.k_max, args.n_seeds = 10, 1
    if args.analyze:
        analyze(args)
    else:
        device = torch.device(args.device if torch.cuda.is_available()
                              or args.device == 'cpu' else 'cpu')
        sweep(args, device)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test on CPU**

Run: `py -3.12 experiments/exp_inference_frontier.py --quick --device cpu`
Expected: prints `[model] kan_110k: ... MFLOPs/step/image` then `[done] kan_110k_s0.10_seed0.json ...` for each of 4 models × 2 σ; 8 part files under `outputs/inference_frontier/parts/`; no traceback. (If `unet` checkpoint keys mismatch, spec contingency: remove `'unet'` from `model_registry` and note it in the commit message.)

- [ ] **Step 3: Verify resumability**

Run the same command again: `py -3.12 experiments/exp_inference_frontier.py --quick --device cpu`
Expected: all cells print `[skip]`, completes in seconds.

- [ ] **Step 4: Smoke-analyze**

Run: `py -3.12 experiments/exp_inference_frontier.py --quick --analyze`
Expected: prints `[PRIMARY] kan_32k_vs_kan_110k: PASS` or `FAIL` (either is fine at smoke scale), writes `frontier.json` + `frontier.png`.

- [ ] **Step 5: Delete smoke parts, commit**

```bash
rm outputs/inference_frontier/parts/*.json outputs/inference_frontier/frontier.json outputs/inference_frontier/frontier.png
git add experiments/exp_inference_frontier.py
git commit -m "feat: fixed-FLOP inference-frontier sweep (resumable, pre-registered criterion)"
```

---

### Task 4: Full frontier run (GPU)

**Files:**
- No code changes. Produces `outputs/inference_frontier/parts/*.json` (4 models × 5 σ × 2 seeds = 40 parts).

- [ ] **Step 1: Launch the full sweep on the 4090 (or 3080 fallback)**

Run (background): `py -3.12 experiments/exp_inference_frontier.py --device cuda`
Expected: ~40 `[done]` lines. Rough budget: each cell is 500 images × 30 steps ≈ the fine-grid cost, minutes per cell on GPU; total well under a few hours. If interrupted, re-run the same command — completed cells are skipped.

- [ ] **Step 2: Verify part count**

Run: `py -3.12 -c "from pathlib import Path; import json; ps=[p for p in Path('outputs/inference_frontier/parts').glob('*.json') if json.loads(p.read_text()).get('done')]; print(len(ps))"`
Expected: `40` (or 30 if `unet` was dropped under the spec contingency).

- [ ] **Step 3: LPIPS at each model's best-K (secondary metric, spec §2)**

Create and run `experiments/frontier_lpips.py`:

```python
# experiments/frontier_lpips.py
"""LPIPS (net='squeeze', repo convention) at each model's best mean-PSNR K,
sigma=0.10, first 100 CIFAR-10 test images. Secondary metric per spec
section 2. Run AFTER the full sweep (reads best-K from the part files).

Run: py -3.12 experiments/frontier_lpips.py --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_inference_frontier import (model_registry, load_test_images,
                                    noise_for, PARTS, OUT_DIR)
from exp_fine_grid_kstar import sequential_denoise_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--sigma', type=float, default=0.10)
    p.add_argument('--n_images', type=int, default=100)
    args = p.parse_args()
    device = torch.device(args.device)

    import lpips
    lp = lpips.LPIPS(net='squeeze').to(device)

    images = load_test_images(args.n_images)
    noisy = (images + noise_for(images, args.sigma, seed=0)).clamp(-1, 1)
    out = {}
    for name, loader in model_registry(device).items():
        part = PARTS / f"{name}_s{args.sigma:.2f}_seed0.json"
        if not part.exists():
            print(f"[skip] no part file for {name}")
            continue
        d = json.loads(part.read_text())
        best_k = int(np.argmax(d['psnr_per_step'])) + 1
        model = loader()
        states = sequential_denoise_record(model, noisy.to(device),
                                           K_max=best_k)
        with torch.no_grad():
            v = lp(states[-1].clamp(-1, 1), images.to(device)).mean().item()
        out[name] = {'best_k': best_k, 'lpips_squeeze': v}
        print(f"{name}: best_k={best_k}  LPIPS={v:.4f}")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (OUT_DIR / 'frontier_lpips.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
```

Run: `py -3.12 experiments/frontier_lpips.py --device cuda`
Expected: one `name: best_k=... LPIPS=...` line per model; `outputs/inference_frontier/frontier_lpips.json` written. LPIPS should broadly agree with the PSNR ordering; if it inverts the primary-pair ordering, report that in the paper section as a caveat sentence (the paper already treats LPIPS as the dissenting metric).

- [ ] **Step 4: Commit the raw parts**

```bash
git add experiments/frontier_lpips.py outputs/inference_frontier/parts outputs/inference_frontier/frontier_lpips.json
git commit -m "data: full inference-frontier sweep + LPIPS secondary metric"
```

---

### Task 5: Analysis + DECISION GATE

**Files:**
- Produces: `outputs/inference_frontier/frontier.json`, `outputs/inference_frontier/frontier.png`.

- [ ] **Step 1: Run analysis**

Run: `py -3.12 experiments/exp_inference_frontier.py --analyze`
Expected: `[PRIMARY] kan_32k_vs_kan_110k: PASS` or `FAIL` with per-seed counts, plus secondary pair lines; figure and JSON written.

- [ ] **Step 2: Record the verdict**

Open `outputs/inference_frontier/frontier.json`, read `verdicts.kan_32k_vs_kan_110k.verdict`. This selects the PASS or FAIL framing in Task 9 — nothing else in the plan changes. Report the verdict (and the per-σ advantage regions) to the user before proceeding to Task 6.

- [ ] **Step 3: Commit**

```bash
git add outputs/inference_frontier/frontier.json outputs/inference_frontier/frontier.png
git commit -m "results: inference-frontier verdict (pre-registered criterion)"
```

---

### Task 6: GroupKAN energy head (`src/group_kan.py`)

**Files:**
- Create: `src/group_kan.py`
- Test: `tests/test_group_kan.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Tasks 7, 8):
  - `GroupKANEnergyModel(n_filters=16, filter_size=5, hidden=640, n_groups=8, n_knots=17, n_channels=3)` with methods `energy(x)->scalar`, `energy_grad(x)`, `denoise(x_noisy, n_steps, dt=0.05, dt_decay=0.97)`, `loss(x_clean, sigma)`, property `n_params` — the exact interface of `ConvSmoothMLPEBM`.
  - Size variants: `hidden=640` → ~33.5K params ("group_kan_32k"); `hidden=136` → ~8.3K params ("group_kan_8k").

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_group_kan.py
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from group_kan import GroupKANEnergyModel, SharedPWL


def test_param_counts_near_targets():
    m32 = GroupKANEnergyModel(hidden=640)
    m8 = GroupKANEnergyModel(hidden=136)
    assert 0.9 * 32000 <= m32.n_params <= 1.15 * 32000, m32.n_params
    assert 0.9 * 8000 <= m8.n_params <= 1.15 * 8000, m8.n_params


def test_energy_is_scalar_and_grad_flows_to_splines():
    m = GroupKANEnergyModel(hidden=136)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    E = m.energy(x)
    assert E.ndim == 0
    E.backward()
    for name, p in m.named_parameters():
        if 'values' in name:            # the shared univariate functions
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_pwl_is_identity_at_init():
    phi = SharedPWL(n_groups=4)
    x = torch.linspace(-2.5, 2.5, 64).unsqueeze(0).repeat(3, 1)  # (3, 64)
    gidx = torch.arange(64) % 4
    y = phi(x, gidx)
    assert torch.allclose(y, x, atol=1e-5)


def test_denoise_shape_and_dsm_loss_finite():
    m = GroupKANEnergyModel(hidden=136)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    out = m.denoise(x + 0.1 * torch.randn_like(x), n_steps=3)
    assert out.shape == x.shape
    loss = m.loss(x, sigma=0.15)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.12 -m pytest tests/test_group_kan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'group_kan'`

- [ ] **Step 3: Write the implementation**

```python
# src/group_kan.py
"""GR-KAN-style grouped-KAN energy head.

Grouping idea from the Kolmogorov-Arnold Transformer (Yang & Wang, ICLR
2025): input channels are split into G groups; each group shares ONE
learnable univariate function, applied elementwise, followed by a standard
nn.Linear mix (matmul-friendly). We claim only the APPLICATION as an EBM
energy head, not the mechanism. The shared univariate function here is a
learnable piecewise-linear spline on a fixed uniform grid (initialized to
the identity), the cheapest GPU-parallel choice.

Backbone (filters + tanh(precision*conv) features) is copied verbatim from
ConvSmoothMLPEBM / KANEnergyModel so the training recipe and descent
protocol transfer unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPWL(nn.Module):
    """G learnable piecewise-linear univariate functions on a fixed grid."""

    def __init__(self, n_groups, n_knots=17, x_min=-3.0, x_max=3.0):
        super().__init__()
        self.n_groups, self.n_knots = n_groups, n_knots
        self.x_min, self.x_max = float(x_min), float(x_max)
        grid = torch.linspace(x_min, x_max, n_knots)
        # identity init: values equal grid positions -> phi(x) == clamp(x)
        self.values = nn.Parameter(grid.unsqueeze(0).repeat(n_groups, 1))

    def forward(self, x, group_idx):
        """x: (N, F); group_idx: (F,) long mapping feature -> group."""
        t = (x.clamp(self.x_min, self.x_max) - self.x_min) \
            / (self.x_max - self.x_min) * (self.n_knots - 1)
        i0 = t.floor().long().clamp(0, self.n_knots - 2)
        frac = t - i0.to(t.dtype)
        flat = self.values.view(-1)                       # (G * n_knots,)
        base = (group_idx * self.n_knots).unsqueeze(0)    # (1, F)
        v0 = flat[base + i0]
        v1 = flat[base + i0 + 1]
        return v0 + frac * (v1 - v0)


class GroupKANLayer(nn.Module):
    def __init__(self, in_features, out_features, n_groups=8, n_knots=17):
        super().__init__()
        assert in_features % n_groups == 0, \
            f"{in_features} not divisible by {n_groups} groups"
        self.phi = SharedPWL(n_groups, n_knots)
        gsize = in_features // n_groups
        self.register_buffer(
            'group_idx', torch.arange(n_groups).repeat_interleave(gsize))
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(self.phi(x, self.group_idx))


class GroupKANEnergyModel(nn.Module):
    """Same conv backbone as ConvSmoothMLPEBM; head = 2 GroupKAN layers."""

    def __init__(self, n_filters=16, filter_size=5, hidden=640, n_groups=8,
                 n_knots=17, n_channels=3):
        super().__init__()
        self.n_filters, self.filter_size = n_filters, filter_size
        self.n_channels = n_channels
        self.filters = nn.Parameter(
            0.01 * torch.randn(n_filters, n_channels, filter_size,
                               filter_size))
        self.log_precision = nn.Parameter(torch.zeros(n_filters))
        in_dim = n_channels * n_filters
        assert hidden % n_groups == 0, "hidden must be divisible by n_groups"
        self.head = nn.Sequential(
            GroupKANLayer(in_dim, hidden, n_groups, n_knots),
            GroupKANLayer(hidden, 1, n_groups, n_knots),
        )

    @property
    def precision(self):
        return F.softplus(self.log_precision)

    def extract_features(self, x):
        B, C, H, W = x.shape
        pad = self.filter_size // 2
        feats = []
        for c in range(C):
            feats.append(F.conv2d(x[:, c:c + 1], self.filters[:, c:c + 1],
                                  padding=pad))
        feats = torch.cat(feats, dim=1)
        prec = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = torch.tanh(feats * prec)
        return feats.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

    def energy(self, x):
        return self.head(self.extract_features(x)).sum()

    def energy_grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.energy(xi), xi)[0].detach()

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        u = x_noisy.clone()
        step = dt
        for _ in range(n_steps):
            u = (u - step * self.energy_grad(u).clamp(-1., 1.)).detach()
            step *= dt_decay
        return u

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi = xn.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.energy(xi), xi,
                                       create_graph=True)[0]
        return F.mse_loss(xi - sigma ** 2 * grad, x_clean)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.12 -m pytest tests/test_group_kan.py -v`
Expected: 4 passed. If `test_param_counts_near_targets` fails, adjust `hidden` (640/136) up or down until within the tolerance band and record the final values — they feed Tasks 7–9.

- [ ] **Step 5: Commit**

```bash
git add src/group_kan.py tests/test_group_kan.py
git commit -m "feat: GR-KAN-style GroupKAN energy head (grouped shared PWL + linear mix)"
```

---

### Task 7: Train GroupKAN under the matched recipe

**Files:**
- Create: `experiments/train_group_kan.py`
- Produces: `outputs/group_kan/group_kan_8k.pt`, `outputs/group_kan/group_kan_32k.pt`, `outputs/group_kan/training_log.json`.

**Interfaces:**
- Consumes: `GroupKANEnergyModel` (Task 6).
- Produces: the two checkpoints above, loadable with `GroupKANEnergyModel(hidden=H)` + `load_state_dict`. Task 8 loads them.

- [ ] **Step 1: Write the training script (recipe copied from `experiments/rebuttal_smooth_mlp.py`, the recipe that produced `conv_mlp_gelu.pt`)**

```python
# experiments/train_group_kan.py
"""Train GroupKAN energy heads under the EXACT recipe used for the
ConvSmoothMLP variants (rebuttal_smooth_mlp.py): AdamW lr=3e-4 wd=1e-4,
CosineAnnealingLR T_max=epochs, 40 epochs, 20,000-image CIFAR-10 train
subset, batch 128, sigma_train=0.15, grad-clip 1.0, AMP on CUDA.
No per-architecture tuning (spec section 3).

Run: py -3.12 experiments/train_group_kan.py --device cuda
Smoke: py -3.12 experiments/train_group_kan.py --device cpu --quick
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from group_kan import GroupKANEnergyModel

OUT = ROOT / 'outputs' / 'group_kan'
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {'group_kan_8k': 136, 'group_kan_32k': 640}
SIGMA_TRAIN = 0.15


def train_one(model, loader, device, n_epochs, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = model.loss(x, SIGMA_TRAIN)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = model.loss(x, SIGMA_TRAIN)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        history.append(tot / nb)
        if ep <= 3 or ep % 5 == 0 or ep == n_epochs:
            print(f"  [{tag}] ep {ep:>3d}/{n_epochs}  loss={tot / nb:.5f}")
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--quick', action='store_true',
                   help='smoke: 2 epochs, 1500 images')
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_train = 2, 1500
    device = torch.device(args.device)

    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True,
                                      download=True, transform=tf)
    subset = torch.utils.data.Subset(ds, list(range(args.n_train)))
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True,
                                         num_workers=0, pin_memory=True,
                                         drop_last=True)

    log = {'recipe': {'lr': 3e-4, 'wd': 1e-4, 'epochs': args.epochs,
                      'n_train': args.n_train, 'batch': 128,
                      'sigma_train': SIGMA_TRAIN, 'clip': 1.0},
           'variants': {}}
    for tag, hidden in VARIANTS.items():
        t0 = time.time()
        model = GroupKANEnergyModel(hidden=hidden).to(device)
        print(f"[train] {tag}: hidden={hidden}, {model.n_params:,} params")
        history = train_one(model, loader, device, args.epochs, tag)
        path = OUT / f"{tag}.pt"
        torch.save(model.state_dict(), path)
        log['variants'][tag] = {'hidden': hidden, 'n_params': model.n_params,
                                'final_loss': history[-1],
                                'seconds': time.time() - t0,
                                'checkpoint': str(path)}
        print(f"[saved] {path}")
    (OUT / 'training_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test on CPU**

Run: `py -3.12 experiments/train_group_kan.py --device cpu --quick`
Expected: both variants train 2 epochs with decreasing loss, two `.pt` files + `training_log.json` written, no traceback.

- [ ] **Step 3: Full training on GPU**

Run: `py -3.12 experiments/train_group_kan.py --device cuda`
Expected: 40 epochs per variant (minutes-to-tens-of-minutes each at this scale); final DSM loss same order of magnitude as the ConvMLP variants' (~the value printed in rebuttal runs). Overwrites the smoke checkpoints.

- [ ] **Step 4: Commit**

```bash
git add experiments/train_group_kan.py outputs/group_kan/training_log.json outputs/group_kan/group_kan_8k.pt outputs/group_kan/group_kan_32k.pt
git commit -m "feat: GroupKAN heads trained under matched ConvMLP recipe"
```

---

### Task 8: GroupKAN on the frontier + fine-grid α

**Files:**
- Modify: `experiments/exp_inference_frontier.py` (add two registry entries)
- Create: `experiments/exp_group_kan_kstar.py`
- Produces: 20 more part files; `outputs/group_kan/fine_grid_group_kan_32k.json` (+ `_8k` if time allows).

**Interfaces:**
- Consumes: checkpoints from Task 7; `evaluate_fine_grid`, `fit_power_law`, `parametric_bootstrap_ci`, `bootstrap_ci_kstar` from `experiments/exp_fine_grid_kstar.py`.
- Produces: α, C, R², CI for GroupKAN-32K → consumed by Task 9's table row.

- [ ] **Step 1: Add GroupKAN to the frontier registry**

In `experiments/exp_inference_frontier.py`, add to imports:

```python
from group_kan import GroupKANEnergyModel
```

and add to the dict in `model_registry` (after `'unet'`):

```python
        'group_kan_8k': lambda: _load_group_kan(
            ROOT / 'outputs/group_kan/group_kan_8k.pt', device, hidden=136),
        'group_kan_32k': lambda: _load_group_kan(
            ROOT / 'outputs/group_kan/group_kan_32k.pt', device, hidden=640),
```

and add this loader above `model_registry`:

```python
def _load_group_kan(path, device, hidden):
    model = GroupKANEnergyModel(hidden=hidden).to(device)
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False))
    model.eval()
    return model
```

- [ ] **Step 2: Run the frontier sweep for the two GroupKAN models only**

Run: `py -3.12 experiments/exp_inference_frontier.py --device cuda --models group_kan_8k group_kan_32k`
Expected: 20 new `[done]` part files. **Quality gate (spec §6):** if GroupKAN-32K's best mean PSNR at σ=0.10 is >2 dB below KAN-32K's (compare `max(psnr_per_step)` in the corresponding parts), drop GroupKAN from the frontier figure and keep only the α row (Step 3) if its fit has R²>0.95; if the fit is also poor, drop GroupKAN entirely and say so in the paper. Report which branch applied.

- [ ] **Step 3: Write and run the fine-grid α script**

```python
# experiments/exp_group_kan_kstar.py
"""Fine-grid K* sweep for GroupKAN-32K -> alpha for the paper's
architecture table. Protocol identical to exp_fine_grid_kstar.py
(5 sigmas, 3 seeds, 500 images, K_max=30).

Run: py -3.12 experiments/exp_group_kan_kstar.py --device cuda
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from group_kan import GroupKANEnergyModel
from exp_fine_grid_kstar import (evaluate_fine_grid, fit_power_law,
                                 parametric_bootstrap_ci, bootstrap_ci_kstar)

OUT = ROOT / 'outputs' / 'group_kan'
SEED = 42
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--hidden', type=int, default=640)
    p.add_argument('--tag', default='group_kan_32k')
    p.add_argument('--n_images', type=int, default=500)
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--k_max', type=int, default=30)
    p.add_argument('--batch', type=int, default=100)
    args = p.parse_args()
    device = torch.device(args.device)

    model = GroupKANEnergyModel(hidden=args.hidden).to(device)
    model.load_state_dict(torch.load(OUT / f"{args.tag}.pt",
                                     map_location=device, weights_only=False))
    model.eval()

    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False,
                                      download=True, transform=tf)
    images = torch.stack([ds[i][0] for i in range(args.n_images)])

    summary = {'tag': args.tag, 'n_params': model.n_params,
               'sigmas': SIGMAS, 'per_sigma': {}}
    K_means = []
    for sigma in SIGMAS:
        stacked = []
        for s in range(args.n_seeds):
            torch.manual_seed(SEED + s * 100)
            np.random.seed(SEED + s * 100)
            per_step = []
            for b in range(0, len(images), args.batch):
                batch = images[b:b + args.batch].to(device)
                per_step.append(evaluate_fine_grid(model, batch, sigma,
                                                   args.k_max))
            stacked.append(np.concatenate(per_step, axis=1))
        mean_psnr = np.stack(stacked, axis=0).mean(axis=0)  # (K, n_img)
        kstar = np.argmax(mean_psnr, axis=0) + 1
        lo, hi = bootstrap_ci_kstar(kstar)
        K_means.append(float(np.mean(kstar)))
        summary['per_sigma'][f'{sigma:.3f}'] = {
            'kstar_mean': float(np.mean(kstar)),
            'kstar_std': float(np.std(kstar, ddof=1)),
            'kstar_ci95': [lo, hi]}
        print(f"sigma={sigma:.2f}: K*={np.mean(kstar):.2f}")

    a, alpha, r2 = fit_power_law(SIGMAS, K_means)
    ci_lo, ci_hi = parametric_bootstrap_ci(SIGMAS, K_means)
    summary['power_law'] = {'alpha': alpha, 'C': math.exp(a), 'R2': r2,
                            'ci95_alpha': [ci_lo, ci_hi]}
    out = OUT / f'fine_grid_{args.tag}.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f"alpha={alpha:.4f}  C={math.exp(a):.1f}  R2={r2:.4f}  "
          f"CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"[done] {out}")


if __name__ == '__main__':
    main()
```

Run: `py -3.12 experiments/exp_group_kan_kstar.py --device cuda`
Expected: five `sigma=... K*=...` lines, then `alpha=... R2=...`; JSON written. R²>0.95 keeps the α row; lower R² → drop the row per the Step 2 quality gate.

- [ ] **Step 4: Re-run consolidated analysis**

Run: `py -3.12 experiments/exp_inference_frontier.py --analyze`
Expected: verdict lines now include secondary pairs against `group_kan_*` only if you add them — the primary verdict must NOT change (it uses the pre-registered pair). Figure regenerated with 6 curves.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp_inference_frontier.py experiments/exp_group_kan_kstar.py outputs/group_kan/fine_grid_group_kan_32k.json outputs/inference_frontier
git commit -m "feat: GroupKAN on the frontier + fine-grid alpha"
```

---

### Task 9: Paper edits (`paper_v8.tex`)

**Files:**
- Modify: `paper_v8.tex` (new subsection after the corruption phase-boundary section; architecture table row; abstract/contributions sentence)

**Interfaces:**
- Consumes: `outputs/inference_frontier/frontier.json` (verdict, regions, FLOPs), `outputs/group_kan/fine_grid_group_kan_32k.json` (α row values), `outputs/inference_frontier/frontier.png` (figure).

- [ ] **Step 1: Insert the new subsection**

Locate the end of the corruption phase-boundary section in `paper_v8.tex` (search for the phase-diagram table, verdicts LAW/BREAKS). Insert after it, filling every `\NUM{...}` placeholder from the JSONs and DELETING the framing that does not match the Task 5 verdict:

```latex
\subsection{The Inference Frontier: Trading Head Capacity for Depth}
\label{sec:frontier}

The depth law fixes how quality scales with iteration count for a given
energy head. A deployment question remains: at a \emph{fixed inference FLOP
budget}, should one spend the budget on a heavier head taking fewer descent
steps, or a lighter head taking more? We measure this directly. For each
head we count the FLOPs of one descent step (one energy forward plus the
input-gradient backward; matmul/conv operations, counted with PyTorch's
\texttt{FlopCounterMode}) and record PSNR after every step
($K{=}0{\dots}30$, five noise levels, 500 CIFAR-10 test images, two noise
seeds), yielding quality-versus-cumulative-FLOPs curves. The comparison
criterion was pre-registered before data collection: the lighter head must
lead by ${\geq}0.3$\,dB over a contiguous budget range in at least three of
five noise levels and both seeds, with the parameter-matched pair
KAN-32K/KAN-110K primary.

% ===== KEEP EXACTLY ONE OF THE TWO PARAGRAPHS BELOW (Task 5 verdict) =====

% --- PASS framing ---
Depth substitutes for capacity. The 32K-parameter head, given its FLOP
budget in extra steps, matches or exceeds the 110K head by
$\NUM{X.X}$--$\NUM{X.X}$\,dB across budgets of
$\NUM{X}$--$\NUM{X}$\,MFLOPs/image (Fig.~\ref{fig:frontier}), and a grouped
KAN head~\citep{yang2025kat} at $\NUM{8.3}$K parameters extends the frontier
further. The depth law of Sec.~3 then tells the practitioner how many steps
that budget should buy at each noise level: the two results compose into a
calibration-free deployment rule.

% --- FAIL framing ---
Depth does \emph{not} substitute for capacity. At every matched budget the
heavier head dominates (max lighter-head advantage
$\NUM{X.X}$\,dB $<$ the pre-registered $0.3$\,dB margin;
Fig.~\ref{fig:frontier}). Together with the absence of per-instance
adaptivity and the dissipative character of refinement (Sec.~6), this adds a
third boundary: iteration buys accuracy only along the noise axis of the
depth law, not as a substitute for model capacity at inference time.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/frontier.png}
\caption{Quality vs.\ cumulative inference FLOPs per image (log axis), one
panel per noise level, one curve per energy head; markers are descent steps.
The pre-registered comparison is KAN-32K vs.\ KAN-110K.}
\label{fig:frontier}
\end{figure}

A sixth architecture on the depth law: the grouped KAN head trained under
the identical denoising-score-matching recipe follows
$K^\ast(\sigma)=C\sigma^\alpha$ with $\alpha=\NUM{X.XXX}$
($R^2=\NUM{0.XXX}$, 95\% CI $[\NUM{X.XX},\NUM{X.XX}]$), consistent with the
iteration-not-architecture finding.
```

Copy `outputs/inference_frontier/frontier.png` to the paper's `figures/` directory (create it if the paper currently keeps figures elsewhere — match the existing `\includegraphics` path convention in `paper_v8.tex`).

- [ ] **Step 2: Add the bibliography entry**

Add to the paper's bibliography (matching its existing entry style):

```latex
\bibitem{yang2025kat} Yang, X.; Wang, X. 2025. Kolmogorov--Arnold
Transformer. In \emph{ICLR 2025}.
```

- [ ] **Step 3: Add the α table row**

In the architecture/α table, add below the existing KAN rows, with values from `outputs/group_kan/fine_grid_group_kan_32k.json`:

```latex
GroupKAN-32K & 33.5K & $\NUM{X.XXX}$ & $\NUM{0.XXX}$ & $[\NUM{X.XX}, \NUM{X.XX}]$ \\
```

Update every occurrence of "five architectures" to "six architectures" ONLY if the α row survives the Task 8 quality gate. For the architecture-independence F-ratio: search the repo for the script that produced F=0.79 (`grep -ri "0.79\|f_ratio\|F-ratio" theory/ experiments/ outputs/`); if found, extend it with the GroupKAN per-seed α values and update the number in the paper; if not found within 30 minutes, keep F=0.79 attributed to the original five architectures and state in the text that the sixth α lies within (or outside) the same band — never report a recomputed statistic whose provenance you cannot trace.

- [ ] **Step 4: One-sentence updates to abstract and contribution list**

Add one sentence to the abstract and one item (or extension of an existing item) to the contributions, scoped exactly to the verdict — PASS: "at fixed inference FLOPs, descent depth substitutes for energy-head capacity"; FAIL: "at fixed inference FLOPs, descent depth does not substitute for head capacity — a third limit". No other claims.

- [ ] **Step 5: Consistency check and commit**

Check: no `\NUM{` placeholders remain (`grep "NUM{" paper_v8.tex` returns nothing); exactly one framing paragraph present; "six architectures" count matches the table; figure file exists at the `\includegraphics` path.

```bash
git add paper_v8.tex figures/frontier.png
git commit -m "paper: inference-frontier section + GroupKAN alpha row (verdict-scoped)"
```

---

### Task 10: Final verification

**Files:** none created; regeneration + checks only.

- [ ] **Step 1: Full test suite**

Run: `py -3.12 -m pytest tests/test_frontier_flops.py tests/test_frontier_analysis.py tests/test_group_kan.py -v`
Expected: all pass.

- [ ] **Step 2: Regenerate analysis from raw parts and diff**

Run: `py -3.12 experiments/exp_inference_frontier.py --analyze`
Expected: `frontier.json` verdict identical to the committed one (byte-level diffs in float formatting are acceptable; the `verdicts` block must match).

- [ ] **Step 3: Trace every new paper number**

For each numeric value added to `paper_v8.tex` in Task 9, name its JSON source (`frontier.json` or `fine_grid_group_kan_32k.json`) and confirm equality. List them in the commit message.

- [ ] **Step 4: Commit and report**

```bash
git add -A outputs/inference_frontier outputs/group_kan
git commit -m "chore: final verification of inference-frontier results"
```

Report to the user: the verdict, the per-σ advantage regions (or their absence), GroupKAN's α and training quality, which contingency branches (if any) fired, and the remaining pre-submission items (citation verification: RaM author list, Christie 2015, Lieder 2020; AAAI template port).

---

### Task 11 (STRETCH — only if Tasks 1–10 are done before Jul 14; first thing to cut)

**Files:**
- Create: `experiments/plot_spline_trajectory.py`
- Produces: `outputs/group_kan/spline_trajectory.png` (qualitative figure, spec §4 stretch goal)

**Interfaces:**
- Consumes: `GroupKANEnergyModel` + `outputs/group_kan/group_kan_32k.pt` (Task 7), `sequential_denoise_record` (via `exp_fine_grid_kstar`).

- [ ] **Step 1: Write and run the visualization**

```python
# experiments/plot_spline_trajectory.py
"""Qualitative figure: where each feature-group's operating point sits on
its FIXED shared univariate function at each descent step, for one image.
This is a VISUALIZATION of the refinement trajectory. The words
'interpretable' / 'reading thoughts' must not appear in the caption or
text (spec section 9): splines are frozen at inference; only the operating
point moves.

Run: py -3.12 experiments/plot_spline_trajectory.py --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from group_kan import GroupKANEnergyModel
from exp_inference_frontier import load_test_images, noise_for
from exp_fine_grid_kstar import sequential_denoise_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--sigma', type=float, default=0.20)
    p.add_argument('--k_max', type=int, default=15)
    args = p.parse_args()
    device = torch.device(args.device)

    model = GroupKANEnergyModel(hidden=640).to(device)
    model.load_state_dict(torch.load(
        ROOT / 'outputs/group_kan/group_kan_32k.pt', map_location=device,
        weights_only=False))
    model.eval()

    img = load_test_images(1)
    noisy = (img + noise_for(img, args.sigma, seed=0)).clamp(-1, 1)
    states = [noisy] + sequential_denoise_record(model, noisy.to(device),
                                                 K_max=args.k_max)

    layer1 = model.head[0]
    phi, gidx = layer1.phi, layer1.group_idx
    G = phi.n_groups
    xs = torch.linspace(phi.x_min, phi.x_max, 200)

    # mean operating point per group per step (features of the whole image)
    ops = []  # (n_steps+1, G)
    for s in states:
        f = model.extract_features(s.to(device))          # (H*W, 48)
        ops.append([f[:, gidx == g].mean().item() for g in range(G)])
    ops = np.array(ops)

    fig, axes = plt.subplots(2, G // 2, figsize=(3 * G // 2, 5))
    with torch.no_grad():
        for g, ax in enumerate(axes.ravel()):
            curve = phi(xs.unsqueeze(0), torch.full((200,), g,
                        dtype=torch.long)).squeeze(0).numpy()
            ax.plot(xs.numpy(), curve, color='0.6', lw=1.5)
            sc = ax.scatter(ops[:, g],
                            np.interp(ops[:, g], xs.numpy(), curve),
                            c=np.arange(len(ops)), cmap='viridis', s=14)
            ax.set_title(f"group {g}", fontsize=8)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label='descent step',
                 shrink=0.7)
    out = ROOT / 'outputs/group_kan/spline_trajectory.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f"[done] {out}")


if __name__ == '__main__':
    main()
```

Run: `py -3.12 experiments/plot_spline_trajectory.py --device cpu`
Expected: an 8-panel figure, fixed grey curves with viridis-colored operating points migrating along them. Include in the paper ONLY if it visibly adds clarity; caption must call it a trajectory visualization.

- [ ] **Step 2: Commit**

```bash
git add experiments/plot_spline_trajectory.py outputs/group_kan/spline_trajectory.png
git commit -m "feat(stretch): spline operating-point trajectory visualization"
```
