"""
Recipe ablation for controllable α.
==============================
Train KAN-EBM (and optionally ConvMLP-GELU) on CIFAR-10 with different training
recipes, then measure the inference-depth exponent α via the fine-grid K* sweep.

Hypothesis: the training recipe changes γ (curvature spectrum), which changes α.
If α = 2γ/β̂ with β̂ fixed by the data, then γ is the only handle. We test whether
different recipes give different γ (implied via α) and whether the ordering holds.

Recipe variants (default: KAN-EBM small, n_train=20k, 1 seed per recipe for speed):
  1. single-sigma   : σ_train = [0.15]  (standard baseline)
  2. multi-sigma    : σ_train = [0.05, 0.10, 0.15, 0.20, 0.30]  (the v1-v4 confound)
  3. wd-0           : weight_decay = 0.0
  4. wd-1e-2        : weight_decay = 1e-2
  5. epochs-20      : 20 epochs
  6. epochs-80      : 80 epochs
  7. batch-128      : batch_size = 128
  8. convmlp-multi  : ConvMLP-GELU + multi-sigma (cross-architecture test)

For each recipe we train, run the tier0 K* protocol (7 σ values, continuous K*),
fit α, and compute implied γ = α · β̂ / 2  (β̂_CIFAR ≈ 2.90 from measure_beta.py).

Output:
  outputs/recipe_ablations/recipe_ablations.json   — all results
  outputs/theory/recipe_ablations.json               — summary for reporting

Usage (full, ~40-50h on RTX 3080):
    py -3.12 theory/recipe_ablation.py --epochs 40 --seeds 0
Quick smoke test (~5 min):
    py -3.12 theory/recipe_ablation.py --quick
Resume-safe: completed runs are skipped on restart.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from torchvision import datasets, transforms
from exp_cifar10 import KANEnergyModel
from rebuttal_smooth_mlp import ConvSmoothMLPEBM

OUT = ROOT / 'outputs' / 'recipe_ablations'
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / 'outputs' / 'theory'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 spectral exponent (from data_beta.json)
BETA_HAT_CIFAR = 2.904

# Tier-0 K* protocol (matches tier0_seeds.py exactly)
SIGMAS = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
K_MAX = 30

# ─────────────────────────────────────────────────────────────────────────────
# Recipes
# ─────────────────────────────────────────────────────────────────────────────

RECIPES = [
    dict(name='single-sigma', arch='kan', sigma_train=[0.15], weight_decay=1e-4,
         epochs=40, batch=32, n_train=20000),
    dict(name='multi-sigma',  arch='kan', sigma_train=[0.05, 0.10, 0.15, 0.20, 0.30],
         weight_decay=1e-4, epochs=40, batch=32, n_train=20000),
    dict(name='wd-0',         arch='kan', sigma_train=[0.15], weight_decay=0.0,
         epochs=40, batch=32, n_train=20000),
    dict(name='wd-1e-2',      arch='kan', sigma_train=[0.15], weight_decay=1e-2,
         epochs=40, batch=32, n_train=20000),
    dict(name='epochs-20',    arch='kan', sigma_train=[0.15], weight_decay=1e-4,
         epochs=20, batch=32, n_train=20000),
    dict(name='epochs-80',    arch='kan', sigma_train=[0.15], weight_decay=1e-4,
         epochs=80, batch=32, n_train=20000),
    dict(name='batch-128',    arch='kan', sigma_train=[0.15], weight_decay=1e-4,
         epochs=40, batch=128, n_train=20000),
    dict(name='convmlp-multi', arch='convmlp_gelu', sigma_train=[0.05, 0.10, 0.15, 0.20, 0.30],
         weight_decay=1e-4, epochs=40, batch=32, n_train=20000),
]

ARCHS = {
    'kan': lambda: KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[48, 16], n_channels=3),
    'convmlp_gelu': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                              n_channels=3, activation='gelu'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_data(n_train, batch_size):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = datasets.CIFAR10(ROOT / 'data', train=True, download=True, transform=tf)
    test_ds  = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    subset = torch.utils.data.Subset(train_ds, list(range(n_train)))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
    return loader, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Training (recipe-aware)
# ─────────────────────────────────────────────────────────────────────────────

def train_recipe(model, loader, device, recipe, tag=''):
    """Train a model with the specified recipe."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=recipe['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe['epochs'])
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    sigma_list = recipe['sigma_train']
    t0 = time.time()
    for ep in range(1, recipe['epochs'] + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device)
            # sample a sigma per batch if multi-sigma
            sigma = float(np.random.choice(sigma_list))
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = model.loss(x, sigma)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = model.loss(x, sigma)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        avg = ep_loss / max(1, nb)
        if ep <= 3 or ep % 5 == 0 or ep == recipe['epochs']:
            print(f"  [{tag}] ep {ep:>3d}/{recipe['epochs']}  loss={avg:.5f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    return time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# K* sweep (tier0 protocol, continuous K*)
# ─────────────────────────────────────────────────────────────────────────────

def psnr_per_image(u, clean):
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).cpu().numpy()


def kstar_at_sigma(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Continuous K* = mean over images of per-image argmax-PSNR step."""
    per_image = []
    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        u = (clean + sigma * torch.randn_like(clean)).to(device)
        series = [psnr_per_image(u, clean)]
        step = dt
        for _ in range(K_MAX):
            with torch.enable_grad():
                xi = u.detach().requires_grad_(True)
                E = model.energy(xi)
                if E.ndim > 0:
                    E = E.sum()
                g = torch.autograd.grad(E, xi)[0].detach()
            u = (u - step * g.clamp(-1, 1)).detach()
            step *= decay
            series.append(psnr_per_image(u, clean))
        arr = np.stack(series, axis=0)            # (K+1, N)
        per_image.append(arr.argmax(axis=0))      # (N,) per-image integer K*
    return float(np.mean(per_image))


def fit_alpha(sigmas, kstars):
    sigmas, kstars = np.array(sigmas, float), np.array(kstars, float)
    ok = kstars > 0
    if ok.sum() < 4 or np.std(kstars[ok]) < 1e-6:
        return float('nan'), float('nan')
    x, y = np.log(sigmas[ok]), np.log(kstars[ok])
    slope, intercept = np.polyfit(x, y, 1)
    yh = intercept + slope * x
    r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(r2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--recipes', nargs='+', default=[r['name'] for r in RECIPES],
                   help='Subset of recipes to run')
    p.add_argument('--quick', action='store_true',
                   help='Smoke test: 3 epochs, 1500 train, 16 eval, 1 eval_seed')
    p.add_argument('--n_eval', type=int, default=64,
                   help='Eval images for K* (tier0 uses 64)')
    p.add_argument('--eval_seeds', type=int, default=2,
                   help='Eval noise seeds per sigma (tier0 uses 2)')
    p.add_argument('--n_train', type=int, default=None,
                   help='Override n_train per recipe (default: recipe default or 1500 if quick)')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override epochs per recipe (default: recipe default or 3 if quick)')
    p.add_argument('--plot', action='store_true',
                   help='Generate summary figure from existing JSON')
    args = p.parse_args()

    res_path = OUT / 'recipe_ablations.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    if not args.plot:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"[device] {device}" + (f"  [{torch.cuda.get_device_name(0)}]"
                                       if device.type == 'cuda' else ""))

        # quick overrides
        quick = args.quick
        n_train = 1500 if quick else (args.n_train if args.n_train is not None else None)
        n_eval = 16 if quick else args.n_eval
        eval_seeds = 1 if quick else args.eval_seeds

        for recipe in RECIPES:
            if recipe['name'] not in args.recipes:
                continue

            for seed in args.seeds:
                key = f"{recipe['name']}|s{seed}"
                if key in results and np.isfinite(results[key].get('alpha', float('nan'))):
                    print(f"[skip] {key} already done (alpha={results[key]['alpha']:.3f})")
                    continue

                print(f"\n{'='*70}")
                print(f"Recipe: {recipe['name']}  seed={seed}")
                print(f"  arch={recipe['arch']}  sigmas={recipe['sigma_train']}  "
                      f"wd={recipe['weight_decay']}  epochs={recipe['epochs']}  "
                      f"batch={recipe['batch']}")
                print(f"{'='*70}", flush=True)

                torch.manual_seed(seed)
                np.random.seed(seed)

                # Determine effective n_train and epochs
                eff_n_train = n_train if n_train is not None else recipe['n_train']
                eff_epochs = args.epochs if args.epochs is not None else recipe['epochs']
                if quick:
                    eff_epochs = 3

                # Data
                loader, test_ds = get_data(eff_n_train, recipe['batch'])
                clean = torch.stack([test_ds[i][0] for i in range(n_eval)]).to(device)

                # Model
                model = ARCHS[recipe['arch']]().to(device)
                print(f"  Params: {model.n_params:,}")

                # Train
                t0 = time.time()
                recipe_eff = dict(recipe)
                recipe_eff['epochs'] = eff_epochs
                train_time = train_recipe(model, loader, device, recipe_eff,
                                              tag=f"{recipe['name']}-s{seed}")
                print(f"  Training complete: {train_time/60:.1f} min")

                # Save checkpoint
                ckpt_path = OUT / f"{recipe['name']}_s{seed}.pt"
                torch.save(model.state_dict(), ckpt_path)

                # K* sweep
                print(f"  K* sweep (sigmas={SIGMAS}, eval_seeds={eval_seeds}, "
                      f"n_eval={n_eval}) ...")
                kstars = [kstar_at_sigma(model, clean, s, device, eval_seeds=eval_seeds)
                          for s in SIGMAS]
                alpha, r2 = fit_alpha(SIGMAS, kstars)
                gamma_implied = alpha * BETA_HAT_CIFAR / 2.0 if np.isfinite(alpha) else float('nan')
                C = math.exp(np.log(kstars[0]) - alpha * np.log(SIGMAS[0])) if np.isfinite(alpha) else float('nan')
                print(f"  K*={['%.2f'%k for k in kstars]}")
                print(f"  alpha={alpha:.3f}  R^2={r2:.3f}  C={C:.2f}  "
                      f"gamma_implied={gamma_implied:.3f}")

                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                results[key] = dict(
                    recipe=recipe['name'],
                    seed=seed,
                    arch=recipe['arch'],
                    sigma_train=recipe['sigma_train'],
                    weight_decay=recipe['weight_decay'],
                    epochs=eff_epochs,
                    batch=recipe['batch'],
                    n_train=eff_n_train,
                    n_eval=n_eval,
                    eval_seeds=eval_seeds,
                    sigmas=SIGMAS,
                    kstars=kstars,
                    alpha=alpha,
                    r2=r2,
                    C=C,
                    gamma_implied=gamma_implied,
                    beta_hat=BETA_HAT_CIFAR,
                    train_time_s=train_time,
                )
                res_path.write_text(json.dumps(results, indent=2))
                print(f"  Saved results -> {res_path}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RECIPE ABALATION SUMMARY")
    print("="*70)
    print(f"{'Recipe':<18} {'Arch':<12} {'Epochs':>6} {'WD':>8} {'Batch':>6} "
          f"{'Alpha':>8} {'R^2':>6} {'Gamma':>8} {'Time(min)':>10}")
    print("-"*90)
    for recipe in RECIPES:
        for seed in args.seeds:
            key = f"{recipe['name']}|s{seed}"
            if key not in results:
                continue
            r = results[key]
            a = r['alpha']
            g = r['gamma_implied']
            t = r['train_time_s'] / 60.0
            print(f"{r['recipe']:<18} {r['arch']:<12} {r['epochs']:>6} "
                  f"{r['weight_decay']:>8.0e} {r['batch']:>6} "
                  f"{a:>8.3f} {r['r2']:>6.3f} {g:>8.3f} {t:>10.1f}")

    # Write summary JSON
    summary = {key: results[key] for key in results}
    (FIG_DIR / 'recipe_ablations.json').write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {FIG_DIR / 'recipe_ablations.json'}")

    if args.plot or (not args.plot and results):
        make_figure(results)


def make_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    # Group by recipe name, average over seeds
    by_recipe = {}
    for key, r in results.items():
        if not np.isfinite(r.get('alpha', float('nan'))):
            continue
        name = r['recipe']
        by_recipe.setdefault(name, []).append(r['alpha'])

    names = []
    alphas = []
    errs = []
    for i, (name, vals) in enumerate(sorted(by_recipe.items())):
        a = np.array(vals)
        names.append(name)
        alphas.append(a.mean())
        errs.append(a.std(ddof=1) if len(a) > 1 else 0.0)
        c = colors[i % 10]
        ax.errorbar(i, alphas[-1], yerr=errs[-1], fmt='o', ms=8, capsize=4, color=c)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(r'Inference exponent $\alpha$')
    ax.set_title('Controllable α: recipe ablation on CIFAR-10')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / 'recipe_ablations.png'), dpi=200)
    fig.savefig(str(FIG_DIR / 'recipe_ablations.pdf'), dpi=200)
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'recipe_ablations.png'}")


if __name__ == '__main__':
    main()
