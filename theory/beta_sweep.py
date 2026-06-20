"""
Controlled go/no-go test of the spectral-shrinkage mechanism: alpha(beta) on
synthetic 1/f^beta Gaussian random fields.
================================================================================
The illustrative theory (THEORY_NOTE.md) says K*(sigma) ~ C sigma^alpha is
early-stopped spectral (Landweber) shrinkage with alpha = 2*gamma/beta. On real
checkpoints the *measured* form failed its go/no-go (measure_kappa.py), with the
suspected confound that real images/architectures violate the data-basis
diagonalization assumption. Gaussian random fields with P(f) = f^{-beta} satisfy
the assumptions BY CONSTRUCTION, so this is the controlled setting in which

        log alpha = log(2*gamma) - 1 * log(beta)

is actually falsifiable: across beta in {1..3} at fixed architecture (=> fixed
gamma), log alpha vs log beta-hat must be linear with slope -1.

  PASS -> theory section upgrades from "illustrative" to "tested where its
          assumptions hold"; CIFAR(alpha~1.38) vs CelebA-64(alpha~1.26) becomes
          the in-the-wild corroboration (see measure_beta.py --alphas).
  FAIL -> reported as the boundary of the spectral account; the empirical law,
          phase diagram and practical rule are unaffected.

Protocol provenance: identical recipe + per-image continuous-K* harness as
tier0_seeds.py / phase_b_diffusion.py (single sigma_train=0.15, 40ep AdamW+cosine
grad-clip 1.0, SIGMAS, K_MAX=30, dt=0.05, decay=0.97). Do not change constants
here without changing them there.

Run (full, ~6-10h on a 4090; 30 runs = 5 betas x 2 families x 3 seeds):
    py -3.12 theory/beta_sweep.py --epochs 40 --seeds 0 1 2
Smoke test (~10 min):
    py -3.12 theory/beta_sweep.py --quick
Split across machines (merge at analysis time):
    py -3.12 theory/beta_sweep.py --families convmlp_gelu --out beta_sweep_ebm.json    # 3080
    py -3.12 theory/beta_sweep.py --families score        --out beta_sweep_score.json  # 4090
    py -3.12 theory/beta_sweep.py --plot_only --out beta_sweep_ebm.json --merge beta_sweep_score.json

The results JSON is written after EVERY run (resume-safe: completed runs are
skipped on restart), checkpoints go to outputs/beta_sweep/ckpt/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
for sub in ('', 'src', 'experiments', 'theory'):
    sys.path.insert(0, str(ROOT / sub))

from tier0_seeds import SIGMAS, kstar_at_sigma, fit_alpha            # noqa: E402
from phase_b_diffusion import ScoreNet, train_score, kstar_score     # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM, train_one          # noqa: E402
from measure_beta import grf_images, radial_spectrum, fit_beta       # noqa: E402

OUT = ROOT / 'outputs' / 'beta_sweep'
CKPT = OUT / 'ckpt'
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / 'outputs' / 'theory'

FAMILIES = {
    'convmlp_gelu': dict(is_score=False,
                         ctor=lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5,
                                                       mlp_hidden=160, n_channels=3,
                                                       activation='gelu')),
    'score':        dict(is_score=True,
                         ctor=lambda: ScoreNet(ch=3, h=64)),
    # optional third family; KAN needs --batch 32 (10GB) and is ~3-4x slower
    'kan':          dict(is_score=False, ctor=None),   # lazily built (heavy import)
}


def _kan_ctor():
    from exp_cifar10 import KANEnergyModel
    return KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[48, 16], n_channels=3)


def build_data(beta, n_train, n_eval, size, std, batch, device):
    """Fixed dataset per beta (seed depends only on beta — all model seeds see the
    same data, mirroring how CIFAR is fixed across seeds in tier0)."""
    ds_seed = 100_000 + int(round(beta * 1000))
    train = grf_images(n_train, size=size, beta=beta, channels=3, seed=ds_seed, std=std)
    clean = grf_images(n_eval, size=size, beta=beta, channels=3, seed=ds_seed + 1,
                       std=std).to(device)
    f, P = radial_spectrum(train[:512])
    beta_hat, r2 = fit_beta(f, P)
    loader = DataLoader(TensorDataset(train), batch_size=batch, shuffle=True, num_workers=0)
    return loader, clean, beta_hat, r2


def run_one(family, beta, seed, args, device, loader, clean):
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = FAMILIES[family]
    model = (_kan_ctor() if family == 'kan' else spec['ctor']()).to(device)
    tag = f"{family}-b{beta:g}-s{seed}"
    print(f"\n=== {tag} ({model.n_params:,} params) ===", flush=True)
    t0 = time.time()
    if spec['is_score']:
        train_score(model, loader, device, n_epochs=args.epochs, sigma_train=0.15)
        kstars = [kstar_score(model, clean, s, device) for s in SIGMAS]
    else:
        train_one(model, loader, device, n_epochs=args.epochs, sigma_train=0.15, tag=tag)
        kstars = [kstar_at_sigma(model, clean, s, device) for s in SIGMAS]
    alpha, r2 = fit_alpha(SIGMAS, kstars)
    torch.save(model.state_dict(), CKPT / f"{tag}.pt")
    print(f"  {tag}: K*={[round(k, 2) for k in kstars]}  alpha={alpha:.3f}  "
          f"R^2={r2:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    n_params = model.n_params
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return dict(family=family, beta_nominal=beta, seed=seed, kstars=kstars,
                alpha=alpha, r2=r2, n_params=n_params,
                epochs=args.epochs, n_train=args.n_train, batch=args.batch,
                sigmas=SIGMAS)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: slope of log(alpha) vs log(beta_hat); theory predicts -1
# ─────────────────────────────────────────────────────────────────────────────

def analyze(results):
    by_fam = {}
    for r in results.values():
        if not (isinstance(r, dict) and np.isfinite(r.get('alpha', float('nan')))):
            continue
        d = by_fam.setdefault(r['family'], {})
        e = d.setdefault(r['beta_nominal'], {'alphas': [], 'beta_hat': r.get('beta_hat')})
        e['alphas'].append(r['alpha'])

    summary = {}
    for fam, d in sorted(by_fam.items()):
        rows = []
        for b in sorted(d):
            a = np.array(d[b]['alphas'])
            bh = d[b]['beta_hat'] if d[b]['beta_hat'] else b
            rows.append((b, bh, a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, len(a)))
        print(f"\n[{fam}]")
        print(f"  {'beta':>5} | {'beta_hat':>8} | {'alpha (mean +/- std)':>22} | seeds")
        for b, bh, am, astd, ns in rows:
            print(f"  {b:>5g} | {bh:>8.3f} | {am:>12.3f} +/- {astd:.3f}   | {ns}")
        if len(rows) < 3:
            print("  (need >=3 beta values for a slope fit)")
            continue
        x = np.log([r[1] for r in rows])
        y = np.log([r[2] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        yh = intercept + slope * x
        ss_res = np.sum((y - yh) ** 2)
        r2 = 1 - ss_res / np.sum((y - y.mean()) ** 2)
        se = np.sqrt(ss_res / max(1, len(x) - 2) / np.sum((x - x.mean()) ** 2))
        gamma = float(np.exp(intercept) / 2.0)
        ab = np.array([r[1] * r[2] for r in rows])     # alpha * beta_hat (theory: = 2*gamma)
        cv = float(ab.std(ddof=1) / ab.mean()) if len(ab) > 1 else 0.0
        print(f"  fit:  log(alpha) = {intercept:+.3f} {slope:+.3f}*log(beta_hat)"
              f"   [slope se={se:.3f}, R^2={r2:.3f}]")
        print(f"  spectral account predicts slope = -1   ->   measured {slope:+.3f}"
              f" ({'within' if abs(slope + 1) <= 2 * se else 'OUTSIDE'} 2 se)")
        print(f"  implied gamma = exp(intercept)/2 = {gamma:.3f};"
              f"   alpha*beta_hat constancy: mean={ab.mean():.3f}, CV={cv:.1%}")
        summary[fam] = dict(rows=[list(r) for r in rows], slope=float(slope),
                            slope_se=float(se), intercept=float(intercept),
                            r2=float(r2), gamma_implied=gamma,
                            alpha_beta_mean=float(ab.mean()), alpha_beta_cv=cv)
    return summary


def make_figure(summary, path_stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    colors = plt.cm.tab10.colors
    for i, (fam, s) in enumerate(sorted(summary.items())):
        rows = np.array(s['rows'])                    # beta, beta_hat, mean, std, n
        bh, am, astd = rows[:, 1], rows[:, 2], rows[:, 3]
        c = colors[i % 10]
        ax1.errorbar(bh, am, yerr=astd, fmt='o', ms=4.5, capsize=3, color=c,
                     label=f"{fam}: slope {s['slope']:+.2f}±{s['slope_se']:.2f}")
        bb = np.linspace(bh.min(), bh.max(), 50)
        ax1.plot(bb, np.exp(s['intercept']) * bb ** s['slope'], '-', lw=1.2, color=c)
        # slope -1 guide anchored at the middle point
        mid = len(bh) // 2
        ax1.plot(bb, am[mid] * (bb / bh[mid]) ** -1.0, '--', lw=0.9, color=c, alpha=0.45)
        ax2.errorbar(bh, am * bh, yerr=astd * bh, fmt='s', ms=4.5, capsize=3, color=c,
                     label=f"{fam}: $2\\gamma$={s['alpha_beta_mean']:.2f} (CV {s['alpha_beta_cv']:.0%})")
        ax2.axhline(s['alpha_beta_mean'], color=c, lw=0.9, alpha=0.45)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'data spectral exponent $\hat\beta$')
    ax1.set_ylabel(r'inference-depth exponent $\alpha$')
    ax1.set_title(r'$\alpha$ vs $\hat\beta$ (theory: slope $-1$, dashed)')
    ax1.legend(fontsize=7.5, frameon=False)
    ax2.set_xlabel(r'data spectral exponent $\hat\beta$')
    ax2.set_ylabel(r'$\alpha \cdot \hat\beta$')
    ax2.set_title(r'$\alpha\hat\beta = 2\gamma$ constancy check')
    ax2.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{path_stem}.{ext}", dpi=180)
    plt.close(fig)
    print(f"\n[fig] {path_stem}.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--families', nargs='+', default=['convmlp_gelu', 'score'],
                   choices=list(FAMILIES))
    p.add_argument('--betas', type=float, nargs='+', default=[1.0, 1.5, 2.0, 2.5, 3.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--n_eval', type=int, default=64)
    p.add_argument('--batch', type=int, default=32, help='32 fits the KAN on a 10GB GPU')
    p.add_argument('--size', type=int, default=32)
    p.add_argument('--img_std', type=float, default=0.35)
    p.add_argument('--out', default='beta_sweep_results.json')
    p.add_argument('--merge', nargs='*', default=[],
                   help='extra results JSONs to merge in at analysis time')
    p.add_argument('--plot_only', action='store_true', help='analysis + figure from JSON only')
    p.add_argument('--quick', action='store_true',
                   help='smoke test: 3 epochs, 2K train, 16 eval, 1 seed, betas {1,2,3}')
    args = p.parse_args()

    if args.quick:
        args.epochs, args.n_train, args.n_eval = 3, 2000, 16
        args.seeds, args.betas = [0], [1.0, 2.0, 3.0]
        args.out = 'beta_sweep_quick.json'

    res_path = OUT / args.out
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    if not args.plot_only:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"[device] {device}"
              + (f" [{torch.cuda.get_device_name(0)}]" if device.type == 'cuda' else ""))
        n_total = len(args.betas) * len(args.families) * len(args.seeds)
        n_done = 0
        for beta in args.betas:
            loader, clean, beta_hat, bh_r2 = build_data(
                beta, args.n_train, args.n_eval, args.size, args.img_std,
                args.batch, device)
            print(f"\n##### beta={beta:g}  (realized beta_hat={beta_hat:.3f},"
                  f" R^2={bh_r2:.3f}) #####", flush=True)
            for family in args.families:
                for seed in args.seeds:
                    key = f"{family}|b{beta:g}|s{seed}"
                    n_done += 1
                    if key in results and np.isfinite(results[key].get('alpha', float('nan'))):
                        print(f"[skip {n_done}/{n_total}] {key} already done "
                              f"(alpha={results[key]['alpha']:.3f})")
                        continue
                    print(f"[run {n_done}/{n_total}]")
                    r = run_one(family, beta, seed, args, device, loader, clean)
                    r['beta_hat'] = beta_hat
                    r['beta_hat_r2'] = bh_r2
                    results[key] = r
                    res_path.write_text(json.dumps(results, indent=2))   # save after EVERY run
            del loader, clean
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        print(f"\n[json] {res_path}")

    for extra in args.merge:
        ep = Path(extra) if Path(extra).exists() else OUT / extra
        if ep.exists():
            merged = json.loads(ep.read_text())
            print(f"[merge] +{len(merged)} runs from {ep}")
            results.update(merged)
        else:
            print(f"[merge] NOT FOUND: {extra}")

    summary = analyze(results)
    if summary:
        stem = 'beta_sweep_quick' if args.quick else 'beta_sweep'
        (OUT / f'{stem}_summary.json').write_text(json.dumps(summary, indent=2))
        make_figure(summary, str(FIG_DIR / stem))


if __name__ == '__main__':
    main()
