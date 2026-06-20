"""
rebuttal_intermediate_sigmas.py
================================
Wave 1b: Add intermediate sigma anchors to push the K*(sigma) fit from
5 to 9 points.

Reviewer concern: "5 noise levels with a 2-parameter model (R^2=0.98 over
5 points is expected from a well-chosen functional form)."

Adds sigmas {0.07, 0.12, 0.17, 0.25} to the original {0.05, 0.10, 0.15,
0.20, 0.30} on the existing CIFAR-10 KAN-EBM. Re-fits the law with all 9
anchors, recomputes 95% CIs, and reports whether the additional points
materially change the fit.

Outputs:
  outputs/finalization/rebuttal/intermediate_sigmas.json
  outputs/finalization/rebuttal/intermediate_sigmas_summary.txt
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_cifar10 import KANEnergyModel as KAN_CIFAR  # noqa: E402

OUT_DIR = ROOT / 'outputs' / 'finalization' / 'rebuttal'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def sweep_kstar(model, batch_clean, sigmas, K_list, device,
                n_seeds=3, dt=0.05, dt_decay=0.97):
    out = {s: [] for s in sigmas}
    for seed in range(n_seeds):
        torch.manual_seed(SEED + seed * 11)
        for sigma in sigmas:
            xc = batch_clean.to(device)
            xn = xc + torch.randn_like(xc) * sigma
            psnrs = {}
            for K in K_list:
                with torch.enable_grad():
                    u = xn.clone()
                    step = dt
                    for _ in range(K):
                        ui = u.detach().requires_grad_(True)
                        E = model.energy(ui)
                        E = E.sum() if E.ndim > 0 else E
                        g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
                        u = (u - step * g).detach()
                        step *= dt_decay
                mse = F.mse_loss(u.clamp(-1, 1), xc).item()
                psnrs[K] = 100.0 if mse < 1e-10 else 10 * math.log10(4.0 / mse)
            kstar = max(psnrs.items(), key=lambda kv: kv[1])[0]
            out[sigma].append(kstar)
    return out


def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(intercept), float(slope), float(r2)


def parametric_bootstrap_ci(sigmas, K_means, B=10000):
    a, b, r2 = fit_power_law(sigmas, K_means)
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    yh = a + b * xs
    resid = ys - yh
    resid_c = resid - resid.mean()
    rng = np.random.default_rng(20260501)
    boot_alphas = np.empty(B)
    for i in range(B):
        e = rng.choice(resid_c, size=len(xs), replace=True)
        ys_boot = yh + e
        slope, _ = np.polyfit(xs, ys_boot, 1)
        boot_alphas[i] = slope
    return float(np.percentile(boot_alphas, 2.5)), float(np.percentile(boot_alphas, 97.5))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # Load CIFAR-10 KAN-EBM (110K params, large variant from results_f32.json)
    ckpt = ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt'
    model = KAN_CIFAR(n_filters=32, filter_size=5, kan_hidden=[96, 16],
                      n_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {ckpt.name}  ({n_params:,} params)")

    # Eval batch (24 CIFAR-10 test images, same as existing protocol)
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    n_eval = 24
    eval_batch = torch.stack([test_ds[i][0] for i in range(n_eval)]).to(device)

    # Original 5 sigmas (already known) + 4 intermediate
    sigmas_original = [0.05, 0.10, 0.15, 0.20, 0.30]
    sigmas_extra    = [0.07, 0.12, 0.17, 0.25]
    sigmas_all      = sorted(set(sigmas_original + sigmas_extra))
    K_list = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    n_seeds = 3

    print(f"  Re-running K* sweep on full 9-anchor sigma set: {sigmas_all}")
    t0 = time.time()
    kstar_seeds = sweep_kstar(model, eval_batch, sigmas_all, K_list, device,
                               n_seeds=n_seeds)
    print(f"  Sweep took {time.time()-t0:.1f}s")

    kstar_means = {s: float(np.mean(kstar_seeds[s])) for s in sigmas_all}
    kstar_stds  = {s: float(np.std(kstar_seeds[s], ddof=1)
                              if len(kstar_seeds[s]) > 1 else 0.0)
                   for s in sigmas_all}

    print("\nK*_obs per sigma:")
    for s in sigmas_all:
        marker = " *" if s in sigmas_extra else "  "
        print(f"  {marker} sigma={s:.3f}  K*={kstar_means[s]:5.2f} +- {kstar_stds[s]:.2f}  "
              f"(seeds={kstar_seeds[s]})")

    # Fit on original 5 vs full 9
    a5, b5, r2_5 = fit_power_law(sigmas_original,
                                  [kstar_means[s] for s in sigmas_original])
    a9, b9, r2_9 = fit_power_law(sigmas_all,
                                  [kstar_means[s] for s in sigmas_all])
    C5 = math.exp(a5)
    C9 = math.exp(a9)
    print(f"\n5-anchor fit:   alpha = {b5:.4f}   C = {C5:.2f}   R^2 = {r2_5:.4f}")
    print(f"9-anchor fit:   alpha = {b9:.4f}   C = {C9:.2f}   R^2 = {r2_9:.4f}")

    ci5_lo, ci5_hi = parametric_bootstrap_ci(sigmas_original,
                                              [kstar_means[s] for s in sigmas_original])
    ci9_lo, ci9_hi = parametric_bootstrap_ci(sigmas_all,
                                              [kstar_means[s] for s in sigmas_all])
    print(f"\n5-anchor 95% CI on alpha: [{ci5_lo:.3f}, {ci5_hi:.3f}]  "
          f"width={ci5_hi-ci5_lo:.3f}")
    print(f"9-anchor 95% CI on alpha: [{ci9_lo:.3f}, {ci9_hi:.3f}]  "
          f"width={ci9_hi-ci9_lo:.3f}")
    print(f"\nReduction in CI width: {(1 - (ci9_hi-ci9_lo)/(ci5_hi-ci5_lo))*100:.1f}%")

    record = {
        'n_params': n_params,
        'n_eval': n_eval,
        'n_seeds': n_seeds,
        'K_list': K_list,
        'sigmas_original': sigmas_original,
        'sigmas_extra': sigmas_extra,
        'sigmas_all': sigmas_all,
        'kstar_per_seed': {str(s): kstar_seeds[s] for s in sigmas_all},
        'kstar_means': {str(s): kstar_means[s] for s in sigmas_all},
        'kstar_stds':  {str(s): kstar_stds[s]  for s in sigmas_all},
        'fit_5_anchor': {'C': C5, 'alpha': b5, 'r_squared': r2_5,
                         'ci95_alpha': [ci5_lo, ci5_hi]},
        'fit_9_anchor': {'C': C9, 'alpha': b9, 'r_squared': r2_9,
                         'ci95_alpha': [ci9_lo, ci9_hi]},
        'ci_width_reduction_pct': (1 - (ci9_hi-ci9_lo)/(ci5_hi-ci5_lo))*100,
    }
    out_path = OUT_DIR / 'intermediate_sigmas.json'
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {out_path}")

    summary = (
        f"Intermediate-sigma extension (5 -> 9 anchors)\n"
        f"=============================================\n"
        f"Original 5 sigmas: {sigmas_original}\n"
        f"Added 4 sigmas:    {sigmas_extra}\n\n"
        f"Per-sigma K*:\n" +
        "\n".join(f"  sigma={s:.3f}  K* = {kstar_means[s]:.2f} +- {kstar_stds[s]:.2f}"
                  for s in sigmas_all) +
        f"\n\nFit comparison:\n"
        f"  5-anchor: alpha = {b5:.4f} +- {(ci5_hi-ci5_lo)/2:.3f}  C = {C5:.2f}  R^2 = {r2_5:.4f}\n"
        f"  9-anchor: alpha = {b9:.4f} +- {(ci9_hi-ci9_lo)/2:.3f}  C = {C9:.2f}  R^2 = {r2_9:.4f}\n"
        f"  CI width shrinks by {(1 - (ci9_hi-ci9_lo)/(ci5_hi-ci5_lo))*100:.1f}%\n"
    )
    (OUT_DIR / 'intermediate_sigmas_summary.txt').write_text(summary)


if __name__ == "__main__":
    main()
