"""
Cross-dataset alpha validation: test predictive-alpha on FashionMNIST and MNIST.
================================================================================
The predictive-alpha analysis (predictive_alpha.py) calibrated gamma from
CIFAR-10 and CelebA-64, then predicted:

    Fashion: alpha_pred = 1.837  (beta_hat = 2.278)
    MNIST:   alpha_pred = TBD    (beta_hat = TBD, measured here)

This script:
  1. Measures beta_hat for MNIST (Fashion already measured).
  2. Trains convmlp_gelu + score models on each dataset (1-channel, 28x28).
  3. Runs the standard K* sweep and fits alpha.
  4. Compares measured alpha to predicted alpha.
  5. If |error| < 15%, predictive-alpha gains a 3rd (and 4th) data point.

Run:  py -3.12 theory/cross_dataset_alpha.py
      py -3.12 theory/cross_dataset_alpha.py --quick          # smoke test
      py -3.12 theory/cross_dataset_alpha.py --datasets fashion
      py -3.12 theory/cross_dataset_alpha.py --datasets mnist

Outputs: outputs/cross_dataset/cross_dataset_results.json
"""
import argparse
import json
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
sys.path.insert(0, str(ROOT / 'theory'))

from torchvision import datasets, transforms
from rebuttal_smooth_mlp import ConvSmoothMLPEBM, train_one
from phase_b_diffusion import ScoreNet, train_score
from measure_beta import load_images, radial_spectrum, fit_beta
from tier0_seeds import psnr_per_image, fit_alpha

OUT = ROOT / 'outputs' / 'cross_dataset'
OUT.mkdir(parents=True, exist_ok=True)

SIGMAS = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
K_MAX = 30

DATASET_INFO = {
    'fashion': {
        'loader': lambda root: datasets.FashionMNIST(
            root / 'data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])),
        'test_loader': lambda root: datasets.FashionMNIST(
            root / 'data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])),
        'n_channels': 1,
        'size': 28,
    },
    'mnist': {
        'loader': lambda root: datasets.MNIST(
            root / 'data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])),
        'test_loader': lambda root: datasets.MNIST(
            root / 'data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])),
        'n_channels': 1,
        'size': 28,
    },
}


def get_data(ds_name, n_train, batch_size=32):
    info = DATASET_INFO[ds_name]
    train_ds = info['loader'](ROOT)
    test_ds = info['test_loader'](ROOT)
    subset = torch.utils.data.Subset(train_ds, list(range(min(n_train, len(train_ds)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, test_ds, info['n_channels']


def measure_beta_for_dataset(ds_name, n_images=2000):
    """Measure power-spectrum exponent beta_hat for a dataset."""
    imgs = load_images(ds_name, n_images)
    if imgs.shape[1] == 1:
        imgs_spec = imgs.repeat(1, 3, 1, 1)
    else:
        imgs_spec = imgs
    f, P = radial_spectrum(imgs_spec)
    beta_hat, r2 = fit_beta(f, P)
    print(f"  [{ds_name}] beta_hat = {beta_hat:.3f}  (R2 = {r2:.4f}, {imgs.shape[0]} images @ {imgs.shape[-1]}px)")
    return beta_hat, r2


def kstar_ebm(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Per-image continuous K* for energy-based model."""
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
        arr = np.stack(series, axis=0)
        per_image.append(arr.argmax(axis=0))
    return float(np.mean(per_image))


def kstar_score(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Per-image continuous K* for score model."""
    per_image = []
    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        u = (clean + sigma * torch.randn_like(clean)).to(device)
        series = [psnr_per_image(u, clean)]
        step = dt
        for _ in range(K_MAX):
            with torch.no_grad():
                s = model(u).clamp(-1, 1)
            u = u + step * s
            step *= decay
            series.append(psnr_per_image(u, clean))
        arr = np.stack(series, axis=0)
        per_image.append(arr.argmax(axis=0))
    return float(np.mean(per_image))


FAMILIES = {
    'convmlp_gelu': {
        'build': lambda ch: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                              n_channels=ch, activation='gelu'),
        'is_score': False,
    },
    'score': {
        'build': lambda ch: ScoreNet(ch=ch, h=64),
        'is_score': True,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--n_eval', type=int, default=64)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--datasets', nargs='+', default=['fashion', 'mnist'])
    p.add_argument('--families', nargs='+', default=['convmlp_gelu', 'score'])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_train, args.n_eval = 5, 5000, 16
        args.seeds = [0]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}"
          + (f" [{torch.cuda.get_device_name(0)}]" if device.type == 'cuda' else ""))

    # Load existing predictive-alpha calibration
    pred_path = ROOT / 'outputs' / 'theory' / 'predictive_alpha.json'
    if pred_path.exists():
        pred_data = json.loads(pred_path.read_text())
        gamma_mean = pred_data['gamma_mean']
        print(f"[predictive-alpha] gamma_bar = {gamma_mean:.3f} (from CIFAR+CelebA calibration)")
    else:
        gamma_mean = None
        print("[predictive-alpha] No calibration found -- will measure alpha only")

    # Load existing beta measurements
    beta_path = ROOT / 'outputs' / 'theory' / 'data_beta.json'
    beta_data = json.loads(beta_path.read_text()) if beta_path.exists() else {}

    res_path = OUT / 'cross_dataset_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    for ds_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        # Step 1: measure or load beta_hat
        if ds_name in beta_data and 'beta_hat' in beta_data[ds_name]:
            beta_hat = beta_data[ds_name]['beta_hat']
            print(f"  [beta] loaded beta_hat = {beta_hat:.3f} from data_beta.json")
        else:
            print(f"  [beta] measuring beta_hat for {ds_name}...")
            beta_hat, beta_r2 = measure_beta_for_dataset(ds_name)
            beta_data[ds_name] = {'beta_hat': beta_hat, 'r2': beta_r2,
                                  'n_images': 2000, 'size': DATASET_INFO[ds_name]['size']}
            (ROOT / 'outputs' / 'theory' / 'data_beta.json').write_text(
                json.dumps(beta_data, indent=2))

        # Step 2: compute predicted alpha
        if gamma_mean is not None:
            alpha_pred = 2 * gamma_mean / beta_hat
            print(f"  [prediction] alpha_pred = 2 * {gamma_mean:.3f} / {beta_hat:.3f} = {alpha_pred:.3f}")
        else:
            alpha_pred = None

        if ds_name not in results:
            results[ds_name] = {'beta_hat': beta_hat, 'alpha_pred': alpha_pred, 'families': {}}

        # Step 3: load data
        loader, test_ds, n_ch = get_data(ds_name, args.n_train, batch_size=args.batch)
        clean = torch.stack([test_ds[i][0] for i in range(min(args.n_eval, len(test_ds)))]).to(device)
        print(f"  [data] {ds_name}: {n_ch}ch x {clean.shape[-1]}px, {args.n_train} train, {clean.shape[0]} eval")

        # Step 4: train + K-sweep for each family
        for fam_name in args.families:
            fam = FAMILIES[fam_name]
            if fam_name not in results[ds_name]['families']:
                results[ds_name]['families'][fam_name] = {'alpha': [], 'r2': [], 'kstars': []}

            for seed in args.seeds:
                run_key = f"{fam_name}|{ds_name}|s{seed}"
                # Skip if already completed
                existing_alphas = results[ds_name]['families'][fam_name]['alpha']
                if len(existing_alphas) > seed:
                    print(f"\n  [skip] {run_key} already completed (alpha={existing_alphas[seed]:.3f})")
                    continue

                torch.manual_seed(seed)
                np.random.seed(seed)
                model = fam['build'](n_ch).to(device)
                print(f"\n  === {run_key} ({model.n_params:,} params) ===", flush=True)

                if fam['is_score']:
                    train_score(model, loader, device, n_epochs=args.epochs,
                                sigma_train=0.15, lr=3e-4)
                else:
                    train_one(model, loader, device, n_epochs=args.epochs,
                              sigma_train=0.15, tag=run_key)

                # K-sweep
                kstars = []
                for sig in SIGMAS:
                    if fam['is_score']:
                        ks = kstar_score(model, clean, sig, device)
                    else:
                        ks = kstar_ebm(model, clean, sig, device)
                    kstars.append(ks)

                alpha, r2 = fit_alpha(SIGMAS, kstars)
                results[ds_name]['families'][fam_name]['alpha'].append(alpha)
                results[ds_name]['families'][fam_name]['r2'].append(r2)
                results[ds_name]['families'][fam_name]['kstars'].append(kstars)
                results[ds_name]['families'][fam_name]['n_params'] = model.n_params

                print(f"  {run_key}: K*={[f'{k:.2f}' for k in kstars]}")
                print(f"  {run_key}: alpha={alpha:.3f}  R2={r2:.3f}", flush=True)

                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Save after every run (resume-safe)
                res_path.write_text(json.dumps(results, indent=2))

        # Step 5: summary for this dataset
        print(f"\n--- {ds_name} summary ---")
        print(f"  beta_hat = {beta_hat:.3f}")
        if alpha_pred is not None:
            print(f"  alpha_pred = {alpha_pred:.3f}")
        for fam_name in args.families:
            r = results[ds_name]['families'].get(fam_name, {})
            alphas = [a for a in r.get('alpha', []) if np.isfinite(a)]
            if alphas:
                a_mean = np.mean(alphas)
                a_std = np.std(alphas, ddof=1) if len(alphas) > 1 else 0
                print(f"  {fam_name}: alpha = {a_mean:.3f} +/- {a_std:.3f} ({len(alphas)} seeds)")
                if alpha_pred is not None:
                    err = a_mean - alpha_pred
                    rel_err = err / alpha_pred
                    ok = "OK" if abs(rel_err) < 0.10 else "~" if abs(rel_err) < 0.20 else "X"
                    print(f"    {ok} prediction error: {err:+.3f} ({rel_err:+.1%})")

    # Final comparison table
    print(f"\n{'='*72}")
    print("CROSS-DATASET PREDICTIVE-ALPHA VALIDATION")
    print(f"{'='*72}")
    print(f"{'dataset':>12} | {'beta_hat':>8} | {'alpha_pred':>10} | {'alpha_meas':>10} | {'error':>8} | {'verdict':>8}")
    print("-" * 72)

    all_ds = ['cifar10', 'celeba64'] + args.datasets
    for ds in all_ds:
        if ds in results:
            r = results[ds]
            b = r.get('beta_hat', 0)
            ap = r.get('alpha_pred')
            # average measured alpha across all families
            all_alphas = []
            for fam_r in r.get('families', {}).values():
                all_alphas.extend([a for a in fam_r.get('alpha', []) if np.isfinite(a)])
            if all_alphas:
                am = np.mean(all_alphas)
                if ap is not None:
                    err = am - ap
                    rel = err / ap
                    v = "OK" if abs(rel) < 0.10 else "~" if abs(rel) < 0.20 else "FAIL"
                    print(f"{ds:>12} | {b:>8.3f} | {ap:>10.3f} | {am:>10.3f} | {rel:>+7.1%} | {v:>8}")
                else:
                    print(f"{ds:>12} | {b:>8.3f} | {'N/A':>10} | {am:>10.3f} | {'N/A':>8} | {'cal':>8}")
        elif ds in beta_data:
            b = beta_data[ds].get('beta_hat', 0)
            ap_val = 2 * gamma_mean / b if gamma_mean else None
            ap_str = f"{ap_val:.3f}" if ap_val else "N/A"
            known_alpha = beta_data[ds].get('alpha')
            if known_alpha:
                err = known_alpha - ap_val if ap_val else None
                rel = err / ap_val if err is not None else None
                v = "OK" if rel and abs(rel) < 0.10 else "~" if rel and abs(rel) < 0.20 else "ref"
                print(f"{ds:>12} | {b:>8.3f} | {ap_str:>10} | {known_alpha:>10.3f} | {rel:>+7.1%} | {v:>8}")
            else:
                print(f"{ds:>12} | {b:>8.3f} | {ap_str:>10} | {'???':>10} | {'???':>8} | {'wait':>8}")

    print(f"\n[json] {res_path}")

    # Update predictive_alpha.json with new measurements
    update_predictive_alpha(results, beta_data, gamma_mean)


def update_predictive_alpha(results, beta_data, gamma_mean):
    """Update the predictive_alpha.json with newly measured alphas."""
    pred_path = ROOT / 'outputs' / 'theory' / 'predictive_alpha.json'
    if not pred_path.exists():
        return
    pred = json.loads(pred_path.read_text())
    updated = False

    for ds_name, r in results.items():
        all_alphas = []
        for fam_r in r.get('families', {}).values():
            all_alphas.extend([a for a in fam_r.get('alpha', []) if np.isfinite(a)])
        if not all_alphas:
            continue

        a_mean = float(np.mean(all_alphas))
        a_std = float(np.std(all_alphas, ddof=1)) if len(all_alphas) > 1 else 0.0
        b = r.get('beta_hat', beta_data.get(ds_name, {}).get('beta_hat'))
        if b is None:
            continue

        # Add to measured_alphas
        pred['measured_alphas'][ds_name] = {
            'alpha_mean': a_mean,
            'alpha_std': a_std,
            'n_runs': len(all_alphas),
            'source': f'cross_dataset_alpha ({ds_name})',
            'resolution': f"{DATASET_INFO.get(ds_name, {}).get('size', '?')}px",
        }
        # Move from unmeasured to loo
        if ds_name in pred.get('unmeasured_predictions', {}):
            p = pred['unmeasured_predictions'].pop(ds_name)
            pred['loo_predictions'][ds_name] = {
                'alpha_true': a_mean,
                'alpha_pred': p['alpha_pred'],
                'error': p['alpha_pred'] - a_mean,
                'rel_error': (p['alpha_pred'] - a_mean) / a_mean,
                'beta_hat': b,
                'gamma_cal': gamma_mean,
                'calibrated_on': ['cifar10', 'celeba64'],
            }
        # Recompute gamma
        gamma_new = a_mean * b / 2.0
        pred['gammas'][ds_name] = gamma_new
        updated = True

    if updated:
        # Recompute gamma stats
        g_vals = list(pred['gammas'].values())
        pred['gamma_mean'] = float(np.mean(g_vals))
        pred['gamma_std'] = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else 0.0
        pred['gamma_cv'] = pred['gamma_std'] / pred['gamma_mean'] if pred['gamma_mean'] > 0 else float('nan')
        pred_path.write_text(json.dumps(pred, indent=2))
        print(f"\n[updated] {pred_path}")
        print(f"  gamma_bar = {pred['gamma_mean']:.3f} +/- {pred['gamma_std']:.3f} (CV = {pred['gamma_cv']:.1%})")
        print(f"  datasets with measured alpha: {list(pred['gammas'].keys())}")


if __name__ == '__main__':
    main()
