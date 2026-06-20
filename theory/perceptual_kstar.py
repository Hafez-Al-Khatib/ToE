"""
Perceptual K*: does the scaling law hold under SSIM and LPIPS, not just PSNR?
=============================================================================
Reviewer concern: "K* that maximises PSNR may differ from K* that maximises
perceptual quality. If so, the scaling law is a PSNR artifact."

This script loads EXISTING tier0 checkpoints (no training) and runs the K*
sweep using three metrics:
  1. PSNR  (peak signal-to-noise ratio, pixel-level)
  2. SSIM  (structural similarity, window-based)
  3. LPIPS (learned perceptual image patch similarity, AlexNet backbone)

For each metric, we fit K*(sigma) = C sigma^alpha and report alpha + R^2.
If alpha_SSIM ~ alpha_PSNR ~ alpha_LPIPS, the law is metric-independent.

Run:  py -3.12 theory/perceptual_kstar.py
      py -3.12 theory/perceptual_kstar.py --quick
      py -3.12 theory/perceptual_kstar.py --archs kan convmlp_gelu

Outputs: outputs/theory/perceptual_kstar.json, perceptual_kstar.pdf/.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'theory'))

from torchvision import datasets, transforms
from exp_cifar10 import KANEnergyModel
from rebuttal_smooth_mlp import ConvSmoothMLPEBM
from tier0_seeds import SIGMAS, K_MAX, fit_alpha

OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / 'outputs' / 'tier0_seeds'

ARCHS = {
    'kan':          lambda: KANEnergyModel(n_filters=16, filter_size=5,
                                           kan_hidden=[48, 16], n_channels=3),
    'convmlp_gelu': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                             n_channels=3, activation='gelu'),
    'convmlp_silu': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                             n_channels=3, activation='silu'),
    'convmlp_tanh': lambda: ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                                             n_channels=3, activation='tanh'),
}


# --- Metric functions ---

def psnr_per_image(u, clean):
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).cpu().numpy()


def ssim_per_image(u, clean):
    """Compute SSIM per image using torchmetrics (handles batches efficiently)."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    u_01 = (u.clamp(-1, 1) + 1) / 2
    c_01 = (clean + 1) / 2
    vals = []
    for i in range(u.shape[0]):
        s = ssim_fn(u_01[i:i+1], c_01[i:i+1], data_range=1.0)
        vals.append(float(s))
    return np.array(vals)


_lpips_net = None

def _get_lpips():
    global _lpips_net
    if _lpips_net is None:
        import lpips
        _lpips_net = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            _lpips_net = _lpips_net.cuda()
    return _lpips_net


def lpips_per_image(u, clean):
    """Compute LPIPS per image (lower = better perceptual quality)."""
    net = _get_lpips()
    u_c = u.clamp(-1, 1)
    with torch.no_grad():
        d = net(u_c, clean)
    return d.squeeze().cpu().numpy()


METRICS = {
    'psnr':  {'fn': psnr_per_image,  'higher_better': True,  'unit': 'dB'},
    'ssim':  {'fn': ssim_per_image,  'higher_better': True,  'unit': ''},
    'lpips': {'fn': lpips_per_image, 'higher_better': False, 'unit': ''},
}


def kstar_multi_metric(model, clean, sigma, device, metrics, eval_seeds=2, dt=0.05, decay=0.97):
    """Run K-step gradient descent and track all metrics simultaneously.
    Returns dict: metric_name -> continuous K* (float)."""
    per_image = {m: [] for m in metrics}

    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        u = (clean + sigma * torch.randn_like(clean)).to(device)

        series = {m: [metrics[m]['fn'](u, clean)] for m in metrics}

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
            for m in metrics:
                series[m].append(metrics[m]['fn'](u, clean))

        for m in metrics:
            arr = np.stack(series[m], axis=0)  # (K+1, N)
            if metrics[m]['higher_better']:
                per_image[m].append(arr.argmax(axis=0))
            else:
                per_image[m].append(arr.argmin(axis=0))

    return {m: float(np.mean(per_image[m])) for m in metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--n_eval', type=int, default=64)
    p.add_argument('--archs', nargs='+', default=['kan', 'convmlp_gelu'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'lpips'])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.n_eval = 16
        args.seeds = [0]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}"
          + (f" [{torch.cuda.get_device_name(0)}]" if device.type == 'cuda' else ""))

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    clean = torch.stack([test_ds[i][0] for i in range(args.n_eval)]).to(device)

    active_metrics = {m: METRICS[m] for m in args.metrics}

    res_path = OUT / 'perceptual_kstar.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    for arch in args.archs:
        for seed in args.seeds:
            run_key = f"{arch}_s{seed}"
            if run_key in results and all(m in results[run_key] for m in args.metrics):
                print(f"[skip] {run_key} already done")
                continue

            ckpt_path = CKPT_DIR / f"{arch}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"[skip] {ckpt_path} not found")
                continue

            model = ARCHS[arch]()
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            model = model.to(device).eval()
            print(f"\n=== {run_key} ({model.n_params:,} params) ===", flush=True)

            run_result = {}
            for m in active_metrics:
                run_result[m] = {'kstars': [], 'alpha': None, 'r2': None}

            t0 = time.time()
            for sig in SIGMAS:
                ks_dict = kstar_multi_metric(model, clean, sig, device, active_metrics)
                for m in active_metrics:
                    run_result[m]['kstars'].append(ks_dict[m])
                ks_str = "  ".join(f"{m}={ks_dict[m]:.2f}" for m in active_metrics)
                print(f"  sig={sig:.2f}: {ks_str}", flush=True)

            for m in active_metrics:
                alpha, r2 = fit_alpha(SIGMAS, run_result[m]['kstars'])
                run_result[m]['alpha'] = alpha
                run_result[m]['r2'] = r2

            results[run_key] = run_result
            elapsed = time.time() - t0
            print(f"  [{run_key}] elapsed={elapsed:.0f}s", flush=True)

            for m in active_metrics:
                better = "higher" if active_metrics[m]['higher_better'] else "lower"
                print(f"    {m} ({better} better): alpha={run_result[m]['alpha']:.3f}  "
                      f"R2={run_result[m]['r2']:.3f}  "
                      f"K*={[f'{k:.1f}' for k in run_result[m]['kstars']]}")

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            res_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*80}")
    print("PERCEPTUAL K* SUMMARY: alpha by metric")
    print(f"{'='*80}")
    print(f"{'run':>20} |" + "".join(f" {'alpha_'+m:>12} {'R2_'+m:>8} |" for m in args.metrics))
    print("-" * 80)

    all_alphas = {m: [] for m in args.metrics}
    for run_key in sorted(results):
        r = results[run_key]
        parts = [f"{run_key:>20} |"]
        for m in args.metrics:
            if m in r:
                a = r[m]['alpha']
                r2 = r[m]['r2']
                parts.append(f" {a:>12.3f} {r2:>8.3f} |")
                if np.isfinite(a):
                    all_alphas[m].append(a)
            else:
                parts.append(f" {'N/A':>12} {'N/A':>8} |")
        print("".join(parts))

    print("-" * 80)
    parts = [f"{'MEAN +/- STD':>20} |"]
    for m in args.metrics:
        vals = all_alphas[m]
        if vals:
            mean_a = np.mean(vals)
            std_a = np.std(vals, ddof=1) if len(vals) > 1 else 0
            parts.append(f" {mean_a:>6.3f}+/-{std_a:<4.3f} {'':>8} |")
        else:
            parts.append(f" {'N/A':>12} {'':>8} |")
    print("".join(parts))

    # Metric agreement assessment
    print(f"\n--- Metric Agreement ---")
    for m1 in args.metrics:
        for m2 in args.metrics:
            if m1 >= m2:
                continue
            v1, v2 = all_alphas[m1], all_alphas[m2]
            if v1 and v2:
                diff = abs(np.mean(v1) - np.mean(v2))
                pct = diff / np.mean(v1) * 100
                ok = "AGREE" if pct < 10 else "MARGINAL" if pct < 20 else "DISAGREE"
                print(f"  {m1} vs {m2}: |delta_alpha| = {diff:.3f} ({pct:.1f}%) -> {ok}")

    print(f"\n[json] {res_path}")
    make_figure(results, args.metrics)


def make_figure(results, metric_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.5 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    sigmas = np.array(SIGMAS)

    for mi, metric in enumerate(metric_names):
        ax = axes[mi]
        alphas_all = []
        for ri, run_key in enumerate(sorted(results)):
            r = results[run_key]
            if metric not in r:
                continue
            kstars = np.array(r[metric]['kstars'])
            alpha = r[metric]['alpha']
            r2 = r[metric]['r2']
            ok = kstars > 0
            if ok.any():
                ax.plot(sigmas[ok], kstars[ok], 'o', ms=4, color=colors[ri % 10], alpha=0.6)
                # fit line
                if np.isfinite(alpha):
                    sig_fit = np.linspace(sigmas[ok].min(), sigmas[ok].max(), 50)
                    C = np.exp(np.polyfit(np.log(sigmas[ok]), np.log(kstars[ok]), 1)[1])
                    ax.plot(sig_fit, C * sig_fit ** alpha, '--', color=colors[ri % 10],
                            lw=1, alpha=0.5, label=f"{run_key} a={alpha:.2f}")
                    alphas_all.append(alpha)

        if alphas_all:
            mean_a = np.mean(alphas_all)
            std_a = np.std(alphas_all, ddof=1) if len(alphas_all) > 1 else 0
            ax.set_title(f"{metric.upper()}: alpha = {mean_a:.3f} +/- {std_a:.3f}")
        else:
            ax.set_title(f"{metric.upper()}")
        ax.set_xlabel('sigma')
        ax.set_ylabel(f'K* ({metric})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.15)
        ax.legend(fontsize=6, frameon=False)

    fig.suptitle("K*(sigma) scaling law: metric independence test", fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'perceptual_kstar.{ext}', dpi=180)
    plt.close(fig)
    print(f"[fig] {OUT / 'perceptual_kstar.pdf'}")


if __name__ == '__main__':
    main()
