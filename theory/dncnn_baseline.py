"""
DnCNN/feedforward baseline: shows feedforward denoisers have NO K-dial.
=========================================================================
Trains a micro-DnCNN (~30-80K params, matched to the EBM budget) on
CIFAR-10 with the same recipe (single σ=0.15, 40ep, AdamW+cosine).

The point is NOT to beat DnCNN on PSNR — a feedforward net of matched
size will probably win at K=1 (it doesn't waste capacity on a landscape).
The point is: feedforward models produce one output. There is no K. The
"compute-quality knob" is exclusive to iterative models. The law governs
how to set that knob.

Also trains a larger DnCNN (~300K) to serve as a "modern baseline upper
bound" for absolute PSNR comparison (addresses reviewer M3).

Run:  py -3.12 theory/dncnn_baseline.py --epochs 40
      py -3.12 theory/dncnn_baseline.py --quick   # 5 epochs, smoke test

Outputs: outputs/theory/dncnn_results.json, dncnn_comparison.pdf/.png
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
sys.path.insert(0, str(ROOT / 'theory'))
from tier0_seeds import get_data, SIGMAS, K_MAX, psnr_per_image, fit_alpha  # noqa: E402

OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)


class MicroDnCNN(nn.Module):
    """Feedforward denoiser: noisy → clean in ONE pass. No energy, no K-dial."""
    def __init__(self, ch=3, depth=5, width=48):
        super().__init__()
        layers = [nn.Conv2d(ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, ch, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)   # residual learning (predict noise)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


CONFIGS = {
    'dncnn_micro':  dict(depth=5,  width=48),    # ~33K params
    'dncnn_small':  dict(depth=7,  width=64),    # ~115K params
    'dncnn_medium': dict(depth=10, width=64),    # ~170K params
}


def train_dncnn(model, loader, device, n_epochs, sigma_train=0.15, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device)
            xn = x + torch.randn_like(x) * sigma_train
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = F.mse_loss(model(xn), x)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = F.mse_loss(model(xn), x)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep <= 3 or ep % 10 == 0 or ep == n_epochs:
            print(f"  [dncnn] ep {ep:3d}/{n_epochs} loss={tot/nb:.5f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)


def eval_dncnn(model, clean, device, eval_seeds=2):
    """Evaluate feedforward denoiser at each σ — single-pass, no K."""
    results = {}
    for sig in SIGMAS:
        psnrs = []
        for es in range(eval_seeds):
            torch.manual_seed(10_000 + es)
            with torch.no_grad():
                xn = clean + sig * torch.randn_like(clean)
                out = model(xn)
            psnrs.append(float(psnr_per_image(out, clean).mean()))
        results[f'{sig:.2f}'] = {'psnr': float(np.mean(psnrs)),
                                  'K': 1,
                                  'note': 'feedforward — single pass, no K-dial'}
    return results


def eval_ebm_kstar(clean, sig, device, model_class_and_ckpt):
    """Load an existing EBM checkpoint and get its K*-optimal PSNR for comparison."""
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--n_eval', type=int, default=64)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--configs', nargs='+', default=['dncnn_micro', 'dncnn_small'])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_train, args.n_eval = 5, 5000, 16
        args.seeds = [0]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}"
          + (f" [{torch.cuda.get_device_name(0)}]" if device.type == 'cuda' else ""))
    loader, test_ds = get_data(args.n_train, device, batch_size=args.batch)
    clean = torch.stack([test_ds[i][0] for i in range(args.n_eval)]).to(device)

    res_path = OUT / 'dncnn_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        results[cfg_name] = {'seeds': {}, 'config': cfg}
        for seed in args.seeds:
            key = f"s{seed}"
            torch.manual_seed(seed); np.random.seed(seed)
            model = MicroDnCNN(ch=3, **cfg).to(device)
            n_p = model.n_params
            print(f"\n=== {cfg_name} seed={seed} ({n_p:,} params) ===", flush=True)
            train_dncnn(model, loader, device, n_epochs=args.epochs)
            ev = eval_dncnn(model, clean, device)
            results[cfg_name]['seeds'][key] = ev
            results[cfg_name]['n_params'] = n_p
            for sig_key, v in ev.items():
                print(f"  sig={sig_key}: PSNR={v['psnr']:.2f} dB (K=1, feedforward)")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        res_path.write_text(json.dumps(results, indent=2))

    # Print comparison table
    print("\n" + "=" * 72)
    print("DnCNN (feedforward, K=1 only) -- PSNR per sigma")
    print("-" * 72)
    for cfg_name in args.configs:
        r = results[cfg_name]
        print(f"\n{cfg_name} ({r['n_params']:,} params):")
        for sig in SIGMAS:
            psnrs = [r['seeds'][sk][f'{sig:.2f}']['psnr']
                     for sk in r['seeds']]
            m = np.mean(psnrs)
            s = np.std(psnrs, ddof=1) if len(psnrs) > 1 else 0
            print(f"  sig={sig:.2f}: {m:.2f} +/- {s:.2f} dB")
    print(f"\nContrast: EBM models have K*(sigma) ~ C sigma^1.376. DnCNN has K=1 always.")
    print(f"DnCNN may win on absolute PSNR but has no compute-quality knob.")
    print(f"\n[json] {res_path}")

    make_figure(results, args.configs)


def make_figure(results, configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = plt.cm.tab10.colors
    for i, cfg in enumerate(configs):
        r = results[cfg]
        psnrs_mean = []
        for sig in SIGMAS:
            vals = [r['seeds'][sk][f'{sig:.2f}']['psnr'] for sk in r['seeds']]
            psnrs_mean.append(np.mean(vals))
        ax.plot(SIGMAS, psnrs_mean, 's--', ms=4, color=colors[i],
                label=f"{cfg} ({r['n_params']//1000}K, feedforward K=1)")

    # reference: EBM K=1 and K=K* lines from the tier0 data if available
    tier0_path = ROOT / 'outputs' / 'tier0_seeds' / 'tier0_results.json'
    if tier0_path.exists():
        t0 = json.loads(tier0_path.read_text())
        if 'convmlp_gelu' in t0:
            # K* line (the benefit of iterative refinement)
            ax.annotate('← EBMs have K*(σ) ≈ Cσ^α here\n    (iterative refinement)',
                        xy=(0.20, psnrs_mean[4] + 1.5), fontsize=8,
                        color=colors[-2], fontstyle='italic')

    ax.set_xlabel('noise level σ')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Feedforward baselines: strong at K=1, but no K-dial')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'dncnn_comparison.{ext}', dpi=180)
    plt.close(fig)
    print(f"[fig] {OUT / 'dncnn_comparison.pdf'}")


if __name__ == '__main__':
    main()
