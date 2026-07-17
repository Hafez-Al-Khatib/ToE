# experiments/exp_inference_frontier.py
"""Fixed-FLOP inference frontier: head capacity vs descent depth.

Spec: docs/superpowers/specs/2026-07-02-inference-frontier-groupkan-design.md
Sweep phase writes resumable part files; --analyze consolidates them into
outputs/inference_frontier/frontier.json + frontier.png and prints the
pre-registered verdict (primary pair: kan_32k vs kan_110k).

Models: kan_110k, kan_32k, conv_mlp_gelu, unet. 'unet' FLOP counting hits a
torch 2.5.1 FlopCounterMode/autograd.grad bug on UNetEBM's backward graph
(see frontier_flops.py docstring and .superpowers/sdd/task-3-report.md);
frontier_flops.flops_per_step_detail() transparently falls back to
forward-only-measurement x2 for any model that trips this bug, so 'unet'
stays in the sweep. Check 'flops_method' ('measured' vs 'forward_x2') in
each part file / frontier.json meta entry to see which path was used.

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
from frontier_flops import flops_per_step_detail
from frontier_analysis import curve_from_psnr, evaluate_pass
from group_kan import GroupKANEnergyModel

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


def _load_group_kan(path, device, hidden):
    model = GroupKANEnergyModel(hidden=hidden).to(device)
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False))
    model.eval()
    return model


def model_registry(device):
    """name -> (loader lambda). Params strictly ordered for the primary pair.

    NOTE: 'unet' FLOP measurement hits a torch 2.5.1 FlopCounterMode/
    autograd interaction bug ("A leaf node was passed to
    _will_engine_execute_node but we are currently running autograd.grad()")
    on UNetEBM's MaxPool2d -> ConvTranspose2d backward graph when the full
    forward+backward is measured inside a single FlopCounterMode block.
    Plain torch.autograd.grad on the same model works fine (confirmed via
    minimal repro); only wrapping it in FlopCounterMode fails. Rather than
    dropping 'unet' from the registry, frontier_flops.flops_per_step_detail
    now falls back to forward-only measurement x2 (validated FLOP identity:
    total = forward + grad_input-backward approx= 2x forward) whenever the
    full measurement raises RuntimeError, so 'unet' is included here with
    its flops accounted via that fallback (see 'flops_method' in each part
    file / meta entry).
    """
    def _load_ladder(path, hidden):
        from rebuttal_smooth_mlp import ConvSmoothMLPEBM
        m = ConvSmoothMLPEBM(n_filters=16, mlp_hidden=hidden,
                             activation='gelu').to(device)
        m.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=True))
        m.eval()
        return m

    reg = {
        'kan_110k': lambda: load_kan(ROOT / 'outputs/cifar10/kan_ebm_f32.pt',
                                     device, n_filters=32, kan_hidden=[96, 16]),
        'kan_32k': lambda: load_kan(
            ROOT / 'outputs/finalization/kstar_param_scaling/kan_small.pt',
            device, n_filters=16, kan_hidden=[48, 16]),
        'conv_mlp_gelu': lambda: load_conv_mlp(
            ROOT / 'outputs/finalization/rebuttal/conv_mlp_gelu.pt', device),
        'unet': lambda: load_unet_ebm(ROOT / 'outputs/unet_ebm/unet_ebm.pt',
                                      device),
        'group_kan_8k': lambda: _load_group_kan(
            ROOT / 'outputs/group_kan/group_kan_8k.pt', device, hidden=136),
        'group_kan_32k': lambda: _load_group_kan(
            ROOT / 'outputs/group_kan/group_kan_32k.pt', device, hidden=640),
    }
    # Batch-v2 additions (panel R1-W4 replication + scale ladder). Same
    # matched recipe as train_group_kan.py; trained by
    # train_replication_ladder.py. Registered unconditionally; loading
    # fails loudly if the checkpoint is absent.
    kan_sizes = {'kan_32k': dict(n_filters=16, kan_hidden=[48, 16]),
                 'kan_110k': dict(n_filters=32, kan_hidden=[96, 16])}
    for _tag, _kw in kan_sizes.items():
        for _ts in (1, 2):
            reg[f'{_tag}_ts{_ts}'] = (
                lambda t=_tag, k=_kw, s=_ts: load_kan(
                    ROOT / f'outputs/replication/{t}_ts{s}.pt', device, **k))
    for _tag, _hidden in {'ladder_1m': 1000, 'ladder_8m': 2800,
                          'ladder_30m': 5450}.items():
        reg[_tag] = (lambda t=_tag, h=_hidden: _load_ladder(
            ROOT / f'outputs/scale_ladder/{t}.pt', h))
    return reg


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
        detail = flops_per_step_detail(model, device)
        fps = detail['total']
        print(f"[model] {name}: {n_params:,} params, "
              f"{fps / 1e6:.1f} MFLOPs/step/image "
              f"(flops_method={detail['method']})")
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
                                'flops_step': fps,
                                'flops_method': detail['method'],
                                'done': True})
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
                   'flops_method': d.get('flops_method', 'measured'),
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
                          'per step = one energy forward + input-gradient backward; '
                          'matmul/conv FLOPs via FlopCounterMode; '
                          'per-model method in meta[<model>].flops_method: "measured" '
                          '(counted directly) or "forward_x2" '
                          '(2 x forward-only count; validated ratio 2.0 on measured heads)'},
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
