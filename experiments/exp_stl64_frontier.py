# experiments/exp_stl64_frontier.py
"""STL-10 64x64 frontier sweep + pre-registered verdict (extension of the
CIFAR frontier; registration in .superpowers/sdd/progress.md, 2026-07-18).

Protocol identical to exp_inference_frontier: K=0..30, sigmas
{0.05,0.10,0.15,0.20,0.30}, 500 held-out test images (STL-10 labeled test
split, resized 64x64), evaluation-noise seeds {0,1}, deterministic noise per
(sigma, seed) cell, FLOPs via frontier_flops. Resumable part files in
outputs/stl64/parts/. When all 20 cells are done, applies
frontier_analysis.evaluate_pass (0.3 dB / >=3 of 5 sigmas / both seeds) to
kan_32k_stl64 (light) vs kan_110k_stl64 (heavy) and records peak gaps.

Run: py experiments/exp_stl64_frontier.py --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel
from exp_inference_frontier import run_cell
from frontier_flops import flops_per_step_detail
from frontier_analysis import evaluate_pass

SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]
SEEDS = [0, 1]
K_MAX = 30
N_IMAGES = 500
LIGHT, HEAVY = 'kan_32k_stl64', 'kan_110k_stl64'
VARIANTS = {
    LIGHT: dict(n_filters=16, kan_hidden=[48, 16]),
    HEAVY: dict(n_filters=32, kan_hidden=[96, 16]),
}
PARTS = ROOT / 'outputs' / 'stl64' / 'parts'


def load_model(tag, device):
    m = KANEnergyModel(**VARIANTS[tag]).to(device)
    m.load_state_dict(torch.load(ROOT / f'outputs/stl64/{tag}.pt',
                                 map_location=device, weights_only=True))
    m.eval()
    return m


def load_stl_test(n):
    tf = T.Compose([T.Resize(64), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.STL10(ROOT / 'data', split='test',
                                    download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)])


def sweep(device):
    PARTS.mkdir(parents=True, exist_ok=True)
    images = load_stl_test(N_IMAGES)
    for tag in VARIANTS:
        model = None
        flops = None
        for sigma in SIGMAS:
            for seed in SEEDS:
                path = PARTS / f'{tag}_s{sigma:.2f}_seed{seed}.json'
                if path.exists() and json.loads(path.read_text()).get('done'):
                    continue
                if model is None:
                    model = load_model(tag, device)
                    flops = flops_per_step_detail(model, device,
                                                  img_shape=(3, 64, 64))
                    print(f'[{tag}] {flops["total"]/1e6:.1f} MF/step '
                          f'({flops["method"]})')
                payload = run_cell(model, images, sigma, seed, K_MAX, device)
                payload.update(model=tag, flops_step=flops['total'],
                               flops_method=flops['method'], done=True)
                path.write_text(json.dumps(payload))
                print(f'[{tag}] sigma={sigma:.2f} seed={seed} done '
                      f'(K*_mean={np.mean(payload["kstar_per_image"]):.1f})')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def analyze():
    cells = {}
    for tag in VARIANTS:
        cells[tag] = {}
        for sigma in SIGMAS:
            s = f'{sigma:.2f}'
            cells[tag][s] = {}
            for seed in SEEDS:
                p = PARTS / f'{tag}_s{s}_seed{seed}.json'
                if not p.exists():
                    print(f'[analyze] missing {p.name}; sweep incomplete')
                    return
                d = json.loads(p.read_text())
                cells[tag][s][seed] = {k: d[k] for k in
                                       ('psnr_per_step', 'psnr_k0',
                                        'flops_step')}
    verdict = evaluate_pass(cells, LIGHT, HEAVY)
    peaks = {}
    for s in cells[LIGHT]:
        pk = {t: float(np.mean([max(cells[t][s][sd]['psnr_per_step'])
                                for sd in SEEDS])) for t in VARIANTS}
        peaks[s] = {'light': pk[LIGHT], 'heavy': pk[HEAVY],
                    'gap_heavy_minus_light': pk[HEAVY] - pk[LIGHT]}
    out = {'preregistration': '.superpowers/sdd/progress.md 2026-07-18 entry',
           'dataset': 'STL-10 test split, 500 images, 64x64',
           'verdict': verdict, 'peaks': peaks}
    dst = ROOT / 'outputs/stl64/stl64_frontier.json'
    dst.write_text(json.dumps(out, indent=1))
    print(f'[STL64] VERDICT: {verdict["verdict"]} '
          f'(per-seed sigma-pass {verdict["per_seed_pass_count"]})')
    for s, p in peaks.items():
        print(f'  sigma={s}: peak light {p["light"]:.2f}, heavy '
              f'{p["heavy"]:.2f}, gap {p["gap_heavy_minus_light"]:+.3f} dB')
    print('wrote', dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()
    if not args.analyze:
        sweep(torch.device(args.device))
    analyze()


if __name__ == '__main__':
    main()
