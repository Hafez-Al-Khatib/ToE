"""
Oracle-gain analysis (panel finding M3): what would per-image adaptive depth
actually buy, in dB?

For kan_32k on 500 CIFAR-10 test images at sigma in {0.10, 0.20, 0.30}
(seed 0, protocol-identical to the frontier parts), compare:
  fixed  : PSNR at the single K* maximizing the MEAN curve (the paper's rule)
  oracle : mean over images of each image's OWN best-K PSNR

oracle - fixed is the ceiling on what any per-image depth controller can add.
CPU-friendly (32K-param model). Writes outputs/review_fixes/oracle_gain.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)   # this machine segfaults in threaded CPU einsum

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_inference_frontier import (model_registry, load_test_images,
                                    noise_for, psnr_batch)
from exp_fine_grid_kstar import sequential_denoise_record

SIGMAS = [0.10, 0.20, 0.30]
N_IMAGES = 500
K_MAX = 30
BATCH = 100
SEED = 0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    device = torch.device(args.device)
    model = model_registry(device)['kan_32k']()
    images = load_test_images(N_IMAGES)
    out = {}
    for sigma in SIGMAS:
        noisy_all = (images + noise_for(images, sigma, SEED)).clamp(-1, 1)
        per_step = []
        for b in range(0, N_IMAGES, BATCH):
            clean = images[b:b + BATCH]
            noisy = noisy_all[b:b + BATCH]
            states = sequential_denoise_record(model, noisy, K_max=K_MAX)
            per_step.append(np.stack([psnr_batch(s, clean) for s in states]))
        per_step = np.concatenate(per_step, axis=1)      # (K_MAX, n)
        mean_curve = per_step.mean(axis=1)
        k_fixed = int(np.argmax(mean_curve))             # 0-based index
        fixed = float(per_step[k_fixed].mean())
        oracle = float(per_step.max(axis=0).mean())
        out[f'{sigma:.2f}'] = {
            'k_fixed': k_fixed + 1,
            'fixed_psnr': fixed,
            'oracle_psnr': oracle,
            'gain_db': oracle - fixed,
        }
        print(f'sigma={sigma:.2f}: fixed K*={k_fixed+1} {fixed:.3f} dB, '
              f'oracle {oracle:.3f} dB, gain {oracle-fixed:+.3f} dB')
    dst = ROOT / 'outputs/review_fixes/oracle_gain.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print('wrote', dst)


if __name__ == '__main__':
    main()
