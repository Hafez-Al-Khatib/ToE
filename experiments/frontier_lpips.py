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
