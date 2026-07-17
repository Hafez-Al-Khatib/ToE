# experiments/frontier_timing.py
"""Wall-clock per descent step per model at batch sizes {1, 32, 256}
(panel R3-W1: the frontier trades a parallel resource for a serial one;
FLOPs alone do not price latency).

GPU: warmup 3 steps, then time K=10 steps x 3 repeats, report ms/step/image.
Writes outputs/inference_frontier/timing.json.

Run: py experiments/frontier_timing.py --device cuda [--models a b c]
"""
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_inference_frontier import model_registry, load_test_images, noise_for
from exp_fine_grid_kstar import sequential_denoise_record

BATCHES = [1, 32, 256]
K_TIME = 10
REPEATS = 3
SIGMA = 0.10


def time_model(model, images, device):
    out = {}
    for bs in BATCHES:
        x = (images[:bs] + noise_for(images[:bs], SIGMA, 0)).clamp(-1, 1)
        x = x.to(device)
        sequential_denoise_record(model, x, K_max=3)          # warmup
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(REPEATS):
            t0 = time.time()
            sequential_denoise_record(model, x, K_max=K_TIME)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        best = min(times)
        out[str(bs)] = {'ms_per_step_per_image': 1000.0 * best / (K_TIME * bs),
                        'ms_per_step_batch': 1000.0 * best / K_TIME}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--models', nargs='+', default=None)
    args = p.parse_args()
    device = torch.device(args.device)
    registry = model_registry(device)
    names = args.models or ['kan_110k', 'kan_32k', 'conv_mlp_gelu', 'unet',
                            'group_kan_8k', 'group_kan_32k']
    images = load_test_images(max(BATCHES))
    result = {'device': (torch.cuda.get_device_name(0)
                         if device.type == 'cuda' else 'cpu'),
              'sigma': SIGMA, 'k_time': K_TIME, 'repeats': REPEATS,
              'models': {}}
    for name in names:
        try:
            model = registry[name]()
        except Exception as e:
            print(f'[skip] {name}: {e}')
            continue
        result['models'][name] = time_model(model, images, device)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        r = result['models'][name]
        print(f"{name:16s} " + "  ".join(
            f"bs{bs}: {r[str(bs)]['ms_per_step_per_image']:.4f} ms/img"
            for bs in BATCHES))
    dst = ROOT / 'outputs/inference_frontier/timing.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=1))
    print('wrote', dst)


if __name__ == '__main__':
    main()
