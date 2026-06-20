"""
exp_diffusion_kstar_schedule.py
==============================
Test whether the K* law can replace hand-designed step schedules in
diffusion models.

Key idea:
    The K* law says: K* = exp(σ* · α) where σ* = σ/σ_max, α ≈ 1.376
    Instead of using a fixed K (1, 5, 10, 20) for all noise levels, we
    compute K* per noise level and use that as the number of steps.

    This turns the diffusion model into a parameter-free, adaptive
    schedule: the noise level itself tells you how many steps to take.

Comparison:
    - Baseline: fixed K values (1, 5, 10, 20) with hand-designed eta decay
    - K* law: K* = round(exp(σ/σ_max * α)) with the same eta decay

Run:
    py -3.12 experiments/exp_diffusion_kstar_schedule.py --device cuda
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from metrics import compute_ssim
from exp_diffusion_cifar10 import TinyUNet, iterative_score_denoise

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ── K* law parameters ─────────────────────────────────────────────────────────
ALPHA = 1.376          # measured universal exponent
SIGMA_MAX = 0.30       # max training noise level (must match training)


def compute_kstar(sigma, alpha=ALPHA, sigma_max=SIGMA_MAX):
    """Compute K* from the scaling law: K* = exp(σ/σ_max * α)."""
    sigma_star = sigma / sigma_max
    kstar = math.exp(sigma_star * alpha)
    return max(1, round(kstar))  # at least 1 step


def evaluate_kstar(model, device, test_loader, sigmas):
    """Evaluate diffusion model using K* law schedule per noise level."""
    model.eval()
    results = {}

    for sigma in sigmas:
        K_star = compute_kstar(sigma)
        K_list = [1, 2, 5, 10, 20, K_star]
        K_list = sorted(set(K_list))  # deduplicate if K_star overlaps

        results[sigma] = {}
        for K in K_list:
            psnrs = []
            ssims = []
            for x, _ in test_loader:
                x = x.to(device)
                xn = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
                with torch.no_grad():
                    pred = iterative_score_denoise(model, xn, K, sigma, device)
                mse = F.mse_loss(pred, x, reduction='none').mean(dim=[1, 2, 3])
                psnr_batch = (10.0 * torch.log10(4.0 / mse)).cpu().numpy()
                psnrs.extend(psnr_batch.tolist())
                for i in range(pred.shape[0]):
                    s = compute_ssim(x[i:i+1], pred[i:i+1], data_range=2.0)
                    ssims.append(s)

            results[sigma][K] = {
                'psnr_mean': float(np.mean(psnrs)),
                'psnr_std': float(np.std(psnrs)),
                'ssim_mean': float(np.mean(ssims)),
                'ssim_std': float(np.std(ssims)),
            }

    return results


def print_summary(results, sigmas):
    """Print comparison table: K* law vs fixed K values."""
    print("\n" + "=" * 90)
    print("Diffusion Schedule Replacement: K* Law vs Fixed K")
    print("=" * 90)
    print(f"\nK* law: K* = exp(σ/σ_max * α)  |  α={ALPHA}, σ_max={SIGMA_MAX}")
    print("-" * 90)

    for sigma in sigmas:
        K_star = compute_kstar(sigma)
        print(f"\nσ = {sigma:.2f}  →  K* = {K_star}")
        print("-" * 50)
        # Show K* first, then fixed K values
        display_order = [K_star] + [k for k in [1, 5, 10, 20] if k != K_star]
        for K in display_order:
            if K not in results[sigma]:
                continue
            r = results[sigma][K]
            marker = " ★ K*" if K == K_star else ""
            print(f"  K={K:2d}{marker}: PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f} dB  "
                  f"SSIM={r['ssim_mean']:.3f}±{r['ssim_std']:.3f}")

    # ── Compute how close K* is to optimal ────────────────────────────────────
    print("\n" + "=" * 90)
    print("K* Law vs. Oracle Optimal K")
    print("=" * 90)
    for sigma in sigmas:
        K_star = compute_kstar(sigma)
        psnrs = {K: results[sigma][K]['psnr_mean'] for K in results[sigma]}
        K_best = max(psnrs, key=psnrs.get)
        psnr_star = psnrs[K_star]
        psnr_best = psnrs[K_best]
        gap = psnr_best - psnr_star
        print(f"σ={sigma:.2f}: K*={K_star} → PSNR={psnr_star:.2f} dB  |  "
              f"optimal K={K_best} → PSNR={psnr_best:.2f} dB  |  "
              f"gap={gap:+.2f} dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/diffusion_cifar10/diffusion_cifar10.pt')
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--sigma_max', type=float, default=0.30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    global SIGMA_MAX
    SIGMA_MAX = args.sigma_max

    # ── Load data ─────────────────────────────────────────────────────────────
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_ds, range(args.n_eval)),
        batch_size=100, shuffle=False, num_workers=2)

    # ── Load model ────────────────────────────────────────────────────────────
    model = TinyUNet(n_channels=3, base_ch=args.base_ch).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run exp_diffusion_cifar10.py first to train the diffusion model.")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded diffusion model: {n_params:,} params from {ckpt_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"\nEvaluating K* schedule on {args.n_eval} images...")
    t0 = time.time()
    results = evaluate_kstar(model, device, eval_loader, sigmas)
    eval_time = time.time() - t0
    print(f"Evaluation complete in {eval_time:.1f}s")

    # ── Print & save ──────────────────────────────────────────────────────────
    print_summary(results, sigmas)

    out_dir = ROOT / 'outputs' / 'diffusion_kstar_schedule'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'diffusion_kstar_results.json'
    out_path.write_text(json.dumps({
        'alpha': ALPHA,
        'sigma_max': SIGMA_MAX,
        'n_eval': args.n_eval,
        'n_params': n_params,
        'results': results,
    }, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
