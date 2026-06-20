"""
Triple-Role Experiment — REAL inference (no hardcoded numbers).

Predecessor commit ec47... contained a `run_scaling_ablation` that PLOTTED
hardcoded PSNR values (`kan_psnr = [21.92, 23.43, ...]`) and a planning demo
that bypassed the learned encoder. Both were flagged as a reviewer-detectable
landmine in the 2026-04-19 audit (paper readiness assessment).

This rewrite runs every reported number through actual model inference on a
loaded checkpoint. If a checkpoint is missing the script SKIPS the relevant
panel rather than fabricating data.

Outputs
-------
results/triple_role/
  ├── kan_kscaling_mnist.json     — real PSNR vs K from results/kan_ebm.pt
  ├── kan_kscaling_mnist.png      — figure rendered from the JSON
  ├── perceptual_transfer.json    — zero-shot maze success rates
  └── eikonal_field.png           — Eikonal field from learned slowness

Tied to REGISTERED_CLAIMS.md P4.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from wave_solver import EikonalSolver, extract_path  # noqa: E402

OUT_DIR = ROOT / "results" / "triple_role"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Load the trained KAN-EBM checkpoint
# ---------------------------------------------------------------------------

def load_kan_ebm(device: torch.device, ckpt_path: Path) -> nn.Module:
    """Re-build the KANEnergyModel architecture used for results/kan_ebm.pt."""
    from exp_multitask import KANEnergyModel  # canonical 28x28 KAN-EBM
    kan = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32], n_channels=1)
    state = torch.load(ckpt_path, map_location=device)
    kan.load_state_dict(state)
    kan.to(device).eval()
    return kan


# ---------------------------------------------------------------------------
# 2.  Real K-scaling curve on MNIST
# ---------------------------------------------------------------------------

def psnr(clean: torch.Tensor, pred: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(pred, clean).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(data_range ** 2 / mse)


def run_real_kscaling(
    kan: nn.Module,
    device: torch.device,
    sigma: float = 0.2,
    n_imgs: int = 200,
    ks: List[int] = (1, 2, 5, 10, 20, 50, 100),
) -> Dict:
    """Real PSNR-vs-K curve on MNIST test split. No hardcoded numbers."""
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tfm)
    xs = torch.stack([ds[i][0] for i in range(n_imgs)]).to(device)

    torch.manual_seed(0)
    noise = torch.randn_like(xs) * sigma
    xs_noisy = (xs + noise).clamp(-1, 1)

    out = {"sigma": sigma, "n_imgs": n_imgs, "ks": list(ks), "psnr": {}, "psnr_std": {}}
    for k in ks:
        with torch.no_grad():
            preds = kan.denoise(xs_noisy, n_steps=int(k))
        per_img = [psnr(xs[i:i+1], preds[i:i+1].clamp(-1, 1)) for i in range(xs.shape[0])]
        out["psnr"][str(k)] = float(np.mean(per_img))
        out["psnr_std"][str(k)] = float(np.std(per_img))
        print(f"  K={k:>3}  PSNR={out['psnr'][str(k)]:.3f} ± {out['psnr_std'][str(k)]:.3f} dB")
    return out


def plot_kscaling(out: Dict, save_path: Path):
    ks = out["ks"]
    means = [out["psnr"][str(k)] for k in ks]
    stds = [out["psnr_std"][str(k)] for k in ks]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(ks, means, yerr=stds, marker="o", linewidth=2, capsize=3,
                 label=f"KAN-EBM (real inference, σ={out['sigma']})")
    plt.xscale("log")
    plt.xticks(ks, labels=[str(k) for k in ks])
    plt.xlabel("Inference steps K (log scale)")
    plt.ylabel("PSNR (dB)")
    plt.title("Test-Time Compute Scaling — measured, not simulated")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 3.  Zero-shot perceptual transfer:  KAN energy → Eikonal slowness
# ---------------------------------------------------------------------------

def kan_energy_to_slowness(
    kan: nn.Module,
    maze_image: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    Pixelwise slowness n(x) = alpha + beta * E_local(x) where E_local is the
    per-pixel contribution of the KAN energy. We compute it as the absolute
    value of the per-pixel feature-energy: KAN sums per-pixel and we keep the
    contribution before the sum.

    This is the operational definition of "denoising energy = planning
    slowness" tested by P4 in REGISTERED_CLAIMS.md.
    """
    with torch.enable_grad():
        x = maze_image.detach().requires_grad_(True)
        feats = kan.extract_features(x)              # (B*H*W, F)
        per_pixel_E = kan.kan(feats).abs().squeeze(-1)  # (B*H*W,)
    B, _, H, W = maze_image.shape
    e = per_pixel_E.view(B, H, W).detach()
    e_min, e_max = e.min(), e.max()
    e_norm = (e - e_min) / (e_max - e_min + 1e-8)
    return alpha + beta * e_norm  # (B, H, W)


def run_perceptual_transfer(
    kan: nn.Module,
    device: torch.device,
    maze_size: int = 32,
    n_test: int = 100,
    wall_density: float = 0.25,
):
    """
    Zero-shot perceptual transfer benchmark.

    Pipeline:
      1. Generate a random maze (grayscale 0/1).
      2. Compute slowness n(x) from the *denoising* KAN energy.
      3. Solve the Eikonal equation with that n(x).
      4. Extract a path; check validity with strict collision rules.

    Compared baselines (random and uniform-slowness) are also reported so the
    reader can see whether the perceptual prior is doing the work.
    """
    from train_phase3 import generate_maze_with_path
    from exp_planning_benchmarks import is_valid_path

    solver = EikonalSolver(grid_size=(maze_size, maze_size), n_sweeps=8).to(device)

    succ_kan = 0
    succ_uniform = 0
    succ_random = 0

    print(f"\n  Zero-shot perceptual transfer: KAN-EBM → Eikonal on {maze_size}x{maze_size}")
    for i in range(n_test):
        maze, source, target, _ = generate_maze_with_path(
            size=maze_size, wall_density=wall_density, ensure_path=True, device=device
        )

        # KAN-as-slowness
        n_kan = kan_energy_to_slowness(kan, maze, alpha=0.5, beta=4.0)
        u_kan = solver(n_kan, source)
        path_kan = extract_path(u_kan, source, target)
        if is_valid_path(path_kan, maze):
            succ_kan += 1

        # Uniform-slowness baseline (no perception): n=1 in passages, n=10 in walls
        n_uniform = torch.where(maze[:, 0] < 0.5, torch.full_like(maze[:, 0], 10.0),
                                torch.full_like(maze[:, 0], 1.0))
        u_uniform = solver(n_uniform, source)
        path_uniform = extract_path(u_uniform, source, target)
        if is_valid_path(path_uniform, maze):
            succ_uniform += 1

        # Random-slowness baseline (control: any per-pixel field works?)
        torch.manual_seed(i)
        n_random = 0.5 + 4.0 * torch.rand_like(n_kan)
        u_random = solver(n_random, source)
        path_random = extract_path(u_random, source, target)
        if is_valid_path(path_random, maze):
            succ_random += 1

    rec = {
        "maze_size": maze_size,
        "n_test": n_test,
        "wall_density": wall_density,
        "kan_perceptual_success_rate": succ_kan / n_test,
        "uniform_oracle_success_rate": succ_uniform / n_test,
        "random_field_success_rate": succ_random / n_test,
    }
    print(
        f"  KAN-perceptual: {rec['kan_perceptual_success_rate']*100:.1f}% | "
        f"Uniform oracle: {rec['uniform_oracle_success_rate']*100:.1f}% | "
        f"Random: {rec['random_field_success_rate']*100:.1f}%"
    )
    return rec


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", default=str(ROOT / "results" / "kan_ebm.pt"))
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--n-imgs", type=int, default=200)
    p.add_argument("--maze-size", type=int, default=32)
    p.add_argument("--n-mazes", type=int, default=100)
    p.add_argument("--skip-kscaling", action="store_true")
    p.add_argument("--skip-transfer", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[triple_role] device={device}")

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[triple_role] CHECKPOINT MISSING at {ckpt}.")
        print("[triple_role] Refusing to fabricate numbers. Train via "
              "experiments/exp_multitask.py --save-checkpoint results/kan_ebm.pt first.")
        sys.exit(2)

    kan = load_kan_ebm(device, ckpt)
    print(f"[triple_role] Loaded KAN-EBM ({sum(p.numel() for p in kan.parameters())} params) from {ckpt}")

    if not args.skip_kscaling:
        print("\n=== Real K-scaling on MNIST ===")
        out = run_real_kscaling(kan, device, sigma=args.sigma, n_imgs=args.n_imgs)
        out["timestamp"] = int(time.time())
        out["ckpt"] = str(ckpt)
        with open(OUT_DIR / "kan_kscaling_mnist.json", "w") as fh:
            json.dump(out, fh, indent=2)
        plot_kscaling(out, OUT_DIR / "kan_kscaling_mnist.png")
        print(f"  wrote {OUT_DIR/'kan_kscaling_mnist.json'} and .png")

    if not args.skip_transfer:
        print("\n=== Zero-shot perceptual transfer (P4) ===")
        rec = run_perceptual_transfer(kan, device,
                                      maze_size=args.maze_size,
                                      n_test=args.n_mazes)
        rec["timestamp"] = int(time.time())
        rec["ckpt"] = str(ckpt)
        with open(OUT_DIR / "perceptual_transfer.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  wrote {OUT_DIR/'perceptual_transfer.json'}")

        # Single visualization for the paper figure: one maze + slowness + travel-time
        from train_phase3 import generate_maze_with_path
        torch.manual_seed(123)
        maze, source, target, _ = generate_maze_with_path(
            size=args.maze_size, wall_density=0.25, ensure_path=True, device=device
        )
        n_field = kan_energy_to_slowness(kan, maze)
        solver = EikonalSolver(grid_size=(args.maze_size, args.maze_size), n_sweeps=8).to(device)
        u_field = solver(n_field, source)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(maze[0, 0].cpu(), cmap="gray")
        axes[0].set_title("Maze (input)"); axes[0].axis("off")
        im1 = axes[1].imshow(n_field[0].cpu(), cmap="hot")
        axes[1].set_title("n(x) from denoising KAN-EBM\n(zero-shot, no planning training)")
        axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
        u_np = u_field[0].detach().cpu().numpy()
        u_np = np.clip(u_np, 0, np.percentile(u_np[np.isfinite(u_np)], 99))
        im2 = axes[2].imshow(u_np, cmap="viridis")
        axes[2].set_title("Travel time u(x) (Eikonal)")
        axes[2].scatter(source[1], source[0], c="red", marker="*", s=120)
        axes[2].scatter(target[1], target[0], c="lime", marker="o", s=120)
        axes[2].axis("off"); plt.colorbar(im2, ax=axes[2], fraction=0.046)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "eikonal_field.png", dpi=150)
        plt.close()
        print(f"  wrote {OUT_DIR/'eikonal_field.png'}")

    print("\n[triple_role] Done.")


if __name__ == "__main__":
    main()
