"""
exp_lane_b_reasoning.py
=======================
Lane B: Abstract Reasoning via Energy Minimization on Grid Puzzles.

This is the System-2 probe for the KAN-EBM. The pitch is that the same
energy-minimization mechanism that handles perceptual inverse problems
(Lane A) also handles abstract constraint satisfaction. We do NOT touch
natural language; we work in the modality the architecture was built for
(2D grids), but on tasks where the underlying structure is a *rule*
rather than image statistics.

Three tasks of increasing difficulty:
  1. STRIPES  -- horizontal/vertical stripe patterns; mask center 4x4 quadrant.
                 Simple translation-invariant statistic.
  2. CHECKER  -- 2x2 block checkerboards in two colors; mask 4x4 quadrant.
                 Local periodic constraint.
  3. LATIN-4  -- 4x4 Latin square (each row/col contains each of 4 colors once);
                 random cell mask. Genuine global constraint satisfaction.

Each puzzle is rendered as a 16x16 grayscale image (4x4 grid * 4-pixel cells).
A small KAN-EBM is trained on the corresponding distribution of valid puzzles
via DSM. At test time, masked puzzles are completed by gradient descent on
E_theta(u) with hard projection on observed cells.

Comparison: same architecture, MLP-EBM head replaces KAN head. Fair, matched
parameter budget. Reports cell-accuracy and puzzle-accuracy vs. K.

Output: outputs/lane_b/results.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from exp_cifar10 import KANEnergyModel, MLPEnergyModel  # noqa: E402

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Puzzle generators
# All output 1-channel images in [-1, 1], shape (1, 16, 16).
# Cells are 4x4 patches within a 4x4 grid. K = number of "colors" in the alphabet.
# We map color k in {0,...,K-1} to value -1 + 2*k/(K-1) (uniformly spaced in [-1,1]).
# ─────────────────────────────────────────────────────────────────────────────

GRID = 4   # 4x4 cells
CELL = 4   # 4 px per cell
H = W = GRID * CELL  # 16


def color_to_value(k: int, K: int) -> float:
    return -1.0 + 2.0 * k / (K - 1)


def values_to_color(v: torch.Tensor, K: int) -> torch.Tensor:
    """Quantize back to nearest color index."""
    levels = torch.tensor([-1.0 + 2.0 * k / (K - 1) for k in range(K)], device=v.device)
    return (v.unsqueeze(-1) - levels).abs().argmin(dim=-1)


def render_grid(grid: torch.Tensor, K: int) -> torch.Tensor:
    """grid: (GRID, GRID) integer in [0, K). Returns (1, H, W) float in [-1, 1]."""
    img = torch.zeros(1, H, W)
    for i in range(GRID):
        for j in range(GRID):
            v = color_to_value(int(grid[i, j].item()), K)
            img[0, i*CELL:(i+1)*CELL, j*CELL:(j+1)*CELL] = v
    return img


def cells_from_image(img: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse of render_grid: (1, H, W) -> (GRID, GRID) int color indices."""
    cells = torch.zeros(GRID, GRID, dtype=torch.long, device=img.device)
    for i in range(GRID):
        for j in range(GRID):
            patch = img[0, i*CELL:(i+1)*CELL, j*CELL:(j+1)*CELL]
            cells[i, j] = values_to_color(patch.mean().unsqueeze(0), K)[0]
    return cells


def gen_stripes(n: int, K: int = 2) -> torch.Tensor:
    """Horizontal or vertical stripe patterns. Returns (n, 1, H, W)."""
    out = torch.empty(n, 1, H, W)
    for s in range(n):
        orient = np.random.randint(2)        # 0=horizontal, 1=vertical
        offset = np.random.randint(K)         # phase
        grid = torch.zeros(GRID, GRID, dtype=torch.long)
        for i in range(GRID):
            for j in range(GRID):
                idx = (i if orient == 0 else j)
                grid[i, j] = (idx + offset) % K
        out[s] = render_grid(grid, K)
    return out


def gen_checker(n: int, K: int = 2) -> torch.Tensor:
    """Checkerboard with random color permutation."""
    out = torch.empty(n, 1, H, W)
    for s in range(n):
        perm = np.random.permutation(K)
        grid = torch.zeros(GRID, GRID, dtype=torch.long)
        for i in range(GRID):
            for j in range(GRID):
                grid[i, j] = perm[(i + j) % K]
        out[s] = render_grid(grid, K)
    return out


def random_latin_square(K: int = 4) -> torch.Tensor:
    """Random K x K Latin square via cyclic-shift method + row/col shuffle."""
    base = torch.tensor([[(i + j) % K for j in range(K)] for i in range(K)], dtype=torch.long)
    # Random row permutation
    base = base[torch.randperm(K)]
    # Random column permutation
    base = base[:, torch.randperm(K)]
    # Random color relabel
    relabel = torch.randperm(K)
    base = relabel[base]
    return base


def gen_latin(n: int, K: int = 4) -> torch.Tensor:
    out = torch.empty(n, 1, H, W)
    for s in range(n):
        out[s] = render_grid(random_latin_square(K), K)
    return out


PUZZLES = {
    'stripes':  (gen_stripes,  2),
    'checker':  (gen_checker,  2),
    'latin4':   (gen_latin,    4),
}


# ─────────────────────────────────────────────────────────────────────────────
# Mask generators (cell-level: hide whole cells, observe whole cells)
# ─────────────────────────────────────────────────────────────────────────────

def cell_mask_quadrant() -> torch.Tensor:
    """Mask = 1 where observed. Hide bottom-right 2x2 cells."""
    cm = torch.ones(GRID, GRID)
    cm[GRID//2:, GRID//2:] = 0
    return cm


def cell_mask_random(rate: float = 0.5) -> torch.Tensor:
    cm = (torch.rand(GRID, GRID) > rate).float()
    return cm


def cell_mask_to_image_mask(cm: torch.Tensor) -> torch.Tensor:
    """Upsample cell-level mask (GRID, GRID) -> image-level (1, H, W)."""
    return cm.repeat_interleave(CELL, dim=0).repeat_interleave(CELL, dim=1).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Models (matched-param KAN vs MLP, both with the same filter bank)
# ─────────────────────────────────────────────────────────────────────────────

def build_kan(n_filters=8, kan_hidden=None):
    if kan_hidden is None:
        kan_hidden = [16]
    return KANEnergyModel(n_filters=n_filters, filter_size=5,
                          kan_hidden=kan_hidden, n_channels=1)


class FilteredMLPEnergyModel(nn.Module):
    """MLP-head EBM with the SAME filter bank as the KAN model.
    Isolates the KAN-vs-MLP architectural difference."""
    def __init__(self, n_filters=8, filter_size=5, hidden=64, n_channels=1):
        super().__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_channels = n_channels
        self.filters = nn.Parameter(0.01 * torch.randn(n_filters, n_channels, filter_size, filter_size))
        self.log_precision = nn.Parameter(torch.zeros(n_filters))
        self.mlp = nn.Sequential(
            nn.Linear(n_filters * n_channels, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    @property
    def precision(self):
        return F.softplus(self.log_precision)

    def extract_features(self, x):
        B, C, Hh, Ww = x.shape
        pad = self.filter_size // 2
        feats = []
        for c in range(C):
            feats.append(F.conv2d(x[:, c:c+1], self.filters[:, c:c+1], padding=pad))
        feats = torch.cat(feats, dim=1)
        prec = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = torch.tanh(feats * prec)
        return feats.permute(0, 2, 3, 1).contiguous().view(B * Hh * Ww, -1)

    def energy(self, x):
        return self.mlp(self.extract_features(x)).sum()

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi = xn.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.energy(xi), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma**2 * grad, x_clean)


# ─────────────────────────────────────────────────────────────────────────────
# Training + Inference
# ─────────────────────────────────────────────────────────────────────────────

def train_ebm(model, data, epochs, sigma, lr=1e-3, batch_size=64, tag=''):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = data.shape[0]
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_losses = []
        for i in range(0, n, batch_size):
            x = data[perm[i:i+batch_size]]
            loss = model.loss(x, sigma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        avg = float(np.mean(ep_losses))
        losses.append(avg)
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"  [{tag}] ep {ep+1:>2d}/{epochs}  loss={avg:.5f}")
    return losses


def refine_with_anchor(model, u_init, n_steps, anchor_mask_img, anchor_vals_img,
                       dt=0.05, dt_decay=0.97):
    u = u_init.clone()
    step = dt
    for _ in range(n_steps):
        with torch.enable_grad():
            ui = u.detach().requires_grad_(True)
            g = torch.autograd.grad(model.energy(ui), ui)[0].detach().clamp(-1, 1)
        u = (u.detach() - step * g)
        # Hard projection on observed cells
        u = torch.where(anchor_mask_img.bool(), anchor_vals_img, u)
        u = u.clamp(-1, 1)
        step *= dt_decay
    return u


def evaluate(model, test_imgs, K_list, K_colors, mask_fn, n_eval=200):
    """For each test image, mask + refine + score cell-accuracy and puzzle-accuracy."""
    cell_acc = {k: [] for k in K_list}
    puzzle_acc = {k: [] for k in K_list}
    n = min(n_eval, test_imgs.shape[0])
    device = test_imgs.device
    for i in range(n):
        x = test_imgs[i:i+1]                                   # (1, 1, H, W)
        cm = mask_fn().to(device)                               # (GRID, GRID)
        m_img = cell_mask_to_image_mask(cm).unsqueeze(0).to(device)  # (1, 1, H, W)
        # The masked region needs an init; fill with grand mean (= 0 in [-1,1])
        u0 = torch.where(m_img.bool(), x, torch.zeros_like(x))
        true_cells = cells_from_image(x[0], K_colors)
        hidden_mask = (cm == 0)                                 # GRID, GRID
        n_hidden = int(hidden_mask.sum().item())
        for k in K_list:
            if k == 0:
                u_hat = u0
            else:
                u_hat = refine_with_anchor(model, u0, k, m_img, x)
            pred_cells = cells_from_image(u_hat[0], K_colors)
            correct = ((pred_cells == true_cells) & hidden_mask).sum().item()
            cell_acc[k].append(correct / max(1, n_hidden))
            puzzle_acc[k].append(1.0 if correct == n_hidden else 0.0)
    return ({k: float(np.mean(v)) for k, v in cell_acc.items()},
            {k: float(np.mean(v)) for k, v in puzzle_acc.items()})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n_train', type=int, default=2000)
    ap.add_argument('--n_test', type=int, default=200)
    ap.add_argument('--sigma', type=float, default=0.3)
    ap.add_argument('--mlp_hidden', type=int, default=64)
    ap.add_argument('--n_filters', type=int, default=8)
    ap.add_argument('--puzzles', nargs='+', default=list(PUZZLES.keys()))
    args = ap.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    out_dir = ROOT / 'outputs' / 'lane_b'
    out_dir.mkdir(parents=True, exist_ok=True)

    K_list = [0, 1, 2, 5, 10, 20]
    summary = {
        'config': vars(args),
        'puzzles': {},
    }
    t0 = time.time()
    for puzzle_name in args.puzzles:
        gen_fn, K_colors = PUZZLES[puzzle_name]
        print("\n" + "=" * 70)
        print(f"PUZZLE: {puzzle_name.upper()} (K={K_colors} colors, "
              f"{args.n_train} train / {args.n_test} test)")
        print("=" * 70)

        train_data = gen_fn(args.n_train, K=K_colors).to(device)
        test_data = gen_fn(args.n_test * 2, K=K_colors).to(device)

        # Build matched-param KAN and Filtered-MLP
        kan = build_kan(n_filters=args.n_filters).to(device)
        mlp = FilteredMLPEnergyModel(n_filters=args.n_filters, filter_size=5,
                                     hidden=args.mlp_hidden, n_channels=1).to(device)
        n_kan = sum(p.numel() for p in kan.parameters())
        n_mlp = sum(p.numel() for p in mlp.parameters())
        print(f"  KAN params: {n_kan:,} | MLP-filt params: {n_mlp:,}")

        # Train
        kan_losses = train_ebm(kan, train_data, args.epochs, args.sigma, tag='KAN')
        mlp_losses = train_ebm(mlp, train_data, args.epochs, args.sigma, tag='MLP')

        # Evaluate on two mask types
        kan.eval(); mlp.eval()

        # Random-cell mask (50% of cells hidden)
        random_mask_fn = lambda rate=0.5: cell_mask_random(rate)
        # Quadrant mask
        quad_mask_fn = cell_mask_quadrant

        eval_results = {}
        for mask_label, mfn in [('random50', random_mask_fn),
                                 ('quadrant', quad_mask_fn)]:
            torch.manual_seed(SEED)  # reproducible mask seq
            np.random.seed(SEED)
            kan_cell, kan_puz = evaluate(kan, test_data, K_list, K_colors, mfn,
                                         n_eval=args.n_test)
            torch.manual_seed(SEED); np.random.seed(SEED)
            mlp_cell, mlp_puz = evaluate(mlp, test_data, K_list, K_colors, mfn,
                                         n_eval=args.n_test)
            eval_results[mask_label] = {
                'kan_cell_acc': kan_cell, 'kan_puzzle_acc': kan_puz,
                'mlp_cell_acc': mlp_cell, 'mlp_puzzle_acc': mlp_puz,
            }
            print(f"\n  --- mask={mask_label} ---")
            print(f"  K  | KAN cell-acc | MLP cell-acc | KAN puz-acc | MLP puz-acc")
            for k in K_list:
                print(f"  {k:>2} |     {kan_cell[k]*100:>5.1f}%   |     "
                      f"{mlp_cell[k]*100:>5.1f}%   |    {kan_puz[k]*100:>5.1f}%   |    "
                      f"{mlp_puz[k]*100:>5.1f}%")

        summary['puzzles'][puzzle_name] = {
            'K_colors': K_colors,
            'n_kan_params': n_kan, 'n_mlp_params': n_mlp,
            'final_kan_loss': kan_losses[-1], 'final_mlp_loss': mlp_losses[-1],
            'eval': eval_results,
        }

    summary['wall_clock_s'] = round(time.time() - t0, 1)
    out_path = out_dir / 'results.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 70)
    print(f"Lane B complete in {summary['wall_clock_s']/60:.1f} min")
    print(f"Saved -> {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
