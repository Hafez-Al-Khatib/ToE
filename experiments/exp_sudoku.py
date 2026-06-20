"""
Paper 1 — Experiment 2: Sudoku Constraint Satisfaction
=======================================================

Compares KAN-Hamiltonian vs MLP-Baseline on Sudoku solving
(framed as energy minimization over a 9-channel field on 9x9 grid).

This is the key experiment showing that KAN learns constraint-aware
energy landscapes that enforce row/column/box uniqueness.

Metrics: Cell accuracy (empty cells), solve rate, constraint violations
Ablations: n_clues ∈ {25, 30, 35, 40}, evolution steps

Usage:
    python experiments/exp_sudoku.py --epochs 30
    python experiments/exp_sudoku.py --epochs 50 --n-clues 30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hamiltonian_field import HamiltonianField
from mlp_field import MLPField
from metrics import (compute_sudoku_accuracy, ExperimentLogger,
                     print_comparison_table)


# ============================================================================
# Sudoku Data Generation
# ============================================================================

def generate_solved_sudoku(rng=None):
    """Generate a random valid solved Sudoku grid (9x9, values 0-8)."""
    if rng is None:
        rng = np.random

    grid = np.zeros((9, 9), dtype=int)

    def is_valid(g, r, c, n):
        if n in g[r, :]: return False
        if n in g[:, c]: return False
        br, bc = 3*(r//3), 3*(c//3)
        if n in g[br:br+3, bc:bc+3]: return False
        return True

    def solve(g):
        for i in range(9):
            for j in range(9):
                if g[i, j] == 0:
                    nums = list(range(1, 10))
                    rng.shuffle(nums)
                    for n in nums:
                        if is_valid(g, i, j, n):
                            g[i, j] = n
                            if solve(g): return True
                            g[i, j] = 0
                    return False
        return True

    solve(grid)
    return grid - 1  # Convert to 0-8 range


def grid_to_onehot(grid):
    """Convert (9,9) grid with values 0-8 to (9,9,9) one-hot tensor."""
    onehot = torch.zeros(9, 9, 9, dtype=torch.float32)
    for i in range(9):
        for j in range(9):
            if grid[i, j] >= 0:
                onehot[grid[i, j], i, j] = 1.0
    return onehot


def create_puzzle_batch(batch_size, n_clues=35, device='cpu', rng=None):
    """
    Generate a batch of Sudoku puzzles with solutions.

    Returns:
        inputs: (B, 9, 9, 9) — one-hot with empty cells as uniform 1/9
        targets: (B, 9, 9) — ground truth digit indices (0-8)
        masks: (B, 9, 9) — True where clues exist
    """
    if rng is None:
        rng = np.random

    inputs_list = []
    targets_list = []
    masks_list = []

    for _ in range(batch_size):
        solution = generate_solved_sudoku(rng)
        target = torch.tensor(solution, dtype=torch.long)

        # Create mask
        mask = torch.zeros(9, 9, dtype=torch.bool)
        indices = torch.randperm(81)[:n_clues]
        mask.view(-1)[indices] = True

        # Input: one-hot for clues, uniform for empty
        inp = torch.zeros(9, 9, 9, dtype=torch.float32)
        for i in range(9):
            for j in range(9):
                if mask[i, j]:
                    inp[solution[i, j], i, j] = 1.0
                else:
                    inp[:, i, j] = 1.0 / 9.0

        inputs_list.append(inp)
        targets_list.append(target)
        masks_list.append(mask)

    return (torch.stack(inputs_list).to(device),
            torch.stack(targets_list).to(device),
            torch.stack(masks_list).to(device))


# ============================================================================
# Training
# ============================================================================

def train_sudoku(model, config):
    """
    Train field model on Sudoku solving via multi-step evolution + CE loss.
    """
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 30)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 16)
    n_clues = config.get('n_clues', 35)
    n_train_steps = config.get('n_train_steps', 5)
    n_puzzles_per_epoch = config.get('n_puzzles_per_epoch', 256)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': [], 'accuracy': []}
    rng = np.random.RandomState(42)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for start in range(0, n_puzzles_per_epoch, batch_size):
            actual_batch = min(batch_size, n_puzzles_per_epoch - start)
            inputs, targets, masks = create_puzzle_batch(
                actual_batch, n_clues=n_clues, device=device, rng=rng)

            optimizer.zero_grad()

            # Forward: evolve field with clue clamping
            state = inputs.clone().requires_grad_(True)
            mask_expanded = masks.unsqueeze(1).float()  # (B, 1, 9, 9)

            final_state = model.trainable_evolve(
                state, n_steps=n_train_steps,
                anchor=inputs.detach(),
                anchor_mask=mask_expanded
            )

            # Cross-entropy loss
            loss = F.cross_entropy(final_state, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track accuracy
            with torch.no_grad():
                pred = final_state.argmax(dim=1)
                empty_mask = ~masks
                if empty_mask.any():
                    acc = (pred[empty_mask] == targets[empty_mask]).float().mean().item()
                else:
                    acc = 1.0

            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: loss={avg_loss:.4f}  "
                  f"empty_acc={avg_acc:.1%}  lr={scheduler.get_last_lr()[0]:.6f}")

    return history


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_sudoku_model(model, n_puzzles=100, n_clues=35, n_steps=50,
                          device='cpu'):
    """Evaluate Sudoku solving on fresh puzzles."""
    model.eval()
    rng = np.random.RandomState(999)

    all_preds = []
    all_targets = []
    all_masks = []
    total_solved = 0

    for _ in range(0, n_puzzles, 16):
        batch_size = min(16, n_puzzles - len(all_preds) * 0 )
        # Generate fresh puzzles
        inputs, targets, masks = create_puzzle_batch(
            16, n_clues=n_clues, device=device, rng=rng)

        # Evolve
        state = inputs.clone().requires_grad_(True)
        mask_expanded = masks.unsqueeze(1).float()

        with torch.enable_grad():
            final = model.evolve(
                state, n_steps=n_steps,
                anchor=inputs.detach(),
                anchor_mask=mask_expanded)

        # Collect predictions
        pred = final.argmax(dim=1)  # (B, 9, 9)

        # Check solve rate per puzzle
        for b in range(pred.shape[0]):
            empty = ~masks[b]
            if empty.any():
                correct = (pred[b][empty] == targets[b][empty]).all().item()
                total_solved += int(correct)

        all_preds.append(final.cpu())
        all_targets.append(targets.cpu())
        all_masks.append(masks.cpu())

    # Aggregate metrics
    preds_cat = torch.cat(all_preds, dim=0)[:n_puzzles]
    targets_cat = torch.cat(all_targets, dim=0)[:n_puzzles]
    masks_cat = torch.cat(all_masks, dim=0)[:n_puzzles]

    metrics = compute_sudoku_accuracy(preds_cat, targets_cat, masks_cat)
    metrics['solve_rate'] = total_solved / n_puzzles

    return metrics


# ============================================================================
# Visualization
# ============================================================================

def visualize_sudoku_solving(model, n_clues=35, n_steps=50, device='cpu',
                             save_path='sudoku_results.png'):
    """Visualize one puzzle being solved."""
    model.eval()
    rng = np.random.RandomState(42)

    inputs, targets, masks = create_puzzle_batch(
        1, n_clues=n_clues, device=device, rng=rng)

    # Evolve and record snapshots
    state = inputs.clone().requires_grad_(True)
    mask_expanded = masks.unsqueeze(1).float()
    snapshot_steps = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    snapshots = []

    u = state
    with torch.enable_grad():
        for step in range(n_steps):
            energy = model.compute_energy(u)
            grad = torch.autograd.grad(energy.sum(), u, create_graph=False)[0]
            u_new = u - model.step_size * grad

            # Anchor clues
            if mask_expanded is not None:
                u_new = torch.where(mask_expanded > 0.5, inputs.detach(), u_new)
            u_new = u_new + model.data_weight * (inputs.detach() - u_new) * (1 - mask_expanded)

            u = u_new.detach().requires_grad_(True)

            if step in snapshot_steps:
                pred = u.argmax(dim=1).squeeze(0).cpu().numpy()  # (9, 9)
                snapshots.append((step, pred))

    target_grid = targets[0].cpu().numpy()
    mask_grid = masks[0].cpu().numpy()

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4*len(snapshots), 4))
    for idx, (step, pred) in enumerate(snapshots):
        ax = axes[idx]
        colors = np.ones((9, 9, 3))

        for i in range(9):
            for j in range(9):
                if mask_grid[i, j]:
                    colors[i, j] = [0.85, 0.85, 0.85]
                elif pred[i, j] == target_grid[i, j]:
                    colors[i, j] = [0.7, 1.0, 0.7]
                else:
                    colors[i, j] = [1.0, 0.7, 0.7]

        ax.imshow(colors, extent=[0, 9, 9, 0])
        for i in range(9):
            for j in range(9):
                ax.text(j+0.5, i+0.5, str(pred[i, j]+1),
                        ha='center', va='center', fontsize=8, fontweight='bold')

        for k in range(10):
            lw = 2 if k % 3 == 0 else 0.5
            ax.axhline(y=k, color='black', linewidth=lw)
            ax.axvline(x=k, color='black', linewidth=lw)

        empty = ~mask_grid
        n_correct = (pred[empty] == target_grid[empty]).sum()
        n_empty = empty.sum()
        ax.set_title(f'Step {step}: {n_correct}/{n_empty}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Sudoku Solving via Energy Minimization ({n_clues} clues)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper 1: Sudoku Experiments")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n-clues', type=int, default=35)
    parser.add_argument('--n-train-steps', type=int, default=5)
    parser.add_argument('--n-eval-steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='./results/sudoku')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # ---- KAN-Hamiltonian ----
    print(f"\n{'='*60}")
    print(f"  Training KAN-Hamiltonian Sudoku Solver")
    print(f"{'='*60}")

    kan_model = HamiltonianField(
        n_channels=9, height=9, width=9,
        n_filters=9, filter_size=5,
        kan_hidden=[128, 1], kan_grid_size=5
    )
    n_params_kan = sum(p.numel() for p in kan_model.parameters())
    print(f"  Parameters: {n_params_kan:,}")

    config = {
        'device': device, 'epochs': args.epochs, 'lr': 1e-3,
        'batch_size': 16, 'n_clues': args.n_clues,
        'n_train_steps': args.n_train_steps,
        'n_puzzles_per_epoch': 256
    }

    t0 = time.time()
    kan_history = train_sudoku(kan_model, config)
    kan_time = time.time() - t0

    kan_metrics = evaluate_sudoku_model(
        kan_model, n_puzzles=100, n_clues=args.n_clues,
        n_steps=args.n_eval_steps, device=device)
    kan_metrics['params'] = n_params_kan
    kan_metrics['train_time_s'] = kan_time
    results['KAN-Hamiltonian'] = kan_metrics

    # ---- MLP-Baseline ----
    print(f"\n{'='*60}")
    print(f"  Training MLP-Baseline Sudoku Solver")
    print(f"{'='*60}")

    mlp_model = MLPField(
        n_channels=9, height=9, width=9,
        n_filters=9, mlp_hidden=[128]
    )
    n_params_mlp = sum(p.numel() for p in mlp_model.parameters())
    print(f"  Parameters: {n_params_mlp:,}")

    t0 = time.time()
    mlp_history = train_sudoku(mlp_model, config)
    mlp_time = time.time() - t0

    mlp_metrics = evaluate_sudoku_model(
        mlp_model, n_puzzles=100, n_clues=args.n_clues,
        n_steps=args.n_eval_steps, device=device)
    mlp_metrics['params'] = n_params_mlp
    mlp_metrics['train_time_s'] = mlp_time
    results['MLP-Baseline'] = mlp_metrics

    # ---- Results ----
    print_comparison_table(results, f"Sudoku Results ({args.n_clues} clues)")

    # Visualize
    visualize_sudoku_solving(
        kan_model, n_clues=args.n_clues, n_steps=args.n_eval_steps,
        device=device,
        save_path=os.path.join(args.output_dir, 'sudoku_kan.png'))
    visualize_sudoku_solving(
        mlp_model, n_clues=args.n_clues, n_steps=args.n_eval_steps,
        device=device,
        save_path=os.path.join(args.output_dir, 'sudoku_mlp.png'))

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(kan_history['loss'], label='KAN', color='#2196F3', linewidth=2)
    ax1.plot(mlp_history['loss'], label='MLP', color='#FF5722', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(kan_history['accuracy'], label='KAN', color='#2196F3', linewidth=2)
    ax2.plot(mlp_history['accuracy'], label='MLP', color='#FF5722', linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Empty Cell Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle('Sudoku Training: KAN vs MLP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sudoku_training.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save models
    torch.save(kan_model.state_dict(),
               os.path.join(args.output_dir, 'kan_sudoku.pth'))
    torch.save(mlp_model.state_dict(),
               os.path.join(args.output_dir, 'mlp_sudoku.pth'))

    # Save results
    logger = ExperimentLogger('sudoku', args.output_dir)
    logger.log_config(vars(args))
    logger.log_final(results)
    logger.save()

    print("\n✓ Sudoku experiments complete!")
    print(f"  Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
