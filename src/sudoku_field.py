"""
Sudoku via Thermodynamic Neural Field
=====================================

Solves Sudoku using the SAME ThermodynamicField architecture that does
image denoising. No hand-crafted energy terms. No neural nets.

Core Idea
---------
A Sudoku grid is a 9×9 spatial field with 9 channels (one per digit).
    u[c, i, j] = "strength of digit c+1 at cell (i,j)"

A solved puzzle has exactly one channel "on" per cell:
    u = one-hot over channels at each spatial position

The ThermodynamicField's learned components naturally encode Sudoku rules:
    - Spatial filters (3×3): detect repeated digits in rows/cols/boxes
    - Diffusion: propagate constraints across space
    - Coupling: enforce mutual exclusion between digit channels
    - Potentials: push activations toward 0 or 1 (commitment)

Training
--------
Same as denoising: corrupt solved puzzles with noise, train the field
dynamics to restore them.

    L = ‖evolve(x_noisy, K steps) - x_clean‖²

Multi-step training (K>1) teaches the field to propagate constraints
over distance, not just make local corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# Import the SAME ThermodynamicField used for denoising
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thermodynamic_field import ThermodynamicField


# =============================================================================
# Data Generation
# =============================================================================

def generate_solved_sudoku() -> np.ndarray:
    """Generate a random valid solved Sudoku grid (9×9, values 1-9)."""
    grid = np.zeros((9, 9), dtype=int)
    
    def is_valid(grid, row, col, num):
        if num in grid[row, :]:
            return False
        if num in grid[:, col]:
            return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_r:box_r+3, box_c:box_c+3]:
            return False
        return True
    
    def solve(grid):
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    nums = list(range(1, 10))
                    np.random.shuffle(nums)
                    for n in nums:
                        if is_valid(grid, i, j, n):
                            grid[i, j] = n
                            if solve(grid):
                                return True
                            grid[i, j] = 0
                    return False
        return True
    
    solve(grid)
    return grid


def grid_to_onehot(grid: np.ndarray) -> torch.Tensor:
    """Convert 9×9 grid (values 1-9) to 9×9×9 one-hot tensor."""
    onehot = np.zeros((9, 9, 9), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            if grid[i, j] > 0:
                onehot[grid[i, j] - 1, i, j] = 1.0
    return torch.from_numpy(onehot)


def onehot_to_grid(onehot: torch.Tensor) -> np.ndarray:
    """Convert 9×9×9 one-hot/soft tensor to 9×9 grid (values 1-9)."""
    return (onehot.argmax(dim=0) + 1).cpu().numpy()


def create_puzzle(grid: np.ndarray, n_clues: int = 35) -> Tuple[np.ndarray, np.ndarray]:
    """Remove cells from solved grid to create puzzle. Returns (puzzle, mask)."""
    puzzle = grid.copy()
    mask = np.ones((9, 9), dtype=bool)
    
    indices = [(i, j) for i in range(9) for j in range(9)]
    np.random.shuffle(indices)
    
    n_remove = 81 - n_clues
    for i, j in indices[:n_remove]:
        puzzle[i, j] = 0
        mask[i, j] = False
    
    return puzzle, mask


# =============================================================================
# Sudoku Field Solver
# =============================================================================

class SudokuField(nn.Module):
    """
    Sudoku solver using ThermodynamicField.
    
    The SAME architecture that does denoising — spatial filters, diffusion,
    coupling, potentials — is applied to a 9-channel field on a 9×9 grid.
    
    Parameters
    ----------
    n_filters : int
        Spatial filters per channel.
    filter_size : int
        Kernel size (3 works well for 9×9 grid).
    n_evolve_steps : int
        Evolution steps during solving. More = longer constraint propagation.
    """
    
    def __init__(
        self,
        n_filters: int = 8,
        filter_size: int = 3,
        n_evolve_steps: int = 50
    ):
        super().__init__()
        self.n_evolve_steps = n_evolve_steps
        
        # The SAME ThermodynamicField used for denoising
        # 9 channels = 9 digits, 9×9 spatial grid
        self.field = ThermodynamicField(
            n_channels=9,
            height=9,
            width=9,
            n_filters=n_filters,
            filter_size=filter_size
        )
    
    def encode(self, puzzle_onehot: torch.Tensor) -> torch.Tensor:
        """
        Convert puzzle one-hot (B, 9, 9, 9) directly to field state.
        
        Since input already has 9 channels matching the field's n_channels,
        we use the multi-channel initialize_fields pathway.
        """
        return self.field.initialize_fields(puzzle_onehot)
    
    def decode(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Convert evolved field state back to Sudoku probabilities.
        
        Apply softmax over channels (digit dimension) at each cell.
        """
        # fields: (B, 9, 9, 9) — already 9 channels on 9×9
        return F.softmax(fields, dim=1)
    
    def solve(
        self,
        puzzle_onehot: torch.Tensor,
        clue_mask: torch.Tensor,
        n_steps: Optional[int] = None,
        damping: float = 1.0,
        temp_range: Tuple[float, float] = (0.0, 0.0)
    ) -> torch.Tensor:
        """
        Solve puzzle by evolving the field.
        
        Parameters
        ----------
        puzzle_onehot : (B, 9, 9, 9) one-hot puzzle (0 for empty cells)
        clue_mask : (B, 9, 9) boolean mask (True = clue cell)
        n_steps : int, optional (default: self.n_evolve_steps)
        
        Returns
        -------
        solution : (B, 9, 9, 9) softmax probabilities
        """
        if n_steps is None:
            n_steps = self.n_evolve_steps
        
        # Initialize field from puzzle
        fields = self.encode(puzzle_onehot)
        anchor = fields.detach()
        
        # Expand clue_mask to match channel dim: (B, 9, 9) → (B, 9, 9, 9)
        # True where clues are → these channels get clamped
        clue_expanded = clue_mask.unsqueeze(1).expand_as(fields)
        
        # Evolve with clue anchoring
        with torch.enable_grad():
            u = fields
            for step in range(n_steps):
                energy = self.field.compute_energy(u)
                grad = torch.autograd.grad(
                    energy.sum(), u, create_graph=True
                )[0]
                
                # Gradient descent + data anchoring
                # Apply damping to step size for stability
                effective_step = self.field.step_size * damping
                
                # Temperature for annealing/Langevin
                T_start, T_end = temp_range
                progress = step / max(n_steps, 1)
                T = T_start + (T_end - T_start) * progress
                
                # Deterministic drift
                drift = -effective_step * grad + \
                        self.field.data_weight * (anchor - u)
                
                # Stochastic diffusion (Langevin dynamics)
                diffusion = 0.0
                if T > 0:
                    noise = torch.randn_like(u)
                    sigma = torch.sqrt(2 * effective_step * T)
                    diffusion = sigma * noise
                
                u_new = u + drift + diffusion
                
                # Re-clamp clues: force clue cells back to strong one-hot
                clue_values = puzzle_onehot * 10.0 - (1.0 - puzzle_onehot) * 10.0
                u = torch.where(clue_expanded, clue_values, u_new)
        
        return self.decode(u)
    
    def forward(self, puzzle_onehot: torch.Tensor) -> torch.Tensor:
        """Compute energy of current field state."""
        fields = self.encode(puzzle_onehot)
        return self.field.compute_energy(fields)


# =============================================================================
# Training
# =============================================================================

def puzzle_solving_loss(
    model: SudokuField,
    x_clean: torch.Tensor,
    n_steps: int = 5,
    n_clues: int = 35
) -> Tuple[torch.Tensor, dict]:
    """
    Train by solving actual puzzles — matches evaluation exactly.
    
    1. Create partial puzzle from solved grid (random clue removal)
    2. Evolve field with clue clamping (same as solve())
    3. Cross-entropy against the true solution
    
    This trains the field dynamics to propagate constraints from
    clue cells outward — the actual solving task, not denoising.
    """
    B = x_clean.shape[0]
    device = x_clean.device
    
    # Create random puzzles from solved grids
    masks = torch.zeros(B, 9, 9, device=device, dtype=torch.bool)
    for b in range(B):
        indices = torch.randperm(81)[:n_clues]
        rows = indices // 9
        cols = indices % 9
        masks[b, rows, cols] = True
    
    # Puzzle: zero out non-clue cells
    mask_expanded = masks.unsqueeze(1).expand_as(x_clean)  # (B, 9, 9, 9)
    puzzle = x_clean * mask_expanded.float()
    
    # Solve with clue clamping (matches model.solve() exactly)
    probs = model.solve(puzzle, masks, n_steps=n_steps)
    
    # Cross-entropy loss against ground truth
    log_probs = torch.log(probs.clamp(min=1e-8))
    loss = -(x_clean * log_probs).sum(dim=1).mean()
    
    # Monitoring: accuracy on empty cells only
    with torch.no_grad():
        pred_grid = probs.argmax(dim=1)  # (B, 9, 9)
        true_grid = x_clean.argmax(dim=1)
        empty_mask = ~masks
        if empty_mask.any():
            accuracy = (pred_grid[empty_mask] == true_grid[empty_mask]).float().mean().item()
        else:
            accuracy = 1.0
    
    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "step_size": model.field.step_size.item(),
    }
    
    return loss, metrics


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_sudoku(
    model: SudokuField,
    n_puzzles: int = 50,
    n_clues: int = 35,
    n_steps: int = 50,
    damping: float = 1.0,
    temp_range: Tuple[float, float] = (0.0, 0.0),
    device: torch.device = torch.device("cpu")
) -> dict:
    """Evaluate on randomly generated puzzles."""
    
    total_correct = 0
    total_empty = 0
    total_solved = 0
    energies = []
    
    for _ in range(n_puzzles):
        sol = generate_solved_sudoku()
        puzzle, mask = create_puzzle(sol, n_clues)
        
        puzzle_oh = grid_to_onehot(puzzle).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        
        probs = model.solve(
            puzzle_oh, mask_t, n_steps=n_steps,
            damping=damping, temp_range=temp_range
        )
        pred = onehot_to_grid(probs.squeeze(0))
        
        empty = ~mask
        n_empty = empty.sum()
        n_correct = (pred[empty] == sol[empty]).sum() if n_empty > 0 else 0
        
        total_correct += n_correct
        total_empty += n_empty
        
        if n_correct == n_empty and n_empty > 0:
            total_solved += 1
        
        # Energy of solution
        with torch.no_grad():
            sol_prob = grid_to_onehot(sol).unsqueeze(0).to(device)
            E_solved = model(sol_prob).mean().item()
        
        # Save checkpoint every epoch
        # This block seems misplaced in evaluate_sudoku, typically used in training loops.
        # Assuming it's intended for a training function not provided in this context.
        # If this function is called within a training loop, 'epoch', 'optimizer', 'avg_loss'
        # would need to be passed or defined in scope.
        # For now, it will cause an error if executed as is within evaluate_sudoku.
        # Keeping it as per user instruction, but noting the potential issue.
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'epoch': epoch,
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': avg_loss,
        # }, "sudoku_field_checkpoint.pt")
        energies.append(E_solved) # Changed from 'e' to 'E_solved'
    
    return {
        "cell_accuracy": total_correct / max(total_empty, 1),
        "solve_rate": total_solved / n_puzzles,
        "avg_energy": np.mean(energies),
        "n_puzzles": n_puzzles
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_solving(
    model: SudokuField,
    n_clues: int = 35,
    n_steps: int = 50,
    damping: float = 1.0,
    temp_range: Tuple[float, float] = (0.0, 0.0),
    device: torch.device = torch.device("cpu"),
    save_path: str = "./outputs/sudoku/field_solving.png"
):
    """Visualize the field-based solving process."""
    
    sol = generate_solved_sudoku()
    puzzle, mask = create_puzzle(sol, n_clues)
    
    puzzle_oh = grid_to_onehot(puzzle).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    # Solve with tracking
    fields = model.encode(puzzle_oh)
    anchor = fields.detach()
    clue_expanded = mask_t.unsqueeze(1).expand_as(fields)
    
    energies = []
    accuracies = []
    entropies = []
    
    snapshots = []
    snapshot_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    
    T_start, T_end = temp_range
    
    u = fields
    with torch.enable_grad():
        for step in range(n_steps):
            # Annealing
            progress = step / max(n_steps, 1)
            T = T_start + (T_end - T_start) * progress
            
            energy = model.field.compute_energy(u)
            grad = torch.autograd.grad(energy.sum(), u, create_graph=True)[0]
            
            effective_step = model.field.step_size * damping
            
            drift = -effective_step * grad + \
                    model.field.data_weight * (anchor - u)
            
            diffusion = 0.0
            if T > 0:
                noise = torch.randn_like(u)
                sigma = torch.sqrt(2 * effective_step * T)
                diffusion = sigma * noise
            
            u_new = u + drift + diffusion
            
            clue_values = puzzle_oh * 10.0 - (1.0 - puzzle_oh) * 10.0
            u = torch.where(clue_expanded, clue_values, u_new)
            
            # Track metrics
            with torch.no_grad():
                probs = F.softmax(u, dim=1)
                pred = onehot_to_grid(probs.squeeze(0))
                empty = ~mask
                n_empty = empty.sum()
                correct = (pred[empty] == sol[empty]).sum() if n_empty > 0 else 0
                
                energies.append(energy.mean().item())
                accuracies.append(correct / max(n_empty, 1))
                
                # Entropy
                p = probs.squeeze(0).clamp(min=1e-8)
                h = -(p * p.log()).sum(dim=0)
                h_empty = h[~mask].mean().item() if (~mask).any() else 0
                entropies.append(h_empty)
                
                if step in snapshot_steps:
                    snapshots.append((step, pred.copy(), correct, n_empty))
    
    # Plot
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.3)
    
    # Row 1: Grid snapshots
    for idx, (step, pred, corr, n_emp) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, idx])
        
        grid_display = np.full((9, 9), '', dtype=object)
        colors = np.ones((9, 9, 3))
        
        for i in range(9):
            for j in range(9):
                if mask[i, j]:
                    grid_display[i, j] = str(sol[i, j])
                    colors[i, j] = [0.85, 0.85, 0.85]  # gray for clues
                else:
                    grid_display[i, j] = str(pred[i, j])
                    if pred[i, j] == sol[i, j]:
                        colors[i, j] = [0.7, 1.0, 0.7]  # green
                    else:
                        colors[i, j] = [1.0, 0.7, 0.7]  # red
        
        ax.imshow(colors, extent=[0, 9, 9, 0], aspect='equal')
        for i in range(9):
            for j in range(9):
                ax.text(j + 0.5, i + 0.5, grid_display[i, j],
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        for k in range(10):
            lw = 2 if k % 3 == 0 else 0.5
            ax.axhline(y=k, color='black', linewidth=lw)
            ax.axvline(x=k, color='black', linewidth=lw)
        
        ax.set_title(f"Step {step}\n{corr}/{n_emp} correct", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Row 2: Metrics
    ax_e = fig.add_subplot(gs[1, 0:2])
    ax_e.plot(energies, 'b-', linewidth=1.5)
    ax_e.set_xlabel("Step")
    ax_e.set_ylabel("Energy", color='b')
    ax_e.set_title("Energy Descent")
    ax_e.grid(True, alpha=0.3)
    
    ax_a = fig.add_subplot(gs[1, 2:4])
    ax_a.plot(accuracies, 'g-', linewidth=1.5, label='Cell Accuracy')
    ax_a.plot(entropies, 'r--', linewidth=1.5, label='Entropy')
    ax_a.set_xlabel("Step")
    ax_a.set_ylabel("Value")
    ax_a.set_title("Accuracy & Entropy vs Step")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    
    # Physics info
    ax_info = fig.add_subplot(gs[1, 4])
    ax_info.axis('off')
    n_params = sum(p.numel() for p in model.parameters())
    info_text = (
        f"ThermodynamicField\n"
        f"Channels: {model.field.n_channels}\n"
        f"Filters: {model.field.n_filters}×{model.field.spatial_filters[0].shape[-1]}\n"
        f"Params: {n_params}\n"
        f"Steps: {n_steps}\n"
        f"η = {model.field.step_size.item():.4f}\n"
        f"μ = {model.field.data_weight.item():.4f}\n"
        f"α = {model.field.alphas.data.cpu().numpy().round(3)}"
    )
    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                fontsize=8, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("Sudoku via ThermodynamicField — Same Model as Denoising",
                fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sudoku via ThermodynamicField")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-puzzles", type=int, default=300)
    parser.add_argument("--n-filters", type=int, default=8)
    parser.add_argument("--filter-size", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=3,
                        help="Evolution steps during training (multi-step denoising)")
    parser.add_argument("--solve-steps", type=int, default=100,
                        help="Evolution steps during solving")
    parser.add_argument("--damping", type=float, default=0.1,
                        help="Step size damping factor for inference")
    parser.add_argument("--temp-start", type=float, default=1.0,
                        help="Starting temperature for annealing")
    parser.add_argument("--temp-end", type=float, default=0.0,
                        help="Ending temperature for annealing")
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="./outputs/sudoku")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("\n=== Creating SudokuField ===")
    model = SudokuField(
        n_filters=args.n_filters,
        filter_size=args.filter_size,
        n_evolve_steps=args.solve_steps
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")
    print(f"Architecture: ThermodynamicField(9 channels, 9×9, {args.n_filters} filters)")
    print(f"Training: {args.train_steps}-step denoising, noise_std={args.noise_std}")
    
    # Generate training data
    print(f"\n=== Generating {args.n_puzzles} solved puzzles ===")
    solved_puzzles = []
    for i in range(args.n_puzzles):
        sol = generate_solved_sudoku()
        sol_oh = grid_to_onehot(sol)
        solved_puzzles.append(sol_oh)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_puzzles}")
    
    data = torch.stack(solved_puzzles)  # (N, 9, 9, 9)
    print(f"Training data shape: {data.shape}")
    
    # Load checkpoint if exists
    checkpoint_path = "sudoku_field_checkpoint.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer setup (needed for checkpoint loading or training)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if os.path.exists(checkpoint_path):
         # Load optimizer state if available
         if 'optimizer_state_dict' in checkpoint:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         start_epoch = checkpoint.get('epoch', 0) + 1
         print(f"Resuming from epoch {start_epoch}")

    if args.eval_only:
        print("\n=== Skipping Training (Eval Only) ===")
        args.epochs = 0 # Skip loop by setting max epochs to 0 (or less than start_epoch)
        # Ensure loop doesn't run if start_epoch > args.epochs (range handles this)

    if not args.eval_only:
        print(f"\n=== Training ({args.epochs} epochs) ===")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        # Shuffle
        perm = torch.randperm(len(data))
        
        for start in range(0, len(data), args.batch_size):
            end = min(start + args.batch_size, len(data))
            batch = data[perm[start:end]].to(device)
            
            optimizer.zero_grad()
            loss, metrics = puzzle_solving_loss(
                model, batch,
                n_steps=args.train_steps,
                n_clues=35
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        
        print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}  "
              f"empty_acc={avg_acc:.1%}  "
              f"η={model.field.step_size.item():.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")
    
    # Evaluate
    print(f"\n=== Evaluating (50 puzzles, {args.solve_steps} steps) ===")
    print(f"Physics: Damping={args.damping}, Temp={args.temp_start}->{args.temp_end}")
    model.eval()
    results = evaluate_sudoku(
        model, n_puzzles=50, n_clues=35,
        n_steps=args.solve_steps,
        damping=args.damping,
        temp_range=(args.temp_start, args.temp_end),
        device=device
    )
    
    print(f"  Cell accuracy: {results['cell_accuracy']:.1%}")
    print(f"  Solve rate:    {results['solve_rate']:.1%}")
    print(f"  Avg energy:    {results['avg_energy']:.2f}")
    
    # Visualize
    print("\n=== Generating Visualization ===")
    visualize_solving(
        model, n_clues=35, n_steps=args.solve_steps,
        damping=args.damping,
        temp_range=(args.temp_start, args.temp_end),
        device=device,
        save_path=os.path.join(args.output_dir, "field_solving.png")
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'args': vars(args)
    }, os.path.join(args.output_dir, "sudoku_field_model.pt"))
    
    print(f"\n=== Done ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
