"""
Sudoku TNF: Constraint Satisfaction via Energy Minimization
============================================================

This script proves that Thermodynamic Neural Field (TNF) energy
minimization can solve hard LOGICAL constraints — not just visual
denoising.

The Core Idea
-------------
A Sudoku puzzle is a Constraint Satisfaction Problem (CSP):
- Each cell must contain exactly one digit 1-9
- Each row must contain all digits 1-9 (no repeats)
- Each column must contain all digits 1-9 (no repeats)
- Each 3x3 box must contain all digits 1-9 (no repeats)

We encode these constraints as ENERGY TERMS in a differentiable
energy functional. The "solved" state is the global energy minimum.

    Unsolved puzzle = High energy (constraints violated)
    Solved puzzle   = Zero energy (all constraints satisfied)
    Solving         = Gradient descent on E(u)

This is identical in principle to how thermodynamic_field.py uses
energy minimization for denoising — but here the "noise" is missing
digits and the "energy landscape" encodes logical rules.

Representation
--------------
- State u: (9, 9, 9) tensor — continuous relaxation of one-hot
  u[i, j, d] ≈ P(cell (i,j) = digit d+1)
- Softmax over digit dimension ensures valid probabilities
- Clues: hard-clamped to one-hot vectors

Usage
-----
    python src/sudoku_tnf.py                    # Train + test
    python src/sudoku_tnf.py --n-clues 30       # Harder puzzles (fewer clues)
    python src/sudoku_tnf.py --visualize        # Save convergence plots
"""

import argparse
import os
import sys
import random
from typing import Tuple, List, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================================
# Sudoku Puzzle Generator
# =========================================================================

def generate_solved_sudoku() -> np.ndarray:
    """
    Generate a random fully solved 9x9 Sudoku grid.
    
    Uses a backtracking solver with random digit ordering
    to produce uniformly random valid solutions.
    
    Returns
    -------
    grid : np.ndarray, shape (9, 9), dtype int
        Solved Sudoku grid with values 1-9.
    """
    grid = np.zeros((9, 9), dtype=int)
    
    def is_valid(g, row, col, num):
        # Check row
        if num in g[row, :]:
            return False
        # Check column
        if num in g[:, col]:
            return False
        # Check 3x3 box
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if num in g[box_r:box_r+3, box_c:box_c+3]:
            return False
        return True
    
    def solve(g):
        for i in range(9):
            for j in range(9):
                if g[i, j] == 0:
                    digits = list(range(1, 10))
                    random.shuffle(digits)
                    for num in digits:
                        if is_valid(g, i, j, num):
                            g[i, j] = num
                            if solve(g):
                                return True
                            g[i, j] = 0
                    return False
        return True
    
    solve(grid)
    return grid


def create_puzzle(solution: np.ndarray, n_clues: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a puzzle by removing digits from a solved grid.
    
    Parameters
    ----------
    solution : np.ndarray (9, 9)
        Fully solved grid.
    n_clues : int
        Number of given clues (17-80). Fewer = harder.
    
    Returns
    -------
    puzzle : np.ndarray (9, 9)
        Puzzle with 0 = empty cell.
    mask : np.ndarray (9, 9)
        Boolean mask, True = clue given.
    """
    n_clues = max(17, min(n_clues, 80))  # Minimum 17 for unique solution
    
    puzzle = solution.copy()
    mask = np.ones((9, 9), dtype=bool)
    
    # Randomly remove cells
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    
    n_remove = 81 - n_clues
    for i, j in cells[:n_remove]:
        puzzle[i, j] = 0
        mask[i, j] = False
    
    return puzzle, mask


def grid_to_onehot(grid: np.ndarray) -> torch.Tensor:
    """Convert integer grid (9,9) with values 0-9 to one-hot (9,9,9)."""
    onehot = torch.zeros(9, 9, 9)
    for i in range(9):
        for j in range(9):
            if grid[i, j] > 0:
                onehot[i, j, grid[i, j] - 1] = 1.0
    return onehot


def onehot_to_grid(onehot: torch.Tensor) -> np.ndarray:
    """Convert probabilities (9,9,9) to integer grid (9,9) via argmax."""
    return (onehot.argmax(dim=-1) + 1).cpu().numpy()


# =========================================================================
# Sudoku Energy Functional
# =========================================================================

class SudokuTNF(nn.Module):
    """
    Thermodynamic Neural Field for Sudoku constraint satisfaction.
    
    Energy E(u) = w_row·E_row + w_col·E_col + w_box·E_box 
                + w_cell·E_cell + w_clue·E_clue
    
    Each term is a differentiable penalty for violating a Sudoku rule.
    The solved state is the global minimum with E = 0.
    
    Parameters
    ----------
    temperature : float
        Softmax temperature. Lower = sharper decisions.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        
        # Learnable constraint weights — the model discovers
        # the relative importance of each constraint type
        self.log_w_row = nn.Parameter(torch.tensor(0.0))
        self.log_w_col = nn.Parameter(torch.tensor(0.0))
        self.log_w_box = nn.Parameter(torch.tensor(0.0))
        self.log_w_cell = nn.Parameter(torch.tensor(0.0))
        self.log_w_clue = nn.Parameter(torch.tensor(2.0))  # Clues weighted high
        
        # Learnable step size for gradient descent
        self.log_step_size = nn.Parameter(torch.tensor(-1.0))
        
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature)))
    
    @property
    def step_size(self):
        return torch.exp(self.log_step_size)
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def compute_energy(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute total constraint energy.
        
        Parameters
        ----------
        u : torch.Tensor, shape (batch, 9, 9, 9)
            Soft digit probabilities (after softmax).
        
        Returns
        -------
        energy : torch.Tensor, shape (batch,)
            Total energy — 0 iff all constraints satisfied.
        """
        B = u.shape[0]
        
        w_row = torch.exp(self.log_w_row)
        w_col = torch.exp(self.log_w_col)
        w_box = torch.exp(self.log_w_box)
        w_cell = torch.exp(self.log_w_cell)
        
        # --- E_row: each digit should appear exactly once per row ---
        # Sum probabilities of digit d across columns in each row
        # If exactly one cell has digit d, sum = 1. Penalty = (sum - 1)²
        row_sums = u.sum(dim=2)  # (B, 9, 9) — sum over columns
        E_row = ((row_sums - 1.0) ** 2).sum(dim=(1, 2))
        
        # --- E_col: each digit should appear exactly once per column ---
        col_sums = u.sum(dim=1)  # (B, 9, 9) — sum over rows
        E_col = ((col_sums - 1.0) ** 2).sum(dim=(1, 2))
        
        # --- E_box: each digit should appear exactly once per 3x3 box ---
        E_box = torch.zeros(B, device=u.device)
        for br in range(3):
            for bc in range(3):
                box = u[:, br*3:(br+1)*3, bc*3:(bc+1)*3, :]  # (B, 3, 3, 9)
                box_sums = box.sum(dim=(1, 2))  # (B, 9)
                E_box = E_box + ((box_sums - 1.0) ** 2).sum(dim=1)
        
        # --- E_cell: each cell should commit to one digit ---
        # Entropy-like penalty: maximize sharpness
        # For a perfect one-hot, max(u) = 1 → penalty = 0
        cell_max = u.max(dim=-1).values  # (B, 9, 9)
        E_cell = ((1.0 - cell_max) ** 2).sum(dim=(1, 2))
        
        # Total energy
        energy = w_row * E_row + w_col * E_col + w_box * E_box + w_cell * E_cell
        
        return energy
    
    def one_step_denoise(
        self,
        u_logits: torch.Tensor,
        clue_onehot: torch.Tensor,
        clue_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Single gradient descent step on the energy landscape.
        
        Parameters
        ----------
        u_logits : torch.Tensor (batch, 9, 9, 9)
            Current logits (pre-softmax).
        clue_onehot : torch.Tensor (batch, 9, 9, 9)
            One-hot encoded clues.
        clue_mask : torch.Tensor (batch, 9, 9)
            Binary mask: 1 = clue given.
        
        Returns
        -------
        u_logits_new : torch.Tensor (batch, 9, 9, 9)
            Updated logits after one energy descent step.
        """
        u_logits = u_logits.requires_grad_(True)
        
        # Softmax to get probabilities
        u_prob = F.softmax(u_logits / self.temperature, dim=-1)
        
        # Compute energy
        energy = self.compute_energy(u_prob)
        
        # Gradient of energy w.r.t. logits
        grad = torch.autograd.grad(
            energy.sum(), u_logits, create_graph=True
        )[0]
        
        # Gradient descent step
        u_logits_new = u_logits - self.step_size * grad
        
        # Re-clamp clues: force clue cells back to their known values
        # Use large logits for clue cells to ensure softmax → one-hot
        clue_logits = clue_onehot * 20.0 - (1.0 - clue_onehot) * 20.0
        mask_expanded = clue_mask.unsqueeze(-1).expand_as(u_logits_new)
        u_logits_new = torch.where(mask_expanded.bool(), clue_logits, u_logits_new)
        
        return u_logits_new
    
    def solve(
        self,
        puzzle_onehot: torch.Tensor,
        clue_mask: torch.Tensor,
        n_steps: int = 100,
        return_trajectory: bool = False,
        T_start: float = 2.0,
        T_end: float = 0.05,
        noise_scale: float = 0.3,
        noise_decay: float = 0.97
    ) -> torch.Tensor:
        """
        Solve a Sudoku puzzle via simulated annealing on the energy landscape.
        
        Uses a cosine temperature schedule (T_start → T_end) combined with
        decaying noise injection to escape local minima.
        
        Parameters
        ----------
        puzzle_onehot : torch.Tensor (batch, 9, 9, 9)
            One-hot encoded puzzle (clues filled, empties = 0).
        clue_mask : torch.Tensor (batch, 9, 9)
            Binary mask: 1 = clue given.
        n_steps : int
            Number of gradient descent iterations.
        return_trajectory : bool
            If True, return all intermediate grids.
        T_start : float
            Initial temperature (high = explore, escape local minima).
        T_end : float
            Final temperature (low = sharpen, commit to digits).
        noise_scale : float
            Initial scale of noise injection for escaping local minima.
        noise_decay : float
            Multiplicative decay of noise per step.
        
        Returns
        -------
        solution : torch.Tensor (batch, 9, 9, 9) or (n_steps, batch, 9, 9, 9)
            Solved probabilities.
        """
        B = puzzle_onehot.shape[0]
        device = puzzle_onehot.device
        
        # Initialize logits: clues → strong one-hot, empties → slight noise
        clue_logits = puzzle_onehot * 20.0 - (1.0 - puzzle_onehot) * 20.0
        noise_logits = torch.randn(B, 9, 9, 9, device=device) * 0.5
        
        mask_expanded = clue_mask.unsqueeze(-1).expand_as(clue_logits)
        u_logits = torch.where(mask_expanded.bool(), clue_logits, noise_logits)
        
        trajectory = []
        current_noise = noise_scale
        
        with torch.enable_grad():
            for step in range(n_steps):
                u_logits = u_logits.detach().requires_grad_(True)
                
                # Cosine annealing: T_start → T_end
                progress = step / max(n_steps - 1, 1)
                T = T_end + 0.5 * (T_start - T_end) * (1 + np.cos(np.pi * progress))
                
                # Softmax with annealed temperature
                u_prob = F.softmax(u_logits / T, dim=-1)
                
                # Compute energy
                energy = self.compute_energy(u_prob)
                
                # Gradient descent
                grad = torch.autograd.grad(energy.sum(), u_logits)[0]
                
                u_logits_new = u_logits.detach() - self.step_size.detach() * grad.detach()
                
                # Noise injection: decaying kicks to escape local minima
                # Only on empty cells, scaled by current noise level
                if current_noise > 0.01:
                    noise = torch.randn_like(u_logits_new) * current_noise
                    noise = torch.where(mask_expanded.bool(), torch.zeros_like(noise), noise)
                    u_logits_new = u_logits_new + noise
                    current_noise *= noise_decay
                
                # Re-clamp clues
                clue_logits_step = puzzle_onehot * 20.0 - (1.0 - puzzle_onehot) * 20.0
                u_logits = torch.where(mask_expanded.bool(), clue_logits_step, u_logits_new)
                
                if return_trajectory:
                    u_prob_snap = F.softmax(u_logits / T, dim=-1)
                    trajectory.append(u_prob_snap.detach())
        
        # Final probabilities at coldest temperature
        u_final = F.softmax(u_logits / T_end, dim=-1)
        
        if return_trajectory:
            return torch.stack(trajectory, dim=0), u_final
        
        return u_final
    
    def forward(self, u_prob: torch.Tensor) -> torch.Tensor:
        """Compute energy for monitoring."""
        return self.compute_energy(u_prob)


# =========================================================================
# Training
# =========================================================================

def train_sudoku(
    model: SudokuTNF,
    n_puzzles: int = 2000,
    n_epochs: int = 30,
    batch_size: int = 32,
    n_clues: int = 35,
    lr: float = 3e-3,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Train the SudokuTNF using 1-step denoising on solved puzzles.
    
    Training loop:
    1. Generate solved Sudoku → one-hot target
    2. Create puzzle (remove cells) → partially filled input
    3. Run one_step_denoise → predicted probabilities
    4. Loss = MSE(predicted, target) over empty cells
    
    This teaches the energy gradient to restore valid solutions.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Pre-generate training data
    print("Generating Sudoku puzzles...")
    solutions = []
    puzzles = []
    masks = []
    
    for _ in tqdm(range(n_puzzles)):
        sol = generate_solved_sudoku()
        puz, mask = create_puzzle(sol, n_clues=n_clues)
        solutions.append(grid_to_onehot(sol))
        puzzles.append(grid_to_onehot(puz))
        masks.append(torch.from_numpy(mask.astype(np.float32)))
    
    solutions = torch.stack(solutions)  # (N, 9, 9, 9)
    puzzles = torch.stack(puzzles)      # (N, 9, 9, 9)
    masks = torch.stack(masks)          # (N, 9, 9)
    
    history = {"loss": [], "energy_solved": [], "energy_puzzle": []}
    
    print(f"\nTraining SudokuTNF ({sum(p.numel() for p in model.parameters())} params)")
    print(f"  Puzzles: {n_puzzles}, Clues: {n_clues}, Batch: {batch_size}")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle
        perm = torch.randperm(n_puzzles)
        
        for start in range(0, n_puzzles, batch_size):
            end = min(start + batch_size, n_puzzles)
            idx = perm[start:end]
            
            sol_batch = solutions[idx].to(device)
            puz_batch = puzzles[idx].to(device)
            mask_batch = masks[idx].to(device)
            
            optimizer.zero_grad()
            
            # Initialize logits from puzzle
            clue_logits = puz_batch * 20.0 - (1.0 - puz_batch) * 20.0
            noise_logits = torch.randn_like(puz_batch) * 0.1
            mask_exp = mask_batch.unsqueeze(-1).expand_as(puz_batch)
            u_logits = torch.where(mask_exp.bool(), clue_logits, noise_logits)
            
            # One step denoising
            u_logits_new = model.one_step_denoise(u_logits, puz_batch, mask_batch)
            
            # Predicted probabilities
            u_pred = F.softmax(u_logits_new / model.temperature, dim=-1)
            
            # Loss: MSE on EMPTY cells only (clues are already correct)
            empty_mask = (~mask_batch.bool()).unsqueeze(-1).expand_as(u_pred).float()
            n_empty = empty_mask.sum().clamp(min=1.0)
            loss = (empty_mask * (u_pred - sol_batch) ** 2).sum() / n_empty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        
        # Monitor energy
        with torch.no_grad():
            sample_sol = solutions[:16].to(device)
            sample_puz = puzzles[:16].to(device)
            puz_prob = F.softmax(sample_puz * 10.0 / model.temperature, dim=-1)
            sol_prob = F.softmax(sample_sol * 10.0 / model.temperature, dim=-1)
            E_puzzle = model(puz_prob).mean().item()
            E_solved = model(sol_prob).mean().item()
        
        history["loss"].append(avg_loss)
        history["energy_solved"].append(E_solved)
        history["energy_puzzle"].append(E_puzzle)
        
        step_size = model.step_size.item()
        temp = model.temperature.item()
        
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"E(solved)={E_solved:.2f} E(puzzle)={E_puzzle:.2f} | "
              f"η={step_size:.3f} T={temp:.3f}")
    
    return history


# =========================================================================
# Evaluation
# =========================================================================

@torch.no_grad()
def evaluate_sudoku(
    model: SudokuTNF,
    n_test: int = 100,
    n_clues: int = 35,
    n_steps: int = 200,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Evaluate Sudoku solving accuracy.
    
    Returns
    -------
    metrics : dict
        solve_rate, cell_accuracy, avg_energy, etc.
    """
    model.eval()
    
    correct_puzzles = 0
    total_cells_correct = 0
    total_empty_cells = 0
    energies = []
    
    for _ in tqdm(range(n_test), desc="Testing"):
        sol = generate_solved_sudoku()
        puz, mask = create_puzzle(sol, n_clues=n_clues)
        
        sol_tensor = grid_to_onehot(sol).unsqueeze(0).to(device)
        puz_tensor = grid_to_onehot(puz).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
        
        # Solve
        u_final = model.solve(puz_tensor, mask_tensor, n_steps=n_steps)
        
        # Convert to grid
        predicted = onehot_to_grid(u_final.squeeze(0))
        
        # Compute energy
        energy = model(u_final).item()
        energies.append(energy)
        
        # Accuracy on empty cells only
        empty_cells = ~mask
        n_empty = empty_cells.sum()
        
        if n_empty > 0:
            correct = (predicted[empty_cells] == sol[empty_cells]).sum()
            total_cells_correct += correct
            total_empty_cells += n_empty
            
            if correct == n_empty:
                correct_puzzles += 1
    
    cell_accuracy = total_cells_correct / max(total_empty_cells, 1) * 100
    solve_rate = correct_puzzles / n_test * 100
    avg_energy = np.mean(energies)
    
    print(f"\n=== Sudoku Evaluation ===")
    print(f"  Test puzzles:    {n_test}")
    print(f"  Clues per puzzle: {n_clues}")
    print(f"  Solve steps:     {n_steps}")
    print(f"  Cell accuracy:   {cell_accuracy:.1f}%")
    print(f"  Full solve rate: {solve_rate:.1f}%")
    print(f"  Avg energy:      {avg_energy:.4f}")
    
    return {
        "cell_accuracy": cell_accuracy,
        "solve_rate": solve_rate,
        "avg_energy": avg_energy,
    }


# =========================================================================
# Visualization
# =========================================================================

def visualize_sudoku_solving(
    model: SudokuTNF,
    n_clues: int = 35,
    n_steps: int = 200,
    device: torch.device = torch.device("cpu"),
    save_path: str = "./outputs/sudoku_solving.png"
):
    """
    Visualize step-by-step Sudoku solving via energy descent.
    
    Creates a figure showing:
    - Row 1: Puzzle → intermediate states → solution
    - Row 2: Energy + Temperature over iterations
    """
    model.eval()
    
    # Generate a puzzle
    sol = generate_solved_sudoku()
    puz, mask = create_puzzle(sol, n_clues=n_clues)
    
    sol_tensor = grid_to_onehot(sol).unsqueeze(0).to(device)
    puz_tensor = grid_to_onehot(puz).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    
    # Solve with trajectory
    trajectory, u_final = model.solve(
        puz_tensor, mask_tensor, n_steps=n_steps, return_trajectory=True
    )
    
    # Compute energy and entropy at each step
    energies = []
    entropies = []
    temperatures = []
    for t in range(trajectory.shape[0]):
        u_t = trajectory[t]  # (1, 9, 9, 9)
        e = model(u_t).item()
        energies.append(e)
        
        # Per-cell entropy: H = -Σ p·log(p) over empty cells
        p = u_t[0]  # (9, 9, 9)
        log_p = torch.log(p.clamp(min=1e-8))
        cell_entropy = -(p * log_p).sum(dim=-1)  # (9, 9)
        empty_mask = ~mask_tensor[0].bool()  # (9, 9)
        avg_entropy = cell_entropy[empty_mask].mean().item() if empty_mask.any() else 0.0
        entropies.append(avg_entropy)
        
        # Temperature schedule
        progress = t / max(n_steps - 1, 1)
        T = 0.05 + 0.5 * (2.0 - 0.05) * (1 + np.cos(np.pi * progress))
        temperatures.append(T)
    
    # Check final accuracy
    predicted = onehot_to_grid(u_final.squeeze(0))
    empty_cells = ~mask
    n_empty = empty_cells.sum()
    correct = (predicted[empty_cells] == sol[empty_cells]).sum() if n_empty > 0 else 0
    
    # Select frames to show
    n_frames = 8
    frame_indices = np.linspace(0, len(energies) - 1, n_frames).astype(int)
    
    fig = plt.figure(figsize=(n_frames * 2.5, 10))
    
    # Row 1: Sudoku grids
    for col, t_idx in enumerate(frame_indices):
        ax = fig.add_subplot(3, n_frames, col + 1)
        grid_t = onehot_to_grid(trajectory[t_idx, 0])
        T_now = temperatures[t_idx]
        draw_sudoku_grid(ax, grid_t, puz, mask, title=f"Step {t_idx}\nT={T_now:.2f}")
    
    # Row 2: Energy + Temperature
    ax_energy = fig.add_subplot(3, 1, 2)
    ax_energy.plot(energies, 'b-', linewidth=2, label='Energy E(u)')
    ax_energy.set_ylabel("Energy E(u)", fontsize=12, color='blue')
    ax_energy.tick_params(axis='y', labelcolor='blue')
    ax_energy.grid(True, alpha=0.3)
    ax_twin = ax_energy.twinx()
    ax_twin.plot(temperatures, 'r--', linewidth=1.5, alpha=0.7, label='Temperature T')
    ax_twin.set_ylabel("Temperature T", fontsize=12, color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')
    ax_energy.set_title(f"Simulated Annealing: Energy Descent + Cooling ({correct}/{n_empty} cells correct)", fontsize=13)
    for t_idx in frame_indices:
        ax_energy.axvline(t_idx, color='gray', alpha=0.3, linestyle='--')
    lines1, labels1 = ax_energy.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_energy.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Row 3: Entropy
    ax_entropy = fig.add_subplot(3, 1, 3)
    ax_entropy.plot(entropies, 'g-', linewidth=2)
    ax_entropy.axhline(0.0, color='black', linestyle=':', alpha=0.5, label='Perfect certainty')
    ax_entropy.axhline(np.log(9), color='red', linestyle=':', alpha=0.5, label='Max entropy (uniform)')
    ax_entropy.set_xlabel("Gradient Descent Step", fontsize=12)
    ax_entropy.set_ylabel("Avg Cell Entropy H(p)", fontsize=12)
    ax_entropy.set_title("Entropy: Uncertainty → Confidence (lower = more committed)", fontsize=13)
    ax_entropy.legend()
    ax_entropy.grid(True, alpha=0.3)
    for t_idx in frame_indices:
        ax_entropy.axvline(t_idx, color='gray', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_entropy_heatmap(
    model: SudokuTNF,
    n_clues: int = 35,
    n_steps: int = 300,
    device: torch.device = torch.device("cpu"),
    save_path: str = "./outputs/sudoku_entropy.png"
):
    """
    Visualize per-cell entropy heatmap at different stages.
    Shows whether errors are 'confidently wrong' or 'confused'.
    """
    model.eval()
    
    sol = generate_solved_sudoku()
    puz, mask = create_puzzle(sol, n_clues=n_clues)
    
    puz_tensor = grid_to_onehot(puz).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    
    trajectory, u_final = model.solve(
        puz_tensor, mask_tensor, n_steps=n_steps, return_trajectory=True
    )
    
    # Show entropy heatmaps at 4 stages
    stages = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for col, t_idx in enumerate(stages):
        p = trajectory[t_idx, 0]  # (9, 9, 9)
        
        # Entropy heatmap
        log_p = torch.log(p.clamp(min=1e-8))
        cell_entropy = -(p * log_p).sum(dim=-1).cpu().numpy()  # (9, 9)
        
        ax = axes[0, col]
        im = ax.imshow(cell_entropy, cmap='RdYlGn_r', vmin=0, vmax=np.log(9))
        ax.set_title(f"Step {t_idx}\nH avg={cell_entropy[~mask].mean():.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Mark clue cells
        for i in range(9):
            for j in range(9):
                if mask[i, j]:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=False, edgecolor='blue', linewidth=1.5))
        
        # Show digit grid below
        ax2 = axes[1, col]
        grid_t = onehot_to_grid(trajectory[t_idx, 0])
        draw_sudoku_grid(ax2, grid_t, puz, mask, title="")
    
    fig.colorbar(im, ax=axes[0, :], label='Entropy H (0=certain, 2.2=random)', shrink=0.8)
    fig.suptitle('Per-Cell Entropy Heatmap: Red=Confused, Green=Confident', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def draw_sudoku_grid(ax, grid, puzzle, mask, title=""):
    """Draw a Sudoku grid with colors for clues vs solved cells."""
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Fill cells
    for i in range(9):
        for j in range(9):
            val = grid[i, j]
            is_clue = mask[i, j]
            
            # Background color
            if is_clue:
                color = '#e8e8e8'  # Light gray for clues
            else:
                color = '#d4edda' if val == 0 else '#c3e6f5'  # Green-ish for empty, blue for filled
            
            rect = plt.Rectangle((j, 8 - i), 1, 1, facecolor=color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
            
            # Digit text
            if val > 0:
                fontweight = 'bold' if is_clue else 'normal'
                fontcolor = 'black' if is_clue else '#0066cc'
                ax.text(j + 0.5, 8 - i + 0.5, str(val),
                       ha='center', va='center', fontsize=7,
                       fontweight=fontweight, color=fontcolor)
    
    # Draw thick box borders
    for i in range(4):
        ax.axhline(i * 3, color='black', linewidth=2)
        ax.axvline(i * 3, color='black', linewidth=2)


def visualize_energy_landscape(
    model: SudokuTNF,
    n_clues: int = 35,
    device: torch.device = torch.device("cpu"),
    save_path: str = "./outputs/sudoku_energy.png"
):
    """
    Show that E(solved) < E(partial) < E(random).
    """
    model.eval()
    
    n_samples = 50
    E_solved = []
    E_partial = []
    E_random = []
    
    for _ in range(n_samples):
        sol = generate_solved_sudoku()
        puz, mask = create_puzzle(sol, n_clues=n_clues)
        
        sol_oh = grid_to_onehot(sol).unsqueeze(0).to(device)
        puz_oh = grid_to_onehot(puz).unsqueeze(0).to(device)
        
        # Solved: sharp one-hot
        sol_prob = F.softmax(sol_oh * 10.0, dim=-1)
        E_solved.append(model(sol_prob).item())
        
        # Partial: clues sharp, empties uniform
        partial_prob = puz_oh.clone()
        empty = (~torch.from_numpy(mask).bool()).unsqueeze(-1).expand_as(partial_prob)
        partial_prob[empty] = 1.0 / 9.0  # Uniform over digits
        E_partial.append(model(partial_prob.to(device)).item())
        
        # Random: all cells uniform
        random_prob = torch.ones(1, 9, 9, 9, device=device) / 9.0
        E_random.append(model(random_prob).item())
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    positions = [1, 2, 3]
    bp = ax.boxplot(
        [E_solved, E_partial, E_random],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        labels=["Solved\n(all constraints\nsatisfied)", 
                "Partial\n(clues only,\nempties uniform)",
                "Random\n(all cells\nuniform)"]
    )
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Energy E(u)", fontsize=13)
    ax.set_title("Sudoku Energy Landscape:\nSolved = Low Energy, Random = High Energy", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Sudoku TNF: Constraint Satisfaction via Energy")
    parser.add_argument("--n-puzzles", type=int, default=2000, help="Training puzzles")
    parser.add_argument("--n-test", type=int, default=100, help="Test puzzles")
    parser.add_argument("--n-clues", type=int, default=35, help="Clues per puzzle (17-80)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--solve-steps", type=int, default=200, help="Gradient descent steps for solving")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output-dir", type=str, default="./outputs/sudoku")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = SudokuTNF(temperature=0.5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SudokuTNF parameters: {n_params}")
    print(f"  Learnable: constraint weights (5), step size (1), temperature (1)")
    
    # Train
    print("\n=== Training ===")
    history = train_sudoku(
        model, 
        n_puzzles=args.n_puzzles,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_clues=args.n_clues,
        lr=args.lr,
        device=device
    )
    
    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["loss"], 'b-', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("1-Step Denoising Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history["energy_solved"], 'g-', linewidth=2, label="E(solved)")
    ax2.plot(history["energy_puzzle"], 'r-', linewidth=2, label="E(puzzle)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Energy")
    ax2.set_title("Learned Energy: E(solved) < E(puzzle)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Evaluate
    print("\n=== Evaluation ===")
    metrics = evaluate_sudoku(
        model,
        n_test=args.n_test,
        n_clues=args.n_clues,
        n_steps=args.solve_steps,
        device=device
    )
    
    # Visualizations
    print("\n=== Visualizations ===")
    
    visualize_energy_landscape(
        model, n_clues=args.n_clues, device=device,
        save_path=os.path.join(args.output_dir, "energy_landscape.png")
    )
    
    visualize_sudoku_solving(
        model, n_clues=args.n_clues, n_steps=args.solve_steps, device=device,
        save_path=os.path.join(args.output_dir, "solving_evolution.png")
    )
    
    visualize_entropy_heatmap(
        model, n_clues=args.n_clues, n_steps=args.solve_steps, device=device,
        save_path=os.path.join(args.output_dir, "entropy_heatmap.png")
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'history': history,
        'args': vars(args)
    }, os.path.join(args.output_dir, "sudoku_tnf.pt"))
    
    # Final summary
    print("\n" + "="*60)
    print("SUDOKU TNF — RESULTS SUMMARY")
    print("="*60)
    print(f"  Architecture:    SudokuTNF ({n_params} learnable params)")
    print(f"  Training:        {args.n_puzzles} puzzles, {args.epochs} epochs")
    print(f"  Clues:           {args.n_clues} / 81")
    print(f"  Cell accuracy:   {metrics['cell_accuracy']:.1f}%")
    print(f"  Full solve rate: {metrics['solve_rate']:.1f}%")
    print(f"  Avg energy:      {metrics['avg_energy']:.4f}")
    print(f"  Step size (η):   {model.step_size.item():.4f}")
    print(f"  Temperature:     {model.temperature.item():.4f}")
    print(f"  Output:          {args.output_dir}/")
    print("="*60)
    
    if metrics['solve_rate'] > 0:
        print("\n✓ TNF successfully solved Sudoku puzzles via energy minimization!")
        print("  This proves: logical constraints CAN be encoded as energy barriers.")
    else:
        print("\n⚠ Solve rate is 0%. The model learned constraint structure but")
        print("  needs more steps/training/tuning to fully solve puzzles.")
        print("  Cell accuracy shows partial constraint satisfaction.")


if __name__ == "__main__":
    main()
