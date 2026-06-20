
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.insert(0, "./src")
from predictive_coding_field import PredictiveCodingField

# --- Sudoku Generator (Copy-Paste for simplicity) ---
class SudokuGenerator:
    def __init__(self):
        self.base = 3
        self.side = self.base * self.base

    def pattern(self, r, c): 
        return (self.base * (r % self.base) + r // self.base + c) % self.side

    def shuffle(self, s): 
        return random.sample(s, len(s))

    def generate_board(self):
        rBase = range(self.base)
        rows  = [g * self.base + r for g in self.shuffle(rBase) for r in self.shuffle(rBase)]
        cols  = [g * self.base + c for g in self.shuffle(rBase) for c in self.shuffle(rBase)]
        nums  = self.shuffle(range(1, self.base * self.base + 1))
        board = [ [nums[self.pattern(r,c)] for c in cols] for r in rows ]
        return torch.tensor(board, dtype=torch.long) - 1

    def mask_board(self, board, n_holes=40):
        mask = torch.zeros_like(board, dtype=torch.bool)
        flat_indices = torch.randperm(81)[:n_holes]
        mask.view(-1)[flat_indices] = True
        puzzle = board.clone()
        puzzle[mask] = -1 
        return puzzle, mask

def visualize_sudoku():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Visualizing on {device}")
    
    # Load Model
    model = PredictiveCodingField(
        n_channels=9,
        height=9,
        width=9,
        n_filters=9, 
        filter_size=17, 
        kan_hidden=[128, 1], 
        kan_grid_size=5
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("./outputs/kan_sudoku.pth", map_location=device))
        print("Loaded Sudoku Model.")
    except:
        print("Model not found.")
        return

    # Generate Sample
    gen = SudokuGenerator()
    solution = gen.generate_board()
    puzzle, mask = gen.mask_board(solution)
    
    # Prepare Input
    input_tensor = torch.zeros(1, 9, 9, 9).to(device)
    filled_mask = (puzzle != -1)
    rows, cols = torch.where(filled_mask)
    vals = puzzle[rows, cols]
    input_tensor[0, vals, rows, cols] = 1.0
    input_tensor[0, :, ~filled_mask] = 1.0/9.0
    
    # Evolve with trajectory
    model.eval()
    print("Solving...")
    
    # Create mask for inference
    # input_tensor is (1, 9, 9, 9)
    # channels max > 0.5 means clue
    mask = (input_tensor.max(dim=1, keepdim=True)[0] > 0.5).float()
    
    trajectory = model.evolve(
        input_tensor, 
        n_steps=100, 
        return_trajectory=True,
        anchor_mask=mask
    ) 
    
    true_digits = solution
    puzzle_disp = puzzle.clone()
    
    # Viz: Show Steps 0, 10, 50, 99
    steps_to_show = [0, 10, 50, 99]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    def plot_board(ax, board, title):
        ax.imshow(board, cmap='tab20c', vmin=0, vmax=9)
        # Add text
        for r in range(9):
            for c in range(9):
                val = board[r, c].item()
                if val != -1:
                    ax.text(c, r, str(val+1), ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')

    plot_board(axes[0], puzzle_disp, "Input")
    
    for i, step in enumerate(steps_to_show):
        state = trajectory[step, 0] # (9, 9, 9)
        pred = state.argmax(dim=0).cpu() # (9, 9)
        
        # Calculate accuracy at this step
        acc = (pred == true_digits).float().mean()
        plot_board(axes[i+1], pred, f"Step {step}\nAcc: {acc:.1%}")
    
    plt.tight_layout()
    plt.savefig("./outputs/kan_sudoku_result.png")
    print("Saved ./outputs/kan_sudoku_result.png")
    
    # Batch Evaluation
    print("\nRunning Batch Evaluation (N=100)...")
    total_boards = 100
    solved_boards = 0
    total_acc = 0.0
    
    # Create batch of 100
    # Manually loops to avoid OOM
    for i in range(total_boards):
        solution = gen.generate_board() # (9, 9)
        puzzle, mask_idx = gen.mask_board(solution)
        
        # Input
        input_tensor = torch.zeros(1, 9, 9, 9).to(device)
        filled_mask = (puzzle != -1)
        rows, cols = torch.where(filled_mask)
        vals = puzzle[rows, cols]
        input_tensor[0, vals, rows, cols] = 1.0
        input_tensor[0, :, ~filled_mask] = 1.0/9.0
        
        
        # Simulated Annealing Loop
        anchor_mask = (input_tensor.max(dim=1, keepdim=True)[0] > 0.5).float()
        u = input_tensor.clone()
        n_steps = 100
        noise_scale = 0.5 # Temperature
        
        for t in range(n_steps):
            # Anneal Temperature
            T = max(0, (1.0 - t/float(n_steps))) * noise_scale
            
            # Injection
            if T > 0:
                noise = torch.randn_like(u) * T
                # Apply noise only to free cells (masked by anchor)
                # mask is 1 for clues. 1-mask is free.
                u = u + noise * (1.0 - anchor_mask)
            
            # Evolve 1 step
            u = model.evolve(
                u, 
                n_steps=1,
                anchor_mask=anchor_mask
            )
            
        final_state = u
        pred = final_state.argmax(dim=1)[0].cpu()
        
        # Check
        matches = (pred == solution).float().mean()
        total_acc += matches.item()
        if matches.item() == 1.0:
            solved_boards += 1
            
    print(f"Batch Metrics:")
    print(f"  Pixel Accuracy: {total_acc / total_boards:.2%}")
    print(f"  Perfect Boards: {solved_boards}/{total_boards} ({solved_boards/total_boards:.1%})")

if __name__ == "__main__":
    visualize_sudoku()
