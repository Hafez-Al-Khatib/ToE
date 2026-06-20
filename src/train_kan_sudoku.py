
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.insert(0, "./src")
from predictive_coding_field import PredictiveCodingField

# --- Sudoku Generator ---
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

        # Produce tensor (9, 9)
        board = [ [nums[self.pattern(r,c)] for c in cols] for r in rows ]
        return torch.tensor(board, dtype=torch.long) - 1 # 0-8 range

    def mask_board(self, board, n_holes=40):
        # Create puzzle by masking
        mask = torch.zeros_like(board, dtype=torch.bool)
        flat_indices = torch.randperm(81)[:n_holes]
        mask.view(-1)[flat_indices] = True
        
        puzzle = board.clone()
        puzzle[mask] = -1 # -1 for empty
        return puzzle, mask

class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, length=1000):
        self.length = length
        self.gen = SudokuGenerator()
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        solution_indices = self.gen.generate_board() # (9, 9) values 0-8
        puzzle_indices, mask = self.gen.mask_board(solution_indices)
        
        # Convert to One-Hot (9, 9, 9) -> Permute to (9, 9, 9) H,W,C? No (C, H, W)
        # Tensor shape: (9, 9, 9)
        
        # Target: Indices (9, 9) Long
        target = torch.tensor(solution_indices, dtype=torch.long)
        
        # Input: Puzzle
        # Empty cells (-1) -> Zero vector (all 0.1/9)
        # Filled cells -> One-hot
        input_tensor = torch.zeros(9, 9, 9)
        filled_mask = (puzzle_indices != -1)
        
        rows, cols = torch.where(filled_mask)
        vals = puzzle_indices[rows, cols]
        input_tensor[vals, rows, cols] = 1.0 # Sharp input
        
        # Empty cells are ambiguous
        empty_mask = ~filled_mask
        input_tensor[:, empty_mask] = 1.0/9.0
        
        return input_tensor, target

def train_kan_sudoku():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Sudoku on {device}")
    
    # 1. Data
    trainset = SudokuDataset(length=2000)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    # 2. Model
    model = PredictiveCodingField(
        n_channels=9,
        height=9,
        width=9,
        n_filters=9, 
        filter_size=17, # Global Vis (Reverted)
        kan_hidden=[128, 1], 
        kan_grid_size=5
    ).to(device)
    
    # PHILOSOPHY ALIGNMENT: 
    # Do not hardcode "Sombrero". allow KAN to learn it.
    # But remove the "Gravity Well" bias that pulls to 0.
    # Initialize polynomial potential to FLAT (0.0) so KAN drives dynamics.
    # PHILOSOPHY ALIGNMENT: 
    # Do not hardcode "Sombrero". allow KAN to learn it.
    # The KAN is initialized with small random weights (unbiased) by default.
    # No need to manually zero polynomials as they are gone.
    
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR for stability
    
    # 3. Training
    epochs = 50 
    print("Starting Sudoku Training (Unbiased Physics)...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Initialize:
            state = inputs.clone().requires_grad_(True)
            
            # Anchor Mask: Identifies Clues (Fixed) vs Holes (Free)
            # Clues have value 1.0 in some channel. Holes have 1.0/9.0 (~0.11).
            # (B, 9, 9, 9) -> max over channels -> (B, 9, 9)
            # Threshold at 0.5
            mask = (inputs.max(dim=1, keepdim=True)[0] > 0.5).float() # (B, 1, 9, 9)
            
            # Evolve
            n_steps = 5 
            final_state = model.trainable_evolve(
                state,
                n_steps=n_steps,
                anchor=inputs.detach(),
                anchor_mask=mask
            )
            
            # Loss: CrossEntropy over 9 channels
            # final_state: (B, 9, 9, 9) -> (B, 9, 9, 9) -> CrossEntropy expects (B, C, H, W)
            # targets: (B, 9, 9) indices
            loss = F.cross_entropy(final_state, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {running_loss / len(trainloader):.4f}")
        # Save every epoch
        torch.save(model.state_dict(), "./outputs/kan_sudoku.pth")

    # Save
    torch.save(model.state_dict(), "./outputs/kan_sudoku.pth")
    print("Training Complete. Verifying...")
    
    # Verify
    model.eval()
    sample_in, sample_tgt = next(iter(trainloader))
    sample_in = sample_in[0:1].to(device) # (1, 9, 9, 9)
    sample_tgt = sample_tgt[0:1].to(device)
    
    # Evolve
    # Multi-step
    final_state = model.evolve(sample_in, n_steps=100)
    
    # Discretize (Argmax)
    pred_digits = final_state.argmax(dim=1) # (1, 9, 9)
    true_digits = sample_tgt.argmax(dim=1)
    
    acc = (pred_digits == true_digits).float().mean()
    print(f"Validation Accuracy (Pixel-wise): {acc.item():.2%}")
    
    # Viz
    # ... (omitted for brevity, just numbers for now)

if __name__ == "__main__":
    train_kan_sudoku()
