"""
Phase 3 Training: Wave Brain for Maze Solving
==============================================

This script trains the Wave Brain model on maze pathfinding tasks.

The Training Process
--------------------
1. Generate random mazes with known optimal paths
2. Encode maze → refractive index field
3. Solve Eikonal equation → travel time field
4. Extract path and compare to ground truth
5. Backpropagate through the entire pipeline

Key Insight: We don't search for paths. We learn to predict
the "refractive index" field such that wave propagation finds
the optimal path instantly.

Usage
-----
Basic training:
    python src/train_phase3.py

With options:
    python src/train_phase3.py --epochs 50 --maze-size 32 --visualize
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from wave_solver import (
    WaveBrain, 
    EikonalSolver, 
    MazeEncoder,
    extract_path,
    generate_random_maze,
    visualize_solution
)


def generate_maze_with_path(
    size: int = 32,
    wall_density: float = 0.25,
    ensure_path: bool = True,
    device: str = "cpu"
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], Optional[List]]:
    """
    Generate a maze with guaranteed path from source to target.
    
    Algorithm:
    1. Generate random walls
    2. Place source and target at opposite corners
    3. Use BFS to ensure path exists
    4. If no path, carve one by removing walls
    
    Parameters
    ----------
    size : int
        Maze size.
    
    wall_density : float
        Fraction of walls.
    
    ensure_path : bool
        If True, guarantee a path exists.
    
    device : str
        Device for tensors.
    
    Returns
    -------
    maze : torch.Tensor
        Maze tensor (1, 1, H, W). 0=wall, 1=passage.
    
    source : tuple
        Source position (y, x).
    
    target : tuple
        Target position (y, x).
    
    path : list or None
        Ground truth path if computed.
    """
    maze_np = np.random.rand(size, size) > wall_density
    maze_np = maze_np.astype(np.float32)
    
    # Fixed source and target
    source = (1, 1)
    target = (size - 2, size - 2)
    
    # Ensure source and target are passages
    maze_np[source[0], source[1]] = 1
    maze_np[target[0], target[1]] = 1
    
    if ensure_path:
        # BFS to check connectivity
        path = bfs_path(maze_np, source, target)
        
        if path is None:
            # Carve a path
            path = carve_path(maze_np, source, target)
    else:
        path = None
    
    # Convert to tensor
    maze = torch.from_numpy(maze_np).to(device)
    maze = maze.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return maze, source, target, path


def bfs_path(
    maze: np.ndarray,
    source: Tuple[int, int],
    target: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """BFS to find shortest path in maze."""
    from collections import deque
    
    H, W = maze.shape
    visited = np.zeros((H, W), dtype=bool)
    parent = {}
    
    queue = deque([source])
    visited[source] = True
    
    while queue:
        y, x = queue.popleft()
        
        if (y, x) == target:
            # Reconstruct path
            path = [(y, x)]
            while (y, x) != source:
                y, x = parent[(y, x)]
                path.append((y, x))
            return path[::-1]
        
        # Check neighbors
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if not visited[ny, nx] and maze[ny, nx] > 0.5:
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (y, x)
                    queue.append((ny, nx))
    
    return None


def carve_path(
    maze: np.ndarray,
    source: Tuple[int, int],
    target: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Carve a straight(ish) path from source to target."""
    y, x = source
    ty, tx = target
    path = [(y, x)]
    
    while (y, x) != (ty, tx):
        # Move toward target
        if y < ty and maze[y + 1, x] >= 0:  # Can potentially carve
            y += 1
        elif x < tx and maze[y, x + 1] >= 0:
            x += 1
        elif y > ty and maze[y - 1, x] >= 0:
            y -= 1
        elif x > tx and maze[y, x - 1] >= 0:
            x -= 1
        else:
            # Stuck, force a move
            if abs(ty - y) >= abs(tx - x):
                y = y + 1 if y < ty else y - 1
            else:
                x = x + 1 if x < tx else x - 1
        
        # Ensure within bounds
        y = max(1, min(y, maze.shape[0] - 2))
        x = max(1, min(x, maze.shape[1] - 2))
        
        # Carve passage
        maze[y, x] = 1.0
        path.append((y, x))
        
        # Safety limit
        if len(path) > 2 * (maze.shape[0] + maze.shape[1]):
            break
    
    return path


class MazeDataset(Dataset):
    """
    Dataset of random mazes with ground truth paths.
    
    Parameters
    ----------
    n_mazes : int
        Number of mazes to generate.
    
    size : int
        Maze size.
    
    wall_density : float
        Fraction of walls.
    """
    
    def __init__(
        self,
        n_mazes: int = 1000,
        size: int = 32,
        wall_density: float = 0.25
    ):
        self.n_mazes = n_mazes
        self.size = size
        self.wall_density = wall_density
        
        # Pre-generate mazes
        self.mazes = []
        self.sources = []
        self.targets = []
        self.paths = []
        
        print("Generating mazes...")
        for _ in tqdm(range(n_mazes)):
            maze, source, target, path = generate_maze_with_path(
                size=size,
                wall_density=wall_density,
                ensure_path=True
            )
            self.mazes.append(maze)
            self.sources.append(source)
            self.targets.append(target)
            self.paths.append(path)
    
    def __len__(self):
        return self.n_mazes
    
    def __getitem__(self, idx):
        return {
            "maze": self.mazes[idx].squeeze(0),  # (1, H, W)
            "source": self.sources[idx],
            "target": self.targets[idx],
            "path": self.paths[idx]
        }


def collate_fn(batch):
    """Custom collate for maze data."""
    mazes = torch.stack([b["maze"] for b in batch])
    sources = [b["source"] for b in batch]
    targets = [b["target"] for b in batch]
    paths = [b["path"] for b in batch]
    
    return {
        "maze": mazes,
        "sources": sources,
        "targets": targets,
        "paths": paths
    }


def path_distance_loss(
    u: torch.Tensor,
    maze: torch.Tensor,
    ground_truth_path: List[Tuple[int, int]],
    source: Tuple[int, int],
    target: Tuple[int, int]
) -> torch.Tensor:
    """
    Loss that encourages the travel time field to match optimal path.
    
    Three components:
    1. Target reachability: u(target) should be reasonable
    2. Wall penalty: walls should have high travel time
    3. Path optimality: travel time along ground truth path should be low
    """
    # Loss 1: Target should be reachable
    target_loss = u[0, target[0], target[1]]
    
    # Loss 2: Wall penalty — walls should have high travel time
    wall_mask = maze[0, 0] < 0.5  # 0 = wall
    wall_u = u[0, wall_mask]
    if len(wall_u) > 0:
        wall_loss = F.relu(10.0 - wall_u.mean())
    else:
        wall_loss = torch.tensor(0.0, device=u.device)
    
    # Loss 3: Path should have low cumulative cost (DIFFERENTIABLE)
    # Ground truth path cells should have low travel time
    if ground_truth_path is not None and len(ground_truth_path) > 0:
        # Index travel time directly (keeps gradients flowing)
        path_times = torch.stack([
            u[0, p[0], p[1]] for p in ground_truth_path
        ])
        path_loss = path_times.mean()
    else:
        path_loss = torch.tensor(0.0, device=u.device)
    
    # ALL three terms contribute — path_loss was previously discarded!
    return target_loss + 0.1 * wall_loss + 0.5 * path_loss


def train_epoch(
    model: WaveBrain,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        mazes = batch["maze"].to(device)
        sources = batch["sources"]
        targets = batch["targets"]
        paths = batch["paths"]
        
        # For simplicity, use first maze's source/target for the batch
        source = sources[0]
        target = targets[0]
        
        optimizer.zero_grad()
        
        # Forward pass
        n, u, _ = model(mazes, source, target_pos=None)
        
        # Compute loss
        batch_loss = 0.0
        for i in range(len(mazes)):
            loss_i = path_distance_loss(
                u[i:i+1], mazes[i:i+1], paths[i], source, target
            )
            batch_loss = batch_loss + loss_i
        batch_loss = batch_loss / len(mazes)
        
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        n_batches += 1
        
        pbar.set_postfix({"loss": f"{batch_loss.item():.3f}"})
    
    return {"loss": total_loss / n_batches}


@torch.no_grad()
def evaluate(
    model: WaveBrain,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str
) -> dict:
    """Evaluate and visualize results."""
    model.eval()
    
    total_loss = 0.0
    path_found_count = 0
    n_samples = 0
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        mazes = batch["maze"].to(device)
        sources = batch["sources"]
        targets = batch["targets"]
        paths = batch["paths"]
        
        source = sources[0]
        target = targets[0]
        
        # Forward pass with path extraction
        n, u, extracted_path = model(mazes[:1], source, target_pos=target)
        
        # Check if path was found
        if extracted_path and len(extracted_path) > 0:
            path_found_count += 1
        
        # Compute loss
        loss = path_distance_loss(u, mazes[:1], paths[0], source, target)
        total_loss += loss.item()
        n_samples += 1
        
        # Visualize first few
        if n_samples <= 3:
            visualize_solution(
                mazes[:1], n, u, extracted_path, source, target,
                save_path=os.path.join(output_dir, f"solution_{n_samples}.png")
            )
    
    return {
        "loss": total_loss / n_samples,
        "path_found_rate": path_found_count / n_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Train Wave Brain on mazes")
    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maze-size", type=int, default=32)
    parser.add_argument("--wall-density", type=float, default=0.25)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-sweeps", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./outputs/wave_brain")
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_dataset = MazeDataset(
        n_mazes=args.n_train,
        size=args.maze_size,
        wall_density=args.wall_density
    )
    test_dataset = MazeDataset(
        n_mazes=args.n_test,
        size=args.maze_size,
        wall_density=args.wall_density
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, collate_fn=collate_fn
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = WaveBrain(
        grid_size=(args.maze_size, args.maze_size),
        n_sweeps=args.n_sweeps,
        use_field_encoder=True,
        n_evolve_steps=10,
        n_field_channels=4,
        n_field_filters=8,
        field_filter_size=5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print("\n=== Training ===")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}")
        
        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            eval_metrics = evaluate(model, test_loader, device, args.output_dir)
            print(f"  Eval: loss={eval_metrics['loss']:.4f}, "
                  f"path_found={eval_metrics['path_found_rate']:.1%}")
            
            # Save best model
            if eval_metrics['loss'] < best_loss:
                best_loss = eval_metrics['loss']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss
                }, os.path.join(args.output_dir, "best_model.pt"))
                print(f"  ✓ Saved best model")
        
        scheduler.step()
    
    print("\n=== Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
