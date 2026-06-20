import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "./src")
from wave_solver import WaveBrain, extract_path
from train_phase3 import generate_maze_with_path, path_distance_loss


def is_valid_path(path, maze):
    """
    Strict collision check (PLANNING_AUDIT.md fix D).

    A path is valid iff every sub-pixel sample (spacing 0.5 px) along each
    segment lies in a passage cell. Diagonal corner-cuts between two adjacent
    walls are rejected by also requiring at least one axis-aligned neighbor at
    each interior sample to be a passage.
    """
    if path is None or len(path) == 0:
        return False
    if maze.dim() == 4:
        m = maze[0, 0]
    elif maze.dim() == 3:
        m = maze[0]
    else:
        m = maze
    H, W = m.shape

    def cell_ok(y, x):
        yi = int(round(float(y)))
        xi = int(round(float(x)))
        if not (0 <= yi < H and 0 <= xi < W):
            return False
        return float(m[yi, xi]) >= 0.5

    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        dy = float(y1) - float(y0)
        dx = float(x1) - float(x0)
        seg_len = max(abs(dy), abs(dx), 1e-6)
        n_samples = int(math.ceil(seg_len / 0.5)) + 1
        for k in range(n_samples + 1):
            t = k / max(n_samples, 1)
            y = float(y0) + t * dy
            x = float(x0) + t * dx
            if not cell_ok(y, x):
                return False
            yi = int(round(y)); xi = int(round(x))
            if 0 < t < 1 and abs(dy) > 0 and abs(dx) > 0:
                if not (cell_ok(yi, xi - int(np.sign(dx))) or
                        cell_ok(yi - int(np.sign(dy)), xi)):
                    return False
    return True

class CNNPlanner(nn.Module):
    """
    Baseline CNN-based planner predicting a non-negative travel-time field.

    Audit fix (PLANNING_AUDIT.md A): the original linear output head let the
    network minimize loss by emitting unbounded negative values (loss reached
    -2.24e12). All downstream losses are defined for u(x) >= 0, so the head
    is now Softplus-bounded.
    """
    def __init__(self):
        super().__init__()
        # Input: 2 channels (Maze geometry, Source position one-hot)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, maze, source_pos):
        B, C, H, W = maze.shape
        source_map = torch.zeros(B, 1, H, W, device=maze.device)
        for i in range(B):
            source_map[i, 0, source_pos[0], source_pos[1]] = 1.0

        x = torch.cat([maze, source_map], dim=1)
        raw = self.net(x).squeeze(1)
        return F.softplus(raw)

def create_training_batch(size, batch_size, wall_density, device):
    mazes, sources, targets, paths = [], [], [], []
    for _ in range(batch_size):
        m, s, t, p = generate_maze_with_path(size, wall_density, True, device)
        mazes.append(m.squeeze(0))
        sources.append(s)
        targets.append(t)
        paths.append(p)
    return torch.stack(mazes), sources, targets, paths

def train_models(
    device,
    train_size=16,
    epochs=50,
    batches_per_epoch=20,
    batch_size=16,
    jumpstart_steps: int = 300,
    n_wall_target: float = 5.0,
    n_passage_target: float = 1.0,
):
    print(f"--- Training Models on {train_size}x{train_size} Mazes ---")

    wave_brain = WaveBrain(
        grid_size=(train_size, train_size),
        n_sweeps=8,
        use_field_encoder=True,
        n_field_channels=4,
        n_field_filters=8,
        field_filter_size=5,
    ).to(device)

    cnn_planner = CNNPlanner().to(device)

    # PLANNING_AUDIT fix C: jumpstart the WaveBrain encoder so n(x) is wall-vs-
    # passage-discriminative *before* the Eikonal loss runs.
    if jumpstart_steps > 0:
        print(f"\n=== Encoder Jumpstart ({jumpstart_steps} steps) ===")
        opt_pre = torch.optim.Adam(wave_brain.encoder.parameters(), lr=3e-4)
        for step in range(jumpstart_steps):
            mazes, _, _, _ = create_training_batch(
                train_size, batch_size, 0.25, device
            )
            n = wave_brain.encoder(mazes)
            n_target = torch.where(
                mazes[:, 0] < 0.5,
                torch.full_like(n, n_wall_target),
                torch.full_like(n, n_passage_target),
            )
            loss = F.mse_loss(n, n_target)
            opt_pre.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wave_brain.encoder.parameters(), 1.0)
            opt_pre.step()
            if (step + 1) % 50 == 0:
                print(f"  pretrain step {step+1}/{jumpstart_steps}  mse={loss.item():.4f}")

    opt_wave = torch.optim.Adam(wave_brain.parameters(), lr=1e-3)
    opt_cnn = torch.optim.Adam(cnn_planner.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        loss_w_total = 0
        loss_c_total = 0
        
        for _ in range(batches_per_epoch):
            mazes, sources, targets, paths = create_training_batch(train_size, batch_size, 0.25, device)
            
            # --- Train WaveBrain ---
            opt_wave.zero_grad()
            source = sources[0] # Simplified: Assuming batch uses same source for the tensor operation
            target = targets[0]
            
            # WaveBrain forward
            n_out, u_out, _ = wave_brain(mazes, source)
            
            # We average loss over the batch items
            loss_w = 0
            for i in range(batch_size):
                loss_w += path_distance_loss(n_out[i:i+1], u_out[i:i+1], mazes[i:i+1], paths[i], sources[i], targets[i])
            loss_w /= batch_size
            
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(wave_brain.parameters(), 1.0)
            opt_wave.step()
            loss_w_total += loss_w.item()
            
            # --- Train CNN Baseline ---
            opt_cnn.zero_grad()
            
            # CNN forward
            pred_u = cnn_planner(mazes, source)
            
            # CNN Loss: We train it to mimic the "Travel Time" field u directly
            # by penalizing high values on the ground truth path and low values on walls.
            loss_c = 0
            for i in range(batch_size):
                path = paths[i]
                maze = mazes[i:i+1]
                
                # Target reachability
                tgt_loss = pred_u[i, targets[i][0], targets[i][1]]
                
                # Wall penalty (CNN should predict high travel time on walls)
                wall_mask = maze[0, 0] < 0.5
                wall_u = pred_u[i][wall_mask]
                wall_loss = F.relu(10.0 - wall_u.mean()) if len(wall_u) > 0 else torch.tensor(0.0, device=device)
                
                # Path optimality (CNN should predict low travel time on path)
                if path is not None and len(path) > 0:
                    path_times = torch.stack([pred_u[i, p[0], p[1]] for p in path])
                    path_loss = path_times.mean()
                else:
                    path_loss = torch.tensor(0.0, device=device)
                
                loss_c += tgt_loss + 0.1 * wall_loss + 0.5 * path_loss
                
            loss_c /= batch_size
            loss_c.backward()
            opt_cnn.step()
            loss_c_total += loss_c.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Wave Loss: {loss_w_total/batches_per_epoch:.3f} | CNN Loss: {loss_c_total/batches_per_epoch:.3f}")
            
    return wave_brain, cnn_planner

def evaluate_zero_shot(wave_brain, cnn_planner, test_sizes, n_test=50, device="cpu"):
    """
    Zero-shot evaluation across maze sizes (PLANNING_AUDIT.md fix D).

    Per size, reports:
      success_rate            — strict is_valid_path() pass rate
      length_ratio_p50/p90    — extracted_len / Dijkstra_len, percentiles over
                                successful runs (P3 metric in REGISTERED_CLAIMS)
    """
    print("\n--- Zero-Shot Generalization Benchmark ---")
    results = {"sizes": list(test_sizes), "wave": [], "cnn": []}

    for size in test_sizes:
        print(f"Testing Maze Size: {size}x{size}")

        wave_brain.grid_size = (size, size)
        wave_brain.encoder.field.height = size
        wave_brain.encoder.field.width = size
        wave_brain.solver.grid_size = (size, size)

        wave_succ = 0
        cnn_succ = 0
        wave_ratios = []
        cnn_ratios = []

        for _ in tqdm(range(n_test)):
            mazes, sources, targets, paths = create_training_batch(
                size, 1, 0.25, device
            )
            maze, source, target = mazes[0:1], sources[0], targets[0]
            gt_len = len(paths[0]) if paths[0] is not None else None

            with torch.no_grad():
                _, _, ext_w = wave_brain(maze, source, target_pos=target)
            if is_valid_path(ext_w, maze):
                wave_succ += 1
                if gt_len:
                    wave_ratios.append(len(ext_w) / gt_len)

            with torch.no_grad():
                pred_u = cnn_planner(maze, source)
                ext_c = extract_path(pred_u, source, target)
            if is_valid_path(ext_c, maze):
                cnn_succ += 1
                if gt_len:
                    cnn_ratios.append(len(ext_c) / gt_len)

        rate_w = wave_succ / n_test
        rate_c = cnn_succ / n_test

        def _pct(arr, q):
            return float(np.percentile(arr, q)) if arr else float("nan")

        rec_w = {
            "success_rate": rate_w,
            "length_ratio_p50": _pct(wave_ratios, 50),
            "length_ratio_p90": _pct(wave_ratios, 90),
        }
        rec_c = {
            "success_rate": rate_c,
            "length_ratio_p50": _pct(cnn_ratios, 50),
            "length_ratio_p90": _pct(cnn_ratios, 90),
        }
        results["wave"].append(rec_w)
        results["cnn"].append(rec_c)

        print(
            f"  WaveBrain: succ={rate_w*100:.1f}%  L/L*~{rec_w['length_ratio_p50']:.2f} "
            f"| CNN: succ={rate_c*100:.1f}%  L/L*~{rec_c['length_ratio_p50']:.2f}"
        )

    return results

def main():
    import json
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("./results/planning_benchmarks", exist_ok=True)

    wave_brain, cnn_planner = train_models(device, train_size=16, epochs=50)

    sizes = [16, 24, 32, 48, 64]
    results = evaluate_zero_shot(
        wave_brain, cnn_planner, sizes, n_test=200, device=device
    )

    log_path = (
        f"./results/planning_benchmarks/"
        f"{int(time.time())}_zero_shot_scaling.json"
    )
    with open(log_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved structured log: {log_path}")

    rates_w = [r["success_rate"] for r in results["wave"]]
    rates_c = [r["success_rate"] for r in results["cnn"]]

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, rates_w, "b-o", linewidth=2, markersize=8,
             label="WaveBrain (Eikonal Physics)")
    plt.plot(sizes, rates_c, "r--s", linewidth=2, markersize=8,
             label="CNN Baseline (Fixed Receptive Field)")
    plt.title("Zero-Shot Generalization Across Maze Sizes (Verified)",
              fontsize=14)
    plt.xlabel("Maze Size (N x N)", fontsize=12)
    plt.ylabel("Success Rate (strict no-corner-cutting)", fontsize=12)
    plt.xticks(sizes)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("./results/planning_benchmarks/generalization_scaling.png",
                dpi=150)
    print("Saved plot: ./results/planning_benchmarks/generalization_scaling.png")


if __name__ == "__main__":
    main()
