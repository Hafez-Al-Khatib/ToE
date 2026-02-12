
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, "./src")
from thermodynamic_field import ThermodynamicField
from wave_solver import WaveBrain
from unified_planner import UnifiedPlanner

def create_environment(size=20):
    """
    Create a 20x20 grid with:
    - Walls (Border + some internal)
    - Start pos
    - Goal pos
    - Block pos (between start and goal)
    """
    walls = torch.zeros(1, 1, size, size)
    # Borders
    walls[:, :, 0, :] = 1
    walls[:, :, -1, :] = 1
    walls[:, :, :, 0] = 1
    walls[:, :, :, -1] = 1
    
    # Internal wall to force path
    # walls[:, :, 5:15, 10] = 1 
    
    start = (size//2, 2)
    goal = (size//2, size-3)
    block = (size//2, size//2) # Block in the middle
    
    return walls, start, goal, block

def gaussian_blob(size, center, sigma=1.5):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    d2 = (y - center[0])**2 + (x - center[1])**2
    return torch.exp(-d2 / (2 * sigma**2))

def manual_physics_init(model):
    """
    Initialize field with 'Zero-Shot Physics' parameters.
    """
    # 2 Channels: 0=Agent, 1=Block
    # Strong repulsion between them
    nn.init.constant_(model.coupling_weights, 5.0) 
    
    # Diffusion: Lower diffusion to keep blobs compact
    with torch.no_grad():
        model.log_alphas[0] = np.log(0.05) 
        model.log_alphas[1] = np.log(0.05) 
        
    # Potentials: Asymmetric Double Well (Stable 0 and Stable 1)
    # V(u) = u^2 * (u-1)^2 = u^4 - 2u^3 + u^2
    # Coefficients for (u^4, u^3, u^2, u)
    # We have potential = a*u^2 + b*u^4.
    
    # ---------------------------------------------------------
    # CRITICAL FIX: Spatial Filters
    # Random filters cause "messy interpolations" (noise).
    # We set them to Laplacian (Diffusion) + Identity.
    # ---------------------------------------------------------
    
    # K=5 Laplacian Kernel (Approx)
    laplacian = torch.tensor([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=torch.float32) / 16.0
    
    # Identity Kernel (No Blur)
    # Semi-Lagrangian Advection provides implicit numerical viscosity.
    # We don't need extra Gaussian blur. We need SHARPENING.
    identity = torch.zeros(5, 5)
    identity[2, 2] = 1.0
    
    with torch.no_grad():
        # Set all filters to a mix of Identity and Laplacian
        for c in range(model.n_channels):
            # 1. Spatial Filters
            model.spatial_filters[c].fill_(0.0)
            
            # Filter 0: Identity (Persistence)
            model.spatial_filters[c][0, 0] = identity
            # Filter 1: Laplacian (Diffusion/Smoothing)
            model.spatial_filters[c][1, 0] = laplacian
            
            # 2. Filter Weights
            model.filter_weights[c].fill_(0.0)
            model.filter_weights[c][0] = 1.0 # Identity importance
            model.filter_weights[c][1] = 0.1 # LOW Diffusion (just enough for stability)
            
    # Deep Double Well (-8, 4) for STRONG SHARPENING
    # V' = -16u + 16u^3. Zeros at 0, +/- 1.
    # This forces the "smears" back into solid chunks.
    nn.init.constant_(model.potential_a, -8.0)
    nn.init.constant_(model.potential_b, 4.0)
    # Wait, ThermodynamicField.py potential is:
    # potential = self.potential_a * u**2 + self.potential_b * u**4
    # It doesn't have a u^3 term!
    # Problem. We can only have symmetric potentials with current code?
    # Let's check ThermodynamicField.py.
    
    # If we are limited to even powers, we can't have minima at 0 and 1.
    # We can have minima at 0 and +/- 1?
    # V = u^2 * (u^2 - 1)^2 = u^2 (u^4 - 2u^2 + 1) = u^6 - 2u^4 + u^2.
    # We don't have u^6.
    
    # Alternative: Use standard double well -u^2 + u^4.
    # Minima at +/- 0.7. Max at 0.
    # If we want background to be 0, we must initialize background at -1 (or +1)?
    # No, that's "filling the universe".
    
    # We want "Spot" solutions. 
    # Reaction-Diffusion systems (Gray-Scott, etc) support spots.
    # They require different kinetics.
    
    # HACK: If we can't change the potential form, we fix the visualization to handle signed values.
    # AND we initialize the background to -1 (stable) and Agent to +1 (stable).
    # Then "Pushing" is interaction between +1 and +1 domains in a -1 universe.
    # Or, we can just accept that 0 is unstable and the background will eventually turn on.
    
    # Let's try initialized background to -1?
    # Or just fix the viz to show abs(u).
    
    # Let's stick to the parameters we had but FIX THE VIZ and MAYBE the potential balance.
    # If agent went to -1.5, it flipped.
    # Let's try to keep it positive.
    # Advection might have pushed it negative?
    
    # Let's try a simpler potential: Just attractive 0 for background? 
    # No, then agent dies.
    
    # Okay, sticking with Double Well (-4, 2)
    # But checking viz.
    pass

def custom_viz(trajectory, walls, goal, size):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    indices = np.linspace(0, len(trajectory)-1, 6).astype(int)
    
    for ax, idx in zip(axes, indices):
        s = trajectory[idx][0] # (2, H, W)
        
        # Visualize Magnitude?
        # Agent (Red) - show Abs value
        agent_map = s[0].abs()
        block_map = s[1].abs()
        
        # Viz Clamp: REMOVED. Use raw values to verify True Physics.
        # agent_map[agent_map < 0.5] = 0
        # block_map[block_map < 0.5] = 0
        
        rgb = torch.zeros(size, size, 3)
        rgb[:, :, 0] = agent_map / agent_map.max().clamp(min=1.0) # Normalize locally
        rgb[:, :, 2] = block_map / block_map.max().clamp(min=1.0)
        
        # Walls
        w = walls[0, 0].cpu()
        rgb[:, :, 1] = w * 0.4
        
        rgb = torch.clamp(rgb, 0, 1)
        
        ax.imshow(rgb)
        ax.set_title(f"Step {idx}")
        ax.plot(goal[1], goal[0], 'yx', markersize=10, markeredgewidth=2)
        
    plt.tight_layout()
    plt.savefig("./outputs/unified_planner/push_block_test.png")
    print("Saved with mag visualization.")

def main():
    device = "cpu"
    os.makedirs("./outputs/unified_planner", exist_ok=True)
    
    # 1. Setup
    size = 20
    walls, start, goal, block_pos = create_environment(size)
    walls = walls.to(device)
    
    # 2. State Init
    # Initialize background to -1.0 (Stable Minima of V = -4u^2 + 2u^4)
    # Agent/Block to +1.0 (Other Stable Minima)
    state = torch.full((1, 2, size, size), -0.7, device=device) # Start near mins
    
    agent_blob = gaussian_blob(size, start).unsqueeze(0).unsqueeze(0)
    block_blob = gaussian_blob(size, block_pos).unsqueeze(0).unsqueeze(0)
    
    # Add blobs: push specific areas to +1 (or closer to +1)
    # If background is -0.7, adding 1.7 makes it +1.
    # Let's add MORE to be safe. +3.0 => +2.3.
    state[:, 0:1] = state[:, 0:1] + agent_blob * 4.0
    state[:, 1:2] = state[:, 1:2] + block_blob * 4.0
    
    # 3. Models
    # Dynamics: Agent (0) = Non-Conserved (Signal), Block (1) = Conserved (Matter)
    tnf = ThermodynamicField(
        n_channels=2, 
        height=size, 
        width=size, 
        n_filters=8
    ).to(device)
    
    # Physics Params
    # Repulsion
    nn.init.constant_(tnf.coupling_weights, 5.0) 
    
    # Diffusion: Keep moderate (0.1) for stability
    with torch.no_grad():
        tnf.log_alphas[0] = np.log(0.1) 
        tnf.log_alphas[1] = np.log(0.1) 
        
    # Potential: Relaxed Double Well (-4, 2). Confining but allows movement.
    nn.init.constant_(tnf.potential_a, -4.0)
    nn.init.constant_(tnf.potential_b, 2.0)
    
    # Step size: 0.04 for speed (CFL limit approx 0.05)
    nn.init.constant_(tnf.log_step_size, np.log(0.04))
    
    wave = WaveBrain(grid_size=(size, size), use_field_encoder=False).to(device)
    # Drift Weight: STRONG Advection (10.0) to move the solitons
    planner = UnifiedPlanner(tnf, wave, drift_weight=10.0)
    
    # 4. Run
    print("Running...")
    trajectory = []
    
    def map_func(s): return walls # Fixed map for now
    
    # 1. Setup
    size = 16 # Reduce size for faster/stable run
    walls, start, goal, block_pos = create_environment(size)
    walls = walls.to(device)
    
    # Re-initialize state for new size
    state = torch.full((1, 2, size, size), -0.7, device=device) 
    agent_blob = gaussian_blob(size, start).unsqueeze(0).unsqueeze(0)
    block_blob = gaussian_blob(size, block_pos).unsqueeze(0).unsqueeze(0)
    state[:, 0:1] = state[:, 0:1] + agent_blob * 4.0
    state[:, 1:2] = state[:, 1:2] + block_blob * 4.0

    # Physics Params (Balanced for Mobility & Stability)
    # Potential: -4, 2 (Standard Double Well)
    nn.init.constant_(tnf.potential_a, -4.0)
    nn.init.constant_(tnf.potential_b, 2.0)
    # Step size: 0.005 for Cahn-Hilliard Stability (4th order need small steps)
    nn.init.constant_(tnf.log_step_size, np.log(0.005))
    
    # Drift: 1.0 (1 pixel per step approx)
    # Semi-Lagrangian is stable, but we shouldn't jump too far.
    planner = UnifiedPlanner(tnf, wave, drift_weight=1.0)

    # Run
    path_gen = planner.solve(state, map_func, start, goal, max_macro_steps=40)
    
    for i, step_state in enumerate(path_gen):
        trajectory.append(step_state.detach().cpu())
        
        # Check Agent/Block Max Value Location
        agent_max_val, agent_max_idx = step_state[0, 0].view(-1).max(0)
        block_max_val, block_max_idx = step_state[0, 1].view(-1).max(0)
        
        ay, ax = agent_max_idx.item() // size, agent_max_idx.item() % size
        by, bx = block_max_idx.item() // size, block_max_idx.item() % size
        
        # Mass Check: Sum of Block Channel
        # Subtract background (-0.7) to measure "excess mass"?
        # Or just sum u.
        mass = step_state[0, 1].sum().item()
        agent_mass = step_state[0, 0].sum().item()

        print(f"Step {i}: Agent@({ay},{ax})={agent_max_val:.2f} (M={agent_mass:.1f}), Block@({by},{bx})={block_max_val:.2f} (M={mass:.1f})")

    # 5. Viz
    custom_viz(trajectory, walls, goal, size)

if __name__ == "__main__":
    main()
