
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, "./src")
from hamiltonian_field import HamiltonianField

def gaussian_blob(size, center, sigma=1.5):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    d2 = (y - center[0])**2 + (x - center[1])**2
    return torch.exp(-d2 / (2 * sigma**2))

def verify_physics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verifying on {device}")
    
    size = 16
    
    # Load Model
    model = HamiltonianField(
        n_channels=2, 
        height=size, 
        width=size, 
        n_filters=8,
        kan_hidden=[16, 1],
        kan_grid_size=5
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("./outputs/kan_physics.pth", map_location=device))
        print("Loaded trained model.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Scenario
    start_pos = (size//2, 3)
    block_pos = (size//2, 8)
    
    agent_blob = gaussian_blob(size, start_pos, sigma=1.5).unsqueeze(0).unsqueeze(0).to(device)
    block_blob = gaussian_blob(size, block_pos, sigma=1.5).unsqueeze(0).unsqueeze(0).to(device)
    
    state = torch.full((1, 2, size, size), -0.7, device=device)
    state[:, 0:1] += agent_blob * 2.0
    state[:, 1:2] += block_blob * 2.0
    
    # Advection: Agent=Right, Block=Static
    advection_field = torch.zeros(1, 2, 2, size, size, device=device)
    advection_field[:, 0, 1, :, :] = 1.0 # Agent vx=1
    
    # Run
    # Use inference evolve (n_steps=40)
    # NOTE: We MUST NOT use torch.no_grad() because evolve uses autograd to compute dynamics!
    print("Simulating...")
    trajectory = model.evolve(
        state, 
        n_steps=40, 
        return_trajectory=True, 
        advection_field=advection_field
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    indices = np.linspace(0, 40, 6).astype(int)
    
    masses = []
    
    for i, idx in enumerate(indices):
        s = trajectory[idx] # (1, 2, H, W)
        
        # Physics Check
        agent = s[0,0]
        block = s[0,1]
        
        mass = block.sum().item()
        masses.append(mass)
        
        # Viz
        rgb = torch.zeros(size, size, 3)
        # Normalize for viz (0..1)
        # Agent Red, Block Blue
        # Clamp negative values for display
        a_disp = (agent + 1.0) / 3.0 
        b_disp = (block + 1.0) / 3.0
        
        rgb[:, :, 0] = a_disp.clamp(0, 1)
        rgb[:, :, 2] = b_disp.clamp(0, 1)
        
        axes[i].imshow(rgb.cpu())
        axes[i].set_title(f"Step {idx}\nBlock Mass: {mass:.1f}")
        
    plt.tight_layout()
    plt.savefig("./outputs/kan_physics_verification.png")
    print("Saved ./outputs/kan_physics_verification.png")
    
    # Plot Mass Conservation
    plt.figure()
    traj_masses = [step[0,1].sum().item() for step in trajectory]
    plt.plot(traj_masses)
    plt.title("Block Mass Conservation (Learned)")
    plt.xlabel("Step")
    plt.ylabel("Integral Mass")
    plt.savefig("./outputs/kan_mass_plot.png")
    print("Saved ./outputs/kan_mass_plot.png")

if __name__ == "__main__":
    verify_physics()
