
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, "./src")
from hamiltonian_field import HamiltonianField
from wave_solver import WaveBrain
from unified_planner import UnifiedPlanner

def gaussian_blob(size, center, sigma=1.5):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    d2 = (y - center[0])**2 + (x - center[1])**2
    return torch.exp(-d2 / (2 * sigma**2))

def create_training_batch(batch_size=4, size=20, device='cpu'):
    """
    Create a batch of "Push" scenarios.
    Agent at left, Block in middle. Target is Block at right.
    """
    # Fixed Locations for simplicity
    start_pos = (size//2, 3)
    block_pos = (size//2, 8)
    target_pos = (size//2, 14) # Goal for Block
    
    # Create blobs
    agent_blob = gaussian_blob(size, start_pos, sigma=1.5).unsqueeze(0).unsqueeze(0).to(device)
    block_blob = gaussian_blob(size, block_pos, sigma=1.5).unsqueeze(0).unsqueeze(0).to(device)
    target_blob = gaussian_blob(size, target_pos, sigma=1.5).unsqueeze(0).unsqueeze(0).to(device)
    
    # State: 2 channels (Agent, Block)
    # Background -0.7 (approx), Blobs +1.0
    state = torch.full((batch_size, 2, size, size), -0.7, device=device)
    state[:, 0:1] += agent_blob * 2.0 # ~1.3 max
    state[:, 1:2] += block_blob * 2.0
    
    # Target: Block channel should look like target_blob
    target_block = torch.full((batch_size, 1, size, size), -0.7, device=device)
    target_block += target_blob * 2.0
    
    return state, target_block

def train_kan_physics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    size = 16 # Smaller for faster training
    
    # 1. Model: Hamiltonian KAN
    # 2 Channels (Agent, Block) -> Energy
    # Filters=8 -> KAN Input=8
    model = HamiltonianField(
        n_channels=2, 
        height=size, 
        width=size, 
        n_filters=8,
        kan_hidden=[16, 1], # 8->16->1
        kan_grid_size=5
    ).to(device)
    
    # Initialize Parameters carefully
    # We want "some" physics to start with, not complete noise.
    # Convolutions: Identity + Random?
    # KAN: Initialized near zero?
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 2. Planner Helper
    # We don't need the full planner loop, just the evolution steps.
    # We simulate "Agent Drift" by adding an advection term manually.
    # Agent moves Right: v = (0, 1). Block stays: v = (0, 0).
    
    # 5D Advection Field: (B, C, 2, H, W)
    # B=1 (broadcasted), C=2
    advection_field = torch.zeros(1, 2, 2, size, size, device=device)
    
    # Channel 0 (Agent): vx = 1.0 (Right)
    advection_field[:, 0, 1, :, :] = 1.0 
    
    # Channel 1 (Block): vx = 0.0 (Static)
    # This FORCES the KAN to learn interaction to move the block!
    
    # 3. Training Loop
    epochs = 100
    history = []
    
    print("Starting Physics Discovery...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get Batch
        state, target_block = create_training_batch(batch_size=4, size=size, device=device)
        
        # Evolve for K steps
        # If we evolve too long, gradients vanish/explode. Start small.
        n_steps = 15 
        
        # Use trainable_evolve
        # Anchor is initial state? No, we want dynamics to handle it.
        # But we need data_weight to keep Agent from disappearing? 
        # Or let the Task Loss handle it?
        
        # We assume Agent is DRIVEN by advection (will/intention).
        # We assume Block is DRIVEN by physics (Hamiltonian).
        
        final_state = model.trainable_evolve(
            state, 
            n_steps=n_steps, 
            advection_field=advection_field,
            anchor=state.detach() # Anchor to self? Or None?
        )
        
        # Loss
        # 1. Task Loss: Block should be at target
        block_final = final_state[:, 1:2]
        task_loss = F.mse_loss(block_final, target_block)
        
        # 2. Agent Survival Loss (Optional? Agent should move)
        # If agent disappears, advection stops working? No, advection is external field here.
        
        # 3. Regularization?
        
        loss = task_loss
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        history.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")
            
            # Check Mass Conservation (Diagnostic)
            mass_initial = state[0, 1].sum().item()
            mass_final = final_state[0, 1].sum().item()
            print(f"  Block Mass: {mass_initial:.1f} -> {mass_final:.1f}")

    # Save
    torch.save(model.state_dict(), "./outputs/kan_physics.pth")
    print("Training Complete.")
    
    # Plot Loss
    plt.figure()
    plt.plot(history)
    plt.title("KAN Physics Learning Loss")
    plt.savefig("./outputs/kan_training_loss.png")
    
if __name__ == "__main__":
    train_kan_physics()
