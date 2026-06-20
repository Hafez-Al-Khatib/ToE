
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, "./src")
from thermodynamic_field import ThermodynamicField

def audit_kan_landscape():
    device = "cpu"
    # Create a TNF with 2 channels
    model = ThermodynamicField(n_channels=2, height=16, width=16).to(device)
    
    # We want to see the energy E(u_agent, u_block)
    # as u_agent and u_block vary.
    # Theoretical ideal: high energy when both are 1.0
    
    u_vals = torch.linspace(-1.5, 1.5, 50)
    U_A, U_B = torch.meshgrid(u_vals, u_vals, indexing='ij')
    
    # Reshape for KAN input: (N, 2)
    inputs = torch.stack([U_A.flatten(), U_B.flatten()], dim=1)
    
    with torch.no_grad():
        # The KAN energy head in TNF evaluates local density energy
        # For simplicity, let's check the potential_kan directly
        # E_pot = model.potential_kan(inputs)
        
        # Also check the coupling part
        # E_total = V(u_a) + V(u_b) + beta * u_a^2 * u_b^2
        
        # TNF v3 potential calculation:
        pot = model.potential_kan(inputs).view(50, 50)
        
        # Coupling term (1D in TNF v3)
        coupling = (model.coupling_weights[0] * (U_A**2) * (U_B**2))
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(U_A.numpy(), U_B.numpy(), pot.numpy(), levels=20)
    plt.colorbar(label='KAN Potential V(u)')
    plt.xlabel('Agent Density (u_a)')
    plt.ylabel('Block Density (u_b)')
    plt.title('Learned KAN Landscape')
    
    plt.subplot(1, 2, 2)
    plt.contourf(U_A.numpy(), U_B.numpy(), (pot + coupling).numpy(), levels=20)
    plt.colorbar(label='Total Local Energy E')
    plt.xlabel('Agent Density (u_a)')
    plt.ylabel('Block Density (u_b)')
    plt.title('Total Energy (Potential + Coupling)')
    
    os.makedirs("./results/audit", exist_ok=True)
    plt.savefig("./results/audit/kan_energy_audit.png")
    print("Audit Figure Saved: results/audit/kan_energy_audit.png")

if __name__ == "__main__":
    audit_kan_landscape()
