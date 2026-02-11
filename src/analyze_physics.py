
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.append('src')
from thermodynamic_field import ThermodynamicField

def analyze_physics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Control Model
    model_path = 'outputs/mnist_tnf_none.pt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading {model_path}...")
    # Architecture params from prove_theory.py defaults
    model = ThermodynamicField(n_channels=4, n_filters=16).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Turing Coefficients (Alphas)
    alphas = model.alphas.detach().cpu().numpy()
    print("\n=== Turing Coefficients (Diffusion Rates) ===")
    print(f"Alphas: {alphas}")
    
    # Check divergence
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    print(f"Mean: {mean_alpha:.4f}, Std: {std_alpha:.4f}")
    if std_alpha > 0.1 * mean_alpha:
        print(">> Divergence DETECTED (System learned distinct diffusers)")
    else:
        print(">> Homogeneous diffusion (System failed to differentiate)")

    # 2. Potential Shape V(u) = a*u^2 + b*u^4
    a = model.potential_a.detach().cpu().numpy()
    b = model.potential_b.detach().cpu().numpy()
    
    print("\n=== KAN Potentials (Energy Wells) ===")
    print(f"a (quadratic): {a}")
    print(f"b (quartic):   {b}")

    # Plot Potentials
    u = np.linspace(-2, 2, 100)
    plt.figure(figsize=(10, 6))
    
    for i in range(len(a)):
        # V(u) = a*u^2 + b*u^4
        # We plot the learned potential shape
        v = a[i] * (u**2) + b[i] * (u**4)
        label = f"Ch {i}: a={a[i]:.2f}, b={b[i]:.2f}"
        plt.plot(u, v, label=label, linewidth=2)
        
        # Check type
        if a[i] < 0 and b[i] > 0:
            print(f"Channel {i}: DOUBLE WELL (Bistable)")
        elif a[i] > 0 and b[i] > 0:
            print(f"Channel {i}: SINGLE WELL (Monostable)")
        else:
            print(f"Channel {i}: Complex/Unstable (a={a[i]:.2f}, b={b[i]:.2f})")

    plt.xlabel("Field Value (u)")
    plt.ylabel("Potential Energy V(u)")
    plt.title("Learned Energy Landscapes per Channel")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 5) # Focus on the well
    plt.savefig('outputs/physics_analysis_potentials.png')
    print("\nSaved outputs/physics_analysis_potentials.png")

    # 3. Inference Step Size
    print(f"\nLearned Inference Step Size (η): {model.step_size.item():.4f}")

if __name__ == "__main__":
    analyze_physics()
