
import torch
import sys
sys.path.insert(0, "./src")
from hamiltonian_field import HamiltonianField

def inspect_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inspecting Model on {device}")
    
    # Load Model
    model = HamiltonianField(
        n_channels=1, 
        height=28, 
        width=28, 
        n_filters=16,
        kan_hidden=[32, 1],
        kan_grid_size=5
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("./outputs/kan_denoise.pth"))
        print("Loaded ./outputs/kan_denoise.pth")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Inspect Scalar Parameters
    print("\n--- Learned Scalars ---")
    print(f"Log Step Size: {model.log_step_size.item():.4f} -> Step Size: {model.step_size.item():.4f}")
    print(f"Log Data Weight: {model.log_data_weight.item():.4f} -> Data Weight: {model.data_weight.item():.4f}")
    
    # Inspect KAN
    print("\n--- KAN Statistics ---")
    kan_params = list(model.kan.parameters())
    param_abs_mean = [p.abs().mean().item() for p in kan_params]
    print(f"KAN Param Mean Abs: {sum(param_abs_mean)/len(param_abs_mean):.4f}")
    
    # Test Gradients on a dummy image
    print("\n--- Gradient Test ---")
    x = torch.randn(1, 1, 28, 28).to(device)
    x.requires_grad_(True)
    
    e = model.compute_energy(x)
    grad = torch.autograd.grad(e.sum(), x)[0]
    
    print(f"Energy: {e.item():.4f}")
    print(f"Gradient Mean Abs: {grad.abs().mean().item():.4f}")
    print(f"Gradient Max: {grad.abs().max().item():.4f}")
    
    # Dynamics Terms
    # update = -step_size * grad + data_weight * (anchor - u)
    # Let's say anchor is 0
    
    term_grad = -model.step_size.item() * grad.abs().mean().item()
    term_fidelity = model.data_weight.item() * 1.0 # arbitrary error
    
    print(f"\n--- Dynamics Magnitude (Approx) ---")
    print(f"Gradient Term (Force): {abs(term_grad):.4f}")
    print(f"Fidelity Term (Anchor): {abs(term_fidelity):.4f}")
    
    if abs(term_grad) < 1e-4:
        print(">> WARNING: Gradient term is negligible. Model is frozen or flat.")
    
    if model.step_size.item() < 1e-3:
        print(">> WARNING: Step size is vanishingly small.")

if __name__ == "__main__":
    inspect_model()
