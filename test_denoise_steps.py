
import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./src")
from hamiltonian_field import HamiltonianField
from train_kan_denoise import SyntheticShapesDataset

def test_inference_steps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Inference Steps on {device}")
    
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

    # Data
    dataset = SyntheticShapesDataset(length=10)
    sample, _ = dataset[0] # Deterministic if seeded, otherwise random
    sample = sample.unsqueeze(0).to(device) # (1, 1, 28, 28)
    
    noise_std = 0.3
    noisy = sample + torch.randn_like(sample) * noise_std
    
    # Run long inference
    print("Running 100 steps...")
    traj_long = model.evolve(noisy, n_steps=100, return_trajectory=True)
    
    # Viz
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    axes[0].imshow(sample[0,0].cpu(), cmap='gray')
    axes[0].set_title("Clean")
    
    axes[1].imshow(noisy[0,0].cpu(), cmap='gray')
    axes[1].set_title("Noisy")
    
    indices = [10, 30, 60, 99]
    for i, idx in enumerate(indices):
        img = traj_long[idx, 0, 0].detach().cpu()
        axes[i+2].imshow(img, cmap='gray')
        axes[i+2].set_title(f"Step {idx}")
        
    plt.savefig("./outputs/kan_denoise_long.png")
    print("Saved to ./outputs/kan_denoise_long.png")

if __name__ == "__main__":
    test_inference_steps()
