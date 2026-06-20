
import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./src")
from hamiltonian_field import HamiltonianField
from train_kan_denoise import SyntheticShapesDataset

def test_inference_anneal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Inference Annealing on {device}")
    
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
    sample, _ = dataset[0] 
    sample = sample.unsqueeze(0).to(device) 
    
    noise_std = 0.3
    noisy = sample + torch.randn_like(sample) * noise_std
    
    # === STRATEGY: Cut The Rope ===
    # Phase 1: 50 steps WITH anchor (Find the basin of attraction)
    # Phase 2: 50 steps WITHOUT anchor (Slide to the bottom)
    
    print("Phase 1: Anchored Evolve (50 steps)...")
    z = model.evolve(noisy, n_steps=50, anchor=noisy)
    
    print("Phase 2: Free Evolve (50 steps)...")
    # anchor=None means the model anchors to the *current state*, 
    # effectively meaning drift is only limited by self-consistency.
    # Wait, if anchor=None in evolve(), it anchors to the input image? 
    # Let's check source code.
    # evolve: if anchor is None: anchor = image.detach().clone()
    # So if we pass z as image, it anchors to z.
    # We want NO anchor force.
    # But evolve() enforces fidelity = data_weight * (anchor - u).
    # We can't easily disable it via arguments unless we pass a mask of zeros?
    # Or we can temporarily zero out model.log_data_weight.
    
    # Hack: set data_weight to -20 (effectively zero)
    original_dw = model.log_data_weight.data.clone()
    model.log_data_weight.data.fill_(-10.0) # exp(-10) is tiny
    
    z_final = model.evolve(z, n_steps=50, anchor=z) # Anchor doesn't matter if weight is 0
    
    # Restore
    model.log_data_weight.data = original_dw
    
    # Viz
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    axes[0].imshow(sample[0,0].cpu(), cmap='gray')
    axes[0].set_title("Clean")
    
    axes[1].imshow(noisy[0,0].cpu(), cmap='gray')
    axes[1].set_title("Noisy")
    
    axes[2].imshow(z[0,0].cpu(), cmap='gray')
    axes[2].set_title("Phase 1 (Anchored)")
    
    axes[3].imshow(z_final[0,0].cpu(), cmap='gray')
    axes[3].set_title("Phase 2 (Free)")
        
    plt.savefig("./outputs/kan_denoise_anneal.png")
    print("Saved to ./outputs/kan_denoise_anneal.png")

if __name__ == "__main__":
    test_inference_anneal()
