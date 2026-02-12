
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./src")
from hamiltonian_field import HamiltonianField
# import torchvision # Broken in this env
# import torchvision.transforms as transforms

class SyntheticShapesDataset(torch.utils.data.Dataset):
    """
    Generates random images of circles and squares.
    """
    def __init__(self, size=28, length=1000):
        self.size = size
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Generate random shape
        img = torch.zeros(1, self.size, self.size) - 1.0 # Background -1
        
        # Random params
        shape_type = torch.randint(0, 2, (1,)).item()
        cx = torch.randint(5, self.size-5, (1,)).item()
        cy = torch.randint(5, self.size-5, (1,)).item()
        r = torch.randint(3, 8, (1,)).item()
        
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing='ij')
        
        if shape_type == 0: # Circle
            mask = ((x - cx)**2 + (y - cy)**2) <= r**2
        else: # Square
            mask = (x >= cx-r) & (x <= cx+r) & (y >= cy-r) & (y <= cy+r)
            
        img[0][mask] = 1.0 # Foreground +1
        return img, 0 # Dummy label

def train_kan_denoise():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    # 1. Data
    print("Generating Synthetic Shapes Dataset...")
    trainset = SyntheticShapesDataset(length=1000)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=0
    )
    
    # 2. Model: Hamiltonian KAN
    # Single Channel (Grayscale) -> Energy
    # n_filters=16 -> KAN Input=16
    model = HamiltonianField(
        n_channels=1, 
        height=28, 
        width=28, 
        n_filters=16,
        kan_hidden=[32, 1], # 16->32->1
        kan_grid_size=5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 5
    noise_std = 0.3
    history = []
    
    print("Starting Denoising Training...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            
            # Corrupt
            noise = torch.randn_like(inputs) * noise_std
            noisy_inputs = inputs + noise
            
            optimizer.zero_grad()
            
            # One-Step Denoising Loss
            # x_denoised = x_noisy - step_size * dE/dx
            # We use manually computed gradient step
            
            # Ensure gradients
            noisy_inputs.requires_grad_(True)
            energy = model.compute_energy_from_image(noisy_inputs) # (B, 1, H, W) -> scalar
            
            grad = torch.autograd.grad(
                energy.sum(), noisy_inputs, create_graph=True
            )[0]
            
            # Step size is trainable parameter of model (log_step_size)
            step_size = model.step_size
            
            denoised_pred = noisy_inputs - step_size * grad
            
            # Loss: Prediction should be closer to Clean
            loss = F.mse_loss(denoised_pred, inputs)
            
            # Regularization for stability
            reg = 0.001 * (grad ** 2).mean()
            loss = loss + reg
            
            loss.backward()
            
            # Clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss / 100:.4f}")
                history.append(running_loss / 100)
                running_loss = 0.0
                
    # Save
    torch.save(model.state_dict(), "./outputs/kan_denoise.pth")
    print("Training Complete.")
    
    # Validation / Visualization
    model.eval()
    
    # Take a sample
    sample, _ = next(iter(trainloader))
    sample = sample[0:1].to(device)
    noisy_sample = sample + torch.randn_like(sample) * noise_std
    
    # Multi-step Denoising (Inference)
    # We MUST disable no_grad because evolve uses autograd internally!
    # But we can detach outputs.
    print("Running Inference...")
    denoised_trajectory = model.evolve(
        noisy_sample, 
        n_steps=30, 
        return_trajectory=True
    ) # (Steps, B, C, H, W)
    
    # Viz
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    # Original
    axes[0].imshow(sample[0,0].cpu(), cmap='gray')
    axes[0].set_title("Clean")
    
    # Noisy
    axes[1].imshow(noisy_sample[0,0].cpu(), cmap='gray')
    axes[1].set_title("Noisy")
    
    # Steps
    step_indices = [0, 10, 29]
    for i, step_idx in enumerate(step_indices):
        img = denoised_trajectory[step_idx, 0, 0].detach().cpu()
        axes[i+2].imshow(img, cmap='gray')
        axes[i+2].set_title(f"Step {step_idx}")

    plt.tight_layout()
    plt.savefig("./outputs/kan_denoise_result.png")
    print("Saved result to ./outputs/kan_denoise_result.png")

if __name__ == "__main__":
    train_kan_denoise()
