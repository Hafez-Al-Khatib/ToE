
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('src')
from thermodynamic_field import ThermodynamicField

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=8, shuffle=True)

def debug_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    # Try to load the best model from Exp 1 (Control) first
    model_path = 'outputs/mnist_tnf_none.pt'
    if not os.path.exists(model_path):
        model_path = 'outputs/mnist_model.pt' # Fallback
    
    if not os.path.exists(model_path):
        print("No model found!")
        return

    print(f"Loading {model_path}...")
    # Architecture params from prove_theory.py defaults
    model = ThermodynamicField(n_channels=4, n_filters=16).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Learned step size: {model.step_size.item()}")

    # Get data
    loader = get_data()
    x_clean, _ = next(iter(loader))
    x_clean = x_clean.to(device)
    x_noisy = x_clean + torch.randn_like(x_clean) * 0.3

    # Test different step size scalings
    scalings = [1.0, 0.5, 0.1, 0.01]
    
    results = []

    for scale in scalings:
        current = x_noisy.clone()
        original_step = model.log_step_size.item()
        
        # Adjust step size temporarily
        # log_step_size = log(scale * exp(original)) = log(scale) + original
        import math
        model.log_step_size.data = torch.tensor(math.log(scale) + original_step).to(device)
        print(f"\nTesting scaling {scale} (effective η = {model.step_size.item():.4f})")

        psnrs = []
        for i in range(50):
            current = model.one_step_denoise(current).detach()
            # Calculate PSNR
            mse = F.mse_loss(current, x_clean)
            psnr = -10 * torch.log10(mse).item()
            psnrs.append(psnr)
            
            if i % 10 == 0:
                print(f"Step {i}: PSNR={psnr:.2f} dB")
        
        results.append((scale, psnrs, current.cpu()))
        
        # Restore
        model.log_step_size.data = torch.tensor(original_step).to(device)

    # Plot
    plt.figure(figsize=(10, 6))
    for scale, psnrs, _ in results:
        plt.plot(psnrs, label=f"Scale {scale}")
    plt.xlabel("Step")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.title("Inference Stability Analysis")
    plt.grid(True)
    plt.savefig('outputs/debug_inference_psnr.png')
    print("Saved outputs/debug_inference_psnr.png")

    # Visualize images
    fig, axes = plt.subplots(len(scalings)+2, 8, figsize=(16, 2*(len(scalings)+2)))
    
    # Row 0: Clean
    for i in range(8):
        axes[0,i].imshow(x_clean[i,0].cpu(), cmap='gray')
        axes[0,i].axis('off')
    axes[0,0].set_title("Clean")

    # Row 1: Noisy
    for i in range(8):
        axes[1,i].imshow(x_noisy[i,0].cpu(), cmap='gray')
        axes[1,i].axis('off')
    axes[1,0].set_title("Noisy")

    # Rows 2+: Results
    for idx, (scale, _, imgs) in enumerate(results):
        row = idx + 2
        for i in range(8):
            axes[row,i].imshow(imgs[i,0], cmap='gray')
            axes[row,i].axis('off')
        axes[row,0].set_title(f"Scale {scale}")

    plt.tight_layout()
    plt.savefig('outputs/debug_inference_images.png')
    print("Saved outputs/debug_inference_images.png")

if __name__ == "__main__":
    debug_inference()
