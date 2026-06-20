import json
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

import landscape_geometry as lg
from rebuttal_smooth_mlp import ConvSmoothMLPEBM

# Load model
model = ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160,
                         n_channels=3, activation="tanh").cuda()
ckpt = ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_tanh.pt"
state = torch.load(ckpt, map_location='cuda', weights_only=False)
model.load_state_dict(state)
model.eval()

# Load data
ds = lg.load_cifar10()
images = [ds[i][0] for i in range(100)]

# Measure Hessian at sigma=0.15 (closest to table)
print("Measuring Hessian for Tanh at sigma=0.15...")
sigma = 0.15
lambda_max_values = []
for img_tensor in images[:8]:  # 8 samples like existing data
    torch.manual_seed(42)
    x_noisy = img_tensor + torch.randn_like(img_tensor) * sigma
    x_noisy = x_noisy.cuda().unsqueeze(0)
    with torch.enable_grad():
        x_noisy.requires_grad_(True)
        E = model.energy(x_noisy)
        grad = torch.autograd.grad(E.sum(), x_noisy, create_graph=True)[0]
        # Power iteration for lambda_max
        v = torch.randn_like(x_noisy)
        for _ in range(50):
            v = v / (v.norm() + 1e-8)
            hvp = torch.autograd.grad(grad, x_noisy, v, retain_graph=True)[0]
            v = hvp
        lambda_max = v.norm().item()
        lambda_max_values.append(lambda_max)
    print(f"  Sample {len(lambda_max_values)}: lambda_max={lambda_max:.2f}")

print(f"\nMean lambda_max for Tanh at sigma={sigma}: {np.mean(lambda_max_values):.1f}")
