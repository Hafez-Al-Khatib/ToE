"""
gen_qualitative_comparison.py
=============================
Generate a qualitative comparison figure for the paper.

Shows side-by-side:
  - Clean image
  - Noisy image
  - KAN-EBM denoised (K=optimal)
  - FFN-DSM denoised
  - ConvMLP-GELU denoised (K=optimal)

Run: python experiments/gen_qualitative_comparison.py --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel, FFNDenoiser
from rebuttal_smooth_mlp import ConvSmoothMLPEBM


def load_kan(path, device):
    model = KANEnergyModel(n_filters=32, kan_hidden=[96, 16]).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_conv_mlp(path, device, activation='gelu'):
    model = ConvSmoothMLPEBM(n_filters=16, mlp_hidden=160, activation=activation).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_ffn(path, device):
    model = FFNDenoiser(hidden=512).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def iterative_denoise(model, noisy, K, eta0=0.05, delta=0.97):
    x = noisy.clone()
    for t in range(K):
        x.requires_grad_(True)
        E = model.energy(x)
        grad = torch.autograd.grad(E, x, create_graph=False)[0]
        eta = eta0 * (delta ** t)
        with torch.no_grad():
            x = x - eta * grad
        x = x.detach().clamp(-1, 1)
    return x


def to_image(tensor):
    """Convert [-1,1] tensor to [0,1] numpy image."""
    img = tensor.cpu().numpy()
    img = (img + 1) / 2
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_examples', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--K_kan', type=int, default=5)
    parser.add_argument('--K_mlp', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Load data
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=False, download=True, transform=tf)

    # Load models
    kan = load_kan(ROOT / 'outputs/cifar10/kan_ebm_f32.pt', device)
    conv_mlp = load_conv_mlp(ROOT / 'outputs/finalization/rebuttal/conv_mlp_gelu.pt', device, 'gelu')
    ffn = load_ffn(ROOT / 'outputs/cifar10/ffn_dsm_f32.pt', device)

    # Select diverse examples
    indices = [0, 15, 42, 77][:args.n_examples]
    examples = [test_ds[i][0] for i in indices]

    fig, axes = plt.subplots(args.n_examples, 5, figsize=(12, 3 * args.n_examples))
    if args.n_examples == 1:
        axes = axes.reshape(1, -1)

    col_labels = ['Clean', f'Noisy ($\\sigma$={args.sigma})', 'KAN-EBM', 'ConvMLP-GELU', 'FFN-DSM']

    for row, img in enumerate(examples):
        img = img.to(device)
        noisy = (img + torch.randn_like(img) * args.sigma).clamp(-1, 1)

        with torch.no_grad():
            ffn_out = ffn(noisy.unsqueeze(0)).squeeze(0).clamp(-1, 1)

        kan_out = iterative_denoise(kan, noisy.unsqueeze(0), args.K_kan).squeeze(0)
        mlp_out = iterative_denoise(conv_mlp, noisy.unsqueeze(0), args.K_mlp).squeeze(0)

        imgs = [img, noisy, kan_out, mlp_out, ffn_out]
        for col, im in enumerate(imgs):
            axes[row, col].imshow(to_image(im))
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=11)

    plt.tight_layout()
    out_path = ROOT / 'outputs' / 'scaled_eval' / 'qualitative_comparison.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved qualitative comparison to {out_path}")


if __name__ == '__main__':
    main()
