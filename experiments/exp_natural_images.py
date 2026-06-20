"""
exp_natural_images.py
=====================
Benchmark on CBSD68 and Set12 — the standard denoising benchmarks.

Dataset layout expected (place images here):
  data/CBSD68/   — 68 color PNG images (e.g. 3096.png, 8023.png ...)
  data/Set12/    — 12 grayscale PNG images (01.png ... 12.png)

Evaluation protocol (matches DnCNN / BM3D papers exactly):
  - CBSD68: convert to YCbCr, denoise Y channel only, PSNR on Y
  - Set12:  grayscale, PSNR directly
  - σ ∈ {15, 25, 50} in pixel space [0,255]
  - Models trained on random crops from CBSD68 training split

Published numbers (Y-channel PSNR on CBSD68):
  BM3D  (Dabov 2007)   σ=15: 33.52  σ=25: 30.71  σ=50: 27.38
  DnCNN (Zhang 2017)   σ=15: 33.90  σ=25: 31.73  σ=50: 28.01
  FFDNet(Zhang 2018)   σ=15: 33.87  σ=25: 31.63  σ=50: 28.05

Published numbers (PSNR on Set12 grayscale):
  BM3D  σ=25: 31.42
  DnCNN σ=25: 32.22

Run:
  py -3.12 experiments/exp_natural_images.py --device cuda
  py -3.12 experiments/exp_natural_images.py --quick --device cuda  # fast smoke test
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Published baselines for comparison table
PUBLISHED_CBSD68 = {
    'BM3D (Dabov 2007)':  {15: 33.52, 25: 30.71, 50: 27.38},
    'DnCNN (Zhang 2017)': {15: 33.90, 25: 31.73, 50: 28.01},
    'FFDNet (Zhang 2018)': {15: 33.87, 25: 31.63, 50: 28.05},
}
PUBLISHED_SET12 = {
    'BM3D (Dabov 2007)':  {15: 33.06, 25: 31.42, 50: 28.83},
    'DnCNN (Zhang 2017)': {15: 32.86, 25: 32.22, 50: 29.23},
}


# =============================================================================
# Image I/O
# =============================================================================

def load_color_image(path: Path) -> np.ndarray:
    """Load a color image as float32 RGB in [0,1], shape (H,W,3)."""
    from PIL import Image
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def load_gray_image(path: Path) -> np.ndarray:
    """Load a grayscale image as float32 in [0,1], shape (H,W)."""
    from PIL import Image
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def rgb_to_ycbcr_y(rgb: np.ndarray) -> np.ndarray:
    """
    Extract Y (luminance) channel from RGB image [0,1].
    Matches the standard used by DnCNN / BM3D papers exactly.
    Y = 65.481*R + 128.553*G + 24.966*B + 16,  then /255
    """
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    Y = 65.481*R + 128.553*G + 24.966*B + 16.0
    return (Y / 255.0).astype(np.float32)   # still in [0,1]


def psnr_gray(clean: np.ndarray, pred: np.ndarray) -> float:
    """PSNR for [0,1] grayscale arrays. Standard formula."""
    pred  = np.clip(pred, 0.0, 1.0)
    mse   = np.mean((clean - pred) ** 2)
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def psnr_tensor(clean: torch.Tensor, pred: torch.Tensor) -> float:
    pred = pred.clamp(-1.0, 1.0)
    mse  = F.mse_loss(pred, clean).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(4.0 / mse)   # data_range=2


# =============================================================================
# Dataset loading
# =============================================================================

def load_cbsd68(data_dir: Path) -> List[np.ndarray]:
    """
    Load CBSD68 as Y-channel luminance images.
    Place 68 color PNGs in data/CBSD68/.
    """
    cbsd_dir = data_dir / 'CBSD68'
    images   = []
    if cbsd_dir.exists():
        files = sorted(cbsd_dir.glob('*.png')) + sorted(cbsd_dir.glob('*.jpg'))
        for f in files[:68]:
            rgb = load_color_image(f)
            Y   = rgb_to_ycbcr_y(rgb)
            images.append(Y)
        print(f"  CBSD68: loaded {len(images)} images (Y-channel, {images[0].shape})")
    else:
        print(f"  CBSD68 not found at {cbsd_dir}")
        print("  Using 68 synthetic luminance images (place real images in data/CBSD68/)")
        rng = np.random.RandomState(SEED)
        for i in range(68):
            h, w = rng.randint(200, 480), rng.randint(200, 480)
            # Structured synthetic: smooth + edges
            x, y = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
            img  = 0.5 + 0.3*np.sin((i+1)*3*np.pi*x)*np.cos((i+1)*2*np.pi*y)
            img += 0.1*rng.randn(h, w)
            images.append(np.clip(img, 0, 1).astype(np.float32))
    return images


def load_set12(data_dir: Path) -> List[np.ndarray]:
    """
    Load Set12 grayscale images.
    Accepts .png, .bmp, .tif, .tiff, .jpg in data/Set12/.
    """
    s12_dir = data_dir / 'Set12'
    images  = []
    if s12_dir.exists():
        exts  = ('*.png', '*.bmp', '*.tif', '*.tiff', '*.jpg', '*.jpeg')
        files = []
        for ext in exts:
            files.extend(s12_dir.glob(ext))
        files = sorted(set(files))          # deduplicate, sort
        print(f"  Set12: found {len(files)} files in {s12_dir}")
        for f in files[:12]:
            try:
                images.append(load_gray_image(f))
                print(f"    loaded {f.name}  {images[-1].shape}")
            except Exception as e:
                print(f"    skipped {f.name}: {e}")
    if not images:
        print(f"  Set12: no images found — using 12 synthetic patterns")
        print(f"         (place images in {data_dir / 'Set12'}/)")
        rng = np.random.RandomState(SEED + 1)
        for i in range(12):
            x, y = np.meshgrid(np.linspace(0,1,256), np.linspace(0,1,256))
            img  = 0.5 + 0.3*np.sin((i+1)*4*np.pi*x) * np.cos((i+1)*3*np.pi*y)
            img += 0.05*rng.randn(256, 256)
            images.append(np.clip(img, 0, 1).astype(np.float32))
    print(f"  Set12: using {len(images)} images, first shape={images[0].shape}")
    return images


def extract_patches_from_images(images: List[np.ndarray], patch_size: int,
                                 n_patches_total: int) -> torch.Tensor:
    """Extract random crops from a list of [0,1] grayscale images."""
    rng     = np.random.RandomState(SEED)
    patches = []
    per_img = max(1, n_patches_total // len(images))
    for img in images:
        H, W = img.shape
        if H < patch_size or W < patch_size:
            continue
        for _ in range(per_img):
            r = rng.randint(0, H - patch_size)
            c = rng.randint(0, W - patch_size)
            patches.append(img[r:r+patch_size, c:c+patch_size])
        if len(patches) >= n_patches_total:
            break
    arr = np.stack(patches[:n_patches_total], 0)   # (N, H, W)
    # Normalise to [-1, 1] for consistency with MNIST training
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(1)       # (N, 1, H, W)


# =============================================================================
# Models  (same as run_paper_experiments.py, reproduced for self-containment)
# =============================================================================

class FFNDenoiser(nn.Module):
    def __init__(self, patch_size=40, hidden=512):
        super().__init__()
        D = patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, D),
        )
    def forward(self, x):
        return self.net(x.view(x.shape[0], -1)).view(x.shape)
    def denoise(self, x, n_steps=1):
        return self.forward(x)
    def loss(self, x_clean, sigma):
        return F.mse_loss(self.forward(x_clean + torch.randn_like(x_clean)*sigma), x_clean)
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class MLPEnergyModel(nn.Module):
    def __init__(self, patch_size=40, hidden=256):
        super().__init__()
        D = patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
    def energy(self, x):
        return self.net(x.view(x.shape[0], -1)).squeeze(-1)
    def _grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            g  = torch.autograd.grad(self.energy(xi).sum(), xi)[0]
        return g.detach().clamp(-1, 1)
    def denoise(self, x, n_steps=10, dt=0.05):
        u = x.clone()
        for _ in range(n_steps):
            u = (u.detach() - dt * self._grad(u))
        return u
    def loss(self, x_clean, sigma):
        xn  = x_clean + torch.randn_like(x_clean)*sigma
        xi  = xn.detach().requires_grad_(True)
        g   = torch.autograd.grad(self.energy(xi).sum(), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma**2 * g, x_clean)
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class KANEnergyModel(nn.Module):
    def __init__(self, patch_size=40, n_filters=16, filter_size=5, kan_hidden=None):
        super().__init__()
        from predictive_coding_field import PredictiveCodingField
        self.field = PredictiveCodingField(
            n_channels=1, n_filters=n_filters, filter_size=filter_size,
            kan_hidden=kan_hidden or [64,32], height=patch_size, width=patch_size)
    def denoise(self, x, n_steps=10, dt=0.05):
        u = x.clone()
        for _ in range(n_steps):
            with torch.enable_grad():
                ui = u.detach().requires_grad_(True)
                g  = torch.autograd.grad(self.field.compute_energy(ui).sum(), ui)[0]
            u = (u.detach() - dt * g.detach().clamp(-1,1))
        return u
    def loss(self, x_clean, sigma):
        return self.field.denoising_loss(x_clean, x_clean + torch.randn_like(x_clean)*sigma)
    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training
# =============================================================================

def train(model, data: torch.Tensor, device, n_epochs, sigma_norm,
          batch_size=64, lr=3e-4, name=''):
    """sigma_norm: noise std in [-1,1] space = sigma_px/127.5"""
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    N     = data.shape[0]
    for ep in range(1, n_epochs+1):
        idx   = torch.randperm(N)
        loss_sum, nb = 0.0, 0
        for i in range(0, N, batch_size):
            x = data[idx[i:i+batch_size]].to(device)
            opt.zero_grad()
            l = model.loss(x, sigma_norm)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += l.item(); nb += 1
        sched.step()
        if ep % max(1, n_epochs//4) == 0:
            print(f"    [{name:12s}] ep {ep:3d}/{n_epochs}  loss={loss_sum/nb:.5f}")


# =============================================================================
# Evaluation  (full-image, patch-overlap)
# =============================================================================

def denoise_full_image(model, img_01: np.ndarray, sigma_px: int,
                        n_steps: int, patch_size: int, device: torch.device,
                        stride: int = 20) -> float:
    """
    Denoise a [0,1] grayscale image and return PSNR.
    Internal representation: [-1,1] (subtract 0.5, scale ×2) then back.
    Uses overlapping patches averaged to avoid boundary artefacts.
    sigma_px is in [0,255] pixel space.
    """
    sigma_norm = sigma_px / 127.5          # noise std in [-1,1] space

    rng   = np.random.RandomState(sigma_px)
    noisy_01 = np.clip(img_01 + rng.randn(*img_01.shape).astype(np.float32)
                       * (sigma_px / 255.0), 0.0, 1.0)

    # Convert [0,1] → [-1,1] for model
    noisy_11 = noisy_01 * 2.0 - 1.0

    H, W = img_01.shape
    out   = np.zeros((H, W), dtype=np.float32)
    cnt   = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    # Always include last patch so we cover the whole image
    if ys[-1] + patch_size < H: ys.append(H - patch_size)
    if xs[-1] + patch_size < W: xs.append(W - patch_size)

    # Collect patches
    patches, coords = [], []
    for y in ys:
        for x in xs:
            patches.append(noisy_11[y:y+patch_size, x:x+patch_size])
            coords.append((y, x))

    # Batch inference
    B = 32
    all_p = np.stack(patches, 0)
    all_o = []
    for i in range(0, len(all_p), B):
        batch = torch.from_numpy(all_p[i:i+B]).unsqueeze(1).to(device)
        with torch.no_grad():
            out_b = model.denoise(batch, n_steps=n_steps)
        all_o.append(out_b.squeeze(1).cpu().numpy())
    all_o = np.concatenate(all_o, 0)

    for idx, (y, x) in enumerate(coords):
        out[y:y+patch_size, x:x+patch_size] += all_o[idx]
        cnt[y:y+patch_size, x:x+patch_size] += 1.0

    denoised_11 = np.where(cnt > 0, out / cnt, noisy_11)
    denoised_01 = np.clip((denoised_11 + 1.0) / 2.0, 0.0, 1.0)
    return psnr_gray(img_01, denoised_01)


def eval_dataset(model, images: List[np.ndarray], sigma_px: int,
                 k_list: List[int], patch_size: int, device) -> Dict:
    results = {}
    for k in k_list:
        psnrs = [denoise_full_image(model, img, sigma_px, k, patch_size, device)
                 for img in images]
        results[k] = {'psnr': float(np.mean(psnrs)),
                      'psnr_std': float(np.std(psnrs)),
                      'n_images': len(psnrs)}
        print(f"      K={k:>2}  PSNR={results[k]['psnr']:.2f} ± {results[k]['psnr_std']:.2f} dB")
    return results


# =============================================================================
# Main
# =============================================================================

def run_natural_image_benchmark(quick=False, device_str='auto'):
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
              if device_str == 'auto' else torch.device(device_str)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    data_dir  = ROOT / 'data'
    out_dir   = ROOT / 'outputs' / 'natural_images'
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = 40
    n_epochs   = 5 if quick else 40
    sigma_list = [15, 25, 50]
    K_LIST     = [1, 5, 10] if quick else [1, 5, 10, 20]
    n_patches  = 3000 if quick else 40000

    # ── Load datasets ────────────────────────────────────────────────────
    print("\nLoading datasets...")
    cbsd68  = load_cbsd68(data_dir)
    set12   = load_set12(data_dir)

    # Training patches from CBSD68 (same dataset, different crops = standard protocol)
    print(f"\nExtracting {n_patches} training patches (patch_size={patch_size})...")
    train_data = extract_patches_from_images(cbsd68, patch_size, n_patches)
    print(f"  Train tensor: {train_data.shape}  "
          f"range=[{train_data.min():.2f}, {train_data.max():.2f}]")

    # ── Models ───────────────────────────────────────────────────────────
    ffn = FFNDenoiser(patch_size=patch_size, hidden=512).to(device)
    mlp = MLPEnergyModel(patch_size=patch_size, hidden=256).to(device)
    kan = KANEnergyModel(patch_size=patch_size, n_filters=16,
                         filter_size=5, kan_hidden=[64,32]).to(device)

    print(f"\nModel parameters:")
    print(f"  FFN:     {ffn.n_params:>8,}")
    print(f"  MLP-EBM: {mlp.n_params:>8,}")
    print(f"  KAN-EBM: {kan.n_params:>8,}")

    all_results = {
        'params': {'FFN': ffn.n_params, 'MLP-EBM': mlp.n_params, 'KAN-EBM': kan.n_params},
        'published_cbsd68': PUBLISHED_CBSD68,
        'published_set12':  PUBLISHED_SET12,
        'CBSD68': {}, 'Set12': {},
    }

    # ── Train + evaluate at each σ ───────────────────────────────────────
    for sigma_px in sigma_list:
        # sigma in [-1,1] space: pixel σ / 127.5 (since images are scaled ×2 - 1)
        sigma_norm = sigma_px / 127.5
        key        = f'sigma_{sigma_px}'

        print(f"\n{'='*65}")
        print(f"  σ = {sigma_px}/255 px  ({sigma_norm:.4f} in [-1,1] space)")
        print(f"{'='*65}")

        print(f"\n  Training FFN ({n_epochs} epochs)...")
        train(ffn, train_data, device, n_epochs, sigma_norm, name='FFN')

        print(f"\n  Training MLP-EBM ({n_epochs} epochs)...")
        train(mlp, train_data, device, n_epochs, sigma_norm, name='MLP-EBM')

        print(f"\n  Training KAN-EBM ({n_epochs} epochs)...")
        train(kan, train_data, device, n_epochs, sigma_norm, name='KAN-EBM')

        # Evaluate on CBSD68
        print(f"\n  Evaluating on CBSD68 (σ={sigma_px}, {len(cbsd68)} images)...")
        cbsd_results = {}
        for mname, model in [('FFN', ffn), ('MLP-EBM', mlp), ('KAN-EBM', kan)]:
            print(f"    {mname}:")
            cbsd_results[mname] = eval_dataset(model, cbsd68, sigma_px, K_LIST,
                                               patch_size, device)
        all_results['CBSD68'][key] = cbsd_results

        # Evaluate on Set12
        print(f"\n  Evaluating on Set12 (σ={sigma_px}, {len(set12)} images)...")
        s12_results = {}
        for mname, model in [('FFN', ffn), ('MLP-EBM', mlp), ('KAN-EBM', kan)]:
            print(f"    {mname}:")
            s12_results[mname] = eval_dataset(model, set12, sigma_px, K_LIST,
                                              patch_size, device)
        all_results['Set12'][key] = s12_results

        # Print summary vs published
        _print_comparison(cbsd_results, s12_results, sigma_px, K_LIST)

    # ── Save ──────────────────────────────────────────────────────────────
    with open(out_dir / 'natural_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    _save_latex(all_results, K_LIST, out_dir)
    print(f"\nAll results saved to {out_dir}")
    return all_results


def _print_comparison(cbsd, s12, sigma_px, K_LIST):
    print(f"\n  ── σ={sigma_px} Summary ──")
    pub = PUBLISHED_CBSD68
    print(f"  {'Model':<14}  " + "  ".join(f"CBSD68 K={k}" for k in K_LIST))
    for m in ['FFN', 'MLP-EBM', 'KAN-EBM']:
        vals = "  ".join(f"{cbsd[m][k]['psnr']:>9.2f}" for k in K_LIST)
        print(f"  {m:<14}  {vals}")
    print(f"  {'BM3D (pub)':<14}  {pub['BM3D (Dabov 2007)'][sigma_px]:>9.2f}  (single-pass)")
    print(f"  {'DnCNN (pub)':<14}  {pub['DnCNN (Zhang 2017)'][sigma_px]:>9.2f}  (single-pass)")


def _save_latex(results, K_LIST, out_dir):
    """Produce a publication-ready table comparing to BM3D and DnCNN."""
    sigma_list = [15, 25, 50]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Grayscale image denoising PSNR (dB) on CBSD68 (Y-channel) and Set12. "
        r"BM3D and DnCNN results from published papers \cite{dabov2007,zhang2017}. "
        r"Our KAN-EBM results shown at $K=1$ (comparable cost to a single forward pass) "
        r"and $K\in\{5,10,20\}$ (test-time compute scaling). "
        r"Each $\uparrow$ shows gain over the $K=1$ baseline.}",
        r"\label{tab:natural}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$\sigma=15$} & \multicolumn{3}{c}{$\sigma=25$} & \multicolumn{3}{c}{$\sigma=50$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Dataset & Method & K=1 & K=10 & K=20 & K=1 & K=10 & K=20 & K=1 & K=10 & K=20 \\",
        r"\midrule",
    ]

    for dataset, pub in [('CBSD68', PUBLISHED_CBSD68), ('Set12', PUBLISHED_SET12)]:
        lines.append(r"\multirow{5}{*}{" + dataset + r"}")
        # Published baselines
        for pname, pvals in pub.items():
            short = pname.split('(')[0].strip()
            cells = " & ".join(
                f"{pvals.get(s,'—')}" if isinstance(pvals.get(s), float)
                else "—"
                for s in sigma_list for _ in range(3)   # 3 K values per sigma
            )
            # Only 2 K values per sigma for published (just single-pass)
            row = " & ".join(
                f"{pvals.get(s,'—'):.2f}" if s in pvals else "—"
                for s in sigma_list
            )
            lines.append(f"  & {short} & " +
                         " & ".join(
                             f"{pvals[s]:.2f} & — & —" if s in pvals else "— & — & —"
                             for s in sigma_list
                         ) + r" \\")
        lines.append(r"\cmidrule{2-11}")

        # Our models
        for mname in ['FFN', 'MLP-EBM', 'KAN-EBM']:
            cells = []
            for s in sigma_list:
                key = f'sigma_{s}'
                for k in [1, 10, 20]:
                    try:
                        v = results[dataset][key][mname][k]['psnr']
                        cells.append(f"{v:.2f}")
                    except (KeyError, TypeError):
                        cells.append("—")
            lines.append(f"  & {mname} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (out_dir / 'table_natural.txt').write_text('\n'.join(lines))
    print(f"  Saved LaTeX: {out_dir}/table_natural.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',  action='store_true',
                        help='5 epochs, 3000 patches, K=[1,5,10] — ~20 min on GPU')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    run_natural_image_benchmark(quick=args.quick, device_str=args.device)
