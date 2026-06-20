"""
exp_celeba.py
=============
KAN-EBM on CelebA: 64x64 face images, 3-channel RGB.

Reviewer demand: "Scale beyond CIFAR-10 to higher-resolution, more complex
distributions (faces)."

This script trains KAN-EBM, MLP-EBM, and FFN-DSM on CelebA (3-channel,
64x64) and evaluates PSNR at K in {1, 2, 5, 10, 20, 50} inference steps.
Batch size is set to 8 to fit within 10 GB VRAM with create_graph=True.

Run:
  python experiments/exp_celeba.py --device cuda
  python experiments/exp_celeba.py --device cuda --quick
  python experiments/exp_celeba.py --device cuda --epochs 50 --sigma 0.2

Output:
  outputs/celeba/results.json
  outputs/celeba/psnr_vs_k.pdf
  outputs/celeba/table_celeba.tex
"""

import argparse
import math
import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# KANLinear + KAN  (self-contained, bug-fixed)
# ──────────────────────────────────────────────────────────────────────────────

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_range=(-1, 1)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer('grid', grid)
        self.base_weight   = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_base  = scale_base; self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        nn.init.kaiming_uniform_(self.base_weight,   a=math.sqrt(5) * scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * scale_spline)
        with torch.no_grad():
            s = self.grid[0, spline_order].item()
            e = self.grid[0, -1 - spline_order].item()
            x = torch.linspace(s, e, grid_size + spline_order
                               ).expand(in_features, -1).t().contiguous()
            noise = (torch.rand(grid_size + spline_order, in_features, out_features)
                     - 0.5) * scale_noise / grid_size
            self.spline_weight.data.copy_(self.curve2coeff(x, noise))

    def b_splines(self, x):
        x     = x.unsqueeze(-1); grid = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
              + (grid[:, k+1:] - x)    / (grid[:, k+1:] - grid[:, 1:-k])    * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A   = self.b_splines(x).transpose(0, 1)
        B   = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(self.b_splines(x).view(x.size(0), -1),
                              self.scaled_spline_weight.view(self.out_features, -1))
        return base_out + spline_out


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(layers_hidden[i], layers_hidden[i+1],
                      grid_size=grid_size, spline_order=spline_order,
                      scale_noise=scale_noise, scale_base=scale_base)
            for i in range(len(layers_hidden) - 1)
        ])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class KANEnergyModel(nn.Module):
    """
    KAN-EBM for n_channels >= 1.  n_channels=3 for CelebA.
    Each of C channels gets filtered independently by n_filters filters,
    yielding C*n_filters = 48 features per pixel (for C=3, n_filters=16).
    """
    def __init__(self, n_filters=16, filter_size=5, kan_hidden=None, n_channels=3):
        super().__init__()
        if kan_hidden is None: kan_hidden = [48, 16]
        self.n_filters   = n_filters
        self.filter_size = filter_size
        self.n_channels  = n_channels
        self.filters     = nn.Parameter(
            0.01 * torch.randn(n_filters, n_channels, filter_size, filter_size))
        kan_in = n_channels * n_filters
        self.kan         = KAN([kan_in] + kan_hidden + [1], grid_size=5)
        self.log_precision = nn.Parameter(torch.zeros(n_filters))

    @property
    def precision(self): return F.softplus(self.log_precision)

    def extract_features(self, x):
        """x: (B, C, H, W) -> (B*H*W, C*n_filters)"""
        B, C, H, W = x.shape
        pad   = self.filter_size // 2
        feats = []
        for c in range(C):
            feats.append(F.conv2d(x[:, c:c+1], self.filters[:, c:c+1], padding=pad))
        feats = torch.cat(feats, dim=1)
        prec  = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = torch.tanh(feats * prec)
        return feats.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

    def energy(self, x):
        return self.kan(self.extract_features(x)).sum()

    def energy_grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.energy(xi), xi)[0].detach()

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        u = x_noisy.clone(); step = dt
        for _ in range(n_steps):
            u = (u - step * self.energy_grad(u).clamp(-1., 1.)).detach()
            step *= dt_decay
        return u

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi   = xn.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.energy(xi), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma ** 2 * grad, x_clean)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class MLPEnergyModel(nn.Module):
    """MLP-EBM baseline for CelebA. Global flattened input -> scalar energy."""
    def __init__(self, img_size=64, n_channels=3, hidden=256):
        super().__init__()
        D = img_size * img_size * n_channels
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def energy(self, x): return self.net(x.view(x.shape[0], -1)).squeeze(-1)

    def energy_grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.energy(xi).sum(), xi)[0].detach().clamp(-1, 1)

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        u = x_noisy.clone(); step = dt
        for _ in range(n_steps):
            u = (u - step * self.energy_grad(u)).detach()
            step *= dt_decay
        return u

    def loss(self, x_clean, sigma):
        xn  = x_clean + torch.randn_like(x_clean) * sigma
        xi  = xn.detach().requires_grad_(True)
        g   = torch.autograd.grad(self.energy(xi).sum(), xi, create_graph=True)[0]
        return F.mse_loss(xi - sigma ** 2 * g, x_clean)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class FFNDenoiser(nn.Module):
    """Feed-forward denoiser baseline (flat K scaling)."""
    def __init__(self, img_size=64, n_channels=3, hidden=512):
        super().__init__()
        D = img_size * img_size * n_channels
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, D))
        self.shape = (n_channels, img_size, img_size)

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1)).view(x.shape)

    def denoise(self, x, n_steps=1, **kw):
        return self.forward(x)

    def loss(self, x_clean, sigma):
        return F.mse_loss(self(x_clean + torch.randn_like(x_clean) * sigma), x_clean)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_celeba(n_train=None, n_test=None, img_size=64):
    """Load CelebA, center-cropped and resized to img_size, normalized to [-1, 1]."""
    import torchvision, torchvision.transforms as T

    tf = T.Compose([
        T.CenterCrop(140),
        T.Resize(img_size, antialias=True),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)

    try:
        ds_tr = torchvision.datasets.CelebA(data_dir, split='train',
                                             download=True, transform=tf)
        ds_te = torchvision.datasets.CelebA(data_dir, split='test',
                                             download=True, transform=tf)
    except Exception as e:
        print(f"\n[!] CelebA auto-download failed: {e}")
        print("[!] Please download CelebA manually:")
        print("    1. Go to https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("    2. Download img_align_celeba.zip, list_eval_partition.txt, identity_CelebA.txt")
        print("    3. Extract into data/celeba/")
        print("    4. Re-run this script")
        sys.exit(1)

    if n_train is not None:
        ds_tr = torch.utils.data.Subset(ds_tr, range(min(n_train, len(ds_tr))))
    if n_test is not None:
        ds_te = torch.utils.data.Subset(ds_te, range(min(n_test, len(ds_te))))

    # batch_size=8 to fit 64x64x3 images with create_graph in 10GB VRAM
    tr = torch.utils.data.DataLoader(ds_tr, batch_size=8, shuffle=True,
                                      num_workers=0, pin_memory=True, drop_last=True)
    te = torch.utils.data.DataLoader(ds_te, batch_size=8, shuffle=False,
                                      num_workers=0, pin_memory=True)
    return tr, te


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def psnr(clean, pred, data_range=2.0):
    pred = pred.clamp(-1, 1)
    mse  = F.mse_loss(pred, clean).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(data_range ** 2 / mse)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, loader, device, n_epochs, sigma, tag=''):
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    best_loss = float('inf')
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = model.loss(x, sigma)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss = model.loss(x, sigma)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item(); nb += 1
        avg = tot / nb
        sched.step()
        if avg < best_loss: best_loss = avg
        if ep % max(1, n_epochs // 5) == 0 or ep == 1:
            print(f'  [{tag:14s}] ep {ep:3d}/{n_epochs}  loss={avg:.5f}  best={best_loss:.5f}')

    return best_loss


@torch.no_grad()
def evaluate(model, loader, device, sigma, k_list, model_type='ebm'):
    model.eval()
    results = {k: [] for k in k_list}
    for batch in loader:
        xc = batch[0].to(device)
        xn = xc + torch.randn_like(xc) * sigma
        for k in k_list:
            xp = model.denoise(xn, n_steps=k)
            results[k].append(psnr(xc, xp))
    return {k: float(np.mean(v)) for k, v in results.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--quick',  action='store_true')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (default: 50 normal, 5 quick)')
    parser.add_argument('--sigma',  type=float, default=0.2,
                        help='Gaussian noise std in [-1,1] space (default: 0.2)')
    parser.add_argument('--n_filters', type=int, default=16)
    args = parser.parse_args()

    device   = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
               if args.device == 'auto' else torch.device(args.device)
    n_epochs = args.epochs if args.epochs else (5 if args.quick else 50)
    sigma    = args.sigma
    K_LIST   = [1, 2, 5, 10, 20] if args.quick else [1, 2, 5, 10, 20, 50]
    n_train  = 5000  if args.quick else 30000   # CelebA has ~162K train; use 30K for tractability
    n_test   = 500   if args.quick else 2000

    out_dir = ROOT / 'outputs' / 'celeba'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nDevice: {device}  |  epochs: {n_epochs}  |  sigma={sigma}')
    print(f'K_LIST: {K_LIST}  |  n_filters={args.n_filters}  |  img_size=64')

    # -- Load CelebA --
    print('\nLoading CelebA (64x64)...')
    tr, te = load_celeba(n_train=n_train, n_test=n_test, img_size=64)
    print(f'  Train: {len(tr.dataset)} images  |  Test: {len(te.dataset)} images')

    # -- Build models --
    kan_hidden = [args.n_filters * 3, 16]
    models = {
        'KAN-EBM':  (KANEnergyModel(n_filters=args.n_filters, filter_size=5,
                                     kan_hidden=kan_hidden, n_channels=3).to(device), 'ebm'),
        'MLP-EBM':  (MLPEnergyModel(img_size=64, n_channels=3, hidden=256).to(device), 'ebm'),
        'FFN-DSM':  (FFNDenoiser(img_size=64, n_channels=3, hidden=512).to(device),    'ffn'),
    }

    print('\nParameter counts:')
    for name, (m, _) in models.items():
        print(f'  {name:<12}: {m.n_params:>9,}')

    results     = {}
    train_times = {}
    t0_all = time.time()

    for name, (model, mtype) in models.items():
        print(f'\n{"~"*60}')
        print(f'  Training {name} ({n_epochs} epochs, sigma={sigma})')
        t0 = time.time()
        best = train_model(model, tr, device, n_epochs, sigma, tag=name)
        dt   = time.time() - t0
        train_times[name] = dt
        print(f'  Trained in {dt/60:.1f} min  (best loss={best:.5f})')

        print(f'  Evaluating at K = {K_LIST}')
        res = evaluate(model, te, device, sigma, K_LIST, model_type=mtype)
        results[name] = res
        for k, v in res.items():
            print(f'    K={k:>2}: {v:.3f} dB')

    # -- Summary --
    print(f'\n{"="*60}')
    print(f'CelebA 64x64 COMPARISON  (sigma={sigma})')
    print(f'{"="*60}')
    header = f'{"Model":<14}  ' + '  '.join(f'K={k:>2}' for k in K_LIST) + '  Gain'
    print(header); print('-' * len(header))
    for name, res in results.items():
        row  = '  '.join(f'{res[k]:>6.2f}' for k in K_LIST)
        gain = res[K_LIST[-1]] - res[K_LIST[0]]
        print(f'{name:<14}  {row}  {gain:+.2f}')

    total_time = time.time() - t0_all
    print(f'\nTotal runtime: {total_time/60:.1f} min')

    # -- Save JSON --
    out = {
        'dataset':     'CelebA',
        'img_size':    64,
        'sigma':       sigma,
        'K_LIST':      K_LIST,
        'n_epochs':    n_epochs,
        'n_filters':   args.n_filters,
        'kan_hidden':  kan_hidden,
        'n_params':    {n: m.n_params for n, (m, _) in models.items()},
        'train_times': train_times,
        'results':     {n: {str(k): v for k, v in r.items()} for n, r in results.items()},
        'device':      str(device),
        'total_time_s': total_time,
    }
    (out_dir / 'results.json').write_text(json.dumps(out, indent=2))

    # -- Plot --
    colors = {'KAN-EBM': '#3B5BDB', 'MLP-EBM': '#E64980', 'FFN-DSM': '#0CA678'}
    styles = {'KAN-EBM': 'o-',      'MLP-EBM': 's--',     'FFN-DSM': '^:'}

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        ks    = np.array(K_LIST)
        psnrs = np.array([res[k] for k in K_LIST])
        ax.plot(ks, psnrs, styles[name], color=colors[name], linewidth=2.0,
                markersize=6, label=f'{name} ({models[name][0].n_params//1000}K params)')

    ax.set_xlabel('Inference steps K', fontsize=13)
    ax.set_ylabel('PSNR (dB)', fontsize=13)
    ax.set_title(f'CelebA 64x64 Denoising: K-Step Scaling  (sigma={sigma})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    if max(K_LIST) >= 20:
        ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(out_dir / 'psnr_vs_k.pdf', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'psnr_vs_k.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- LaTeX table --
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{PSNR (dB) on CelebA 64$\times$64 ($\sigma=' + str(sigma) + r'$) '
        r'vs inference steps $K$. KAN-EBM uses $n_\text{filters}='
        + str(args.n_filters) + r'$ filters per channel (48 total KAN inputs for RGB). '
        r'MLP-EBM and FFN-DSM serve as ablation baselines.}',
        r'\label{tab:celeba}',
        r'\begin{tabular}{l' + 'c' * len(K_LIST) + 'c}',
        r'\toprule',
        r'Method & ' + ' & '.join(f'$K={k}$' for k in K_LIST) + r' & Gain \\',
        r'\midrule',
    ]
    for name, res in results.items():
        cells = ' & '.join(f'{res[k]:.2f}' for k in K_LIST)
        gain  = res[K_LIST[-1]] - res[K_LIST[0]]
        sign  = '+' if gain >= 0 else ''
        lines.append(f'{name} & {cells} & {sign}{gain:.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out_dir / 'table_celeba.tex').write_text('\n'.join(lines))

    print(f'\nAll outputs -> {out_dir}/')
    print(f'  outputs/celeba/results.json')
    print(f'  outputs/celeba/psnr_vs_k.pdf')
    print(f'  outputs/celeba/table_celeba.tex')


if __name__ == '__main__':
    main()
