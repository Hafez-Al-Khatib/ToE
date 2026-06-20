"""
run_full_benchmark.py
=====================
Sequential master runner for all KAN-EBM benchmark experiments.

Runs ONE experiment at a time and calls torch.cuda.empty_cache() between
each run so the GPU never OOMs.  All batch sizes are reduced to GPU-safe
values (32-64).

Experiments executed (in order):
  1. DDPM / DEQ comparison       -> outputs/diffusion_comparison/
  2. Multi-task SR & Inpainting  -> results/multitask/
  3. KAEM ablation baseline      -> (stdout summary)
  4. OOD detection (CIFAR/SVHN)  -> outputs/ood_cifar/

Run:
  py -3.12 experiments/run_full_benchmark.py --device cuda

Flags:
  --quick      5-epoch fast run for smoke-testing
  --device     cuda / cpu / auto (default: auto)
  --skip       Comma-separated list of experiment names to skip,
               e.g. --skip ddpm,kaem
"""

import sys
import gc
import time
import argparse
import traceback
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def psnr(clean, pred, data_range=2.0):
    pred = pred.clamp(-1, 1)
    mse  = F.mse_loss(pred, clean).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(data_range**2 / mse)


def flush_gpu():
    """Release all cached GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [GPU] cache flushed.\n")


def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def get_mnist(quick=False, batch_size=32):
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    n_train = 5000  if quick else 50000
    n_test  = 500   if quick else 2000
    ds_tr = torchvision.datasets.MNIST(ROOT/'data', train=True,  download=True, transform=tf)
    ds_te = torchvision.datasets.MNIST(ROOT/'data', train=False, download=True, transform=tf)
    ds_tr = torch.utils.data.Subset(ds_tr, range(n_train))
    ds_te = torch.utils.data.Subset(ds_te, range(n_test))
    tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    te = torch.utils.data.DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  MNIST: {n_train} train / {n_test} test  |  batch_size={batch_size}")
    return tr, te


def generic_train(model, loader, device, n_epochs, sigma, name=''):
    opt    = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            opt.zero_grad()
            ctx = torch.amp.autocast('cuda') if scaler else torch.no_grad.__class__()  # plain context
            if scaler:
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
        sched.step()
        if ep % max(1, n_epochs // 4) == 0:
            print(f"    [{name:15s}] ep {ep:3d}/{n_epochs}  loss={tot/nb:.5f}")


@torch.no_grad()
def generic_eval(model, loader, device, sigma, k_list, denoise_fn):
    model.eval()
    res = {k: [] for k in k_list}
    for batch in loader:
        xc = batch[0].to(device, non_blocking=True)
        xn = xc + torch.randn_like(xc) * sigma
        for k in k_list:
            xp = denoise_fn(model, xn, k)
            res[k].append(psnr(xc, xp))
    return {k: float(np.mean(v)) for k, v in res.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: DDPM / DEQ comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_ddpm_comparison(device, quick, n_epochs):
    section("EXP 1 / 4 : DDPM & DEQ vs KAN-EBM")

    # --- local model definitions (copied from exp_diffusion_comparison.py) ---
    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.GroupNorm(4, dim), nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(4, dim), nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        def forward(self, x): return x + self.net(x)

    class ScoreNetUNet(nn.Module):
        def __init__(self, in_ch=1, base_ch=16, n_sigma=10):
            super().__init__()
            self.sigma_embed = nn.Embedding(n_sigma, base_ch)
            self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), ResBlock(base_ch))
            self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), ResBlock(base_ch*2))
            self.mid  = ResBlock(base_ch*2)
            self.dec2 = nn.Sequential(nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2), ResBlock(base_ch))
            self.out  = nn.Conv2d(base_ch*2, in_ch, 1)
            self.n_sigma = n_sigma
        def forward(self, x, sigma_idx):
            e  = self.sigma_embed(sigma_idx).view(-1, 16, 1, 1)
            h1 = self.enc1(x) + e
            h2 = self.mid(self.enc2(h1))
            h  = self.dec2(h2)
            return self.out(torch.cat([h, h1], dim=1))
        @property
        def n_params(self): return sum(p.numel() for p in self.parameters())

    class DDPMDenoiser(nn.Module):
        def __init__(self, in_ch=1, base_ch=16):
            super().__init__()
            sigmas = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
            self.register_buffer('sigma_values', torch.tensor(sigmas))
            self.score_net = ScoreNetUNet(in_ch, base_ch, n_sigma=len(sigmas))
        def _sidx(self, sigma): return (self.sigma_values - sigma).abs().argmin().expand(1)
        def denoise(self, x_noisy, sigma, n_steps=10, step_size=None):
            step_size = step_size or sigma**2 / 10.0
            u = x_noisy.clone()
            sidx = self._sidx(sigma).to(x_noisy.device).expand(x_noisy.shape[0])
            with torch.no_grad():
                for _ in range(n_steps):
                    score = self.score_net(u, sidx)
                    u = (u + (step_size/2)*score*sigma**2 + torch.randn_like(u)*math.sqrt(step_size)*0.1).clamp(-1,1)
            return u
        def loss(self, x_clean, sigma):
            noise   = torch.randn_like(x_clean) * sigma
            x_noisy = x_clean + noise
            sidx    = self._sidx(sigma).to(x_clean.device).expand(x_clean.shape[0])
            pred    = self.score_net(x_noisy, sidx)
            true_score = -noise / sigma**2
            return sigma**2 * F.mse_loss(pred, true_score)
        @property
        def n_params(self): return sum(p.numel() for p in self.parameters())

    class DEQDenoiser(nn.Module):
        def __init__(self, obs_dim=784, hidden=64):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(obs_dim*2, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),    nn.SiLU(),
                nn.Linear(hidden, obs_dim),
            )
        def denoise(self, x_noisy, n_steps=10):
            B = x_noisy.shape[0]
            xf = x_noisy.view(B, -1)
            z  = xf.clone()
            for _ in range(n_steps):
                z = self.f(torch.cat([z, xf], -1)) * 0.1 + xf
            return z.view(x_noisy.shape)
        def loss(self, x_clean, sigma):
            noise   = torch.randn_like(x_clean) * sigma
            x_noisy = x_clean + noise
            z_star  = self.denoise(x_noisy, n_steps=5)
            return F.mse_loss(z_star, x_clean)
        @property
        def n_params(self): return sum(p.numel() for p in self.parameters())

    class KANEBMWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            from predictive_coding_field import PredictiveCodingField
            self.field = PredictiveCodingField(n_channels=1, n_filters=16, filter_size=5,
                                               kan_hidden=[32], height=28, width=28)
        def denoise(self, x_noisy, n_steps=10):
            u = x_noisy.clone()
            for i in range(n_steps):
                with torch.enable_grad():
                    ui = u.detach().requires_grad_(True)
                    g  = torch.autograd.grad(self.field.compute_energy(ui).sum(), ui)[0].detach().clamp(-1,1)
                u = u.detach() - 0.05 * (0.97**i) * g
            return u
        def loss(self, x_clean, sigma):
            return self.field.denoising_loss(x_clean, x_clean + torch.randn_like(x_clean)*sigma)
        @property
        def n_params(self): return sum(p.numel() for p in self.parameters())

    sigma  = 0.2
    K_LIST = [1, 2, 5, 10, 20]
    tr, te = get_mnist(quick=quick, batch_size=32)

    models = {
        'KAN-EBM':    (KANEBMWrapper().to(device),            lambda m,x,k: m.denoise(x, n_steps=k)),
        'DDPM-score': (DDPMDenoiser(in_ch=1, base_ch=16).to(device), lambda m,x,k: m.denoise(x, sigma=sigma, n_steps=k)),
        'DEQ':        (DEQDenoiser(obs_dim=784, hidden=64).to(device), lambda m,x,k: m.denoise(x, n_steps=k)),
    }

    print("\nParameter counts:")
    for name, (m, _) in models.items():
        print(f"  {name:<15}: {m.n_params:>8,}")

    results = {}
    for name, (model, denoise_fn) in models.items():
        print(f"\n  Training {name} ({n_epochs} epochs) ...")
        generic_train(model, tr, device, n_epochs, sigma, name=name)
        print(f"  Evaluating {name} ...")
        res = generic_eval(model, te, device, sigma, K_LIST, denoise_fn)
        results[name] = res
        for k, v in res.items():
            print(f"    K={k:>2}: {v:.2f} dB")
        # free this model right away
        del model
        flush_gpu()

    out = ROOT / 'outputs' / 'diffusion_comparison'
    out.mkdir(parents=True, exist_ok=True)
    import json
    with open(out / 'comparison_results.json', 'w') as f:
        json.dump({'results': results, 'K_LIST': K_LIST, 'sigma': sigma}, f, indent=2)

    print(f"\n  Results summary (sigma={sigma}, MNIST):")
    print(f"  {'Model':<15}  " + "  ".join(f"K={k:>2}" for k in K_LIST))
    print("  " + "-" * 53)
    for name, res in results.items():
        row = "  ".join(f"{res[k]:>6.2f}" for k in K_LIST)
        print(f"  {name:<15}  {row}")
    print(f"\n  Saved to {out}/")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Multi-task (SR, Inpainting)
# ─────────────────────────────────────────────────────────────────────────────

def run_multitask(device, quick, n_epochs):
    section("EXP 2 / 4 : Multi-Task Restoration (SR & Inpainting)")

    # delegate to the existing script's main functions
    sys.argv = ['exp_multitask.py', '--device', str(device)]
    if quick:
        sys.argv.append('--quick')
    try:
        import importlib.util, importlib
        spec = importlib.util.spec_from_file_location(
            "exp_multitask",
            ROOT / 'experiments' / 'exp_multitask.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # patch batch sizes to GPU-safe values
        mod.train_kan.__defaults__ = (n_epochs, 0.2, 32, True)   # batch_size=32
        mod.train_ffn_task.__defaults__ = (10, 0.2, 64)          # batch_size=64
        mod.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"  [multitask] ERROR: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: KAEM ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_kaem_ablation(device, quick, n_epochs):
    section("EXP 3 / 4 : KAEM Ablation (Naive KAN vs Our KAN-EBM)")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp_kaem_baseline",
            ROOT / 'experiments' / 'exp_kaem_baseline.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_ablation(device_str=str(device), quick=quick)
    except Exception as e:
        print(f"  [kaem_ablation] ERROR: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: OOD detection (CIFAR vs SVHN)
# ─────────────────────────────────────────────────────────────────────────────

def run_ood_cifar(device, quick):
    section("EXP 4 / 4 : OOD Detection (CIFAR-10 vs SVHN)")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp_ood_cifar",
            ROOT / 'experiments' / 'exp_ood_cifar.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # The ood script reads sys.argv, so set it up
        sys.argv = ['exp_ood_cifar.py', '--device', str(device)]
        if quick:
            sys.argv.append('--quick')
        mod.main() if hasattr(mod, 'main') else None
    except SystemExit:
        pass
    except Exception as e:
        print(f"  [ood_cifar] ERROR: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Sequential KAN-EBM full benchmark')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--quick',  action='store_true', help='Fast 5-epoch smoke test')
    parser.add_argument('--skip',   default='', help='Comma-separated exps to skip: ddpm,multitask,kaem,ood')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    n_epochs = 5 if args.quick else 25   # reduced from 30 to 25 to ease pressure

    skip = {s.strip().lower() for s in args.skip.split(',') if s.strip()}

    print(f"\n{'#'*60}")
    print(f"  KAN-EBM Full Benchmark")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {n_epochs}")
    print(f"  Quick   : {args.quick}")
    print(f"  Skip    : {skip or 'none'}")
    print(f"{'#'*60}\n")

    t_start = time.time()

    if 'ddpm' not in skip:
        try:
            run_ddpm_comparison(device, args.quick, n_epochs)
        except Exception as e:
            print(f"[DDPM exp] FAILED: {e}")
            traceback.print_exc()
        flush_gpu()

    if 'multitask' not in skip:
        try:
            run_multitask(device, args.quick, n_epochs)
        except Exception as e:
            print(f"[Multitask exp] FAILED: {e}")
            traceback.print_exc()
        flush_gpu()

    if 'kaem' not in skip:
        try:
            run_kaem_ablation(device, args.quick, n_epochs)
        except Exception as e:
            print(f"[KAEM exp] FAILED: {e}")
            traceback.print_exc()
        flush_gpu()

    if 'ood' not in skip:
        try:
            run_ood_cifar(device, args.quick)
        except Exception as e:
            print(f"[OOD exp] FAILED: {e}")
            traceback.print_exc()
        flush_gpu()

    elapsed = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  All experiments done in {elapsed/60:.1f} min")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
