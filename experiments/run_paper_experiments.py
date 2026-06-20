"""
run_paper_experiments.py
========================
Self-contained experiment runner for paper results.
Run with:  py -3.12 experiments/run_paper_experiments.py [--quick] [--device cuda]

Produces three families of results:

  Table 1  —  PSNR / SSIM denoising vs FFN and MLP-EBM baselines (MNIST, FashionMNIST)
  Table 2  —  Mathematical chain verification (cosine similarities, H-drift ratio)
  Figure 1 —  PSNR vs K inference steps  (the test-time compute scaling curve)

All results written to:  outputs/paper_results/
  results.json           — machine-readable numbers
  table1_denoising.txt   — formatted LaTeX table
  table2_math.txt        — formatted validation table
  psnr_vs_k.png          — Figure 1

Requirements:  torch, torchvision, numpy, matplotlib
"""

import argparse
import json
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Metrics
# =============================================================================

def psnr(clean: torch.Tensor, pred: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(pred, clean).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(data_range ** 2 / mse)


def ssim(clean: torch.Tensor, pred: torch.Tensor,
         data_range: float = 2.0, window_size: int = 7) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = g.ger(g).unsqueeze(0).unsqueeze(0).to(clean.device)

    ssim_vals = []
    C = clean.shape[1]
    for c in range(C):
        x = clean[:, c:c+1]
        y = pred[:, c:c+1]
        mu_x = F.conv2d(x, kernel, padding=window_size // 2)
        mu_y = F.conv2d(y, kernel, padding=window_size // 2)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sig_x2 = F.conv2d(x ** 2, kernel, padding=window_size // 2) - mu_x2
        sig_y2 = F.conv2d(y ** 2, kernel, padding=window_size // 2) - mu_y2
        sig_xy = F.conv2d(x * y, kernel, padding=window_size // 2) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        ssim_vals.append((num / den).mean().item())
    return float(np.mean(ssim_vals))


# =============================================================================
# Models
# =============================================================================

class FFNDenoiser(nn.Module):
    """Pure feedforward baseline — no iterative inference."""
    def __init__(self, obs_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, x):
        shape = x.shape
        return self.net(x.view(x.shape[0], -1)).view(shape)

    def denoise(self, x_noisy, n_steps=1):
        # FFN ignores n_steps — single forward pass regardless
        return self.forward(x_noisy)

    def loss(self, x_clean, sigma):
        noise = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        pred = self.forward(x_noisy)
        return F.mse_loss(pred, x_clean)


class MLPEnergyModel(nn.Module):
    """EBM with MLP energy — same iterative inference as KAN-EBM, different energy fn."""
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def energy(self, x):
        return self.energy_net(x.view(x.shape[0], -1)).squeeze(-1)

    def energy_grad(self, x):
        # enable_grad so this works inside @torch.no_grad() at eval time
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            E = self.energy(x_in).sum()
            return torch.autograd.grad(E, x_in)[0].detach()

    def denoise(self, x_noisy, n_steps=10, dt=0.1):
        u = x_noisy.clone()
        for _ in range(n_steps):
            u = (u - dt * self.energy_grad(u)).detach()
        return u

    def loss(self, x_clean, sigma):
        """Denoising score matching via score-implicit MSE.

        We need create_graph=True so that gradients flow back through
        autograd.grad() into the energy network weights.
        """
        noise = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        x_in = x_noisy.detach().requires_grad_(True)
        E = self.energy(x_in).sum()
        grad = torch.autograd.grad(E, x_in, create_graph=True)[0]
        pred = x_in - sigma ** 2 * grad
        return F.mse_loss(pred, x_clean)


class KANEnergyModel(nn.Module):
    """
    KAN-parameterised energy model with spatial filter bank.
    Uses the PredictiveCodingField from src/.
    Wrapped here for a clean training interface.
    """
    def __init__(self, n_channels: int, height: int, width: int,
                 n_filters: int = 16, filter_size: int = 5,
                 kan_hidden: List[int] = None):
        super().__init__()
        if kan_hidden is None:
            kan_hidden = [64, 32]
        from predictive_coding_field import PredictiveCodingField
        self.field = PredictiveCodingField(
            n_channels=n_channels,
            n_filters=n_filters,
            filter_size=filter_size,
            kan_hidden=kan_hidden,
            height=height,
            width=width,
        )
        self.obs_dim = n_channels * height * width

    def denoise(self, x_noisy, n_steps=10, dt=0.05):
        """Gradient descent on the KAN energy. enable_grad works under no_grad context."""
        u = x_noisy.clone()
        for _ in range(n_steps):
            with torch.enable_grad():
                u_in = u.detach().requires_grad_(True)
                E = self.field.compute_energy(u_in).sum()
                grad = torch.autograd.grad(E, u_in)[0].detach()
            # Clip to prevent divergence from under-trained energy
            grad = grad.clamp(-1.0, 1.0)
            u = (u.detach() - dt * grad)
        return u

    def loss(self, x_clean, sigma):
        noise = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        return self.field.denoising_loss(x_clean, x_noisy)


# =============================================================================
# Data
# =============================================================================

def get_mnist(quick: bool = False, fashion: bool = False) -> Tuple:
    try:
        import torchvision
        import torchvision.transforms as T
        name = 'FashionMNIST' if fashion else 'MNIST'
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        data_dir = ROOT / 'data'
        cls = torchvision.datasets.FashionMNIST if fashion else torchvision.datasets.MNIST
        train_ds = cls(data_dir, train=True,  download=True, transform=transform)
        test_ds  = cls(data_dir, train=False, download=True, transform=transform)
        n_train = 5000 if quick else len(train_ds)
        n_test  = 500  if quick else 2000
        train_ds = torch.utils.data.Subset(train_ds, range(n_train))
        test_ds  = torch.utils.data.Subset(test_ds,  range(n_test))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
        test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)
        print(f"  {name}: {n_train} train / {n_test} test")
        return train_loader, test_loader, name
    except Exception as e:
        # Synthetic fallback: random 28×28
        print(f"  torchvision not available ({e}), using synthetic 28×28 data")
        n_train = 2000 if quick else 10000
        n_test  = 200  if quick else 1000
        train_x = torch.randn(n_train, 1, 28, 28)
        test_x  = torch.randn(n_test,  1, 28, 28)
        train_ds = torch.utils.data.TensorDataset(train_x, torch.zeros(n_train, dtype=torch.long))
        test_ds  = torch.utils.data.TensorDataset(test_x,  torch.zeros(n_test,  dtype=torch.long))
        return (torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True),
                torch.utils.data.DataLoader(test_ds,  batch_size=128),
                'Synthetic28x28')


# =============================================================================
# Training loop
# =============================================================================

def train_model(model: nn.Module, train_loader, device, n_epochs: int,
                sigma: float, lr: float = 3e-4, name: str = '') -> List[float]:
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    # AMP scaler — halves VRAM and speeds up KAN's create_graph autograd
    use_amp = (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    losses  = []
    for ep in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device)
            opt.zero_grad()
            loss = model.loss(x, sigma)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if ep % max(1, n_epochs // 5) == 0:
            print(f"    [{name}] epoch {ep:3d}/{n_epochs}  loss={avg:.5f}")
    return losses


@torch.no_grad()
def evaluate_model(model, test_loader, device, sigma: float,
                   k_list: List[int]) -> Dict:
    model.eval()
    results = {k: {'psnr': [], 'ssim': []} for k in k_list}

    for batch in test_loader:
        x_clean = batch[0].to(device)
        x_noisy = x_clean + torch.randn_like(x_clean) * sigma
        for k in k_list:
            try:
                x_pred = model.denoise(x_noisy, n_steps=k)
            except TypeError:
                x_pred = model.denoise(x_noisy)
            x_pred = x_pred.clamp(-1, 1)
            results[k]['psnr'].append(psnr(x_clean, x_pred))
            results[k]['ssim'].append(ssim(x_clean, x_pred))

    return {k: {'psnr': float(np.mean(v['psnr'])), 'ssim': float(np.mean(v['ssim']))}
            for k, v in results.items()}


# =============================================================================
# Experiment 1: Denoising Benchmark  (Table 1)
# =============================================================================

def run_denoising_benchmark(device, quick: bool = False,
                            mnist_only: bool = False,
                            sigma_filter: float = None) -> Dict:
    print("\n" + "="*70)
    print("EXPERIMENT 1: Denoising Benchmark (Table 1)")
    print("="*70)

    n_epochs  = 5 if quick else 30
    sigma_list = [sigma_filter] if sigma_filter else [0.1, 0.2, 0.3]
    K_EVAL    = [1, 5, 10]
    datasets  = [False] if mnist_only else [False, True]   # False=MNIST, True=Fashion
    all_results = {}

    for fashion in datasets:
        train_loader, test_loader, dataset_name = get_mnist(quick=quick, fashion=fashion)
        all_results[dataset_name] = {}

        for sigma in sigma_list:
            print(f"\n  Dataset={dataset_name}  σ={sigma}")
            sigma_key = f"sigma_{sigma}"
            all_results[dataset_name][sigma_key] = {}

            # ── FFN Baseline ───────────────────────────────────────────────
            print("  Training FFN baseline...")
            ffn = FFNDenoiser(obs_dim=784, hidden=512).to(device)
            train_model(ffn, train_loader, device, n_epochs, sigma, name='FFN')
            ffn_res = evaluate_model(ffn, test_loader, device, sigma, K_EVAL)
            all_results[dataset_name][sigma_key]['FFN'] = ffn_res
            print(f"    FFN  PSNR@K=1: {ffn_res[1]['psnr']:.2f} dB  SSIM: {ffn_res[1]['ssim']:.4f}")

            # ── MLP-EBM Baseline ───────────────────────────────────────────
            print("  Training MLP-EBM baseline...")
            mlp_ebm = MLPEnergyModel(obs_dim=784, hidden=256).to(device)
            train_model(mlp_ebm, train_loader, device, n_epochs, sigma, name='MLP-EBM')
            mlp_res = evaluate_model(mlp_ebm, test_loader, device, sigma, K_EVAL)
            all_results[dataset_name][sigma_key]['MLP-EBM'] = mlp_res
            print(f"    MLP-EBM  PSNR@K=10: {mlp_res[10]['psnr']:.2f} dB  SSIM: {mlp_res[10]['ssim']:.4f}")

            # ── KAN-EBM ────────────────────────────────────────────────────
            print("  Training KAN-EBM (PredictiveCodingField)...")
            # Smaller KAN hidden dims to keep create_graph autograd tractable
            kan_ebm = KANEnergyModel(n_channels=1, height=28, width=28,
                                     n_filters=16, filter_size=5,
                                     kan_hidden=[32]).to(device)
            train_model(kan_ebm, train_loader, device, n_epochs, sigma, name='KAN-EBM')
            kan_res = evaluate_model(kan_ebm, test_loader, device, sigma, K_EVAL)
            all_results[dataset_name][sigma_key]['KAN-EBM'] = kan_res
            print(f"    KAN-EBM  PSNR@K=10: {kan_res[10]['psnr']:.2f} dB  SSIM: {kan_res[10]['ssim']:.4f}")

    return all_results


# =============================================================================
# Experiment 2: Mathematical Chain Verification  (Table 2)
# =============================================================================

def run_math_verification(device) -> Dict:
    print("\n" + "="*70)
    print("EXPERIMENT 2: Mathematical Chain Verification (Table 2)")
    print("="*70)

    results = {}

    # ── 2A: Allen-Cahn ≡ Score ≡ Prediction Error ─────────────────────────
    print("\n  [2A] Allen-Cahn ↔ Score ↔ PC equivalence (Gaussian energy)")
    torch.manual_seed(SEED)
    D = 128
    mu_data = torch.zeros(D, device=device)
    sigma_data = 0.3
    o = (mu_data + sigma_data * torch.randn(D, device=device)).unsqueeze(0)
    sigma2 = sigma_data ** 2

    x_var = o.clone().requires_grad_(True)
    E = ((x_var - mu_data) ** 2).sum() / (2 * sigma2)
    grad_ac = torch.autograd.grad(E, x_var)[0]

    score = -(o - mu_data) / sigma2
    pred_error = (1.0 / sigma_data) * (o - mu_data)

    def cos_sim(a, b):
        return F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()

    sim_ac_score = cos_sim(grad_ac, -score)
    sim_ac_pc    = cos_sim(grad_ac, pred_error)
    sim_score_pc = cos_sim(-score, pred_error)

    print(f"    cos(Allen-Cahn, Score)  = {sim_ac_score:.8f}  (expect 1.0)")
    print(f"    cos(Allen-Cahn, PC-err) = {sim_ac_pc:.8f}  (expect 1.0)")
    print(f"    cos(Score, PC-err)      = {sim_score_pc:.8f}  (expect 1.0)")
    results['chain_verification'] = {
        'ac_vs_score': sim_ac_score,
        'ac_vs_pc':    sim_ac_pc,
        'score_vs_pc': sim_score_pc,
        'all_pass':    all(s > 0.9999 for s in [sim_ac_score, sim_ac_pc, sim_score_pc]),
    }

    # ── 2B: Symplectic vs Naive Euler Hamiltonian drift ────────────────────
    # Use a hand-crafted isotropic quadratic potential V(z) = ||z||²/2
    # so H(z,p) = ||p||²/2 + ||z||²/2 (harmonic oscillator).
    # Analytic: naive Euler injects energy O(dt²) per step → grows linearly.
    # Symplectic Euler preserves a shadow Hamiltonian → bounded drift.
    # dt=0.3 makes the difference visible in 200 steps.
    print("\n  [2B] Symplectic vs Naive Euler H-conservation (200 steps, quadratic V)")
    torch.manual_seed(SEED)
    D  = 32
    B  = 64
    dt = 0.3
    n_steps = 200

    z0 = torch.randn(B, D, device=device)
    p0 = torch.randn(B, D, device=device)

    def quad_potential(z):
        return 0.5 * (z ** 2).sum(-1)          # (B,)

    def quad_hamiltonian(z, p):
        return 0.5 * (p ** 2).sum(-1) + quad_potential(z)   # (B,)

    H0_val = quad_hamiltonian(z0, p0).mean().item()

    # Symplectic Euler: p_new = p - dt*∇V(z),  z_new = z + dt*p_new  (uses updated p)
    z_s, p_s = z0.clone(), p0.clone()
    H_symp = [H0_val]
    for _ in range(n_steps):
        p_s = p_s - dt * z_s                   # ∇V(z) = z for quadratic
        z_s = z_s + dt * p_s                   # updated p → symplectic
        H_symp.append(quad_hamiltonian(z_s, p_s).mean().item())

    # Naive Euler: p_new = p - dt*∇V(z),  z_new = z + dt*p_OLD  (uses old p → energy leak)
    z_n, p_n = z0.clone(), p0.clone()
    H_naive = [H0_val]
    for _ in range(n_steps):
        p_new = p_n - dt * z_n
        z_n   = z_n + dt * p_n                 # old p → naive
        p_n   = p_new
        H_naive.append(quad_hamiltonian(z_n, p_n).mean().item())

    drift_symp  = max(abs(h - H0_val) for h in H_symp)
    drift_naive = max(abs(h - H0_val) for h in H_naive)
    ratio = drift_naive / max(drift_symp, 1e-10)

    print(f"    H0                  = {H0_val:.4f}")
    print(f"    Symplectic max |ΔH| = {drift_symp:.6f}  (bounded oscillation)")
    print(f"    Naive      max |ΔH| = {drift_naive:.6f}  (growing injection)")
    print(f"    Ratio (naive/symp)  = {ratio:.1f}×  (expect >>10)")
    results['hamiltonian_conservation'] = {
        'H0': H0_val,
        'symplectic_max_drift': drift_symp,
        'naive_max_drift':      drift_naive,
        'naive_over_symplectic': ratio,
        'symplectic_wins':      ratio > 5,
    }

    # ── 2C: PC free energy monotone decrease ──────────────────────────────
    print("\n  [2C] PC inference — free energy monotone decrease")
    from hierarchical_pc_kan import HierarchicalPCKAN
    pc = HierarchicalPCKAN(dims=[64, 32, 16], hidden_dims=[32],
                           inference_lr=0.05, n_inference_steps=30).to(device)
    torch.manual_seed(SEED)
    x_obs = torch.randn(16, 64, device=device)
    _, _, info = pc.inference(x_obs, n_steps=30, return_trajectory=True)
    traj = info.get('F_trajectory', [])
    if len(traj) > 1:
        n_decrease = sum(1 for i in range(1, len(traj)) if traj[i] < traj[i-1])
        pct_decrease = n_decrease / (len(traj) - 1) * 100
        print(f"    F decreasing steps: {n_decrease}/{len(traj)-1}  ({pct_decrease:.1f}%)")
        results['pc_convergence'] = {
            'steps':       len(traj) - 1,
            'n_decrease':  n_decrease,
            'pct_decrease': pct_decrease,
            'monotone':    pct_decrease > 90,
        }
    else:
        print("    Warning: no trajectory returned")
        results['pc_convergence'] = {'monotone': None}

    return results


# =============================================================================
# Experiment 3: PSNR vs K  (Figure 1)
# =============================================================================

def run_psnr_vs_k(device, quick: bool = False) -> Dict:
    print("\n" + "="*70)
    print("EXPERIMENT 3: PSNR vs Inference Steps K  (Figure 1)")
    print("="*70)

    n_epochs = 5 if quick else 25
    sigma = 0.2
    K_LIST = [0, 1, 2, 5, 10, 20] if not quick else [0, 1, 5, 10]
    train_loader, test_loader, _ = get_mnist(quick=quick, fashion=False)

    print(f"\n  σ={sigma}, {n_epochs} epochs")

    ffn = FFNDenoiser(obs_dim=784, hidden=512).to(device)
    train_model(ffn, train_loader, device, n_epochs, sigma, name='FFN')

    mlp_ebm = MLPEnergyModel(obs_dim=784, hidden=256).to(device)
    train_model(mlp_ebm, train_loader, device, n_epochs, sigma, name='MLP-EBM')

    kan_ebm = KANEnergyModel(n_channels=1, height=28, width=28,
                              n_filters=16, filter_size=5, kan_hidden=[64, 32]).to(device)
    train_model(kan_ebm, train_loader, device, n_epochs, sigma, name='KAN-EBM')

    print("\n  Evaluating across K values...")
    ffn_res     = evaluate_model(ffn,     test_loader, device, sigma, K_LIST)
    mlp_res     = evaluate_model(mlp_ebm, test_loader, device, sigma, K_LIST)
    kan_res     = evaluate_model(kan_ebm, test_loader, device, sigma, K_LIST)

    print(f"\n  {'K':>4}  {'FFN':>8}  {'MLP-EBM':>8}  {'KAN-EBM':>8}")
    print(f"  {'':-<4}  {'':-<8}  {'':-<8}  {'':-<8}")
    for k in K_LIST:
        ffn_p = ffn_res[k]['psnr']
        mlp_p = mlp_res[k]['psnr']
        kan_p = kan_res[k]['psnr']
        marker = '  ←' if kan_p == max(kan_res[kk]['psnr'] for kk in K_LIST) else ''
        print(f"  {k:>4}  {ffn_p:>8.2f}  {mlp_p:>8.2f}  {kan_p:>8.2f}{marker}")

    return {'FFN': ffn_res, 'MLP-EBM': mlp_res, 'KAN-EBM': kan_res, 'K_list': K_LIST}


# =============================================================================
# Output formatting
# =============================================================================

def save_table1(bench_results: Dict, out_dir: Path):
    lines = ["\\begin{table}[t]",
             "\\centering",
             "\\caption{Denoising PSNR (dB) / SSIM on MNIST and FashionMNIST. "
             "EBM models are evaluated at K=1 and K=10 inference steps.}",
             "\\label{tab:denoising}",
             "\\begin{tabular}{llccccccc}",
             "\\toprule",
             " & & \\multicolumn{3}{c}{K=1} & \\multicolumn{3}{c}{K=10} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Dataset & $\\sigma$ & FFN & MLP-EBM & KAN-EBM & FFN & MLP-EBM & KAN-EBM \\\\",
             "\\midrule"]

    for dataset, sigma_dict in bench_results.items():
        for sigma_key, model_dict in sigma_dict.items():
            sigma = sigma_key.replace('sigma_', '')
            row_parts = [dataset, f"{sigma}"]
            for k in [1, 10]:
                for model_name in ['FFN', 'MLP-EBM', 'KAN-EBM']:
                    if model_name in model_dict and k in model_dict[model_name]:
                        p = model_dict[model_name][k]['psnr']
                        s = model_dict[model_name][k]['ssim']
                        row_parts.append(f"{p:.2f}/{s:.3f}")
                    else:
                        row_parts.append("—")
            lines.append(" & ".join(row_parts) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / "table1_denoising.txt").write_text("\n".join(lines))
    print(f"\n  Saved: {out_dir}/table1_denoising.txt")


def save_table2(math_results: Dict, out_dir: Path):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Mathematical validation results.}",
        "\\label{tab:math}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Experiment & Value & Pass \\\\",
        "\\midrule",
    ]
    cr = math_results.get('chain_verification', {})
    lines.append(f"cos(Allen-Cahn, Score)  & {cr.get('ac_vs_score', 'N/A'):.8f} & {'\\checkmark' if cr.get('all_pass') else '\\times'} \\\\")
    lines.append(f"cos(Allen-Cahn, PC-err) & {cr.get('ac_vs_pc',    'N/A'):.8f} & {'\\checkmark' if cr.get('all_pass') else '\\times'} \\\\")
    lines.append(f"cos(Score, PC-err)      & {cr.get('score_vs_pc', 'N/A'):.8f} & {'\\checkmark' if cr.get('all_pass') else '\\times'} \\\\")
    hc = math_results.get('hamiltonian_conservation', {})
    lines.append("\\midrule")
    lines.append(f"Symplectic max $|\\Delta H|$ & {hc.get('symplectic_max_drift', 'N/A'):.6f} & — \\\\")
    lines.append(f"Naive max $|\\Delta H|$       & {hc.get('naive_max_drift', 'N/A'):.6f} & — \\\\")
    lines.append(f"Ratio naive/symplectic        & {hc.get('naive_over_symplectic', 'N/A'):.1f}$\\times$ & {'\\checkmark' if hc.get('symplectic_wins') else '\\times'} \\\\")
    pc = math_results.get('pc_convergence', {})
    lines.append("\\midrule")
    pct = pc.get('pct_decrease', None)
    lines.append(f"PC free energy decrease & {pct:.1f}\\% & {'\\checkmark' if pc.get('monotone') else '\\times'} \\\\" if pct is not None else "PC free energy decrease & N/A & — \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / "table2_math.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir}/table2_math.txt")


def save_figure1(scaling_results: Dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        K_LIST = scaling_results['K_list']
        ffn_p  = [scaling_results['FFN'][k]['psnr']     for k in K_LIST]
        mlp_p  = [scaling_results['MLP-EBM'][k]['psnr'] for k in K_LIST]
        kan_p  = [scaling_results['KAN-EBM'][k]['psnr'] for k in K_LIST]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(K_LIST, ffn_p,  'k--o', label='FFN (no iteration)',   markersize=5)
        ax.plot(K_LIST, mlp_p,  'b-^',  label='MLP-EBM',              markersize=5)
        ax.plot(K_LIST, kan_p,  'r-s',  label='KAN-EBM (ours)',        markersize=6, linewidth=2)
        ax.set_xlabel('Inference steps K', fontsize=12)
        ax.set_ylabel('PSNR (dB)',         fontsize=12)
        ax.set_title('Test-time compute scaling: PSNR vs K\n(MNIST, σ=0.2)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(K_LIST)
        fig.tight_layout()
        out_path = out_dir / 'psnr_vs_k.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  matplotlib not available — skipping Figure 1")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TIM Paper Experiments')
    parser.add_argument('--quick',      action='store_true',
                        help='Reduce epochs/data for a fast smoke test (~5 min)')
    parser.add_argument('--device',     default='auto',
                        help='cuda | cpu | auto (default: auto-detect)')
    parser.add_argument('--exp',        choices=['all', 'bench', 'math', 'scaling'],
                        default='all',  help='Which experiment to run')
    parser.add_argument('--mnist-only', action='store_true',
                        help='bench: skip FashionMNIST, run MNIST only (~half the time)')
    parser.add_argument('--sigma',      type=float, default=None,
                        help='bench: run a single sigma value (0.1 | 0.2 | 0.3)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Quick mode: {args.quick}")

    out_dir = ROOT / 'outputs' / 'paper_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {'device': str(device), 'quick': args.quick}
    t_start = time.time()

    # ── Run selected experiments ───────────────────────────────────────────
    if args.exp in ('all', 'bench'):
        bench = run_denoising_benchmark(device, quick=args.quick,
                                        mnist_only=args.mnist_only,
                                        sigma_filter=args.sigma)
        all_results['denoising_benchmark'] = bench
        save_table1(bench, out_dir)

    if args.exp in ('all', 'math'):
        math_r = run_math_verification(device)
        all_results['math_verification'] = math_r
        save_table2(math_r, out_dir)

    if args.exp in ('all', 'scaling'):
        scaling = run_psnr_vs_k(device, quick=args.quick)
        all_results['psnr_vs_k'] = scaling
        save_figure1(scaling, out_dir)

    # ── Save JSON ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    all_results['elapsed_seconds'] = round(elapsed, 1)
    json_path = out_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"All results saved to:  {out_dir}")
    print(f"  results.json           — machine-readable")
    print(f"  table1_denoising.txt   — LaTeX Table 1")
    print(f"  table2_math.txt        — LaTeX Table 2")
    print(f"  psnr_vs_k.png          — Figure 1")
    print(f"Total time: {elapsed/60:.1f} min")

    # ── Quick summary ──────────────────────────────────────────────────────
    if 'math_verification' in all_results:
        cr = all_results['math_verification'].get('chain_verification', {})
        hc = all_results['math_verification'].get('hamiltonian_conservation', {})
        print(f"\nKey numbers for paper:")
        print(f"  Chain cosine similarity:     {cr.get('ac_vs_score', '—'):.6f}  (Theory: 1.0)")
        print(f"  H-drift ratio naive/symp:    {hc.get('naive_over_symplectic', '—'):.1f}×")
    if 'psnr_vs_k' in all_results:
        kan = all_results['psnr_vs_k']['KAN-EBM']
        ffn = all_results['psnr_vs_k']['FFN']
        k_max = max(all_results['psnr_vs_k']['K_list'])
        print(f"  KAN-EBM PSNR @ K={k_max}: {kan[k_max]['psnr']:.2f} dB")
        print(f"  FFN     PSNR @ K={k_max}: {ffn[k_max]['psnr']:.2f} dB")
        print(f"  Gain from iteration:         +{kan[k_max]['psnr'] - ffn[k_max]['psnr']:.2f} dB")


if __name__ == '__main__':
    main()
