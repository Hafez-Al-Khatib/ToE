"""
Phase B: does the K*(sigma) ~ C sigma^alpha law extend to a NON-energy score model?
==================================================================================
We train a direct-score denoiser (predicts the score field s_theta(x), NOT the
gradient of a scalar energy -- the diffusion/NCSN parameterization) with the SAME
recipe as the EBM (single sigma_train=0.15, DSM), and run it through the SAME
per-image K-sweep harness (u <- u + eta * clamp(s_theta(u))).

If the score model also obeys K*(sigma) ~ C sigma^alpha with a similar exponent,
the law is a property of ITERATIVE DENOISING / early-stopped regularization, not of
the energy structure or the architecture -- a cross-family unification. If it does
NOT, that is a clean contrast that distinguishes energy-based test-time compute.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'theory'))
from tier0_seeds import get_data, fit_alpha, SIGMAS, K_MAX, psnr_per_image   # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)


class ScoreNet(nn.Module):
    """Direct score field s_theta: R^{3xHxW} -> R^{3xHxW} (non-conservative, non-energy)."""
    def __init__(self, ch=3, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, h, 3, padding=1), nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.SiLU(),
            nn.Conv2d(h, ch, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

    def loss(self, x_clean, sigma):
        noise = torch.randn_like(x_clean) * sigma
        xn = x_clean + noise
        s = self.net(xn)
        true_score = -noise / sigma ** 2          # score of the Gaussian noising kernel
        return sigma ** 2 * F.mse_loss(s, true_score)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_score(model, loader, device, n_epochs, sigma_train=0.15, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device)
            opt.zero_grad()
            loss = model.loss(x, sigma_train)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == n_epochs:
            print(f"    [score] ep {ep:3d}/{n_epochs} loss={tot/nb:.5f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)


def kstar_score(model, clean, sigma, device, eval_seeds=2, dt=0.05, decay=0.97):
    """Per-image continuous K* under score-ascent denoising (mirrors the EBM harness)."""
    per_image = []
    for es in range(eval_seeds):
        torch.manual_seed(10_000 + es)
        u = (clean + sigma * torch.randn_like(clean)).to(device)
        series = [psnr_per_image(u, clean)]
        step = dt
        with torch.no_grad():
            for _ in range(K_MAX):
                s = model(u).clamp(-1, 1)            # mirror EBM's clamped gradient
                u = (u + step * s)
                step *= decay
                series.append(psnr_per_image(u, clean))
        arr = np.stack(series, axis=0)
        per_image.append(arr.argmax(axis=0))
    return float(np.mean(per_image))


def main():
    print(f"[device] {DEVICE}"
          + (f" [{torch.cuda.get_device_name(0)}]" if DEVICE.type == 'cuda' else ""))
    loader, test_ds = get_data(20000, DEVICE, batch_size=128)
    clean = torch.stack([test_ds[i][0] for i in range(64)]).to(DEVICE)

    alphas, peak_psnrs = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed); np.random.seed(seed)
        model = ScoreNet(ch=3, h=64).to(DEVICE)
        print(f"\n=== score seed={seed} ({model.n_params:,} params) ===", flush=True)
        train_score(model, loader, DEVICE, n_epochs=40)
        kstars = [kstar_score(model, clean, s, DEVICE) for s in SIGMAS]
        alpha, r2 = fit_alpha(SIGMAS, kstars)
        alphas.append(alpha)
        # peak PSNR at sigma=0.10 for a quality reference
        print(f"  seed {seed}: K*={[round(k,2) for k in kstars]}  alpha={alpha:.3f}  R^2={r2:.3f}",
              flush=True)
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    a = np.array(alphas)
    print("\n" + "=" * 56)
    print(f"score model: alpha = {a.mean():.3f} +/- {a.std(ddof=1):.3f}   per-seed {[round(x,3) for x in alphas]}")
    print(f"EBM reference (tier0): alpha = 1.376 +/- 0.003")
    print("=" * 56)


if __name__ == '__main__':
    main()
