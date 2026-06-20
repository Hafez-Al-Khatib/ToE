"""
Practical payoff: the calibration-free K*(sigma) rule vs naive alternatives.
============================================================================
Claim to support: a 5-point calibration of K*(sigma)=C sigma^alpha gives a
SEARCH-FREE early-stopping rule that recovers ~oracle PSNR -- eliminating the
per-deployment K-sweep -- and beats the naive fixed-K and single-pass choices.

We compare, per noise level sigma (leave-one-out for the rule):
  - ORACLE  : best single K (argmax mean PSNR)            [unachievable upper bound]
  - RULE    : K = round(C sigma^alpha), (C,alpha) fit on the OTHER sigmas
  - FIXED-K : one K used for all sigma (best on average)  [naive practitioner]
  - K=1     : single pass, no test-time compute
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'theory'))
from measure_kappa import MODELS, load_cifar_test, DEVICE          # noqa: E402

SIGMAS = np.array([0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35])
K_MAX = 30
N_IMG = 100
EVAL_SEEDS = 2


def psnr_img(u, clean):
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).cpu().numpy()


def psnr_curve_and_kstar(model, clean, sigma, dt=0.05, decay=0.97):
    """Return (mean PSNR per K  shape (K+1,), continuous per-image K*)."""
    curves, per_img_kstar = [], []
    for es in range(EVAL_SEEDS):
        torch.manual_seed(10_000 + es)
        u = clean + sigma * torch.randn_like(clean)
        series = [psnr_img(u, clean)]
        step = dt
        for _ in range(K_MAX):
            with torch.enable_grad():
                xi = u.detach().requires_grad_(True)
                E = model.energy(xi)
                if E.ndim > 0:
                    E = E.sum()
                g = torch.autograd.grad(E, xi)[0].detach()
            u = (u - step * g.clamp(-1, 1)).detach()
            step *= decay
            series.append(psnr_img(u, clean))
        arr = np.stack(series, axis=0)            # (K+1, N)
        curves.append(arr.mean(1))
        per_img_kstar.append(arr.argmax(0))
    return np.mean(curves, 0), float(np.mean(per_img_kstar))


def fit_law(sig, kst):
    sl, ic = np.polyfit(np.log(sig), np.log(kst), 1)
    return np.exp(ic), sl   # C, alpha


def main():
    name = 'kan_f32'
    cfg = MODELS[name]
    model = cfg['cls'](**cfg['kw']).to(DEVICE)
    model.load_state_dict(torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"[device] {DEVICE}  model={name} ({sum(p.numel() for p in model.parameters()):,} params)")

    test = load_cifar_test()
    clean = torch.stack([test[i][0] for i in range(N_IMG)]).to(DEVICE)

    curves, kstars = {}, {}
    for s in SIGMAS:
        c, ks = psnr_curve_and_kstar(model, clean, float(s))
        curves[float(s)] = c
        kstars[float(s)] = ks

    # oracle (best single K per sigma)
    oracle_K = {s: int(np.argmax(curves[s])) for s in curves}
    oracle_P = {s: curves[s][oracle_K[s]] for s in curves}

    # fixed-K: the single K maximizing average PSNR across sigma
    avg_by_K = np.mean([curves[s] for s in curves], axis=0)
    K_fix = int(np.argmax(avg_by_K))

    # leave-one-out rule
    print(f"\n{'sigma':>6} | {'oracle':>14} | {'RULE (LOO)':>16} | {'fixed-K':>14} | {'K=1':>8}")
    print(f"{'':>6} | {'K   PSNR':>14} | {'K   PSNR  rec%':>16} | {'K   PSNR':>14} | {'PSNR':>8}")
    print("-" * 70)
    rule_rec, fix_rec, k1_gap, rule_gap, fix_gap = [], [], [], [], []
    for s in SIGMAS:
        s = float(s)
        others = [o for o in SIGMAS if float(o) != s]
        C, al = fit_law(np.array(others), np.array([kstars[float(o)] for o in others]))
        K_rule = min(max(int(round(C * s**al)), 0), K_MAX)
        P_rule = curves[s][K_rule]
        P_fix = curves[s][K_fix]
        P_k1 = curves[s][1]
        rec = 100 * P_rule / oracle_P[s]
        frec = 100 * P_fix / oracle_P[s]
        rule_rec.append(rec); fix_rec.append(frec)
        rule_gap.append(oracle_P[s] - P_rule); fix_gap.append(oracle_P[s] - P_fix)
        k1_gap.append(oracle_P[s] - P_k1)
        print(f"{s:>6.3f} | {oracle_K[s]:>2d}  {oracle_P[s]:>8.2f} | "
              f"{K_rule:>2d}  {P_rule:>6.2f}  {rec:>5.1f} | "
              f"{K_fix:>2d}  {P_fix:>6.2f} | {P_k1:>8.2f}")

    print("\n" + "=" * 70)
    print(f"RULE (leave-one-out):  mean recovery {np.mean(rule_rec):.2f}% of oracle  "
          f"(worst {np.min(rule_rec):.2f}%), mean gap {np.mean(rule_gap):.3f} dB, "
          f"max gap {np.max(rule_gap):.3f} dB")
    print(f"FIXED-K (K={K_fix} for all): mean recovery {np.mean(fix_rec):.2f}%  "
          f"(worst {np.min(fix_rec):.2f}%), mean gap {np.mean(fix_gap):.3f} dB, "
          f"max gap {np.max(fix_gap):.3f} dB")
    print(f"No test-time compute (K=1): mean oracle gap {np.mean(k1_gap):.2f} dB "
          f"(up to {np.max(k1_gap):.2f} dB at high sigma)")
    print(f"Compute: the rule needs ZERO per-sigma sweep; the sweep alternative costs "
          f"{K_MAX}x inference passes per sigma to locate K.")


if __name__ == '__main__':
    main()
