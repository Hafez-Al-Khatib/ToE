"""Decisive diagnostic: run the clean per-image K* eval on the PAPER'S OWN checkpoints.

If the paper's trained KAN and ConvMLP show KAN~=ConvMLP under this eval, the headline
exponent gap is an EVAL-methodology artifact. If they show the gap, then the reduced
re-training (tier0) is what washed it out.
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'theory'))
from measure_kappa import MODELS, load_cifar_test, DEVICE          # noqa: E402
from tier0_seeds import kstar_at_sigma, fit_alpha, SIGMAS          # noqa: E402

EVAL = ['unet', 'kan_f32', 'kan_small', 'conv_mlp_gelu', 'conv_mlp_silu', 'conv_mlp_tanh']
PAPER_ALPHA = {'unet': 1.434, 'kan_f32': 1.365, 'kan_small': 1.358, 'conv_mlp_gelu': 1.132,
               'conv_mlp_silu': 1.126, 'conv_mlp_tanh': 1.136}

print(f"[device] {DEVICE}")
test_ds = load_cifar_test()
clean = torch.stack([test_ds[i][0] for i in range(64)]).to(DEVICE)

print(f"\n{'model':>14} | {'alpha (my eval)':>15} | {'paper alpha':>11} | {'R^2':>6}")
print("-" * 56)
for name in EVAL:
    cfg = MODELS[name]
    if not cfg['ckpt'].exists():
        print(f"{name:>14} | checkpoint missing"); continue
    model = cfg['cls'](**cfg['kw']).to(DEVICE)
    model.load_state_dict(torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=True))
    model.eval()
    kstars = [kstar_at_sigma(model, clean, s, DEVICE) for s in SIGMAS]
    alpha, r2 = fit_alpha(SIGMAS, kstars)
    print(f"{name:>14} | {alpha:>15.3f} | {PAPER_ALPHA[name]:>11.3f} | {r2:>6.3f}")
    del model
