"""Smoke test: build all 4 model variants, run forward + grad + denoise."""
import sys
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from wav_kan_ebm import WavKANEnergyModel  # noqa: E402

torch.manual_seed(0)
device = torch.device('cpu')
B = 2
x = torch.randn(B, 3, 32, 32, device=device).clamp_(-1, 1)

configs = [
    dict(front='free', head='bspline', n_filters=16, kan_hidden=[48, 16]),
    dict(front='free', head='wavkan', n_filters=16, kan_hidden=[48, 16], n_atoms=8),
    dict(front='gabor', head='bspline', n_filters=16, kan_hidden=[48, 16]),
    dict(front='gabor', head='wavkan', n_filters=16, kan_hidden=[48, 16], n_atoms=8),
]

print(f"{'config':40s} {'n_params':>10s} {'E forward (s)':>14s} {'grad (s)':>10s} {'denoise K=5 (s)':>16s}")
print('-' * 92)
for c in configs:
    model = WavKANEnergyModel(**c).to(device)
    n = sum(p.numel() for p in model.parameters())
    tag = f"front={c['front']:5s} head={c['head']:7s}"
    t0 = time.time()
    E = model.energy(x)
    t1 = time.time()
    g = model.energy_grad(x)
    t2 = time.time()
    u = model.denoise(x, n_steps=5, dt=0.05, dt_decay=0.97)
    t3 = time.time()
    print(f"{tag:40s} {n:10,d} {t1-t0:14.3f} {t2-t1:10.3f} {t3-t2:16.3f}    "
          f"E={float(E):+.2f}  ||g||={g.norm().item():.2f}  ||u||={u.norm().item():.2f}")
