"""Micro-benchmark: real per-step cost of KAN vs ConvMLP DSM training on this GPU."""
import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
from exp_cifar10 import KANEnergyModel
from rebuttal_smooth_mlp import ConvSmoothMLPEBM

dev = torch.device('cuda')
print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

def bench(name, model, batch, n=4):
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(batch, 3, 32, 32, device=dev)
    # warmup (not timed)
    opt.zero_grad(); loss = model.loss(x, 0.15); loss.backward(); opt.step()
    torch.cuda.synchronize()
    for i in range(n):
        torch.cuda.synchronize(); t = time.time()
        opt.zero_grad(); loss = model.loss(x, 0.15); loss.backward(); opt.step()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {name} bs={batch} step {i}: {time.time()-t:.2f}s  loss={loss.item():.4f}  peakmem={mem:.2f}GB", flush=True)
    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

for bs in (8, 16, 32):
    bench('KAN', KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[48,16], n_channels=3), batch=bs, n=3)
bench('ConvMLP', ConvSmoothMLPEBM(n_filters=16, filter_size=5, mlp_hidden=160, n_channels=3, activation='gelu'), batch=128, n=3)
print("[done]", flush=True)
