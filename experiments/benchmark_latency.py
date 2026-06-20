"""
benchmark_latency.py
====================
Wall-clock latency benchmark for inference — addressing reviewer concern
about practical value.

Measures per-step latency, total inference time at various K, and throughput
for KAN-EBM, ConvMLP-EBM, FFN-DSM, and Micro-DnCNN on CIFAR-10.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel, FFNDenoiser
from rebuttal_smooth_mlp import ConvSmoothMLPEBM
from exp_dncnn_cifar10 import MicroDnCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = ROOT / 'outputs' / 'landscape_geometry'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ebm_denoise_step(model, xn, K=1, step=0.05, dt_decay=0.97):
    """Manual gradient-descent denoising for EBM models."""
    u = xn.clone()
    step_size = step
    for _ in range(K):
        ui = u.detach().requires_grad_(True)
        E = model.energy(ui)
        # Handle both scalar and tensor energies
        if E.ndim > 0:
            E = E.sum()
        g = torch.autograd.grad(E, ui)[0].detach().clamp(-1, 1)
        u = (u - step_size * g).detach()
        step_size *= dt_decay
    return u


def benchmark_ebm(model, xn, K_list, n_reps=50):
    """Benchmark an EBM model. Returns dict with timing results."""
    model.eval()
    
    # Warm-up
    for _ in range(10):
        _ = ebm_denoise_step(model, xn, K=1)
    torch.cuda.synchronize()
    
    # Per-step latency (K=1)
    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ebm_denoise_step(model, xn, K=1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    per_step_ms = float(np.mean(times) * 1000)
    per_step_std_ms = float(np.std(times) * 1000)
    
    # Total time at various K values
    total_times = {}
    for K in K_list:
        times_k = []
        for _ in range(n_reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = ebm_denoise_step(model, xn, K=K)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_k.append(t1 - t0)
        total_times[K] = float(np.mean(times_k) * 1000)
    
    # Throughput at K=1: images per second
    throughput = 16.0 / (per_step_ms / 1000.0)
    
    return {
        'per_step_ms': {'mean': per_step_ms, 'std': per_step_std_ms},
        'total_ms': {f'K{k}': v for k, v in total_times.items()},
        'throughput_images_per_sec': throughput,
    }


def benchmark_forward(model, xn, n_reps=50):
    """Benchmark a feed-forward model (FFN or DnCNN)."""
    model.eval()
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(xn)
    torch.cuda.synchronize()
    
    # Forward pass latency
    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(xn)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    per_step_ms = float(np.mean(times) * 1000)
    per_step_std_ms = float(np.std(times) * 1000)
    
    # Throughput
    throughput = 16.0 / (per_step_ms / 1000.0)
    
    return {
        'per_step_ms': {'mean': per_step_ms, 'std': per_step_std_ms},
        'total_ms': {'K1': per_step_ms},  # forward models only have K=1
        'throughput_images_per_sec': throughput,
    }


def load_checkpoint(model, path):
    """Load checkpoint, skipping if not found or incompatible."""
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        return True
    except Exception as e:
        print(f"  [SKIP] Failed to load {path}: {e}")
        return False


def main():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Generate a fixed batch of 16 noisy CIFAR-10 images
    batch_size = 16
    sigma = 0.15
    xn = torch.randn(batch_size, 3, 32, 32, device=DEVICE) * sigma
    
    models_config = [
        {
            'key': 'kan_f32',
            'name': 'KAN-EBM f32',
            'class': KANEnergyModel,
            'kwargs': {'n_filters': 32, 'filter_size': 5, 'kan_hidden': [96, 16], 'n_channels': 3},
            'checkpoint': ROOT / 'outputs' / 'cifar10' / 'kan_ebm_f32.pt',
            'type': 'ebm',
            'expected_params': 112608,
            'K_list': [2, 5, 10],
        },
        {
            'key': 'kan_small',
            'name': 'KAN-EBM small',
            'class': KANEnergyModel,
            'kwargs': {'n_filters': 16, 'filter_size': 5, 'kan_hidden': [48, 16], 'n_channels': 3},
            'checkpoint': ROOT / 'outputs' / 'finalization' / 'kstar_param_scaling' / 'kan_small.pt',
            'type': 'ebm',
            'expected_params': 33440,
            'K_list': [2, 5, 10],
        },
        {
            'key': 'conv_mlp_gelu',
            'name': 'ConvMLP-GELU',
            'class': ConvSmoothMLPEBM,
            'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'gelu'},
            'checkpoint': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_gelu.pt',
            'type': 'ebm',
            'expected_params': 34977,
            'K_list': [2, 5, 10],
        },
        {
            'key': 'conv_mlp_relu',
            'name': 'ConvMLP-ReLU',
            'class': ConvSmoothMLPEBM,
            'kwargs': {'n_filters': 16, 'filter_size': 5, 'mlp_hidden': 160, 'n_channels': 3, 'activation': 'relu'},
            'checkpoint': ROOT / 'outputs' / 'finalization' / 'rebuttal' / 'conv_mlp_relu.pt',
            'type': 'ebm',
            'expected_params': 34977,
            'K_list': [2, 5, 10],
        },
        {
            'key': 'ffn_dsm',
            'name': 'FFN-DSM',
            'class': FFNDenoiser,
            'kwargs': {'img_size': 32, 'n_channels': 3, 'hidden': 512},
            'checkpoint': ROOT / 'outputs' / 'cifar10' / 'ffn_dsm_f32.pt',
            'type': 'forward',
            'expected_params': 3674624,
            'K_list': [1],
        },
        {
            'key': 'micro_dncnn',
            'name': 'Micro-DnCNN',
            'class': MicroDnCNN,
            'kwargs': {'n_channels': 3, 'depth': 5, 'n_filters': 16},
            'checkpoint': ROOT / 'outputs' / 'dncnn_baseline' / 'dncnn_cifar10.pt',
            'type': 'forward',
            'expected_params': 7939,
            'K_list': [1],
        },
    ]
    
    results = {}
    
    for cfg in models_config:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {cfg['name']}")
        print(f"{'='*60}")
        
        # Instantiate model
        model = cfg['class'](**cfg['kwargs']).to(DEVICE)
        n_params = count_params(model)
        print(f"  Params: {n_params:,} (expected ~{cfg['expected_params']:,})")
        
        # Load checkpoint
        if not load_checkpoint(model, cfg['checkpoint']):
            continue
        print(f"  Checkpoint loaded: {cfg['checkpoint']}")
        
        # Benchmark
        if cfg['type'] == 'ebm':
            bench = benchmark_ebm(model, xn, cfg['K_list'], n_reps=50)
        else:
            bench = benchmark_forward(model, xn, n_reps=50)
        
        print(f"  Per-step latency: {bench['per_step_ms']['mean']:.3f} ± {bench['per_step_ms']['std']:.3f} ms")
        for k, v in bench['total_ms'].items():
            print(f"  Total time {k}: {v:.3f} ms")
        print(f"  Throughput: {bench['throughput_images_per_sec']:.1f} images/sec")
        
        # Build entry
        entry = {
            'params': n_params,
            'per_step_ms': bench['per_step_ms'],
            'throughput_images_per_sec': bench['throughput_images_per_sec'],
        }
        
        # Add total_ms fields with explicit keys
        for k, v in bench['total_ms'].items():
            # k is like 'K2', 'K5', etc. → total_ms_at_K2
            k_num = k.replace('K', '')
            entry[f'total_ms_at_K{k_num}'] = v
        
        results[cfg['key']] = entry
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save JSON
    out = {'models': results, 'device': str(DEVICE), 'batch_size': batch_size, 'sigma': sigma}
    out_path = OUT_DIR / 'latency_benchmark.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to: {out_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("LATENCY BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<18} {'Params':>10} {'Step (ms)':>14} {'K=2 (ms)':>10} {'K=5 (ms)':>10} {'K=10 (ms)':>10} {'Img/s':>10}")
    print("-"*80)
    
    for cfg in models_config:
        key = cfg['key']
        if key not in results:
            continue
        r = results[key]
        name = cfg['name']
        params = r['params']
        step = f"{r['per_step_ms']['mean']:.2f}±{r['per_step_ms']['std']:.2f}"
        k2 = r.get('total_ms_at_K2', '-')
        k2_str = f"{k2:.2f}" if isinstance(k2, (int, float)) else str(k2)
        k5 = r.get('total_ms_at_K5', '-')
        k5_str = f"{k5:.2f}" if isinstance(k5, (int, float)) else str(k5)
        k10 = r.get('total_ms_at_K10', '-')
        k10_str = f"{k10:.2f}" if isinstance(k10, (int, float)) else str(k10)
        imgs = f"{r['throughput_images_per_sec']:.1f}"
        print(f"{name:<18} {params:>10,} {step:>14} {k2_str:>10} {k5_str:>10} {k10_str:>10} {imgs:>10}")
    
    print("="*80)


if __name__ == '__main__':
    main()
