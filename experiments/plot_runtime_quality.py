"""
Plot runtime vs quality (PSNR) trade-off curve.
Shows per-image latency (ms) vs peak PSNR for all models at sigma=0.10.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
with open('outputs/landscape_geometry/latency_benchmark.json') as f:
    latency_data = json.load(f)

with open('outputs/scaled_eval/cifar10_scaled_500.json') as f:
    scaled_data = json.load(f)

# Model mapping
models = {
    'KAN-EBM f32': {
        'latency_ms': latency_data['models']['kan_f32']['per_step_ms']['mean'],
        'peak_K': 2,
        'scaled_name': 'KAN-EBM-110K',
        'color': '#d62728',  # red
        'marker': 'o',
    },
    'KAN-EBM small': {
        'latency_ms': latency_data['models']['kan_small']['per_step_ms']['mean'],
        'peak_K': 2,
        'scaled_name': 'KAN-EBM-32K',
        'color': '#ff7f0e',  # orange
        'marker': 's',
    },
    'ConvMLP-GELU': {
        'latency_ms': latency_data['models']['conv_mlp_gelu']['per_step_ms']['mean'],
        'peak_K': 2,
        'scaled_name': 'ConvMLP-GELU',
        'color': '#2ca02c',  # green
        'marker': '^',
    },
    'FFN-DSM': {
        'latency_ms': latency_data['models']['ffn_dsm']['per_step_ms']['mean'],
        'peak_K': 1,
        'scaled_name': 'FFN-DSM',
        'color': '#9467bd',  # purple
        'marker': 'D',
    },
    'Micro-DnCNN': {
        'latency_ms': latency_data['models']['micro_dncnn']['per_step_ms']['mean'],
        'peak_K': 1,
        'scaled_name': None,
        'color': '#17becf',  # cyan
        'marker': 'v',
    },
}

sigma = '0.1'
points = []

for label, cfg in models.items():
    total_ms = cfg['latency_ms'] * cfg['peak_K']
    
    if cfg['scaled_name']:
        psnr = scaled_data['results'][sigma][cfg['scaled_name']][str(cfg['peak_K'])]['psnr']
    else:
        psnr = 31.35  # Micro-DnCNN from Table 2
    
    points.append({
        'label': label,
        'latency_ms': total_ms,
        'psnr': psnr,
        'color': cfg['color'],
        'marker': cfg['marker'],
        'peak_K': cfg['peak_K'],
    })

# Add K=1 points for iterative models
for label, cfg in models.items():
    if cfg['peak_K'] > 1 and cfg['scaled_name']:
        total_ms_k1 = cfg['latency_ms'] * 1
        psnr_k1 = scaled_data['results'][sigma][cfg['scaled_name']]['1']['psnr']
        points.append({
            'label': f"{label} (K=1)",
            'latency_ms': total_ms_k1,
            'psnr': psnr_k1,
            'color': cfg['color'],
            'marker': cfg['marker'],
            'peak_K': 1,
            'alpha': 0.5,
        })

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

for p in points:
    alpha = p.get('alpha', 1.0)
    size = 80 if alpha == 1.0 else 50
    ax.scatter(p['latency_ms'], p['psnr'], 
               color=p['color'], marker=p['marker'], s=size, 
               alpha=alpha, edgecolors='black', linewidth=0.5,
               zorder=3)

# Annotate peak points
for p in points:
    if p.get('alpha', 1.0) == 1.0:
        offset = (8, 5) if p['latency_ms'] < 50 else (-8, 5)
        ha = 'left' if p['latency_ms'] < 50 else 'right'
        ax.annotate(p['label'], 
                    xy=(p['latency_ms'], p['psnr']),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, ha=ha, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

# Connect K=1 to peak for iterative models
for label, cfg in models.items():
    if cfg['peak_K'] > 1 and cfg['scaled_name']:
        k1_psnr = scaled_data['results'][sigma][cfg['scaled_name']]['1']['psnr']
        peak_psnr = scaled_data['results'][sigma][cfg['scaled_name']][str(cfg['peak_K'])]['psnr']
        k1_ms = cfg['latency_ms'] * 1
        peak_ms = cfg['latency_ms'] * cfg['peak_K']
        ax.annotate('', xy=(peak_ms, peak_psnr), xytext=(k1_ms, k1_psnr),
                    arrowprops=dict(arrowstyle='->', color=cfg['color'], lw=1.5, ls='--'))

ax.set_xlabel('Total inference latency per image (ms)', fontsize=11)
ax.set_ylabel('Peak PSNR (dB) at σ=0.10', fontsize=11)
ax.set_xscale('log')
ax.set_xlim(0.3, 200)
ax.set_ylim(26, 32)
ax.grid(True, alpha=0.3, which='both')
ax.set_title('Runtime vs. Quality Trade-off (CIFAR-10, σ=0.10)', fontsize=12)

# Add note
ax.text(0.02, 0.02, 'Arrows: K=1 → peak K.\nLower-left is better (faster + higher PSNR).',
        transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('outputs/finalization/rebuttal/runtime_quality_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig('outputs/finalization/rebuttal/runtime_quality_tradeoff.png', dpi=300, bbox_inches='tight')
print("Saved runtime_quality_tradeoff.{pdf,png}")

# Throughput bar chart
fig2, ax2 = plt.subplots(figsize=(7, 4))
model_names = ['Micro-DnCNN', 'FFN-DSM', 'ConvMLP-GELU', 'KAN-EBM small', 'KAN-EBM f32']
throughputs = [17743, 33282, 5961, 618, 327]
psnrs = [31.35, 27.32, 29.69, 30.28, 30.35]
colors = ['#17becf', '#9467bd', '#2ca02c', '#ff7f0e', '#d62728']

bars = ax2.barh(model_names, throughputs, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Throughput (images / second)', fontsize=11)
ax2.set_xscale('log')
ax2.set_xlim(100, 100000)

for i, (bar, psnr) in enumerate(zip(bars, psnrs)):
    width = bar.get_width()
    ax2.text(width * 1.2, bar.get_y() + bar.get_height()/2, 
             f'{psnr:.2f} dB', va='center', fontsize=9)

ax2.set_title('Throughput vs. Peak PSNR (CIFAR-10, σ=0.10)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/finalization/rebuttal/throughput_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('outputs/finalization/rebuttal/throughput_comparison.png', dpi=300, bbox_inches='tight')
print("Saved throughput_comparison.{pdf,png}")
