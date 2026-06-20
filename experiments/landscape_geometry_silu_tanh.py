"""
landscape_geometry_silu_tanh.py
================================
W2: Extends landscape_geometry.py to ConvMLP-SiLU and ConvMLP-Tanh.
Loads the existing outputs/landscape_geometry/basin_geometry.json (which has
KAN, GELU, ReLU), runs basin_radius + barrier on the missing two variants,
and writes back a complete table.
"""
import json
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

# Reuse helpers from landscape_geometry
import landscape_geometry as lg  # noqa: E402
from rebuttal_smooth_mlp import ConvSmoothMLPEBM  # noqa: E402

OUT = ROOT / 'outputs' / 'landscape_geometry'
PATH = OUT / 'basin_geometry.json'

# Load existing
existing = json.loads(PATH.read_text())
have = set(existing['models'].keys())
print(f"Existing variants in basin_geometry.json: {sorted(have)}")

# Missing variants to add
NEW_SPECS = [
    {
        "key": "conv_mlp_silu",
        "kwargs": dict(n_filters=16, filter_size=5, mlp_hidden=160,
                       n_channels=3, activation="silu"),
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_silu.pt",
    },
    {
        "key": "conv_mlp_tanh",
        "kwargs": dict(n_filters=16, filter_size=5, mlp_hidden=160,
                       n_channels=3, activation="tanh"),
        "ckpt": ROOT / "outputs" / "finalization" / "rebuttal" / "conv_mlp_tanh.pt",
    },
]

ds = lg.load_cifar10()
n_imgs = max(lg.N_IMAGES_BASIN, lg.N_IMAGES_BARRIER)
images = [ds[i][0] for i in range(n_imgs)]

for spec in NEW_SPECS:
    key = spec["key"]
    if key in have:
        print(f"  [skip] {key} already present")
        continue
    if not spec["ckpt"].exists():
        print(f"  [skip] checkpoint missing: {spec['ckpt']}")
        continue
    print(f"\n{'='*60}")
    print(f"Model: {key}")
    print(f"{'='*60}")
    model = ConvSmoothMLPEBM(**spec["kwargs"]).cuda()
    state = torch.load(spec["ckpt"], map_location='cuda', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} params")

    print("  -- Basin radius --")
    basin = lg.measure_basin_radius(model, images)
    print("  -- Energy barrier --")
    with torch.no_grad():
        barrier = lg.measure_barrier(model, images)

    existing['models'][key] = {
        "basin_radius": basin,
        "barrier": barrier,
    }
    print(f"  basin_radius_threshold: {basin['basin_radius_threshold']}")
    print(f"  mean barrier height: {np.mean(barrier['barrier_heights']):.1f}")

    del model
    torch.cuda.empty_cache()

# Save back
PATH.write_text(json.dumps(existing, indent=2))
print(f"\nUpdated {PATH}  (variants: {sorted(existing['models'].keys())})")

# Print a clean summary table
print("\n" + "=" * 70)
print(f"{'Variant':<20}  {'Basin radius':>13}  {'Mean barrier':>13}")
print("-" * 70)
for k, v in existing['models'].items():
    br = v['basin_radius']['basin_radius_threshold']
    bh = float(np.mean(v['barrier']['barrier_heights']))
    print(f"{k:<20}  {br:>13.3f}  {bh:>13.1f}")
