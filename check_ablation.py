import json

with open('outputs/finalization/rebuttal/smooth_mlp_ablation.json') as f:
    data = json.load(f)

for v in data['variants']:
    variant = data['variants'][v]
    print(f'\n{v}:')
    print(f'  psnr_per_K_avg: {variant.get("psnr_per_K_avg", "N/A")}')
    if 'fit' in variant:
        print(f'  fit: {variant["fit"]}')
