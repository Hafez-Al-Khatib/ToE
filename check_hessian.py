import json

with open('outputs/landscape_geometry/hessian_spectra.json') as f:
    data = json.load(f)

for model in ['kan_f32', 'conv_mlp_gelu', 'conv_mlp_relu', 'conv_mlp_silu']:
    d = data['models'][model]
    print(f'\n{model}:')
    for k, v in d.items():
        if 'lambda_max' in v:
            lm = v['lambda_max']
            if isinstance(lm, list):
                print(f'  {k}: lambda_max={lm}')
            else:
                print(f'  {k}: lambda_max={lm:.1f}')
