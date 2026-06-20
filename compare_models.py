import json
import numpy as np
import math

with open('outputs/scaled_eval/cifar10_scaled_500.json') as f:
    data = json.load(f)

kan = data['results']['kan_ebm_f32']
sigmas = [float(s) for s in data['sigmas']]

kstar_obs = {}
for sigma in sigmas:
    psnrs = {int(k): v for k, v in kan[str(sigma)].items()}
    kstar = max(psnrs, key=psnrs.get)
    kstar_obs[sigma] = kstar

print('K* observed:', kstar_obs)

sig_arr = np.array(sigmas)
ks_arr = np.array([kstar_obs[s] for s in sigmas], dtype=float)

# Power law
log_s = np.log(sig_arr)
log_k = np.log(ks_arr)
slope_pl, int_pl = np.polyfit(log_s, log_k, 1)
pl_pred = np.exp(int_pl) * sig_arr**slope_pl
pl_r2 = 1 - np.sum((ks_arr - pl_pred)**2) / np.sum((ks_arr - ks_arr.mean())**2)

# Logarithmic
slope_log, int_log = np.polyfit(log_s, ks_arr, 1)
log_pred = int_log + slope_log * log_s
log_r2 = 1 - np.sum((ks_arr - log_pred)**2) / np.sum((ks_arr - ks_arr.mean())**2)

# Exponential
slope_exp, int_exp = np.polyfit(sig_arr, log_k, 1)
exp_pred = np.exp(int_exp + slope_exp * sig_arr)
exp_r2 = 1 - np.sum((ks_arr - exp_pred)**2) / np.sum((ks_arr - ks_arr.mean())**2)

# Linear
slope_lin, int_lin = np.polyfit(sig_arr, ks_arr, 1)
lin_pred = int_lin + slope_lin * sig_arr
lin_r2 = 1 - np.sum((ks_arr - lin_pred)**2) / np.sum((ks_arr - ks_arr.mean())**2)

# Polynomial degree 2
poly2 = np.polyfit(sig_arr, ks_arr, 2)
poly2_pred = np.polyval(poly2, sig_arr)
poly2_r2 = 1 - np.sum((ks_arr - poly2_pred)**2) / np.sum((ks_arr - ks_arr.mean())**2)

models = {
    'Power law': {'pred': pl_pred, 'r2': pl_r2, 'params': 2},
    'Logarithmic': {'pred': log_pred, 'r2': log_r2, 'params': 2},
    'Exponential': {'pred': exp_pred, 'r2': exp_r2, 'params': 2},
    'Linear': {'pred': lin_pred, 'r2': lin_r2, 'params': 2},
    'Quadratic': {'pred': poly2_pred, 'r2': poly2_r2, 'params': 3},
}

print('\nModel Comparison (KAN-EBM f32, 500-image eval):')
print('=' * 70)
header = f"{'Model':<15} {'R^2':>8} {'RMSE':>8} {'AICc':>10}"
print(header)
print('-' * 70)

n = len(sig_arr)
results = {}
for name, m in models.items():
    rmse = np.sqrt(np.mean((ks_arr - m['pred'])**2))
    k = m['params']
    # AICc for small sample sizes
    rss = np.sum((ks_arr - m['pred'])**2)
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else float('inf')
    results[name] = {'r2': float(m['r2']), 'rmse': float(rmse), 'aicc': float(aicc)}
    print(f"{name:<15} {m['r2']:>8.4f} {rmse:>8.3f} {aicc:>10.2f}")

with open('outputs/landscape_geometry/model_comparison.json', 'w') as f:
    json.dump({'kstar_observed': kstar_obs, 'models': results}, f, indent=2)
print('\nSaved to outputs/landscape_geometry/model_comparison.json')
