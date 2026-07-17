"""
Calibration-cost analysis (panel finding M5): what does depth-rule error cost
in delivered PSNR, as a function of the assumed exponent alpha?

Pure numpy over the existing frontier parts for kan_32k (mean of 2 seeds).
Rule: K(sigma) = round(C * sigma^alpha), C fit so the rule is exact at
sigma_cal = 0.15 (the grid's middle point). Cost = peak PSNR - PSNR at the
rule's K, per sigma; we report the max over sigma for each alpha.

Writes outputs/review_fixes/calibration_cost.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / 'outputs/inference_frontier/parts'
SIGMAS = ['0.05', '0.10', '0.15', '0.20', '0.30']
SIGMA_CAL = 0.15
ALPHAS = [1.13, 1.20, 1.37, 1.60]


def mean_curve(sigma):
    curves = []
    for seed in (0, 1):
        d = json.loads(
            (PARTS / f'kan_32k_s{sigma}_seed{seed}.json').read_text())
        curves.append(d['psnr_per_step'])
    return np.mean(curves, axis=0)                       # index k-1 -> K=k


def main():
    kstar_cal = int(np.argmax(mean_curve(f'{SIGMA_CAL:.2f}'))) + 1
    out = {'sigma_cal': SIGMA_CAL, 'kstar_cal': kstar_cal, 'alphas': {}}
    for alpha in ALPHAS:
        C = kstar_cal / SIGMA_CAL ** alpha
        costs = {}
        for s in SIGMAS:
            sigma = float(s)
            curve = mean_curve(s)
            k_rule = int(np.clip(round(C * sigma ** alpha), 1, len(curve)))
            cost = float(curve.max() - curve[k_rule - 1])
            costs[s] = {'k_rule': k_rule, 'k_peak': int(np.argmax(curve)) + 1,
                        'cost_db': cost}
        worst = max(v['cost_db'] for v in costs.values())
        out['alphas'][f'{alpha:.2f}'] = {'C': C, 'per_sigma': costs,
                                         'max_cost_db': worst}
        print(f'alpha={alpha:.2f}: worst-case PSNR cost {worst:.3f} dB '
              f'(K_rule per sigma: '
              f'{[v["k_rule"] for v in costs.values()]})')
    dst = ROOT / 'outputs/review_fixes/calibration_cost.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print('wrote', dst)


if __name__ == '__main__':
    main()
