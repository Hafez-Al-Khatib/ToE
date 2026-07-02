import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from frontier_analysis import curve_from_psnr, advantage_region, evaluate_pass


def _cell(psnr_per_step, flops_step, psnr_k0=20.0):
    return {'psnr_per_step': list(psnr_per_step), 'psnr_k0': psnr_k0,
            'flops_step': flops_step}


def test_curve_includes_k0_origin():
    f, q = curve_from_psnr([21.0, 22.0, 22.5], flops_step=10.0, psnr_k0=20.0)
    assert f[0] == 0.0 and q[0] == 20.0
    assert len(f) == 4 and f[-1] == 30.0 and q[-1] == 22.5


def test_advantage_region_found_when_light_dominates():
    # light: cheap steps, +1 dB everywhere over the shared budget range
    fl, ql = curve_from_psnr(np.linspace(22, 26, 30), 1.0, 20.0)
    fh, qh = curve_from_psnr(np.linspace(21, 25, 30), 3.0, 20.0)
    regions = advantage_region(fl, ql, fh, qh)
    assert len(regions) >= 1
    lo, hi = regions[0]
    assert hi > lo > 0


def test_no_advantage_region_when_heavy_dominates():
    fl, ql = curve_from_psnr(np.linspace(21, 23, 30), 1.0, 20.0)
    fh, qh = curve_from_psnr(np.linspace(24, 27, 30), 3.0, 20.0)
    assert advantage_region(fl, ql, fh, qh) == []


def test_evaluate_pass_requires_3_sigmas_both_seeds():
    win = list(np.linspace(23, 27, 30))    # strong light curve
    lose = list(np.linspace(21, 24, 30))   # weaker heavy curve at 3x step cost
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = {'light': {}, 'heavy': {}}
    for s in sigmas:
        results['light'][s] = {0: _cell(win, 1.0), 1: _cell(win, 1.0)}
        results['heavy'][s] = {0: _cell(lose, 3.0), 1: _cell(lose, 3.0)}
    out = evaluate_pass(results, light='light', heavy='heavy')
    assert out['verdict'] == 'PASS'
    # break seed 1 everywhere -> FAIL (criterion demands BOTH seeds)
    for s in sigmas:
        results['light'][s][1] = _cell(lose, 3.0)
        results['heavy'][s][1] = _cell(win, 1.0)
    out = evaluate_pass(results, light='light', heavy='heavy')
    assert out['verdict'] == 'FAIL'
