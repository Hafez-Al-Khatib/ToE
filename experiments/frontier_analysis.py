"""Quality-vs-FLOPs curves and the PRE-REGISTERED pass criterion.

Criterion (spec docs/superpowers/specs/2026-07-02-inference-frontier-
groupkan-design.md section 2): PASS iff the lighter head beats the heavier
head by >= 0.3 dB PSNR at matched total FLOPs over a contiguous budget
range, across >= 3 of 5 sigma values, in BOTH seeds. Do not change the
constants below after data collection has started.
"""
import numpy as np

MARGIN_DB = 0.3
MIN_SIGMAS_PASS = 3


def curve_from_psnr(psnr_per_step, flops_step, psnr_k0):
    """(cumulative FLOPs, PSNR) curve; K=0 point (noisy input, 0 FLOPs) first."""
    K_max = len(psnr_per_step)
    flops = np.arange(0, K_max + 1, dtype=float) * float(flops_step)
    psnr = np.concatenate([[float(psnr_k0)],
                           np.asarray(psnr_per_step, dtype=float)])
    return flops, psnr


def advantage_region(flops_light, psnr_light, flops_heavy, psnr_heavy,
                     n_grid=200, margin_db=MARGIN_DB):
    """Contiguous budget ranges where light - heavy >= margin_db.

    Both curves are interpolated onto a shared log-spaced FLOP grid spanning
    the overlap of their nonzero-FLOP supports. A region must span more than
    one grid point (a range, not a point).
    """
    lo = max(flops_light[1], flops_heavy[1])
    hi = min(flops_light[-1], flops_heavy[-1])
    if hi <= lo:
        return []
    grid = np.geomspace(lo, hi, n_grid)
    ql = np.interp(grid, flops_light, psnr_light)
    qh = np.interp(grid, flops_heavy, psnr_heavy)
    adv = (ql - qh) >= margin_db

    regions, start = [], None
    for i in range(len(adv)):
        if adv[i] and start is None:
            start = i
        closing = (not adv[i]) or (i == len(adv) - 1)
        if closing and start is not None:
            end = i if adv[i] else i - 1
            if end > start:
                regions.append((float(grid[start]), float(grid[end])))
            start = None
    return regions


def evaluate_pass(results, light, heavy):
    """Apply the pre-registered criterion to results[model][sigma][seed]."""
    sigmas = sorted(results[light].keys())
    seeds = sorted(next(iter(results[light].values())).keys())
    per_seed_pass_count, detail = {}, {}
    for seed in seeds:
        n_pass = 0
        for s in sigmas:
            rl, rh = results[light][s][seed], results[heavy][s][seed]
            fl, ql = curve_from_psnr(rl['psnr_per_step'], rl['flops_step'],
                                     rl['psnr_k0'])
            fh, qh = curve_from_psnr(rh['psnr_per_step'], rh['flops_step'],
                                     rh['psnr_k0'])
            regions = advantage_region(fl, ql, fh, qh)
            detail[f'sigma={s}/seed={seed}'] = regions
            if regions:
                n_pass += 1
        per_seed_pass_count[seed] = n_pass
    verdict = all(c >= MIN_SIGMAS_PASS for c in per_seed_pass_count.values())
    return {'verdict': 'PASS' if verdict else 'FAIL',
            'per_seed_pass_count': per_seed_pass_count,
            'regions': detail, 'pair': [light, heavy],
            'margin_db': MARGIN_DB}
