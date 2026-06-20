"""
compute_oracle_recovery.py
==========================
Computes the oracle recovery percentage for the K* prediction law.

Reads:
  outputs/finalization/kstar/kstar_spectrum.json

For each sigma:
  - Oracle PSNR = max_K PSNR(sigma, K)
  - Predicted K = round(C * sigma^alpha)  (C=86.682, alpha=1.534)
  - If K_pred is not in the discrete grid, fall back to the nearest grid K.
  - Recovery ratio = PSNR(sigma, K_pred) / Oracle PSNR

Outputs:
  outputs/finalization/rebuttal/oracle_recovery.json
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECTRUM = ROOT / 'outputs' / 'finalization' / 'kstar' / 'kstar_spectrum.json'
OUT = ROOT / 'outputs' / 'finalization' / 'rebuttal'
OUT_FILE = OUT / 'oracle_recovery.json'

C = 86.682
ALPHA = 1.534


def nearest_grid_k(K_pred, grid):
    return min(grid, key=lambda k: abs(k - K_pred))


def main():
    if not SPECTRUM.exists():
        raise SystemExit(f"Spectrum file not found: {SPECTRUM}")

    data = json.loads(SPECTRUM.read_text())
    psnr_table = data['psnr_table']

    first_sigma = list(psnr_table.keys())[0]
    grid = sorted(int(k) for k in psnr_table[first_sigma].keys())

    results = {}
    recoveries = []
    for sigma_str, psnrs in psnr_table.items():
        sigma = float(sigma_str)
        oracle_psnr = max(float(v) for v in psnrs.values())
        K_pred_raw = C * (sigma ** ALPHA)
        K_pred_rounded = round(K_pred_raw)
        if str(K_pred_rounded) in psnrs:
            K_pred = K_pred_rounded
            fallback_note = ""
        else:
            K_pred = nearest_grid_k(K_pred_rounded, grid)
            fallback_note = f" (fallback to nearest grid K={K_pred}, raw={K_pred_raw:.2f})"

        pred_psnr = float(psnrs[str(K_pred)])
        recovery = pred_psnr / oracle_psnr
        recoveries.append(recovery)
        results[sigma_str] = {
            'oracle_psnr': oracle_psnr,
            'K_pred_raw': K_pred_raw,
            'K_pred_used': K_pred,
            'pred_psnr': pred_psnr,
            'recovery_ratio': recovery,
        }
        print(f"sigma={sigma_str:>5}  K_pred={K_pred:>2}{fallback_note:<45}  "
              f"oracle={oracle_psnr:.3f}  pred={pred_psnr:.3f}  recovery={recovery:.4f}")

    avg_recovery = sum(recoveries) / len(recoveries)
    out_data = {
        'C': C,
        'alpha': ALPHA,
        'average_recovery_ratio': avg_recovery,
        'average_recovery_percent': avg_recovery * 100,
        'per_sigma': results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out_data, indent=2))
    print(f"\nAverage oracle recovery: {avg_recovery*100:.2f}%")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
