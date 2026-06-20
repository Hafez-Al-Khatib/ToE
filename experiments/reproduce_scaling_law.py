"""
reproduce_scaling_law.py
========================
End-to-end reproducibility script for the KAN-EBM scaling-law claim.

What it does:
  1. Checks for the CIFAR-10 KAN-EBM checkpoint (outputs/cifar10/kan_ebm_f32.pt).
  2. If missing, trains it by delegating to experiments/exp_cifar10.py.
  3. Runs the K* sweep (step 5 of run_paper_finalization.py).
  4. Fits the power law K*(sigma) ~ C * sigma^alpha and computes bootstrap CIs.
  5. Generates the main figures (Pareto, scaling law, K* cross-validation).
  6. Writes a reproducibility_report.json summarising everything.

Run:
  python experiments/reproduce_scaling_law.py --device cuda
  python experiments/reproduce_scaling_law.py --device cuda --quick

Expected runtime (full): ~3 h on RTX 3080
Expected runtime (quick): ~30 min on RTX 3080
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
REPORT_PATH = OUT / "reproducibility_report.json"

SEED = 42

# ── Paths to existing checkpoints / prerequisites ─────────────────────────────
CIFAR_CKPT = OUT / "cifar10" / "kan_ebm_f32.pt"
CIFAR_RESULTS = OUT / "cifar10" / "results_f32.json"
FLOP_RESULTS = OUT / "flop_benchmark" / "flop_results.json"
KSTAR_VAL_JSON = OUT / "finalization" / "kstar_validation" / "kstar_validation.json"

# ── Helper: run a Python script via subprocess, streaming stdout/stderr ───────

def run_script(script_path: Path, args: list[str], cwd: Path = ROOT) -> dict:
    """Run a Python script and return a structured log entry."""
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    dt = time.time() - t0
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "elapsed_seconds": round(dt, 1),
        "cwd": str(cwd),
    }


def ensure_cifar_checkpoint(device: str, quick: bool) -> dict:
    """Train CIFAR-10 KAN-EBM if the checkpoint is missing."""
    if CIFAR_CKPT.exists():
        print(f"[check] Found existing checkpoint: {CIFAR_CKPT}")
        return {"action": "skipped", "reason": "checkpoint exists", "path": str(CIFAR_CKPT)}

    print("[check] CIFAR-10 KAN-EBM checkpoint missing. Training now...")
    args = [f"--device", device]
    if quick:
        args.append("--quick")
    # exp_cifar10.py trains KAN-EBM, MLP-EBM, and FFN-DSM together.
    log = run_script(ROOT / "experiments" / "exp_cifar10.py", args)
    log["action"] = "trained"
    return log


def run_kstar_sweep(device: str, quick: bool) -> dict:
    """Delegate to run_paper_finalization.py step 5."""
    args = [f"--device", device, "--only", "step5"]
    if quick:
        args.append("--quick")
    return run_script(ROOT / "experiments" / "run_paper_finalization.py", args)


def run_bootstrap_ci() -> dict:
    """Delegate to rebuttal_bootstrap_ci.py (CPU-only, fast)."""
    return run_script(ROOT / "experiments" / "rebuttal_bootstrap_ci.py", [])


def run_pareto_figure() -> dict:
    """Delegate to rebuttal_pareto_v2.py (reads existing JSONs)."""
    return run_script(ROOT / "experiments" / "rebuttal_pareto_v2.py", [])


def run_plot_scaling_laws() -> dict:
    """Delegate to plot_scaling_laws.py (reads existing JSONs)."""
    return run_script(ROOT / "experiments" / "plot_scaling_laws.py", [])


# ── Fit power law locally (mirrors logic in run_paper_finalization.py) ────────

def fit_power_law(sigmas, K_means):
    xs = np.log(np.asarray(sigmas, dtype=float))
    ys = np.log(np.asarray(K_means, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yh = intercept + slope * xs
    ss_res = float(np.sum((ys - yh) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(slope), float(r2)


def compute_local_powerlaw() -> dict:
    """Re-fit the power law from the generated kstar_validation.json."""
    if not KSTAR_VAL_JSON.exists():
        return {"error": f"{KSTAR_VAL_JSON} not found; cannot fit power law."}

    rec = json.loads(KSTAR_VAL_JSON.read_text())
    sigmas_str = rec.get("sigmas", [])
    sigmas = [float(s) for s in sigmas_str]

    fits = {}
    for tag, exp in rec.get("experiments", {}).items():
        K_means = [exp["mean"][str(s)] for s in sigmas_str]
        if len(set(K_means)) < 2:
            fits[tag] = {"note": "degenerate K* (constant); no fit possible"}
            continue
        a, b, r2 = fit_power_law(sigmas, K_means)
        fits[tag] = {
            "C": math.exp(a),
            "alpha": b,
            "r_squared": r2,
            "formula": f"K*(sigma) ~= {math.exp(a):.2f} * sigma^{b:.3f}",
        }
    return fits


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip CIFAR-10 training even if checkpoint is missing")
    args = ap.parse_args()

    device = args.device
    quick = args.quick
    t_start = time.time()

    report = {
        "script": "reproduce_scaling_law.py",
        "seed": SEED,
        "device_requested": device,
        "quick_mode": quick,
        "steps": [],
    }

    # ── Step 1: checkpoint / training ────────────────────────────────────────
    step1 = ensure_cifar_checkpoint(device, quick) if not args.skip_training else {
        "action": "skipped", "reason": "--skip-training"
    }
    report["steps"].append({"name": "ensure_cifar_checkpoint", "result": step1})
    if step1.get("returncode", 0) != 0 and not args.skip_training:
        print("[FATAL] CIFAR-10 training failed. Aborting.")
        report["status"] = "FAILED"
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        sys.exit(1)

    # ── Step 2: K* sweep ─────────────────────────────────────────────────────
    step2 = run_kstar_sweep(device, quick)
    report["steps"].append({"name": "kstar_sweep", "result": step2})
    if step2["returncode"] != 0:
        print("[FATAL] K* sweep failed. Aborting.")
        report["status"] = "FAILED"
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        sys.exit(1)

    # ── Step 3: bootstrap CIs ────────────────────────────────────────────────
    step3 = run_bootstrap_ci()
    report["steps"].append({"name": "bootstrap_ci", "result": step3})

    # ── Step 4: Pareto figure ────────────────────────────────────────────────
    step4 = run_pareto_figure()
    report["steps"].append({"name": "pareto_figure", "result": step4})

    # ── Step 5: scaling-law plot ─────────────────────────────────────────────
    step5 = run_plot_scaling_laws()
    report["steps"].append({"name": "plot_scaling_laws", "result": step5})

    # ── Step 6: local power-law refit ────────────────────────────────────────
    report["power_law_fits"] = compute_local_powerlaw()

    # ── Finalise report ──────────────────────────────────────────────────────
    report["total_elapsed_seconds"] = round(time.time() - t_start, 1)
    report["status"] = "SUCCESS"
    report["outputs_generated"] = {
        "cifar10_results": str(CIFAR_RESULTS) if CIFAR_RESULTS.exists() else None,
        "cifar10_checkpoint": str(CIFAR_CKPT) if CIFAR_CKPT.exists() else None,
        "kstar_validation_json": str(KSTAR_VAL_JSON) if KSTAR_VAL_JSON.exists() else None,
        "pareto_honest_pdf": str(OUT / "finalization" / "rebuttal" / "pareto_honest.pdf"),
        "bootstrap_ci_summary": str(OUT / "finalization" / "rebuttal" / "bootstrap_ci_summary.txt"),
        "scaling_laws_pdf": str(OUT / "cifar10" / "paper_scaling_laws.pdf"),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n{'='*70}")
    print(f"Reproducibility report saved to: {REPORT_PATH}")
    print(f"Total time: {report['total_elapsed_seconds'] / 60:.1f} min")
    print(f"Status: {report['status']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
