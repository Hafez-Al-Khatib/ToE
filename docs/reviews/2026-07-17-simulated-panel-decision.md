# Simulated Peer-Review Panel — Editorial Decision Record

**Paper:** paper_v8.tex — "The Limits of Test-Time Compute in Iterative Denoisers"
**Panel run:** 2026-07-17 (5 independent reviewer agents; academic-paper-reviewer full mode)
**Verification:** every load-bearing factual claim re-checked against `outputs/` on 2026-07-17/18 before adoption.

## Editorial Decision: MAJOR REVISION

Per-reviewer: EIC Major Revision (conf 4/5) · R1 Methodology Major Revision (4/5) ·
R2 Domain Major Revision (4/5) · R3 Perspective Major Revision (4/5) ·
Devil's Advocate: reject-and-resubmit unless claims re-scoped to full evidence base.
DA CRITICAL findings C1–C3 bar Accept (iron rule) and set the fix priority.

Panel consensus on strengths (all five): pre-registration integrity (verified in
`frontier.json`), numbers-trace-to-evidence discipline, honest negative results,
scoping language ("regime-scoped, not universal"). The DA explicitly REJECTED:
K_MAX-truncation confound, pre-registration gaming, and seed-count objections —
and confirmed the σ≥0.10 light-head advantage survives a correct envelope analysis.

## Findings register

| ID | Finding (verified source) | Sev | Fix task |
|----|---------------------------|-----|----------|
| C1 | ±0.010 is protocol repeatability, not uncertainty; bootstrap CI [1.23,1.60]; excluded measured exponents ddpm_V2=1.205, CelebA-64=1.265, CelebA-256=1.610 ("known deviations" hardcoded); cohort switches 25↔26 between claims (`consolidated_evidence.json`, `consolidate_all_results.py:149`) | CRIT | T3 |
| C2 | Fine-grid protocol separates KAN from ConvMLP (1.365 [1.325,1.408] vs 1.132 [1.083,1.184], pairwise bootstrap p=1.0), contradicting the abstract's F=0.79 headline (`finegrid_ci_summary.json`) | CRIT | T3 |
| C3 | Frontier σ=0.05 "capacity wins ~4.4 dB" is a forced-overspend artifact; peak-vs-peak gap is +0.050 dB (verified from parts). Advantage_region assumes budget exhaustion past K* | CRIT | T6 |
| M1 | Heavy head's peak exceeds light's by only +0.05..+0.09 dB at every σ → "capacity" buys ~nothing; heavy head is Pareto-dominated (parts, verified) | MAJ | T6 |
| M2 | OOD "destroys": at deployable K=2–5 AUROC is 0.910–0.881 (vs 0.916); at σ=0.35 refined EK BEATS static E0 for far-OOD, +0.114..+0.135 across all 4 checkpoints (`gate_b_kcheck.json`, `gate_b_verification.json`, verified) | MAJ | T7 |
| M3 | §8.1 "std ~0.6 (range 3–6)" false vs frontier parts: std 1.33 range 4–21 (σ=0.20), std 1.95 range 3–30 (σ=0.30) (verified). Decisive quantity = oracle gain in dB (batch v2) | MAJ | T7 |
| M4 | "Law" from 7 points over 0.78 decades; R² non-discriminating; K*=1 floor censoring at σ=0.05 | MAJ | T3 |
| M5 | dB cost of calibration error never reported. Computed (`calibration_cost.json`): α=1.37 rule ≤0.115 dB worst-case; α∈[1.13,1.6] all ≤0.349 dB — rule is forgiving; also bounds value of exponent precision | MAJ | T3 |
| M6 | DnCNN-micro (33K, single pass) 27.29–27.43 dB at σ=0.2 beats every energy head's peak (max 26.6) — undisclosed; §5 baseline (FFN-DSM 3.7M, 25.29) is the weaker one (verified `dncnn_results.json`) | MAJ | T4 |
| M7 | "Grounded in classical iterative-regularization theory": classical stopping index DECREASES with noise (k*∝δ^(−2/(2ν+1))), ours increases — settings differ, mapping never derived; own spectral-mechanism test failed (β-sweep slope −0.09±0.44 vs −1) and is undisclosed | MAJ | T8 |
| R3-W1 | FLOPs price a serial resource as if parallel. Wall-clock (parts): kan_32k 0.469, kan_110k 0.868 ms/step (1.85× — within-family survives), conv_mlp 0.030 (15.7× faster/step than kan_32k — cross-family inverts) | CRIT→scoped | T6 |
| R3-W2 | LLM/over-thinking bridge decorative; snell2024 miscited for overthinking | MAJ | T8 |
| R3-W3 | "Calibration-free" overclaim (needs C, α, and σ at inference) | MAJ | T3/T6 |
| R3-W4 | No inversion condition for substitution (peak-ceiling condition unstated) | MAJ | T6 + batch v2 ladder |
| W2-EIC/R2 | Missing literatures: PnP/RED, Kadkhodaie–Simoncelli, Tweedie, DIP, LISTA, Cold Diffusion/InDI/DPS, DDIM/DPM-Solver/distillation/EDM, Kaplan/Chinchilla, Nalisnick, early-exit, overthinking | MAJ | T8 |
| W3-R2 | Phase-boundary mechanism loose: speckle multiplicative, Poisson signal-dependent; "no residual to remove" false for blur; Cold-Diffusion/InDI counterexamples → scope to DSM-trained prior-only descent | MAJ | T5 |
| W5-R2 | OOD positioning: "explains"→"consistent with"; Liu2020 is supervised logsumexp; Nalisnick anomaly unaddressed; "destroys information"→separability of scalar readout | MAJ | T7 |
| W5-R1 | Bookkeeping: "four datasets" (JSON has 3); n=25/26 cohort switch; Fig3 code-name labels; Fig5 mid-sentence placement | MIN | T3/T9 |
| W3-EIC | §8.1 one paragraph, no figure, co-headline billing | MAJ | T7 + batch v2 oracle gain |

## Fix vehicles

- **Workstream A (text/analysis, local):** plan `docs/superpowers/plans/2026-07-18-review-fixes.md` Tasks 1–9.
- **Workstream B (Colab batch v2, approved "Full batch"):** `scripts/remote_run2.sh` — training-seed
  replication (verdict stability vs 0.3 dB margin), 1M/8M/30M scale ladder, wall-clock at batch {1,32,256},
  oracle gain, blur-trained control. Commit 2fc8844.
