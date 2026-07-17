# Review-Fixes Implementation Plan (paper_v8, post-panel)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. THIS RUN: inline execution chosen (deadline; verified numbers already in orchestrator context).

**Goal:** Repair every verified finding from the 2026-07-17 five-reviewer simulated panel so paper_v8's claims match the full evidence base, before the AAAI abstract deadline (2026-07-21); queue one optional Colab batch for seed-replication + scale ladder before the full deadline (2026-07-28).

**Architecture:** Workstream A = local text/analysis fixes only (all numbers verified against `outputs/` on 2026-07-17/18; two small CPU analysis scripts produce the two missing quantities). Workstream B = one gated Colab batch (~$15–30) mirroring `scripts/remote_run.sh` patterns. Editorial decision synthesized from panel: **Major Revision — consensus 4× Major Revision + DA reject-and-resubmit; DA CRITICALs C1–C3 verified true and are the priority order.**

**Tech Stack:** LaTeX (plain article + numbers-natbib), python 3.12 + numpy/torch (CPU) for analysis scripts, existing frontier parts JSONs as data.

## Global Constraints

- Do NOT commit anything under `outputs/` (repo convention, user-confirmed).
- Pre-registered frontier criterion text (0.3 dB, ≥3/5 σ, both seeds) must remain reported as-is; fixes change *interpretation and framing*, never the criterion or verdict (PASS 4/5 stands).
- ASCII-only console prints (Windows cp1256).
- Every number added to the paper must trace to a named JSON/script; record the trace in the commit message.
- No heavy GPU work locally; CPU-only analysis scripts are fine (models ≤110K params, 32×32).
- Numbers verified 2026-07-18 against parts (do not re-derive): heavy-vs-light peak gaps +0.050/+0.061/+0.085/+0.086/+0.070 dB at σ=0.05/0.10/0.15/0.20/0.30; wall-clock ms/step/img kan_32k 0.4688, kan_110k 0.8680, conv_mlp_gelu 0.0299, group_kan_8k 0.0492, group_kan_32k 0.1548, unet 0.0614; per-image K* (kan_32k seed0) std 0.08/0.50/0.68/1.33/1.95, ranges 1–2/2–4/3–8/4–21/3–30; DnCNN-micro (33K) σ=0.2 single-pass 27.29–27.43 dB over 3 seeds; gate_b σ=0.35 far-OOD E0→EK gains +0.114..+0.135 across all 4 checkpoints; fine-grid ConvMLP-GELU α=1.132 CI[1.083,1.184] vs KAN 1.365 CI[1.325,1.408] (pairwise bootstrap separates, p=1.0); excluded measured exponents ddpm_V2=1.205, CelebA-64=1.265, CelebA-HQ-256=1.610; β-sweep slope −0.09±0.44 R²=0.014 vs predicted −1.

---

### Task 1: Archive the panel + decision record

**Files:**
- Create: `docs/reviews/2026-07-17-simulated-panel-decision.md`

**Interfaces:** Produces the durable findings list (IDs C1–C3, M1–M7, W-series per reviewer) that later tasks' commit messages reference.

- [ ] **Step 1:** Write the decision record: editorial decision (Major Revision), per-reviewer recommendation + confidence, consolidated findings table with columns [ID | finding | severity | verified-against | fix task #], and the DA's rejected-attacks list (K_MAX truncation, pre-registration gaming, seed counts — all rejected). Mark C1 (selective exclusion/cohort switching), C2 (fine-grid contradicts F=0.79 headline), C3 (forced-overspend margins at σ=0.05) as the DA CRITICALs that bar Accept.
- [ ] **Step 2:** Commit: `git add docs/reviews/... && git commit -m "docs: simulated-panel decision record (Major Revision; findings C1-C3, M1-M7)"`

### Task 2: Two local analysis scripts (the missing quantities)

**Files:**
- Create: `theory/oracle_gain.py` — per-image oracle-K* PSNR minus fixed-K* PSNR, kan_32k, 500 CIFAR test images, σ∈{0.10,0.20,0.30}, CPU. Loads `outputs/finalization/kstar_param_scaling/kan_small.pt`, reuses `sequential_denoise_record` protocol (dt=0.05, decay 0.97, clamp ±1), deterministic noise via `torch.Generator(seed=10000*0+round(sigma*1000))` (protocol-identical to frontier parts). Output: `outputs/review_fixes/oracle_gain.json` with per-σ {oracle_mean_psnr, fixed_kstar_psnr, gain_db}.
- Create: `theory/calibration_cost.py` — pure-numpy on existing parts: for each σ, PSNR delivered by K=round(C·σ^α) for α∈{1.13,1.2,1.37,1.6} (C fit at σ=0.16 each) vs argmax PSNR; output `outputs/review_fixes/calibration_cost.json`.

**Interfaces:** Produces `gain_db` (consumed by Task 7's §8.1 rewrite) and `max_psnr_cost_db` per α (consumed by Task 4's calibration paragraph).

- [ ] **Step 1:** Write both scripts (calibration_cost is parts-only numpy; oracle_gain mirrors `exp_inference_frontier.py`'s eval loop with per-image argmax vs fixed-K readout).
- [ ] **Step 2:** Run: `py -3.12 theory/calibration_cost.py` then `py -3.12 theory/oracle_gain.py` (CPU; expect ≤~20 min). Record both JSONs' numbers in the ledger.
- [ ] **Step 3:** Commit scripts: `git commit -m "feat: oracle-gain and calibration-cost analyses (panel M3/M5)"`

### Task 3: §4 statistical reframe (C1, C2, W1-methodology, W5 bookkeeping)

**Files:** Modify `paper_v8.tex` §4 + abstract α sentence.

- [ ] Headline becomes protocol-conditional: report α=1.370±0.010 explicitly as *within-protocol run-to-run repeatability*, immediately followed by the bootstrap 95% CI [1.23,1.60] as the uncertainty statement, and the protocol-shift disclosure: same architecture moves 1.378→1.132 under the per-image fine-grid protocol; measured exponents outside the matched regime span 1.13–1.61 (ddpm_V2 1.205; CelebA-64 1.265; CelebA-HQ-256 1.610 — previously excluded as "known deviations", now disclosed in a short "Exponent spread beyond the matched regime" paragraph with the exclusion rule stated ex-post honestly).
- [ ] F-test sentence: add df and n (n=3 seeds/arch), state "failure to reject with n=12 is weak evidence of equivalence", add the fine-grid contradiction sentence (pairwise bootstrap separates KAN from ConvMLP, mean diff 0.23, CI [0.17,0.30]) and the shared-noise caveat (common noise realizations bias F downward). The abstract drops "an F-test cannot distinguish architectures" as a headline; replaced per Task 8.
- [ ] Fix "four datasets" → "three datasets"; state the 25-core/26-calibration cohort rule in one sentence where each is used.
- [ ] "Calibration" paragraph: rename "one-shot calibration" → "one-point calibration"; add the dB-cost sentence from `calibration_cost.json` (rule's PSNR cost vs oracle; robustness across α∈[1.13,1.6]) — frames peak-flatness as *forgiveness of the rule*, honestly noting it also bounds the practical value of exponent precision.
- [ ] Add functional-form honesty: σ spans 0.78 decades; note R² is not discriminating over this range and the power law is the simplest adequate description (drop "law" upgrades where they appear; title handled in Task 8).
- [ ] Commit with number traces.

### Task 4: §5 iteration reframe + DnCNN disclosure (M6)

**Files:** Modify `paper_v8.tex` §5.

- [ ] Reframe thesis: §5 shows only iterative operators have a *compute dial* — not that iteration wins on quality. Disclose: a parameter-matched feedforward DnCNN (33K params) reaches 27.4 dB at σ=0.2 in a single pass, above every energy head's semiconvergence peak (max 26.6 dB); cite `dncnn_results.json` values (27.29–27.43 over 3 seeds). State plainly: for pure fixed-noise denoising quality, a matched feedforward baseline is superior; the object of study is the compute-quality *curve*, which feedforward lacks.
- [ ] Note the FFN-DSM flatness is definitional (single-pass architecture); iterated feedforward (PnP-style) is future work, cited in Task 8's related-work additions.
- [ ] Commit.

### Task 5: §6 phase-boundary rewrite (domain W3, DA m3)

**Files:** Modify `paper_v8.tex` §6.

- [ ] Taxonomy fix: drop "signal-independent" (speckle is multiplicative; Poisson is signal-dependent); the supported dichotomy is stochastic vs deterministic.
- [ ] Mechanism fix: replace "a deterministic operator leaves none to remove" with: the residual exists (x − h∗x ≠ 0) but is structured and signal-correlated, outside the span of a score field trained only on isotropic additive noise; inverting it requires the forward operator (PnP/RED/DPS) or degradation-specific training (Cold Diffusion, InDI).
- [ ] Scope the claim to "DSM-trained, prior-only descent" everywhere the boundary is stated (incl. abstract phrasing per Task 8); add the JPEG-thinness caveat (K*∈{0,1,2}, weakly increasing).
- [ ] Commit.

### Task 6: §7 frontier rewrite (C3, M1, R3-W1, R1-W4)

**Files:** Modify `paper_v8.tex` §7 + Fig. 4 caption + Practical Takeaways (3).

- [ ] Add the envelope analysis: heavy head's peak exceeds light's by only +0.05–0.09 dB at every σ (per-σ numbers in Global Constraints); the honest statement of the frontier is two-sided — (i) at matched budgets light wins by +0.8..+3.9 dB for σ≥0.10 (survives the envelope analysis: at ~1.4 GF, heavy's best-within-budget ≈19.7 dB vs light 23.65 dB), and (ii) at unconstrained budget the heavy head buys ≤0.09 dB at 3.4× step cost — the oversized head is Pareto-dominated, and "substitution" is bounded by the light head's own peak.
- [ ] Correct the σ=0.05 boundary: the previous "heavier head wins by ~4.4 dB" was a forced-overspend artifact (advantage_region assumes budget exhaustion past K*); peak-vs-peak the gap is 0.05 dB at K*=1 for both. New statement: at σ=0.05 there is no depth to trade and no meaningful quality gap — the criterion fails there because no ≥0.3 dB *advantage region* exists, not because capacity wins. Fix Takeaway (3)'s "unless operating in the low-noise regime" accordingly (light head remains the rational choice at σ=0.05 — same quality at 29% of the single-step cost).
- [ ] Wall-clock scoping paragraph: FLOPs are the pre-registered budget metric; on the run's GPU, kan_32k costs 0.469 ms/step vs kan_110k 0.868 (1.85×, not the 3.4× FLOP ratio → within-family conclusion survives latency accounting, attenuated), while conv_mlp_gelu costs 0.030 ms/step (15.7× faster per step than kan_32k despite more FLOPs → cross-family FLOP conclusions do NOT transfer to latency). State the KAN elementwise-ops caveat (FlopCounterMode counts matmul/conv only; KAN gathers are uncounted, direction of bias stated). Scope Takeaway (3) to throughput/FLOP-budgeted serving.
- [ ] State that all winning regions begin at exactly one KAN-110K step (406.8 MF), the earliest comparable budget.
- [ ] Note the frontier rests on one trained model per head (2 evaluation-noise seeds); training-seed replication queued (Workstream B) — one honest sentence in Limitations.
- [ ] Commit with traces.

### Task 7: §8.1 + §8.2 rewrites (M2, M3, domain W5)

**Files:** Modify `paper_v8.tex` §8.1, §8.2, Fig. 5 caption.

- [ ] §8.1: replace "std ∼0.6 steps (range 3–6)" with the per-σ table from frontier parts (std 0.08→1.95, ranges up to 3–30 at σ=0.30). The claim becomes: spread grows with σ, but the *decisive* quantity is the oracle gain — insert `gain_db` from Task 2 (expected small; if oracle gain > 0.3 dB at any σ, the negative result must be weakened to that σ-range and the contributions list adjusted — decision gate, report to user).
- [ ] §8.2: (a) add deployable-depth numbers (AUROC 0.910–0.881 at K=2–5 vs 0.916 at K=0 — degradation at K* is 0.5–3.5 points, "destroys" reserved for the K→30 trend); (b) disclose the σ=0.35 reversal (refined energy beats static for far-OOD, +0.11..+0.14 across all four checkpoints; the score-before-it-thinks rule is noise-conditioned); (c) "destroys distributional information" → "erases the separability of the scalar energy score" (descent map is injective); (d) near-OOD sentence reframed: AUROC≈0.51 at K=0 — there is no signal to dissipate; (e) "explains" → "is consistent with" (Graham); (f) note Liu 2020 energy is supervised logsumexp, ours generative-DSM; (g) add Nalisnick likelihood-pathology paragraph: our DSM energy behaves as distance-to-manifold, escaping the CIFAR/SVHN likelihood inversion — state the energy-gap direction.
- [ ] Commit.

### Task 8: Related work, citations, theory-grounding, abstract/title/contributions

**Files:** Modify `paper_v8.tex` (Related Work, Discussion, bibliography, abstract, title footnote-level wording, contributions).

- [ ] New Related Work paragraph "Denoisers as priors and learned iterative schemes": Venkatakrishnan 2013 (PnP), Romano/Elad/Milanfar 2017 (RED), Kadkhodaie & Simoncelli 2021, Ulyanov 2018 (DIP), Gregor & LeCun 2010 (LISTA), Efron 2011 (Tweedie), Song & Ermon 2020 (Langevin steps-per-level heuristics) — with the one-sentence delta statement (measured exponent + cross-arch statistics + one-point calibration + limits).
- [ ] New paragraph "Step-budget trade-offs at scale": DDIM (already cited song2021 — split), DPM-Solver (Lu 2022), progressive distillation (Salimans & Ho 2022), consistency models (Song 2023), EDM (Karras 2022), Kaplan 2020 + Hoffmann 2022 (fixed-budget allocation), early-exit (Teerapittayanon 2016) next to §8.1.
- [ ] Restoration-under-deterministic-degradation: Cold Diffusion (Bansal 2022), InDI (Delbracio & Milanfar 2023), DPS (Chung 2023) — cited from §6 rewrite.
- [ ] OOD: Nalisnick 2019 (+optionally Serrà 2020).
- [ ] Over-thinking fix: cite an overthinking-specific reference (e.g., Chen et al. 2024 "Do NOT Think That Much") and keep snell2024 for allocation only; soften "denoising analogue" to "structurally similar rise-then-decay", add transfer-conditions sentence (contractive refinement operators, known noise level ≈ sufficient statistic; expected to fail for expansive/autoregressive iteration).
- [ ] Theory grounding: abstract's "We ground both the law and its limits in classical iterative-regularization theory" → "Classical semiconvergence theory predicts the rise-then-decay; the K*(σ) scaling in our descent-from-the-observation setting is empirical — notably, classical stopping-index rates for Landweber iteration *decrease* with noise, opposite to our K*, because the classical iteration starts from an uninformed initialization rather than the noisy observation." Add Engl/Hanke/Neubauer + Hansen citations. Disclose the failed spectral mechanism test in one Limitations sentence (β-sweep slope −0.09±0.44 vs predicted −1; mechanism remains open).
- [ ] Abstract + contributions rewritten to the reframed claims (α precision honest, F-test demoted, frontier two-sided statement, OOD noise-conditioned, adaptivity pending oracle-gain gate). All ~14 new bibitems added in existing style.
- [ ] Commit.

### Task 9: Consistency pass + ledger + push

- [ ] Run `scratchpad/check_paper.py` (env balance, refs, cites, figures) — expect clean.
- [ ] Grep for stale claims: `grep -nE "4\.4|0\.94|four datasets|calibration-free|signal-independent|destroys distributional" paper_v8.tex` — expect no hits.
- [ ] Number-trace list in commit message; update `.superpowers/sdd/progress.md`; push.

### Task 10 (GATED — user approval of spend): Colab batch v2

**Files:** Create `scripts/remote_run2.sh`, extend `experiments/train_group_kan.py`-style trainer for ConvMLP scale ladder, extend registry in `exp_inference_frontier.py`.

Contents (resumable, watchdog, mirrors remote_run.sh): (a) train primary-pair heads with 2 extra training seeds (kan_32k, kan_110k recipe-matched) + 20 frontier cells/seed → verdict stability check vs 0.3 dB margin; (b) 1M/8M/30M matched-recipe ConvMLP ladder + 30 frontier cells → scale panel for Fig. 4 + peak-PSNR-vs-params plot (R3 W4's inversion condition); (c) optional blur-trained EBM control (answers domain Q2: does blur move to "law holds" when trained on it); (d) wall-clock per step at batch {1,32,256}. Estimated 4–8 A100-h ≈ $6–12 (2× margin: ≤$25). Only (a) is required for the paper's current claims; (b)–(d) upgrade it.

---

## Self-Review

- Spec coverage: C1→T3, C2→T3, C3→T6, M1→T6, M2→T7, M3→T7+T2, M4→T3, M5→T3+T2, M6→T4, M7→T8, R3-W1→T6, R3-W2→T8, R3-W3→T3(rename)+T8, R3-W4→T10b(+T6 condition sentence), domain W1→T8, W2→T8, W3→T5, W4→T8, W5→T7, EIC W2→T8, W3→T7, W4 polish items→T9 grep + Fig3 labels deferred (archived figure; regenerate only if time permits — noted, acceptable). Gap check: none blocking.
- Placeholder scan: Task 2 outputs feed Tasks 3/7 (explicit decision gate if oracle-gain >0.3 dB). No TBDs.
- Type consistency: n/a (text edits + 2 scripts with named outputs).
