# Supervisor Meeting Prep — 2026-04-17

## One-sentence summary of where you are

The core empirical result (monotonic K-scaling on MNIST with KAN energy vs.
divergence with MLP energy) is real, reproducible, and strong; the NeurIPS
reviewer's critique is primarily about empirical breadth (MNIST-only training,
outdated baselines) and one specific theoretical overclaim, not about the
central finding.

## Open with (3 minutes)

1. "I sent the draft to a rigorous NeurIPS-style reviewer and got a 4/10 /
   borderline-reject with a clear path to acceptance."
2. "I've already acted on the critiques that don't require new experiments:
   narrowed the theoretical claim, reframed BM3D as a reference point, added
   a patch-based extension, and softened the natural-image framing."
3. "I have a concrete 5–7 day plan to address the remaining critiques with
   in-distribution CIFAR-10 training and a modern flow-matching baseline."

Hand over `OVERHAUL_PLAN.md` if they want the details.

## Show (in order)

1. **Figure 5 (PSNR vs K on MNIST).** This is the central empirical claim.
   Monotonic 22.9 → 27.2 dB with KAN; MLP actively diverges; FFN flat.
2. **Table 2 (ablation A–F).** Controlled ablation isolating KAN vs. MLP at
   matched parameters. The divergence of MLP-EBM under K-scaling is the cleanest
   single finding in the paper.
3. **Figure 3 (learned spline activations) + Figure 2 (filter bank).** The
   interpretability story. These visuals are persuasive with experienced advisors.
4. **Section 4 as rewritten** — the strengthened Proposition 1 (Algorithmic
   equivalence for any smooth E). Lead with the stronger form; note that it
   both generalizes and de-risks the original claim.

## Flag proactively (do not let them find)

- **The "Triple Equivalence" claim in earlier drafts was overclaimed** — it held
  only for quadratic E. I've rewritten Section 4 to state the stronger,
  more honest version (equivalence holds for any smooth E as an inference
  operator; the frameworks differ in how they construct E). This is also
  consistent with `RESEARCH_MAP.md` / `false_claims/06_unified_theory/`, which
  flagged this claim internally months ago.
- **CBSD68/Set12 and DDPM/DEQ numbers are from offline runs.** The data and
  logs exist on my machine but aren't committed to the repo. I'm going to pull
  them in as part of Phase 0 of the overhaul.
- **Natural-image numbers are zero-shot, not a primary claim.** Current abstract
  now states this explicitly.

## Do not dwell on

- The "FEP / unified theory" framing from earlier project docs. Your own
  research map already retired this and the paper no longer leans on it.
- Sudoku. The internal map flags it as structurally limited (`false_claims/04_sudoku_np_hardness/`); it's not in this paper, so don't open that door.
- Absolute PSNR vs. BM3D/DnCNN. You've reframed the paper so this is not the
  comparison that matters.

## Questions you should expect

- *"Why not just train on CIFAR-10 from the start?"*
  Honest answer: MNIST was the minimal controlled setting for isolating the
  KAN-vs-MLP architectural claim in the ablation. CIFAR-10 in-distribution
  training is the natural Phase 1 extension and is in the overhaul plan.

- *"What's the mechanism? Why does the KAN enable scaling but MLP doesn't?"*
  The learnable B-spline per-edge nonlinearity lets the energy carve sharp,
  local basins with smooth gradient fields between them. Fixed ReLU/GELU
  produce piecewise-linear energies whose gradient field is not aligned with
  the clean-data manifold; repeated descent walks off the manifold.

- *"Is this really test-time scaling, or just iterative refinement?"*
  The distinction is academic at the level of one image, but matters for
  framing. What we show is that investing K× more gradient steps at inference
  yields monotonic quality gains — which is the operational definition of
  test-time scaling. It's not best-of-N search; it's a deterministic
  inference-compute-for-quality curve.

- *"What's the risk in the overhaul?"*
  Medium risk that CIFAR-10 K-scaling is less clean than MNIST (energy landscape
  harder to shape). Mitigation is step-size / grid-size tuning (1 extra day).
  Low risk that the flow-matching baseline beats KAN-EBM at low K — that's
  actually compatible with the story as long as KAN-EBM overtakes at large K.

## If they ask whether to submit to NeurIPS now

Don't. Current state would get rejected. Finish the overhaul (5–7 days) and
submit the revised version. The submission deadline consideration should drive
whether this is NeurIPS 2026 or ICLR 2027.

## Leave-behinds

- `paper.tex` — current revised draft with writing fixes applied
- `OVERHAUL_PLAN.md` — concrete post-meeting execution plan
- `MEETING_PREP.md` (this file) — your talking points

## If asked about timing

Phase 1 CIFAR-10 runs can start tonight if your GPU is free. Results available
in 18–24 hours wall-clock. Flow-matching baseline Phase 2 runs in parallel the
day after. Realistic revised-draft-ready date: **one week from the meeting**.
