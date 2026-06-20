# Orchestration Plan: AAAI 2027 Strategic Decision & Submission Guide

**Date:** 2026-06-14  
**Deadline:** AAAI 2027 Abstract — July 21, 2026 (~37 days) | Full Paper — July 28, 2026 (~44 days)  
**Objective:** Produce a definitive strategic guide that (1) recommends UPGRADE over SWITCH, (2) documents the exact 5-week workflow, and (3) delivers a AAAI-ready research inference package.

---

## Stage 1 — Synthesis & Decision (Orchestrator, THIS TURN)
**Skill:** None needed — this is pure synthesis of project files already read.  
**Deliverable:** `plan.md` (this file) + `AAAI_Strategic_Guide.html` (interactive document).  
**Status:** IN PROGRESS.

### Key findings from file audit:
1. **KAN-EBM narrative collapsed** from "KAN is special" (α=1.53 vs 1.26) to "the law is universal" (α=1.376 ± 0.003 across ALL architectures + score nets). This is actually a *stronger* scientific result.
2. **NeurIPS failed** due to overclaiming, weak baselines, and FLOP dishonesty. The new narrative is honest and defensible.
3. **AAAI 2027 deadline is July 28** — only ~44 days. A switch to any new direction is impossible.
4. **The remaining experiments** (β-sweep, DDPM, DnCNN, BSD500) are well-scoped and achievable in 3-4 weeks.
5. **The paper rewrite** is the real risk — needs 2 solid weeks.

### Decision: UPGRADE, not SWITCH.
- **Upgrade path:** Continue the universal scaling law paper with the locked June 5 thesis. Add the remaining experiments. Rewrite paper_v5.tex around the new narrative.
- **Switch is impossible:** Any new direction requires 3-6 months minimum.

---

## Stage 2 — HTML Document Production (Orchestrator, THIS TURN)
**Skill:** None — using PythonRun for HTML generation with embedded CSS/JS.  
**Deliverable:** `AAAI_Strategic_Guide.html` — an interactive, self-contained strategic guide with:
- Executive summary with decision rationale
- Timeline visualization (Gantt-style)
- Risk matrix
- Experiment priority matrix
- Daily/weekly task breakdown
- Paper structure template
- AAAI positioning guide

---

## Stage 3 — Post-Delivery Guidance (Future turns, if needed)
If the user requests:
- **Sub-agent `coder`:** Could help generate LaTeX templates, experiment scripts, or paper skeleton.
- **Sub-agent `explore`:** Could research specific AAAI accepted papers in the test-time compute / energy-based model space for citation positioning.
- **Sub-agent `plan`:** Could review the paper_v5 outline for structural coherence.

**No sub-agents needed for the current deliverable** — it is a pure synthesis and document production task.

---

## Deliverable Map

| File | Path | Purpose |
|------|------|---------|
| `plan.md` | `C:\Users\hafez\Desktop\AUB Research\ToE\plan.md` | Orchestration blueprint |
| `AAAI_Strategic_Guide.html` | `C:\Users\hafez\Desktop\AUB Research\ToE\AAAI_Strategic_Guide.html` | Interactive strategic guide |

---

## Quality Gates
- [ ] HTML renders correctly in browser
- [ ] All timeline dates computed from June 14, 2026 anchor
- [ ] All experiment references trace to actual files in repo
- [ ] Decision rationale is evidence-backed (cites specific review files)
- [ ] Workflow is actionable (not vague "do experiments")
