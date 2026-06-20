# AAAI 2027 Final Push — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all remaining experiments and write paper_v5.tex for AAAI 2027 submission (abstract July 21, full paper July 28, 2026).

**Architecture:** Run falsifiable FashionMNIST + MNIST K-sweeps to validate the predictive-alpha relationship (3rd/4th data points), upload Colab DDPM notebook for pretrained-diffusion universality test, then write the paper around the locked thesis with all results.

**Tech Stack:** PyTorch, torchvision, matplotlib, LaTeX (AAAI format)

## Global Constraints

- Python 3.12 on Windows (cp1256 terminal — NO Unicode in print statements)
- GPU: RTX 3080 10GB local, Colab Pro for DDPM (expires end of June 2026)
- All training: single-sigma recipe (sigma_train=0.15), 40 epochs, AdamW + cosine LR
- Data range [-1, 1], PSNR peak = 4.0
- AAAI 2027 format: 7 pages + 1 references, double-column

---

## Results Inventory (what we have)

| Experiment | Status | Key result |
|-----------|--------|------------|
| Tier-0 seeds (CIFAR-10, 4 EBM heads) | DONE | alpha=1.377+/-0.003 |
| Cross-family (score net, CIFAR-10) | DONE | alpha matches EBMs |
| Scale-up 64px (CelebA-64) | DONE | alpha=1.26 |
| Phase diagram (corruption types) | DONE | Stochastic: R^2>0.95, Deterministic: R^2~0 |
| Spectral shrinkage theory figure | DONE | Illustrative mechanism |
| Practical payoff (5-pt calibration) | DONE | 99.83% oracle PSNR |
| DnCNN feedforward baseline | DONE | K=1 always, no compute-quality knob |
| beta-sweep on synthetic GRFs | DONE | FAILED go/no-go (slope=-0.09, expected -1) |
| Predictive-alpha (CIFAR+CelebA) | DONE | LOO <10%, gamma CV=6.3% (only 2 points) |
| measure_beta (FFT spectra) | DONE | CIFAR beta=2.90, CelebA=3.46, Fashion=2.28 |

## What's MISSING (this plan)

1. **FashionMNIST K-sweep** — predicted alpha=1.837, falsifiable 3rd data point
2. **MNIST K-sweep** — 4th data point (need to measure beta first)  
3. **Colab DDPM baseline** — pretrained 35.7M-param diffusion model universality
4. **Paper rewrite** — paper_v5.tex around locked thesis with all new results
5. **ARS reviewer pass** — simulated peer review before submission

---

### Task 1: FashionMNIST + MNIST K-sweep — COMPLETED

- [x] Created `theory/fashion_mnist_kstar.py` (supports --dataset fashion/mnist)
- [x] Smoke tested with --quick
- [x] Launched full overnight: Fashion 40ep 3 seeds + MNIST 40ep 3 seeds
- [x] **Results (2026-06-20):**
  - Fashion: alpha = 1.372 +/- 0.005 (EBM=1.371, score=1.374, R^2=0.992)
  - MNIST: alpha = 1.356 +/- 0.009 (EBM=1.349, score=1.364, R^2=0.999)
  - **KEY FINDING:** alpha~1.37 is near-constant across all 28-32px datasets
  - Predictive-alpha (alpha=2gamma/beta) FAILED: Fashion predicted 1.84, measured 1.37

---

### Task 3: Upload and run Colab DDPM notebook (URGENT — Colab Pro expires June 30)

**Files:**
- Upload: `theory/colab_diffusion_baseline.ipynb` to Google Colab
- Output: saved to Google Drive at `AAAI_kstar/ddpm_baseline/`

- [ ] **Step 1: Upload notebook to Colab**
- [ ] **Step 2: Run all cells, monitor for disconnects**
- [ ] **Step 3: Download results JSON from Drive**

---

### Task 4: Re-run predictive-alpha with new data points

**Files:**
- Modify: `theory/predictive_alpha.py` (add fashion/mnist alpha loading)
- Output: updated `outputs/theory/predictive_alpha.json` and figure

- [ ] **Step 1: Add fashion alpha loading to predictive_alpha.py**
- [ ] **Step 2: Run analysis with 3+ data points**
- [ ] **Step 3: Evaluate LOO prediction quality**

---

### Task 5: Paper rewrite (paper_v5.tex)

**Files:**
- Modify: `paper_v5.tex`
- Reference: `theory/PAPER_OUTLINE.md` (locked thesis)

- [ ] **Step 1: Use /ars-outline to generate detailed section outline**
- [ ] **Step 2: Write each section following PAPER_OUTLINE.md structure**
- [ ] **Step 3: Integrate all figures and results tables**
- [ ] **Step 4: Use /ars-reviewer for simulated peer review**
- [ ] **Step 5: Revise based on review feedback**

---

### Task 6: Abstract submission (July 21 deadline)

- [ ] **Step 1: Use /ars-abstract to draft bilingual abstract**
- [ ] **Step 2: Polish to 150-word AAAI limit**
- [ ] **Step 3: Submit**

---

## Timeline

| Date | Task | Priority |
|------|------|----------|
| June 20 (tonight) | Launch Fashion+MNIST overnight | P0 |
| June 21-22 | Analyze overnight results, re-run predictive-alpha | P0 |
| June 21-25 | Upload + run Colab DDPM (before Pro expires!) | P0 |
| June 23-30 | Paper_v5 draft | P0 |
| July 1-7 | /ars-reviewer pass + revisions | P1 |
| July 8-14 | Final polishing, code release prep | P1 |
| July 15-20 | Abstract finalization | P0 |
| July 21 | Abstract submission | DEADLINE |
| July 22-27 | Final paper polish | P0 |
| July 28 | Full paper submission | DEADLINE |
