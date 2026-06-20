"""
Generates AAAI_Progress_Update_June2026.pptx — advisor/lab-seminar deck
pitching the KAN-EBM inference-depth law paper updates.
Run: py -3.12 make_aaai_slides.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree
from pathlib import Path

OUT = Path(__file__).parent / "AAAI_Progress_Update_June2026.pptx"

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG        = RGBColor(0x0D, 0x1B, 0x2A)   # deep navy
C_ACCENT    = RGBColor(0x00, 0xB4, 0xD8)   # electric cyan
C_ACCENT2   = RGBColor(0x90, 0xE0, 0xEF)   # pale cyan
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT     = RGBColor(0xCA, 0xD3, 0xE0)   # muted blue-grey
C_DARK_BOX  = RGBColor(0x14, 0x2B, 0x40)   # card bg
C_GREEN     = RGBColor(0x2D, 0xD4, 0x8A)   # positive
C_ORANGE    = RGBColor(0xFF, 0x9F, 0x1C)   # warning / pivot
C_RED       = RGBColor(0xFF, 0x4D, 0x6D)   # negative / falsified

W, H = Inches(13.33), Inches(7.5)   # 16:9


# ── Helpers ───────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def fill_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, l, t, w, h, color, alpha=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def txt(slide, text, l, t, w, h,
        size=20, bold=False, color=C_WHITE, align=PP_ALIGN.LEFT,
        italic=False, wrap=True):
    tf_box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tf_box.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tf_box


def txt_block(slide, lines, l, t, w, h,
              size=18, bold=False, color=C_WHITE, align=PP_ALIGN.LEFT,
              line_spacing_pt=None):
    """lines: list of (text, bold_override, color_override) or plain strings."""
    tf_box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tf_box.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, str):
            text, b, c = item, bold, color
        else:
            text, b, c = item
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = b
        run.font.color.rgb = c
    return tf_box


def accent_bar(slide, t=0.52, h=0.055):
    box(slide, 0, t, 13.33, h, C_ACCENT)


def section_header(slide, label, color=C_ACCENT):
    box(slide, 0.35, 0.28, 0.06, 0.38, color)
    txt(slide, label, 0.55, 0.25, 12.0, 0.55, size=26, bold=True, color=C_WHITE)


def bullet(slide, items, l, t, w, size=17, gap=0.38, color=C_LIGHT,
           accent=C_ACCENT2, marker="▸"):
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            m, text, c = item
        else:
            m, text, c = marker, item, color
        txt(slide, m, l, t + i * gap, 0.25, gap, size=size, color=accent)
        txt(slide, text, l + 0.28, t + i * gap, w - 0.28, gap, size=size, color=c)


def card(slide, l, t, w, h, title, body_lines, title_color=C_ACCENT,
         body_size=15.5, title_size=16):
    box(slide, l, t, w, h, C_DARK_BOX)
    # left accent strip
    box(slide, l, t, 0.04, h, title_color)
    txt(slide, title, l + 0.12, t + 0.06, w - 0.2, 0.32,
        size=title_size, bold=True, color=title_color)
    tf_box = slide.shapes.add_textbox(
        Inches(l + 0.12), Inches(t + 0.38), Inches(w - 0.24), Inches(h - 0.45))
    tf = tf_box.text_frame
    tf.word_wrap = True
    first = True
    for line in body_lines:
        if first:
            p = tf.paragraphs[0]; first = False
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line if isinstance(line, str) else line[0]
        run.font.size = Pt(body_size)
        run.font.color.rgb = line[1] if isinstance(line, tuple) else C_LIGHT


def stat_box(slide, l, t, w, h, number, label, num_color=C_ACCENT):
    box(slide, l, t, w, h, C_DARK_BOX)
    box(slide, l, t, w, 0.04, num_color)
    txt(slide, number, l, t + 0.12, w, 0.55,
        size=32, bold=True, color=num_color, align=PP_ALIGN.CENTER)
    txt(slide, label, l, t + 0.65, w, 0.6,
        size=13, color=C_LIGHT, align=PP_ALIGN.CENTER)


# ═════════════════════════════════════════════════════════════════════════════
# Slides
# ═════════════════════════════════════════════════════════════════════════════

prs = new_prs()

# ── SLIDE 1: Title ────────────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
box(sl, 0, 0, 13.33, 0.12, C_ACCENT)
box(sl, 0, 7.38, 13.33, 0.12, C_ACCENT)
box(sl, 0.35, 1.5, 12.63, 0.06, C_ACCENT)

txt(sl, "AAAI 2027 · Progress Update · June 2026",
    0.5, 0.9, 12.3, 0.45, size=15, color=C_ACCENT2, bold=False)

txt(sl, "Optimal Test-Time Compute\nfor Denoisers is Set by\nNoise, Not Architecture",
    0.5, 1.7, 12.3, 2.6, size=40, bold=True, color=C_WHITE)

txt(sl, "A Universal Inference-Depth Law for Iterative Denoisers",
    0.5, 4.45, 12.3, 0.55, size=22, color=C_ACCENT, bold=False, italic=True)

txt(sl, "Hafez Al-Khatib  ·  AUB Research",
    0.5, 5.25, 12.3, 0.4, size=16, color=C_LIGHT)

txt(sl, "K*(σ) ≈ C σ^α    α = 1.376 ± 0.003   (architecture-independent)",
    0.5, 5.85, 12.3, 0.5, size=18, bold=True, color=C_GREEN)

# ── SLIDE 2: The Journey ──────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Where We Started  →  Where We Are")

phases = [
    ("Original claim (v1–v4)",
     "KAN-EBM has architecture-dependent exponent. KAN α = 1.45, ConvMLP α = 1.22. Δα = +0.23.",
     C_RED),
    ("NeurIPS outcome",
     "Desk reject (scope/fit). Own adversarial review: 3/10. Core empirical claim flagged as artifact.",
     C_ORANGE),
    ("Tier-0 audit  (June 4–5)",
     "3 training seeds + per-image continuous K*. Result: gap collapsed to zero. The LAW survived.",
     C_ORANGE),
    ("Phase B  (June 5)",
     "Direct score network (non-energy, 77K) gives α = 1.376 ± 0.002 — IDENTICAL to all EBMs.",
     C_GREEN),
    ("Reframed thesis  (locked June 5)",
     "The law is universal across architectures AND model families. Noise sets α, not the model.",
     C_GREEN),
]

for i, (title, body, c) in enumerate(phases):
    y = 0.8 + i * 1.22
    box(sl, 0.35, y, 0.06, 0.9, c)
    txt(sl, title, 0.55, y, 5.5, 0.4, size=16, bold=True, color=c)
    txt(sl, body, 0.55, y + 0.38, 5.5, 0.52, size=14, color=C_LIGHT)

# right: the pivot arrow
box(sl, 6.9, 1.0, 5.9, 5.8, C_DARK_BOX)
box(sl, 6.9, 1.0, 5.9, 0.04, C_ACCENT)
txt(sl, "The pivot in one equation", 7.05, 1.1, 5.6, 0.38,
    size=16, bold=True, color=C_ACCENT)
txt(sl, "Before:", 7.05, 1.6, 5.5, 0.35, size=15, bold=True, color=C_RED)
txt(sl, "α(KAN) > α(ConvMLP)  →  KAN is special",
    7.05, 1.95, 5.5, 0.45, size=14, color=C_LIGHT, italic=True)
txt(sl, "After:", 7.05, 2.6, 5.5, 0.35, size=15, bold=True, color=C_GREEN)
txt(sl, "α(KAN) = α(ConvMLP) = α(score)\n        = 1.376 ± 0.003",
    7.05, 2.95, 5.5, 0.65, size=16, bold=True, color=C_GREEN)
txt(sl, "The architecture gap was a training-recipe\nartifact (richer recipe → higher α for ALL\nmodels, not just KAN).",
    7.05, 3.75, 5.5, 0.85, size=14, color=C_LIGHT)
txt(sl, "New claim: the exponent is a property\nof the task (noise + data), not the model.",
    7.05, 4.72, 5.5, 0.75, size=14, bold=True, color=C_ACCENT2)

# ── SLIDE 3: Tier-0 Audit Detail ─────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Tier-0 Audit: Killing the Credibility Smell")

txt(sl, "Problem flagged by reviewer M6: all α values identical to 16 digits (same training seed).",
    0.5, 0.85, 12.3, 0.45, size=16, color=C_ORANGE)

cols = [
    ("What we ran",    C_ACCENT,  ["3 independent training seeds per architecture",
                                   "Per-image continuous K* (not argmax-of-mean)",
                                   "Matched recipe: 40ep / 20K / bs=32 / single σ=0.15",
                                   "4 energy heads: KAN, GELU, SiLU, Tanh"]),
    ("What we found",  C_GREEN,   ["α: KAN 1.376 | GELU 1.378 | SiLU 1.379 | Tanh 1.373",
                                   "ALL within 1.376 ± 0.003 — statistically identical",
                                   "R² = 0.98–0.997 across all seeds",
                                   "Seed std = 0.001–0.006 (genuine variance, not zero)"]),
    ("What died",      C_RED,     ["Δα = +0.23 (KAN vs ConvMLP): NOT reproducible",
                                   "Old checkpoints had richer recipe (more epochs,\nmulti-σ) → inflated α for KAN and U-Net",
                                   "Under matched training: zero architecture effect",
                                   "Paper v4 central claim is false"]),
]

for i, (title, col, items) in enumerate(cols):
    x = 0.35 + i * 4.28
    box(sl, x, 1.45, 4.05, 5.6, C_DARK_BOX)
    box(sl, x, 1.45, 4.05, 0.04, col)
    txt(sl, title, x + 0.12, 1.55, 3.8, 0.38, size=16, bold=True, color=col)
    for j, item in enumerate(items):
        txt(sl, "▸  " + item, x + 0.12, 2.05 + j * 1.1, 3.8, 1.0,
            size=13.5, color=C_LIGHT)

txt(sl, "Takeaway: the scaling LAW is rock-solid. The architecture narrative was the artifact.",
    0.5, 7.05, 12.3, 0.35, size=15, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)

# ── SLIDE 4: Universality Result ─────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "The Core Result: Architecture-Independent Universality")

# stat boxes row
stats = [
    ("α = 1.376", "KAN energy head\n3 seeds", C_ACCENT),
    ("α = 1.378", "GELU energy head\n3 seeds", C_ACCENT),
    ("α = 1.379", "SiLU energy head\n3 seeds", C_ACCENT),
    ("α = 1.373", "Tanh energy head\n3 seeds", C_ACCENT),
    ("α = 1.376", "Direct score net\n(non-energy, 77K)\n3 seeds", C_GREEN),
]
for i, (num, lbl, c) in enumerate(stats):
    stat_box(sl, 0.35 + i * 2.52, 0.82, 2.3, 1.55, num, lbl, c)

txt(sl, "All five models: α = 1.376 ± 0.003",
    0.35, 2.52, 12.5, 0.45, size=22, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)

# two key sub-claims
card(sl, 0.35, 3.1, 6.0, 3.85,
     "Cross-architecture (EBM heads)",
     ["Same energy parameterization family, different",
      "activation functions: KAN B-splines, GELU, SiLU,",
      "Tanh. Identical α under matched recipe.",
      "",
      "→ The exponent is NOT a property of",
      "   the activation or basis function."],
     C_ACCENT)

card(sl, 6.6, 3.1, 6.25, 3.85,
     "Cross-family (energy vs. score)",
     ["ScoreNet predicts the score field s_θ(x) directly —",
      "it has NO scalar energy, NO gradient structure.",
      "Trained with DSM under the same recipe.",
      "",
      "→ The exponent is NOT a property of the",
      "   energy-based parameterization at all.",
      "→ It is a property of iterative denoising itself."],
     C_GREEN)

# ── SLIDE 5: Phase Diagram ────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Where the Law Holds: Corruption Phase Diagram")

txt(sl, "Tested on existing checkpoints (KAN / ConvMLP / U-Net) across 6 corruption types.",
    0.5, 0.85, 12.3, 0.4, size=15, color=C_LIGHT)

rows = [
    ("Gaussian noise",     "R² > 0.97", "✓ Obeys", C_GREEN),
    ("Speckle noise",      "R² > 0.96", "✓ Obeys", C_GREEN),
    ("Poisson noise",      "R² > 0.95", "✓ Obeys", C_GREEN),
    ("Gaussian blur",      "R² ≈ 0.00", "✗ Breaks", C_RED),
    ("JPEG compression",   "R² ≈ 0.00", "✗ Breaks", C_RED),
    ("Salt-and-pepper",    "R² ≈ 0.45", "~ Partial", C_ORANGE),
]

hdrs = ["Corruption type", "Fit quality", "Law status"]
hx = [0.4, 5.5, 9.8]
hw = [4.8, 3.8, 3.0]
for j, (h, x, w) in enumerate(zip(hdrs, hx, hw)):
    box(sl, x, 1.35, w, 0.38, C_ACCENT)
    txt(sl, h, x + 0.1, 1.38, w - 0.15, 0.35, size=14, bold=True,
        color=C_BG, align=PP_ALIGN.CENTER)
for i, (name, r2, status, c) in enumerate(rows):
    y = 1.82 + i * 0.75
    bg = C_DARK_BOX if i % 2 == 0 else RGBColor(0x11, 0x22, 0x33)
    for j, (x, w) in enumerate(zip(hx, hw)):
        box(sl, x, y, w, 0.72, bg)
    txt(sl, name, hx[0] + 0.15, y + 0.12, hw[0] - 0.2, 0.5, size=15, color=C_LIGHT)
    txt(sl, r2,   hx[1] + 0.1,  y + 0.12, hw[1] - 0.15, 0.5, size=15,
        color=c, align=PP_ALIGN.CENTER, bold=True)
    txt(sl, status, hx[2] + 0.1, y + 0.12, hw[2] - 0.15, 0.5, size=15,
        color=c, align=PP_ALIGN.CENTER, bold=True)

txt(sl,
    "Mechanism insight: additive stochastic noise → Gaussian-smoothed posterior → iterative regularization applies.\n"
    "Deterministic operators (blur, JPEG) → structured degradation → different signal-noise decomposition → law breaks.",
    0.4, 6.42, 12.5, 0.85, size=14, color=C_LIGHT)

# ── SLIDE 6: Scale-Up ─────────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Scale-Up to 64px: Law Holds, α is Data-Dependent")

txt(sl, "CelebA-64 (64×64 RGB, 8K training images) — same protocol, same families.",
    0.5, 0.85, 12.3, 0.4, size=15, color=C_LIGHT)

# stats
stats64 = [
    ("α = 1.255", "Score net\n(64px CelebA)\n2 seeds", C_GREEN),
    ("α = 1.268", "ConvMLP-GELU\n(64px CelebA)\n3 seeds", C_ACCENT),
    ("α = 1.271", "KAN energy\n(64px CelebA)\n2 seeds", C_ACCENT),
    ("α = 1.376", "All models\n(32px CIFAR-10)\nreference", C_ACCENT2),
]
for i, (num, lbl, c) in enumerate(stats64):
    stat_box(sl, 0.4 + i * 3.1, 1.32, 2.75, 1.55, num, lbl, c)

box(sl, 0.35, 3.05, 12.63, 0.04, C_ORANGE)

card(sl, 0.35, 3.2, 6.1, 3.7,
     "α is architecture-independent  ✓",
     ["At 64px: score (1.255) ≈ ConvMLP (1.268) ≈ KAN (1.271)",
      "Same family clustering as at 32px.",
      "Cross-family universality HOLDS at higher resolution.",
      "",
      "R² > 0.999 across all 64px runs — even cleaner\npower law than at 32px."],
     C_GREEN)

card(sl, 6.65, 3.2, 6.05, 3.7,
     "α is data-dependent  ✓",
     ["CIFAR-32 α ≈ 1.376  vs  CelebA-64 α ≈ 1.26",
      "Δα ≈ 0.12  (well outside seed noise of ±0.003)",
      "",
      "Consistent with α = 2γ/β:",
      "  CelebA has smoother spectrum (larger β)",
      "  → smaller α (fewer steps needed per σ)",
      "",
      "paper_v4 claim 'Δα ≤ 0.005 across datasets'\nwas a methodology artifact — DROPPED."],
     C_ORANGE)

# ── SLIDE 7: Mechanism ────────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Mechanism: Early-Stopped Spectral Regularization")

txt(sl, "Why does noise set α?  A tractable local model gives the answer.",
    0.5, 0.82, 12.3, 0.38, size=15, color=C_LIGHT)

# left: the derivation sketch
box(sl, 0.35, 1.3, 6.3, 5.65, C_DARK_BOX)
box(sl, 0.35, 1.3, 6.3, 0.04, C_ACCENT)
txt(sl, "The derivation sketch", 0.5, 1.38, 6.0, 0.38, size=16, bold=True, color=C_ACCENT)

steps = [
    ("1. Setup", "Near attractor, energy is locally quadratic: E(u) = ½ uᵀAu. "
                 "GD in the eigenbasis: u_K[i] = (1−ηλᵢ)^K · ũ[i]"),
    ("2. Each step = shrinkage", "K gradient steps act as a Landweber spectral filter "
                                  "a_i(K) = (1−ηλᵢ)^K  — the same as Tikhonov/iterated regularization."),
    ("3. Optimal K per mode", "Bias-variance balance gives K*(σ) via the crossover mode: "
                               "modes above i* = σ^{−2/β} are noise-dominated."),
    ("4. The exponent", "K* ∝ λ_{i*}^{−1} ∝ σ^{2γ/β}   →   α = 2γ/β\n"
                        "β = data power spectrum exponent (P(f) ∝ f^{−β})\n"
                        "γ = learned curvature spectrum exponent"),
]
for i, (label, body) in enumerate(steps):
    y = 1.85 + i * 1.15
    txt(sl, label, 0.5, y, 1.6, 0.38, size=13.5, bold=True, color=C_ACCENT2)
    txt(sl, body, 0.5, y + 0.35, 5.9, 0.75, size=12.5, color=C_LIGHT)

# right: honest scope + predictions
box(sl, 6.85, 1.3, 6.1, 2.7, C_DARK_BOX)
box(sl, 6.85, 1.3, 6.1, 0.04, C_GREEN)
txt(sl, "What the theory correctly predicts", 7.0, 1.38, 5.8, 0.38,
    size=15, bold=True, color=C_GREEN)
preds = ["✓  Sign: more noise → more steps",
         "✓  Band: α ∈ [1.2, 1.5] for natural images",
         "✓  Data-dependence: larger β → smaller α",
         "✓  Phase boundary: breaks for non-additive noise",
         "✓  Architecture-independence (γ is recipe-set)"]
for i, p in enumerate(preds):
    txt(sl, p, 7.0, 1.88 + i * 0.38, 5.7, 0.38, size=13.5, color=C_LIGHT)

box(sl, 6.85, 4.15, 6.1, 2.8, C_DARK_BOX)
box(sl, 6.85, 4.15, 6.1, 0.04, C_ORANGE)
txt(sl, "Honest scope (important!)", 7.0, 4.23, 5.8, 0.38,
    size=15, bold=True, color=C_ORANGE)
txt(sl, "The go/no-go on real checkpoints FAILED\n(measured κ underpredicts α, anti-correlates\nfor some models). Confound: data-basis\ndiagonalization assumption breaks in practice.\n\nPresented as: illustrative mechanism +\nrigorous regularization-theory framing.\nNOT a fitted or measured law.",
    7.0, 4.65, 5.8, 2.15, size=13, color=C_LIGHT)

# ── SLIDE 8: Practical Payoff ─────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Practical Payoff: A Search-Free Early-Stopping Rule")

txt(sl, "The law is not just descriptive — it gives a deployable inference protocol.",
    0.5, 0.82, 12.3, 0.38, size=15, color=C_LIGHT)

stat_box(sl, 0.4,  1.35, 2.85, 1.6, "99.83%", "oracle PSNR\nrecovered", C_GREEN)
stat_box(sl, 3.5,  1.35, 2.85, 1.6, "30×",    "fewer inference\npasses vs K-sweep", C_ACCENT)
stat_box(sl, 6.6,  1.35, 2.85, 1.6, "−5.2 dB","cost of naive\nfixed-K at high σ", C_RED)
stat_box(sl, 9.7,  1.35, 2.85, 1.6, "5-point", "calibration\nper model", C_ACCENT2)

card(sl, 0.35, 3.1, 8.0, 3.9,
     "The protocol",
     ["1.  Calibrate: for 5 known σ values, measure K*(σ) on a small val set.",
      "2.  Fit: estimate C and α via log-log OLS (two parameters, five points).",
      "3.  Deploy: at test time, for any noise level σ, stop at K = round(C σ^α).",
      "4.  No K-sweep, no validation set at test time, no hyperparameter search.",
      "",
      "Leave-one-out evaluation on KAN-110K / CIFAR-10:",
      "  Rule: 99.83% oracle  |  max gap: 0.20 dB",
      "  Fixed K=10: 90.6%    |  worst case: −5.2 dB at high σ",
      "  K=1 (no refinement): up to −6.9 dB"],
     C_ACCENT)

card(sl, 8.55, 3.1, 4.5, 3.9,
     "Why reviewers care",
     ["The standard test-time-compute\nproblem is: how much to spend?",
      "",
      "Our answer: a closed-form rule\nfrom 5 calibration points.",
      "",
      "The law SAVES compute relative\nto running K=K_max every time.",
      "",
      "Connection to classical early-\nstopping literature (Yao 2007,\nRaskutti 2014) is the theory hook."],
     C_GREEN)

# ── SLIDE 9: In Progress ──────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "In Progress: Strengthening the Two Remaining Gaps")

card(sl, 0.35, 1.0, 6.15, 5.95,
     "β-sweep: controlled theory test  (4090 overnight)",
     ["What: train EBM + score models on synthetic 1/f^β Gaussian",
      "random fields for β ∈ {1.0, 1.5, 2.0, 2.5, 3.0}.",
      "These obey the spectral assumptions BY CONSTRUCTION.",
      "",
      "Test: does log α vs log β̂ fit a line with slope −1?",
      "Theory predicts: YES (α = 2γ/β → log α = log 2γ − log β).",
      "",
      "PASS → theory section upgrades from 'illustrative' to",
      "        'tested where assumptions hold'; CIFAR vs CelebA",
      "        becomes the in-the-wild corroboration.",
      "FAIL → reported as the boundary of the spectral account;",
      "        law, diagram, and rule are unaffected.",
      "",
      "Code: theory/beta_sweep.py  (resume-safe, saves after each run)",
      "Expected runtime: ~6–10h, 5 β × 2 families × 3 seeds = 30 runs"],
     C_ACCENT)

card(sl, 6.7, 1.0, 6.3, 5.95,
     "Pretrained DDPM baseline  (Colab Pro this month)",
     ["What: google/ddpm-cifar10-32  (35.7M params, production-trained)",
      "run through the IDENTICAL K-sweep protocol.",
      "",
      "VP → VE score conversion:",
      "  s(u) = −ε̂(√ᾱ_t · u, t) / σ_t",
      "  conditioning fixed at t matching σ=0.15 (apples-to-apples)",
      "",
      "If α ≈ 1.38: cross-family claim now spans",
      "  5.8K params (micro EBM)  →  35.7M params (DDPM)",
      "  trained-from-scratch  →  production-pretrained",
      "  energy model  →  score model",
      "  → Reviewer M1 fully retired.",
      "",
      "Code: theory/colab_diffusion_baseline.ipynb",
      "  Drive-mounted, saves after every σ, auto-resumes.",
      "Expected runtime: ~15 min on Colab T4"],
     C_GREEN)

# ── SLIDE 10: What's Still Needed ─────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Remaining Work to AAAI Submission")

rows_todo = [
    ("Weeks 1–2",  "Colab + 4090",  "DDPM baseline + β-sweep training",              "Colab Pro / 4090 SSH"),
    ("Weeks 3–4",  "4090 + 3080",   "DnCNN/NAFNet-small baseline (feedforward, no K-dial)",
                                                                                       "4090 training"),
    ("Weeks 3–4",  "4090",          "BSD500 64px: 3rd dataset for α(data) trend",     "4090 training"),
    ("Weeks 4–6",  "3080",          "Ablations: recipe (single- vs multi-σ), parameter-count sweep, "
                                    "eval-protocol, optimizer", "3080 eval"),
    ("Weeks 4–6",  "3080",          "Extend phase diagram: noise mixtures",           "3080 eval"),
    ("Weeks 6–8",  "Writing",       "Rewrite paper_v4.tex around locked thesis; code release",
                                                                                       "Deadline: ~Aug 2026"),
]

hdrs = ["Timeline", "Resource", "Task", "Note"]
hx = [0.35, 1.9, 3.5, 10.7]
hw = [1.45, 1.5, 7.05, 2.3]
for j, (h, x, w) in enumerate(zip(hdrs, hx, hw)):
    box(sl, x, 0.82, w, 0.38, C_ACCENT)
    txt(sl, h, x + 0.08, 0.84, w - 0.12, 0.36, size=13.5, bold=True,
        color=C_BG, align=PP_ALIGN.CENTER)

for i, (week, res, task, note) in enumerate(rows_todo):
    y = 1.28 + i * 0.98
    bg = C_DARK_BOX if i % 2 == 0 else RGBColor(0x11, 0x22, 0x33)
    for x, w in zip(hx, hw):
        box(sl, x, y, w, 0.95, bg)
    c_week = C_ACCENT if i < 2 else (C_ACCENT2 if i < 4 else C_ORANGE)
    txt(sl, week, hx[0]+0.08, y+0.12, hw[0]-0.12, 0.72, size=13, bold=True, color=c_week)
    txt(sl, res,  hx[1]+0.08, y+0.12, hw[1]-0.12, 0.72, size=12, color=C_LIGHT)
    txt(sl, task, hx[2]+0.08, y+0.06, hw[2]-0.12, 0.85, size=12.5, color=C_LIGHT)
    txt(sl, note, hx[3]+0.08, y+0.12, hw[3]-0.12, 0.72, size=11.5, color=C_ACCENT2)

txt(sl, "Scope discipline: NO Eikonal, latent-HKAN, or composition in this paper. One core, deepened.",
    0.35, 7.1, 12.6, 0.32, size=13.5, bold=True, color=C_RED, align=PP_ALIGN.CENTER)

# ── SLIDE 11: Honest Positioning ──────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Honest Positioning for AAAI")

card(sl, 0.35, 1.05, 4.0, 5.95,
     "What this paper IS",
     ["A clean, universal empirical law with:",
      "",
      "  • Cross-architecture universality",
      "    (5 heads, matched training)",
      "  • Cross-family universality",
      "    (EBM + score + DDPM pending)",
      "  • Mechanistic framing",
      "    (spectral regularization)",
      "  • Sharp phase boundary",
      "    (additive ✓ / deterministic ✗)",
      "  • Practical deployable rule",
      "    (99.83% oracle, 30× faster)",
      "  • Data-dependence explained",
      "    (α = 2γ/β)"],
     C_GREEN)

card(sl, 4.55, 1.05, 4.3, 5.95,
     "The reviewer risk (and response)",
     ["Risk: 'expected from regularization",
      "theory — nothing new.'",
      "",
      "Response:",
      "  (a) Cross-family universality was",
      "      NOT previously demonstrated",
      "  (b) Phase boundary is new",
      "  (c) Practical calibration rule is new",
      "  (d) Connecting test-time-compute",
      "      (hot topic) to classical iterated",
      "      regularization is the contribution",
      "",
      "AAAI explicitly invites empirical,",
      "integrative, and analysis papers.",
      "This is that lane."],
     C_ORANGE)

card(sl, 9.05, 1.05, 4.0, 5.95,
     "What this paper is NOT",
     ["✗ A claim that KAN is special",
      "  (dropped entirely)",
      "",
      "✗ A new architecture",
      "",
      "✗ Better than SOTA denoisers",
      "  (DnCNN/NAFNet beat us in",
      "   absolute PSNR — that's fine)",
      "",
      "✗ A unified theory of intelligence",
      "  (that's the long game)",
      "",
      "✗ A paradigm shift",
      "  It's a wedge. A clean wedge."],
     C_RED)

# ── SLIDE 12: Summary & Ask ────────────────────────────────────────────────────
sl = blank(prs); fill_bg(sl, C_BG)
accent_bar(sl)
section_header(sl, "Summary")

box(sl, 0.35, 1.05, 12.63, 2.15, C_DARK_BOX)
box(sl, 0.35, 1.05, 12.63, 0.05, C_ACCENT)
txt(sl, "The one-line thesis (locked):",
    0.6, 1.12, 12.0, 0.38, size=15, bold=True, color=C_ACCENT)
txt(sl,
    "The optimal number of inference steps for an iterative denoiser is a universal power law\n"
    "in the noise level — K*(σ) ≈ C σ^α, with α set by the data spectrum, not the model.",
    0.6, 1.55, 12.0, 0.58, size=18, bold=True, color=C_WHITE)

bullets_summary = [
    (C_GREEN,  "DONE",     "Universality (5 heads + score, α=1.376±0.003) · Phase diagram · Theory framing · "
                           "Practical rule (99.83%) · 64px scale-up · 3-seed rigor"),
    (C_ORANGE, "RUNNING",  "β-sweep (4090) — controlled theory test · DDPM baseline (Colab) — cross-family capstone"),
    (C_ACCENT, "WEEKS 3–6","DnCNN baseline · BSD500 64px · ablation suite · paper rewrite"),
    (C_RED,    "DEADLINE", "~August 2026  (AAAI 2027)"),
]
for i, (c, label, body) in enumerate(bullets_summary):
    y = 3.4 + i * 0.88
    box(sl, 0.35, y, 1.5, 0.72, c)
    txt(sl, label, 0.42, y + 0.12, 1.38, 0.5, size=14, bold=True,
        color=C_BG, align=PP_ALIGN.CENTER)
    txt(sl, body, 2.05, y + 0.12, 10.8, 0.58, size=14.5, color=C_LIGHT)

txt(sl, "K*(σ) ≈ C σ^α      α = 1.376 ± 0.003      architecture-independent      data-dependent",
    0.5, 7.1, 12.3, 0.32, size=15, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
prs.save(str(OUT))
print(f"Saved: {OUT}  ({len(prs.slides)} slides)")
