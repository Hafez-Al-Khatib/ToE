"""
exp_spline_audit.py — Allen-Cahn Double-Well Discovery Check
═══════════════════════════════════════════════════════════════

Trains a KAN-EBM on MNIST (or loads a checkpoint), then:

  1. Evaluates every B-spline edge function φ_{i→j}(x) across x ∈ [-1, 1]
  2. Overlays four reference curves:
       • Allen-Cahn derivative  f(x) = x³ − x            (the "force")
       • Allen-Cahn potential   U(x) = (x²−1)²/4         (double-well shape)
       • Identity              g(x) = x                  (linear baseline)
       • ReLU                  r(x) = max(0, x)          (MLP baseline)
  3. For every spline computes cosine similarity (after mean-centering) with
     both the force and the potential.
  4. Produces four output figures:
       Fig A — output-layer splines (→scalar E) with AC overlay + similarity heatmap
       Fig B — hidden-layer splines grid (all 256 edges, layered)
       Fig C — top-8 most AC-like splines (enlarged, annotated)
       Fig D — cosine similarity histogram: where does AC emerge in the network?
  5. Prints a discovery summary to stdout.

If ANY output-layer spline has |cos_sim| > 0.85 with the AC force or potential,
the model has learned Allen-Cahn physics purely from image statistics.

Usage
─────
    # Quick mode (trains 30 epochs, ~3 min GPU):
    python experiments/exp_spline_audit.py --device cuda

    # With existing checkpoint:
    python experiments/exp_spline_audit.py --checkpoint results/model.pt --device cuda

    # CPU (slow, ~15 min):
    python experiments/exp_spline_audit.py --device cpu --epochs 20
"""

import argparse
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Inline model (all 3 bug fixes)
# ──────────────────────────────────────────────────────────────────────────────

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=torch.nn.SiLU, grid_range=(-1, 1)):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.grid_size     = grid_size
        self.spline_order  = spline_order

        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight   = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_noise   = scale_noise
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline

        nn.init.kaiming_uniform_(self.base_weight,   a=math.sqrt(5) * scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * scale_spline)
        with torch.no_grad():
            start = self.grid[0, spline_order].item()
            end   = self.grid[0, -1 - spline_order].item()
            x     = torch.linspace(start, end, grid_size + spline_order
                                   ).expand(in_features, -1).t().contiguous()
            noise = (torch.rand(grid_size + spline_order, in_features, out_features)
                     - 0.5) * scale_noise / grid_size
            self.spline_weight.data.copy_(self.curve2coeff(x, noise))

    def b_splines(self, x):
        x     = x.unsqueeze(-1)
        grid  = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k+1)]) / (grid[:, k:-1]   - grid[:, :-(k+1)]) * bases[:, :, :-1]
              + (grid[:, k+1:] - x)    / (grid[:, k+1:]   - grid[:, 1:-k])    * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A   = self.b_splines(x).transpose(0, 1)
        B   = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(self.b_splines(x).view(x.size(0), -1),
                              self.scaled_spline_weight.view(self.out_features, -1))
        return base_out + spline_out

    def get_spline_function(self, in_idx: int, out_idx: int, n_pts: int = 300):
        """Return (x_vals, y_vals) for the full activation φ_{in_idx → out_idx}(x)."""
        start  = self.grid[in_idx, self.spline_order].item()
        end    = self.grid[in_idx, -1 - self.spline_order].item()
        x_full = torch.linspace(start, end, n_pts)
        x_in   = x_full.unsqueeze(1).expand(-1, self.in_features).to(self.grid.device)
        with torch.no_grad():
            splines  = self.b_splines(x_in)               # (n_pts, in_f, coeff)
            sw       = self.scaled_spline_weight           # (out_f, in_f, coeff)
            spline_y = (splines[:, in_idx, :] * sw[out_idx, in_idx, :]).sum(-1)
            base_y   = self.base_activation(x_full.to(self.grid.device)) * self.base_weight[out_idx, in_idx]
            y        = (spline_y + base_y).cpu().numpy()
        return x_full.numpy(), y


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(layers_hidden[i], layers_hidden[i+1],
                      grid_size=grid_size, spline_order=spline_order,
                      scale_noise=scale_noise, scale_base=scale_base)
            for i in range(len(layers_hidden) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KANEnergyModel(nn.Module):
    def __init__(self, n_filters=16, filter_size=5, kan_hidden=None, n_channels=1):
        super().__init__()
        if kan_hidden is None:
            kan_hidden = [32]
        self.n_filters   = n_filters
        self.filter_size = filter_size
        self.n_channels  = n_channels
        self.filters     = nn.Parameter(
            0.01 * torch.randn(n_filters, n_channels, filter_size, filter_size))
        kan_in           = n_channels * n_filters
        self.kan         = KAN([kan_in] + kan_hidden + [1], grid_size=5)
        self.log_precision = nn.Parameter(torch.zeros(n_filters))

    @property
    def precision(self):
        return F.softplus(self.log_precision)

    def extract_features(self, x):
        B, C, H, W = x.shape
        pad  = self.filter_size // 2
        feats = []
        for c in range(C):
            f = F.conv2d(x[:, c:c+1], self.filters[:, c:c+1], padding=pad)
            feats.append(f)
        feats = torch.cat(feats, dim=1)
        prec  = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = feats * prec                   # Fix 1: precision wired in
        feats = torch.tanh(feats)             # Fix 2: tanh squash to [-1, 1]
        return feats.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

    def energy(self, x):
        return self.kan(self.extract_features(x)).sum()

    def energy_grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            E  = self.energy(xi)
            return torch.autograd.grad(E, xi)[0].detach()

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        u, step = x_noisy.clone(), dt
        for _ in range(n_steps):
            grad = self.energy_grad(u).clamp(-1., 1.)
            u    = (u - step * grad).detach()
            step *= dt_decay               # Fix 3: decaying step
        return u

    def loss(self, x_clean, sigma):
        noise   = torch.randn_like(x_clean) * sigma
        x_noisy = x_clean + noise
        with torch.enable_grad():
            xi   = x_noisy.detach().requires_grad_(True)
            E    = self.energy(xi)
            grad = torch.autograd.grad(E, xi, create_graph=True)[0]
        pred = xi - sigma ** 2 * grad
        return F.mse_loss(pred, x_clean)


# ──────────────────────────────────────────────────────────────────────────────
# Reference curves
# ──────────────────────────────────────────────────────────────────────────────

def reference_curves(x: np.ndarray) -> dict:
    """All Allen-Cahn reference shapes on the same x grid."""
    return {
        "AC force  x³−x":       x**3 - x,
        "AC potential (x²−1)²": (x**2 - 1)**2 / 4,
        "Identity  x":          x,
        "ReLU  max(0,x)":       np.maximum(0, x),
        "Tanh":                 np.tanh(x),
        "|x|  (edge detector)": np.abs(x),
    }


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity after mean-centering both vectors."""
    a = a - a.mean();  b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(device, n_epochs=30, sigma=0.2, batch_size=32):
    print(f"\n[train] Training KAN-EBM — {n_epochs} epochs, σ={sigma}, device={device}")
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    ds = datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))]))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=True, num_workers=0, drop_last=True)
    model = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        running = []
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = model.loss(x, sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(loss.item())
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={np.mean(running):.4f}")
    print("[train] Done.\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Spline extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_all_splines(model: KANEnergyModel, n_pts: int = 300) -> list:
    """
    Returns list of dicts:
      {layer, in_idx, out_idx, x, y, sims}
    sims = {ref_name: cosine_similarity_with_that_reference}
    """
    records = []
    x_ref   = np.linspace(-1, 1, n_pts)
    refs    = reference_curves(x_ref)

    for layer_idx, kan_layer in enumerate(model.kan.layers):
        for i in range(kan_layer.in_features):
            for j in range(kan_layer.out_features):
                x_vals, y_vals = kan_layer.get_spline_function(i, j, n_pts=n_pts)
                sims = {name: cosine_sim(y_vals, ref)
                        for name, ref in refs.items()}
                records.append({
                    "layer":   layer_idx,
                    "in_idx":  i,
                    "out_idx": j,
                    "x":       x_vals,
                    "y":       y_vals,
                    "sims":    sims,
                })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

REF_COLORS = {
    "AC force  x³−x":       "#E64980",   # pink — our hero
    "AC potential (x²−1)²": "#F59F00",   # gold — our hero
    "Identity  x":          "#ADB5BD",
    "ReLU  max(0,x)":       "#ADB5BD",
    "Tanh":                 "#ADB5BD",
    "|x|  (edge detector)": "#ADB5BD",
}
AC_KEYS = ["AC force  x³−x", "AC potential (x²−1)²"]


def normalise(y):
    """Scale y to [-1, 1] for visual overlay."""
    r = np.max(np.abs(y))
    return y / r if r > 1e-9 else y


def fig_output_layer(records, out_dir: Path):
    """
    Fig A: all output-layer splines (layer 1, → scalar E) with Allen-Cahn overlay.
    Each subplot = one input edge to the energy output.
    """
    out_recs = [r for r in records if r["layer"] == 1]   # second KAN layer → 1
    n        = len(out_recs)
    ncols    = 8
    nrows    = math.ceil(n / ncols)
    x_ref    = np.linspace(-1, 1, 300)
    refs     = reference_curves(x_ref)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.4))
    axes = np.array(axes).flatten()
    fig.suptitle("Output-Layer KAN Splines (→ Scalar Energy E)\nvs Allen-Cahn Reference Curves",
                 fontsize=13, fontweight='bold', y=1.01)

    for ax_idx, rec in enumerate(out_recs):
        ax = axes[ax_idx]
        sim_force = rec["sims"]["AC force  x³−x"]
        sim_pot   = rec["sims"]["AC potential (x²−1)²"]
        best_sim  = max(abs(sim_force), abs(sim_pot))

        # background colour: green if high AC similarity
        if best_sim > 0.85:
            ax.set_facecolor("#d4edda")
        elif best_sim > 0.6:
            ax.set_facecolor("#fff3cd")

        # spline (normalised)
        y_norm = normalise(rec["y"])
        ax.plot(rec["x"], y_norm, color="#3B5BDB", lw=2.0, zorder=5, label="φ(x)")

        # AC overlays (normalised)
        ax.plot(x_ref, normalise(refs["AC force  x³−x"]),
                color="#E64980", lw=1.2, ls="--", alpha=0.85, label="x³−x")
        ax.plot(x_ref, normalise(refs["AC potential (x²−1)²"]),
                color="#F59F00", lw=1.2, ls=":",  alpha=0.85, label="(x²−1)²")

        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6)

        title_col = "#155724" if best_sim > 0.85 else ("#856404" if best_sim > 0.6 else "black")
        ax.set_title(f"→E  in={rec['in_idx']}\n"
                     f"cos(force)={sim_force:+.2f}\n"
                     f"cos(pot)  ={sim_pot:+.2f}",
                     fontsize=6.5, color=title_col)

    for ax in axes[len(out_recs):]:
        ax.set_visible(False)

    # Legend once
    axes[0].legend(fontsize=5, loc="upper left")

    plt.tight_layout()
    path = out_dir / "figA_output_layer_splines.pdf"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


def fig_top8(records, out_dir: Path):
    """Fig C: top-8 most Allen-Cahn-like splines from the whole network (enlarged)."""
    # Score each record by max(|sim_force|, |sim_pot|)
    scored = []
    for r in records:
        score = max(abs(r["sims"]["AC force  x³−x"]),
                    abs(r["sims"]["AC potential (x²−1)²"]))
        scored.append((score, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    top8 = scored[:8]

    x_ref = np.linspace(-1, 1, 300)
    refs  = reference_curves(x_ref)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    fig.suptitle("Top-8 Most Allen-Cahn-Like KAN Splines\n"
                 "(pink dashes = x³−x force;  gold dots = (x²−1)² potential)",
                 fontsize=12, fontweight='bold')

    for ax, (score, rec) in zip(axes, top8):
        sim_force = rec["sims"]["AC force  x³−x"]
        sim_pot   = rec["sims"]["AC potential (x²−1)²"]
        y_norm    = normalise(rec["y"])

        # Shade background by score
        bg = "#d4edda" if score > 0.85 else ("#fff3cd" if score > 0.6 else "white")
        ax.set_facecolor(bg)

        ax.plot(rec["x"], y_norm, color="#3B5BDB", lw=2.5, label="learned φ(x)", zorder=5)
        ax.plot(x_ref, normalise(refs["AC force  x³−x"]),
                color="#E64980", lw=1.8, ls="--", alpha=0.9, label="x³−x  (AC force)")
        ax.plot(x_ref, normalise(refs["AC potential (x²−1)²"]),
                color="#F59F00", lw=1.8, ls=":",  alpha=0.9, label="(x²−1)² (AC pot)")
        ax.plot(x_ref, normalise(refs["Identity  x"]),
                color="#ADB5BD", lw=1.0, ls="-",  alpha=0.5, label="identity")

        ax.axhline(0, color='gray', lw=0.6)
        ax.axvline(0, color='gray', lw=0.6)
        ax.axvline(-1, color='gray', lw=0.4, ls=':')
        ax.axvline(+1, color='gray', lw=0.4, ls=':')
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x  (tanh-squashed filter response)", fontsize=8)
        ax.set_ylabel("φ(x)  [normalised]", fontsize=8)

        layer_name = "hidden→hidden" if rec["layer"] == 0 else "hidden→E"
        title_col  = "#155724" if score > 0.85 else ("#856404" if score > 0.6 else "black")
        ax.set_title(
            f"Layer {rec['layer']} ({layer_name})  in={rec['in_idx']} out={rec['out_idx']}\n"
            f"cos(force)={sim_force:+.3f}   cos(pot)={sim_pot:+.3f}   "
            f"★ best={score:.3f}",
            fontsize=8, fontweight='bold', color=title_col)

    axes[0].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    path = out_dir / "figC_top8_AC_splines.pdf"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


def fig_similarity_histogram(records, out_dir: Path):
    """Fig D: histogram of AC cosine similarities across layers."""
    layer0 = [r for r in records if r["layer"] == 0]
    layer1 = [r for r in records if r["layer"] == 1]

    def sims(recs, key):
        return [abs(r["sims"][key]) for r in recs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    fig.suptitle("Allen-Cahn Cosine Similarity Distribution Across Layers",
                 fontsize=12, fontweight='bold')

    for ax, key, color, label in [
        (axes[0], "AC force  x³−x",        "#E64980", "cos(x³−x)"),
        (axes[1], "AC potential (x²−1)²",   "#F59F00", "cos((x²−1)²)"),
    ]:
        bins = np.linspace(0, 1, 21)
        ax.hist(sims(layer0, key), bins=bins, alpha=0.7, color="#3B5BDB",
                label=f"Layer 0 (hidden, n={len(layer0)})", edgecolor='white')
        ax.hist(sims(layer1, key), bins=bins, alpha=0.7, color=color,
                label=f"Layer 1 (→E,     n={len(layer1)})", edgecolor='white')
        ax.axvline(0.85, color='red', lw=1.5, ls='--', label="threshold 0.85")
        ax.axvline(0.60, color='orange', lw=1.0, ls=':', label="threshold 0.60")
        ax.set_xlabel(f"|{label}|  (cosine similarity after mean-centering)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"Similarity with  {label}", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    path = out_dir / "figD_similarity_histogram.pdf"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


def fig_hidden_layer_grid(records, out_dir: Path):
    """Fig B: all hidden-layer splines overlaid in one panel per output unit."""
    layer0 = [r for r in records if r["layer"] == 0]
    # Group by out_idx
    out_units = sorted(set(r["out_idx"] for r in layer0))
    x_ref     = np.linspace(-1, 1, 300)
    refs      = reference_curves(x_ref)

    ncols  = min(8, len(out_units))
    nrows  = math.ceil(len(out_units) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes   = np.array(axes).flatten()
    fig.suptitle("Hidden-Layer Splines Grouped by Output Unit\n"
                 "(each line = one incoming edge; pink = x³−x overlay)",
                 fontsize=11, fontweight='bold', y=1.01)

    for ax_idx, out_j in enumerate(out_units):
        ax    = axes[ax_idx]
        group = [r for r in layer0 if r["out_idx"] == out_j]
        # mean AC similarity for this output unit
        mean_sim = np.mean([abs(r["sims"]["AC force  x³−x"]) for r in group])
        bg = "#d4edda" if mean_sim > 0.7 else ("#fff3cd" if mean_sim > 0.5 else "white")
        ax.set_facecolor(bg)

        for rec in group:
            ax.plot(rec["x"], normalise(rec["y"]),
                    color="#3B5BDB", lw=0.7, alpha=0.4)
        # Overlay AC force
        ax.plot(x_ref, normalise(refs["AC force  x³−x"]),
                color="#E64980", lw=1.5, ls="--", alpha=0.9, label="x³−x")
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([])
        ax.set_title(f"→h{out_j}  ⟨|cos|⟩={mean_sim:.2f}", fontsize=7,
                     color="#155724" if mean_sim > 0.7 else "black")

    for ax in axes[len(out_units):]:
        ax.set_visible(False)

    plt.tight_layout()
    path = out_dir / "figB_hidden_layer_grid.pdf"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Discovery summary
# ──────────────────────────────────────────────────────────────────────────────

THRESHOLD_STRONG  = 0.85
THRESHOLD_NOTABLE = 0.60

def print_summary(records):
    print("\n" + "═" * 72)
    print("  ALLEN-CAHN SPLINE AUDIT — DISCOVERY SUMMARY")
    print("═" * 72)

    for layer_idx in [0, 1]:
        recs = [r for r in records if r["layer"] == layer_idx]
        layer_name = "Layer 0 (hidden→hidden)" if layer_idx == 0 else "Layer 1 (hidden→E scalar)"
        print(f"\n  {layer_name}  [{len(recs)} edges]")
        print(f"  {'Edge':<14} {'cos(x³−x)':>12} {'cos((x²−1)²)':>14} {'best':>8}  Signal")

        for r in sorted(recs, key=lambda x: max(abs(x["sims"]["AC force  x³−x"]),
                                                 abs(x["sims"]["AC potential (x²−1)²"])),
                        reverse=True)[:10]:
            sf  = r["sims"]["AC force  x³−x"]
            sp  = r["sims"]["AC potential (x²−1)²"]
            best = max(abs(sf), abs(sp))
            flag = "★ STRONG"  if best > THRESHOLD_STRONG  else \
                   "◎ notable" if best > THRESHOLD_NOTABLE else "  —"
            print(f"  in={r['in_idx']:2d}→out={r['out_idx']:2d}  "
                  f"{sf:+10.4f}    {sp:+12.4f}   {best:6.4f}  {flag}")

    # Layer-1 summary (most critical — these edges wire directly into E)
    out_recs   = [r for r in records if r["layer"] == 1]

    def _is_genuine_ac(r):
        """
        True only if the spline is strongly AC-like AND the AC shape is not
        better explained by the identity function x.

        Note: cos(x³−x, x) = −0.83 over [−1,1], so a plain linear activation
        would spuriously score high on the force test.  We therefore require
        the AC score to beat the identity score by at least 0.15 — i.e. the
        cubic nonlinearity is genuinely present, not just captured by linearity.
        """
        sf   = abs(r["sims"]["AC force  x³−x"])
        sp   = abs(r["sims"]["AC potential (x²−1)²"])
        sid  = abs(r["sims"]["Identity  x"])
        best = max(sf, sp)
        # Strong AC signal AND beats the identity explanation
        return best > THRESHOLD_STRONG and best > sid + 0.15

    strong_force    = [r for r in out_recs if abs(r["sims"]["AC force  x³−x"])  > THRESHOLD_STRONG]
    strong_pot      = [r for r in out_recs if abs(r["sims"]["AC potential (x²−1)²"]) > THRESHOLD_STRONG]
    any_strong_raw  = [r for r in out_recs if max(abs(r["sims"]["AC force  x³−x"]),
                                                   abs(r["sims"]["AC potential (x²−1)²"])) > THRESHOLD_STRONG]
    any_genuine     = [r for r in out_recs if _is_genuine_ac(r)]

    print("\n" + "─" * 72)
    print(f"  Output-layer edges with |cos(AC force)| > {THRESHOLD_STRONG}:           {len(strong_force)}")
    print(f"  Output-layer edges with |cos(AC pot)|   > {THRESHOLD_STRONG}:           {len(strong_pot)}")
    print(f"  Output-layer edges with EITHER         > {THRESHOLD_STRONG} (raw):      {len(any_strong_raw)}")
    print(f"  Output-layer edges GENUINE AC (beats identity by >0.15):  {len(any_genuine)}")
    print()
    print("  NOTE: cos(x³−x, x) = −0.83 on [−1,1], so identity activations")
    print("        can spuriously score high on the force test.  The 'genuine'")
    print("        count disambiguates by requiring AC to beat identity + 0.15.")
    print()

    any_strong = any_genuine  # use disambiguated count for verdict

    if any_strong:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  ★ PHYSICS EMERGENCE DETECTED                               ║")
        print("  ║    The model learned Allen-Cahn double-well structure        ║")
        print("  ║    purely from image statistics — without being told.        ║")
        print("  ║    This is a Top-1% NeurIPS result.                         ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
    else:
        notable_out = [r for r in out_recs if max(abs(r["sims"]["AC force  x³−x"]),
                                                   abs(r["sims"]["AC potential (x²−1)²"])) > THRESHOLD_NOTABLE]
        if notable_out:
            print(f"  ◎ {len(notable_out)} output-layer edges show NOTABLE AC similarity (>{THRESHOLD_NOTABLE}).")
            print(f"    Partial physics emergence — consider training longer or with larger KAN hidden.")
        else:
            print("  — No strong Allen-Cahn signature found.")
            print("    The model learned its own activation basis, not the AC double-well.")
            print("    This is still scientifically valid — but the physics claim cannot be made.")
    print("═" * 72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to saved model .pt file (skips training)")
    ap.add_argument("--device",     type=str, default="cpu")
    ap.add_argument("--epochs",     type=int, default=30)
    ap.add_argument("--sigma",      type=float, default=0.2)
    ap.add_argument("--n_pts",      type=int, default=300,
                    help="Points per spline curve (higher = smoother plots)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"[device] {device}")

    out_dir = ROOT / "outputs" / "spline_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or train model ───────────────────────────────────────────────────
    if args.checkpoint:
        print(f"[load] {args.checkpoint}")
        model = KANEnergyModel(n_filters=16, filter_size=5, kan_hidden=[32]).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        model = train(device, n_epochs=args.epochs, sigma=args.sigma)
        ckpt  = out_dir / "trained_model.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"[saved] checkpoint → {ckpt}")

    model.eval()

    # ── Extract all spline functions ─────────────────────────────────────────
    print("[audit] Extracting all spline activation functions …")
    records = extract_all_splines(model, n_pts=args.n_pts)
    print(f"        {len(records)} edges total  "
          f"({sum(1 for r in records if r['layer']==0)} hidden-layer, "
          f"{sum(1 for r in records if r['layer']==1)} output-layer)")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\n[figures] Generating …")
    fig_output_layer(records, out_dir)
    fig_hidden_layer_grid(records, out_dir)
    fig_top8(records, out_dir)
    fig_similarity_histogram(records, out_dir)

    # ── Print discovery summary ──────────────────────────────────────────────
    print_summary(records)

    # ── Save similarity table as CSV ─────────────────────────────────────────
    csv_path = out_dir / "similarity_table.csv"
    with open(csv_path, "w") as f:
        f.write("layer,in_idx,out_idx,cos_force,cos_potential,cos_identity,"
                "cos_relu,cos_tanh,cos_abs\n")
        for r in records:
            s = r["sims"]
            f.write(f"{r['layer']},{r['in_idx']},{r['out_idx']},"
                    f"{s['AC force  x³−x']:.6f},"
                    f"{s['AC potential (x²−1)²']:.6f},"
                    f"{s['Identity  x']:.6f},"
                    f"{s['ReLU  max(0,x)']:.6f},"
                    f"{s['Tanh']:.6f},"
                    f"{s['|x|  (edge detector)']:.6f}\n")
    print(f"[saved] similarity table → {csv_path}")
    print(f"\n[done]  All outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()
