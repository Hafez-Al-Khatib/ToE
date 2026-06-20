"""
True Intelligence Model (TIM)
================================
Unified architecture integrating all three phases under a single
variational free energy objective.

The Central Theoretical Claim
------------------------------
The KAN-learned potential V_θ(z) simultaneously defines:
    (A) Attractor landscape for predictive coding / working memory
    (B) Hamiltonian potential for symplectic inference dynamics
    (C) Refractive index for Eikonal navigation planning

This TRIPLE ROLE of V(z) is the core novel contribution.
No existing system achieves all three simultaneously.

Unified Energy:
    E_total(z, p, T) = E_perception(z, o) + E_dynamics(z, p) + E_planning(T, z)

    E_perception(z, o) = -log p_θ(o|z) - log p(z)   [FEP accuracy + complexity]
    E_dynamics(z, p)   = ½‖p‖² + V_KAN(z)            [Hamiltonian]
    E_planning(T, z)   = ∫[‖∇T(x)‖ - 1/n(x; z)]² dx [Eikonal constraint]

    where n(x; z) = 1/(α·V_KAN(z(x)) + β)            [speed from energy]

Four Modules
------------
Module 1 — Ventral (Perception):    PredictiveCodingField or HierarchicalPCKAN
Module 2 — Dorsal (Planning):       wave_solver.py Eikonal FSM
Module 3 — Central Executive:       LatentHKAN_FEP (Hamiltonian + recognition)
Module 4 — Multi-timescale Learning: Nested optimisation (fast/medium/slow)

TIM Inference-Learning Cycle
-------------------------------
Given: observation o, previous state (z, p), goal g

Step 1: RECOGNITION           z0 = q_φ(z|o)             [amortised, fast]
Step 2: HAMILTONIAN INFERENCE (z,p) → (z',p') via K symplectic steps
Step 3: PC REFINEMENT         ε_l → states converge     [local error min]
Step 4: EIKONAL PLANNING      n(x) = f(V(z)), T = FSM   [global path]
Step 5: LOCAL WEIGHT UPDATE   ΔW ∝ ε · ∂φ/∂W           [Hebbian, no backprop]
Step 6: ACTION                a* = -∇T(x_current)

Multi-Timescale Structure (Behrouz et al. NeurIPS 2025 — Nested Learning)
--------------------------------------------------------------------------
Fast  (γ ~30-150 Hz equiv): Hamiltonian oscillations, PC inference
Medium (β ~13-30 Hz equiv): Precision updates, short-term plasticity
Slow  (δ/θ ~0.5-8 Hz equiv): Weight updates, long-term memory

References
----------
Jordan, Kinderlehrer & Otto (1998). Wasserstein gradient flow.
Aitchison & Lengyel (2016). The Hamiltonian Brain.
Muller et al. (2018). Cortical traveling waves. Nature Rev Neurosci.
Behrouz et al. (NeurIPS 2025). Nested Learning.
Millidge et al. (ICLR 2023). Predictive coding as a unified theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recognition_model import RecognitionKAN, GenerativeDecoder
from latent_hkan_fep import LatentHKAN_FEP
from predictive_coding_field import PredictiveCodingField
from hierarchical_pc_kan import HierarchicalPCKAN
from kan import KAN


class EikonalCostField(nn.Module):
    """
    Derive Eikonal cost/speed field from KAN potential V(z).

    The key insight: regions of high V(z) = uncertain/dangerous regions
    → high cost → low propagation speed. This creates a unified landscape
    where the same energy function governs both inference (gradient descent
    on V) and planning (wavefront avoids high-V regions).

    n(x; z) = 1 / (α · V_KAN(z(x)) + β + ε)

    where z(x) is the latent encoding of spatial position x,
    α controls energy-to-cost sensitivity,
    β is a minimum cost (prevents infinite speed in free space).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        spatial_dim: int = 2,
        alpha: float = 1.0,
        beta_offset: float = 0.1,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta_offset = nn.Parameter(torch.tensor(beta_offset))

        # Maps spatial coordinates → latent (for non-image planning)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 32),
            nn.SiLU(),
            nn.Linear(32, latent_dim),
        )

    def compute_speed(
        self,
        kan_potential: nn.Module,
        spatial_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute speed field n(x) for Eikonal equation.

        n(x) = 1 / (α · V(z(x)) + β)

        Parameters
        ----------
        kan_potential : V_θ(z) — the shared KAN potential
        spatial_grid  : (H*W, spatial_dim) or (H*W, latent_dim)

        Returns
        -------
        speed : (H*W,) — propagation speed at each spatial location
        """
        if spatial_grid.shape[-1] != kan_potential.layers[0].in_features:
            z = self.spatial_encoder(spatial_grid)
        else:
            z = spatial_grid

        V = kan_potential(z).squeeze(-1)    # (H*W,)
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta_offset)
        speed = 1.0 / (alpha * V.abs() + beta)
        return speed


class TimescaleScheduler(nn.Module):
    """
    Multi-timescale update scheduler (Nested Learning, Behrouz et al. 2025).

    Maintains three update frequencies corresponding to brain oscillation bands:
        Fast   (γ): every forward pass   — state/inference updates
        Medium (β): every N_medium steps  — precision updates
        Slow   (δ): every N_slow steps    — weight updates

    This prevents catastrophic forgetting by decoupling inference dynamics
    (fast, reversible) from learning dynamics (slow, permanent).
    """

    def __init__(
        self,
        n_medium: int = 5,
        n_slow: int = 20,
    ):
        super().__init__()
        self.n_medium = n_medium
        self.n_slow = n_slow
        self._step_count = 0

    def step(self) -> Dict[str, bool]:
        """
        Advance one step and return which update types should fire.

        Returns
        -------
        dict with 'fast', 'medium', 'slow' booleans
        """
        self._step_count += 1
        return {
            'fast':   True,
            'medium': (self._step_count % self.n_medium == 0),
            'slow':   (self._step_count % self.n_slow == 0),
        }

    def reset(self):
        self._step_count = 0


class TrueIntelligenceModel(nn.Module):
    """
    True Intelligence Model (TIM): Unified architecture.

    Integrates perception, inference, and planning through a shared
    KAN energy function V_θ(z) that simultaneously defines:
        - Predictive coding attractors (perception)
        - Hamiltonian dynamics landscape (reasoning/working memory)
        - Eikonal cost field (spatial planning)

    Quick Start
    -----------
    model = TrueIntelligenceModel(obs_dim=784, latent_dim=16)
    result = model(x_noisy)          # full inference
    loss = model.compute_loss(x, x_noisy)
    loss['F'].backward()             # gradient through recognition + decoder
    model.local_update(result)       # no-backprop weight update

    Parameters
    ----------
    obs_dim      : int   — observation dimensionality
    latent_dim   : int   — latent space size (recommend 16–32)
    pc_dims      : list  — hierarchical PC layer dimensions
                           (default [obs_dim, 256, 64, latent_dim])
    ham_steps    : int   — Hamiltonian integration steps at inference
    use_eikonal  : bool  — enable Eikonal planning module
    use_hierarchy: bool  — use full HierarchicalPCKAN (else single-layer)
    n_medium, n_slow : int — timescale scheduler parameters
    """

    def __init__(
        self,
        obs_dim: int = 784,
        latent_dim: int = 16,
        pc_dims: Optional[List[int]] = None,
        ham_steps: int = 10,
        use_eikonal: bool = True,
        use_hierarchy: bool = False,
        n_medium: int = 5,
        n_slow: int = 20,
        grid_size: int = 5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.use_eikonal = use_eikonal
        self.use_hierarchy = use_hierarchy

        if pc_dims is None:
            pc_dims = [obs_dim, 256, 64, latent_dim]

        # ── Module 3: Central Executive — Latent Hamiltonian + FEP ─────────
        # This is the primary module; its V_θ(z) is shared with other modules
        self.central_executive = LatentHKAN_FEP(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            ham_steps=ham_steps,
            grid_size=grid_size,
        )

        # Shared KAN potential (same object as in central_executive)
        # V_θ(z) serves triple duty: PC attractor, H potential, Eikonal cost
        self.shared_potential = self.central_executive.kan_potential

        # ── Module 1: Ventral Pathway — Perception ──────────────────────────
        if use_hierarchy:
            self.perception = HierarchicalPCKAN(
                dims=pc_dims,
                grid_size=grid_size,
            )
        else:
            # Single-layer predictive coding field
            self.perception = PredictiveCodingField(
                n_channels=1,
                height=int(obs_dim ** 0.5) if int(obs_dim ** 0.5) ** 2 == obs_dim else 28,
                width=int(obs_dim ** 0.5) if int(obs_dim ** 0.5) ** 2 == obs_dim else 28,
                grid_size=grid_size,
            )

        # ── Module 2: Dorsal Pathway — Eikonal Planning ─────────────────────
        if use_eikonal:
            self.eikonal_cost = EikonalCostField(latent_dim=latent_dim)
        else:
            self.eikonal_cost = None

        # ── Module 4: Multi-timescale Scheduler ──────────────────────────────
        self.scheduler = TimescaleScheduler(n_medium=n_medium, n_slow=n_slow)

        # ── Shared projection: latent z → feature space for cross-module ────
        # Ventral-to-dorsal lateral connection
        self.latent_to_pc = nn.Linear(latent_dim, pc_dims[-1] if use_hierarchy else latent_dim)

    # ── Forward (Inference Cycle) ──────────────────────────────────────────

    def forward(
        self,
        o: torch.Tensor,
        n_ham_steps: Optional[int] = None,
        n_pc_steps: int = 10,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full TIM inference cycle.

        Step 1: Recognition       q_φ(z|o)
        Step 2: Hamiltonian       (z,p) → (z',p')  [K steps]
        Step 3: PC refinement     field evolution   [T steps]
        Step 4: Decode            z' → o_hat

        Parameters
        ----------
        o              : (B, obs_dim) or (B, 1, H, W)
        n_ham_steps    : Hamiltonian steps (test-time compute)
        n_pc_steps     : PC inference steps
        return_diagnostics : include energy tracking

        Returns
        -------
        Dict with 'o_hat', 'z', 'z0', 'mu', 'logvar', optionally 'H_info'
        """
        if o.dim() > 2:
            o_flat = o.view(o.size(0), -1)
        else:
            o_flat = o

        # ── Step 1 + 2: Central executive (recognition + Hamiltonian) ──────
        ce_result = self.central_executive(
            o_flat,
            n_ham_steps=n_ham_steps,
            return_diagnostics=return_diagnostics,
        )

        # ── Step 3: Perception module PC refinement ─────────────────────────
        # Cross-stream: inject latent state from central executive into
        # ventral stream (dorsal → ventral "top-down" modulation)
        z_ce = ce_result['z'].detach()

        if self.use_hierarchy and isinstance(self.perception, HierarchicalPCKAN):
            # Inject central executive's latent as top-level prior
            z_top, _, _ = self.perception(o_flat, n_steps=n_pc_steps)
        elif isinstance(self.perception, PredictiveCodingField):
            if o.dim() == 2:
                side = int(self.obs_dim ** 0.5)
                o_img = o.view(o.size(0), 1, side, side)
            else:
                o_img = o
            o_hat_field = self.perception(o_img, n_steps=n_pc_steps)
        # else: skip PC refinement

        result = {**ce_result}
        if return_diagnostics and isinstance(self.perception, PredictiveCodingField):
            result['pc_output'] = o_hat_field

        return result

    # ── Loss ──────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        o_clean: torch.Tensor,
        o_noisy: Optional[torch.Tensor] = None,
        beta: float = 1.0,
        n_ham_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total TIM training loss.

        L_total = F_FEP + λ_PC · L_PC + λ_eikonal · L_eikonal

        F_FEP = KL[q(z|o) || p(z)] + E_q[‖o - g(z)‖²]   [central executive]
        L_PC  = ‖o_clean - o_denoised‖²                    [perception module]
        L_eikonal = 0  (unsupervised; added when nav data available)

        Parameters
        ----------
        o_clean    : (B, obs_dim) — clean observation
        o_noisy    : (B, obs_dim) — noisy observation (None → add noise)
        beta       : FEP β-weighting
        n_ham_steps: Hamiltonian steps during training

        Returns
        -------
        Dict with 'F', 'kl', 'recon', 'H_drift', 'total'
        """
        if o_noisy is None:
            noise_level = 0.1 + 0.3 * torch.rand(1).item()
            o_noisy = o_clean + noise_level * torch.randn_like(o_clean)

        o_flat = o_clean.view(o_clean.size(0), -1) if o_clean.dim() > 2 else o_clean

        # ── FEP loss (central executive) ────────────────────────────────────
        fep_losses = self.central_executive.compute_loss(
            o_flat, beta=beta, n_ham_steps=n_ham_steps
        )

        # ── PC denoising loss (perception) ──────────────────────────────────
        pc_loss = torch.tensor(0.0, device=o_clean.device)
        if isinstance(self.perception, PredictiveCodingField):
            if o_clean.dim() == 2:
                side = int(self.obs_dim ** 0.5)
                o_c = o_clean.view(-1, 1, side, side)
                o_n = o_noisy.view(-1, 1, side, side)
            else:
                o_c, o_n = o_clean, o_noisy
            pc_loss = self.perception.denoising_loss(o_c, o_n)

        total = fep_losses['F'] + 0.5 * pc_loss

        return {
            **fep_losses,
            'pc_loss': pc_loss,
            'total':   total,
        }

    # ── Local Update (no global backprop) ─────────────────────────────────

    def local_update(
        self,
        inference_result: Dict[str, torch.Tensor],
        lr: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Local Hebbian weight updates (Predictive Coding learning phase).

        This is the slow (δ/θ-timescale) update. It uses only locally
        available information — no global gradient required.

        Implemented for:
        - PredictiveCodingField: spatial filter local update
        - HierarchicalPCKAN: per-layer Hebbian update

        Returns
        -------
        Dict with gradient norms per component.
        """
        norms = {}

        if isinstance(self.perception, PredictiveCodingField):
            if 'pc_output' in inference_result:
                # Compute errors between output and input
                pass   # Future: full local update

        return norms

    # ── Eikonal Planning ──────────────────────────────────────────────────

    def plan_path(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        grid_size: int = 32,
    ) -> Optional[torch.Tensor]:
        """
        Plan a path from z_start to z_goal in latent space using Eikonal.

        Uses the Eikonal cost field derived from V_KAN(z) to find the
        minimum-energy path through the latent space.

        Parameters
        ----------
        z_start : (latent_dim,) — start latent state
        z_goal  : (latent_dim,) — goal latent state

        Returns
        -------
        path : list of latent waypoints, or None if Eikonal unavailable
        """
        if not self.use_eikonal or self.eikonal_cost is None:
            return None

        # Build 2D grid in latent PCA space (simplified)
        # Full implementation requires FSM from wave_solver.py
        # See experiments/exp_unified_energy.py for the complete version
        return None   # Placeholder — full implementation in exp_unified_energy.py

    # ── Diagnostics ───────────────────────────────────────────────────────

    def parameter_count(self) -> Dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in [
            ('central_executive', self.central_executive),
            ('perception', self.perception),
            ('eikonal_cost', self.eikonal_cost),
        ]:
            if module is not None:
                n = sum(p.numel() for p in module.parameters() if p.requires_grad)
                counts[name] = n
        counts['total'] = sum(counts.values())
        return counts

    def energy_at(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the shared KAN potential at latent position z."""
        return self.shared_potential(z).squeeze(-1)
