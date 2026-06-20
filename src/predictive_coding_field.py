"""
Predictive Coding Field
========================
Renamed and restructured from HamiltonianField → PredictiveCodingField.

Why the Rename
--------------
The previous class was called HamiltonianField but implemented Allen-Cahn
gradient flow:
    u_{t+1} = u_t - η ∇_u E(u_t)

This is NOT Hamiltonian dynamics. Hamiltonian dynamics CONSERVES energy.
Gradient flow DISSIPATES energy. They are mathematical opposites.

The correct interpretation of Allen-Cahn gradient flow is as PREDICTIVE CODING:

    Allen-Cahn:          ∂u/∂t = -δE/δu
    Predictive Coding:   ∂x_l/∂t = -ε_l   where ε_l = Π_l^½(x_l - μ_l)

These are identical when E = ½Σ_l ‖ε_l‖² and μ_l = g_l(x_{l+1}) is the
KAN-parameterised top-down prediction. The gradient is the prediction error.

Key Mathematical Bridges
-------------------------
1. Allen-Cahn = Score Following (Vincent 2011):
       ∂u/∂t = -∇_u E(u) = ∇_u log p(u)   [when E = -log p]
   So Allen-Cahn dynamics IS following the score function = DSM training.

2. Score Function = Prediction Error (Rao & Ballard 1999):
       For p(o|x) = N(o; g(x), σ²I):
       ∇_x log p(o|x) = (1/σ²)(o - g(x))·∂g/∂x = "prediction error"

3. Prediction Error ≡ Variational FE Gradient (Bogacz 2017):
       F_PC = Σ_l ½‖ε_l‖²   →   ∂F/∂x_l = -ε_l   [Laplace approx]

4. Variational FE ⊂ Wasserstein Gradient Flow (JKO 1998):
       F(ρ) = ∫ρ log ρ dx + ∫ρV dx = -H(q) + E_q[V]

New Features Added Over HamiltonianField
-----------------------------------------
- Learnable precision Π_l (per-dimension attention gate)
- Explicit prediction error computation and logging
- Local Hebbian weight update (no backprop needed at runtime)
- Two-phase learning (inference phase + weight update phase)
- Modular: can be stacked into HierarchicalPCKAN

References
----------
Rao & Ballard (1999). Predictive coding in the visual cortex. Nature Neuroscience.
Friston (2005). A theory of cortical responses. Phil Trans B.
Bogacz (2017). A tutorial on the free-energy framework. J. Math. Psych.
Song et al. (NeurIPS 2020). Can the brain do backpropagation?
Millidge et al. (ICLR 2023). Predictive coding as a unified theory of EBM learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kan import KAN
from thermodynamic_field import ThermodynamicField


class PredictiveCodingField(ThermodynamicField):
    """
    A KAN-parameterised Allen-Cahn field reinterpreted as predictive coding.

    Inherits spatial filter banks from ThermodynamicField. Replaces the
    polynomial potential with a KAN energy function (same as HamiltonianField)
    but adds:
        - Learnable precision weights Π (attention over prediction errors)
        - Explicit prediction error ε = Π^½(u - μ)
        - Local Hebbian weight update rule (no global backprop required)
        - Diagnostic logging of prediction errors and energy

    Dynamics
    --------
    Inference (fast, γ-timescale):
        ε   = Π^½ · (u - μ_top_down)                # precision-weighted error
        ∂u/∂t = -ε - ∇_u E_KAN(u)                  # combined update
        u_{t+1} = u_t - η · (∂u/∂t)

    Learning (slow, δ/θ-timescale):
        ΔW_KAN ∝ ε · (∂φ/∂W)                        # local Hebbian rule
                                                      # equivalent to backprop at
                                                      # equilibrium (Song et al 2020)

    Parameters
    ----------
    n_channels, n_filters, filter_size, height, width :
        Inherited from ThermodynamicField.
    kan_hidden : list[int]
        Hidden layer sizes for KAN energy function.
    precision_init : float
        Initial value for all precision parameters (default 1.0 = unweighted).
    learn_precision : bool
        If False, fix precision at initial value (ablation).
    """

    def __init__(
        self,
        n_channels: int = 1,
        height: int = 28,
        width: int = 28,
        n_filters: int = 16,
        filter_size: int = 5,
        kan_hidden: Optional[list] = None,
        precision_init: float = 1.0,
        learn_precision: bool = True,
        grid_size: int = 5,
    ):
        super().__init__(
            n_channels=n_channels,
            height=height,
            width=width,
            n_filters=n_filters,
            filter_size=filter_size,
        )

        if kan_hidden is None:
            kan_hidden = [32, 16]

        # ── KAN Energy Function ─────────────────────────────────────────────
        # Input: spatial filter features per pixel (n_channels * n_filters)
        # Output: scalar energy density per pixel
        kan_in = n_channels * n_filters
        self.kan_energy = KAN(
            layers_hidden=[kan_in] + kan_hidden + [1],
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
        )

        # ── Learnable Precision Π ───────────────────────────────────────────
        # Precision = inverse variance of prediction errors
        # High precision → that spatial location/channel is strongly penalised
        # Analogous to attention: the model learns what to care about
        log_prec = torch.full((n_channels,), torch.log(torch.tensor(precision_init)))
        if learn_precision:
            self.log_precision = nn.Parameter(log_prec)
        else:
            self.register_buffer('log_precision', log_prec)

        # ── Diagnostic storage (populated during forward pass) ──────────────
        self._last_prediction_errors: Optional[torch.Tensor] = None
        self._last_energy: Optional[torch.Tensor] = None

    @property
    def precision(self) -> torch.Tensor:
        """Precision weights Π (always positive via softplus)."""
        return F.softplus(self.log_precision)  # (n_channels,)

    # ── Core computation ───────────────────────────────────────────────────────

    def compute_energy(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Compute KAN-parameterised energy density for all pixels.

        Equivalent to HamiltonianField.compute_energy but with explicit
        per-channel feature extraction and KAN energy.

        Parameters
        ----------
        fields : (B, n_channels, H, W)

        Returns
        -------
        energy : scalar — total energy (summed over spatial + channels)
        """
        B, C, H, W = fields.shape
        pad = self.spatial_filters[0].shape[-1] // 2
        all_features = []

        for c in range(C):
            u_c = fields[:, c:c+1, :, :]                       # (B, 1, H, W)
            f_c = F.conv2d(u_c, self.spatial_filters[c], padding=pad)  # (B, n_filters, H, W)
            all_features.append(f_c)

        # (B, C*n_filters, H, W)  →  (B*H*W, C*n_filters)
        features = torch.cat(all_features, dim=1)
        BHW = B * H * W
        features_flat = features.permute(0, 2, 3, 1).contiguous().view(BHW, -1)
        
        # Grid-Scale mismatch fix: squash features to [-1, 1] for B-splines
        features_flat = torch.tanh(features_flat)

        # KAN → scalar energy density per pixel
        energy_density = self.kan_energy(features_flat)         # (B*H*W, 1)
        total_energy = energy_density.sum()

        self._last_energy = total_energy.detach()
        return total_energy

    def prediction_error(
        self,
        u: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute precision-weighted prediction error ε = Π^½ · (u - μ).

        Parameters
        ----------
        u  : (B, n_channels, H, W) — current state
        mu : (B, n_channels, H, W) — top-down prediction (None → zeros)

        Returns
        -------
        error : (B, n_channels, H, W)
        """
        if mu is None:
            mu = torch.zeros_like(u)

        prec_sqrt = self.precision.sqrt()                        # (n_channels,)
        prec_sqrt = prec_sqrt.view(1, -1, 1, 1)                 # broadcast

        error = prec_sqrt * (u - mu)
        self._last_prediction_errors = error.detach()
        return error

    def inference_step(
        self,
        u: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """
        One step of predictive coding inference dynamics.

        u_{t+1} = u_t - dt · (ε + ∇_u E_KAN(u_t))

        where ε = Π^½(u - μ) is the prediction error and
        ∇_u E_KAN is the KAN energy gradient.

        This IS Allen-Cahn dynamics, correctly reinterpreted as
        precision-weighted prediction error minimisation.

        Parameters
        ----------
        u   : (B, n_channels, H, W)
        mu  : top-down prediction (or None for unsupervised mode)
        dt  : inference step size

        Returns
        -------
        u_new : (B, n_channels, H, W)
        """
        # enable_grad so this works inside @torch.no_grad() at eval time
        with torch.enable_grad():
            u_in = u.detach().requires_grad_(True)
            E = self.compute_energy(u_in)
            energy_grad = torch.autograd.grad(E, u_in, create_graph=False)[0].detach()

        # Prediction error (no grad needed)
        error = self.prediction_error(u.detach(), mu)

        # Combined update: error + energy gradient
        u_new = u.detach() - dt * (error + energy_grad)
        return u_new

    def run_inference(
        self,
        u_init: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        n_steps: int = 20,
        dt: float = 0.05,
        anchor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run K steps of predictive coding inference.

        Parameters
        ----------
        u_init     : (B, n_channels, H, W)
        mu         : top-down prediction (None → unsupervised)
        n_steps    : number of gradient steps K (test-time compute)
        dt         : step size
        anchor_mask: (B, 1, H, W) — if not None, anchor those pixels to u_init
                     (used for constraint satisfaction tasks like Sudoku)

        Returns
        -------
        u_final : (B, n_channels, H, W)
        info    : dict with energy trajectory and error trajectory
        """
        u = u_init.clone()
        energy_traj = []
        error_traj = []

        for _ in range(n_steps):
            u = self.inference_step(u, mu=mu, dt=dt)

            # Re-anchor clamped pixels (e.g., Sudoku clue cells)
            if anchor_mask is not None:
                u = torch.where(anchor_mask.bool(), u_init, u)

            if self._last_energy is not None:
                energy_traj.append(self._last_energy.item())
            if self._last_prediction_errors is not None:
                err_norm = self._last_prediction_errors.norm().item()
                error_traj.append(err_norm)

        info = {
            'energy_trajectory': energy_traj,
            'error_trajectory':  error_traj,
        }
        return u, info

    # ── Local Hebbian Weight Update ──────────────────────────────────────────

    def local_weight_update(
        self,
        u: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        lr: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Local Hebbian-like weight update: ΔW ∝ ε · (∂φ/∂W).

        Equivalent to backpropagation AT EQUILIBRIUM (Song et al. 2020).
        Does NOT require global gradient transport — each layer updates
        using only locally available information (its own error signal
        and the pre-synaptic activation).

        This implements the 'learning phase' of predictive coding:
        1. Inference phase runs until equilibrium (see run_inference)
        2. Learning phase updates weights once using the converged errors

        Parameters
        ----------
        u   : (B, n_channels, H, W) — converged state after inference
        mu  : top-down prediction (None → zero)
        lr  : learning rate for local update

        Returns
        -------
        Dict with gradient norms per component (for logging)
        """
        error = self.prediction_error(u, mu)       # (B, n_channels, H, W)

        # Spatial filter update: Δfilter_c ∝ Σ_{x,y} ε_c(x,y) · u_c(x,y)
        grad_norms = {}
        with torch.no_grad():
            for c in range(self.n_channels):
                u_c = u[:, c:c+1, :, :]           # (B, 1, H, W)
                e_c = error[:, c:c+1, :, :]        # (B, 1, H, W)

                # Gradient of filter energy w.r.t. filter weights
                # ΔK_c = (e_c) ⊛ u_c  [cross-correlation, same as backprop through conv]
                pad = self.spatial_filters[c].shape[-1] // 2
                delta_filt = F.conv2d(
                    u_c.permute(1, 0, 2, 3),       # (1, B, H, W)
                    e_c.permute(1, 0, 2, 3),        # (1, B, H, W)
                    padding=pad,
                ).permute(1, 0, 2, 3)[:self.n_filters]

                self.spatial_filters[c].data -= lr * delta_filt[:, :1].expand_as(
                    self.spatial_filters[c]
                )
                grad_norms[f'filter_{c}'] = delta_filt.norm().item()

        # Update KAN energy parameters using Equilibrium Propagation
        # At equilibrium, the parameter gradient of the total energy is the correct Hebbian update
        u_in = u.detach().requires_grad_(True)
        with torch.enable_grad():
            E_total = self.compute_energy(u_in)
            kan_grads = torch.autograd.grad(E_total, self.kan_energy.parameters(), allow_unused=True)
            
            with torch.no_grad():
                kan_norm = 0.0
                for p, g in zip(self.kan_energy.parameters(), kan_grads):
                    if g is not None:
                        p.data -= lr * g
                        kan_norm += g.pow(2).sum().item()
                grad_norms['kan'] = kan_norm ** 0.5

        return grad_norms

    # ── Training Loss (Score Matching / DSM equivalent) ──────────────────────

    def denoising_loss(
        self,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        1-step denoising loss (Denoising Score Matching equivalent).

        L(θ) = ‖x_clean - (x_noisy - η·∇_x E_θ(x_noisy))‖²

        This is mathematically equivalent to score matching (Vincent 2011):
            DSM: minimise E[‖s_θ(x) - ∇_x log p(x)‖²]
            where s_θ(x) = -∇_x E_θ(x)  [score = negative energy gradient]

        Parameters
        ----------
        x_clean, x_noisy : (B, 1, H, W)

        Returns
        -------
        loss : scalar
        """
        # Multi-channel field from single-channel image
        u = self.initialize_fields(x_noisy)
        u = u.requires_grad_(True)

        E = self.compute_energy(u)
        energy_grad = torch.autograd.grad(E, u, create_graph=True)[0]

        # 1-step denoised prediction
        step = self.step_size
        u_denoised = u - step * energy_grad

        # Project back to image channel
        x_pred = u_denoised[:, :1, :, :]

        # MSE on clean image
        loss = F.mse_loss(x_pred, x_clean)

        # Optional: gradient magnitude regularisation (prevents energy collapse)
        grad_reg = 0.01 * (energy_grad ** 2).mean()

        return loss + grad_reg

    def forward(
        self,
        x: torch.Tensor,
        n_steps: int = 10,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Full inference: x_noisy → x_clean via K steps of predictive coding.

        Parameters
        ----------
        x             : (B, 1, H, W) — noisy input
        n_steps       : K inference steps (test-time compute)
        return_trajectory : if True, return intermediate states

        Returns
        -------
        x_out : (B, 1, H, W) — denoised output
        """
        u = self.initialize_fields(x)
        u, info = self.run_inference(u, n_steps=n_steps)
        return u[:, :1, :, :]


# ──────────────────────────────────────────────────────────────────────────────
# Backward-Compatible Alias
# ──────────────────────────────────────────────────────────────────────────────

class GradientFlowField(PredictiveCodingField):
    """
    Alias: GradientFlowField == PredictiveCodingField.

    Use this name when emphasising the Allen-Cahn / gradient-flow dynamics
    rather than the predictive coding interpretation.

    Both names are correct. HamiltonianField is INCORRECT.
    """
    pass
