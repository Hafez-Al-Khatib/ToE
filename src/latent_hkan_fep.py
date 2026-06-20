"""
Latent H-KAN with Full FEP Compliance
========================================
Extends the existing latent_hkan.py to a complete variational inference
system satisfying the requirements of the Free Energy Principle.

The Missing Piece in the Original latent_hkan.py
-------------------------------------------------
The original LatentHKAN had:
    ✓ Encoder: x → z
    ✓ KAN energy: V(z)
    ✓ Hamiltonian dynamics: (z,p) → (z',p') via symplectic Euler
    ✗ NO recognition model q(z|o) — missing FEP component
    ✗ NO generative model p(o|z) — missing FEP component
    ✗ NO variational free energy — just gradient flow on E(x)

This file adds those missing pieces. The result:
    ✓ Recognition model q_φ(z|o)   — amortised variational inference
    ✓ Generative decoder p_θ(o|z)  — explicit generative model
    ✓ Variational Free Energy F    — proper FEP objective
    ✓ Hamiltonian dynamics         — symplectic exploration of posterior

Neuroscience Justification: The Hamiltonian Brain (Aitchison & Lengyel 2016)
-----------------------------------------------------------------------------
Aitchison & Lengyel proved that Hamiltonian Monte Carlo (HMC) maps exactly
onto excitatory-inhibitory neural circuits:
    - Position variables z  ←→  excitatory neuron firing rates
    - Momentum variables p  ←→  inhibitory neuron activity
    - E/I balance           ←→  symplectic structure (Liouville theorem)
    - Cortical oscillations ←→  Hamiltonian trajectories in neural state space

The KAN potential V(z) = -log p(z|o) defines attractor states corresponding
to recognised objects. Hamiltonian dynamics CONSERVE energy (unlike
Allen-Cahn gradient flow), meaning the trajectories explore the posterior
without losing information — equivalent to exact MCMC vs variational approx.

Connections to Working Memory (Ye & Wang 2023)
----------------------------------------------
PFC dynamics during working memory are governed by:
    - Multiple stable attractors (resting + stimulus-selective states)
    - Barrier heights (memory stability = energy barrier between attractors)
    - Minimum-action transition paths (thinking = Hamiltonian trajectories)
All three are computable from V(z): attractors = minima, barriers = saddle
points, transitions = minimum-energy paths.

Test-Time Compute Scaling
-------------------------
More Hamiltonian steps K → more thorough exploration of the posterior.
Energy conservation guarantees NO information loss with more steps.
Quality monotonically improves with K (unlike vanilla gradient descent
which can overshoot). This is the correct test-time scaling mechanism.

References
----------
Aitchison & Lengyel (2016). The Hamiltonian Brain. PLOS Computational Biology.
Friston (2005). A theory of cortical responses.
Ye & Wang (2023). Hamiltonian mechanics of neural manifolds.
Greydanus et al. (NeurIPS 2019). Hamiltonian Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kan import KAN
from recognition_model import RecognitionKAN, GenerativeDecoder, variational_free_energy
from latent_hkan import ConvEncoder, MLPEncoder


class LatentHKAN_FEP(nn.Module):
    """
    Latent Hamiltonian KAN with Full Free Energy Principle compliance.

    Components
    ----------
    1. Recognition model  q_φ(z|o):  KAN encoder → (μ_z, σ_z)
    2. KAN potential      V_θ(z):    Learnable B-spline energy in latent space
    3. Hamiltonian system H(z,p):    T(p) + V(z), symplectic integration
    4. Generative decoder p_ψ(o|z):  KAN decoder → reconstructed observation

    Objective: Variational Free Energy
    -----------------------------------
    F = KL[q(z|o) || p(z)] + E_q[-log p(o|z)]
      = Complexity         + Accuracy

    Hamiltonian steps serve as the 'thinking' phase: the system explores
    the posterior via energy-conserving trajectories, generating diverse
    hypotheses before committing to a reconstruction.

    Parameters
    ----------
    obs_dim     : int   — observation dimensionality (e.g., 784 for MNIST)
    latent_dim  : int   — latent space dimensionality (recommend 8–32)
    kan_hidden  : list  — KAN hidden layer dims for energy function
    enc_hidden  : list  — hidden dims for recognition KAN encoder
    dec_hidden  : list  — hidden dims for generative KAN decoder
    encoder_type: str   — 'mlp' or 'conv'
    ham_steps   : int   — default number of Hamiltonian integration steps
    dt          : float — symplectic Euler step size
    damping     : float — optional damping (0 = pure Hamiltonian, >0 = friction)
    grid_size   : int   — B-spline grid resolution
    """

    def __init__(
        self,
        obs_dim: int = 784,
        latent_dim: int = 16,
        kan_hidden: Optional[list] = None,
        enc_hidden: Optional[list] = None,
        dec_hidden: Optional[list] = None,
        encoder_type: str = 'mlp',
        ham_steps: int = 10,
        dt: float = 0.1,
        damping: float = 0.0,
        grid_size: int = 5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.ham_steps = ham_steps
        self.dt = dt
        self.damping = damping

        if kan_hidden is None:
            kan_hidden = [32, 16]
        if enc_hidden is None:
            enc_hidden = [min(256, obs_dim // 2), 64]
        if dec_hidden is None:
            dec_hidden = [64, min(256, obs_dim // 2)]

        # ── 1. Recognition Model q_φ(z|o) ──────────────────────────────────
        self.recognition = RecognitionKAN(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=enc_hidden,
            grid_size=grid_size,
        )

        # ── 2. KAN Potential V_θ(z) ─────────────────────────────────────────
        # Input: latent z ∈ ℝ^latent_dim
        # Output: scalar potential energy V(z) ∈ ℝ
        # V(z) = -log p(z) - log p(o|z)  [when combined with prior and likelihood]
        self.kan_potential = KAN(
            layers_hidden=[latent_dim] + kan_hidden + [1],
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
        )

        # ── 3. Generative Decoder p_ψ(o|z) ─────────────────────────────────
        self.decoder = GenerativeDecoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dims=dec_hidden,
            grid_size=grid_size,
            output_activation='sigmoid',
        )

        # ── Diagnostics ─────────────────────────────────────────────────────
        self._last_H_trajectory: Optional[torch.Tensor] = None

    # ── Hamiltonian Mechanics ──────────────────────────────────────────────

    def potential(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate KAN potential V(z).

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        V : (B,) — potential energy per sample
        """
        return self.kan_potential(z).squeeze(-1)

    def hamiltonian(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        H(z,p) = T(p) + V(z) = ½‖p‖² + V_KAN(z).

        Kinetic energy: T = ½‖p‖²    (isotropic mass matrix = I)
        Potential energy: V = KAN(z)

        Parameters
        ----------
        z, p : (B, latent_dim)

        Returns
        -------
        H : (B,) — Hamiltonian per sample
        """
        T = 0.5 * (p ** 2).sum(-1)
        V = self.potential(z)
        return T + V

    def symplectic_step(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of symplectic Euler integration.

        The symplectic Euler scheme (also called semi-implicit Euler or
        leapfrog) PRESERVES the symplectic structure of Hamiltonian dynamics,
        meaning it conserves a shadow Hamiltonian H̃ ≈ H to O(dt²).

        Scheme:
            p_{t+1} = p_t - dt · ∇_z V(z_t)
            z_{t+1} = z_t + dt · p_{t+1}       [uses updated p!]

        Note: this is DIFFERENT from the naive Euler which uses p_t.
        The symplectic version is stable and energy-conserving.
        The naive version spirals outward (energy grows unboundedly).

        Parameters
        ----------
        z, p : (B, latent_dim) — position and momentum
        dt   : step size (overrides self.dt if provided)

        Returns
        -------
        z_new, p_new : updated position and momentum
        """
        if dt is None:
            dt = self.dt

        # Gradient of potential w.r.t. z
        if getattr(self, 'training', False) or z.requires_grad:
            z_in = z
            V = self.potential(z_in).sum()
            dV_dz = torch.autograd.grad(V, z_in, create_graph=torch.is_grad_enabled())[0]
        else:
            z_in = z.detach().requires_grad_(True)
            V = self.potential(z_in).sum()
            dV_dz = torch.autograd.grad(V, z_in)[0]

        # Symplectic Euler:
        p_new = p - dt * dV_dz                     # momentum update (uses old z)
        if self.damping > 0:
            p_new = p_new * (1 - self.damping * dt) # optional friction
        z_new = z + dt * p_new                     # position update (uses NEW p)

        if getattr(self, 'training', False) or z.requires_grad:
            return z_new, p_new
        return z_new.detach(), p_new.detach()

    def hamiltonian_trajectory(
        self,
        z0: torch.Tensor,
        p0: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Run K steps of Hamiltonian dynamics from (z0, p0).

        More steps = more thorough exploration of the posterior.
        Energy is approximately conserved throughout (shadow Hamiltonian).

        Parameters
        ----------
        z0      : (B, latent_dim) — initial position
        p0      : (B, latent_dim) — initial momentum (None → N(0,I))
        n_steps : K integration steps (default: self.ham_steps)
        dt      : step size (default: self.dt)
        return_all : if True, return full trajectory

        Returns
        -------
        z_final, p_final : final state
        info    : dict with H_initial, H_final, H_trajectory (if return_all)
        """
        if n_steps is None:
            n_steps = self.ham_steps
        if p0 is None:
            p0 = torch.randn_like(z0)

        z, p = z0, p0
        trajectory = []
        H_traj = []

        with torch.no_grad():
            H_init = self.hamiltonian(z, p)

        for _ in range(n_steps):
            z, p = self.symplectic_step(z, p, dt=dt)
            if return_all:
                trajectory.append(z)
                with torch.no_grad():
                    H_traj.append(self.hamiltonian(z, p).mean().item())

        with torch.no_grad():
            H_final = self.hamiltonian(z, p)

        info = {
            'H_initial':    H_init.mean().item(),
            'H_final':      H_final.mean().item(),
            'H_drift':      (H_final - H_init).abs().mean().item(),
            'trajectory':   trajectory if return_all else None,
            'H_trajectory': H_traj if return_all else None,
        }
        self._last_H_trajectory = H_traj

        return z, p, info

    # ── Full Inference Cycle ───────────────────────────────────────────────

    def forward(
        self,
        o: torch.Tensor,
        n_ham_steps: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference cycle:
            1. Amortised recognition:  o → (μ_z, σ_z)
            2. Posterior sample:       z ~ q(z|o)
            3. Hamiltonian exploration: (z,p) → (z',p') via K steps
            4. Generative reconstruction: z' → o_hat

        Parameters
        ----------
        o              : (B, obs_dim) or (B, 1, H, W)
        n_ham_steps    : K Hamiltonian steps (test-time compute)
        return_diagnostics : include Hamiltonian energy tracking

        Returns
        -------
        dict with keys:
            'o_hat'  : (B, obs_dim) — reconstructed observation
            'z'      : (B, latent_dim) — final latent (post-Hamiltonian)
            'z0'     : (B, latent_dim) — initial latent (from recognition)
            'mu'     : (B, latent_dim) — posterior mean
            'logvar' : (B, latent_dim) — posterior log-variance
            'H_info' : dict — Hamiltonian diagnostics (if requested)
        """
        if o.dim() > 2:
            o = o.view(o.size(0), -1)

        # ── Step 1: Amortised recognition ──────────────────────────────────
        z0, mu, logvar = self.recognition.sample(o)

        # ── Step 2: Hamiltonian exploration ────────────────────────────────
        # Initialise momentum from kinetic energy prior p ~ N(0, I)
        p0 = torch.randn_like(z0)
        z_final, p_final, H_info = self.hamiltonian_trajectory(
            z0=z0,
            p0=p0,
            n_steps=n_ham_steps,
            return_all=return_diagnostics,
        )

        # ── Step 3: Decode ──────────────────────────────────────────────────
        o_hat = self.decoder(z_final)

        result = {
            'o_hat':  o_hat,
            'z':      z_final,
            'z0':     z0,
            'mu':     mu,
            'logvar': logvar,
        }
        if return_diagnostics:
            result['H_info'] = H_info

        return result

    # ── Training ───────────────────────────────────────────────────────────

    def compute_loss(
        self,
        o: torch.Tensor,
        beta: float = 1.0,
        n_ham_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute variational free energy F = KL + Reconstruction.

        F = KL[q(z|o) || p(z)]  +  E_q[‖o - g(z)‖²]
          = Complexity            +  Accuracy

        Parameters
        ----------
        o           : (B, obs_dim)
        beta        : β-VAE weighting of KL (1.0 = standard FEP)
        n_ham_steps : Hamiltonian steps during training (1 is efficient)

        Returns
        -------
        dict with 'F', 'kl', 'recon', 'H_drift'
        """
        result = self.forward(o, n_ham_steps=n_ham_steps, return_diagnostics=False)

        # Reconstruction loss  -E_q[log p(o|z)]
        o_flat = o.view(o.size(0), -1) if o.dim() > 2 else o
        recon = F.mse_loss(result['o_hat'], o_flat)

        # KL divergence  KL[q(z|o) || p(z)]
        kl = self.recognition.kl_divergence(result['mu'], result['logvar'])

        # Total variational free energy
        free_energy = recon + beta * kl

        # Hamiltonian consistency loss (optional): penalise H-drift
        # This is a physics regulariser — encourages true energy conservation
        # Not needed for training; useful for analysis / ablation
        H_init = self.hamiltonian(result['z0'], torch.zeros_like(result['z0']))
        H_final = self.hamiltonian(result['z'], torch.zeros_like(result['z']))
        h_drift = (H_final - H_init).abs().mean()

        return {
            'F':       free_energy,
            'kl':      kl,
            'recon':   recon,
            'H_drift': h_drift,
        }

    # ── Active Inference (Action Selection) ───────────────────────────────

    def expected_free_energy(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        transition_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute Expected Free Energy G(π) for action selection.

        G(π) = E_q[H[p(o|z_τ)]] + E_q[KL[q(z_τ|π)||p(z_τ)]]
             = Ambiguity         + Risk

        Ambiguity: uncertainty about future observations (explore)
        Risk:      divergence from preferred states (exploit)

        This implements the core active inference action selection:
        a* = argmin_a G(a)

        Currently simplified: G ≈ V_KAN(z_next) [potential at next state]
        Full implementation requires a learned transition model p(z'|z,a).

        Parameters
        ----------
        z        : (B, latent_dim) — current latent state
        actions  : (n_actions, action_dim) — candidate actions
        transition: optional learned p(z'|z,a)

        Returns
        -------
        G : (B, n_actions) — expected free energy per action
        """
        n_actions = actions.shape[0]
        B = z.shape[0]

        if transition_model is None:
            # Simplified: G ≈ V_KAN(z + action_effect)
            # Action effect modelled as additive perturbation in latent space
            z_expanded = z.unsqueeze(1).expand(B, n_actions, -1)   # (B, A, D)
            a_expanded = actions.unsqueeze(0).expand(B, -1, -1)    # (B, A, D)
            z_next = z_expanded + 0.1 * a_expanded[:, :, :self.latent_dim]
            z_flat = z_next.view(B * n_actions, self.latent_dim)
            G = self.potential(z_flat).view(B, n_actions)
        else:
            # Full active inference with transition model
            raise NotImplementedError(
                "Full active inference with explicit transition model. "
                "See docs/TIM_Architecture.md for implementation notes."
            )

        return G

    def select_action(
        self,
        o: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action by minimising Expected Free Energy.

        a* = argmin_a G(z, a)

        Returns
        -------
        best_action : (B, action_dim)
        G           : (B, n_actions) — G values for all actions
        """
        result = self.forward(o, n_ham_steps=self.ham_steps)
        z = result['z']
        G = self.expected_free_energy(z, actions)
        best_idx = G.argmin(dim=-1)
        best_action = actions[best_idx]
        return best_action, G
