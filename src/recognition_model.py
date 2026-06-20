"""
Recognition Model — q_φ(z | o)
================================
The highest-leverage addition to make the codebase FEP-compliant.

Mathematical Role
-----------------
Under the Free Energy Principle (Friston, 2005), inference requires two objects:

    Generative model:    p(o, z) = p(o | z) · p(z)
    Recognition model:   q_φ(z | o) ≈ p(z | o)     ← THIS FILE

Variational Free Energy:
    F = E_q[log q(z|o)] - E_q[log p(o, z)]
      = KL[q(z|o) || p(z)] - E_q[log p(o|z)]
      = Complexity       +  Accuracy

Without q(z|o), the system can ONLY do gradient descent on E(x) (Allen-Cahn/
score matching). Adding q(z|o) transforms it into variational inference, enabling:
    1. Amortised inference (fast, single-pass recognition at test time)
    2. Generative sampling   (dream / imagination via p(o|z))
    3. Active inference      (action selection via expected free energy G)
    4. True FEP compliance   (not just score matching — full VI)

Chain of Mathematical Equivalences (Jordan-Kinderlehrer-Otto, 1998)
--------------------------------------------------------------------
Allen-Cahn ⊂ Langevin ⊂ Wasserstein gradient flow of F(ρ) ⊃ FEP

    Allen-Cahn:   ∂u/∂t = -δE/δu          (deterministic, zero-temperature)
    Langevin:     dx = -∇E dt + √(2T) dW  (stochastic, temperature T)
    FP equation:  ∂ρ/∂t = ∇·(ρ∇V) + Δρ  (population, Wasserstein gradient flow)

The FREE ENERGY F(ρ) = ∫ρ log ρ dx + ∫ρV dx = -H(q) + E_q[E(x)]
is EXACTLY the variational free energy under Gaussian approximation (Bogacz 2017).

Connections Implemented Here
-----------------------------
Link 1: Allen-Cahn ≡ score following (Vincent 2011 via DSM equivalence)
Link 2: Score matching → predictive coding (Rao & Ballard 1999)
Link 3: Predictive coding → variational FE (Bogacz 2017, Laplace approx)
Link 4: All ⊂ Wasserstein gradient flow (JKO 1998)

References
----------
Friston, K. (2005). A theory of cortical responses. Philosophical Transactions.
Jordan, Kinderlehrer & Otto (1998). The variational formulation of the FP equation.
Bogacz, R. (2017). A tutorial on the free-energy framework. J. Math. Psych.
Vincent, P. (2011). A connection between score matching and denoising autoencoders.
Rao & Ballard (1999). Predictive coding in the visual cortex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kan import KAN


# ──────────────────────────────────────────────────────────────────────────────
# Recognition Model: q_φ(z | o)
# ──────────────────────────────────────────────────────────────────────────────

class RecognitionKAN(nn.Module):
    """
    Amortised recognition model q_φ(z|o) parameterised by a KAN encoder.

    Maps observations o → approximate posterior parameters (μ_z, σ_z),
    enabling:
        - Fast amortised inference (single forward pass at test time)
        - Reparameterisation trick for differentiable VI
        - KL-divergence computation against Gaussian prior p(z) = N(0,I)

    Architecture
    ------------
    o  →  [KAN encoder]  →  (μ_z, log σ_z)
                             ↓
                        z = μ + σ·ε,   ε ~ N(0,I)   [reparameterisation]

    KAN vs MLP choice
    -----------------
    KAN is used here because the recognition function z = f(o) is typically
    smooth and interpretable (e.g., brightness → position in latent space),
    matching KAN's design assumptions. The B-spline activations let us
    inspect what features drive the posterior.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of observation (e.g., 784 for flat MNIST).
    latent_dim : int
        Dimensionality of latent space z.
    hidden_dims : list[int]
        Hidden layer sizes for the KAN encoder.
    use_conv : bool
        If True, prepend a small CNN to the KAN (for image inputs).
    conv_channels : tuple[int]
        Channel sizes for the optional CNN front-end.
    grid_size : int
        B-spline grid resolution for KAN layers.
    min_logvar : float
        Minimum log-variance (prevents posterior collapse).
    max_logvar : float
        Maximum log-variance (prevents unconstrained uncertainty).
    """

    def __init__(
        self,
        obs_dim: int = 784,
        latent_dim: int = 16,
        hidden_dims: Optional[list] = None,
        use_conv: bool = False,
        conv_channels: Tuple[int, ...] = (32, 64),
        grid_size: int = 5,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        if hidden_dims is None:
            hidden_dims = [min(256, obs_dim // 2), 64]

        # ── Optional CNN front-end for image observations ──────────────────
        self.use_conv = use_conv
        if use_conv:
            # Infer spatial size (assumes square image)
            side = int(obs_dim ** 0.5)
            assert side * side == obs_dim, \
                f"use_conv=True requires square image; got obs_dim={obs_dim}"
            self.conv_side = side

            conv_layers = []
            in_ch = 1
            for out_ch in conv_channels:
                conv_layers += [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                ]
                in_ch = out_ch
            self.cnn = nn.Sequential(*conv_layers)

            # Compute CNN output dimension
            dummy = torch.zeros(1, 1, side, side)
            with torch.no_grad():
                cnn_out = self.cnn(dummy)
            kan_in_dim = cnn_out.numel()
        else:
            self.cnn = None
            kan_in_dim = obs_dim

        # ── KAN encoder  →  mean and log-variance ──────────────────────────
        # Output: 2 * latent_dim  (first half = μ, second half = log σ²)
        layers_hidden = [kan_in_dim] + hidden_dims + [2 * latent_dim]
        self.kan_encoder = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
        )

        # Learnable precision (inverse variance) weighting per latent dim
        # Initialised to 1; acts as an attention gate over latent dimensions
        self.log_precision = nn.Parameter(torch.zeros(latent_dim))

    # ── Core methods ──────────────────────────────────────────────────────────

    def encode(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map observation → posterior parameters.

        Parameters
        ----------
        o : torch.Tensor  shape (B, obs_dim) or (B, 1, H, W)

        Returns
        -------
        mu     : (B, latent_dim)  — posterior mean
        logvar : (B, latent_dim)  — posterior log-variance
        """
        if o.dim() > 2:
            o = o.view(o.size(0), -1)

        if self.use_conv:
            B = o.size(0)
            o_img = o.view(B, 1, self.conv_side, self.conv_side)
            h = self.cnn(o_img).view(B, -1)
        else:
            h = o

        out = self.kan_encoder(h)             # (B, 2*latent_dim)
        mu, logvar = out.chunk(2, dim=-1)

        # Clamp log-variance for training stability
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar

    def sample(
        self,
        o: torch.Tensor,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample z ~ q(z|o) via reparameterisation trick.

        Parameters
        ----------
        o         : (B, obs_dim)
        n_samples : int  — number of posterior samples per observation

        Returns
        -------
        z      : (B*n_samples, latent_dim)  — posterior sample
        mu     : (B, latent_dim)            — posterior mean
        logvar : (B, latent_dim)            — posterior log-variance
        """
        mu, logvar = self.encode(o)
        sigma = torch.exp(0.5 * logvar)

        if n_samples == 1:
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
        else:
            # Multiple samples: (B, n_samples, latent_dim)
            B, D = mu.shape
            eps = torch.randn(B, n_samples, D, device=mu.device)
            z = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).view(B * n_samples, D)
            mu = mu.unsqueeze(1).expand(B, n_samples, D).reshape(B * n_samples, D)
            logvar = logvar.unsqueeze(1).expand(B, n_samples, D).reshape(B * n_samples, D)

        return z, mu, logvar

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL[ q(z|o) || p(z) ] for Gaussian q and standard-normal prior.

        KL = -0.5 * Σ_d (1 + log σ²_d - μ²_d - σ²_d)

        Precision weighting: multiply by exp(log_precision) to let the model
        selectively penalise uninformative latent dimensions.

        Returns
        -------
        kl : scalar  — mean KL across batch
        """
        precision = torch.exp(self.log_precision)          # (latent_dim,)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_weighted = (kl_per_dim * precision).sum(dim=-1)
        return kl_weighted.mean()

    def log_posterior(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        log q(z | o) = -0.5 * Σ_d [(z_d - μ_d)² / σ²_d + log σ²_d + log 2π]

        Used for importance weighting (IWAE bound).
        """
        log2pi = torch.log(torch.tensor(2 * 3.14159265358979, device=z.device))
        return -0.5 * ((z - mu).pow(2) / logvar.exp() + logvar + log2pi).sum(-1)

    def forward(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: sample + return diagnostics.

        Returns  z, mu, logvar
        """
        return self.sample(o)


# ──────────────────────────────────────────────────────────────────────────────
# Generative Decoder: p_θ(o | z)
# ──────────────────────────────────────────────────────────────────────────────

class GenerativeDecoder(nn.Module):
    """
    Generative model p_θ(o | z), parameterised by a KAN decoder.

    Maps latent z → reconstructed observation o.
    KAN choice: the generation function should be smooth and
    interpretable — when z changes continuously, o should change
    continuously and predictably.

    Parameters
    ----------
    latent_dim : int
    obs_dim    : int
    hidden_dims : list[int]
    use_deconv : bool   — if True, use transposed-conv up-sampling
    output_activation : str  — 'sigmoid' (images), 'none' (continuous data)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        obs_dim: int = 784,
        hidden_dims: Optional[list] = None,
        use_deconv: bool = False,
        output_activation: str = 'sigmoid',
        grid_size: int = 5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.output_activation = output_activation

        if hidden_dims is None:
            hidden_dims = [64, min(256, obs_dim // 2)]

        # KAN decoder
        layers_hidden = [latent_dim] + hidden_dims + [obs_dim]
        self.kan_decoder = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
        )

        # Learnable observation noise (log σ_obs)
        self.log_obs_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z → observation.

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        o_hat : (B, obs_dim)
        """
        o_logit = self.kan_decoder(z)
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(o_logit)
        elif self.output_activation == 'tanh':
            return torch.tanh(o_logit)
        return o_logit

    def log_likelihood(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        log p(o | z) = -||o - g(z)||² / (2σ²_obs) - D/2 · log(2πσ²)

        Returns negative mean reconstruction error (per sample).
        """
        o_hat = self.forward(z)
        if o.dim() > 2:
            o = o.view(o.size(0), -1)
        sigma = self.log_obs_sigma.exp().clamp(min=1e-4)
        recon = ((o - o_hat) ** 2).sum(-1) / (2 * sigma ** 2)
        const = 0.5 * self.obs_dim * torch.log(2 * torch.tensor(3.14159265) * sigma ** 2)
        return -(recon + const).mean()

    def reconstruction_loss(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss (simpler alternative to log-likelihood)."""
        o_hat = self.forward(z)
        if o.dim() > 2:
            o = o.view(o.size(0), -1)
        return F.mse_loss(o_hat, o)


# ──────────────────────────────────────────────────────────────────────────────
# Variational Free Energy objective
# ──────────────────────────────────────────────────────────────────────────────

def variational_free_energy(
    o: torch.Tensor,
    recognition: RecognitionKAN,
    generative: GenerativeDecoder,
    n_samples: int = 1,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute variational free energy F = Complexity + β·Accuracy.

    F = KL[q(z|o) || p(z)] - E_q[log p(o|z)]
      = KL divergence      + reconstruction error

    Parameters
    ----------
    o          : (B, obs_dim) — observations
    recognition: RecognitionKAN
    generative : GenerativeDecoder
    n_samples  : Monte Carlo samples for E_q[log p(o|z)]
    beta       : β-VAE weighting of KL term (β=1 → standard VAE/FEP)

    Returns
    -------
    Dict with keys: 'F' (total), 'kl', 'recon', 'z', 'o_hat'
    """
    z, mu, logvar = recognition.sample(o, n_samples=n_samples)

    # Reconstruction  -E_q[log p(o|z)]
    o_rep = o.repeat_interleave(n_samples, dim=0) if n_samples > 1 else o
    recon = generative.reconstruction_loss(o_rep, z)

    # Complexity  KL[q||p]
    kl = recognition.kl_divergence(mu, logvar)

    # Free energy
    F = recon + beta * kl

    with torch.no_grad():
        o_hat = generative(z[:o.size(0)])   # first sample for visualisation

    return {
        'F':     F,
        'kl':    kl,
        'recon': recon,
        'z':     z,
        'o_hat': o_hat,
    }
