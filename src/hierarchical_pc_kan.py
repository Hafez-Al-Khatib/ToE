"""
Hierarchical Predictive Coding KAN
=====================================
Multi-layer predictive coding hierarchy where each layer uses a KAN
as its generative function (top-down prediction generator).

Architecture
------------
Layers L, L-1, ..., 1, 0 (top to bottom):

    x_L  →  KAN_{L-1}  →  μ_{L-1}  ─┐
    x_{L-1}  ─────────────────────────┤→  ε_{L-1} = Π_{L-1}^½(x_{L-1} - μ_{L-1})
    x_{L-1}  →  KAN_{L-2}  →  μ_{L-2} ─┐
    ...                                  ...
    x_1  →  KAN_0  →  μ_0   ─────────┤→  ε_0  [bottom: image prediction errors]
    x_0 = observation (clamped)

Inference dynamics (fast, γ-timescale):
    dx_l/dt = -ε_l + (∂KAN_{l-1}/∂x_l)^T · ε_{l-1}
            = error FROM above + error BACK-projected FROM below

At equilibrium (dx_l/dt = 0), this is equivalent to backpropagation
(Song et al. NeurIPS 2020; Millidge et al. ICLR 2023).

Weight update (slow, δ/θ-timescale):
    ΔW_l = η · ε_l · (∂KAN_l/∂W_l)   [fully local, no backward pass needed]

Multi-timescale structure (Behrouz et al. Nested Learning, NeurIPS 2025)
------------------------------------------------------------------------
Fast  (γ, every inference step):   state updates dx_l/dt
Medium (β, every few steps):       precision Π_l updates
Slow  (δ/θ, every batch):          weight W_l updates

References
----------
Rao & Ballard (1999). Predictive coding in the visual cortex.
Song et al. (NeurIPS 2020). Can the brain do backpropagation?
Millidge et al. (ICLR 2023). Predictive coding: towards a future of AI.
Behrouz et al. (NeurIPS 2025). Nested Learning: The Illusion of Deep Learning Architecture.
Whittington & Bogacz (2017). An approximation of the error backpropagation algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kan import KAN


class PCLayer(nn.Module):
    """
    A single predictive coding layer.

    Responsibilities:
        1. Store state x_l
        2. Generate top-down prediction μ_{l-1} = KAN_l(x_l)
        3. Compute precision-weighted prediction error ε_l = Π_l^½(x_l - μ_l)
        4. Update state based on errors from above and below

    Parameters
    ----------
    input_dim   : int   — dim of this layer's state x_l
    output_dim  : int   — dim of the layer below (μ_{l-1} has this dim)
    grid_size   : int   — B-spline grid resolution
    precision   : float — initial precision value
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        grid_size: int = 5,
        precision_init: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [max(input_dim, output_dim)]

        # KAN generative function: x_l → μ_{l-1}
        self.kan = KAN(
            layers_hidden=[input_dim] + hidden_dims + [output_dim],
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
        )

        # Learnable log-precision — sized output_dim because errors live
        # at the LOWER layer (which has dimension output_dim).
        self.log_precision = nn.Parameter(
            torch.full((output_dim,), torch.log(torch.tensor(precision_init)))
        )

    @property
    def precision(self) -> torch.Tensor:
        """Precision weights (always positive)."""
        return F.softplus(self.log_precision)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate top-down prediction for layer below.

        Parameters
        ----------
        x : (B, input_dim) — this layer's state

        Returns
        -------
        mu : (B, output_dim) — prediction for layer below
        """
        return self.kan(x)

    def compute_error(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Precision-weighted prediction error ε = Π^½(x - μ).

        x and mu both live at the LOWER layer (dimension = output_dim).
        precision has shape (output_dim,) after the bug-fix in __init__.

        Parameters
        ----------
        x  : (B, output_dim) — lower layer state
        mu : (B, output_dim) — top-down prediction for lower layer

        Returns
        -------
        error : (B, output_dim)
        """
        prec_sqrt = self.precision.sqrt().unsqueeze(0)   # (1, output_dim)
        return prec_sqrt * (x - mu)

    def local_weight_update(
        self,
        x: torch.Tensor,
        error: torch.Tensor,
        lr: float = 1e-3,
    ) -> float:
        """
        ΔW_l = η · ε_l · (∂KAN_l/∂W_l)   [local Hebbian update].

        Equivalent to the gradient of the free energy F = ½Σ‖ε_l‖²
        w.r.t. KAN weights, evaluated at the current state.

        This does NOT require knowledge of errors in other layers.
        It is strictly local to this layer.
        """
        # Compute KAN gradient with respect to weights
        x_in = x.detach().requires_grad_(True)
        pred = self.kan(x_in)                            # (B, output_dim)

        # The prediction affects ε_{l-1} = Π_{l-1}^½(x_{l-1} - KAN_l(x_l))
        # ΔW_l ∝ -∂ε_{l-1}/∂W_l · ε_{l-1}
        # = -∂(x_{l-1} - KAN_l(x_l))/∂W_l · ε_{l-1}
        # = ∂KAN_l(x_l)/∂W_l · ε_{l-1}   [positive: reduce error]
        loss = (pred * error[:, :pred.shape[-1]].detach()).sum()

        # Only update KAN parameters, not x
        for param in self.kan.parameters():
            if param.grad is not None:
                param.grad.zero_()

        loss.backward()

        with torch.no_grad():
            for param in self.kan.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
                    param.grad.zero_()

        return loss.item()


class HierarchicalPCKAN(nn.Module):
    """
    L-layer predictive coding hierarchy with KAN generative functions.

    This is the full multi-layer version of PredictiveCodingField.
    While PredictiveCodingField operates on spatial pixel fields (2D),
    HierarchicalPCKAN operates on vectorised feature representations.

    Usage
    -----
    model = HierarchicalPCKAN(dims=[784, 256, 64, 16])

    # Forward (amortised inference):
    x_top, states, errors = model.forward(x_obs, n_steps=50)

    # Training (local Hebbian updates):
    model.local_update(states, errors, lr=1e-3)

    Parameters
    ----------
    dims : list[int]
        Layer dimensions from bottom (observation) to top.
        dims[0] = observation dim (clamped during inference).
        dims[-1] = highest-level representation.
    hidden_dims : list[int] or None
        Hidden dims for KAN in each layer (same for all layers).
    grid_size : int
        B-spline grid size for all KANs.
    inference_lr : float
        Step size for state update during inference.
    n_inference_steps : int
        Default number of inference steps (test-time compute).
    """

    def __init__(
        self,
        dims: List[int],
        hidden_dims: Optional[List[int]] = None,
        grid_size: int = 5,
        inference_lr: float = 0.1,
        n_inference_steps: int = 50,
    ):
        super().__init__()
        assert len(dims) >= 2, "Need at least 2 layers"

        self.dims = dims
        self.n_layers = len(dims) - 1    # number of KAN generative functions
        self.inference_lr = inference_lr
        self.n_inference_steps = n_inference_steps

        # Each PCLayer maps dim[l+1] → dim[l]  (top-down prediction)
        self.layers = nn.ModuleList([
            PCLayer(
                input_dim=dims[l + 1],
                output_dim=dims[l],
                hidden_dims=hidden_dims,
                grid_size=grid_size,
            )
            for l in range(self.n_layers)
        ])

        # Bottom-most KAN: maps dims[-1] → dims[-2] (only for inference)
        # Top layer has no "layer above" → prior: p(x_L) = N(0,I)

    def _init_states(self, x_obs: torch.Tensor) -> List[torch.Tensor]:
        """
        Initialise layer states.

        x_obs: (B, dims[0]) — observation (clamped at bottom)
        States x[0] = x_obs (fixed), x[1..L] = small noise
        """
        B = x_obs.size(0)
        states = [x_obs.detach()]           # layer 0: clamped to observation
        for l in range(1, len(self.dims)):
            x_l = 0.01 * torch.randn(B, self.dims[l], device=x_obs.device)
            states.append(x_l)
        return states

    def _compute_errors(
        self,
        states: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute prediction errors at each layer.

        ε_l = Π_l^½ (x_l - KAN_l(x_{l+1}))
        for l = 0 (observation level) to L-1 (penultimate level).
        Top layer L has prior error: ε_L = Π_L^½ · x_L  (vs N(0,I) prior)
        """
        errors = []
        for l in range(self.n_layers):
            # Top-down prediction from above
            mu_l = self.layers[l].predict(states[l + 1])      # (B, dims[l])
            # Error at this level
            e_l = self.layers[l].compute_error(states[l], mu_l)
            errors.append(e_l)

        # Top-level prior error: x_L should be close to 0 (N(0,I) prior)
        x_top = states[-1]
        prior_error = x_top   # derivative of -log N(x;0,I) = x
        errors.append(prior_error)

        return errors

    def _state_update(
        self,
        states: List[torch.Tensor],
        errors: List[torch.Tensor],
        lr: float,
    ) -> List[torch.Tensor]:
        """
        Update internal states (not observation layer) via error propagation.

        ∂x_l/∂t = -ε_l + (∂KAN_{l-1}/∂x_l)^T · ε_{l-1}

        The second term is the 'back-projected' prediction error from below.
        At convergence, this is equivalent to the backpropagation gradient
        (Whittington & Bogacz 2017; Song et al. 2020).
        """
        new_states = [states[0]]    # layer 0 clamped

        for l in range(1, len(states)):
            x_l = states[l].detach().requires_grad_(True)

            # Prediction FROM this layer to layer below
            if l < self.n_layers:
                pred_below = self.layers[l - 1].predict(x_l)   # (B, dims[l-1])
                # Gradient of lower-level error w.r.t. x_l
                # ε_{l-1} = Π^½(x_{l-1} - KAN_{l-1}(x_l))
                # ∂ε_{l-1}/∂x_l = -Π^½ · ∂KAN_{l-1}/∂x_l
                # Update contribution: +(∂KAN_{l-1}/∂x_l)^T · ε_{l-1}
                error_below = errors[l - 1].detach()   # (B, dims[l-1])
                proj_loss = (pred_below * error_below).sum()
                grad_from_below = torch.autograd.grad(
                    proj_loss, x_l, retain_graph=False
                )[0]
            else:
                grad_from_below = torch.zeros_like(x_l)

            # Error AT this level (from above)
            error_here = errors[l].detach()             # (B, dims[l])

            # State update: -error_here + grad_from_below
            delta = -error_here[:, :x_l.shape[-1]] + grad_from_below
            x_l_new = x_l.detach() + lr * delta
            new_states.append(x_l_new)

        return new_states

    def inference(
        self,
        x_obs: torch.Tensor,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        """
        Run predictive coding inference (minimise free energy by updating states).

        Parameters
        ----------
        x_obs           : (B, dims[0]) — observation (clamped at bottom)
        n_steps         : inference steps K (test-time compute)
        return_trajectory: track free energy F over steps

        Returns
        -------
        states  : list of (B, dims[l]) — converged states per layer
        errors  : list of (B, dims[l]) — final prediction errors
        info    : dict with F trajectory and per-layer error norms
        """
        if n_steps is None:
            n_steps = self.n_inference_steps

        if x_obs.dim() > 2:
            x_obs = x_obs.view(x_obs.size(0), -1)

        states = self._init_states(x_obs)
        info = {'F_trajectory': [], 'error_norms': []}

        for step in range(n_steps):
            errors = self._compute_errors(states)

            # Free energy F = ½Σ_l ‖ε_l‖²
            F = sum(0.5 * e.pow(2).sum() for e in errors)

            if return_trajectory:
                info['F_trajectory'].append(F.item())

            # Adaptive step size: reduce near convergence
            step_lr = self.inference_lr * (1.0 / (1 + 0.01 * step))
            states = self._state_update(states, errors, lr=step_lr)

        # Final errors
        errors = self._compute_errors(states)
        info['error_norms'] = [e.norm().item() for e in errors]
        info['F_final'] = sum(0.5 * e.pow(2).sum().item() for e in errors)

        return states, errors, info

    def local_update(
        self,
        states: List[torch.Tensor],
        errors: List[torch.Tensor],
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Local weight updates using converged states and errors.

        ΔW_l = η · ε_l · (∂KAN_l/∂W_l)

        This is the 'learning phase' — called once per batch AFTER
        inference has converged. It is fully local (no backprop needed
        through the full computation graph).

        At convergence, this is mathematically equivalent to the
        backpropagation gradient (Song et al. NeurIPS 2020).

        Returns
        -------
        Dict with loss per layer (for logging).
        """
        losses = {}
        for l in range(self.n_layers):
            loss = self.layers[l].local_weight_update(
                x=states[l + 1],
                error=errors[l],
                lr=lr,
            )
            losses[f'layer_{l}'] = loss
        return losses

    def forward(
        self,
        x_obs: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Full forward pass with inference.

        Parameters
        ----------
        x_obs   : (B, obs_dim) or (B, 1, H, W)
        n_steps : test-time compute budget

        Returns
        -------
        z_top  : (B, dims[-1]) — top-level representation
        states : list of states per layer
        errors : list of prediction errors per layer
        """
        if x_obs.dim() > 2:
            x_obs = x_obs.view(x_obs.size(0), -1)

        states, errors, _ = self.inference(x_obs, n_steps=n_steps)
        z_top = states[-1]
        return z_top, states, errors

    def backprop_forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass using backpropagation (for comparison baseline).

        Chains KAN layers top-down and computes a classification/regression head.
        Used to compare local learning vs backprop on the same architecture.
        """
        if x_obs.dim() > 2:
            x_obs = x_obs.view(x_obs.size(0), -1)

        # Bottom-up pass (inverse of top-down generation)
        # For comparison: just use the last KAN layer as an encoder
        # Full comparison requires training a separate bottom-up path
        raise NotImplementedError(
            "Backprop comparison should use a separate bottom-up MLP/KAN encoder "
            "trained with standard cross-entropy. See experiments/exp_local_learning.py."
        )

    def free_energy(self, x_obs: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Compute the variational free energy F = ½Σ_l ‖ε_l‖² after inference.

        Lower F = better fit to observation under the model's priors.
        """
        if x_obs.dim() > 2:
            x_obs = x_obs.view(x_obs.size(0), -1)
        _, errors, info = self.inference(x_obs, n_steps=n_steps)
        F = sum(0.5 * e.pow(2).sum() for e in errors)
        return F


class ClassificationHead(nn.Module):
    """
    Small classification head on top of HierarchicalPCKAN.

    Used to compare:
        (A) Local PC learning (HierarchicalPCKAN.local_update)
        (B) Backpropagation baseline

    Parameters
    ----------
    input_dim  : int   — top-level representation dimension (= dims[-1])
    n_classes  : int
    """

    def __init__(self, input_dim: int, n_classes: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
