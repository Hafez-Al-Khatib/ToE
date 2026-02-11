"""
Energy-Based Model for Natural Intelligence
============================================

This module implements the core "Ball in a Bowl" paradigm:
- An energy function E(x) that assigns low energy to "correct" states
- Langevin dynamics that lets the system "roll" to energy minima
- Contrastive training that shapes the energy landscape

Mathematical Foundation
-----------------------
We define a Boltzmann distribution over states:

    p(x) = (1/Z) exp(-E(x)/T)

where:
    - E(x) is the energy function (this neural network)
    - Z is the partition function (intractable)
    - T is the temperature

Key insight: We don't need to compute Z. We only need energy *differences*.

Inference as Physics
--------------------
Traditional NN: y = f(x)     # Single forward pass
Energy Model:   x* = argmin E(x)  # Iterative relaxation

We find x* by running gradient descent on the energy:

    x_{t+1} = x_t - ε ∇_x E(x_t)

This is equivalent to a ball rolling downhill on the energy landscape.

References
----------
- Hinton, G. (2002). "Training Products of Experts by Minimizing Contrastive Divergence"
- LeCun, Y. (2006). "A Tutorial on Energy-Based Learning"
- Du, Y. & Mordatch, I. (2019). "Implicit Generation and Modeling with Energy-Based Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math


class EnergyMLP(nn.Module):
    """
    Energy function parameterized by a Multi-Layer Perceptron.
    
    Maps input x ∈ ℝⁿ to scalar energy E(x) ∈ ℝ.
    
    Architecture Design Choices
    ---------------------------
    1. **Layer Normalization**: Stabilizes the energy scale across training.
       Without it, energies can grow unboundedly, causing gradient explosion.
    
    2. **SiLU Activation** (Swish): x * sigmoid(x)
       - Smooth and differentiable everywhere (important for Langevin dynamics)
       - Non-monotonic, allowing richer gradient landscapes
       - No "dead neurons" problem unlike ReLU
    
    3. **No output activation**: Energy is unbounded.
       We want E(x) ∈ ℝ, not constrained to [0,1] or similar.
    
    4. **Spectral Normalization** (optional): Controls Lipschitz constant.
       This bounds ||∇E||, preventing explosive gradients during inference.
    
    Mathematical Properties
    -----------------------
    - Smoothness: E(x) is infinitely differentiable (due to SiLU)
    - Continuity: Small input changes → small energy changes
    - Expressivity: Universal approximator for smooth energy functions
    
    Example
    -------
    >>> model = EnergyMLP(input_dim=784)  # For 28x28 images
    >>> x = torch.randn(32, 784)
    >>> energies = model(x)  # Shape: (32,)
    >>> 
    >>> # Energy should be a single scalar per sample
    >>> print(energies.shape)
    torch.Size([32])
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        use_spectral_norm: bool = True,
        activation: str = "silu"
    ):
        """
        Initialize the energy network.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input vectors (e.g., 784 for 28x28 MNIST)
        
        hidden_dims : tuple of int
            Sizes of hidden layers. Deeper = more expressive energy landscape,
            but also harder to train and slower inference.
        
        use_spectral_norm : bool
            If True, apply spectral normalization to all layers.
            This bounds the Lipschitz constant: ||E(x) - E(y)|| ≤ L||x - y||
            Stabilizes Langevin dynamics by bounding gradient magnitude.
        
        activation : str
            Activation function. Options: "silu", "gelu", "elu"
            - "silu" (recommended): Smooth, non-monotonic, no dead neurons
            - "gelu": Similar to SiLU, used in Transformers
            - "elu": Smooth, but less expressive than SiLU
        """
        super().__init__()
        
        # Store config for debugging/logging
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Choose activation function
        #
        # Why SiLU (Swish)?
        # -----------------
        # The gradient ∂E/∂x must be well-behaved for Langevin dynamics.
        # ReLU has discontinuous gradient at 0 → unstable dynamics.
        # SiLU is smooth everywhere: d/dx[x·σ(x)] = σ(x) + x·σ(x)·(1-σ(x))
        #
        activations = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU()
        }
        self.activation = activations.get(activation, nn.SiLU())
        
        # Build the network
        #
        # Each hidden layer: Linear → LayerNorm → Activation
        #
        # LayerNorm is critical:
        # - Prevents energy scale from drifting during training
        # - Normalizes activations, improving gradient flow
        # - More stable than BatchNorm for small batch sizes
        #
        layers = []
        dims = [input_dim] + list(hidden_dims)
        
        for i in range(len(dims) - 1):
            # Linear layer (optionally with spectral normalization)
            linear = nn.Linear(dims[i], dims[i + 1])
            
            if use_spectral_norm:
                # Spectral normalization divides weights by their spectral norm
                # This ensures the layer is 1-Lipschitz: ||Wx|| ≤ ||x||
                # Combined across layers: ||E(x) - E(y)|| ≤ ||x - y||
                linear = nn.utils.spectral_norm(linear)
            
            layers.extend([
                linear,
                nn.LayerNorm(dims[i + 1]),  # Normalize activations
                self.activation              # Smooth nonlinearity
            ])
        
        self.feature_net = nn.Sequential(*layers)
        
        # Final layer: project to scalar energy
        # No activation on output — energy is ℝ, not bounded
        self.energy_head = nn.Linear(hidden_dims[-1], 1)
        if use_spectral_norm:
            self.energy_head = nn.utils.spectral_norm(self.energy_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy E(x) for input states.
        
        The energy is a scalar measure of how "incorrect" or "unlikely" the
        input is according to the model. In our Boltzmann interpretation:
        
            p(x) ∝ exp(-E(x))
        
        So lower energy = higher probability = "better" state.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Energies of shape (batch_size,)
            Each entry is the scalar energy of that sample.
        
        Example
        -------
        >>> model = EnergyMLP(input_dim=784)
        >>> clean = torch.randn(16, 784) * 0.1   # Low variance (clean-ish)
        >>> noisy = torch.randn(16, 784) * 1.0   # High variance (noisy)
        >>> 
        >>> # After training, we expect:
        >>> E_clean = model(clean).mean()
        >>> E_noisy = model(noisy).mean()
        >>> assert E_noisy > E_clean  # Noisy has higher energy
        """
        # Ensure input is flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Extract features through hidden layers
        h = self.feature_net(x)
        
        # Project to scalar energy
        # squeeze(-1) removes the last dimension: (batch, 1) → (batch,)
        energy = self.energy_head(h).squeeze(-1)
        
        return energy


def langevin_dynamics(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    n_steps: int = 100,
    step_size: float = 10.0,
    noise_scale: float = 0.005,
    clip_grad: Optional[float] = 1.0,
    return_trajectory: bool = False
) -> torch.Tensor:
    """
    Run Langevin dynamics to sample/optimize from an energy function.
    
    Mathematical Background
    -----------------------
    Langevin dynamics is a physics-inspired algorithm for sampling from
    a Boltzmann distribution p(x) ∝ exp(-E(x)).
    
    The update rule is:
    
        x_{t+1} = x_t - (ε/2) ∇_x E(x_t) + √ε · η_t
    
    where:
        - ε is the step size
        - ∇_x E(x_t) is the energy gradient (computed via backprop)
        - η_t ~ N(0, I) is Gaussian noise
    
    The first term performs gradient descent (minimizes energy).
    The second term adds random exploration (prevents getting stuck).
    
    Convergence Theorem
    -------------------
    Under mild conditions, as t → ∞, the distribution of x_t converges
    to the target distribution:
    
        lim_{t→∞} p(x_t) = (1/Z) exp(-E(x))
    
    For our purposes (finding low-energy states), we typically:
    1. Start with noise_scale > 0 (exploration)
    2. Decrease noise_scale over time (simulated annealing)
    3. End with noise_scale ≈ 0 (pure gradient descent)
    
    Parameters
    ----------
    energy_fn : Callable
        Energy function E(x). Takes (batch, dim) tensor, returns (batch,) energies.
    
    x_init : torch.Tensor
        Initial state of shape (batch_size, dim).
        For denoising: the noisy input image.
    
    n_steps : int
        Number of Langevin steps. More steps = cleaner result but slower.
        Typical: 50-200 for denoising.
    
    step_size : float
        The ε parameter. Larger = faster but less stable.
        Typical: 1-100, depending on energy scale.
    
    noise_scale : float
        Scale of the stochastic term. Set to 0 for pure gradient descent.
        Typical: 0.005-0.01 for slight randomness, 0 for deterministic.
    
    clip_grad : float or None
        If not None, clip gradient to [-clip_grad, clip_grad].
        Prevents explosive updates when far from data manifold.
    
    return_trajectory : bool
        If True, return full trajectory instead of just final state.
        Useful for visualization.
    
    Returns
    -------
    torch.Tensor
        If return_trajectory is False: Final state, shape (batch_size, dim)
        If return_trajectory is True: Trajectory, shape (n_steps+1, batch_size, dim)
    
    Example
    -------
    >>> model = EnergyMLP(784)
    >>> noisy_image = torch.randn(1, 784) * 0.5 + clean_image
    >>> 
    >>> # "Denoise" by rolling down the energy landscape
    >>> denoised = langevin_dynamics(model, noisy_image, n_steps=100)
    >>>
    >>> # The energy should decrease
    >>> print(f"Before: {model(noisy_image).item():.2f}")
    >>> print(f"After: {model(denoised).item():.2f}")
    """
    # Clone so we don't modify the input
    x = x_init.clone().detach().requires_grad_(True)
    
    # Storage for trajectory if requested
    if return_trajectory:
        trajectory = [x.detach().clone()]
    
    for step in range(n_steps):
        # Compute energy
        energy = energy_fn(x)
        
        # Compute gradient: ∇_x E(x)
        # We sum energies because autograd computes ∂(sum)/∂x
        grad = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]
        
        # Optional: Clip gradients to prevent explosion
        # This is important when x is far from the data manifold
        # (energy gradient can be huge in unexplored regions)
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)
        
        # Langevin update
        # 
        #   x_{t+1} = x_t - ε∇E(x_t) + √(2ε)·noise
        #
        # The √(2ε) factor ensures correct stationary distribution
        # when sampling from the Boltzmann distribution.
        # For optimization (finding minima), this is less critical.
        #
        noise = torch.randn_like(x) * noise_scale
        x = x - step_size * grad + noise
        
        # Detach and re-enable gradients for next iteration
        x = x.detach().requires_grad_(True)
        
        if return_trajectory:
            trajectory.append(x.detach().clone())
    
    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    else:
        return x.detach()


def annealed_langevin_dynamics(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    n_steps: int = 100,
    step_size_init: float = 100.0,
    step_size_final: float = 1.0,
    noise_schedule: str = "linear",
    clip_grad: Optional[float] = 1.0
) -> torch.Tensor:
    """
    Langevin dynamics with annealed (decreasing) step size.
    
    Why Annealing?
    --------------
    Fixed step size faces a trade-off:
    - Large ε: Explores well, but oscillates near minimum
    - Small ε: Converges precisely, but can get stuck in local minima
    
    Solution: Start large (exploration), gradually decrease (exploitation).
    
    This is analogous to "simulated annealing" in optimization:
    - High temperature → Random exploration
    - Low temperature → Greedy descent
    
    The schedule is:
    
        ε_t = ε_init · (ε_final / ε_init)^(t / T)   [exponential]
    or
        ε_t = ε_init - (ε_init - ε_final) · (t / T)  [linear]
    
    Parameters
    ----------
    energy_fn : Callable
        Energy function E(x).
    
    x_init : torch.Tensor
        Initial state (noisy input).
    
    n_steps : int
        Total number of Langevin steps.
    
    step_size_init : float
        Initial (large) step size.
    
    step_size_final : float
        Final (small) step size.
    
    noise_schedule : str
        "linear" or "exponential" decay of step size and noise.
    
    Returns
    -------
    torch.Tensor
        Denoised / optimized state.
    """
    x = x_init.clone().detach().requires_grad_(True)
    
    for step in range(n_steps):
        # Compute annealed step size
        t = step / max(n_steps - 1, 1)  # t ∈ [0, 1]
        
        if noise_schedule == "exponential":
            # Exponential decay: ε_t = ε_0 · r^t, where r = ε_T / ε_0
            ratio = step_size_final / (step_size_init + 1e-8)
            step_size = step_size_init * (ratio ** t)
        else:  # Linear
            # Linear decay: ε_t = ε_0 + (ε_T - ε_0) · t
            step_size = step_size_init + (step_size_final - step_size_init) * t
        
        # Noise also decreases with step size (prevents late-stage oscillation)
        noise_scale = 0.01 * math.sqrt(step_size / step_size_init)
        
        # Standard Langevin step
        energy = energy_fn(x)
        grad = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]
        
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)
        
        noise = torch.randn_like(x) * noise_scale
        x = x - step_size * grad + noise
        x = x.detach().requires_grad_(True)
    
    return x.detach()


class SampleBuffer:
    """
    Persistent buffer for negative samples in Contrastive Divergence training.
    
    Background: Contrastive Divergence (CD)
    ---------------------------------------
    Training an EBM requires samples from the model distribution p_θ(x).
    
    The gradient of log-likelihood is:
    
        ∇_θ log p(x) = -∇_θ E(x_data) + E_{x~p_θ}[∇_θ E(x)]
    
    The first term pushes down energy of real data.
    The second term pushes up energy of model samples.
    
    Problem: Sampling from p_θ(x) requires running MCMC until convergence —
    this is extremely slow.
    
    Solution 1: CD-k
    ----------------
    Start MCMC from real data, run only k steps. The chain hasn't converged,
    but the endpoint is a "negative sample" of sorts. This works because:
    - The chain starts near real data (low energy)
    - After k steps, it drifts toward where the model "wants" to go
    - Raising energy there penalizes unwanted modes
    
    Problem: Short chains provide poor negative samples.
    
    Solution 2: Persistent CD (PCD) — This Class
    ---------------------------------------------
    Maintain a buffer of particles that persist across training steps.
    Each step:
    1. Sample a batch of particles from the buffer
    2. Run k Langevin steps to refine them
    3. Use as negative samples for the gradient
    4. Store updated particles back in buffer
    
    Over time, the buffer particles mix properly and provide high-quality
    negative samples without running new chains from scratch.
    
    Buffer Initialization
    ---------------------
    We reinitialize a fraction of particles each step to:
    - Prevent the buffer from becoming stale
    - Explore new modes as the energy landscape changes
    - Balance exploitation (refine existing) vs exploration (start fresh)
    
    Reference: Tieleman, T. (2008). "Training Restricted Boltzmann Machines
               using Approximations to the Likelihood Gradient"
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        sample_dim: int = 784,
        reinit_prob: float = 0.05
    ):
        """
        Initialize the sample buffer.
        
        Parameters
        ----------
        buffer_size : int
            Number of particles to maintain.
            Larger = better mixing, but more memory.
        
        sample_dim : int
            Dimension of each sample (e.g., 784 for MNIST).
        
        reinit_prob : float
            Probability of reinitializing each particle from noise.
            Higher = more exploration, Lower = better exploitation.
            Typical: 0.01-0.1
        """
        self.buffer_size = buffer_size
        self.sample_dim = sample_dim
        self.reinit_prob = reinit_prob
        
        # Initialize buffer with random noise
        # Using uniform [-1, 1] to cover typical data range
        self.buffer = torch.rand(buffer_size, sample_dim) * 2 - 1
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of particles from the buffer.
        
        Returns both the particles and their indices (for updating later).
        
        Parameters
        ----------
        batch_size : int
            Number of particles to sample.
        
        Returns
        -------
        samples : torch.Tensor
            Particles of shape (batch_size, sample_dim)
        indices : torch.Tensor
            Buffer indices, for updating after Langevin steps
        """
        indices = torch.randint(0, self.buffer_size, (batch_size,))
        samples = self.buffer[indices].clone()
        
        # Randomly reinitialize some samples (exploration)
        reinit_mask = torch.rand(batch_size) < self.reinit_prob
        n_reinit = reinit_mask.sum().item()
        if n_reinit > 0:
            samples[reinit_mask] = torch.rand(n_reinit, self.sample_dim, device=self.buffer.device) * 2 - 1
        
        return samples, indices
    
    def update(self, indices: torch.Tensor, new_samples: torch.Tensor):
        """
        Update buffer with refined particles.
        
        After running Langevin dynamics on sampled particles, store the
        results back in the buffer. This is the "persistent" part of PCD.
        
        Parameters
        ----------
        indices : torch.Tensor
            Buffer indices (from sample())
        new_samples : torch.Tensor
            Updated particles after Langevin steps
        """
        # Keep samples on the same device as buffer (detach to prevent memory leaks)
        self.buffer[indices] = new_samples.detach().to(self.buffer.device)
    
    def to(self, device: torch.device) -> "SampleBuffer":
        """Move buffer to device."""
        self.buffer = self.buffer.to(device)
        return self


def contrastive_divergence_loss(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x_positive: torch.Tensor,
    x_negative: torch.Tensor,
    alpha: float = 1.0,
    reg_weight: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the Contrastive Divergence loss.
    
    Mathematical Derivation
    -----------------------
    We want to maximize log-likelihood:
    
        L(θ) = E_{x~data}[log p_θ(x)]
             = E_{x~data}[-E_θ(x)] - log Z_θ
    
    Taking the gradient:
    
        ∇L = E_{data}[-∇E(x)] - E_{p_θ}[-∇E(x)]
           = -E_{data}[∇E(x)] + E_{p_θ}[∇E(x)]
    
    This means:
        - Decrease energy of real data (first term)
        - Increase energy of model samples (second term)
    
    The loss we minimize is the negative of this:
    
        Loss = E_{data}[E(x)] - E_{model}[E(x)]
    
    If we minimize this:
        - E(x_data) decreases (good: real data becomes more probable)
        - E(x_model) increases (good: "hallucinations" become less probable)
    
    Regularization
    --------------
    We add an L2 penalty on energies to prevent:
    1. Energy scale from growing unboundedly
    2. Numerical instability in training
    
        Loss = E(x_pos) - E(x_neg) + λ(E(x_pos)² + E(x_neg)²)
    
    Parameters
    ----------
    energy_fn : Callable
        The energy network E(x).
    
    x_positive : torch.Tensor
        Real data samples (clean images).
    
    x_negative : torch.Tensor
        Model samples (from buffer + Langevin).
    
    alpha : float
        Weight on the negative (model) term.
        Higher = more aggressive "pushing up" of negatives.
    
    reg_weight : float
        L2 regularization weight on energy magnitudes.
    
    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    
    metrics : dict
        Dictionary with diagnostic metrics:
        - "energy_pos": Mean energy of positive samples
        - "energy_neg": Mean energy of negative samples
        - "energy_diff": Difference (should be negative if training well)
    """
    # Compute energies
    E_pos = energy_fn(x_positive)
    E_neg = energy_fn(x_negative)
    
    # Contrastive loss: push down positives, push up negatives
    # We want E_pos < E_neg, so we minimize E_pos - E_neg
    cd_loss = E_pos.mean() - alpha * E_neg.mean()
    
    # L2 regularization on energy magnitudes
    # Prevents energy scale from exploding
    reg_loss = reg_weight * (E_pos.pow(2).mean() + E_neg.pow(2).mean())
    
    # Total loss
    loss = cd_loss + reg_loss
    
    # Diagnostics
    metrics = {
        "energy_pos": E_pos.mean().item(),
        "energy_neg": E_neg.mean().item(),
        "energy_diff": (E_pos.mean() - E_neg.mean()).item(),
        "cd_loss": cd_loss.item(),
        "reg_loss": reg_loss.item()
    }
    
    return loss, metrics


def denoising_score_matching_loss(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x_clean: torch.Tensor,
    noise_std: float = 0.3
) -> Tuple[torch.Tensor, dict]:
    """
    Denoising Score Matching loss — an alternative to Contrastive Divergence.
    
    Mathematical Background
    -----------------------
    Score Matching trains the energy function by matching its gradient
    (the "score") to the true data score:
    
        L(θ) = E_{x~data}[||∇_x E_θ(x) - ∇_x log p_data(x)||²]
    
    Problem: We don't know ∇_x log p_data(x).
    
    Solution: Denoising Score Matching (Vincent, 2011)
    If we corrupt data with known noise, the optimal score at the noisy
    point can be computed analytically:
    
        x̃ = x + σε,  where ε ~ N(0, I)
    
    The optimal denoising score is:
    
        ∇_x̃ log p(x̃|x) = -(x̃ - x)/σ² = -ε/σ
    
    So we train:
        L(θ) = E_{x,ε}[||∇_x̃ E_θ(x̃) + ε/σ||²]
    
    Intuitively: the energy gradient should point from noisy → clean.
    
    Comparison to CD
    ----------------
    | Aspect | Contrastive Divergence | Denoising Score Matching |
    |--------|------------------------|--------------------------|
    | Requires | MCMC sampling | Only noise corruption |
    | Speed | Slow (many Langevin steps) | Fast (single forward) |
    | Stability | Can diverge | Stable |
    | Expressivity | Full distribution | Only local gradients |
    
    Parameters
    ----------
    energy_fn : Callable
        The energy network.
    
    x_clean : torch.Tensor
        Clean data samples.
    
    noise_std : float
        Standard deviation of noise corruption.
        Higher = easier optimization, Lower = more precise scores.
    
    Returns
    -------
    loss : torch.Tensor
        The DSM loss.
    
    metrics : dict
        Diagnostic metrics.
    """
    # Sample noise and corrupt
    noise = torch.randn_like(x_clean) * noise_std
    x_noisy = x_clean + noise
    
    # Enable gradients for x_noisy
    x_noisy = x_noisy.requires_grad_(True)
    
    # Compute energy and its gradient w.r.t. input
    energy = energy_fn(x_noisy)
    score_estimate = torch.autograd.grad(
        energy.sum(), x_noisy, create_graph=True
    )[0]
    
    # The target score: -noise / noise_std
    # (Points from noisy back toward clean)
    target_score = -noise / noise_std
    
    # Score matching loss: MSE between estimated and target scores
    loss = (score_estimate - target_score).pow(2).mean()
    
    metrics = {
        "score_norm": score_estimate.norm(dim=-1).mean().item(),
        "target_norm": target_score.norm(dim=-1).mean().item(),
        "dsm_loss": loss.item()
    }
    
    return loss, metrics


class EnergyBasedDenoiser:
    """
    High-level wrapper for energy-based image denoising.
    
    This class provides a convenient interface for:
    1. Training an energy model on clean images
    2. Denoising corrupted images via Langevin dynamics
    3. Visualizing the energy landscape
    
    Usage Example
    -------------
    >>> from torchvision import datasets, transforms
    >>> 
    >>> # Load MNIST
    >>> transform = transforms.Compose([
    ...     transforms.ToTensor(),
    ...     transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ... ])
    >>> train_data = datasets.MNIST('./data', train=True, transform=transform)
    >>> 
    >>> # Create denoiser
    >>> denoiser = EnergyBasedDenoiser(input_dim=784)
    >>> 
    >>> # Train
    >>> denoiser.train(train_data, epochs=20)
    >>> 
    >>> # Denoise an image
    >>> noisy = clean + 0.3 * torch.randn_like(clean)
    >>> denoised = denoiser.denoise(noisy, n_steps=100)
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the energy-based denoiser.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input (784 for MNIST).
        
        hidden_dims : tuple
            Hidden layer sizes for the energy network.
        
        device : str
            Device for computation ("cuda" or "cpu").
        """
        self.device = torch.device(device)
        self.model = EnergyMLP(input_dim, hidden_dims).to(self.device)
        self.buffer = SampleBuffer(
            buffer_size=10000,
            sample_dim=input_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        langevin_steps: int = 20,
        langevin_lr: float = 10.0
    ) -> dict:
        """
        Perform one training step with Persistent Contrastive Divergence.
        
        Parameters
        ----------
        x_batch : torch.Tensor
            Batch of clean images, shape (batch_size, input_dim)
        
        langevin_steps : int
            Number of Langevin steps for negative sampling.
        
        langevin_lr : float
            Step size for Langevin dynamics.
        
        Returns
        -------
        dict
            Training metrics.
        """
        x_batch = x_batch.to(self.device)
        batch_size = x_batch.size(0)
        
        # Sample negative particles from buffer
        x_neg, indices = self.buffer.sample(batch_size)
        x_neg = x_neg.to(self.device)
        
        # Refine negatives with Langevin dynamics
        self.model.eval()
        x_neg = langevin_dynamics(
            self.model, x_neg,
            n_steps=langevin_steps,
            step_size=langevin_lr,
            noise_scale=0.01
        )
        self.model.train()
        
        # Update buffer
        self.buffer.update(indices, x_neg)
        
        # Compute loss and update
        self.optimizer.zero_grad()
        loss, metrics = contrastive_divergence_loss(
            self.model, x_batch, x_neg
        )
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return metrics
    
    @torch.no_grad()
    def denoise(
        self,
        x_noisy: torch.Tensor,
        n_steps: int = 100,
        step_size: float = 10.0,
        use_annealing: bool = True
    ) -> torch.Tensor:
        """
        Denoise an image by running Langevin dynamics.
        
        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy input image(s), shape (batch_size, input_dim) or (input_dim,)
        
        n_steps : int
            Number of Langevin steps.
        
        step_size : float
            Step size for dynamics.
        
        use_annealing : bool
            If True, use annealed Langevin (decreasing step size).
        
        Returns
        -------
        torch.Tensor
            Denoised image(s).
        """
        # Ensure batch dimension
        if x_noisy.dim() == 1:
            x_noisy = x_noisy.unsqueeze(0)
        
        x_noisy = x_noisy.to(self.device)
        self.model.eval()
        
        if use_annealing:
            result = annealed_langevin_dynamics(
                self.model, x_noisy,
                n_steps=n_steps,
                step_size_init=step_size * 10,
                step_size_final=step_size * 0.1
            )
        else:
            result = langevin_dynamics(
                self.model, x_noisy,
                n_steps=n_steps,
                step_size=step_size,
                noise_scale=0.001
            )
        
        return result
    
    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy of input."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x.to(self.device))
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check
    print("Testing EnergyMLP...")
    
    model = EnergyMLP(input_dim=100, hidden_dims=(64, 32))
    x = torch.randn(16, 100)
    
    energy = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Energy shape: {energy.shape}")
    print(f"  Energy values: {energy[:4].tolist()}")
    
    print("\nTesting Langevin dynamics...")
    x_init = torch.randn(4, 100)
    x_final = langevin_dynamics(model, x_init, n_steps=10, step_size=1.0)
    
    print(f"  Initial energy: {model(x_init).mean().item():.3f}")
    print(f"  Final energy: {model(x_final).mean().item():.3f}")
    
    print("\nTesting trajectory...")
    trajectory = langevin_dynamics(
        model, x_init[:1], n_steps=20, return_trajectory=True
    )
    print(f"  Trajectory shape: {trajectory.shape}")
    
    print("\n✓ All tests passed!")
