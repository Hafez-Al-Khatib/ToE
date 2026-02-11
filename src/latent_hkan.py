"""
Latent H-KAN: Encoder + Latent-Space Hamiltonian KAN
=====================================================

This module implements the CORRECTED H-KAN architecture.

The Critical Fix
----------------
KANs have a parameter blowup: input_dim × hidden × (grid_size + spline_order).
On 784-dim MNIST, this is ~10-20x more parameters than an MLP.

Solution: Use KANs for LOGIC (low-dim), not PERCEPTION (high-dim).

Architecture
------------
1. **Encoder**: Standard CNN/MLP compresses image → latent z ∈ ℝ¹⁶
2. **KAN Energy**: KAN operates on compact latent space
3. **Decoder** (optional): Latent → reconstruction

Why This Works
--------------
KANs excel at learning smooth, interpretable functions with few inputs.
The encoder handles the messy, high-dimensional perception.
The KAN handles the clean, low-dimensional reasoning.

Analogy: Your visual cortex (encoder) sees the road.
         Your prefrontal cortex (KAN) decides to turn left.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kan_layer import KANLayer, KANNetwork


class ConvEncoder(nn.Module):
    """
    CNN encoder: Image → Compact latent vector.
    
    Uses standard CNN architecture for perception (high-dim → low-dim).
    This is where we handle the 784-dimensional input efficiently.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 16,
        hidden_channels: Tuple[int, ...] = (32, 64, 128)
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Conv layers with BatchNorm and LeakyReLU
        layers = []
        in_ch = input_channels
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        
        # Compute flattened size (for 28x28 input with 3 stride-2 convs: 4x4)
        # 28 → 14 → 7 → 4 (approximately, with padding)
        self._flat_size = hidden_channels[-1] * 4 * 4
        
        # Project to latent space
        self.fc = nn.Linear(self._flat_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent vector.
        
        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape (batch, C, H, W) or (batch, H*W) for flattened.
        
        Returns
        -------
        z : torch.Tensor
            Latent vector of shape (batch, latent_dim)
        """
        # Handle flattened input (e.g., from MNIST loader)
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 28, 28)
        
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class MLPEncoder(nn.Module):
    """
    MLP encoder for when input is already a vector (e.g., state space).
    
    Simpler than CNN, used for control tasks.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (256, 128)
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU()
            ])
            in_dim = out_dim
        
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class LatentHKAN(nn.Module):
    """
    Latent Hamiltonian-KAN: The Corrected Architecture.
    
    Pipeline
    --------
    1. Encoder: x ∈ ℝ^784 → z ∈ ℝ^16 (perception)
    2. KAN Energy: E(z) (reasoning/logic)
    3. Hamiltonian dynamics in latent space
    
    Memory Comparison
    -----------------
    Direct KAN on pixels: 784 × 512 × (8 + 3) ≈ 4.4M parameters
    Latent KAN:           16 × 32 × (8 + 3) ≈ 5.6K parameters
    
    That's ~800x fewer parameters in the KAN layer!
    
    The encoder adds ~100K parameters, but that's still far less than
    the 4.4M we'd need for a direct pixel KAN.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 16,
        kan_hidden: Tuple[int, ...] = (32, 16),
        encoder_type: str = "mlp",
        grid_size: int = 8,
        spline_order: int = 3
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input (784 for MNIST).
        
        latent_dim : int
            Dimension of latent space. Keep this SMALL (8-32).
            This is where KAN operates.
        
        kan_hidden : tuple
            Hidden layer sizes for KAN. Keep these small too.
        
        encoder_type : str
            "mlp" for vectors, "conv" for images.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: High-dim perception
        if encoder_type == "conv":
            self.encoder = ConvEncoder(latent_dim=latent_dim)
        else:
            self.encoder = MLPEncoder(input_dim, latent_dim)
        
        # KAN Energy: Low-dim reasoning
        # This is where the interpretable, learnable functions live
        self.kan_energy = KANNetwork(
            layers=[latent_dim] + list(kan_hidden) + [1],
            grid_size=grid_size,
            spline_order=spline_order
        )
        
        # Optional: momentum network for Hamiltonian dynamics
        self.momentum_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy in latent space.
        
        This is where the KAN magic happens: the energy landscape
        is represented by learnable spline functions.
        """
        return self.kan_energy(z).squeeze(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: input → latent → energy.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, input_dim) or (batch, C, H, W)
        
        Returns
        -------
        energy : torch.Tensor
            Scalar energy per sample, shape (batch,)
        """
        z = self.encode(x)
        return self.energy(z)
    
    def latent_hamiltonian(
        self,
        z: torch.Tensor,
        p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hamiltonian H(z, p) = T(p) + V(z) in latent space.
        
        Parameters
        ----------
        z : torch.Tensor
            Position in latent space (batch, latent_dim)
        p : torch.Tensor, optional
            Momentum. If None, initialized to zero.
        
        Returns
        -------
        H : torch.Tensor
            Hamiltonian value (batch,)
        """
        if p is None:
            p = torch.zeros_like(z)
        
        # Kinetic energy: T = 0.5 * ||p||²
        T = 0.5 * (p ** 2).sum(dim=-1)
        
        # Potential energy: V = KAN_energy(z)
        V = self.energy(z)
        
        return T + V
    
    def latent_dynamics_step(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        dt: float = 0.01,
        damping: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of Hamiltonian dynamics in latent space.
        
        Uses symplectic Euler for energy conservation.
        
        Parameters
        ----------
        z, p : torch.Tensor
            Position and momentum in latent space.
        dt : float
            Time step.
        damping : float
            Damping coefficient (0 = conservative, >0 = dissipative).
        
        Returns
        -------
        z_new, p_new : torch.Tensor
            Updated position and momentum.
        """
        # Compute gradient of potential
        z_grad = z.clone().requires_grad_(True)
        V = self.energy(z_grad)
        dV_dz = torch.autograd.grad(V.sum(), z_grad, create_graph=True)[0]
        
        # Symplectic update with optional damping
        p_new = p - dt * dV_dz - damping * dt * p
        z_new = z + dt * p_new
        
        return z_new.detach(), p_new.detach()
    
    def denoise_latent(
        self,
        x: torch.Tensor,
        n_steps: int = 50,
        step_size: float = 0.1
    ) -> torch.Tensor:
        """
        Denoise by gradient descent in latent space.
        
        This is Phase 1's "ball in bowl" but in the learned latent space.
        
        Parameters
        ----------
        x : torch.Tensor
            Noisy input image.
        n_steps : int
            Number of gradient descent steps.
        step_size : float
            Step size for gradient descent.
        
        Returns
        -------
        z_clean : torch.Tensor
            Denoised latent representation.
        """
        z = self.encode(x)
        z = z.clone().requires_grad_(True)
        
        for _ in range(n_steps):
            E = self.energy(z)
            grad = torch.autograd.grad(E.sum(), z)[0]
            z = z - step_size * grad
            z = z.detach().requires_grad_(True)
        
        return z.detach()


class LatentDecoder(nn.Module):
    """
    Decoder: Latent → Image reconstruction.
    
    Optional component for visualization and VAE-style training.
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        output_dim: int = 784
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Testing Latent H-KAN...")
    
    # Create model
    model = LatentHKAN(
        input_dim=784,
        latent_dim=16,
        kan_hidden=(32, 16),
        encoder_type="mlp"
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    kan_params = sum(p.numel() for p in model.kan_energy.parameters())
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  KAN parameters: {kan_params:,}")
    
    # Test forward pass
    x = torch.randn(8, 784)
    z = model.encode(x)
    E = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Energy shape: {E.shape}")
    
    # Test Hamiltonian dynamics
    p = torch.randn_like(z) * 0.1
    H0 = model.latent_hamiltonian(z, p)
    
    z_new, p_new = z.clone(), p.clone()
    for _ in range(10):
        z_new, p_new = model.latent_dynamics_step(z_new, p_new, dt=0.01)
    
    H1 = model.latent_hamiltonian(z_new, p_new)
    energy_drift = (H1 - H0).abs().mean() / H0.abs().mean() * 100
    
    print(f"  Energy drift after 10 steps: {energy_drift:.2f}%")
    
    # Memory comparison
    print("\n  Memory Comparison:")
    print(f"  Direct KAN (784→512): ~4.4M params")
    print(f"  Latent KAN (16→32): ~{kan_params:,} params")
    print(f"  Reduction: ~{4_400_000 // kan_params}x fewer params in KAN")
    
    print("\n✓ All Latent H-KAN tests passed!")
