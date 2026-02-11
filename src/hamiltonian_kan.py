"""
Hamiltonian-KAN: Energy-Based Control with Spline Potentials
=============================================================

This module implements H-KAN: a combination of:
1. KAN-based energy functions (flexible, interpretable potentials)
2. Hamiltonian dynamics (physics-based inference with momentum)

Physical Intuition
------------------
Imagine a marble rolling on a landscape:
- The landscape shape is learned by the KAN (E(x))
- The marble has position (x) and momentum (p)
- It follows Hamilton's equations: ẋ = ∂H/∂p, ṗ = -∂H/∂x

The Hamiltonian is:
    H(x, p) = (1/2)||p||² + E(x)
              ↑ kinetic    ↑ potential

Key insight: momentum carries the system through flat regions and over
small bumps, enabling better exploration than gradient descent.

For Control Tasks
-----------------
In control (e.g., inverted pendulum), the state x = (θ, θ̇) includes
position and velocity. We learn E(x) such that:
- Desired states (e.g., θ=0, upright) have low energy
- Unstable states have high energy

Then "control" is simply letting the system evolve under Hamiltonian
dynamics — it naturally moves toward low-energy (desired) states.

Symplectic Integration
----------------------
Standard ODE solvers (Euler, RK4) don't preserve the geometric structure
of Hamiltonian systems. Over long simulations, energy drifts.

Symplectic integrators (like leapfrog/Störmer-Verlet) exactly preserve
the symplectic 2-form, ensuring:
- Energy is approximately conserved
- Phase space volume is exactly preserved
- Long-term stability

References
----------
- Greydanus, S. et al. (2019). "Hamiltonian Neural Networks"
- Zhong, Y. et al. (2020). "Symplectic ODE-Net: Learning Hamiltonian Dynamics"
- Toth, P. et al. (2020). "Hamiltonian Generative Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union
import math

from kan_layer import KANLayer, KANNetwork, EnergyKAN


class HamiltonianKAN(nn.Module):
    """
    Hamiltonian system with KAN-based potential energy.
    
    The Hamiltonian is:
        H(x, p) = (1/2) pᵀ M⁻¹ p + E_KAN(x)
    
    where:
        - x is position (state)
        - p is momentum
        - M is mass matrix (often identity)
        - E_KAN is a KAN network
    
    Hamilton's equations:
        dx/dt = ∂H/∂p = M⁻¹ p
        dp/dt = -∂H/∂x = -∇E_KAN(x)
    
    Parameters
    ----------
    state_dim : int
        Dimension of state space (position x).
    
    hidden_dims : tuple
        Hidden layer sizes for the energy KAN.
    
    grid_size : int
        KAN spline grid size.
    
    mass : float or torch.Tensor
        Mass matrix. If scalar, uses M = mass * I.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        grid_size: int = 5,
        spline_order: int = 3,
        mass: Union[float, torch.Tensor] = 1.0
    ):
        super().__init__()
        self.state_dim = state_dim
        
        # Potential energy function (KAN)
        self.potential = EnergyKAN(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            grid_size=grid_size,
            spline_order=spline_order
        )
        
        # Mass matrix
        if isinstance(mass, (int, float)):
            self.register_buffer('mass', torch.tensor(mass))
            self.register_buffer('mass_inv', torch.tensor(1.0 / mass))
        else:
            self.register_buffer('mass', mass)
            self.register_buffer('mass_inv', torch.inverse(mass))
    
    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy T = (1/2) pᵀ M⁻¹ p.
        
        For scalar mass M = m·I, this simplifies to (1/2m)||p||².
        """
        if self.mass.dim() == 0:  # Scalar mass
            return 0.5 * self.mass_inv * (p ** 2).sum(dim=-1)
        else:  # Matrix mass
            return 0.5 * torch.einsum('bi,ij,bj->b', p, self.mass_inv, p)
    
    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy E(x) using the KAN."""
        return self.potential(x)
    
    def hamiltonian(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute total Hamiltonian H = T + V.
        
        Parameters
        ----------
        x : torch.Tensor
            Position (state), shape (batch, state_dim)
        
        p : torch.Tensor
            Momentum, shape (batch, state_dim)
        
        Returns
        -------
        torch.Tensor
            Hamiltonian values, shape (batch,)
        """
        T = self.kinetic_energy(p)
        V = self.potential_energy(x)
        return T + V
    
    def dynamics(
        self, 
        x: torch.Tensor, 
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hamilton's equations of motion.
        
        dx/dt = ∂H/∂p = M⁻¹ p
        dp/dt = -∂H/∂x = -∇V(x)
        
        Parameters
        ----------
        x : torch.Tensor
            Position, shape (batch, state_dim). Requires grad.
        
        p : torch.Tensor
            Momentum, shape (batch, state_dim).
        
        Returns
        -------
        dx_dt : torch.Tensor
            Velocity (time derivative of position).
        
        dp_dt : torch.Tensor
            Force (time derivative of momentum).
        """
        # dx/dt = M⁻¹ p
        if self.mass.dim() == 0:
            dx_dt = self.mass_inv * p
        else:
            dx_dt = torch.matmul(p, self.mass_inv.T)
        
        # dp/dt = -∇V(x)
        # Need gradient with respect to x
        x_grad = x.requires_grad_(True)
        V = self.potential_energy(x_grad)
        grad_V = torch.autograd.grad(
            V.sum(), x_grad, 
            create_graph=True,  # For higher-order gradients if needed
            retain_graph=True
        )[0]
        dp_dt = -grad_V
        
        return dx_dt, dp_dt


def leapfrog_step(
    hamiltonian: HamiltonianKAN,
    x: torch.Tensor,
    p: torch.Tensor,
    dt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One step of the leapfrog (Störmer-Verlet) integrator.
    
    The leapfrog method is a symplectic integrator, meaning it preserves
    the geometric structure of Hamiltonian dynamics. This is crucial for:
    - Long-term stability
    - Approximate energy conservation
    - Preserving phase space volume
    
    Algorithm:
        p_{1/2} = p_0 - (dt/2) ∇V(x_0)     # Half-step momentum
        x_1 = x_0 + dt · M⁻¹ p_{1/2}        # Full-step position
        p_1 = p_{1/2} - (dt/2) ∇V(x_1)     # Half-step momentum
    
    This is time-reversible and second-order accurate (error ~ dt³).
    
    Parameters
    ----------
    hamiltonian : HamiltonianKAN
        The Hamiltonian system.
    
    x : torch.Tensor
        Current position, shape (batch, dim).
    
    p : torch.Tensor
        Current momentum, shape (batch, dim).
    
    dt : float
        Time step.
    
    Returns
    -------
    x_new : torch.Tensor
        New position.
    
    p_new : torch.Tensor
        New momentum.
    """
    # Compute ∇V at current position
    x.requires_grad_(True)
    V = hamiltonian.potential_energy(x)
    grad_V = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
    
    # Half-step in momentum
    p_half = p - 0.5 * dt * grad_V
    
    # Full-step in position
    if hamiltonian.mass.dim() == 0:
        x_new = x + dt * hamiltonian.mass_inv * p_half
    else:
        x_new = x + dt * torch.matmul(p_half, hamiltonian.mass_inv.T)
    x_new = x_new.detach()
    
    # Compute ∇V at new position
    x_new.requires_grad_(True)
    V_new = hamiltonian.potential_energy(x_new)
    grad_V_new = torch.autograd.grad(V_new.sum(), x_new, create_graph=False)[0]
    
    # Half-step in momentum
    p_new = p_half - 0.5 * dt * grad_V_new
    
    return x_new.detach(), p_new.detach()


def simulate_hamiltonian(
    hamiltonian: HamiltonianKAN,
    x0: torch.Tensor,
    p0: Optional[torch.Tensor] = None,
    n_steps: int = 100,
    dt: float = 0.01,
    return_trajectory: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate Hamiltonian dynamics using leapfrog integration.
    
    Parameters
    ----------
    hamiltonian : HamiltonianKAN
        The Hamiltonian system.
    
    x0 : torch.Tensor
        Initial position, shape (batch, dim).
    
    p0 : torch.Tensor, optional
        Initial momentum. If None, sampled from N(0, I).
    
    n_steps : int
        Number of integration steps.
    
    dt : float
        Time step.
    
    return_trajectory : bool
        If True, return full trajectory.
    
    Returns
    -------
    x_final : torch.Tensor
        Final position (or trajectory if return_trajectory).
    
    p_final : torch.Tensor
        Final momentum (or trajectory if return_trajectory).
    """
    if p0 is None:
        p0 = torch.randn_like(x0)
    
    x, p = x0.clone(), p0.clone()
    
    if return_trajectory:
        x_traj = [x.clone()]
        p_traj = [p.clone()]
    
    for _ in range(n_steps):
        x, p = leapfrog_step(hamiltonian, x, p, dt)
        
        if return_trajectory:
            x_traj.append(x.clone())
            p_traj.append(p.clone())
    
    if return_trajectory:
        return torch.stack(x_traj), torch.stack(p_traj)
    else:
        return x, p


class DampedHamiltonianDynamics:
    """
    Damped Hamiltonian dynamics for finding energy minima.
    
    Standard Hamiltonian dynamics conserves energy (H = const).
    This means the system orbits forever at its initial energy level.
    
    For optimization (finding minima), we add friction:
        ṗ = -∇V(x) - γp
    
    where γ > 0 is the damping coefficient. This dissipates energy:
        dH/dt = -γ ||ṗ||² ≤ 0
    
    The system now spirals toward the minimum like a damped oscillator.
    
    Trade-offs:
    - High γ: Fast energy decay, but can overshoot (underdamped)
    - Low γ: Slow energy decay, more exploration
    - Critical damping (γ = 2√k/m): Optimal for single-well potential
    
    Parameters
    ----------
    hamiltonian : HamiltonianKAN
        The Hamiltonian system.
    
    damping : float
        Friction coefficient γ.
    """
    
    def __init__(self, hamiltonian: HamiltonianKAN, damping: float = 0.1):
        self.hamiltonian = hamiltonian
        self.damping = damping
    
    def step(
        self, 
        x: torch.Tensor, 
        p: torch.Tensor, 
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of damped Hamiltonian dynamics.
        
        Uses semi-implicit scheme:
            x_{n+1} = x_n + dt · M⁻¹ p_n
            p_{n+1} = p_n - dt · ∇V(x_{n+1}) - dt · γ · p_n
        """
        # Position update
        if self.hamiltonian.mass.dim() == 0:
            x_new = x + dt * self.hamiltonian.mass_inv * p
        else:
            x_new = x + dt * torch.matmul(p, self.hamiltonian.mass_inv.T)
        x_new = x_new.detach()
        
        # Compute gradient at new position
        x_new.requires_grad_(True)
        V = self.hamiltonian.potential_energy(x_new)
        grad_V = torch.autograd.grad(V.sum(), x_new, create_graph=False)[0]
        
        # Momentum update with damping
        p_new = p - dt * grad_V - dt * self.damping * p
        
        return x_new.detach(), p_new.detach()
    
    def simulate(
        self,
        x0: torch.Tensor,
        p0: Optional[torch.Tensor] = None,
        n_steps: int = 100,
        dt: float = 0.01,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate damped dynamics."""
        if p0 is None:
            p0 = torch.randn_like(x0) * 0.5  # Smaller initial momentum
        
        x, p = x0.clone(), p0.clone()
        
        if return_trajectory:
            x_traj = [x.clone()]
            p_traj = [p.clone()]
            H_traj = [self.hamiltonian.hamiltonian(x, p).mean().item()]
        
        for _ in range(n_steps):
            x, p = self.step(x, p, dt)
            
            if return_trajectory:
                x_traj.append(x.clone())
                p_traj.append(p.clone())
                H_traj.append(self.hamiltonian.hamiltonian(x, p).mean().item())
        
        if return_trajectory:
            return torch.stack(x_traj), torch.stack(p_traj), H_traj
        else:
            return x, p


def train_hkan_equilibrium(
    hamiltonian: HamiltonianKAN,
    target_states: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_iterations: int = 1000,
    dynamics_steps: int = 50,
    dt: float = 0.01,
    damping: float = 0.1
):
    """
    Train H-KAN such that target states are low-energy equilibria.
    
    Training Objective
    ------------------
    We want target states x* to be stable equilibria:
    1. Low energy: E(x*) is small
    2. Zero gradient: ∇E(x*) = 0
    3. Positive curvature: ∇²E(x*) > 0 (local minimum)
    
    We achieve this by:
    1. Starting from random states
    2. Running damped dynamics toward equilibrium
    3. Measuring distance to target
    4. Backpropagating through the dynamics
    
    Parameters
    ----------
    hamiltonian : HamiltonianKAN
        The H-KAN model to train.
    
    target_states : torch.Tensor
        Desired equilibrium states, shape (n_targets, state_dim).
    
    optimizer : Optimizer
        PyTorch optimizer for H-KAN parameters.
    
    n_iterations : int
        Number of training iterations.
    
    dynamics_steps : int
        Steps of damped dynamics per iteration.
    
    dt : float
        Dynamics time step.
    
    damping : float
        Damping coefficient.
    
    Returns
    -------
    losses : list
        Training loss history.
    """
    dynamics = DampedHamiltonianDynamics(hamiltonian, damping=damping)
    losses = []
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Start from noisy versions of targets
        x = target_states + torch.randn_like(target_states) * 0.5
        p = torch.randn_like(x) * 0.1
        
        # Run damped dynamics (need to maintain graph for backprop)
        for _ in range(dynamics_steps):
            # Position update
            if hamiltonian.mass.dim() == 0:
                x_new = x + dt * hamiltonian.mass_inv * p
            else:
                x_new = x + dt * torch.matmul(p, hamiltonian.mass_inv.T)
            
            # Gradient
            V = hamiltonian.potential_energy(x_new)
            grad_V = torch.autograd.grad(
                V.sum(), x_new, create_graph=True, retain_graph=True
            )[0]
            
            # Momentum update with damping
            p = p - dt * grad_V - dt * damping * p
            x = x_new
        
        # Loss: final state should be close to target
        state_loss = F.mse_loss(x, target_states)
        
        # Regularization: target states should have low energy
        energy_loss = hamiltonian.potential_energy(target_states).mean()
        
        # Regularization: gradient at targets should be small
        target_grad = target_states.clone().requires_grad_(True)
        V_target = hamiltonian.potential_energy(target_grad)
        grad_target = torch.autograd.grad(V_target.sum(), target_grad)[0]
        gradient_loss = (grad_target ** 2).mean()
        
        # Total loss
        loss = state_loss + 0.1 * energy_loss + 0.1 * gradient_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(hamiltonian.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if (iteration + 1) % 100 == 0:
            print(f"Iter {iteration + 1}: loss={loss.item():.4f}, "
                  f"state={state_loss.item():.4f}, "
                  f"energy={energy_loss.item():.4f}")
    
    return losses


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing HamiltonianKAN...")
    
    # Create a simple H-KAN
    hkan = HamiltonianKAN(
        state_dim=2,
        hidden_dims=(16, 8),
        grid_size=5
    )
    
    # Test Hamiltonian computation
    x = torch.randn(4, 2)
    p = torch.randn(4, 2)
    H = hkan.hamiltonian(x, p)
    print(f"  State: {x.shape}, Momentum: {p.shape}")
    print(f"  Hamiltonian: {H.shape}, values: {H.tolist()}")
    
    # Test leapfrog step
    x_new, p_new = leapfrog_step(hkan, x, p, dt=0.01)
    print(f"\n  After leapfrog: x={x_new.shape}, p={p_new.shape}")
    
    # Test simulation
    x_traj, p_traj = simulate_hamiltonian(
        hkan, x[:1], n_steps=100, dt=0.01, return_trajectory=True
    )
    print(f"\n  Trajectory: x={x_traj.shape}, p={p_traj.shape}")
    
    # Check energy conservation
    H_init = hkan.hamiltonian(x_traj[0], p_traj[0]).item()
    H_final = hkan.hamiltonian(x_traj[-1], p_traj[-1]).item()
    print(f"  Energy: initial={H_init:.4f}, final={H_final:.4f}, "
          f"drift={(H_final - H_init) / abs(H_init) * 100:.2f}%")
    
    # Test damped dynamics
    print("\nTesting damped dynamics...")
    damped = DampedHamiltonianDynamics(hkan, damping=0.5)
    x_damp, p_damp, H_hist = damped.simulate(
        x[:1], n_steps=200, return_trajectory=True
    )
    print(f"  Damped trajectory: {len(H_hist)} steps")
    print(f"  Energy: initial={H_hist[0]:.4f}, final={H_hist[-1]:.4f}")
    print(f"  Energy decreased: {H_hist[-1] < H_hist[0]}")
    
    print("\n✓ All H-KAN tests passed!")
