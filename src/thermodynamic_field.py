"""
Thermodynamic Neural Field (TNF) v3
====================================

A genuinely novel architecture derived from physics, not borrowed from ML.

No MLPs. No ConvNets. No Transformers. No attention.
The architecture IS a physical system.

Core Idea
---------
Treat data as a physical field u(x,y) on a 2D spatial domain.
Energy is a generalized Ginzburg-Landau functional:

    E[u] = Σₖ wₖ ‖Kₖ * u‖² + λ V(u) + Σᵢⱼ J(uᵢ, uⱼ)
           ↑ learned filters    ↑ poly potential  ↑ channel coupling

Inference = gradient descent on E[u]:
    u ← u - η ∇_u E(u) + μ (u_obs - u)

Training = 1-step denoising:
    L = ‖(x_noisy - η ∇_x E(x_noisy)) - x_clean‖²

ALL components are fully differentiable (no integer indexing).

Design Principles
-----------------
1. Spatial filters Kₖ: Learned 5×5 convolution kernels as energy terms
   - ∇_u |K*u|² = 2 K^T(K*u) — standard conv+transpose, fully differentiable
   - Captures edges, textures, orientations as preferred spatial patterns

2. Polynomial potential: V(u) = a·u² + b·u⁴ (double-well or single-well)
   - Fully differentiable (polynomial), no indexing
   - Models what pixel values the system prefers

3. Multi-channel Turing patterns: Activator-inhibitor dynamics
   - Different diffusion rates create pattern formation
   - Coupling via learned bilinear interaction

References
----------
- Turing (1952). "The Chemical Basis of Morphogenesis"
- Ginzburg & Landau (1950). Mean-field theory of phase transitions
- Swift & Hohenberg (1977). Pattern formation near onset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from kan import KAN


class ThermodynamicField(nn.Module):
    """
    A thermodynamic field system for image denoising.
    
    Architecture = Generalized Ginzburg-Landau energy functional.
    Inference = Energy gradient descent.
    Training = 1-step denoising supervision.
    
    Every component is fully differentiable — no integer indexing,
    no non-differentiable operations. This ensures all parameters
    receive gradients during training.
    """
    
    def __init__(
        self,
        n_channels: int = 4,
        height: int = 28,
        width: int = 28,
        n_filters: int = 16,
        filter_size: int = 5,
        dynamics_mode: Optional[list] = None,
        kan_hidden: Optional[list] = None
    ):
        """
        Parameters
        ----------
        n_channels : int
            Number of interacting field channels (Turing patterns need ≥2).
        n_filters : int
            Number of learned spatial filters per channel.
        filter_size : int
            Kernel size for spatial filters (e.g., 5 = 5×5).
        dynamics_mode : list of str, optional
            List of 'non_conserved' or 'conserved' per channel.
            Default is 'non_conserved' (Allen-Cahn) for all.
        """
        super().__init__()
        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.dynamics_mode = dynamics_mode
        
        # === LEARNED: Spatial filter bank per channel ===
        # Energy: Σₖ wₖ |Kₖ * uᵢ|²
        # Gradient: 2 Σₖ wₖ Kₖᵀ(Kₖ * uᵢ) — fully differentiable
        self.spatial_filters = nn.ParameterList([
            nn.Parameter(torch.randn(n_filters, 1, filter_size, filter_size) * 0.05)
            for _ in range(n_channels)
        ])
        self.filter_weights = nn.ParameterList([
            nn.Parameter(torch.ones(n_filters) * 0.01)
            for _ in range(n_channels)
        ])
        
        # === LEARNED: KAN Potential V_KAN(u) ===
        # Replaces simple polynomial a*u^2 + b*u^4 with a specialized B-Spline network.
        # This acts as the "Memory" of the system, learning complex energy landscapes.
        # Input: n_channels (vector of field values at a pixel). Output: 1 (Energy density).
        if kan_hidden is None:
            kan_hidden = [16]
            
        self.potential_kan = KAN(
            layers_hidden=[n_channels] + kan_hidden + [1],
            grid_size=5,
            spline_order=3,
            scale_noise=0.2, # Stronger initial gradients
            scale_base=1.0,
            scale_spline=1.0,
        )
        # Note: We remove potential_a and potential_b parameters
        
        # === LEARNED: Diffusion coefficients (Turing: slow activator, fast inhibitor) ===
        initial_alphas = torch.zeros(n_channels)
        for i in range(n_channels):
            initial_alphas[i] = -0.5 + (i % 2) * 1.0  # alternating slow/fast
        self.log_alphas = nn.Parameter(initial_alphas)
        
        # === LEARNED: Inter-channel coupling ===
        # Bilinear coupling: J(uᵢ, uⱼ) = cᵢⱼ · uᵢ · uⱼ (per-pixel)
        # Fully differentiable, captures activator-inhibitor interaction
        n_couplings = n_channels * (n_channels - 1) // 2
        self.coupling_weights = nn.Parameter(torch.randn(n_couplings) * 0.01)
        
        # === LEARNED: Channel initialization (per-channel gain + bias) ===
        # For single-channel input: hidden channels = gain * image + bias
        # For multi-channel input: project via learned linear weights
        self.init_gains = nn.Parameter(torch.ones(n_channels - 1) * 0.5)
        self.init_biases = nn.Parameter(torch.zeros(n_channels - 1))
        
        # === LEARNED: Multi-channel input projection ===
        # When input has K channels, project to n_channels via learned weights
        # This is a simple linear combination (NOT a neural net)
        self.input_proj_weight = nn.Parameter(torch.eye(n_channels)[:, :min(n_channels, 9)] * 0.5)
        self.input_proj_bias = nn.Parameter(torch.zeros(n_channels))
        
        # === LEARNED: Inference step size and data weight ===
        self.log_step_size = nn.Parameter(torch.tensor(-1.0))  # exp(-1) ≈ 0.37
        self.log_data_weight = nn.Parameter(torch.tensor(-5.0))  # exp(-5) ≈ 0.0067 (Was -2.0)
        # Lower data weight forces system to rely on Physics (KAN) initially.
    
    @property
    def alphas(self):
        """Diffusion coefficients (always positive)."""
        return torch.exp(self.log_alphas)
    
    @property
    def step_size(self):
        """Learned inference step size."""
        return torch.exp(self.log_step_size)
    
    @property
    def data_weight(self):
        """Learned data fidelity weight."""
        return torch.exp(self.log_data_weight)
    
    def initialize_fields(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create multi-channel field from input.
        
        Supports two modes:
        - Single-channel input (B, 1, H, W): ch0 = image, ch1..K = gain*image + bias
        - Multi-channel input (B, C, H, W): project via learned linear weights
        """
        B, C_in, H, W = image.shape
        
        if C_in == 1:
            # Original behavior: single-channel input → multi-channel field
            channels = [image]
            for k in range(self.n_channels - 1):
                hidden = self.init_gains[k] * image + self.init_biases[k]
                channels.append(hidden)
            return torch.cat(channels, dim=1)
        else:
            # Multi-channel input → project to n_channels via learned linear weights
            # weight: (n_channels, min(n_channels, C_in))
            C_proj = min(C_in, self.input_proj_weight.shape[1])
            # Flatten spatial, apply linear, reshape
            x = image[:, :C_proj]  # (B, C_proj, H, W)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C_proj)
            w = self.input_proj_weight[:, :C_proj]  # (n_channels, C_proj)
            out = torch.matmul(x, w.T) + self.input_proj_bias  # (B, H, W, n_channels)
            return out.permute(0, 3, 1, 2)  # (B, n_channels, H, W)
    
    def compute_energy(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Generalized Ginzburg-Landau energy:
        
            E = Σᵢ [Σₖ wₖ |Kₖ * uᵢ|² + αᵢ |∇uᵢ|² + aᵢ uᵢ² + bᵢ uᵢ⁴]
                + Σᵢ<ⱼ cᵢⱼ ∫ uᵢ · uⱼ
        
        All operations are fully differentiable conv2d, polynomial, bilinear.
        """
        B, C, H, W = fields.shape
        energy = torch.zeros(B, device=fields.device)
        
        alphas = self.alphas
        pad = self.spatial_filters[0].shape[-1] // 2
        
        for i in range(C):
            u_i = fields[:, i:i+1, :, :]  # (B, 1, H, W)
            
            # --- Spatial filter energy: Σₖ |wₖ| · |Kₖ * uᵢ|² ---
            filtered = F.conv2d(u_i, self.spatial_filters[i], padding=pad)
            filter_energy = (self.filter_weights[i].abs().view(1, -1, 1, 1) *
                           filtered.pow(2)).sum(dim=(1, 2, 3))
            energy = energy + filter_energy
            
            # --- Gradient energy: αᵢ |∇uᵢ|² ---
            grad_x = u_i[:, :, :, 1:] - u_i[:, :, :, :-1]
            grad_y = u_i[:, :, 1:, :] - u_i[:, :, :-1, :]
            grad_energy = alphas[i] * (grad_x.pow(2).sum(dim=(1,2,3)) +
                                       grad_y.pow(2).sum(dim=(1,2,3)))
            energy = energy + grad_energy
            
            # --- Polynomial potential REMOVED (Handled by KAN below) ---
            # pot = self.potential_a[i] * u_i.pow(2) + self.potential_b[i] * u_i.pow(4)
            # energy = energy + pot.sum(dim=(1, 2, 3))
        # --- KAN Potential: V(u) ---
        # Reshape fields to (B*H*W, C)
        # fields: (B, C, H, W) -> (B, H, W, C) -> (N, C)
        fields_flat = fields.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Compute V(u) via KAN
        # kan(x) output shape: (B*H*W, 1)
        pot_flat = self.potential_kan(fields_flat)
        pot_per_sample = pot_flat.view(B, H, W).sum(dim=(1, 2))
        
        energy = energy + pot_per_sample
        
        # --- Bilinear coupling: cᵢⱼ ∫ uᵢ · uⱼ ---
        idx = 0
        for i in range(C):
            for j in range(i + 1, C):
                coupling_e = self.coupling_weights[idx] * (
                    fields[:, i] * fields[:, j]
                ).sum(dim=(1, 2))
                energy = energy + coupling_e
                idx += 1
        
        # Normalize
        energy = energy / (H * W)
        
        return energy
    
    def compute_energy_from_image(self, image: torch.Tensor) -> torch.Tensor:
        """Compute energy from image input (any shape)."""
        if image.dim() == 2:
            image = image.view(-1, 1, self.height, self.width)
        elif image.dim() == 3:
            image = image.unsqueeze(1)
        
        fields = self.initialize_fields(image)
        return self.compute_energy(fields)
    
    def one_step_denoise(self, x_noisy: torch.Tensor) -> torch.Tensor:
        """
        Single energy gradient step for denoising.
        
        x_denoised = x_noisy - η * ∇_x E(x_noisy)
        
        Used during training (1-step loss) and chained during inference.
        All operations are differentiable for end-to-end training.
        """
        x_input = x_noisy.requires_grad_(True)
        energy = self.compute_energy_from_image(x_input)
        
        grad = torch.autograd.grad(
            energy.sum(), x_input, create_graph=True
        )[0]
        
        return x_input - self.step_size * grad
    
    def evolve(
        self,
        image: torch.Tensor,
        n_steps: int = 50,
        return_trajectory: bool = False,
        advection_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Denoise by iterated energy gradient descent (+ optional advection).
        
        Equation: ∂u/∂t = -∂E/∂u - v·∇u
        
        Parameters
        ----------
        advection_field : torch.Tensor, optional (B, 2, H, W) or (B, C, 2, H, W)
            Velocity field v=(vx, vy) for advecting the state u.
            Supports per-channel advection.
        """
        if image.dim() == 2:
            image = image.view(-1, 1, self.height, self.width)
        elif image.dim() == 3:
            image = image.unsqueeze(1)
        
        u = image.clone().detach()
        anchor = image.detach().clone()
        
        if return_trajectory:
            trajectory = [u.clone()]
        
        for step in range(n_steps):
            u_input = u.requires_grad_(True)
            energy = self.compute_energy_from_image(u_input)
            
            grad = torch.autograd.grad(
                energy.sum(), u_input, create_graph=False
            )[0]
            
            # Gradient Flow: dt = -grad E + data_fidelity
            update = -self.step_size.detach() * grad + \
                     self.data_weight.detach() * (anchor - u_input.detach())
            
            # Advection: -v·∇u
            if advection_field is not None:
                u_curr = u_input.detach()
                u_pad = F.pad(u_curr, (1, 1, 1, 1), mode='replicate')
                du_dx = (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2]) / 2.0
                du_dy = (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1]) / 2.0
                
                if advection_field.dim() == 5: # (B, C, 2, H, W)
                    vy = advection_field[:, :, 0] 
                    vx = advection_field[:, :, 1]
                else: # (B, 2, H, W)
                    vy = advection_field[:, 0].unsqueeze(1) 
                    vx = advection_field[:, 1].unsqueeze(1)
                
                advection = -(vx * du_dx + vy * du_dy)
                update = update + self.step_size.detach() * advection
            
            u = u_input.detach() + update
            u = torch.clamp(u, -1.5, 1.5)
            
            if return_trajectory:
                trajectory.append(u.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        
        return u.detach()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x)."""
        return self.compute_energy_from_image(x)
    
    def trainable_evolve(
        self,
        fields: torch.Tensor,
        n_steps: int = 10,
        anchor: torch.Tensor = None,
        advection_field: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Evolve fields with full gradient tracking.
        Supports per-channel advection.
        """
        if anchor is None:
            anchor = fields.detach()
        
        u = fields
        # Ensure we can compute dE/du
        if not u.requires_grad:
            u.requires_grad_(True)
        
        for step in range(n_steps):
            energy = self.compute_energy(u)
            
            grad = torch.autograd.grad(
                energy.sum(), u, create_graph=True
            )[0]
            
            # Gradient descent + data fidelity anchoring
            update = -self.step_size * grad + self.data_weight * (anchor - u)
            
            # Advection: -v·∇u (differentiable)
            if advection_field is not None:
                u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
                du_dx = (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2]) / 2.0
                du_dy = (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1]) / 2.0
                
                if advection_field.dim() == 5: # (B, C, 2, H, W)
                    vy = advection_field[:, :, 0] 
                    vx = advection_field[:, :, 1]
                else: # (B, 2, H, W)
                    vy = advection_field[:, 0].unsqueeze(1) 
                    vx = advection_field[:, 1].unsqueeze(1)
                
                advection = -(vx * du_dx + vy * du_dy)
                update = update + self.step_size * advection
                
            u = u + update
        
        return u


def one_step_denoising_loss(
    model: ThermodynamicField,
    x_clean: torch.Tensor,
    noise_std: float = 0.3,
) -> Tuple[torch.Tensor, dict]:
    """
    1-Step Denoising Loss: the efficient training objective.
    
    L = ‖(x_noisy - η * ∇_x E(x_noisy)) - x_clean‖²
    
    This is equivalent to training the energy gradient to be a
    denoising operator. Only requires ONE forward + autograd.grad
    with create_graph=True — fast and memory efficient.
    
    Why it works:
    - ∇_x E contains learned spatial filter responses + potential + coupling
    - Training selects the energy functional whose gradient removes noise
    - The Ginzburg-Landau structure constrains ∇E to physically meaningful forms
    
    Why it's physics-native:
    - The denoiser IS the energy gradient
    - Physical interpretation: noisy data = high energy, denoising = relaxation
    - The energy structure (spatial filters + potential + coupling) is preserved
    """
    H, W = model.height, model.width
    
    if x_clean.dim() == 2:
        x_clean = x_clean.view(-1, 1, H, W)
    
    # Corrupt with noise
    noise = torch.randn_like(x_clean) * noise_std
    x_noisy = x_clean + noise
    
    # 1-step denoising via energy gradient
    x_denoised = model.one_step_denoise(x_noisy)
    
    # Direct supervision
    loss = F.mse_loss(x_denoised, x_clean)
    
    # Monitoring metrics
    with torch.no_grad():
        E_clean = model(x_clean.view(x_clean.size(0), -1)).mean().item()
        E_noisy = model(x_noisy.view(x_noisy.size(0), -1)).mean().item()
    
    metrics = {
        "loss": loss.item(),
        "energy_clean": E_clean,
        "energy_noisy": E_noisy,
        "energy_gap": E_noisy - E_clean,
        "step_size": model.step_size.item(),
    }
    
    return loss, metrics


def multi_step_denoising_loss(
    model: ThermodynamicField,
    x_clean: torch.Tensor,
    n_steps: int = 5,
    noise_std: float = 0.3
) -> Tuple[torch.Tensor, dict]:
    """
    Multi-Step Denoising Loss: Backprop through time (BPTT).
    
    L = ‖evolve(x_noisy, K) - x_clean‖²
    
    Teaches the field dynamics how to solve problems that require
    iterative propagation (like filling in large missing regions
    or global constraint satisfaction). High memory cost!
    """
    H, W = model.height, model.width
    
    if x_clean.dim() == 2:
        x_clean = x_clean.view(-1, 1, H, W)
    elif x_clean.dim() == 3:
        x_clean = x_clean.unsqueeze(1) # B, 1, H, W
        
    # Corrupt
    noise = torch.randn_like(x_clean) * noise_std
    x_noisy = x_clean + noise
    
    # Initialize fields (map input ch -> hidden ch)
    fields = model.initialize_fields(x_noisy)
    
    # Evolve with gradient tracking through all steps
    u_final = model.trainable_evolve(
        fields, 
        n_steps=n_steps, 
        anchor=fields.detach() # Anchor to initial noisy state
    )
    
    # We want the FIRST channel (corresponding to input) to match x_clean
    # Or should we decode? The field IS the image for channel 0.
    # initialize_fields puts image in ch0.
    x_pred = u_final[:, 0:1, :, :]
    
    loss = F.mse_loss(x_pred, x_clean)
    
    return loss, {"loss": loss.item()}


# =============================================================================
# Quick Validation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Thermodynamic Neural Field v3 — Self-Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ThermodynamicField(n_channels=4, n_filters=16, filter_size=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nArchitecture: {model.n_channels}-channel field, {n_params} learnable params")
    print(f"  Spatial filters: {model.n_filters}×{4} channels, {model.spatial_filters[0].shape}")
    print(f"  Diffusion rates (α): {model.alphas.data.cpu().tolist()}")
    print(f"  Step size η: {model.step_size.item():.4f}")
    print(f"  Data weight μ: {model.data_weight.item():.4f}")
    
    x = torch.randn(4, 1, 28, 28, device=device)
    
    # Test energy
    energy = model(x.view(4, 784))
    print(f"\nEnergy: shape={energy.shape}, mean={energy.mean().item():.3f}")
    
    # Test 1-step denoising loss
    loss, metrics = one_step_denoising_loss(model, x, noise_std=0.3)
    print(f"\n1-Step Denoising Loss: {metrics['loss']:.6f}")
    print(f"  Energy gap: {metrics['energy_gap']:.4f}")
    print(f"  Step size: {metrics['step_size']:.4f}")
    
    # Test gradient flow — should be ALL params now
    loss.backward()
    grad_info = []
    for name, p in model.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().max() > 0
        grad_info.append((name, has_grad, tuple(p.shape)))
    
    n_grads = sum(1 for _, g, _ in grad_info if g)
    n_total = len(grad_info)
    print(f"\nGradient flow: {n_grads}/{n_total} params have nonzero gradients")
    for name, has_grad, shape in grad_info:
        status = "✓" if has_grad else "✗"
        print(f"  {status} {name} {shape}")
    
    # Test inference evolution
    model.zero_grad()
    denoised = model.evolve(x, n_steps=20)
    print(f"\nEvolve (inference): {x.shape} → {denoised.shape}")
    
    print(f"\n{'✓' if n_grads == n_total else '⚠'} TNF v3: {n_grads}/{n_total} params trained, {n_params} total")
