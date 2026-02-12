
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from thermodynamic_field import ThermodynamicField
from kan import KAN

class HamiltonianField(ThermodynamicField):
    """
    A Generalist Thermodynamic Field where the Energy Functional is learned by a KAN.
    
    Philosophy:
    -----------
    Instead of hardcoding V(u), Diffusion, and Coupling, we learn:
    E[u] = ∫ KAN( Features(u) ) dx
    
    where Features(u) are local spatial correlations (Conv2d results).
    The KAN learns the "Laws of Physics" (the Hamiltonian density) for the specific task.
    """
    def __init__(
        self,
        n_channels: int = 4,
        height: int = 28,
        width: int = 28,
        n_filters: int = 16,
        filter_size: int = 5,
        kan_hidden: list = [32, 1], # [Hidden, Output]
        kan_grid_size: int = 5,
    ):
        super().__init__(n_channels, height, width, n_filters, filter_size)
        
        # Override the parameter lists from parent to avoid unused params
        # We REUSE self.spatial_filters for feature extraction
        # But we don't need potential_a, potential_b, coupling_weights, etc.
        # We will ignore them or delete them.
        # Ideally, we'd refactor, but inheritance is cleaner for compatibility.
        
        # The KAN takes 'n_filters' spatial features per channel as input
        # So total input features = n_channels * n_filters
        input_dim = n_channels * n_filters
        
        self.kan = KAN(
            layers_hidden=[input_dim] + kan_hidden,
            grid_size=kan_grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            grid_range=[-2, 2],
        )
        
        # We still use the parent's init_fields, evolve, etc.
        # We only override compute_energy.
        
    def compute_energy(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Compute General Learned Energy: E = ∫ KAN( Conv(u) ) dx
        """
        B, C, H, W = fields.shape
        energy = torch.zeros(B, device=fields.device)
        
        # 1. Feature Extraction (Spatial Convolutions)
        # We want to convolve each channel with its filters, or mix them?
        # The parent implementation does: conv(u_i, filters_i).
        # Let's generalize. We want features that represent the local state.
        # Let's concatenate all channels and convolve with 'n_filters' 
        # (treating them as global features).
        # But self.spatial_filters is a ParameterList of (n_filters, 1, k, k).
        # Let's behave similarly to parent but feed gradients to KAN.
        
        # Actually, let's redefine features for the Hamiltonian case.
        # Useful invariants: u_i, Laplacian(u_i), GradMag(u_i).
        # Or just let Conv2d learn features from the multi-channel input.
        
        # To strictly use the self.spatial_filters from parent (which are per-channel):
        # We get (C * n_filters) features.
        # C=2, n_filters=8 => 16 features.
        
        feature_maps = []
        pad = self.spatial_filters[0].shape[-1] // 2
        
        for i in range(C):
            u_i = fields[:, i:i+1, :, :]
            # Convolve u_i with its bank of filters
            # filters: (n_filters, 1, k, k)
            f_i = F.conv2d(u_i, self.spatial_filters[i], padding=pad)
            feature_maps.append(f_i)
            
        # Concatenate all features: (B, C*n_filters, H, W)
        features = torch.cat(feature_maps, dim=1)
        
        # 2. Reshape for KAN: (B*H*W, InputDim)
        # We need to project features learned by filters to KAN input.
        # Wait, if C*n_filters is large, KAN input is large.
        # Let's stick to the architecture.
        # If n_channels=2, n_filters=8 => 16 inputs. KAN can handle that.
        
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        
        # 3. KAN Energy Density
        # Output: (B*H*W, 1)
        energy_density = self.kan(features_flat)
        
        # 4. Integrate
        energy_density = energy_density.view(B, H, W)
        total_energy = energy_density.sum(dim=(1, 2)) / (H * W)
        
        # Add a regularization term for the magnitude of field (prevent explosion)
        # Since KAN is unbounded.
        reg = 0.01 * (fields ** 2).mean(dim=(1, 2, 3))
        
        return total_energy + reg

    
    def evolve(
        self,
        image: torch.Tensor,
        n_steps: int = 50,
        return_trajectory: bool = False,
        advection_field: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Custom evolve supporting anchor_mask for Inpainting/Sudoku.
        """
        if image.dim() == 2:
            image = image.view(-1, 1, self.height, self.width)
        elif image.dim() == 3:
            image = image.unsqueeze(1)
        
        u = image.clone().detach()
        if anchor is None:
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
            update = -self.step_size.detach() * grad 
            
            # Data Fidelity (Anchoring)
            fidelity = self.data_weight.detach() * (anchor - u_input.detach())
            if anchor_mask is not None:
                div_fidelity = fidelity * anchor_mask
            else:
                div_fidelity = fidelity
                
            update = update + div_fidelity
            
            # Advection (Using Parent Logic if needed, omitted here for brevity if unneeded)
            # Or assume no advection for Hamiltonian Sudoku.
            # But let's copy parent logic if advection provided?
            if advection_field is not None:
                 # Minimal implementation
                 pass 

            u = u_input.detach() + update
            # u = torch.clamp(u, -1.5, 1.5) # Removed clamp for flexibility? No, keep it.
            
            if return_trajectory:
                trajectory.append(u.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        
        return u.detach()

    def trainable_evolve(
        self,
        fields: torch.Tensor,
        n_steps: int = 10,
        anchor: torch.Tensor = None,
        advection_field: torch.Tensor = None,
        anchor_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Trainable evolve with mask support.
        """
        if anchor is None:
            anchor = fields.detach()
        
        u = fields
        if not u.requires_grad:
            u.requires_grad_(True)
        
        for step in range(n_steps):
            energy = self.compute_energy(u)
            
            grad = torch.autograd.grad(
                energy.sum(), u, create_graph=True
            )[0]
            
            update = -self.step_size * grad
            
            fidelity = self.data_weight * (anchor - u)
            if anchor_mask is not None:
                fidelity = fidelity * anchor_mask
            
            update = update + fidelity
                
            u = u + update
        
        return u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_energy_from_image(x)
