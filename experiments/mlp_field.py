"""
MLP-Based Field: Baseline for KAN-Hamiltonian Comparison
=========================================================

This implements an MLP-based energy field with the SAME architecture as
HamiltonianField, but replacing the KAN energy density with an MLP.

This is the critical baseline for Paper 1: we want to show that KAN learns
more interpretable and physically meaningful energy functions than MLP,
with comparable or better performance.

Architecture Match
------------------
Both models:
- Use the SAME spatial filter bank (learned 5x5 conv kernels)
- Use the SAME diffusion coefficients
- Output scalar energy density per spatial point
- Use gradient-based inference (Langevin dynamics)

Only difference:
- HamiltonianField: E(features) = KAN(features)  [B-spline basis]
- MLPField:         E(features) = MLP(features)   [ReLU/SiLU basis]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from thermodynamic_field import ThermodynamicField


class MLPEnergyDensity(nn.Module):
    """MLP that maps local features to scalar energy density."""

    def __init__(self, input_dim, hidden_dims=[32], activation='silu'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if activation == 'silu':
                    layers.append(nn.SiLU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPField(ThermodynamicField):
    """
    MLP-based energy field — direct comparison baseline for HamiltonianField.

    Same spatial filters + diffusion structure, but MLP instead of KAN
    for the energy density function.
    """

    def __init__(
        self,
        n_channels,
        height,
        width,
        n_filters=4,
        filter_size=5,
        mlp_hidden=[32],
        activation='silu'
    ):
        # Initialize ThermodynamicField base (we override compute_energy)
        super().__init__(
            n_channels=n_channels,
            height=height,
            width=width,
            n_filters=n_filters,
            filter_size=filter_size
        )

        # Replace KAN with MLP for energy density
        input_dim = n_channels * n_filters
        self.mlp_energy = MLPEnergyDensity(input_dim, mlp_hidden, activation)

        # Remove the KAN from parent class to avoid confusion
        if hasattr(self, 'potential_kan'):
            del self.potential_kan

        # Regularization weight
        self.reg_weight = nn.Parameter(torch.tensor(0.001))

    def compute_energy(self, fields):
        """
        Compute energy using MLP density (mirrors HamiltonianField.compute_energy).

        Parameters
        ----------
        fields : (B, C, H, W) tensor

        Returns
        -------
        energy : (B,) tensor, scalar energy per sample
        """
        B, C, H, W = fields.shape

        # spatial_filters is a ParameterList of length n_channels.
        # Each element has shape (n_filters, 1, filter_size, filter_size).
        # We convolve each channel with its filter bank — exactly matching
        # the parent ThermodynamicField and HamiltonianField approach.
        pad = self.spatial_filters[0].shape[-1] // 2
        all_features = []
        for c in range(C):
            u_c = fields[:, c:c+1, :, :]          # (B, 1, H, W)
            f_c = F.conv2d(u_c, self.spatial_filters[c], padding=pad)
            # f_c: (B, n_filters, H, W)
            all_features.append(f_c)

        # Stack: (B, C*n_filters, H, W)
        features = torch.cat(all_features, dim=1)

        # 2. Reshape for MLP: (B*H*W, C*n_filters)
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # 3. MLP energy density
        energy_density = self.mlp_energy(features_flat)  # (B*H*W, 1)
        energy_density = energy_density.reshape(B, H, W)

        # 4. Integrate over spatial domain
        energy = energy_density.sum(dim=(1, 2))  # (B,)

        # 5. L2 regularization on fields
        reg = self.reg_weight.abs() * (fields ** 2).sum(dim=(1, 2, 3))

        return energy + reg

    def evolve(self, image, n_steps=50, return_trajectory=False,
               advection_field=None, anchor=None, anchor_mask=None):
        """Energy minimization via gradient descent."""
        u = image.clone().requires_grad_(True)
        trajectory = [u.detach().clone()] if return_trajectory else None

        for step in range(n_steps):
            energy = self.compute_energy(u)
            grad = torch.autograd.grad(energy.sum(), u, create_graph=False)[0]

            with torch.no_grad():
                # Gradient step
                u_new = u - self.step_size * grad

                # Data fidelity anchoring
                if anchor is not None:
                    if anchor_mask is not None:
                        u_new = torch.where(anchor_mask > 0.5, anchor, u_new)
                    else:
                        u_new = u_new + self.data_weight * (anchor - u_new)

                u = u_new.requires_grad_(True)

            if return_trajectory:
                trajectory.append(u.detach().clone())

        if return_trajectory:
            return u.detach(), trajectory
        return u.detach()

    def trainable_evolve(self, fields, n_steps=5, anchor=None,
                         advection_field=None, anchor_mask=None):
        """Differentiable evolution for training."""
        u = fields

        for step in range(n_steps):
            energy = self.compute_energy(u)
            grad = torch.autograd.grad(energy.sum(), u, create_graph=True)[0]

            u = u - self.step_size * grad

            if anchor is not None:
                if anchor_mask is not None:
                    u = torch.where(anchor_mask > 0.5, anchor, u)
                else:
                    u = u + self.data_weight * (anchor - u)

        return u


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing MLPField baseline...")

    model = MLPField(
        n_channels=1, height=28, width=28,
        n_filters=16, mlp_hidden=[32]
    )

    x = torch.randn(4, 1, 28, 28)
    E = model.compute_energy(x)
    print(f"  Energy shape: {E.shape}")
    print(f"  Parameters: {count_parameters(model):,}")

    # Compare with HamiltonianField
    from hamiltonian_field import HamiltonianField
    kan_model = HamiltonianField(
        n_channels=1, height=28, width=28,
        n_filters=16, kan_hidden=[32, 1], kan_grid_size=5
    )
    print(f"  KAN model parameters: {count_parameters(kan_model):,}")
    print("  MLPField test passed!")
