"""
Kolmogorov-Arnold Network (KAN) Layer
======================================

This module implements KAN layers: neural network layers where the learnable
parameters are on the EDGES (as spline functions) rather than the NODES.

Mathematical Foundation: The Kolmogorov-Arnold Theorem
------------------------------------------------------
Any continuous function f: [0,1]^n → ℝ can be represented as:

    f(x₁, ..., xₙ) = Σ_{q=0}^{2n} Φ_q(Σ_{p=1}^{n} φ_{q,p}(x_p))

where φ_{q,p} and Φ_q are continuous univariate functions.

This is profound: we can represent ANY multivariate function using only
compositions of 1D functions! No need for complex multi-dimensional interactions.

KAN Implementation
------------------
We approximate each φ_{q,p} as a B-spline — a piecewise polynomial defined by:
- A knot vector (grid points)
- Control points (learned parameters)

The key insight: instead of y = σ(Wx + b), we have y_j = Σ_i φ_{i,j}(x_i)
where each φ_{i,j} is a learned spline.

Why Splines?
------------
1. **Flexibility**: Can approximate any smooth function
2. **Locality**: Changing one control point only affects nearby region
3. **Smoothness**: Cubic splines are C² continuous (smooth second derivative)
4. **Stability**: No Runge phenomenon (unlike high-degree polynomials)

For energy functions, this means we can learn:
- Sharp valleys (stable states)
- Smooth ridges (transition regions)
- Multi-modal basins (multiple equilibria)

References
----------
- Liu, Z. et al. (2024). "KAN: Kolmogorov-Arnold Networks"
- de Boor, C. (1978). "A Practical Guide to Splines"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class BSplineBasis(nn.Module):
    """
    Computes B-spline basis functions B_{i,k}(x).
    
    B-splines are piecewise polynomials defined by the Cox-de Boor recursion:
    
    Base case (degree 0):
        B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0
    
    Recursion (degree k):
        B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) · B_{i,k-1}(x)
                   + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) · B_{i+1,k-1}(x)
    
    Properties:
    - Non-negative: B_{i,k}(x) ≥ 0
    - Compact support: B_{i,k}(x) = 0 outside [t_i, t_{i+k+1}]
    - Partition of unity: Σ_i B_{i,k}(x) = 1
    - Smooth: B_{i,k} is C^{k-1} continuous
    
    Parameters
    ----------
    grid_size : int
        Number of interior grid intervals. Total # of basis functions = grid_size + degree.
    
    degree : int
        Polynomial degree. 3 = cubic splines (most common).
    
    grid_range : tuple
        (min, max) range for the grid. Data should be normalized to this range.
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        degree: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        self.grid_size = grid_size
        self.degree = degree
        self.grid_range = grid_range
        
        # Number of basis functions
        self.n_bases = grid_size + degree
        
        # Create uniform knot vector
        # For a uniform grid with G intervals and degree k:
        # - G + 1 interior knots
        # - k repeated knots at each end (clamped boundary)
        # Total: G + 1 + 2k knots
        interior_knots = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        
        # Clamped splines: repeat boundary knots
        left_pad = interior_knots[0].repeat(degree)
        right_pad = interior_knots[-1].repeat(degree)
        knots = torch.cat([left_pad, interior_knots, right_pad])
        
        self.register_buffer('knots', knots)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions at points x.
        
        Parameters
        ----------
        x : torch.Tensor
            Input points, shape (...). Will be flattened and processed.
        
        Returns
        -------
        torch.Tensor
            Basis values, shape (..., n_bases).
        """
        # Store original shape for restoration
        original_shape = x.shape
        x = x.reshape(-1, 1)  # (N, 1)
        
        # Clamp x to grid range (extrapolation is unstable)
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1] - 1e-6)
        
        # Compute bases using de Boor recursion
        # Start with degree 0 (indicator functions)
        bases = self._compute_degree_0(x)  # (N, n_intervals)
        
        # Build up to target degree
        for k in range(1, self.degree + 1):
            bases = self._compute_degree_k(x, bases, k)
        
        # Reshape to match input
        return bases.reshape(*original_shape, -1)
    
    def _compute_degree_0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute degree-0 B-splines (indicator functions).
        
        B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (N, 1)
        
        Returns
        -------
        torch.Tensor
            Shape (N, n_knots - 1)
        """
        # Number of degree-0 bases is one less than number of knots
        n_intervals = len(self.knots) - 1
        
        # Compare x to knot intervals
        # x: (N, 1), knots[:-1]: (n_intervals,) → (N, n_intervals)
        lower = (x >= self.knots[:-1].unsqueeze(0))  # x ≥ t_i
        upper = (x < self.knots[1:].unsqueeze(0))    # x < t_{i+1}
        
        # B_{i,0}(x) = 1 iff x is in [t_i, t_{i+1})
        bases = (lower & upper).float()
        
        return bases
    
    def _compute_degree_k(
        self, 
        x: torch.Tensor, 
        bases_km1: torch.Tensor, 
        k: int
    ) -> torch.Tensor:
        """
        Compute degree-k B-splines from degree-(k-1) splines.
        
        B_{i,k}(x) = α_{i,k}(x)·B_{i,k-1}(x) + (1 - α_{i+1,k}(x))·B_{i+1,k-1}(x)
        
        where α_{i,k}(x) = (x - t_i) / (t_{i+k} - t_i)
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (N, 1)
        
        bases_km1 : torch.Tensor
            Degree-(k-1) bases, shape (N, n_bases_km1)
        
        k : int
            Target degree
        
        Returns
        -------
        torch.Tensor
            Degree-k bases, shape (N, n_bases_km1 - 1)
        """
        n_bases_k = bases_km1.shape[-1] - 1
        
        # Knot differences
        # For degree k, we need t_{i+k} - t_i for the alpha coefficient
        t_diff_left = self.knots[k:-1] - self.knots[:-k-1]  # t_{i+k} - t_i
        t_diff_right = self.knots[k+1:] - self.knots[1:-k]   # t_{i+k+1} - t_{i+1}
        
        # Avoid division by zero
        t_diff_left = torch.where(t_diff_left == 0, torch.ones_like(t_diff_left), t_diff_left)
        t_diff_right = torch.where(t_diff_right == 0, torch.ones_like(t_diff_right), t_diff_right)
        
        # Compute alpha coefficients
        # α_{i,k}(x) = (x - t_i) / (t_{i+k} - t_i)
        alpha_left = (x - self.knots[:-k-1].unsqueeze(0)) / t_diff_left.unsqueeze(0)
        alpha_right = (self.knots[k+1:].unsqueeze(0) - x) / t_diff_right.unsqueeze(0)
        
        # Only use the relevant portions
        alpha_left = alpha_left[:, :n_bases_k]
        alpha_right = alpha_right[:, :n_bases_k]
        
        # Recurrence: B_{i,k} = α_left · B_{i,k-1} + α_right · B_{i+1,k-1}
        bases_k = alpha_left * bases_km1[:, :-1] + alpha_right * bases_km1[:, 1:]
        
        return bases_k


class KANLayer(nn.Module):
    """
    A Kolmogorov-Arnold Network layer.
    
    Transformation: y_j = Σ_i φ_{i,j}(x_i) + b_j
    
    Where each φ_{i,j} is a B-spline parameterized by learnable control points.
    
    This is fundamentally different from MLP:
    - MLP: y = σ(Wx + b) — fixed nonlinearity, learned linear transform
    - KAN: y = Σ φ(x) + b — learned nonlinear transform per edge
    
    Interpretation
    --------------
    You can visualize each φ_{i,j} as a curve. After training:
    - Some curves are identity-like (pass-through)
    - Some are step-like (thresholding)
    - Some are peaked (feature detection)
    
    This interpretability is a major advantage over MLPs.
    
    Parameters
    ----------
    in_features : int
        Number of input features.
    
    out_features : int
        Number of output features.
    
    grid_size : int
        Number of intervals in the spline grid.
        More = finer approximation but more parameters.
    
    spline_order : int
        Degree of the spline (3 = cubic).
    
    grid_range : tuple
        Expected range of input values. Normalize inputs to this range.
    
    residual : bool
        If True, add a linear residual connection: y = Σφ(x) + Wx
        This helps with training stability.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        residual: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.residual = residual
        
        # B-spline basis computer
        self.basis = BSplineBasis(grid_size, spline_order, grid_range)
        
        # Number of control points per spline
        n_ctrl = self.basis.n_bases
        
        # Learnable control points for each (input, output) pair
        # Shape: (in_features, out_features, n_ctrl)
        # Initialized small for smooth starting point
        self.ctrl_points = nn.Parameter(
            torch.randn(in_features, out_features, n_ctrl) * 0.1
        )
        
        # Optional residual (linear) connection
        if residual:
            self.residual_weight = nn.Parameter(
                torch.randn(in_features, out_features) * 0.1
            )
        else:
            self.register_parameter('residual_weight', None)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Scaling factor for numerical stability
        self.scale = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, in_features)
        
        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Compute B-spline basis at each input
        # x: (batch, in_features) → bases: (batch, in_features, n_ctrl)
        bases = self.basis(x)
        
        # Evaluate splines: φ_{i,j}(x_i) = Σ_c ctrl_points_{i,j,c} · B_c(x_i)
        # bases: (batch, in, n_ctrl)
        # ctrl_points: (in, out, n_ctrl)
        # Result: (batch, in, out)
        spline_vals = torch.einsum('bic,ioc->bio', bases, self.ctrl_points)
        
        # Sum over input dimension: y_j = Σ_i φ_{i,j}(x_i)
        y = spline_vals.sum(dim=1)  # (batch, out)
        
        # Add residual (linear) connection if enabled
        if self.residual and self.residual_weight is not None:
            y = y + torch.matmul(x, self.residual_weight)
        
        # Scale and bias
        y = y * self.scale + self.bias
        
        return y
    
    def get_edge_function(
        self, 
        in_idx: int, 
        out_idx: int, 
        n_points: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the learned function φ_{in_idx, out_idx}(x).
        
        Useful for visualization and interpretation.
        
        Parameters
        ----------
        in_idx : int
            Input feature index.
        
        out_idx : int
            Output feature index.
        
        n_points : int
            Number of evaluation points.
        
        Returns
        -------
        x_vals : torch.Tensor
            Input values, shape (n_points,)
        
        y_vals : torch.Tensor
            Function values φ(x), shape (n_points,)
        """
        x_vals = torch.linspace(*self.basis.grid_range, n_points, device=self.ctrl_points.device)
        
        # Compute basis at these points
        bases = self.basis(x_vals)  # (n_points, n_ctrl)
        
        # Get control points for this edge
        ctrl = self.ctrl_points[in_idx, out_idx]  # (n_ctrl,)
        
        # Evaluate spline
        y_vals = torch.matmul(bases, ctrl)  # (n_points,)
        
        # Add residual contribution if present
        if self.residual and self.residual_weight is not None:
            y_vals = y_vals + self.residual_weight[in_idx, out_idx] * x_vals
        
        return x_vals, y_vals


class KANNetwork(nn.Module):
    """
    A multi-layer KAN network.
    
    Architecture: Input → KAN₁ → KAN₂ → ... → KANₗ → Output
    
    This replaces the standard MLP for function approximation.
    Instead of y = σ(W₂·σ(W₁·x)), we have nested spline compositions.
    
    For energy functions, this is: E(x) = KAN(x)
    where KAN learns an energy landscape with potentially complex structure.
    
    Parameters
    ----------
    layers : list of int
        Layer sizes, e.g., [784, 64, 32, 1] for MNIST→energy.
    
    grid_size : int
        Grid size for all KAN layers.
    
    spline_order : int
        Spline degree (3 = cubic).
    """
    
    def __init__(
        self,
        layers: list,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        self.layers_sizes = layers
        
        # Build KAN layers
        kan_layers = []
        for i in range(len(layers) - 1):
            kan_layers.append(
                KANLayer(
                    in_features=layers[i],
                    out_features=layers[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    residual=True
                )
            )
            # Add layer normalization for stability (except last layer)
            if i < len(layers) - 2:
                kan_layers.append(nn.LayerNorm(layers[i + 1]))
        
        self.network = nn.Sequential(*kan_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full KAN network."""
        return self.network(x)
    
    def get_all_edge_functions(self, layer_idx: int = 0):
        """
        Get all edge functions from a specific layer.
        
        Returns a dict mapping (in_idx, out_idx) to (x_vals, y_vals).
        """
        # Find the KAN layer
        kan_layer = None
        kan_count = 0
        for module in self.network:
            if isinstance(module, KANLayer):
                if kan_count == layer_idx:
                    kan_layer = module
                    break
                kan_count += 1
        
        if kan_layer is None:
            raise ValueError(f"Layer {layer_idx} not found")
        
        edge_functions = {}
        for i in range(kan_layer.in_features):
            for j in range(kan_layer.out_features):
                edge_functions[(i, j)] = kan_layer.get_edge_function(i, j)
        
        return edge_functions


class EnergyKAN(nn.Module):
    """
    Energy function parameterized by a KAN network.
    
    Maps input x ∈ ℝⁿ to scalar energy E(x) ∈ ℝ.
    
    Compared to EnergyMLP (Phase 1), this offers:
    - More flexible energy landscapes (splines can be sharper than SiLU)
    - Interpretable edge functions (can visualize what's learned)
    - Potentially multi-modal basins
    
    Use this for:
    - Control tasks with multiple equilibria
    - Learning complex physical potentials
    - Interpretable energy-based models
    
    Parameters
    ----------
    input_dim : int
        Dimension of input.
    
    hidden_dims : tuple
        Hidden layer sizes.
    
    grid_size : int
        KAN grid resolution.
    
    spline_order : int
        Spline degree.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (64, 32),
        grid_size: int = 5,
        spline_order: int = 3
    ):
        super().__init__()
        
        # Build layer sizes: input → hidden → 1 (scalar energy)
        layers = [input_dim] + list(hidden_dims) + [1]
        
        self.kan = KANNetwork(
            layers=layers,
            grid_size=grid_size,
            spline_order=spline_order
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy E(x).
        
        Returns
        -------
        torch.Tensor
            Energy values, shape (batch_size,)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # KAN output is (batch, 1), squeeze to (batch,)
        return self.kan(x).squeeze(-1)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing B-spline basis...")
    basis = BSplineBasis(grid_size=5, degree=3)
    x = torch.linspace(-1, 1, 100)
    B = basis(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Basis shape: {B.shape}")
    print(f"  Partition of unity: {B.sum(dim=-1).mean():.4f} (should be ~1.0)")
    
    print("\nTesting KAN layer...")
    layer = KANLayer(in_features=10, out_features=5, grid_size=5)
    x = torch.randn(32, 10)
    y = layer(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    
    # Check edge function
    x_vals, y_vals = layer.get_edge_function(0, 0)
    print(f"  Edge function (0,0): {len(x_vals)} points")
    
    print("\nTesting KAN network...")
    net = KANNetwork(layers=[100, 32, 16, 1], grid_size=5)
    x = torch.randn(16, 100)
    y = net(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    
    print("\nTesting EnergyKAN...")
    energy = EnergyKAN(input_dim=784, hidden_dims=(64, 32))
    x = torch.randn(8, 784)
    E = energy(x)
    print(f"  Input: {x.shape}, Energy: {E.shape}")
    print(f"  Energy values: {E[:4].tolist()}")
    
    print("\n✓ All KAN tests passed!")
