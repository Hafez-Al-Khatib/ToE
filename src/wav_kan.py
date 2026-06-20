"""
wav_kan.py
==========
Wav-KAN: KAN with Morlet-wavelet edges.

Each edge function is a finite sum of K Morlet atoms:
    phi_ij(x) = base_w[j,i] * SiLU(x)
              + sum_k a[j,i,k] * exp(-((x - b[j,i,k]) / s[j,i,k])^2 / 2)
                              * cos(omega0 * (x - b[j,i,k]) / s[j,i,k])

Parameters per edge:  a, b (shift), s (scale) for K atoms  -> 3K per edge.
With grid_size G in stock KAN, B-spline edges have G+order params each;
choosing K=G makes parameter count comparable (3K vs G+order=G+3 here).

Drop-in compatible with the KANEnergyModel: replace KAN(...) with WavKAN(...)
and the rest of the pipeline (energy, denoise, DSM training) is unchanged.

Notes on initialization:
- b sampled in grid_range
- s sampled around (grid_range_width / K) to tile the range
- a small Kaiming-uniform
- A learnable carrier frequency omega0 is shared across edges by default
  (a per-edge omega is supported via per_edge_omega=True but inflates params)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WavKANLinear(nn.Module):
    """Wav-KAN equivalent of KANLinear. Same in/out interface."""

    def __init__(self, in_features, out_features,
                 n_atoms=8, grid_range=(-1.0, 1.0),
                 omega0=5.0, learnable_omega=True, per_edge_omega=False,
                 scale_init=None, scale_base=1.0, scale_atom=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_atoms = n_atoms
        self.grid_low, self.grid_high = grid_range
        width = float(self.grid_high - self.grid_low)
        if scale_init is None:
            scale_init = width / max(n_atoms, 1)

        # SiLU residual branch (same as B-spline KAN: separates linear-ish base)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        self.base_activation = nn.SiLU()

        # Atom params (out, in, K)
        amp = torch.empty(out_features, in_features, n_atoms)
        nn.init.kaiming_uniform_(amp, a=math.sqrt(5) * scale_atom)
        self.amp = nn.Parameter(amp)

        # Initial shifts tile grid_range; jitter to break symmetry
        b_init = (torch.arange(n_atoms, dtype=torch.float32) + 0.5) / n_atoms
        b_init = self.grid_low + b_init * width  # (K,)
        b_init = b_init.view(1, 1, n_atoms).expand(out_features, in_features, n_atoms).clone()
        b_init += 0.05 * width * torch.randn_like(b_init)
        self.shift = nn.Parameter(b_init)

        # Initial scales near tiling width; positive via softplus
        s_pre = torch.log(torch.expm1(torch.full((out_features, in_features, n_atoms),
                                                 scale_init)))
        self.scale_pre = nn.Parameter(s_pre)

        # Carrier frequency
        if per_edge_omega:
            o = torch.full((out_features, in_features), omega0)
            self.omega = nn.Parameter(o) if learnable_omega else self.register_buffer('omega_buf', o)
        else:
            o = torch.tensor(float(omega0))
            if learnable_omega:
                self.omega = nn.Parameter(o)
            else:
                self.register_buffer('omega_buf', o)
        self.learnable_omega = learnable_omega
        self.per_edge_omega = per_edge_omega

    def _omega(self):
        if self.learnable_omega:
            return self.omega
        return self.omega_buf

    @property
    def scale(self):
        return F.softplus(self.scale_pre) + 1e-3

    def edge_response(self, x):
        """Evaluate every edge function on a 1-D x tensor of shape (B, in_features)
        and return per-edge values of shape (B, out_features, in_features).
        """
        # x:(B, in) -> (B, 1, in, 1) ;  shift/scale: (out, in, K) -> (1, out, in, K)
        xb = x.unsqueeze(1).unsqueeze(-1)
        b = self.shift.unsqueeze(0)
        s = self.scale.unsqueeze(0)
        t = (xb - b) / s
        env = torch.exp(-0.5 * t * t)
        om = self._omega()
        if self.per_edge_omega:
            om = om.unsqueeze(0).unsqueeze(-1)
        carrier = torch.cos(om * t)
        atoms = env * carrier  # (B, out, in, K)
        a = self.amp.unsqueeze(0)  # (1, out, in, K)
        edge_vals = (a * atoms).sum(dim=-1)  # (B, out, in)
        return edge_vals

    def forward(self, x):
        # SiLU base path  (B, in) -> (B, out)
        base_out = F.linear(self.base_activation(x), self.base_weight)
        # Wavelet edge path: sum_i phi_ij(x_i)
        edge_vals = self.edge_response(x)  # (B, out, in)
        wav_out = edge_vals.sum(dim=-1)    # (B, out)
        return base_out + wav_out


class WavKAN(nn.Module):
    """Sequence of WavKANLinear layers, drop-in for KAN."""

    def __init__(self, layers_hidden, n_atoms=8, grid_range=(-1.0, 1.0),
                 omega0=5.0, learnable_omega=True, per_edge_omega=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for a, b in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(WavKANLinear(
                a, b, n_atoms=n_atoms, grid_range=grid_range,
                omega0=omega0, learnable_omega=learnable_omega,
                per_edge_omega=per_edge_omega))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
