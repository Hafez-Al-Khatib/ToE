"""GR-KAN-style grouped-KAN energy head.

Grouping idea from the Kolmogorov-Arnold Transformer (Yang & Wang, ICLR
2025): input channels are split into G groups; each group shares ONE
learnable univariate function, applied elementwise, followed by a standard
nn.Linear mix (matmul-friendly). We claim only the APPLICATION as an EBM
energy head, not the mechanism. The shared univariate function here is a
learnable piecewise-linear spline on a fixed uniform grid (initialized to
the identity), the cheapest GPU-parallel choice.

Backbone (filters + tanh(precision*conv) features) is copied verbatim from
ConvSmoothMLPEBM / KANEnergyModel so the training recipe and descent
protocol transfer unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPWL(nn.Module):
    """G learnable piecewise-linear univariate functions on a fixed grid."""

    def __init__(self, n_groups, n_knots=17, x_min=-3.0, x_max=3.0):
        super().__init__()
        self.n_groups, self.n_knots = n_groups, n_knots
        self.x_min, self.x_max = float(x_min), float(x_max)
        grid = torch.linspace(x_min, x_max, n_knots)
        # identity init: values equal grid positions -> phi(x) == clamp(x)
        self.values = nn.Parameter(grid.unsqueeze(0).repeat(n_groups, 1))

    def forward(self, x, group_idx):
        """x: (N, F); group_idx: (F,) long mapping feature -> group."""
        t = (x.clamp(self.x_min, self.x_max) - self.x_min) \
            / (self.x_max - self.x_min) * (self.n_knots - 1)
        i0 = t.floor().long().clamp(0, self.n_knots - 2)
        frac = t - i0.to(t.dtype)
        flat = self.values.view(-1)                       # (G * n_knots,)
        base = (group_idx * self.n_knots).unsqueeze(0)    # (1, F)
        v0 = flat[base + i0]
        v1 = flat[base + i0 + 1]
        return v0 + frac * (v1 - v0)


class GroupKANLayer(nn.Module):
    def __init__(self, in_features, out_features, n_groups=8, n_knots=17):
        super().__init__()
        assert in_features % n_groups == 0, \
            f"{in_features} not divisible by {n_groups} groups"
        self.phi = SharedPWL(n_groups, n_knots)
        gsize = in_features // n_groups
        self.register_buffer(
            'group_idx', torch.arange(n_groups).repeat_interleave(gsize))
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(self.phi(x, self.group_idx))


class GroupKANEnergyModel(nn.Module):
    """Same conv backbone as ConvSmoothMLPEBM; head = 2 GroupKAN layers."""

    def __init__(self, n_filters=16, filter_size=5, hidden=640, n_groups=8,
                 n_knots=17, n_channels=3):
        super().__init__()
        self.n_filters, self.filter_size = n_filters, filter_size
        self.n_channels = n_channels
        self.filters = nn.Parameter(
            0.01 * torch.randn(n_filters, n_channels, filter_size,
                               filter_size))
        self.log_precision = nn.Parameter(torch.zeros(n_filters))
        in_dim = n_channels * n_filters
        assert hidden % n_groups == 0, "hidden must be divisible by n_groups"
        self.head = nn.Sequential(
            GroupKANLayer(in_dim, hidden, n_groups, n_knots),
            GroupKANLayer(hidden, 1, n_groups, n_knots),
        )

    @property
    def precision(self):
        return F.softplus(self.log_precision)

    def extract_features(self, x):
        B, C, H, W = x.shape
        pad = self.filter_size // 2
        feats = []
        for c in range(C):
            feats.append(F.conv2d(x[:, c:c + 1], self.filters[:, c:c + 1],
                                  padding=pad))
        feats = torch.cat(feats, dim=1)
        prec = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = torch.tanh(feats * prec)
        return feats.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

    def energy(self, x):
        return self.head(self.extract_features(x)).sum()

    def energy_grad(self, x):
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.energy(xi), xi)[0].detach()

    def denoise(self, x_noisy, n_steps=10, dt=0.05, dt_decay=0.97):
        u = x_noisy.clone()
        step = dt
        for _ in range(n_steps):
            u = (u - step * self.energy_grad(u).clamp(-1., 1.)).detach()
            step *= dt_decay
        return u

    def loss(self, x_clean, sigma):
        xn = x_clean + torch.randn_like(x_clean) * sigma
        with torch.enable_grad():
            xi = xn.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.energy(xi), xi,
                                       create_graph=True)[0]
        return F.mse_loss(xi - sigma ** 2 * grad, x_clean)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
