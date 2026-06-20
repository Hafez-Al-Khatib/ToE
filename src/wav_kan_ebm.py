"""
wav_kan_ebm.py
==============
Wavelet-Inspired KAN-EBM variants.

Four combinations:
    front_free + head_bspline  -- original KAN-EBM
    front_free + head_wavkan   -- Wav-KAN head only
    front_gabor + head_bspline -- parametric Gabor front
    front_gabor + head_wavkan  -- fully wavelet (front + head)

Energy / denoise / DSM training interface matches KANEnergyModel.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

# Reuse the original B-spline KAN definition from exp_cifar10
from exp_cifar10 import KAN as BSplineKAN  # noqa: E402

from wav_kan import WavKAN  # noqa: E402
from gabor_front import GaborFilterBank  # noqa: E402


class WavKANEnergyModel(nn.Module):
    """Configurable variant of KANEnergyModel."""

    def __init__(self,
                 n_filters=16, filter_size=5, n_channels=3,
                 kan_hidden=None,
                 front='free',        # 'free' | 'gabor'
                 head='bspline',      # 'bspline' | 'wavkan'
                 n_atoms=8,
                 omega0=5.0,
                 learnable_omega=True):
        super().__init__()
        if kan_hidden is None:
            kan_hidden = [48, 16]
        self.front_kind = front
        self.head_kind = head
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_channels = n_channels

        # ── Front end ──────────────────────────────────────────────
        if front == 'free':
            self.filters = nn.Parameter(
                0.01 * torch.randn(n_filters, n_channels, filter_size, filter_size))
            self.gabor = None
        elif front == 'gabor':
            self.gabor = GaborFilterBank(n_filters=n_filters, n_channels=n_channels,
                                         filter_size=filter_size)
            self.filters = None
        else:
            raise ValueError(f'unknown front: {front!r}')

        self.log_precision = nn.Parameter(torch.zeros(n_filters))

        # ── Head ───────────────────────────────────────────────────
        kan_in = n_channels * n_filters
        layers = [kan_in] + list(kan_hidden) + [1]
        if head == 'bspline':
            self.kan = BSplineKAN(layers, grid_size=5)
        elif head == 'wavkan':
            self.kan = WavKAN(layers, n_atoms=n_atoms,
                              omega0=omega0, learnable_omega=learnable_omega)
        else:
            raise ValueError(f'unknown head: {head!r}')

    @property
    def precision(self):
        return F.softplus(self.log_precision)

    def extract_features(self, x):
        B, C, H, W = x.shape
        pad = self.filter_size // 2
        feats = []
        if self.front_kind == 'free':
            for c in range(C):
                feats.append(F.conv2d(x[:, c:c + 1], self.filters[:, c:c + 1], padding=pad))
        else:
            for c in range(C):
                feats.append(self.gabor(x[:, c:c + 1], channel_idx=c))
        feats = torch.cat(feats, dim=1)
        prec = self.precision.repeat(C).view(1, C * self.n_filters, 1, 1)
        feats = torch.tanh(feats * prec)
        return feats.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

    def energy(self, x):
        return self.kan(self.extract_features(x)).sum()

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
