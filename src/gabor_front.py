"""
gabor_front.py
==============
Parametric Gabor filter bank as a drop-in for the learned 5x5 conv filters
in KAN-EBM.

Why: the spectral analysis (exp_spectral_conv_filters.py) found that the
trained KAN-EBM filter bank spontaneously discovers Gabor-like / Mexican-hat
structure with mean cosine similarity 0.70 to parametric Gabors. This module
makes the inductive bias explicit: each filter is parameterized as a Gabor
with learnable (sigma, frequency, theta, phase). Per-channel mixing is
preserved so the (F, C, k, k) layout matches the existing model.

Parameters per filter (per channel): 4 (sigma, freq, theta, phase) + 1
amplitude = 5 params. Vs free 5x5 = 25 params. -> 5x parameter compression
at the front end.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborFilterBank(nn.Module):
    """A bank of (F, C) Gabor filters of size kxk."""

    def __init__(self, n_filters=32, n_channels=3, filter_size=5,
                 init_freq=2.0, init_sigma=0.5):
        super().__init__()
        F_, C, k = n_filters, n_channels, filter_size
        self.n_filters = F_
        self.n_channels = C
        self.filter_size = k

        # Initialize a span of orientations and frequencies for diversity
        thetas = torch.linspace(0, math.pi, F_ + 1)[:-1]  # 0 .. pi
        thetas = thetas.view(F_, 1).expand(F_, C).clone()
        thetas += 0.05 * math.pi * torch.randn_like(thetas)

        freqs = torch.full((F_, C), float(init_freq))
        freqs += 0.5 * torch.randn_like(freqs)

        sigmas_pre = torch.log(torch.expm1(torch.full((F_, C), float(init_sigma))))
        amps = 0.1 * torch.randn(F_, C)
        phases = torch.zeros(F_, C)  # learnable

        self.theta = nn.Parameter(thetas)
        self.freq = nn.Parameter(freqs)
        self.sigma_pre = nn.Parameter(sigmas_pre)
        self.phase = nn.Parameter(phases)
        self.amp = nn.Parameter(amps)

        # Coordinate grid
        half = (k - 1) / 2
        xs = (torch.arange(k, dtype=torch.float32) - half) / max(half, 1.0)
        X, Y = torch.meshgrid(xs, xs, indexing='xy')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

    @property
    def sigma(self):
        return F.softplus(self.sigma_pre) + 1e-3

    def kernel(self):
        """Build (F, C, k, k) conv kernel from Gabor parameters."""
        F_, C, k = self.n_filters, self.n_channels, self.filter_size
        theta = self.theta.view(F_, C, 1, 1)
        freq = self.freq.view(F_, C, 1, 1)
        sigma = self.sigma.view(F_, C, 1, 1)
        phase = self.phase.view(F_, C, 1, 1)
        amp = self.amp.view(F_, C, 1, 1)
        X = self.X.view(1, 1, k, k)
        Y = self.Y.view(1, 1, k, k)
        Xr = X * torch.cos(theta) + Y * torch.sin(theta)
        Yr = -X * torch.sin(theta) + Y * torch.cos(theta)
        env = torch.exp(-(Xr * Xr + Yr * Yr) / (2 * sigma * sigma))
        carrier = torch.cos(2 * math.pi * freq * Xr + phase)
        kern = amp * env * carrier
        # Zero-mean each kernel (wavelet condition)
        kern = kern - kern.mean(dim=(-1, -2), keepdim=True)
        return kern

    def forward(self, x_channel, channel_idx):
        """Apply this bank's channel-idx-th column to a single-channel input.
        x_channel: (B, 1, H, W).  Returns (B, F, H, W).
        """
        kern = self.kernel()  # (F, C, k, k)
        kern_c = kern[:, channel_idx:channel_idx + 1]  # (F, 1, k, k)
        pad = self.filter_size // 2
        return F.conv2d(x_channel, kern_c, padding=pad)
