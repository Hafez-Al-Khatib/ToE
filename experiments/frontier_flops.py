"""Per-descent-step FLOP measurement for energy heads.

One descent step = one energy forward + one backward pass to get the input
gradient. Measured with torch.utils.flop_counter.FlopCounterMode, which
counts matmul/conv FLOPs in forward AND backward. Elementwise ops (tanh,
spline basis evaluation, piecewise-linear lookup) are NOT counted; matmul
and conv dominate every head in this study. This counting convention is
stated verbatim in the paper section (spec section 6, FLOP accounting risk).
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def flops_per_step(model, device, img_shape=(3, 32, 32)):
    """FLOPs for one energy forward + input-gradient backward on ONE image."""
    model = model.to(device)
    x = torch.randn(1, *img_shape, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        xi = x.detach().requires_grad_(True)
        E = model.energy(xi)
        E = E.sum() if E.ndim > 0 else E
        torch.autograd.grad(E, xi)
    return int(counter.get_total_flops())
