"""Per-descent-step FLOP measurement for energy heads.

One descent step = one energy forward + one backward pass to get the input
gradient. Measured with torch.utils.flop_counter.FlopCounterMode, which
counts matmul/conv FLOPs in forward AND backward. Elementwise ops (tanh,
spline basis evaluation, piecewise-linear lookup) are NOT counted; matmul
and conv dominate every head in this study. This counting convention is
stated verbatim in the paper section (spec section 6, FLOP accounting risk).

Known torch 2.5.1 bug (see .superpowers/sdd/task-3-report.md): wrapping
certain backward graphs (e.g. UNetEBM's MaxPool2d -> ConvTranspose2d path)
in FlopCounterMode while calling torch.autograd.grad raises
"RuntimeError: A leaf node was passed to _will_engine_execute_node but we
are currently running autograd.grad(). This is currently not supported."
Plain torch.autograd.grad works fine on the same model outside
FlopCounterMode; only the FLOP *measurement* is affected. Task 1 validated
that for input-gradient descent, total FLOPs = forward + grad_input-backward
approx= 2x forward (grad_weight is never formed, since autograd.grad only
requests the input gradient). flops_per_step_detail() uses this fact as a
fallback: when the full forward+backward measurement raises, it measures
forward-only FLOPs and reports total = 2 * forward instead.
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def _forward_only_flops(model, x):
    """FLOPs for a single energy(x) forward pass, no grad tracking."""
    counter = FlopCounterMode(display=False)
    with counter:
        with torch.no_grad():
            model.energy(x)
    return int(counter.get_total_flops())


def flops_per_step_detail(model, device, img_shape=(3, 32, 32)):
    """FLOPs for one energy forward + input-gradient backward on ONE image.

    Returns a dict:
        {'total': int, 'forward': int, 'method': 'measured' | 'forward_x2'}

    Tries the full forward+backward measurement inside a single
    FlopCounterMode block first ('measured'). If that raises RuntimeError
    (torch 2.5.1 FlopCounterMode/autograd.grad bug on some backward graphs,
    see module docstring), falls back to measuring forward-only FLOPs and
    reporting total = 2 * forward ('forward_x2'), per the validated
    forward-backward-input-gradient FLOP identity.
    """
    model = model.to(device)
    x = torch.randn(1, *img_shape, device=device)

    forward_flops = _forward_only_flops(model, x)

    try:
        counter = FlopCounterMode(display=False)
        with counter:
            xi = x.detach().requires_grad_(True)
            E = model.energy(xi)
            E = E.sum() if E.ndim > 0 else E
            torch.autograd.grad(E, xi)
        total = int(counter.get_total_flops())
        return {'total': total, 'forward': forward_flops, 'method': 'measured'}
    except RuntimeError:
        total = 2 * forward_flops
        return {'total': total, 'forward': forward_flops, 'method': 'forward_x2'}


def flops_per_step(model, device, img_shape=(3, 32, 32)):
    """FLOPs for one energy forward + input-gradient backward on ONE image.

    Thin wrapper around flops_per_step_detail(...)['total'] kept for
    backward compatibility with existing call sites.
    """
    return flops_per_step_detail(model, device, img_shape)['total']
