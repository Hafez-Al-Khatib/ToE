import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from group_kan import GroupKANEnergyModel, SharedPWL


def test_param_counts_near_targets():
    m32 = GroupKANEnergyModel(hidden=640)
    m8 = GroupKANEnergyModel(hidden=136)
    assert 0.9 * 32000 <= m32.n_params <= 1.15 * 32000, m32.n_params
    assert 0.9 * 8000 <= m8.n_params <= 1.15 * 8000, m8.n_params


def test_energy_is_scalar_and_grad_flows_to_splines():
    m = GroupKANEnergyModel(hidden=136)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    E = m.energy(x)
    assert E.ndim == 0
    E.backward()
    for name, p in m.named_parameters():
        if 'values' in name:            # the shared univariate functions
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_pwl_is_identity_at_init():
    phi = SharedPWL(n_groups=4)
    x = torch.linspace(-2.5, 2.5, 64).unsqueeze(0).repeat(3, 1)  # (3, 64)
    gidx = torch.arange(64) % 4
    y = phi(x, gidx)
    assert torch.allclose(y, x, atol=1e-5)


def test_denoise_shape_and_dsm_loss_finite():
    m = GroupKANEnergyModel(hidden=136)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    out = m.denoise(x + 0.1 * torch.randn_like(x), n_steps=3)
    assert out.shape == x.shape
    loss = m.loss(x, sigma=0.15)
    assert torch.isfinite(loss)
