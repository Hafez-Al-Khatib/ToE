import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

import frontier_flops
from frontier_flops import flops_per_step, flops_per_step_detail


class DummyLinearEBM(nn.Module):
    """Energy = w . x  (single Linear D->1, no bias interactions to count)."""
    def __init__(self, D=3 * 32 * 32):
        super().__init__()
        self.net = nn.Linear(D, 1, bias=False)

    def energy(self, x):
        return self.net(x.view(x.shape[0], -1)).sum()


def test_flops_matches_analytic_linear():
    D = 3 * 32 * 32
    model = DummyLinearEBM(D)
    got = flops_per_step(model, torch.device('cpu'))
    # forward mm: 2*D; backward computes ONLY grad_input (one mm, 2*D) --
    # autograd.grad(E, xi) never forms grad_weight. Inference descent
    # therefore costs ~2x forward, not the 3x training heuristic.
    expected = 4 * D
    assert abs(got - expected) <= 0.05 * expected, f"got {got}, expected {expected}"


def test_flops_positive_for_repo_model_shape():
    model = DummyLinearEBM()
    assert flops_per_step(model, torch.device('cpu')) > 0


def test_flops_per_step_detail_measured_path():
    D = 3 * 32 * 32
    model = DummyLinearEBM(D)
    detail = flops_per_step_detail(model, torch.device('cpu'))

    assert detail['method'] == 'measured'

    expected_total = 4 * D
    expected_forward = 2 * D
    assert abs(detail['total'] - expected_total) <= 0.05 * expected_total, (
        f"total: got {detail['total']}, expected {expected_total}")
    assert abs(detail['forward'] - expected_forward) <= 0.05 * expected_forward, (
        f"forward: got {detail['forward']}, expected {expected_forward}")

    ratio = detail['total'] / detail['forward']
    assert abs(ratio - 2.0) <= 0.1, f"total/forward ratio {ratio} not ~= 2.0"


def test_flops_per_step_detail_fallback_path(monkeypatch):
    """When the full forward+backward measurement raises RuntimeError
    with the specific torch 2.5.1 FlopCounterMode/autograd.grad bug signature
    '_will_engine_execute_node' (simulating certain backward graphs, e.g.
    UNetEBM), flops_per_step_detail must fall back to forward-only-measurement x2."""
    D = 3 * 32 * 32
    model = DummyLinearEBM(D)

    def _raising_grad(*args, **kwargs):
        raise RuntimeError(
            "A leaf node was passed to _will_engine_execute_node but we "
            "are currently running autograd.grad(). This is currently "
            "not supported.")

    monkeypatch.setattr(frontier_flops.torch.autograd, 'grad', _raising_grad)

    detail = flops_per_step_detail(model, torch.device('cpu'))

    assert detail['method'] == 'forward_x2'
    assert detail['total'] == 2 * detail['forward']


def test_flops_per_step_detail_unrelated_error_reraises(monkeypatch):
    """When the full forward+backward measurement raises RuntimeError
    with a signature other than '_will_engine_execute_node' (some unrelated
    failure), flops_per_step_detail must re-raise instead of falling back."""
    D = 3 * 32 * 32
    model = DummyLinearEBM(D)

    def _raising_grad(*args, **kwargs):
        raise RuntimeError('some unrelated failure')

    monkeypatch.setattr(frontier_flops.torch.autograd, 'grad', _raising_grad)

    with pytest.raises(RuntimeError, match='some unrelated failure'):
        flops_per_step_detail(model, torch.device('cpu'))
