import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

from frontier_flops import flops_per_step


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
    # forward mm: 2*D; backward (grad_input + grad_weight): 2 more mms, 2*D each.
    expected = 6 * D
    assert abs(got - expected) <= 2 * D, f"got {got}, expected ~{expected}"


def test_flops_positive_for_repo_model_shape():
    model = DummyLinearEBM()
    assert flops_per_step(model, torch.device('cpu')) > 0
