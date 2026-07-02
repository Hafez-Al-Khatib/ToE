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
    # forward mm: 2*D; backward computes ONLY grad_input (one mm, 2*D) --
    # autograd.grad(E, xi) never forms grad_weight. Inference descent
    # therefore costs ~2x forward, not the 3x training heuristic.
    expected = 4 * D
    assert abs(got - expected) <= 0.05 * expected, f"got {got}, expected {expected}"


def test_flops_positive_for_repo_model_shape():
    model = DummyLinearEBM()
    assert flops_per_step(model, torch.device('cpu')) > 0
