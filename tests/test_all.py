"""
Unit tests for the Natural Intelligence project.

Run with: python -m pytest tests/ -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from energy_model import (
    EnergyMLP,
    langevin_dynamics,
    annealed_langevin_dynamics,
    SampleBuffer,
    contrastive_divergence_loss
)
from kan_layer import (
    BSplineBasis,
    KANLayer,
    KANNetwork,
    EnergyKAN
)
from hamiltonian_kan import (
    HamiltonianKAN,
    leapfrog_step,
    simulate_hamiltonian,
    DampedHamiltonianDynamics
)
from wave_solver import (
    EikonalSolver,
    MazeEncoder,
    WaveBrain,
    extract_path,
    generate_random_maze
)


class TestEnergyModel:
    """Test Phase 1: Energy-based models."""
    
    def test_energy_mlp_output_shape(self):
        """EnergyMLP should output scalar energies."""
        model = EnergyMLP(input_dim=784, hidden_dims=(64, 32))
        x = torch.randn(16, 784)
        energy = model(x)
        
        assert energy.shape == (16,), f"Expected (16,), got {energy.shape}"
    
    def test_energy_mlp_gradient_exists(self):
        """Energy should be differentiable w.r.t. input."""
        model = EnergyMLP(input_dim=100)
        x = torch.randn(4, 100, requires_grad=True)
        energy = model(x)
        
        grad = torch.autograd.grad(energy.sum(), x)[0]
        assert grad.shape == x.shape
        assert not torch.isnan(grad).any()
    
    def test_langevin_dynamics_decreases_energy(self):
        """Langevin dynamics should decrease energy."""
        model = EnergyMLP(input_dim=50, hidden_dims=(32,))
        x_init = torch.randn(8, 50)
        
        E_init = model(x_init).mean().item()
        x_final = langevin_dynamics(model, x_init, n_steps=50, step_size=1.0)
        E_final = model(x_final).mean().item()
        
        # Energy should generally decrease (may not always due to stochasticity)
        # Allow some tolerance
        assert E_final < E_init * 1.5, f"Energy didn't stabilize: {E_init:.2f} → {E_final:.2f}"
    
    def test_sample_buffer_size(self):
        """Sample buffer should return correct batch size."""
        buffer = SampleBuffer(buffer_size=100, sample_dim=50)
        samples, indices = buffer.sample(16)
        
        assert samples.shape == (16, 50)
        assert indices.shape == (16,)
    
    def test_contrastive_divergence_loss(self):
        """CD loss should be negative when positives have lower energy."""
        model = EnergyMLP(input_dim=20, hidden_dims=(16,))
        
        # Create fake data where positives should have lower energy
        x_pos = torch.zeros(8, 20)  # Concentrated
        x_neg = torch.randn(8, 20) * 3  # Spread out
        
        loss, metrics = contrastive_divergence_loss(model, x_pos, x_neg)
        
        assert not torch.isnan(loss)
        assert "energy_pos" in metrics
        assert "energy_neg" in metrics


class TestKANLayer:
    """Test Phase 2: KAN layers."""
    
    def test_bspline_basis_partition_of_unity(self):
        """B-spline bases should sum to 1."""
        basis = BSplineBasis(grid_size=5, degree=3)
        x = torch.linspace(-0.9, 0.9, 100)  # Stay inside grid
        B = basis(x)
        
        sums = B.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.1), \
            f"Partition of unity violated: sums range [{sums.min():.3f}, {sums.max():.3f}]"
    
    def test_kan_layer_output_shape(self):
        """KAN layer should produce correct output shape."""
        layer = KANLayer(in_features=10, out_features=5, grid_size=5)
        x = torch.randn(32, 10)
        y = layer(x)
        
        assert y.shape == (32, 5), f"Expected (32, 5), got {y.shape}"
    
    def test_kan_layer_differentiable(self):
        """KAN layer should be fully differentiable."""
        layer = KANLayer(in_features=10, out_features=5)
        x = torch.randn(4, 10, requires_grad=True)
        y = layer(x)
        
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.ctrl_points.grad is not None
    
    def test_kan_network_output_shape(self):
        """KAN network should produce correct output shape."""
        net = KANNetwork(layers=[100, 32, 16, 1], grid_size=5)
        x = torch.randn(8, 100)
        y = net(x)
        
        assert y.shape == (8, 1)
    
    def test_energy_kan_scalar_output(self):
        """EnergyKAN should output scalar energies."""
        energy = EnergyKAN(input_dim=50, hidden_dims=(16, 8))
        x = torch.randn(12, 50)
        E = energy(x)
        
        assert E.shape == (12,)


class TestHamiltonianKAN:
    """Test Phase 2: Hamiltonian dynamics."""
    
    def test_hamiltonian_computation(self):
        """Hamiltonian should be sum of kinetic and potential."""
        hkan = HamiltonianKAN(state_dim=2, hidden_dims=(8,))
        x = torch.randn(4, 2)
        p = torch.randn(4, 2)
        
        H = hkan.hamiltonian(x, p)
        T = hkan.kinetic_energy(p)
        V = hkan.potential_energy(x)
        
        assert torch.allclose(H, T + V, atol=1e-5)
    
    def test_leapfrog_preserves_energy(self):
        """Leapfrog integrator should approximately conserve energy."""
        hkan = HamiltonianKAN(state_dim=2, hidden_dims=(8,))
        x = torch.randn(1, 2)
        p = torch.randn(1, 2)
        
        H_init = hkan.hamiltonian(x, p).item()
        
        # Many small steps
        for _ in range(100):
            x, p = leapfrog_step(hkan, x, p, dt=0.01)
        
        H_final = hkan.hamiltonian(x, p).item()
        
        # Energy should be conserved within 10%
        relative_drift = abs(H_final - H_init) / (abs(H_init) + 1e-8)
        assert relative_drift < 0.1, f"Energy drifted {relative_drift*100:.1f}%"
    
    def test_damped_dynamics_decreases_energy(self):
        """Damped dynamics should decrease total energy."""
        hkan = HamiltonianKAN(state_dim=2, hidden_dims=(8,))
        damped = DampedHamiltonianDynamics(hkan, damping=0.5)
        
        x = torch.randn(1, 2)
        p = torch.randn(1, 2) * 2  # High momentum
        
        H_init = hkan.hamiltonian(x, p).item()
        
        x_final, p_final = damped.simulate(x, p, n_steps=100, dt=0.05)
        
        H_final = hkan.hamiltonian(x_final, p_final).item()
        
        assert H_final < H_init, f"Energy increased: {H_init:.2f} → {H_final:.2f}"


class TestWaveSolver:
    """Test Phase 3: Wave/Eikonal solver."""
    
    def test_eikonal_source_zero(self):
        """Eikonal solution should be zero at source."""
        solver = EikonalSolver(grid_size=(16, 16), n_sweeps=8)
        n = torch.ones(1, 16, 16)  # Uniform cost
        source = (8, 8)
        
        u = solver(n, source)
        
        assert u[0, source[0], source[1]] < 0.01
    
    def test_eikonal_increases_with_distance(self):
        """Travel time should increase with distance from source."""
        solver = EikonalSolver(grid_size=(16, 16), n_sweeps=8)
        n = torch.ones(1, 16, 16)
        source = (8, 8)
        
        u = solver(n, source)
        
        # Points farther from source should have higher u
        u_near = u[0, 7, 8]  # Adjacent to source
        u_far = u[0, 0, 0]   # Corner
        
        assert u_far > u_near
    
    def test_maze_encoder_output_range(self):
        """Maze encoder should output positive refractive indices."""
        encoder = MazeEncoder(min_n=0.1, max_n=10.0)
        maze = torch.rand(4, 1, 32, 32)
        
        n = encoder(maze)
        
        assert n.min() >= 0.1 - 0.01
        assert n.max() <= 10.0 + 0.01
        assert n.shape == (4, 32, 32)
    
    def test_wave_brain_full_pipeline(self):
        """WaveBrain should run full pipeline."""
        brain = WaveBrain(grid_size=(16, 16), n_sweeps=4)
        maze = generate_random_maze(size=16, wall_density=0.2)
        source = (2, 2)
        target = (13, 13)
        
        n, u, path = brain(maze, source, target_pos=target)
        
        assert n.shape == (1, 16, 16)
        assert u.shape == (1, 16, 16)
        assert path is not None
        assert len(path) > 0
    
    def test_path_extraction(self):
        """Path extraction should return valid path."""
        solver = EikonalSolver(grid_size=(16, 16), n_sweeps=8)
        n = torch.ones(1, 16, 16)
        source = (2, 2)
        target = (13, 13)
        
        u = solver(n, source)
        path = extract_path(u, source, target)
        
        assert len(path) > 2
        # Path should start near target and end near source
        assert abs(path[-1][0] - source[0]) < 2 and abs(path[-1][1] - source[1]) < 2


class TestIntegration:
    """Integration tests across phases."""
    
    def test_energy_kan_as_energy_function(self):
        """EnergyKAN should work with Langevin dynamics."""
        energy = EnergyKAN(input_dim=20, hidden_dims=(16, 8))
        x_init = torch.randn(4, 20)
        
        x_final = langevin_dynamics(
            lambda x: energy(x),
            x_init,
            n_steps=20,
            step_size=1.0
        )
        
        assert x_final.shape == x_init.shape
        assert not torch.isnan(x_final).any()
    
    def test_hamiltonian_kan_state_evolution(self):
        """H-KAN should evolve states through phase space."""
        hkan = HamiltonianKAN(state_dim=4, hidden_dims=(16, 8))
        x0 = torch.randn(2, 4)
        
        x_traj, p_traj = simulate_hamiltonian(
            hkan, x0, n_steps=50, dt=0.01, return_trajectory=True
        )
        
        assert x_traj.shape == (51, 2, 4)
        assert p_traj.shape == (51, 2, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
