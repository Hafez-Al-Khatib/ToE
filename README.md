# Natural Intelligence: The Physics of Thought

> *"We are done with 'Artificial' Intelligence. We are building Natural Intelligence."*

This repository implements the research manifesto **"From Computational Approximation to Physical Relaxation"** — a paradigm shift where intelligence emerges from energy minimization, not computation.

## Core Philosophy

Traditional neural networks **compute** answers via forward passes. Our approach lets answers **emerge** as a physical system settles into equilibrium.

| Paradigm | Metaphor | Answer Mechanism |
|----------|----------|------------------|
| Feedforward NN | Calculator | `y = f(x)` |
| Energy-Based Model | Ball in Bowl | `x* = argmin E(x)` |

## Project Structure

```
ToE/
├── docs/                    # Deep theoretical documentation
│   ├── theory_foundations.md   # Mathematical foundations
│   ├── phase1_explained.md     # Energy-based denoising theory
│   ├── phase2_explained.md     # Hamiltonian-KAN theory
│   └── phase3_explained.md     # Path integral theory
├── src/
│   ├── energy_model.py      # Phase 1: Energy function + Langevin dynamics
│   ├── train_phase1.py      # Training for MNIST denoising
│   ├── kan_layer.py         # Phase 2: KAN implementation
│   ├── hamiltonian_kan.py   # Phase 2: H-KAN dynamics
│   ├── wave_solver.py       # Phase 3: Eikonal solver
│   └── visualize.py         # Energy landscape visualization
├── notebooks/               # Interactive explorations
├── outputs/                 # Generated visualizations
└── tests/                   # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run Phase 1: Energy-Based Denoising
python src/train_phase1.py --epochs 20

# Visualize energy landscapes
python src/visualize.py
```

## The Three Phases

### Phase 1: Ball in a Bowl
Prove that "settling" works for image denoising on MNIST.

### Phase 2: Hamiltonian-KAN  
Use spline-based KANs to create complex, interpretable energy landscapes.

### Phase 3: Wave Brain
Solve reasoning problems at O(1) depth using wave interference.

---

*Research conducted at AUB. Inspired by Active Inference, Free Energy Principle, and the unity of physical law.*
