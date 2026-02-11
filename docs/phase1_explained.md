# Phase 1: Energy-Based Denoising — The Ball in a Bowl

This document provides the complete mathematical and conceptual explanation of Phase 1: using energy-based models for image denoising.

---

## Table of Contents
1. [The Denoising Problem](#1-the-denoising-problem)
2. [Why Energy Models Are Perfect for Denoising](#2-why-energy-models-are-perfect-for-denoising)
3. [The Mathematical Framework](#3-the-mathematical-framework)
4. [The Architecture](#4-the-architecture)
5. [Training: Contrastive Divergence](#5-training-contrastive-divergence)
6. [Inference: Langevin Dynamics](#6-inference-langevin-dynamics)
7. [Implementation Details](#7-implementation-details)

---

## 1. The Denoising Problem

### Problem Statement

Given a noisy image $\tilde{x} = x + \eta$ where:
- $x \in \mathbb{R}^{784}$ is the clean image (28×28 MNIST digit)
- $\eta \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian noise
- $\tilde{x}$ is the observed noisy image

**Goal**: Recover $x$ from $\tilde{x}$.

### Traditional Approaches

| Method | Mechanism | Limitation |
|--------|-----------|------------|
| Autoencoders | $\hat{x} = f_\theta(\tilde{x})$ | Single forward pass, no refinement |
| Denoising Score Matching | Learn $\nabla_x \log p(x)$ | Requires careful noise scheduling |
| Variational methods | Optimize $\|x - \tilde{x}\|^2 + \lambda R(x)$ | Hand-crafted regularizer $R(x)$ |

### Our Approach: Energy Minimization

We learn an energy function $E_\theta(x)$ where:
- Clean images have **low energy**
- Noisy/corrupted images have **high energy**

**Denoising = Rolling downhill from noise to clean data.**

---

## 2. Why Energy Models Are Perfect for Denoising

### The Physical Intuition

Imagine dropping a ball on a hilly landscape:

```
Energy
  ↑
  │    ╱╲    ╱╲
  │   ╱  ╲  ╱  ╲     ← Noisy images (high energy)
  │  ╱    ╲╱    ╲
  │ ╱      ●      ╲  ← Ball starts here (noisy input)
  │╱        ↓      ╲
  └─────────●───────→ x
            ↑
       Clean image (minimum)
```

The ball naturally rolls to the **bottom of the valley** — the clean image.

### Mathematical Insight

Consider the Boltzmann distribution over images:

$$p(x) = \frac{1}{Z} \exp(-E_\theta(x))$$

If we train $E_\theta$ on clean MNIST digits:
- The 10 digit classes form 10 "valleys" in energy space
- Each valley corresponds to a stable attractor
- Noisy inputs sit on the "hills" between valleys
- Gradient descent naturally flows toward the nearest valley

### Key Advantages

1. **Iterative refinement**: We can run dynamics for as long as needed
2. **Natural denoising**: Energy gradient points toward clean data
3. **Uncertainty**: Energy value indicates confidence
4. **Multi-modal**: Can represent multiple valid reconstructions

---

## 3. The Mathematical Framework

### Energy Function

We parameterize the energy function as a neural network:

$$E_\theta: \mathbb{R}^n \rightarrow \mathbb{R}$$
$$E_\theta(x) = \text{MLP}(x) \in \mathbb{R}$$

The network takes an image as input and outputs a **single scalar** — the energy.

### The Gibbs Distribution

The energy defines a probability distribution:

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}$$

where the partition function is:

$$Z_\theta = \int_{\mathbb{R}^n} \exp(-E_\theta(x)) \, dx$$

**Problem**: $Z_\theta$ is intractable (integral over all possible images).

**Solution**: Use **contrastive methods** that only require energy **differences**.

### The Score Function

The **score** is the gradient of log-probability:

$$s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

This is computable! We just backpropagate through the energy network.

The score points **toward high probability** (low energy) — exactly what we need for denoising.

---

## 4. The Architecture

### Energy Network Design

```
Input x ∈ ℝ⁷⁸⁴ (flattened 28×28 image)
    ↓
Linear(784.model → 512) + LayerNorm + SiLU
    ↓
Linear(512 → 256) + LayerNorm + SiLU
    ↓
Linear(256 → 128) + LayerNorm + SiLU
    ↓
Linear(128 → 1)  ← Scalar energy output
```

### Design Choices

1. **SiLU activation** (Swish): Smooth gradients, no dead neurons
   $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

2. **Layer normalization**: Stabilizes training, normalizes energy scale

3. **No final activation**: Energy is unbounded (can be any real number)

4. **Scalar output**: Energy is a single number, not a vector

### Why This Architecture?

The network must satisfy:
- **Smoothness**: Energy gradient must exist and be continuous
- **Expressivity**: Must capture complex data manifold
- **Stability**: Must not produce extreme energy values

---

## 5. Training: Contrastive Divergence

### The Objective

We want to maximize the log-likelihood of the training data:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]$$

Expanding:

$$\mathcal{L}(\theta) = -\mathbb{E}_{p_{\text{data}}}[E_\theta(x)] - \log Z_\theta$$

The gradient is:

$$\nabla_\theta \mathcal{L} = -\mathbb{E}_{p_{\text{data}}}[\nabla_\theta E_\theta(x)] + \mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x)]$$

### Interpretation

The gradient has two terms:

1. **Positive phase**: $-\nabla_\theta E_\theta(x_{\text{data}})$
   - Push **down** the energy of real data

2. **Negative phase**: $+\nabla_\theta E_\theta(x_{\text{model}})$
   - Push **up** the energy of model samples

**Problem**: Sampling from $p_\theta$ exactly is expensive.

### Contrastive Divergence (CD-k)

**Hinton's insight**: Start MCMC from data, run only k steps.

**Algorithm**:
```
for each mini-batch of clean images x₀:
    # Generate negative samples
    x_k = x_0  # Start from real data
    for i in 1..k:
        x_k = x_k - ε∇ₓE(x_k) + √(2ε)·η   # Langevin step
    
    # Compute loss
    loss = E(x_0) - E(x_k)  # Push down real, push up fake
    
    # Update parameters
    θ ← θ - α·∇θ(loss)
```

### Persistent Contrastive Divergence (PCD)

**Improvement**: Maintain a **buffer** of negative samples across batches.

```python
# Initialize buffer once
buffer = torch.randn(buffer_size, image_dim)

# Each training step
indices = random.sample(buffer_size, batch_size)
x_neg = buffer[indices]  # Get from buffer
x_neg = langevin_step(x_neg, k=10)  # Refine
buffer[indices] = x_neg  # Store back

loss = E(x_pos).mean() - E(x_neg).mean()
```

**Why it works**: The buffer provides better mixing than short chains from data.

---

## 6. Inference: Langevin Dynamics

### The Algorithm

Given noisy input $\tilde{x}$, we recover the clean image by running:

$$x_{t+1} = x_t - \epsilon \nabla_x E_\theta(x_t) + \sqrt{2\epsilon\tau} \cdot \eta_t$$

where:
- $\epsilon$ is the step size (learning rate for inference)
- $\tau$ is the temperature (noise level)
- $\eta_t \sim \mathcal{N}(0, I)$ is random noise

### Deterministic vs Stochastic

**Stochastic Langevin** (with noise):
- Samples from the distribution $p_\theta(x)$
- Good for exploring multiple modes
- Better for highly ambiguous inputs

**Deterministic gradient descent** (τ = 0):
- Finds the MAP estimate (mode of distribution)
- Faster convergence
- Better when input is close to a single mode

For denoising, we typically use **deterministic** with a small noise term:

$$x_{t+1} = x_t - \epsilon \nabla_x E_\theta(x_t)$$

### Annealed Langevin Dynamics

**Problem**: Large step sizes explore but don't converge; small step sizes converge but get stuck.

**Solution**: Start with large steps (exploration), gradually decrease (exploitation).

```python
for t in range(T):
    # Annealing schedule
    epsilon_t = epsilon_start * (epsilon_end / epsilon_start) ** (t / T)
    
    # Langevin update
    grad = torch.autograd.grad(E(x).sum(), x)[0]
    x = x - epsilon_t * grad
```

### Convergence

**Theorem**: Under mild conditions on $E_\theta$, Langevin dynamics converges to a local minimum of the energy.

**In practice**: 50-200 steps is usually sufficient for MNIST denoising.

---

## 7. Implementation Details

### Numerical Stability

1. **Gradient clipping**: Prevents explosion
   ```python
   grad = torch.clamp(grad, -1.0, 1.0)
   ```

2. **Energy regularization**: Penalize extreme energies
   ```python
   loss = E_pos.mean() - E_neg.mean() + 0.01 * (E_pos**2 + E_neg**2).mean()
   ```

3. **Spectral normalization**: Control Lipschitz constant
   - Ensures energy gradient is bounded
   - Improves training stability

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 1e-4 | For training |
| Langevin steps (training) | 10-50 | More = better negatives, slower |
| Langevin steps (inference) | 50-200 | More = cleaner output |
| Langevin step size | 10-100 | Depends on energy scale |
| Buffer size | 10000 | For PCD |
| Noise σ for training | 0.3-0.5 | Of pixel intensity [0,1] |

### Evaluation Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**:
   $$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
   Higher is better. >20 dB is decent, >25 dB is good.

2. **Energy ratio**:
   $$r = \frac{E(\text{noisy})}{E(\text{clean})}$$
   Should be > 1 if model learned correctly.

3. **Visual quality**: The ultimate test — do digits look sharp?

---

## Summary

Phase 1 demonstrates the core principle:

> **Inference is not computation — it's relaxation.**

We train an energy function to assign low values to clean data and high values to noise. Denoising is simply following the energy gradient until we reach a minimum.

This is the "ball in a bowl" paradigm: the answer emerges from physics, not from symbolic computation.

---

*Next: See [phase2_explained.md](phase2_explained.md) for how KAN layers create more expressive energy landscapes.*
