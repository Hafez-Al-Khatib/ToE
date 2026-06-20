# Comprehensive Survey: Optimizations for Kolmogorov-Arnold Networks (KANs)
## Efficiency, Parallelization, and Expressivity Preservation

**Date:** May 2026  
**Scope:** FLOP reduction, hardware acceleration, basis-function alternatives, compression, and critical evaluation of claims across the KAN literature.

---

## Executive Summary

Kolmogorov-Arnold Networks (KANs) have generated significant interest as interpretable alternatives to Multi-Layer Perceptrons (MLPs). However, their practical adoption is severely hindered by computational inefficiency: original B-spline KANs are typically **10× slower** than MLPs with comparable parameter counts, and their FLOP count is substantially higher due to iterative basis evaluation. This survey systematically catalogs and critically evaluates the optimization strategies proposed to address this bottleneck, categorizing them into:

1. **Basis-function substitutions** (replacing B-splines with cheaper alternatives)
2. **Hardware-level acceleration** (CUDA kernels, fused operators, parallelization)
3. **Architectural redesigns** (hybrid MLP-KAN, parameter-lean variants, compression)
4. **Framework-level optimizations** (JAX, Triton, memory layout improvements)

**Key Finding:** While many KAN variants claim dramatic speedups, fair benchmarking under FLOP-controlled or parameter-controlled conditions reveals that most KAN optimizations narrow the gap with MLPs rather than establish a new efficiency frontier. The genuine advantage of KANs remains confined to **symbolic regression and low-dimensional scientific computing**; for general machine learning, computer vision, NLP, and audio tasks, optimized MLPs still dominate when computational budgets are equalized.

---

## 1. The Computational Bottleneck of Original KANs

### 1.1 FLOP Analysis: KAN vs. MLP

The canonical KAN layer replaces the weight matrix \(W \in \mathbb{R}^{d_{out} \times d_{in}}\) of an MLP with a matrix of learnable univariate functions \(\Phi = [\phi_{q,p}(\cdot)]\), where each \(\phi_{q,p}\) is parameterized as a B-spline expansion:

\[
\phi_{q,p}(x) = w_b \cdot b(x) + \sum_{i=1}^{G} c_{q,p,i} \cdot B_i(x)
\]

where \(G\) is the grid size and \(k\) is the spline degree.

**FLOP Formulas (per layer, per sample):**

| Model | Parameters | FLOPs |
|-------|-----------|-------|
| **MLP** | \(d_{in}d_{out} + d_{out}\) | \(2d_{in}d_{out} + 5d_{out}\) (with GELU) |
| **KAN (B-spline)** | \(d_{in}d_{out}(G + k + 3) + d_{out}\) | \(d_{in}d_{out}[9k(G + 1.5k) + 2G - 2.5k + 3] + 7d_{in}\) |
| **KAF (RFF)** | \(d_{in}M + M + 2d_{in} + d_{in}d_{out} + d_{out}\) | \(4d_{in}M + 2d_{in} + 2d_{in}d_{out} + 5d_{in}\) |

*Sources: KANbeFair (Yu et al., 2024), KAF paper (Zhang et al., 2025)*

For typical values (e.g., \(G=10, k=3\)), a KAN layer incurs **~30–100× more FLOPs** than an MLP layer of the same width. This is the fundamental bottleneck: even though KANs may use fewer parameters for a given task, the *per-parameter* computation is far more expensive due to the De Boor-Cox recursion for B-spline basis evaluation.

### 1.2 Empirical Observations

- **KANbeFair (Yu et al., NUS, 2024):** Controlled experiments across ML, CV, NLP, audio, and symbolic regression show that when **FLOPs are matched**, MLPs generally outperform KANs. KANs only reliably win in **symbolic formula representation**.
- **FPGA Implementation (Tran et al., 2024):** Hardware synthesis shows KANs consume **substantially higher LUTs and DSPs** than MLPs for high-dimensional classification, with latency 1.1–2.27× longer despite comparable accuracy.
- **KAN 2.0 Authors' Admission:** "The biggest bottleneck of KAN lies in its slow training. KANs are usually 10× slower than MLPs, given the same number of parameters." (Liu et al., 2024)

---

## 2. Basis-Function Substitutions: The Core Optimization Axis

The dominant strategy for reducing KAN FLOPs is to replace B-splines with basis families that are cheaper to evaluate, better suited to GPU parallelism, or aligned with problem structure. Below is a critical taxonomy.

### 2.1 FastKAN (RBF-Based)
- **Paper/Repo:** `fast-kan` (Li, 2024)
- **Technique:** Replaces B-splines with **Gaussian Radial Basis Functions (RBFs)**. Each edge function is a weighted sum of fixed Gaussian kernels.
- **Claimed Benefit:** "Very fast calculation" — avoids recursive spline evaluation.
- **Analysis:** RBF evaluation is non-iterative and vectorizable. However, RBF width tuning remains a hyperparameter burden. FastKAN is widely adopted as a pragmatic baseline and serves as the foundation for MetaCluster compression.
- **Speedup:** ~2–5× over B-spline KAN in wall-clock time (anecdotal community reports).

### 2.2 FasterKAN (RSWAF Bases)
- **Paper/Repo:** `faster-kan`
- **Technique:** Uses **Reflectional Switch Activation Functions (RSWAF)**.
- **Claimed Benefit:** As of May 2024, claimed to be the fastest KAN variant, only ~2× slower than MLP in backward pass.
- **Analysis:** RSWAFs are designed for efficient gradient computation. Limited independent benchmarking exists; the claim appears plausible for small networks but unverified at scale.

### 2.3 ChebyKAN (Chebyshev Polynomials)
- **Paper:** SS et al., 2024; `ChebyKAN` repo
- **Technique:** Replaces B-splines with **Chebyshev polynomials of the first/second kind**.
- **Claimed Benefit:** Strong speed-accuracy tradeoff; orthogonal polynomials reduce redundancy.
- **Critical Analysis:** 
  - Chebyshev evaluation via recurrence is cheaper than B-spline De Boor-Cox recursion.
  - However, high-degree Chebyshev modes can explode without **tanh normalization** (a recurring theme across polynomial KANs).
  - ACM Computing Surveys (2025) notes Chebyshev variants as "a strong speed-accuracy choice" in preliminary benchmarks.
  - **PolyKAN** (see §3) accelerates ChebyKAN further via fused CUDA kernels.

### 2.4 FourierKAN / KAF / FusedFourierKAN
- **Papers:** 
  - FourierKAN (Mehrabian et al., 2024; Xu et al., 2024)
  - KAF — Kolmogorov-Arnold Fourier Networks (Zhang et al., 2025)
  - FusedFourierKAN — optimized with fused GPU kernels
- **Technique:** Uses **Fourier series** or **Random Fourier Features (RFF)** as edge activations.
- **Claimed Benefit:** Efficient for periodic/oscillatory targets; RFF expansions are GPU-friendly and avoid recursion.
- **Critical Analysis:**
  - KAF provides explicit FLOP formulas showing significant reduction vs. B-spline KAN (see table above).
  - FusedFourierKAN implements custom kernel fusion for the Fourier basis, but gains are reported as modest compared to unfused baselines.
  - Best suited for problems with inherent periodicity; can suffer from spectral leakage on non-periodic domains.

### 2.5 Wav-KAN (Wavelet-Based) — A Critical Assessment
- **Paper:** Bozorgasl & Chen, 2024; arXiv:2405.12832
- **Technique:** Replaces B-splines with **wavelet basis functions** (continuous and discrete wavelet transforms).
- **Claimed Benefits:** 
  - "Enhanced accuracy, faster training speeds, increased robustness"
  - Multi-resolution analysis via DWT "obviates need for recalculation"
  - Better capture of high/low-frequency components
- **Skepticism & Critical Evaluation:**
  1. **Unfair Comparison:** The paper explicitly states that **batch normalization (BN)** significantly improves Wav-KAN accuracy, but notes that the original KAN paper "did not mention and didn't apply batch normalization." This is a critical confounder: BN is known to stabilize and accelerate training independently of the basis choice. The speedup claims may partly reflect BN rather than wavelets.
  2. **False DWT Claim:** Independent code analysis reveals that **there is no actual Discrete Wavelet Transform (DWT) implementation in the Wav-KAN architecture.** The paper claims multi-resolution analysis via DWT, but the implementation consists solely of parametric wavelet-shaped activation functions. The "DWT" claim is mathematically mischaracterized.
  3. **No Noise Experiments:** The "increased robustness" claim is entirely qualitative. No noise-injection experiments, adversarial robustness tests, or out-of-distribution evaluations were performed.
  4. **No Interpretability Analysis:** Despite claiming interpretability as a core advantage, no symbolic regression, pruning visualization, or interpretability metrics are reported.
  5. **Limited Evaluation Scope:** Experiments are restricted to MNIST and simple synthetic tasks. No large-scale validation on ImageNet, CIFAR-100, or modern NLP benchmarks.
  6. **FLOP Reality:** Wavelet transforms are not inherently cheaper than well-implemented B-splines. The paper provides no rigorous FLOP accounting.
  7. **Hype Indicators:** The abstract claims Wav-KAN is "analogous to how water conforms to the shape of its container" and aspires to make wavelets "as widespread as ReLU and sigmoid in UAT." Such language, combined with limited empirical rigor, warrants caution.
- **Verdict:** Wav-KAN is an example of the "KAN variant rush" — a technically modest contribution packaged with overstated claims. It offers a plausible structural prior for multi-scale problems, but its claims of universal speedup, robustness, and interpretability are **not convincingly substantiated**. If considering wavelet-based approaches, **DecoKAN** (which uses actual DWT decomposition + KAN processing) is far more sound. Wav-KAN should be treated as a domain-specific basis choice, not a general KAN acceleration strategy.

### 2.6 ReLU-KAN / HRKAN (ReLU-Power Functions)
- **Paper/Repo:** `relu_kan`, `HRKAN`
- **Technique:** Replaces splines with **ReLU-power functions**: \(\sigma_k(x) = \max(0, x)^k\).
- **Claimed Benefit:** Non-iterative, trivial GPU parallelization, no basis recursion.
- **Analysis:** 
  - The Practitioner's Guide (2025) explicitly recommends ReLU-KAN/HRKAN when **speed is the primary objective**.
  - These are the closest KAN variants to MLPs in computational structure; the edge functions are piecewise polynomial but require no coefficient lookup or recurrence.
  - Interpretability is partially preserved (piecewise linear/quadratic/cubic shapes are still human-inspectable).
  - Expressivity may be reduced for smooth, globally varying functions compared to B-splines.

### 2.7 rKAN (Rational Functions)
- **Paper:** Aghaei, 2024/2025
- **Technique:** Uses **Padé / Jacobi rational functions** as edge activations.
- **Claimed Benefit:** Smoother approximation, improved computational efficiency.
- **Analysis:** Rational functions can approximate poles and singularities well. However, the ACM Computing Surveys review notes that "these benefits come with added design and training complexity, particularly in high-dimensional settings." The efficiency gains are real for specific PDE problems but not generalizable.

### 2.8 fKAN (Fractional / Jacobi Polynomials)
- **Paper:** Aghaei, 2024
- **Technique:** Uses **Jacobi polynomials** with fractional/domain parameters.
- **Claimed Benefit:** Controllable smoothness and boundary shaping.
- **Analysis:** Similar to ChebyKAN but with additional hyperparameters (Jacobi shape parameters). Good for problems with known boundary behavior, but hyperparameter tuning burden offsets efficiency gains.

### 2.9 SincKAN
- **Paper:** Reinhardt & Gleyzer, 2024
- **Technique:** Uses **sinc functions** as basis.
- **Claimed Benefit:** Bandlimited approximations, ideal for signal processing and PINNs with known frequency constraints.
- **Analysis:** Highly specialized. Sinc evaluation requires care near zero crossings. Not a general acceleration strategy.

### 2.10 GramKAN
- **Paper/Repo:** Drokin, 2024; `MetaCluster` paper
- **Technique:** Uses **Gram polynomials**.
- **Analysis:** Included in MetaCluster experiments as a baseline. Gram polynomials are discrete orthogonal polynomials; they offer moderate efficiency but lack the GPU optimization ecosystem of Chebyshev or Fourier variants.

### 2.11 FC-KAN (Function Combinations)
- **Paper:** Ta, 2024; arXiv:2409.01763
- **Technique:** Combines outputs of multiple basis families (B-splines + wavelets + RBFs) via element-wise sum, product, concatenation, etc.
- **Claimed Benefit:** Two variants (B-splines + Derivative-of-Gaussian, and B-splines + quadratic linear transform) outperformed FastKAN, EfficientKAN, FasterKAN on MNIST/Fashion-MNIST.
- **Analysis:** Combining bases increases expressive flexibility but **increases parameter and FLOP counts**, moving in the opposite direction of efficiency. The outperformance on small datasets may reflect overfitting to basis diversity rather than genuine efficiency.

### 2.12 LSS-SKAN (Single-Parameter Learnable Spline Shape)
- **Technique:** Uses a **single-parameter learnable spline shape (LSS)** per edge, dramatically reducing the parameter count per activation function.
- **Claimed Benefit:** Fastest "pure KAN" variant (edge-wise functions preserved) with ~2× slowdown vs. MLP; interpretability simplified due to single-parameter shapes.
- **Analysis:** Often overlooked in the literature. If edge-wise interpretability must be preserved, LSS-SKAN offers the best speed/interpretability trade-off among true KAN architectures. However, the restricted function space may limit expressivity for complex mappings.

### 2.13 FlashKAT / GR-KAN
- **Technique:** System-level optimizations combining **Group KAN (GR-KAN)** parameter sharing with kernel fusion.
- **Claimed Benefit:** **86.5× speedup** over baseline KAT implementation; matching accuracy.
- **Analysis:** A systems-level win rather than a new basis. Group sharing reduces redundant parameters across edges, while fused kernels eliminate launch overhead. Best suited for transformer-scale deployments where KAN layers are integrated into large architectures.

### 2.14 Free-RBF-KAN
- **Technique:** Adaptive RBF widths that are learned per-edge rather than fixed.
- **Analysis:** More flexible than FastKAN's fixed RBF grid but at higher parameter cost. Suitable for scientific computing where function smoothness varies spatially.

### 2.15 PowerMLP: The MLP-with-KAN-Expressivity Approach
- **Paper:** Qiu et al., 2024/2025; arXiv:2412.13571, AAAI 2025
- **Technique:** An MLP-type network using **node-wise polynomial bases** combined with linear mixing, rather than edge-wise splines.
- **Claimed Benefit:** 
  - Trains ~**40× faster** than KAN
  - Theoretically equivalent expressivity to KAN over bounded domains
  - Can be converted back to an equivalent KAN representation (though conversion is non-trivial)
- **Critical Analysis:**
  - PowerMLP is arguably the most credible efficiency breakthrough in the KAN space. By abandoning the per-edge function parameterization and returning to a node-wise activation structure (like MLPs), it recovers near-MLP training speeds while retaining KAN-level approximation theory.
  - **Interpretability Trade-off:** The conversion lemmas show that PowerMLP *can* be mapped to KAN form, but this involves "basis changes and dense coefficient expansions" that may increase parameter complexity and numerical conditioning issues. Edge-level interpretability is lost unless explicitly reconstructed.
  - The Practitioner's Guide (2025) treats PowerMLP as a distinct architecture rather than a pure KAN variant, noting it as an "MLP-type network with KAN-level expressiveness."
- **Verdict:** PowerMLP represents the most promising path for **practical deployment** where KAN-like expressivity is desired but MLP-like training speed is mandatory. It is not a KAN optimization per se but a **reconciliation** of KAN theory with MLP practice.

---

## 3. Hardware-Level Acceleration and Parallelization

Beyond changing the basis, significant speedups come from optimizing how basis functions are evaluated on modern hardware.

### 3.1 PolyKAN: Fused CUDA Kernels for Polynomial KANs
- **Paper:** Zhong et al., Sun Yat-sen University, 2025; arXiv:2511.14852 (PPoPP 2026)
- **Target:** Chebyshev, Legendre, and other polynomial KAN variants
- **Key Innovations:**
  1. **Lookup Table (LUT) with Interpolation:** Pre-computes basis functions offline; runtime uses linear interpolation instead of expensive trigonometric/recurrence evaluation.
  2. **2D Tiling:** Partitions input/output dimensions into rectangular blocks per thread block, improving cache locality.
  3. **Two-Stage Reduction:** Replaces scattered atomic updates with a partial-sum stage in shared memory + combine stage, eliminating atomic contention.
  4. **Coefficient Layout Reordering:** Reorders coefficient tensor from `[in, out, degree]` to `[degree, out, in]` for unit-stride reads.
- **Reported Speedups:**
  - **1.2–10× faster inference** than Triton + cuBLAS baseline
  - **1.4–12× faster training**
  - Identical accuracy on speech, audio enhancement, and tabular regression
- **Significance:** First **general, variant-agnostic** GPU operator library for KANs accepted at a top-tier systems venue (PPoPP 2026). Prior work (e.g., FusedFourierKAN) was tightly coupled to specific bases.

### 3.2 UKAN / warpKAN (NVIDIA Research)
- **Technique:** Custom CUDA kernels using warp-level primitives for B-spline evaluation; supports **unbounded domains** via dynamic grid extension.
- **Reported Speedups:**
  - **3–30× speedup** over pykan baseline
  - **Up to 1000× memory reduction** for large grid sizes
- **Analysis:** The most mature B-spline KAN acceleration for large-scale problems. Unbounded domain support eliminates the need for manual range tuning. NVIDIA-backed implementation with production-quality kernels.

### 3.3 FusionKAN
- **Technique:** Single fused CUDA kernel for entire B-spline KAN forward/backward pass; uses `__ldg` cache for read-only data.
- **Reported Speedups:** **33×** over baseline implementations.
- **Memory:** Achieves **O(1) memory scaling** vs. O(grid) for standard implementations — unmatched for memory-constrained settings.
- **Analysis:** Most extreme memory efficiency among B-spline accelerators. The O(1) scaling is critical for edge devices.

### 3.4 KAT (Kolmogorov-Arnold Transformer) CUDA Kernels
- **Technique:** CUDA/Triton kernels for rational-function KAN layers with **Group KAN (GR-KAN)** parameter sharing.
- **Analysis:** Designed for transformer integration. Group sharing reduces parameters by sharing basis functions across edge groups. FlashKAT (see §2.13) builds on this for 86.5× system-level speedup.

### 3.5 LinearKAN
- **Technique:** Dynamic input-indexed Sparse Matrix-Matrix multiplication (SpMM) via **cuSPARSE**.
- **Basis:** Linear B-splines (degree-1)
- **Claimed Benefit:** "Orders of magnitude" speedup on Ampere+ GPUs due to structured sparsity.
- **Analysis:** Restricted to linear splines, which limits smoothness. Best for problems where piecewise-linear approximations suffice and GPU tensor cores can exploit the sparse structure.

### 3.6 FusedFourierKAN
- **Repo:** `FusedFourierKAN`
- **Technique:** Fused CUDA kernels for FourierKAN forward/backward passes.
- **Analysis:** Gains are limited by simple kernel fusion without the systematic memory-bandwidth optimizations seen in PolyKAN. PolyKAN authors explicitly note FusedFourierKAN yields "limited performance gains" due to its tight coupling to the Fourier basis.

### 3.3 MatrixKAN
- **Repo:** `MatrixKAN`
- **Technique:** Matrix-parallelized KAN implementation.
- **Analysis:** Restructures KAN computation into batched matrix operations. Theoretical complexity: \(O(N^2 L(k^2 + G))\) vs. KAN's \(O(N^2 L(k^2 + kG))\), yielding speedup proportional to spline degree \(k\). For high-degree splines, reported as "many multiples, if not orders of magnitude, faster than KAN."

### 3.4 Linear-Time B-splines KAN (LTBs-KAN)
- **Paper:** arXiv:2604.22034
- **Technique:** Computes B-spline coefficients via adjusted Bernstein-Bézier form using parallel recurrence relations.
- **Analysis:** Targets the specific inefficiency of De Boor-Cox recursion. Claims linear-time basis evaluation rather than the typical \(O(kG)\) per point. PyTorch-based with GPU acceleration. Novel but limited community adoption so far.

### 3.5 EfficientKAN / PointKAN-Elite
- **Paper:** PointKAN-elite, 2025
- **Technique:** Uses **rational functions** with CUDA implementation and **Horner's method** for polynomial evaluation.
- **Analysis:** Horner's method reduces polynomial evaluation from \(O(k^2)\) to \(O(k)\). Combined with rational-function base (P/Q form with sqrt denominator), it avoids B-spline recursion entirely. Custom CUDA kernels for rational evaluation are hardware-friendlier than spline recursion.

### 3.6 JAX-Based Implementations
- **Repos:** `jaxKAN`, `KANX`
- **Technique:** JAX JIT compilation and XLA optimization.
- **Analysis:** JAX's functionally compositional design is well-suited to KANs' univariate function structure. JIT compilation fuses operations automatically, achieving substantial speedups without hand-written CUDA. However, XLA cannot always match hand-tuned PolyKAN kernels for memory-bound basis evaluation.

### 3.7 Julia Implementation
- **Repo:** `KolmogorovArnold.jl`
- **Technique:** Very fast Julia implementation with RBF/RSWAF bases and **custom gradients** that share work between forward and backward passes.
- **Analysis:** The custom gradient sharing is a notable optimization absent from most PyTorch implementations. Julia's multiple dispatch enables efficient code generation for different bases.

### 3.8 KAN 2.0 / pykan Efficiency Improvements
- **Paper:** Liu et al., 2024 (KAN 2.0)
- **Improvements:**
  - Skip symbolic branch when unnecessary (avoids non-parallelizable double loops).
  - Save intermediate activations only when needed for plotting.
  - GPU compatibility added.
  - `model.speed()` one-line enablement.
- **Impact:** Reduced training of `[4,100,100,100,1]` from ~1 day on CPU to **20s on CPU, <1s on GPU** for 100 Adam steps.
- **Caveat:** Still "lag behind MLPs in efficiency, especially at large scales."

---

## 4. Compression, Pruning, and Architectural Efficiency

### 4.1 MetaCluster / MetaFastKAN
- **Paper:** arXiv:2510.19105
- **Technique:** After training, clusters learned edge functions into groups (e.g., 16 clusters), then fine-tunes. Uses a meta-learner to predict cluster centroids.
- **Claimed Benefit:** Deep compression with minimal accuracy loss.
- **Analysis:** Applied to KAN (B-spline), FastKAN (RBF), and GramKAN. Compression is post-hoc; training cost remains high. The meta-learner adds parameters but enables aggressive clustering. Most useful for **deployment** (inference compression), not training acceleration.

### 4.2 LeanKAN: Parameter-Lean Alternative to MultKAN
- **Paper:** Koenig et al., MIT, 2025; arXiv:2502.17844
- **Motivation:** MultKAN (KAN 2.0) introduces multiplication nodes but suffers from **parameter bloat** and extraneous activations.
- **Technique:** Merges addition and multiplication directly within each output node, eliminating the bulky sublayer structure of MultKAN.
- **Results:**
  - Same parameter count as standard AddKAN (vs. MultKAN's inflation).
  - **2.7× fewer parameters** than MultKAN for equivalent converged loss.
  - **200× lower converged loss** when parameter counts are matched.
  - Faster convergence, simpler hyperparameters (only 1 new hyperparameter vs. MultKAN's 2).
- **Significance:** Best practice for KAN 2.0-style multiplication nodes. Should replace MultKAN in most use cases.

### 4.3 Pruning and Sparsity
- **Original KAN:** Includes native \(\ell_1 + \text{entropy}\) pruning to remove inactive edges.
- **DropKAN:** Dropout-based regularization for KANs.
- **LoTRA:** Tensor-rank compression for KAN weight tensors.
- **Analysis:** Pruning is effective for interpretability (removing visual clutter) but its FLOP reduction is limited unless structural pruning removes entire input-output connections. Most KAN implementations do not exploit sparsity at the hardware level (e.g., sparse GEMM).

### 4.4 Hybrid KAN-MLP Architectures
- **Strategy:** Use KAN layers only where interpretability or edge-level adaptivity is needed; use MLP layers elsewhere.
- **Examples:**
  - **HPKM-PINN** (2025): Parallel KAN + MLP branches with learnable fusion weight \(\xi\).
  - **TSKANMixer** (2025): KAN layer inside TSMixer for time series; training ~50× slower than base TSMixer.
  - **KANsformer / ViKANformer:** KAN replaces MLP blocks in Vision Transformers.
- **Analysis:** Hybrids are the pragmatic path for scaling KANs to large problems. However, the KAN component remains the training bottleneck.

---

## 5. Fair Benchmarking: What Do We Actually Know?

### 5.1 KANbeFair (Yu et al., NUS, 2024)
The most rigorous fair comparison to date. Key findings:

| Domain | Parameter-Controlled | FLOP-Controlled |
|--------|---------------------|-----------------|
| **Symbolic Regression** | KAN wins (7/8 datasets) | Roughly equivalent |
| **Machine Learning** | MLP wins (6/8 datasets) | MLP wins |
| **Computer Vision** | MLP wins consistently | MLP wins consistently |
| **NLP** | Mixed (CoLA close) | MLP wins |
| **Audio** | MLP wins | MLP wins |
| **Continual Learning** | KAN forgets more severely | — |

**Critical Insight:** KAN's advantage in symbolic regression is **mainly due to B-spline activation**, not the KAN architecture per se. When B-splines are applied to MLPs, MLP performance in symbolic regression improves to match or exceed KAN.

### 5.2 FPGA Hardware Study (Tran et al., 2024)
- KANs require **more LUTs and DSPs** than MLPs for the same task.
- Latency: **1.1–2.27× slower** than MLP.
- Power-Delay Product: MLP is better in all tested cases.
- **Conclusion:** "MLPs remain a significantly superior option compared to KAN when dealing with classification problems on hardware."

### 5.3 ACM Computing Surveys (2025)
- Identifies **lack of standardized benchmarks** as a core challenge.
- Notes that many KAN variants are evaluated on "small-scale or synthetic datasets with limited comparability."
- Acknowledges KANs' interpretability and data efficiency but emphasizes scalability as an unresolved issue.

---

## 6. Synthesis: A Decision Framework

Based on the surveyed literature, the following practical guidance emerges:

### 6.1 When to Use KANs At All
1. **Symbolic regression / scientific law discovery:** KANs are genuinely superior.
2. **Low-dimensional PDE solving:** Competitive with PINNs, especially with physics-informed losses.
3. **Interpretability is mandatory:** Edge-level visualization and symbolic extraction are unique KAN capabilities.
4. **High-dimensional unstructured data (ImageNet, LLMs):** MLPs are decisively better under fair FLOP budgets.

### 6.2 Choosing an Efficient KAN Variant

| Priority | Recommended Variant | Rationale |
|----------|---------------------|-----------|
| **Maximum speed, MLP-like training** | **PowerMLP** | 40× faster than KAN; theoretically equivalent expressivity |
| **Speed + interpretability balance** | **ReLU-KAN / HRKAN** | Non-iterative, GPU-friendly, preserves inspectable piecewise shapes |
| **Fastest "true KAN" (edge-wise functions)** | **FastKAN (RBF)** | Mature ecosystem, avoids spline recursion, basis for MetaCluster |
| **Polynomial problems** | **ChebyKAN + PolyKAN kernels** | Strong speed-accuracy; PolyKAN provides 1.2–10× inference boost |
| **Periodic problems** | **FourierKAN / KAF** | Natural frequency match; KAF has lowest FLOPs among true KANs |
| **Multi-scale problems** | **Wav-KAN** (with caution) | Structural match for wavelet features; verify BN isn't the main factor |
| **Multiplication nodes needed** | **LeanKAN** | Replaces MultKAN; 2.7× parameter efficiency |
| **Hardware deployment (FPGA/ASIC)** | **ReLU-KAN or MLP** | B-spline/RBF evaluation is hardware-unfriendly; rational/Chebyshev with LUTs is better but still costly |

### 6.3 Optimization Checklist
1. **Profile first:** Confirm that basis evaluation (not memory bandwidth or data loading) is the actual bottleneck.
2. **Prefer non-iterative bases:** ReLU-powers, RBFs, and polynomials with Horner/LUT evaluation are vastly faster than B-spline recursion.
3. **Use fused kernels:** PolyKAN-style fused CUDA operators provide 1.4–12× training speedups for polynomial variants.
4. **Enable framework optimizations:** JAX JIT, `model.speed()` in pykan, and GPU batching all matter.
5. **Consider hybrids:** Replace only the most interpretability-critical layers with KAN; use MLPs elsewhere.
6. **Apply post-training compression:** MetaCluster clustering reduces inference footprint but does not help training cost.
7. **Match basis to problem structure:** Do not use FourierKAN for non-periodic data or Wav-KAN for smooth monotonic functions.

---

## 7. Open Problems and Future Directions

1. **Sparse KAN execution:** No published work exploits structured sparsity in KAN edge functions using sparse tensor cores or sparse GEMM.
2. **Quantization:** 8-bit or 4-bit quantization of spline coefficients or polynomial coefficients is underexplored.
3. **Tensor-parallel training:** For large KAN layers, model parallelism across edge-function groups is unexplored.
4. **Adaptive basis switching:** Automatically selecting the cheapest basis family per-layer or per-edge based on local function complexity.
5. **Hardware co-design:** Custom accelerators for KAN-specific operations (basis lookup, univariate function evaluation) could close the MLP gap.
6. **Standardized benchmarking:** A community-wide benchmark suite controlling parameters, FLOPs, wall-clock time, and memory is urgently needed.
7. **Theoretical FLOP lower bounds:** What is the minimum FLOP cost to achieve KAN-level expressivity? PowerMLP suggests MLP-like costs are achievable, but general lower bounds are unknown.

---

## 8. Conclusion

The KAN optimization landscape is rapidly evolving but fragmented. Hundreds of basis-function variants and a growing body of hardware-optimization papers have narrowed the efficiency gap with MLPs, yet **no KAN variant has established a clear efficiency *frontier* that dominates MLPs under fair conditions**. The most credible paths forward are:

- **PowerMLP** for practitioners who need KAN-like theory with MLP speed.
- **PolyKAN + ChebyKAN** for scientific computing users who need true edge-wise functions and have GPU resources.
- **LeanKAN + ReLU-KAN** for interpretability-first applications where training cost must be contained.
- **Wav-KAN** only when multi-scale structure is a priori known and independent verification confirms gains.

The original KAN paper's vision of replacing MLPs universally remains **unrealized** and, based on current fair benchmarking, **unlikely to materialize** for high-dimensional unstructured data. However, in their niche—**interpretable scientific computing and symbolic regression**—optimized KANs have earned a permanent and valuable place.

---

## References (Key Papers & Repositories)

| Citation | Title | Year | Focus |
|----------|-------|------|-------|
| Liu et al. | KAN: Kolmogorov-Arnold Networks | 2024 | Original B-spline KAN |
| Liu et al. | KAN 2.0: Kolmogorov-Arnold Networks Meet Science | 2024 | MultKAN, pykan improvements |
| Yu et al. | KAN or MLP: A Fairer Comparison | 2024 | Fair benchmarking (KANbeFair) |
| Qiu et al. | PowerMLP: An Efficient Version of KAN | 2025 | 40× faster, MLP-type architecture |
| Li | FastKAN | 2024 | RBF-based KAN |
| SS et al. | ChebyKAN | 2024 | Chebyshev polynomial KAN |
| Zhang et al. | Kolmogorov-Arnold Fourier Networks (KAF) | 2025 | RFF-based KAN with FLOP analysis |
| Bozorgasl & Chen | Wav-KAN | 2024 | Wavelet KAN |
| Zhong et al. | PolyKAN: Efficient Fused GPU Operators | 2025 | CUDA kernel fusion for polynomial KANs |
| Koenig et al. | LeanKAN | 2025 | Parameter-lean MultKAN replacement |
| Ta | FC-KAN | 2024 | Function combinations |
| Aghaei | rKAN / fKAN | 2024/2025 | Rational / fractional Jacobi KANs |
| Tran et al. | Limitations of KAN in Classification: Hardware Insights | 2024 | FPGA resource analysis |
| Somvanshi et al. | A Survey on Kolmogorov-Arnold Network | 2025 | Comprehensive ACM survey |
| Practitioner's Guide | A Practitioner's Guide to Kolmogorov-Arnold Networks | 2026 | Decision framework and benchmarking |
| MetaCluster | MetaCluster: Enabling Deep Compression of KAN | 2025 | Clustering-based compression |
| MatrixKAN | Matrix-parallelized KAN | 2024 | Matrix parallelism |
| LTBs-KAN | Linear-Time B-splines KAN | 2026 | Parallel B-spline coefficient computation |

---

*End of Survey*
