# BREAKTHROUGH IDEAS REPORT
## EBM Test-Time Compute Scaling Project — Post-AAI Research Directions

**Date:** 2026-05-12  
**Prepared by:** OutOfBox_Breakthrough_Thinker  
**Context:** KAN-EBM architecture (conv-backbone + per-pixel head) with iterative denoising via gradient descent on learned energy. Core empirical finding: K* ~ C σ^α. The paper is currently weak because (1) the power law is expected from classical theory, (2) universality is tautological, and (3) theory prediction is off by 3.5×.  
**Goal:** Identify unconventional angles that leverage existing infrastructure (code, checkpoints, datasets) to produce genuinely surprising, publishable results.

---

## Ranking Methodology

Each idea is scored on two dimensions (1–10):
- **Breakthrough Potential (BP):** How transformative would a positive result be? Would it change how the field thinks?
- **Feasibility (F):** Can we test it with existing code/checkpoints/datasets within ~1–2 months?

**Final Rank = BP × F** (higher is better). Ties broken by novelty and surprise factor.

---

## TOP 10 BREAKTHROUGH ANGLES (Ranked)

---

### #1 → DIFFUSION STEP SCHEDULE REPLACEMENT (Score: 56 = BP 7 × F 8)

**Concept:** Modern diffusion models (Stable Diffusion, DALL-E) use hand-designed step schedules (uniform, quadratic, cosine). Your law K* ~ C σ^α predicts the *optimal number of steps* for any noise level. Replace the fixed schedule with a data-adaptive schedule: measure the dataset's β (from spectral analysis), predict α, and set K* accordingly. At inference time, dynamically allocate steps based on the input noise level.

**Why It's Novel:**  
- Adaptive noise schedules exist (ANT for time series, DIFFRACT for diffraction patterns), but **no one has derived a step schedule from a learned power-law scaling of optimal stopping time**.
- The key difference: existing methods adapt the *noise level* per step; you would adapt the *number of steps* based on a theoretically-grounded scaling law.
- This directly addresses the diffusion community's obsession with "how many steps do we need?"

**How to Test It:**  
1. Take a pre-trained diffusion model (e.g., a lightweight DDPM on CIFAR-10 or CelebA).
2. Instead of a fixed T=1000 steps, use your K* law: at each noise level σ_t, compute K*(σ_t) = C σ_t^α.
3. Use this as a non-uniform schedule where regions with higher K* get more steps.
4. Compare PSNR/FID vs. standard uniform/cosine schedules at equal total compute.

**Expected Surprising Result:**  
- A 20–40% reduction in steps with equal or better sample quality, because the schedule concentrates compute where the landscape is hardest (high σ).
- The schedule is *dataset-specific* and *learnable*, not hand-designed.

**Risk:**  
- The law might only hold for your specific EBM architecture, not general diffusion models. If it fails, the result is a negative but still informative (the law is architecture-specific).
- Diffusion models have a fixed noise schedule tied to their training; the K* law might not be directly applicable without retraining.

**Infrastructure Needed:**  
- Existing: your KAN-EBM code, denoising pipeline, checkpoint evaluation scripts.
- New: integration with a standard diffusion model (DDPM codebase) or implementing the adaptive schedule within your EBM framework and comparing to baselines.
- Time: ~2 weeks.

---

### #2 → VIDEO/TEMPORAL EXTENSION: DOES α CHANGE? (Score: 56 = BP 8 × F 7)

**Concept:** Extend the KAN-EBM to video (2D+time). Video has a temporal power spectrum in addition to spatial. If you corrupt a video with spatiotemporal Gaussian noise, does the optimal stopping time still follow a power law? Does the temporal dimension change α? Could you predict K* from the spatial+temporal spectral exponent β_s + β_t?

**Why It's Novel:**  
- **There is zero existing literature on EBM denoising scaling laws for video.** This is a complete blue-ocean question.
- Video denoising is a huge practical problem (surveillance, medical imaging, film restoration).
- The temporal dimension introduces a fascinating theoretical question: does the law become K* ~ C σ^(α_s + α_t) or is there a coupling term?

**How to Test It:**  
1. Use a simple video dataset (e.g., Moving MNIST, KTH actions, or Vimeo-90K).
2. Extend the KAN backbone to 3D convolutions or use a 2D+time factorized architecture.
3. Train the EBM on video denoising (same DSM objective).
4. Measure K*(σ) for different temporal and spatial noise levels.
5. Test whether α differs from the image case and whether the temporal spectral exponent β_t predicts it.

**Expected Surprising Result:**  
- α_video < α_image (faster convergence because temporal redundancy provides extra information), OR α_video > α_image (temporal correlations slow down per-frame convergence).
- The temporal spectral exponent β_t independently predicts a shift in C or α.
- A new "spatiotemporal scaling law" emerges: K* ~ C σ^α f(β_t) for some function f.

**Risk:**  
- Video training is computationally expensive and may be unstable.
- The temporal dynamics might not follow a clean power law at all, making the story less clean.

**Infrastructure Needed:**  
- Existing: EBM training code, spectral analysis scripts.
- New: 3D conv backbone or 2D+time architecture, video data loaders.
- Time: ~3–4 weeks (training is the bottleneck).

---

### #3 → MULTI-TASK ZERO-SHOT: ONE K* FOR ALL DEGRADATIONS (Score: 56 = BP 7 × F 8)

**Concept:** Your model is trained on denoising. Can it generalize to deblurring, super-resolution, and inpainting *with zero-shot adaptation* — not just "does the law hold" but "can the same model with the same K* law handle multiple tasks?" The key is that these tasks can be written as linear inverse problems y = Ax + noise. If the EBM's energy is a good prior, gradient descent on the posterior energy E(x) + λ||y - Ax||² should still follow a K* scaling law, but with the *effective* noise level now depending on the conditioning of A.

**Why It's Novel:**  
- Zero-shot restoration with diffusion/EBMs is active (DPS, DDRM, CM4IR), but **no one has asked whether the optimal stopping time scales in the same way across tasks**.
- The surprising claim would be: "The same K* law, with the same α, predicts optimal stopping for denoising, deblurring, and SR — the only thing that changes is the effective σ."
- This would make your law a universal *task-agnostic* compute-quality tradeoff.

**How to Test It:**  
1. Take your trained denoising EBM.
2. For deblurring: apply a Gaussian blur, then run gradient descent on the posterior. Measure K* for different blur kernels and noise levels.
3. For super-resolution: downsample by 2× or 4×, add noise, run posterior gradient descent. Measure K*.
4. For inpainting: mask 50% of pixels, run posterior gradient descent. Measure K*.
5. Fit the power law for each task and compare C, α.

**Expected Surprising Result:**  
- α is approximately *invariant* across tasks (same exponent), but C changes because the effective noise level depends on the degradation operator A.
- This means: given a new degradation, you only need to estimate its effective σ to predict K* — no retraining needed.
- A single model achieves competitive results on 4 tasks, with compute predicted by your law.

**Risk:**  
- The posterior landscape for inverse problems may have multiple local minima, making K* ill-defined.
- Zero-shot results may be worse than task-specific methods, limiting the impact.

**Infrastructure Needed:**  
- Existing: trained checkpoints, evaluation scripts, linear inverse problem solvers (easily added).
- New: degradation operators (blur kernels, downsampling, masks).
- Time: ~2 weeks.

---

### #4 → INFORMATION GEOMETRY: GEODESIC DISTANCE PREDICTS K* BETTER (Score: 48 = BP 8 × F 6)

**Concept:** The energy landscape defines a Riemannian metric via the Hessian: g = ∇²E. The geodesic distance under this metric between the noisy initialization and the attractor may predict K* better than the simple power law. The classical theory predicts K* ~ σ²/μ, but your empirical law has a different α and a 3.5× offset. The gap might be explained by the *curvature* of the energy landscape: the true distance is not Euclidean but geodesic. This connects to Amari's information geometry and natural gradient descent.

**Why It's Novel:**  
- A very recent paper (2025, "Riemannian Metrics from Energy-Based Models") derives Riemannian metrics from EBMs, but **does not connect them to optimal stopping time or convergence rates**.
- The classical result K* ~ σ²/μ assumes a quadratic energy (Euclidean metric). If the landscape is curved, the geodesic distance is larger, explaining the 3.5× gap.
- This would turn the 3.5× "error" into a **signature of non-Euclidean geometry**.

**How to Test It:**  
1. Compute the Hessian ∇²E at the attractor (or along the denoising trajectory) for your trained EBM.
2. Compute the Riemannian metric g = ∇²E (or a regularized version).
3. Approximate the geodesic distance from the noisy initialization to the clean image using the metric.
4. Test whether geodesic distance / step size predicts K* more accurately than the Euclidean theory.
5. Derive a modified theoretical prediction: K* ≈ d_geodesic / (μ * step_size) and compare to empirical K*.

**Expected Surprising Result:**  
- The geodesic distance prediction reduces the 3.5× gap to <1.5×.
- The α from the power law is close to 2 (the Euclidean prediction) but corrected by a curvature-dependent term: α = 2 + κ, where κ is the average Ricci curvature along the trajectory.
- This establishes a **deep geometric foundation** for the scaling law.

**Risk:**  
- Computing Hessians for high-dimensional images is expensive (need low-rank approximations).
- The geodesic distance may be hard to compute accurately.
- The theory might not explain the gap, leaving the project without a clear result.

**Infrastructure Needed:**  
- Existing: trained EBM, evaluation pipeline.
- New: Hessian-vector product code, geodesic computation (can use SciPy / PyTorch autograd).
- Time: ~3–4 weeks.

---

### #5 → TRAIN THE MODEL TO HAVE A SPECIFIC α (Score: 45 = BP 9 × F 5)

**Concept:** The standard Denoising Score Matching (DSM) objective does not care about the inference scaling. What if you add a regularization term that penalizes the deviation from a target α? Or train the energy landscape to have controlled curvature, which directly controls the convergence rate? This is the inverse problem to "measure α" — you "design α."

**Why It's Novel:**  
- No one has proposed **training an EBM to have a specific power-law convergence exponent**.
- This would be the first "inference-aware training objective" for EBMs: you don't just care about denoising quality at a fixed K, you care about *how quickly* quality improves with K.
- If you can train α to be, say, 1.0 instead of 2.0, you get a model where doubling σ only doubles K* instead of quadrupling it — a massive compute saving at high noise.

**How to Test It:**  
1. Define a target α_target (e.g., 1.0, 1.5).
2. During training, periodically evaluate K*(σ) for a small validation set at multiple noise levels.
3. Fit the power law and compute α_empirical.
4. Add a regularization term: L_total = L_DSM + λ |α_empirical - α_target|.
5. Alternatively, regularize the energy landscape's curvature directly (e.g., penalize the ratio of largest to smallest eigenvalue of the Hessian).
6. Train until convergence and compare the new α, C, and denoising quality.

**Expected Surprising Result:**  
- α is indeed trainable and can be pushed from ~2.0 down to ~1.2 with only a small loss in denoising quality.
- The model with a smaller α achieves the same quality at high noise with 50% fewer steps.
- A new design principle emerges: "EBMs can be designed for fast inference by controlling the energy landscape's curvature."

**Risk:**  
- High risk: the α might be a fundamental property of the data distribution that cannot be changed by training.
- The regularization might make training unstable or degrade denoising quality severely.
- Computing α during training is expensive (requires measuring K* on validation data).

**Infrastructure Needed:**  
- Existing: training code, K* measurement pipeline.
- New: modified loss function with α regularization, efficient K* estimation during training.
- Time: ~4–6 weeks (training is the bottleneck).

---

### #6 → ADVERSARIAL ROBUSTNESS: K* UNDER ADVERSARIAL PERTURBATIONS (Score: 42 = BP 7 × F 6)

**Concept:** If you add adversarial perturbations instead of Gaussian noise, does K* still scale predictably? The EBM's energy landscape is shaped by the training data. Adversarial perturbations are designed to push inputs to low-energy but wrong regions. Could the number of steps needed to "clean" an adversarial example be predicted by your law? Could the EBM's energy landscape provide a certified defense because the attractor basin is large?

**Why It's Novel:**  
- EBMs for adversarial robustness exist (Yin et al. 2022, DAT), but **no one has studied how the optimal inference depth K* scales with adversarial perturbation strength**.
- The surprising hypothesis: adversarial perturbations of strength ε behave like Gaussian noise of strength σ_eff(ε) in terms of K*, making your law a universal "difficulty metric" for both natural and adversarial corruptions.
- If K* increases monotonically with ε, the EBM can *detect* adversarial examples by measuring how many steps it takes to converge.

**How to Test It:**  
1. Generate adversarial examples on your test set using PGD or FGSM (targeting the denoising output).
2. Run your EBM denoising on adversarially perturbed images.
3. Measure K*(ε) for different perturbation strengths ε.
4. Test if K* follows a power law in ε.
5. Compare: if you know ε, can you predict K*? If you measure K*, can you estimate ε?

**Expected Surprising Result:**  
- K* scales as K* ~ C' ε^α' with a similar α to the Gaussian case but a larger C.
- The EBM's iterative denoising can "undo" adversarial perturbations up to a critical ε, beyond which it gets stuck in a spurious attractor.
- This critical ε can be predicted from the energy landscape's geometry (Hessian at the attractor).

**Risk:**  
- Adversarial attacks on EBMs are less studied; the attack might be trivially powerful or trivially weak.
- The result might be negative: the EBM provides no robustness beyond standard denoising.

**Infrastructure Needed:**  
- Existing: trained EBM, denoising pipeline.
- New: adversarial attack code (PGD/FGSM on pixel space), robustness evaluation metrics.
- Time: ~2–3 weeks.

---

### #7 → ARCHITECTURE SEARCH: MAXIMIZE C, MINIMIZE α (Score: 42 = BP 7 × F 6)

**Concept:** Use the scaling law as an architecture search objective. The law says K* ~ C σ^α. For a fixed σ, a smaller C means faster convergence (fewer steps to optimal). A smaller α means the compute penalty for high noise grows more slowly. Design a Neural Architecture Search (NAS) objective that directly optimizes C and α instead of just PSNR.

**Why It's Novel:**  
- NAS for EBMs exists (energy-aware NAS, but that means hardware energy, not energy landscape).
- **No one has used the power-law parameters (C, α) as differentiable architecture search objectives.**
- This would produce architectures that are explicitly designed for "fast inference at high noise."

**How to Test It:**  
1. Define a search space: backbone depth, KAN layer configurations, per-pixel head architecture.
2. For each architecture candidate, quickly estimate C and α by training for a few epochs and measuring K* at 3–5 noise levels.
3. The search objective: maximize PSNR - λ₁ C - λ₂ α (or similar).
4. Use a lightweight search method (random search with early termination, or DARTS-style).

**Expected Surprising Result:**  
- Architectures with skip connections that preserve high-frequency gradients have smaller α.
- Deeper KAN layers reduce C but increase α — there is a Pareto frontier.
- A discovered architecture achieves the same PSNR with 30% fewer inference steps at high noise.

**Risk:**  
- NAS is computationally expensive even with proxy tasks.
- The search space might be too small to find meaningful improvements.
- C and α may be noisy estimates early in training, making the search unstable.

**Infrastructure Needed:**  
- Existing: training code, K* measurement.
- New: NAS framework (can start with simple random search), architecture search space definition.
- Time: ~4–6 weeks.

---

### #8 → UNCERTAINTY QUANTIFICATION: HESSIAN AS CONFIDENCE (Score: 42 = BP 6 × F 7)

**Concept:** The EBM gives not just a denoised image but an energy landscape. The Hessian at the attractor quantifies local curvature: high curvature = sharp minimum = confident prediction. Low curvature = flat minimum = uncertain. The K* scaling might be related to the uncertainty landscape: images that require more steps might be those where the EBM is less confident.

**Why It's Novel:**  
- EBMs for uncertainty quantification are underexplored compared to Bayesian neural networks or ensembling.
- The connection between *optimal stopping time* and *predictive uncertainty* has not been made in the literature.
- This would give your EBM a built-in "uncertainty-aware stopping" mechanism: stop when the uncertainty is low enough, not just when the MSE stops improving.

**How to Test It:**  
1. Compute the Hessian eigenvalues at the final denoised output for each test image.
2. Measure the uncertainty metric: U = 1 / Tr(∇²E) (or use the smallest eigenvalue).
3. Test correlation: does K* correlate with U? (Images with high U need more steps.)
4. Propose an uncertainty-aware stopping criterion: stop when the ratio of uncertainty change per step drops below threshold.
5. Compare to the MSE-based K* — does uncertainty-based stopping match or improve it?

**Expected Surprising Result:**  
- A strong positive correlation: K* ~ γ log(U) for some γ.
- Uncertainty-based stopping predicts the same K* as MSE-based stopping but uses only the energy landscape, no ground truth needed.
- This enables **unsupervised** stopping: you don't need clean images to know when to stop.

**Risk:**  
- The Hessian might be too expensive to compute at every pixel (but low-rank approximations exist).
- The correlation might be weak, making the result less exciting.

**Infrastructure Needed:**  
- Existing: trained EBM, test images.
- New: Hessian eigenvalue computation at attractor.
- Time: ~2 weeks.

---

### #9 → META-LEARNING: PREDICT K* FOR NEW DATASETS (Score: 36 = BP 6 × F 6)

**Concept:** Train on CIFAR-10, CelebA, BSD500. Learn a meta-model that predicts (C, α) for a new dataset from a few calibration images. The meta-model inputs: the empirical spectral exponent β of the calibration set, plus the dataset's image statistics. The output: predicted C and α. This is a meta-learning or few-shot learning framing of the predictive α problem.

**Why It's Novel:**  
- Meta-learning for early stopping exists (Guiroy et al., 2022), but **not for EBM inference depth**.
- The meta-model would be a tiny MLP mapping dataset statistics to (C, α).
- Once trained, you can deploy your EBM on a new dataset with zero tuning: just measure β from 10 images and predict K*.

**How to Test It:**  
1. Collect C, α values for 5–10 datasets (some held out).
2. For each dataset, compute statistics: β, mean variance, image resolution, color channel correlation.
3. Train a small meta-MLP to predict (C, α) from these statistics.
4. On a held-out dataset, measure β from 10 images, predict (C, α), run denoising with predicted K*.
5. Compare to oracle K* (measured from the full test set).

**Expected Surprising Result:**  
- The meta-MLP achieves <10% relative error on (C, α) with just 10 calibration images.
- A new dataset can be denoised optimally without any hyperparameter search.
- The meta-model reveals that β is the dominant predictor, confirming your theoretical insight.

**Risk:**  
- The meta-model might overfit to the training datasets.
- If the relationship between β and α is deterministic, the meta-model is unnecessary and the story is weaker.

**Infrastructure Needed:**  
- Existing: trained EBMs on multiple datasets, K* measurements, β computation.
- New: meta-MLP training code, cross-dataset evaluation protocol.
- Time: ~2–3 weeks.

---

### #10 → FREE ENERGY PRINCIPLE: OPTIMAL STOPPING AS FREE ENERGY MINIMIZATION (Score: 35 = BP 7 × F 5)

**Concept:** View EBM inference as minimizing a variational free energy F = E - H/β, where E is the learned energy and H is the entropy of the posterior. The optimal stopping time K* might be the point where the free energy is minimized, which could be predicted from the entropy of the posterior. This connects to Friston's Free Energy Principle (FEP) and predictive coding literature.

**Why It's Novel:**  
- The FEP is a huge literature in neuroscience, but its application to image denoising is sparse and often hand-wavy.
- The rigorous connection would be: **in an EBM, the posterior entropy H[p(x|y)] can be approximated from the Hessian determinant at the attractor, and the free energy minimum predicts K*.**
- This would turn the FEP from a philosophical framework into a quantitative prediction tool for ML.

**How to Test It:**  
1. Compute the posterior entropy at each step of the denoising trajectory: H ≈ 0.5 log det(∇²E) + const.
2. Compute the free energy F(K) = E(x_K) - T * H(x_K) for a virtual temperature T.
3. Find the K where F(K) is minimized.
4. Compare this K_FEP to the empirically optimal K* (from MSE).
5. Derive a theoretical prediction: K* = argmin_F(K) ≈ C σ^(α_FEP) and compare exponents.

**Expected Surprising Result:**  
- The free energy minimum occurs at a K that closely matches the MSE-optimal K*.
- The FEP explains the power law: as σ increases, the posterior entropy grows, which shifts the free energy minimum to larger K.
- A unified view: "Optimal stopping is free energy minimization at a temperature set by the noise level."

**Risk:**  
- The FEP connection might be seen as "hand-waving" or "neuroscience-washing" unless the math is extremely rigorous.
- The entropy approximation might be too crude.
- The reviewers might be skeptical of FEP-based claims.

**Infrastructure Needed:**  
- Existing: trained EBM, denoising trajectory data.
- New: Hessian determinant computation, free energy formula implementation.
- Time: ~2–3 weeks.

---

## HONORABLE MENTIONS (Outside Top 10)

### 11. Online / Continual Denoising
- Stream images with varying noise, adapt K* on the fly using online regression.
- Novel but limited impact; the meta-learning angle (#9) is stronger.
- **BP: 5, F: 6, Score: 30**

### 12. Neuroscience / Biological Inspiration
- Power law as signature of biological predictive coding.
- Too speculative; hard to test without biological data.
- **BP: 6, F: 3, Score: 18**

---

## STRATEGIC RECOMMENDATIONS

### Immediate Action (Next 2 Weeks)
1. **Start with Angle #3 (Multi-Task)** and Angle #1 (Schedule Replacement) in parallel. Both require minimal new infrastructure and can be done with existing checkpoints.
2. If either shows a positive result, it immediately strengthens the AAAI paper by showing the law is *useful*, not just observed.

### Medium-Term (2–6 Weeks)
3. If multi-task works, **pursue Angle #2 (Video)** — this is the biggest untapped question and could be a standalone follow-up paper.
4. If the 3.5× gap bothers you, **pursue Angle #4 (Information Geometry)** — it turns the gap into a geometric discovery.

### High-Risk / High-Reward (3+ Months)
5. **Angle #5 (Train α)** is the most transformative if it works. It would be a paradigm shift in EBM training. But it's also the riskiest. Consider it only if you have time for a full retraining cycle.

### The Narrative Pivot
The current paper's weakness is: "We observed a power law that theory predicts."  
The breakthrough narrative is: **"The power law is a signature of something deeper. We can use it to predict, design, and generalize."**

- **Predict:** Schedule replacement (#1), meta-learning (#9)
- **Design:** Architecture search (#7), train α (#5)
- **Generalize:** Multi-task (#3), video (#2), adversarial (#6)
- **Explain:** Information geometry (#4), free energy (#10)

Pick at least one from each category and the paper becomes a research program, not a single observation.

---

## LITERATURE MAP SUMMARY

| Angle | Existing Work | Gap |
|-------|---------------|-----|
| #1 Schedule Replacement | ANT (NeurIPS 2024), DIFFRACT (ICCV 2025) | No one uses learned K* law to set step count |
| #2 Video Extension | RTE-VD (embedded), general video denoising | No EBM scaling laws for video |
| #3 Multi-Task | DPS, DDRM, CM4IR (CVPR 2025), DiffPIR | No one studies K* scaling across tasks |
| #4 Info Geometry | Riemannian Metrics from EBMs (2025) | No connection to optimal stopping |
| #5 Train α | DSM, adversarial training for EBMs | No inference-aware training objective |
| #6 Adversarial | DAT (ECCV 2022), Scalable EBMs (2026) | No K* scaling with adversarial strength |
| #7 NAS | Energy-aware NAS (ICCAD 2019) | No NAS for inference-depth scaling |
| #8 Uncertainty | Hessian-based UQ in NNs | No connection to K* |
| #9 Meta-Learning | ABE (Guiroy 2022) for early stopping | No meta-learning for EBM inference depth |
| #10 Free Energy | Friston (2023), diffusion FEP (2025) | No quantitative prediction of K* from FEP |

---

*End of Report*
