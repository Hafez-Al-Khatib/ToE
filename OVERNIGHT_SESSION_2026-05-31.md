# KAN-EBM AAAI 2027 — Overnight Session Log

**Date:** 2026-05-31 (Sunday, dawn, ~4 AM Beirut)
**Status:** Zenith going to sleep. Full CIFAR-10 eval queued.
**GPU:** RTX 3080, ~8GB free after leaked memory

---

## What Is Running Tonight

### Task 1: Full CIFAR-10 Evaluation (10,000 images)

**Script:** `experiments/exp_scaled_eval_cifar10_full.py`
**Models evaluated (9 total):**
1. KAN-EBM-110K (`outputs/cifar10/kan_ebm_f32.pt`)
2. KAN-EBM-32K (`outputs/finalization/kstar_param_scaling/kan_small.pt`)
3. ConvMLP-GELU (`outputs/finalization/rebuttal/conv_mlp_gelu.pt`)
4. ConvMLP-SiLU (`outputs/finalization/rebuttal/conv_mlp_silu.pt`)
5. ConvMLP-Tanh (`outputs/finalization/rebuttal/conv_mlp_tanh.pt`)
6. ConvMLP-ReLU (`outputs/finalization/rebuttal/conv_mlp_relu.pt`)
7. U-Net-EBM (`outputs/unet_ebm/unet_ebm.pt`, 233K params)
8. FFN-DSM (`outputs/cifar10/ffn_dsm_f32.pt`, single-pass)
9. DnCNN (`outputs/dncnn_baseline/dncnn_cifar10.pt`, 7.9K params, single-pass)

**Parameters:**
- n_images: 10,000 (full CIFAR-10 test set)
- batch_size: 128
- device: cuda
- sigmas: [0.05, 0.1, 0.15, 0.2, 0.3]
- K_list: [1, 2, 5, 10, 20]
- Metrics: PSNR and SSIM for each (σ, K, model) combination

**Estimated runtime:** 4-8 hours (depends on UNetEBM speed — largest model)
**Output:** `outputs/scaled_eval/cifar10_full_10000_<timestamp>.json`

### Task 2: Flow Matching Baseline Training (Queue for after eval)

**Script:** `experiments/exp_flow_matching_baseline.py`
**Status:** NOT YET STARTED — will queue after eval completes
**Target:** ~45K param MicroUNet trained on CIFAR-10 with conditional flow matching
**Estimated runtime:** 6-8 hours (60 epochs on CIFAR-10)

---

## Next Tasks (Tomorrow)

1. **Check eval results** — verify all models completed, no NaNs, power-law fits computed
2. **Start flow matching training** if eval completed successfully
3. **Power-law fitting** on full 10K results — recompute α, C, R² with tighter CIs
4. **CelebA-64 native training** — prepare configs, start training (2-3 GPU days)
5. **GitHub repo cleanup** — create public repo, clean code, write README

---

## Known Issues to Watch

1. **GPU leaked memory (~2.3GB)** — not a problem, 8GB still available, but may limit largest models
2. **U-Net EBM is 233K params** — larger than other models, may be slower per-step
3. **ConvMLP-ReLU checkpoint** — verify it exists and loads correctly
4. **FFN-DSM evaluate_ffn** — currently evaluated at all K values (same result each time). Could optimize but keeping protocol consistent.

---

## Handoff Notes for Next Session

**If eval completed successfully:**
- Parse JSON, compute power-law fits for each model
- Compare to 500-image results: α values should be stable, CIs tighter
- Start flow matching training immediately

**If eval crashed/failed:**
- Check error logs
- Likely causes: OOM (reduce batch_size to 64), checkpoint load failure, CUDA error
- Fix and restart

**If user asks before I check:**
- Check `outputs/scaled_eval/` for latest JSON
- Report completion status, estimated time remaining if still running

---

*Written by Telamon, 2026-05-31 04:00 AM (Asia/Beirut)*
