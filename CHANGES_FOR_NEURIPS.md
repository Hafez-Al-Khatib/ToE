# Changes Made for NeurIPS Readiness — Final Status

## Paper Updates (`paper_v3.tex`)

### Text Fixes
- **ImageNet-64 → Tiny-ImageNet-64**: Fixed misleading dataset name in appendix
- **Abstract CI**: Specified "bootstrap 95% CI"
- **Appendix A**: Changed "Algorithm 1" to "Training Details"
- **Limitations**: Added citations for Restormer and SwinIR; added micro-DnCNN to baseline list
- **Bibliography**: Fixed entry types for 7 citations, corrected BM3D title, added note for placeholder `kaem2024`
- **Oracle recovery**: Added note that 99.4% is conservative (exact: 99.84%)
- **K* independence**: Added note about K* quantization and independent fit verification

### New Content
- **Abstract**: Mentions DnCNN baseline and 500-image evaluation with SSIM
- **Introduction (Contributions)**: Added contribution (4) — validation on 500 images and DnCNN comparison
- **New Section 3.4**: "Scaled evaluation and modern baselines"
  - Reports PSNR and SSIM on 500 CIFAR-10 test images
  - Includes micro-DnCNN baseline (~8K params)
  - Shows DnCNN wins on single-pass PSNR at medium noise, but KAN-EBM uniquely scales with K
  - Reports 500-image K* sequence {1,2,5,5,10} and fit alpha=1.29 (consistent with original)
  - New Table `tab:scaled500` with side-by-side comparison

## Reproducibility Infrastructure

- **REPRODUCIBILITY.md**: NeurIPS-style checklist mapping each claim to exact script/command
- **requirements.txt**: Pinned versions (torch==2.10.0, torchvision==0.25.0, numpy==2.4.3, etc.)
- **experiments/reproduce_scaling_law.py**: End-to-end orchestration script

## Statistical Rigor

- **experiments/compute_oracle_recovery.py**: Provenance for 99.4% claim
  - Result: 99.84% (confirms conservative reporting)
  - Saved to `outputs/finalization/rebuttal/oracle_recovery.json`
- **experiments/rebuttal_smooth_mlp_analysis.py**: Independent per-variant fit computation
  - Added note about K* quantization
  - Saved to `smooth_mlp_stats_independent.json`

## Scaled Evaluation (500 images, CIFAR-10)

- **experiments/exp_scaled_eval_cifar10.py**: Full evaluation script
  - Models: KAN-EBM-110K, KAN-EBM-32K, ConvMLP-GELU, FFN-DSM
  - Metrics: PSNR + SSIM
  - Results saved to `outputs/scaled_eval/cifar10_scaled_500.json`
  - **CRITICAL BUG FIX**: Used model's built-in `denoise()` method instead of custom broken inference

### Key Results (500 images)
| sigma | KAN-EBM peak | ConvMLP peak | FFN-DSM | DnCNN |
|-------|-------------|--------------|---------|-------|
| 0.05  | 33.81 @ K=1 | 33.43 @ K=1  | 27.92   | 32.40 |
| 0.10  | 30.35 @ K=2 | 29.69 @ K=2  | 27.32   | 31.35 |
| 0.15  | 28.19 @ K=5 | 26.63 @ K=5  | 26.52   | 29.67 |
| 0.20  | 26.25 @ K=5 | 25.63 @ K=5  | 25.66   | 27.50 |
| 0.30  | 23.54 @ K=10| 22.79 @ K=10 | 23.96   | 23.19 |

- 500-image K* fit: alpha=1.29, C=46.0, R²=0.97 (bootstrap CI [0.94, 1.55])
- Original 24-image fit: alpha=1.53, C=86.7, R²=0.98
- **Conclusion**: Law is robust; exact K* values shift slightly with more data

## Modern Baselines

### Micro-DnCNN (COMPLETE)
- **Script**: `experiments/exp_dncnn_cifar10.py`
- **Architecture**: 5-layer residual CNN, 16 filters, ~8K params
- **Training**: 20 epochs on CIFAR-10, sigma=0.15, GPU (~9 minutes)
- **Results**: Strong single-pass PSNR; outperforms KAN-EBM at medium noise levels
- **Significance**: Addresses reviewer demand for modern CNN baseline

### Diffusion Baseline (INCOMPLETE — CRASHED)
- **Script**: `experiments/exp_diffusion_cifar10.py`
- **Status**: Multiple architecture bugs fixed, but task lost/crashed during training
- **Issue**: U-Net skip connection channel mismatches; may need further debugging
- **Recommendation**: Fix and run overnight, or include as "future work" in rebuttal

## Qualitative Comparisons

- **Script**: `experiments/gen_qualitative_comparison.py`
- **Output**: `outputs/scaled_eval/qualitative_comparison.png`
- Shows clean / noisy / KAN-EBM / ConvMLP-GELU / FFN-DSM side-by-side

## Remaining Gaps
1. **Diffusion baseline**: Needs debugging and training
2. **Theoretical insight**: No explanation for alpha ≈ 1.5
3. **Higher resolution**: No 256×256 experiments
4. **Checker puzzle figure**: Still one sentence, needs table/figure
5. **Calibration guidance**: No procedure for estimating C on new datasets
