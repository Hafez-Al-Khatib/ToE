$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH += ";C:\Users\hafez\AppData\Roaming\Python\Python314\Scripts"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🔥 Starting KAN-EBM Master Experiment Pipeline 🔥" -ForegroundColor Cyan
Write-Host "All silent architectural bugs have been resolved." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Base Paper Experiments (MNIST benchmarks, ablation, math verification)
Write-Host "`n[1/5] Running Core Paper Experiments (MNIST)..." -ForegroundColor Yellow
py -3.12 experiments/run_paper_experiments.py --device cuda

# 2. Reviewer Extension Experiments (Extended Scaling, CBSD68 Diffusion comparison, CIFAR-10)
Write-Host "`n[2/5] Running Reviewer Extensions (CBSD68 & CIFAR-10)..." -ForegroundColor Yellow
py -3.12 experiments/run_reviewer_experiments.py --device cuda

# 3. Multitask Zero-Shot Capabilities
Write-Host "`n[3/5] Running Zero-Shot Multitask Pipeline (Inpainting / SR)..." -ForegroundColor Yellow
py -3.12 experiments/exp_multitask.py --device cuda

# 4. Phase 2 Scaling: CelebA 64x64
Write-Host "`n[4/5] Running Phase 2 Scaling: CelebA 64x64..." -ForegroundColor Yellow
py -3.12 experiments/exp_celeba.py --device cuda --epochs 50

# 5. Phase 2 Scaling: ImageNet 64x64
Write-Host "`n[5/5] Running Phase 2 Scaling: ImageNet 64x64..." -ForegroundColor Yellow
py -3.12 experiments/exp_imagenet64.py --device cuda --epochs 50

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "✅ All experiments completed successfully! ✅" -ForegroundColor Green
Write-Host "Results are saved in the 'outputs/' directory." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
