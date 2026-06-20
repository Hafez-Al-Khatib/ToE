# run_overnight.ps1
# Executes the Parameter Scaling Law ablation for NeurIPS.
# Trains the KAN-EBM at 3 different sizes to prove the K-scaling property
# holds across model scales, cementing the compute-efficiency claim.

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting Overnight Parameter Scaling Run (CIFAR-10)    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Model 1: Nano (approx 16K parameters)
Write-Host "`n[1/3] Training Nano Model (n_filters=8)..." -ForegroundColor Yellow
py -3.12 experiments/exp_cifar10.py --device cuda --epochs 60 --sigma 0.1 --n_filters 8

# Model 2: Base (approx 64K parameters)
Write-Host "`n[2/3] Training Base Model (n_filters=32)..." -ForegroundColor Yellow
py -3.12 experiments/exp_cifar10.py --device cuda --epochs 60 --sigma 0.1 --n_filters 32

# Model 3: Large (approx 256K parameters)
Write-Host "`n[3/3] Training Large Model (n_filters=128)..." -ForegroundColor Yellow
py -3.12 experiments/exp_cifar10.py --device cuda --epochs 60 --sigma 0.1 --n_filters 128

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "   Overnight Run Complete! Check outputs/cifar10/         " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
