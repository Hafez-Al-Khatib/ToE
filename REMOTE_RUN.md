# Remote Run: Inference Frontier + GroupKAN (lab 4090)

One batch generates all remaining experimental data for the paper_v8 AAAI
extension. Everything is resumable — if anything dies, just re-run step 4.

## 1. On the lab machine: clone the branch

```bash
git clone -b feature/hamiltonian-kan https://github.com/Hafez-Al-Khatib/ToE.git
cd ToE
pip install lpips   # torch+CUDA, torchvision, numpy, matplotlib assumed present
```

## 2. Copy the pretrained checkpoints over (they are git-ignored)

From the Windows machine (repo root has `remote_checkpoints.tar.gz`, ~1.7 MB):

```bash
scp "remote_checkpoints.tar.gz" <user>@<lab-host>:~/ToE/
```

On the lab machine, extract at the repo root (paths are preserved):

```bash
cd ~/ToE && tar xzf remote_checkpoints.tar.gz
```

## 3. What will run (in order, ~1-3 h total on a 4090)

1. Train GroupKAN heads (8K + 32K params, 40 epochs, matched recipe).
2. Frontier sweep: 6 models x 5 noise levels x 2 seeds = 60 cells,
   under a 45-min-per-attempt watchdog (a hung attempt is killed and the
   sweep resumes; completed cells are never redone).
3. GroupKAN fine-grid K* sweeps (the alpha table rows; 3 seeds each).
4. LPIPS secondary metric at each model's best step count.

## 4. Run it (inside tmux/screen or nohup so SSH drops don't kill it)

```bash
nohup bash scripts/remote_run.sh > remote_run_console.log 2>&1 &
tail -f remote_run.log
```

## 5. Copy results back

On the lab machine:

```bash
tar czf results_back.tar.gz outputs/inference_frontier outputs/group_kan remote_run.log
```

`scp` it to the Windows repo root, then extract there (Windows 10 has tar):

```powershell
tar xzf results_back.tar.gz
```

Then tell Claude: "remote results are back" — analysis, the pre-registered
decision gate, and the paper edits continue locally from the part files.

## Notes

- The sweep's noise draws are deterministic per (sigma, seed) cell, so the
  remote parts are protocol-identical to the local design.
- If the lab torch version measures UNet FLOPs without the torch 2.5.1
  FlopCounterMode bug, the part files will record flops_method="measured"
  for unet instead of "forward_x2" — both are valid; the analysis reads the
  method per model.
- Do NOT commit anything under outputs/ (repo convention).
