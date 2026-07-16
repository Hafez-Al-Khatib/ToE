# Remote Run: Inference Frontier + GroupKAN (Colab or lab 4090)

One batch generates all remaining experimental data for the paper_v8 AAAI
extension. Everything is resumable — if anything dies, just re-run.

## Option A (recommended): Google Colab — zero setup

Open `colab_run.ipynb` in Colab and `Runtime > Run all` on a GPU runtime:

    https://colab.research.google.com/github/Hafez-Al-Khatib/ToE/blob/feature/hamiltonian-kan/colab_run.ipynb

(If the repo is private, upload `colab_run.ipynb` to Colab manually instead.)

The notebook clones this branch, extracts the committed checkpoint tarball,
symlinks `outputs/` onto your Google Drive (so completed sweep cells survive
runtime disconnects), runs `scripts/remote_run.sh`, and packages
`results_back.tar.gz` for download (with a backup copy in Drive root).
After any disconnect, just Run all again — nothing is redone.

GPU guidance: A100 ~1–3 h, L4 ~2–6 h, T4 ~4–10 h (free tier may need 2–3
sessions; the Drive-backed resume makes that fine).

## Option B: lab machine with a CUDA GPU

### 1. On the lab machine: clone the branch

```bash
git clone -b feature/hamiltonian-kan https://github.com/Hafez-Al-Khatib/ToE.git
cd ToE
pip install lpips   # torch+CUDA, torchvision, numpy, matplotlib assumed present
```

### 2. Extract the pretrained checkpoints

`remote_checkpoints.tar.gz` (~1.5 MB) is committed on this branch, so the
clone already has it. Extract at the repo root (paths are preserved):

```bash
cd ~/ToE && tar xzf remote_checkpoints.tar.gz
```

### 3. What will run (in order, ~1-3 h total on a 4090)

1. Train GroupKAN heads (8K + 32K params, 40 epochs, matched recipe).
2. Frontier sweep: 6 models x 5 noise levels x 2 seeds = 60 cells,
   under a 45-min-per-attempt watchdog (a hung attempt is killed and the
   sweep resumes; completed cells are never redone).
3. GroupKAN fine-grid K* sweeps (the alpha table rows; 3 seeds each).
4. LPIPS secondary metric at each model's best step count.

### 4. Run it (inside tmux/screen or nohup so SSH drops don't kill it)

```bash
nohup bash scripts/remote_run.sh > remote_run_console.log 2>&1 &
tail -f remote_run.log
```

### 5. Copy results back

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
