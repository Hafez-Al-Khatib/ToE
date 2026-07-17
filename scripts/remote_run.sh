#!/usr/bin/env bash
# Remote GPU batch: inference-frontier + GroupKAN experiments (AAAI paper_v8).
# Run from the repo root on the lab machine (Linux, CUDA GPU, e.g. RTX 4090).
# Safe to re-run: every stage is resumable / skip-if-done.
# Expected wall time on a 4090: roughly 1-3 hours total.
#
# Prereqs: python3 with torch+CUDA, torchvision, numpy, matplotlib, lpips
#          (pip install lpips), and the checkpoint tarball extracted at the
#          repo root (see REMOTE_RUN.md).
set -uo pipefail  # no -e: the sweep watchdog tolerates killed attempts

PY=${PY:-python3}
LOG=remote_run.log
echo "[remote] start $(date)" | tee -a "$LOG"

# 0) Sanity: CUDA + required pretrained checkpoints
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$LOG"
for f in outputs/cifar10/kan_ebm_f32.pt \
         outputs/finalization/kstar_param_scaling/kan_small.pt \
         outputs/finalization/rebuttal/conv_mlp_gelu.pt \
         outputs/unet_ebm/unet_ebm.pt; do
  [ -f "$f" ] || { echo "MISSING checkpoint: $f -- extract remote_checkpoints.tar.gz at the repo root first"; exit 1; }
done

# 1) Train GroupKAN heads (matched recipe, 40 epochs x 2 variants; skip if done)
if [ ! -f outputs/group_kan/group_kan_32k.pt ] || [ ! -f outputs/group_kan/group_kan_8k.pt ]; then
  echo "[remote] training GroupKAN variants..." | tee -a "$LOG"
  $PY experiments/train_group_kan.py --device cuda 2>&1 | tee -a "$LOG"
else
  echo "[remote] GroupKAN checkpoints already present, skipping training" | tee -a "$LOG"
fi

# 2) Frontier sweep: 6 models x 5 sigmas x 2 seeds = 60 cells. Resumable.
#    Watchdog: each attempt capped at 45 min; truncated part files from a
#    killed attempt are removed before resuming.
clean_truncated() {
  $PY - <<'EOF'
import json, pathlib
for p in pathlib.Path('outputs/inference_frontier/parts').glob('*.json'):
    try:
        json.loads(p.read_text())
    except Exception:
        print('[clean] removing truncated part:', p)
        p.unlink()
EOF
}
count_done() {
  $PY - <<'EOF'
import json, pathlib
n = 0
for p in pathlib.Path('outputs/inference_frontier/parts').glob('*.json'):
    try:
        if json.loads(p.read_text()).get('done'):
            n += 1
    except Exception:
        pass
print(n)
EOF
}
mkdir -p outputs/inference_frontier/parts
for attempt in $(seq 1 12); do
  clean_truncated
  n=$(count_done)
  echo "[remote] sweep attempt $attempt, parts done: $n/60" | tee -a "$LOG"
  [ "$n" -ge 60 ] && break
  timeout -k 30 2700 $PY experiments/exp_inference_frontier.py --device cuda \
    --models kan_110k kan_32k conv_mlp_gelu unet group_kan_8k group_kan_32k 2>&1 | tee -a "$LOG"
done
clean_truncated
echo "[remote] sweep finished with $(count_done)/60 parts" | tee -a "$LOG"

# 3) GroupKAN fine-grid alpha (32k = paper row; 8k = secondary)
if [ ! -f outputs/group_kan/fine_grid_group_kan_32k.json ]; then
  $PY experiments/exp_group_kan_kstar.py --device cuda --hidden 640 --tag group_kan_32k 2>&1 | tee -a "$LOG"
fi
if [ ! -f outputs/group_kan/fine_grid_group_kan_8k.json ]; then
  $PY experiments/exp_group_kan_kstar.py --device cuda --hidden 136 --tag group_kan_8k 2>&1 | tee -a "$LOG"
fi

# 4) LPIPS secondary metric (reads the sweep parts; downloads lpips weights once)
$PY experiments/frontier_lpips.py --device cuda 2>&1 | tee -a "$LOG"

echo "[remote] DONE $(date)" | tee -a "$LOG"
echo "[remote] Package results:  tar czf results_back.tar.gz outputs/inference_frontier outputs/group_kan remote_run.log"
