#!/usr/bin/env bash
# Colab/remote batch v2: post-review experiments (panel 2026-07-17).
# Run from the repo root AFTER (or alongside) remote_run.sh -- all v1
# outputs are skip-if-done, so running both in sequence is safe and cheap.
# Stages:
#   1. Train replication heads (primary pair x training seeds 1,2)
#      + scale ladder (1M / 8M / 30M ConvMLP-GELU), matched recipe.
#   2. Frontier sweep for the 7 new models (70 cells, resumable watchdog).
#   3. Wall-clock timing at batch {1,32,256} (latency scoping).
#   4. Oracle-gain analysis (per-image adaptive-depth ceiling in dB).
#   5. Blur-trained control (EXPLORATORY, last -- fine to kill).
# Expected wall time on A100: ~2-4 h total.
set -uo pipefail

PY=${PY:-python3}
LOG=remote_run2.log
echo "[v2] start $(date)" | tee -a "$LOG"

$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$LOG"

# 1) Training (skip-if-done per checkpoint)
$PY experiments/train_replication_ladder.py --device cuda 2>&1 | tee -a "$LOG"

# 2) Sweep the 7 new models: 7 x 5 sigmas x 2 seeds = 70 cells. Resumable.
V2_MODELS="kan_32k_ts1 kan_32k_ts2 kan_110k_ts1 kan_110k_ts2 ladder_1m ladder_8m ladder_30m"
count_v2() {
  $PY - <<'EOF'
import json, pathlib
tags = ('kan_32k_ts1','kan_32k_ts2','kan_110k_ts1','kan_110k_ts2',
        'ladder_1m','ladder_8m','ladder_30m')
n = 0
for p in pathlib.Path('outputs/inference_frontier/parts').glob('*.json'):
    try:
        d = json.loads(p.read_text())
        if d.get('done') and d.get('model') in tags:
            n += 1
    except Exception:
        pass
print(n)
EOF
}
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
mkdir -p outputs/inference_frontier/parts
for attempt in $(seq 1 12); do
  clean_truncated
  n=$(count_v2)
  echo "[v2] sweep attempt $attempt, new parts done: $n/70" | tee -a "$LOG"
  [ "$n" -ge 70 ] && break
  timeout -k 30 2700 $PY experiments/exp_inference_frontier.py --device cuda \
    --models $V2_MODELS 2>&1 | tee -a "$LOG"
done
clean_truncated
echo "[v2] sweep finished with $(count_v2)/70 new parts" | tee -a "$LOG"

# 3) Wall-clock timing (all 13 models present at this point)
$PY experiments/frontier_timing.py --device cuda --models \
  kan_110k kan_32k conv_mlp_gelu unet group_kan_8k group_kan_32k \
  ladder_1m ladder_8m ladder_30m 2>&1 | tee -a "$LOG"

# 4) Oracle gain (CPU-crashy locally; trivial here)
if [ ! -f outputs/review_fixes/oracle_gain.json ]; then
  $PY theory/oracle_gain.py --device cuda 2>&1 | tee -a "$LOG"
fi

# 5) Blur-trained control (exploratory; last on purpose)
if [ ! -f outputs/blur_law/blur_law.json ]; then
  $PY experiments/exp_blur_law.py --device cuda 2>&1 | tee -a "$LOG"
fi

echo "[v2] DONE $(date)" | tee -a "$LOG"
echo "[v2] Package: tar czhf results_back2.tar.gz outputs/inference_frontier outputs/replication outputs/scale_ladder outputs/review_fixes outputs/blur_law remote_run2.log"
