#!/usr/bin/env bash
# Colab/remote batch v3: STL-10 64x64 frontier extension (pre-registered
# 2026-07-18; see .superpowers/sdd/progress.md). Run from repo root AFTER
# remote_run.sh / remote_run2.sh (all stages skip-if-done).
# Expected wall time on A100: ~1-2 h (incl. one-time STL-10 download ~2.5GB).
set -uo pipefail

PY=${PY:-python3}
LOG=remote_run3.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[v3] start $(date)" | tee -a "$LOG"

$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" | tee -a "$LOG"

# 1) Train the STL-64 primary pair (skip-if-done)
$PY experiments/train_stl64.py --device cuda 2>&1 | tee -a "$LOG"

# 2) Sweep 20 cells + pre-registered verdict (resumable watchdog)
count_stl() {
  $PY - <<'EOF'
import json, pathlib
n = 0
for p in pathlib.Path('outputs/stl64/parts').glob('*.json'):
    try:
        if json.loads(p.read_text()).get('done'):
            n += 1
    except Exception:
        pass
print(n)
EOF
}
mkdir -p outputs/stl64/parts
for attempt in $(seq 1 8); do
  n=$(count_stl)
  echo "[v3] sweep attempt $attempt, parts done: $n/20" | tee -a "$LOG"
  [ "$n" -ge 20 ] && break
  timeout -k 30 2700 $PY experiments/exp_stl64_frontier.py --device cuda 2>&1 | tee -a "$LOG"
done
$PY experiments/exp_stl64_frontier.py --analyze 2>&1 | tee -a "$LOG"

echo "[v3] DONE $(date)" | tee -a "$LOG"
