#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session=aic_dino_m2f_loveda_vehicle_mix_v5
checkpoint="${V4_BEST_CHECKPOINT:?Set V4_BEST_CHECKPOINT to the V4 best.pth path.}"

cd "$project"
source env.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
test -f "$checkpoint"
python scripts/prepare_loveda_mix.py --config configs/baseline_v1.yaml

if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing duplicate tmux session: $session" >&2
  exit 1
fi

mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
log="${AIC_OUTPUT_ROOT:?Set AIC_OUTPUT_ROOT}/logs/v5_formal_${stamp}.log"
mkdir -p "$(dirname "$log")"
tmux new-session -d -s "$session" \
  "bash -lc 'set -o pipefail; cd $project; source env.sh; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; python -m aic_dino_m2f.train --config configs/baseline_v1.yaml --finetune-from $checkpoint --end-epoch 10 2>&1 | tee $log'"
echo "Started $session; follow with: tmux attach -t $session"
echo "Log: $log"
