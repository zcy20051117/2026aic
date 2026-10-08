#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session=aic_dino_m2f_clean_hard_v4

cd "$project"
source env.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing duplicate tmux session: $session" >&2
  exit 1
fi

mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
log="${AIC_OUTPUT_ROOT:?Set AIC_OUTPUT_ROOT}/logs/v4_formal_${stamp}.log"
mkdir -p "$(dirname "$log")"
tmux new-session -d -s "$session" \
  "bash -lc 'set -euo pipefail; cd $project; source env.sh; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; python -m aic_dino_m2f.train --config configs/baseline_v1.yaml --end-epoch 50 2>&1 | tee $log'"
echo "Started $session; follow with: tmux attach -t $session"
echo "Log: $log"
