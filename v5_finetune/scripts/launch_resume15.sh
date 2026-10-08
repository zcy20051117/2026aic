#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${AIC_OUTPUT_ROOT:?Set AIC_OUTPUT_ROOT}/dinov3_vitl16_m2f_loveda_vehicle_mix_v5_resume15"
session=aic_dino_m2f_loveda_resume15

cd "$project"
source env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing duplicate tmux session: $session" >&2
  exit 1
fi

if [[ ! -d "$output" ]]; then
  python scripts/prepare_resume15.py \
    --source-config configs/baseline_v1.yaml \
    --target-config configs/resume15.yaml \
    --expected-last-epoch 6
fi
test -f "$output/checkpoints/last.pth"

mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
log="${AIC_OUTPUT_ROOT:?Set AIC_OUTPUT_ROOT}/logs/v5_resume15_${stamp}.log"
mkdir -p "$(dirname "$log")"
tmux new-session -d -s "$session" \
  "bash -lc 'set -o pipefail; cd $project; source env.sh; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; python -m aic_dino_m2f.train --config configs/resume15.yaml --resume $output/checkpoints/last.pth --end-epoch 21 2>&1 | tee $log'"
echo "Started $session; follow with: tmux attach -t $session"
echo "Log: $log"
