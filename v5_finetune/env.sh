#!/usr/bin/env bash
set -euo pipefail

export AIC_DINO_M2F_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DINOV3_SOURCE:?Set DINOV3_SOURCE to the local DINOv3 source checkout.}"
export PYTHONPATH="$AIC_DINO_M2F_ROOT:$DINOV3_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$AIC_DINO_M2F_ROOT/.cache/huggingface"
export TORCH_HOME="$AIC_DINO_M2F_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$AIC_DINO_M2F_ROOT/.cache/torch_extensions"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"
