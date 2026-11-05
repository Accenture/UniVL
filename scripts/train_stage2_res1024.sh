#!/usr/bin/env bash
# Stage 2 at 1024x1024 (resolution ablation).
# Usage: bash scripts/train_stage2_res1024.sh   (from the repository root)
# Requires RES1024 = True in univl/pipeline/flux_univl.py.
set -euo pipefail

export UNIVL_CONFIG=${UNIVL_CONFIG:-./configs/stage2_diffusion_res1024.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=true
# Optional: export WANDB_API_KEY=<your key> to log to Weights & Biases.

echo "Config: $UNIVL_CONFIG  GPUs: $CUDA_VISIBLE_DEVICES"
accelerate launch --main_process_port ${MAIN_PROCESS_PORT:-29501} -m univl.training.train
