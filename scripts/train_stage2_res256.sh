#!/usr/bin/env bash
# Stage 2 at 256x256 (resolution ablation).
# Usage: bash scripts/train_stage2_res256.sh   (from the repository root)
# Requires RES256 = True in univl/pipeline/flux_univl.py.
set -euo pipefail

export UNIVL_CONFIG=${UNIVL_CONFIG:-./configs/stage2_diffusion_res256.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=true
# Optional: export WANDB_API_KEY=<your key> to log to Weights & Biases.

echo "Config: $UNIVL_CONFIG  GPUs: $CUDA_VISIBLE_DEVICES"
accelerate launch --main_process_port ${MAIN_PROCESS_PORT:-29501} -m univl.training.train
