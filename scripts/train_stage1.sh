#!/usr/bin/env bash
# Stage 1: align the UniVL encoder output with the VAE latent space (no DiT forward pass).
# Usage: bash scripts/train_stage1.sh   (from the repository root)
set -euo pipefail

export UNIVL_CONFIG=${UNIVL_CONFIG:-./configs/stage1_alignment.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=true
# Optional: export WANDB_API_KEY=<your key> to log to Weights & Biases.

echo "Config: $UNIVL_CONFIG  GPUs: $CUDA_VISIBLE_DEVICES"
accelerate launch --main_process_port ${MAIN_PROCESS_PORT:-29501} -m univl.training.train
