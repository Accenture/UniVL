#!/usr/bin/env bash
# Stage 2: fine-tune the FLUX.1-dev LoRA conditioned on the UniVL embedding.
# Set model.trained_adapter_path in the config to the Stage-1 output first.
# Usage: bash scripts/train_stage2.sh   (from the repository root)
set -euo pipefail

export UNIVL_CONFIG=${UNIVL_CONFIG:-./configs/stage2_diffusion.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=true
# Optional: export WANDB_API_KEY=<your key> to log to Weights & Biases.

echo "Config: $UNIVL_CONFIG  GPUs: $CUDA_VISIBLE_DEVICES"
accelerate launch --main_process_port ${MAIN_PROCESS_PORT:-29501} -m univl.training.train
