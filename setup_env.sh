#!/bin/bash
# UniVL environment setup.
# Creates a conda env named "univl" with Python 3.10 + CUDA 12.1 stack.
#
# Usage:
#   bash setup_env.sh                 # default name "univl"
#   ENV_NAME=my_env bash setup_env.sh # custom name
#
# After completion:
#   conda activate univl
#   python -c "import torch, diffusers, transformers; print(torch.cuda.is_available())"
set -e

ENV_NAME=${ENV_NAME:-univl}
PYTHON_VER=3.10

REPO=$(cd "$(dirname "$0")" && pwd)
echo "Repo: $REPO"
echo "Env:  $ENV_NAME (Python $PYTHON_VER)"

# 1. Detect / find conda
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found. Install Miniconda first: https://docs.conda.io/projects/miniconda/"
    exit 1
  fi
fi
eval "$(conda shell.bash hook)"

# 2. Create or skip
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists. Skipping creation."
else
  conda create -y -n "$ENV_NAME" python=$PYTHON_VER pip
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel setuptools

# 3. Torch (CUDA 12.1) — pin to versions we tested.
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 4. Diffusers / transformers stack.
pip install \
    diffusers==0.31.0 \
    transformers==4.49.0 \
    peft==0.15.2 \
    accelerate==0.34.2 \
    safetensors sentencepiece protobuf

# 5. Training stack.
pip install lightning>=2.5 prodigyopt wandb datasets

# 6. Image / data utilities.
pip install opencv-python pycocotools Pillow 'numpy<2' addict easydict imgaug

# 7. Metrics + evaluation.
pip install torchmetrics torch-fidelity pyiqa timm matplotlib wordcloud

# 8. Misc.
pip install tqdm omegaconf PyYAML

# 9. Sanity check.
python - <<'PY'
import torch, diffusers, transformers, peft, accelerate
print("torch        :", torch.__version__, " | cuda available:", torch.cuda.is_available())
print("diffusers    :", diffusers.__version__)
print("transformers :", transformers.__version__)
print("peft         :", peft.__version__)
print("accelerate   :", accelerate.__version__)
PY

echo ""
echo "Done. To use:"
echo "    conda activate $ENV_NAME"
