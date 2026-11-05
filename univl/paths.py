"""Filesystem locations used by UniVL scripts.

Every location can be overridden with an environment variable, so no script needs
to be edited to run on a new machine. Relative defaults resolve against the
working directory (run scripts from the repository root).
"""
import os

# Root of the UniVL-ImgGen data (LAION images, mask-text records, ImgEdit records).
DATA_ROOT = os.environ.get("UNIVL_DATA_ROOT", "data_root")
# Evaluation sets, generated panels, and metric files.
OUTPUT_ROOT = os.environ.get("UNIVL_OUTPUT_ROOT", "outputs")
# Released ImgEdit single-turn data (input to data/imgedit/*).
IMGEDIT_ROOT = os.environ.get("IMGEDIT_ROOT", "imgedit/Singleturn")
# COCO val2017 (images in val2017/, annotations in annotations/).
COCO_ROOT = os.environ.get("COCO_ROOT", "coco_val2017")
# Vision-encoder weights extracted from DeepSeek-OCR (tools/extract_deepseek_ocr_vision_weights.py).
DEEPSEEK_OCR_WEIGHTS = os.environ.get(
    "DEEPSEEK_OCR_WEIGHTS", "checkpoints/deepseek_ocr_vision.safetensors"
)
# Third-party baseline code bases (evaluation/baselines only).
POWERPAINT_REPO = os.environ.get("POWERPAINT_REPO", "third_party/PowerPaint")
CONTROLNET_INPAINT_REPO = os.environ.get(
    "CONTROLNET_INPAINT_REPO", "third_party/FLUX-Controlnet-Inpainting"
)
