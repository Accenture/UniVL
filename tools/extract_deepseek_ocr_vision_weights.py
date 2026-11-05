"""Extract the vision-encoder weights (SAM, CLIP, projector) from the DeepSeek-OCR checkpoint.

UniVL initializes its CLIP backbone (with the SAM stem) from DeepSeek-OCR. This script keeps
only the `model.sam_model.*`, `model.vision_model.*`, and `model.projector.*` tensors, strips
the leading `model.` prefix, and writes them to one safetensors file. Training reads it from
`model.deepseek_w_path` in the config; evaluation reads it from $DEEPSEEK_OCR_WEIGHTS.

Usage (from the repository root):
    python -m tools.extract_deepseek_ocr_vision_weights          # downloads from the HF Hub
    python -m tools.extract_deepseek_ocr_vision_weights --input /path/to/model-00001-of-000001.safetensors
"""
import argparse
import os

from safetensors.torch import load_file, save_file

from univl.paths import DEEPSEEK_OCR_WEIGHTS

HF_REPO = "deepseek-ai/DeepSeek-OCR"
HF_FILE = "model-00001-of-000001.safetensors"
KEEP_PREFIXES = ("model.sam_model.", "model.vision_model.", "model.projector.")


def extract(input_path, output_path):
    print(f"Reading DeepSeek-OCR weights from: {input_path}")
    state_dict = load_file(input_path)
    # Strip only the leading "model." so keys become sam_model.*, vision_model.*, projector.*,
    # the prefixes expected by the loaders in univl/training/trainer.py and evaluation/eval_univl.py.
    vision_state = {
        key[len("model."):]: tensor
        for key, tensor in state_dict.items()
        if key.startswith(KEEP_PREFIXES)
    }
    if not vision_state:
        raise SystemExit("No sam_model/vision_model/projector tensors found; is this the DeepSeek-OCR checkpoint?")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(vision_state, output_path)
    print(f"Saved {len(vision_state)} tensors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None,
                        help=f"Local DeepSeek-OCR safetensors file (default: download {HF_FILE} from {HF_REPO})")
    parser.add_argument("--output", default=DEEPSEEK_OCR_WEIGHTS, help="Output path (default: $DEEPSEEK_OCR_WEIGHTS)")
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        from huggingface_hub import hf_hub_download
        input_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    extract(input_path, args.output)


if __name__ == "__main__":
    main()
