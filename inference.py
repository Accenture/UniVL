"""Generate an image with UniVL from a source image, mask boxes, and short labels.

Each region is given as a pixel box in the source image plus a short label. The label is
rendered as text inside its box to form the contextual condition, which is the only input
UniVL reads (no text prompt by default). Several regions are generated in one forward pass.

Example (from the repository root):
    python inference.py --ckpt runs/<run>/ckpt/<step> \
        --image photo.jpg \
        --box 40 220 230 470 --label dog \
        --box 260 260 500 470 --label "wooden bench" \
        --out outputs/photo_edit.png

Pass --box/--label once per region. The rendering and sampling settings match
evaluation/eval_univl.py, so results are comparable with the evaluation scripts.
"""
import argparse
import os

import torch
from PIL import Image

import univl.pipeline.flux_univl as flux_module
from evaluation import eval_univl as ev
from univl.pipeline.flux_univl import Condition, generate


def build_condition(image, boxes, labels, size):
    """Render each label inside its box and return (condition image, normalized boxes)."""
    w, h = image.size
    condition = image.convert("RGB").resize(size)
    sx, sy = size[0] / w, size[1] / h
    norm_boxes = []
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        condition = ev._render_text_in_bbox(
            condition, int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), label.replace("-", " ")
        )
        norm_boxes.append([min(1.0, max(0.0, v)) for v in (x1 / w, y1 / h, x2 / w, y2 / h)])
    return condition, (norm_boxes if len(norm_boxes) > 1 else norm_boxes[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint directory containing default.safetensors and omini_adapter.safetensors")
    parser.add_argument("--image", required=True, help="Source image")
    parser.add_argument("--box", type=float, nargs=4, action="append", required=True,
                        metavar=("X1", "Y1", "X2", "Y2"), help="Region box in source-image pixels (repeatable)")
    parser.add_argument("--label", action="append", required=True, help="Short label for the matching --box (repeatable)")
    parser.add_argument("--out", default="outputs/univl_output.png", help="Output image path")
    parser.add_argument("--save-condition", action="store_true", help="Also save the rendered contextual condition")
    parser.add_argument("--seed", type=int, default=ev.SEED)
    parser.add_argument("--steps", type=int, default=ev.NUM_INFERENCE_STEPS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.box) != len(args.label):
        parser.error("Give exactly one --label per --box.")

    flux_module.TWO_CONDITION = False
    pipe = ev.load_model(args.ckpt, args.device)

    image = Image.open(args.image)
    condition_img, pts = build_condition(image, args.box, args.label, ev.CONDITION_SIZE)
    condition = Condition(condition_img, "default", [0, 0], 1.0)
    with torch.no_grad():
        result = generate(
            pipe,
            prompt="",
            conditions=[condition],
            height=ev.TARGET_SIZE[1],
            width=ev.TARGET_SIZE[0],
            num_inference_steps=args.steps,
            guidance_scale=ev.GUIDANCE_SCALE,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
            model_config={"independent_condition": False},
            kv_cache=False,
            pts=pts,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.images[0].save(args.out)
    print(f"Saved {args.out}")
    if args.save_condition:
        stem, ext = os.path.splitext(args.out)
        condition_img.save(f"{stem}_condition{ext}")
        print(f"Saved {stem}_condition{ext}")


if __name__ == "__main__":
    main()
