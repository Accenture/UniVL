#!/usr/bin/env python3
"""
Sharded inference on a JSONL of records (e.g. train_real_4k.jsonl). Each shard
slices the JSONL into shard_n contiguous chunks and runs FLUX inference on its
chunk, saving combined panels [cond | gen | gt] to <out_dir>/.

Designed to be run 4-way in parallel (one per GPU) so 4000 records finish in
~1.5 hours instead of ~6 hours.

Usage (per GPU):
    python3 evaluation/fid/infer_univl_shard.py \\
        --records output_inference/train_real_4k.jsonl \\
        --shard_idx 0 --shard_n 4 --gpu 0 \\
        --ckpt runs/<run>/ckpt/50000 \\
        --out_dir output_inference/eval4k_Cn50 \\
        --two_condition

Reuses inference helpers from evaluation/eval_univl.py to keep semantics identical.
"""
import argparse, json, os, sys, importlib

from evaluation import eval_univl as r200  # reuse load_model + run_inference logic

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--shard_n",   type=int, required=True)
    ap.add_argument("--gpu",       type=int, required=True)
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--two_condition", action="store_true")
    ap.add_argument("--use_text_on_mask", action="store_true", default=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    # Set TWO_CONDITION flag in pipeline (matches eval_univl's behavior)
    import univl.pipeline.flux_univl as flux_module
    flux_module.TWO_CONDITION = args.two_condition
    print(f"[shard {args.shard_idx}/{args.shard_n} gpu={args.gpu}] TWO_CONDITION={flux_module.TWO_CONDITION}")

    # Load full record list and slice contiguous shard
    all_records = []
    with open(args.records) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_records.append(json.loads(line))
    n = len(all_records)
    chunk = (n + args.shard_n - 1) // args.shard_n
    start = args.shard_idx * chunk
    end = min(n, start + chunk)
    my_records = all_records[start:end]
    print(f"[shard {args.shard_idx}] handling records [{start}:{end}] = {len(my_records)} samples")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load FLUX pipeline + checkpoint (same as evaluation/eval_univl.py)
    pipe = r200.load_model(args.ckpt, device, two_condition=args.two_condition)

    # Run inference. Reuse evaluation.eval_univl.run_inference but it indexes into a flat
    # list and writes filename based on idx. To keep filenames unique across
    # shards we offset idx by `start`.
    #
    # run_inference(pipe, records, output_dir, device, use_text_on_mask, two_condition)
    # writes "{idx:04d}_{rec.type}_{basename}_{objn}.jpg" — we need to shift idx
    # by `start` so shards don't collide. Easiest: monkey-patch the formatter.
    # Cleaner: run inference inline here.

    # ---- Inline inference (copied / pruned from evaluation.eval_univl.run_inference) ----
    import torch.nn.functional as F  # noqa
    import torchvision.transforms as T  # noqa
    from univl.pipeline.flux_univl import Condition, generate

    SEED = r200.SEED
    NUM_INFERENCE_STEPS = r200.NUM_INFERENCE_STEPS
    GUIDANCE_SCALE = r200.GUIDANCE_SCALE
    CONDITION_SIZE = r200.CONDITION_SIZE
    TARGET_SIZE = r200.TARGET_SIZE

    for local_i, rec in enumerate(my_records):
        idx = start + local_i
        cond_path = rec["cond_path"]
        gt_path   = rec["gt_path"]
        objn      = rec["objn"]
        bbox      = rec["bbox"]
        rtype     = rec["type"]

        if not os.path.exists(cond_path):
            print(f"  [{idx}] SKIP missing cond {cond_path}")
            continue

        # bbox: [x1, y1, x2, y2] in original-image pixels
        condition_img = Image.open(cond_path).convert("RGB")
        orig_w, orig_h = condition_img.size
        condition_img = condition_img.resize(CONDITION_SIZE)

        pts = [bbox[0]/orig_w, bbox[1]/orig_h, bbox[2]/orig_w, bbox[3]/orig_h]
        pts = [max(0.0, min(1.0, p)) for p in pts]

        # Optionally render objn text on mask area
        if args.use_text_on_mask:
            sx = CONDITION_SIZE[0] / orig_w
            sy = CONDITION_SIZE[1] / orig_h
            bx1, by1 = int(bbox[0]*sx), int(bbox[1]*sy)
            bx2, by2 = int(bbox[2]*sx), int(bbox[3]*sy)
            text = objn.replace("-", " ")
            condition_img = r200._render_text_in_bbox(condition_img, bx1, by1, bx2, by2, text)

        condition = Condition(condition_img, "default", [0, 0], 1.0)
        generator = torch.Generator(device=device).manual_seed(SEED)

        result = generate(
            pipe, prompt="", conditions=[condition],
            height=TARGET_SIZE[1], width=TARGET_SIZE[0],
            num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
            generator=generator, model_config={"independent_condition": False},
            kv_cache=False, pts=pts,
        )
        gen_img = result.images[0]

        gt_img = None
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert("RGB").resize(TARGET_SIZE)
        elif os.path.exists(gt_path.replace("_r.png", "_r.jpg")):
            gt_img = Image.open(gt_path.replace("_r.png", "_r.jpg")).convert("RGB").resize(TARGET_SIZE)

        # Save panel: [cond | gen | gt] (1536 x 512 if gt exists, else 1024 x 512)
        basename = os.path.basename(cond_path).replace(".png", "").replace(".jpg", "")
        out_name = f"{idx:05d}_{rtype}_{basename}_{objn}.jpg"
        panels = [condition_img.resize(TARGET_SIZE), gen_img]
        if gt_img is not None:
            panels.append(gt_img)
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for k, p in enumerate(panels):
            combined.paste(p, (TARGET_SIZE[0]*k, 0))
        combined.save(os.path.join(args.out_dir, out_name), quality=95)

        if (local_i+1) % 25 == 0 or local_i == len(my_records)-1:
            print(f"[shard {args.shard_idx}] [{local_i+1}/{len(my_records)}] {rtype:8s} {objn}")


if __name__ == "__main__":
    main()
