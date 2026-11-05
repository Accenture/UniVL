#!/usr/bin/env python3
"""BrushNet (TencentARC, ECCV 2024) eval on the same 200 single-box + 75 multibox set.

Uses BrushNetX from the BrushEdit HF repo (stronger SD1.5-based BrushNet).
Class-name text prompts ("mask contains <name>" / multi).

Run with:
  python -m evaluation.baselines.eval_brushnet --gpu 0 --mode single
  ... --mode multi  on a different GPU
"""
import argparse, glob, json, os, re, sys
from univl.paths import OUTPUT_ROOT
import torch
import numpy as np
from PIL import Image, ImageDraw

EVAL_JSONL     = OUTPUT_ROOT + "/eval_sample_set_200.jsonl"
MULTIBOX_JSONL = OUTPUT_ROOT + "/multibox_eval_75.jsonl"
TARGET_SIZE    = (512, 512)   # SD1.5 native
SEED           = 42
DTYPE          = torch.float16
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE      = 7.5
BRUSHNET_COND_SCALE = 1.0

BRUSHEDIT_HF = "TencentARC/BrushEdit"
BRUSHNETX_SUBFOLDER = "brushnetX"
BASE_MODEL_SUBFOLDER = "base_model/realisticVisionV60B1_v51VAE"


def load_eval_records():
    out = []
    with open(EVAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out


def load_multibox_records():
    if not os.path.exists(MULTIBOX_JSONL): return []
    out = []
    with open(MULTIBOX_JSONL) as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out


def _build_prompt_single(objn):
    return f"mask contains {objn.replace('-', ' ')}"


def _build_prompt_multi(class_names):
    return " ".join(f"mask{i+1} contains {n.replace('-', ' ')}"
                    for i, n in enumerate(class_names))


def _resolve_gt_path(orig_path):
    """Multibox JSONL points at /.../laion-aes/<shard>/<inner>/<basename>;
    the shard listed (e.g. 00006) may not be present locally — try any present shard."""
    if os.path.exists(orig_path): return orig_path
    m = re.match(r"(.*/laion-aes/)(\d+)(/.*)", orig_path)
    if not m: return None
    prefix, _shard, rest = m.group(1), m.group(2), m.group(3)
    for sh in range(0, 100):
        cand = f"{prefix}{sh:05d}{rest}"
        if os.path.exists(cand): return cand
    return None


def _make_mask(img_size, bboxes_512):
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in bboxes_512:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_size[0], x2), min(img_size[1], y2)
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def _make_masked_input(gt_512, bboxes_512):
    masked = gt_512.copy()
    for x1, y1, x2, y2 in bboxes_512:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(TARGET_SIZE[0], x2), min(TARGET_SIZE[1], y2)
        if x2 > x1 and y2 > y1:
            masked.paste((0, 0, 0), (x1, y1, x2, y2))
    return masked


def load_brushnet_pipe(device):
    from huggingface_hub import snapshot_download
    from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

    repo_root = snapshot_download(BRUSHEDIT_HF,
        allow_patterns=[f"{BRUSHNETX_SUBFOLDER}/*", f"{BASE_MODEL_SUBFOLDER}/*"])
    bn_path = os.path.join(repo_root, BRUSHNETX_SUBFOLDER)
    base_path = os.path.join(repo_root, BASE_MODEL_SUBFOLDER)
    print(f"BrushNet weights: {bn_path}")
    print(f"Base model:        {base_path}")

    bn = BrushNetModel.from_pretrained(bn_path, torch_dtype=DTYPE)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_path, brushnet=bn, torch_dtype=DTYPE, low_cpu_mem_usage=False, safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def run_singlebox(pipe, records, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    for idx, rec in enumerate(records):
        gt_path = rec["gt_path"]; objn = rec["objn"]
        if "bbox" in rec:
            bbox = rec["bbox"]; pt1, pt2 = [bbox[0], bbox[1]], [bbox[2], bbox[3]]
        else:
            pt1, pt2 = rec["coords"]
        if not os.path.exists(gt_path):
            alt = gt_path.replace("_r.png", "_r.jpg")
            if os.path.exists(alt): gt_path = alt
            else:
                cond = rec.get("cond_path", "")
                if cond and os.path.exists(cond): gt_path = cond
                else: print(f"  [{idx}] SKIP missing"); continue

        gt_full = Image.open(gt_path).convert("RGB")
        orig_w, orig_h = gt_full.size
        gt_512 = gt_full.resize(TARGET_SIZE)
        sx, sy = TARGET_SIZE[0]/orig_w, TARGET_SIZE[1]/orig_h
        bx1, by1 = int(pt1[0]*sx), int(pt1[1]*sy)
        bx2, by2 = int(pt2[0]*sx), int(pt2[1]*sy)
        bboxes_512 = [(bx1, by1, bx2, by2)]
        mask = _make_mask(TARGET_SIZE, bboxes_512)
        masked_input = _make_masked_input(gt_512, bboxes_512)

        # BrushNet expects: init_image with the masked region zeroed out, mask as RGB binary
        init_arr = np.array(gt_512)
        mask_arr = np.array(mask) > 127
        init_arr_masked = init_arr * (~mask_arr[..., None])
        init_pil = Image.fromarray(init_arr_masked.astype(np.uint8)).convert("RGB")
        mask_pil = Image.fromarray((mask_arr.astype(np.uint8) * 255)).convert("RGB")

        prompt = _build_prompt_single(objn)
        generator = torch.Generator(device=device).manual_seed(SEED)
        out = pipe(prompt, init_pil, mask_pil,
                   num_inference_steps=NUM_INFERENCE_STEPS,
                   guidance_scale=GUIDANCE_SCALE,
                   generator=generator,
                   brushnet_conditioning_scale=BRUSHNET_COND_SCALE).images[0]
        gen_img = out.resize(TARGET_SIZE)

        basename = os.path.basename(gt_path).replace(".jpg", "").replace(".png", "")
        out_name = f"{idx:04d}_{rec['type']}_{basename}_{objn[:20]}.jpg"
        panels = [masked_input, gen_img, gt_512]
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        if (idx+1) % 25 == 0 or idx == len(records)-1:
            print(f"  [{idx+1}/{len(records)}] {rec['type']:8s} {objn}", flush=True)


@torch.no_grad()
def run_multibox(pipe, mb_records, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    for idx, rec in enumerate(mb_records):
        gt_path = _resolve_gt_path(rec["gt_path"])
        segments = rec["segments"]
        res = rec["resolution"]; img_w, img_h = res["width"], res["height"]
        if gt_path is None:
            print(f"  [mb {idx}] SKIP unresolved {rec['gt_path']}", flush=True)
            continue
        gt_full = Image.open(gt_path).convert("RGB")
        gt_512 = gt_full.resize(TARGET_SIZE)
        sx, sy = TARGET_SIZE[0]/img_w, TARGET_SIZE[1]/img_h
        bboxes_512 = []; class_names = []
        for seg in segments:
            bb = seg["bbox"]
            bx1, by1 = int(bb[0]*sx), int(bb[1]*sy)
            bx2, by2 = int(bb[2]*sx), int(bb[3]*sy)
            bboxes_512.append((bx1, by1, bx2, by2)); class_names.append(seg["class_name"])

        mask = _make_mask(TARGET_SIZE, bboxes_512)
        masked_input = _make_masked_input(gt_512, bboxes_512)

        init_arr = np.array(gt_512)
        mask_arr = np.array(mask) > 127
        init_arr_masked = init_arr * (~mask_arr[..., None])
        init_pil = Image.fromarray(init_arr_masked.astype(np.uint8)).convert("RGB")
        mask_pil = Image.fromarray((mask_arr.astype(np.uint8) * 255)).convert("RGB")

        prompt = _build_prompt_multi(class_names)
        objn = ", ".join(c.replace("-", " ") for c in class_names)

        generator = torch.Generator(device=device).manual_seed(SEED)
        out = pipe(prompt, init_pil, mask_pil,
                   num_inference_steps=NUM_INFERENCE_STEPS,
                   guidance_scale=GUIDANCE_SCALE,
                   generator=generator,
                   brushnet_conditioning_scale=BRUSHNET_COND_SCALE).images[0]
        gen_img = out.resize(TARGET_SIZE)

        safe_objn = re.sub(r"[^A-Za-z0-9 ,._-]", "_", objn[:30].replace("\\'", "'"))
        out_name = f"mb_{idx:04d}_{os.path.basename(gt_path).replace('.jpg','').replace('.png','')}_{safe_objn}.jpg"
        panels = [masked_input, gen_img, gt_512]
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        if (idx+1) % 15 == 0 or idx == len(mb_records)-1:
            print(f"  [mb {idx+1}/{len(mb_records)}] {objn[:40]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    ap.add_argument("--out_dir",
        default=OUTPUT_ROOT + "/eval_brushnet")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    pipe = load_brushnet_pipe(device)

    if args.mode in ("single", "both"):
        records = load_eval_records()
        print(f"\n──── BrushNet single-box ({len(records)}) ────", flush=True)
        run_singlebox(pipe, records, args.out_dir, device)

    if args.mode in ("multi", "both"):
        mb = load_multibox_records()
        print(f"\n──── BrushNet multibox ({len(mb)}) ────", flush=True)
        run_multibox(pipe, mb, args.out_dir + "_multibox", device)

    print(f"\nDone (panels saved). To compute metrics, run a follow-up script after visual review.")


if __name__ == "__main__":
    main()
