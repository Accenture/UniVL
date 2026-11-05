#!/usr/bin/env python3
"""PowerPaint v2.1 (open-mmlab, ECCV 2024) eval on the same 200 single-box + 75 multibox set.

PowerPaint v2.1 is a SD1.5-based text-guided inpainter built on the BrushNet
backbone with learned task tokens (P_obj, P_ctxt, P_shape). Set $POWERPAINT_REPO to a
checkout of https://github.com/open-mmlab/PowerPaint. For class-conditioned
inpainting we use task=text-guided → promptA = "<class> P_obj", promptB = "<class> P_obj".

Run with:
  python -m evaluation.baselines.eval_powerpaint --gpu 2 --mode single
"""
import argparse, glob, json, os, re, sys
from univl.paths import OUTPUT_ROOT, POWERPAINT_REPO
sys.path.insert(0, POWERPAINT_REPO)
import torch
import numpy as np
from PIL import Image, ImageDraw

EVAL_JSONL     = OUTPUT_ROOT + "/eval_sample_set_200.jsonl"
MULTIBOX_JSONL = OUTPUT_ROOT + "/multibox_eval_75.jsonl"
TARGET_SIZE    = (512, 512)
SEED           = 42
DTYPE          = torch.float16
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE      = 7.5
BRUSHNET_COND_SCALE = 1.0
TRADOFF             = 1.0   # fitting_degree

POWERPAINT_HF = "JunhaoZhuang/PowerPaint-v2-1"


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


def load_powerpaint_pipe(device):
    from huggingface_hub import snapshot_download
    from transformers import CLIPTextModel
    from safetensors.torch import load_model
    from diffusers import UniPCMultistepScheduler

    from powerpaint.models.BrushNet_CA import BrushNetModel
    from powerpaint.models.unet_2d_condition import UNet2DConditionModel
    from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
    from powerpaint.utils.utils import TokenizerWrapper, add_tokens

    repo_root = snapshot_download(POWERPAINT_HF)
    base_path = os.path.join(repo_root, "realisticVisionV60B1_v51VAE")
    pp_brushnet_dir = os.path.join(repo_root, "PowerPaint_Brushnet")
    print(f"Base model: {base_path}")
    print(f"PowerPaint BrushNet: {pp_brushnet_dir}")

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=DTYPE)
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=DTYPE)
    brushnet = BrushNetModel.from_unet(unet)

    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        base_path, brushnet=brushnet, text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=DTYPE, low_cpu_mem_usage=False, safety_checker=None,
    )
    pipe.unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=DTYPE)
    pipe.tokenizer = TokenizerWrapper(from_pretrained=base_path, subfolder="tokenizer",
                                      revision=None, torch_type=DTYPE)
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    load_model(pipe.brushnet,
               os.path.join(pp_brushnet_dir, "diffusion_pytorch_model.safetensors"))
    pipe.text_encoder_brushnet.load_state_dict(
        torch.load(os.path.join(pp_brushnet_dir, "pytorch_model.bin"), map_location="cpu"),
        strict=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _build_pp_prompts(prompt):
    """Text-guided (object insertion) → P_obj on both A and B and negatives."""
    promptA = prompt + " P_obj"
    promptB = prompt + " P_obj"
    negA = "P_obj"
    negB = "P_obj"
    return promptA, promptB, negA, negB


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

        # PowerPaint expects: image with mask zeroed, mask as RGB
        init_arr = np.array(gt_512)
        mask_arr = np.array(mask) > 127
        init_arr_masked = init_arr * (~mask_arr[..., None])
        init_pil = Image.fromarray(init_arr_masked.astype(np.uint8)).convert("RGB")
        mask_pil = Image.fromarray((mask_arr.astype(np.uint8) * 255)).convert("RGB")

        prompt = _build_prompt_single(objn)
        pA, pB, nA, nB = _build_pp_prompts(prompt)
        generator = torch.Generator(device=device).manual_seed(SEED)

        out = pipe(
            promptA=pA, promptB=pB, promptU=prompt,
            tradoff=TRADOFF, tradoff_nag=TRADOFF,
            negative_promptA=nA, negative_promptB=nB, negative_promptU="",
            image=init_pil, mask=mask_pil,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            brushnet_conditioning_scale=BRUSHNET_COND_SCALE,
            width=TARGET_SIZE[0], height=TARGET_SIZE[1],
        ).images[0]
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
        pA, pB, nA, nB = _build_pp_prompts(prompt)
        objn = ", ".join(c.replace("-", " ") for c in class_names)

        generator = torch.Generator(device=device).manual_seed(SEED)
        out = pipe(
            promptA=pA, promptB=pB, promptU=prompt,
            tradoff=TRADOFF, tradoff_nag=TRADOFF,
            negative_promptA=nA, negative_promptB=nB, negative_promptU="",
            image=init_pil, mask=mask_pil,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            brushnet_conditioning_scale=BRUSHNET_COND_SCALE,
            width=TARGET_SIZE[0], height=TARGET_SIZE[1],
        ).images[0]
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
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    ap.add_argument("--out_dir",
        default=OUTPUT_ROOT + "/eval_powerpaint")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    pipe = load_powerpaint_pipe(device)

    if args.mode in ("single", "both"):
        records = load_eval_records()
        print(f"\n──── PowerPaint single-box ({len(records)}) ────", flush=True)
        run_singlebox(pipe, records, args.out_dir, device)

    if args.mode in ("multi", "both"):
        mb = load_multibox_records()
        print(f"\n──── PowerPaint multibox ({len(mb)}) ────", flush=True)
        run_multibox(pipe, mb, args.out_dir + "_multibox", device)

    print(f"\nDone (panels saved). Visual review before metrics.")


if __name__ == "__main__":
    main()
