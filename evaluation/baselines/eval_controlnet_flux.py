#!/usr/bin/env python3
"""
ControlNet-FLUX (alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta) eval.

Self-contained. Requires diffusers >= 0.32 (the Beta ControlNet uses
extra_condition_channels=4), so run it in a separate environment from UniVL
(which pins diffusers 0.31). Set $CONTROLNET_INPAINT_REPO to a checkout of
https://github.com/alimama-creative/FLUX-Controlnet-Inpainting for its pipeline code.

Usage:
  python -m evaluation.baselines.eval_controlnet_flux --gpu 0
"""
import argparse, glob, json, os, re, sys
from univl.paths import CONTROLNET_INPAINT_REPO, OUTPUT_ROOT
# Add alimama-creative custom pipeline files to path
sys.path.insert(0, CONTROLNET_INPAINT_REPO)
import torch
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict


EVAL_JSONL     = OUTPUT_ROOT + "/eval_sample_set_200.jsonl"
MULTIBOX_JSONL = OUTPUT_ROOT + "/multibox_eval_75.jsonl"
TARGET_SIZE    = (512, 512)
SEED           = 42
DTYPE          = torch.bfloat16
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE      = 3.5         # standard CFG (used as `guidance_scale`)
TRUE_GUIDANCE_SCALE = 1.0         # alimama Beta default
CONTROLNET_COND_SCALE = 0.9       # alimama default
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

CONTROLNET_REPO = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
BASE_FLUX_REPO  = "black-forest-labs/FLUX.1-dev"


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


def load_controlnet_pipe(device):
    # Use alimama-creative's custom pipeline (extra 4 mask channels not supported in stock diffusers)
    from controlnet_flux import FluxControlNetModel
    from transformer_flux import FluxTransformer2DModel
    from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
    print(f"Loading ControlNet from {CONTROLNET_REPO} ...")
    cnet = FluxControlNetModel.from_pretrained(CONTROLNET_REPO, torch_dtype=DTYPE)
    print(f"Loading FLUX transformer from {BASE_FLUX_REPO} ...")
    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_FLUX_REPO, subfolder="transformer", torch_dtype=DTYPE
    )
    print(f"Loading FLUX pipeline from {BASE_FLUX_REPO} ...")
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        BASE_FLUX_REPO, controlnet=cnet, transformer=transformer, torch_dtype=DTYPE
    ).to(device)
    pipe.transformer.to(DTYPE)
    pipe.controlnet.to(DTYPE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def run_singlebox(pipe, records, output_dir, device, prompt_mode="class"):
    os.makedirs(output_dir, exist_ok=True)
    gen, gt, gc, gtc, text, types = [], [], [], [], [], []
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
        prompt = _build_prompt_single(objn) if prompt_mode == "class" else " "

        generator = torch.Generator(device=device).manual_seed(SEED)
        result = pipe(
            prompt=prompt,
            height=TARGET_SIZE[1], width=TARGET_SIZE[0],
            control_image=gt_512,
            control_mask=mask.convert("RGB"),
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            controlnet_conditioning_scale=CONTROLNET_COND_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            true_guidance_scale=TRUE_GUIDANCE_SCALE,
            negative_prompt="",
        )
        gen_img = result.images[0]

        basename = os.path.basename(gt_path).replace(".jpg", "").replace(".png", "")
        out_name = f"{idx:04d}_{rec['type']}_{basename}_{objn[:20]}.jpg"
        panels = [masked_input, gen_img, gt_512]
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        gen.append(gen_img); gt.append(gt_512); text.append(objn); types.append(rec["type"])
        cx1, cy1 = max(0, bx1), max(0, by1)
        cx2, cy2 = min(TARGET_SIZE[0], bx2), min(TARGET_SIZE[1], by2)
        if cx2 > cx1+4 and cy2 > cy1+4:
            gc.append(gen_img.crop((cx1, cy1, cx2, cy2)))
            gtc.append(gt_512.crop((cx1, cy1, cx2, cy2)))
        if (idx+1) % 25 == 0 or idx == len(records)-1:
            print(f"  [{idx+1}/{len(records)}] {rec['type']:8s} {objn}")
    return gen, gt, gc, gtc, text, types


@torch.no_grad()
def run_multibox(pipe, mb_records, output_dir, device, prompt_mode="class"):
    os.makedirs(output_dir, exist_ok=True)
    gen, gt, gc, gtc, text, types = [], [], [], [], [], []
    for idx, rec in enumerate(mb_records):
        gt_path = rec["gt_path"]
        segments = rec["segments"]
        res = rec["resolution"]; img_w, img_h = res["width"], res["height"]
        if not os.path.exists(gt_path): continue
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
        prompt = _build_prompt_multi(class_names) if prompt_mode == "class" else " "
        objn = ", ".join(c.replace("-", " ") for c in class_names)

        generator = torch.Generator(device=device).manual_seed(SEED)
        result = pipe(
            prompt=prompt,
            height=TARGET_SIZE[1], width=TARGET_SIZE[0],
            control_image=gt_512,
            control_mask=mask.convert("RGB"),
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            controlnet_conditioning_scale=CONTROLNET_COND_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            true_guidance_scale=TRUE_GUIDANCE_SCALE,
            negative_prompt="",
        )
        gen_img = result.images[0]

        safe_objn = re.sub(r"[^A-Za-z0-9 ,._-]", "_", objn[:30].replace("\\'", "'"))
        out_name = f"mb_{idx:04d}_{os.path.basename(gt_path).replace('.jpg','').replace('.png','')}_{safe_objn}.jpg"
        panels = [masked_input, gen_img, gt_512]
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        gen.append(gen_img); gt.append(gt_512); text.append(objn); types.append("multibox")
        cx1, cy1, cx2, cy2 = bboxes_512[0]
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(TARGET_SIZE[0], cx2), min(TARGET_SIZE[1], cy2)
        if cx2 > cx1+4 and cy2 > cy1+4:
            gc.append(gen_img.crop((cx1, cy1, cx2, cy2)))
            gtc.append(gt_512.crop((cx1, cy1, cx2, cy2)))
        if (idx+1) % 15 == 0 or idx == len(mb_records)-1:
            print(f"  [mb {idx+1}/{len(mb_records)}] {objn[:40]}")
    return gen, gt, gc, gtc, text, types


@torch.no_grad()
def compute_metrics(gen, gt, gen_crops, gt_crops, text_labels, types, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from transformers import CLIPModel, CLIPProcessor
    import pyiqa
    to_tensor = T.ToTensor()
    out = {}
    n = len(gen); print(f"\nComputing metrics on {n} pairs ...")
    print("  SSIM ..."); ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_vals = []
    for g, gt_i in zip(gen, gt):
        gt_t = to_tensor(gt_i).unsqueeze(0).to(device)
        g_t = to_tensor(g.resize(gt_i.size)).unsqueeze(0).to(device)
        ssim_vals.append(ssim(g_t, gt_t).item())
    out["SSIM"] = float(np.mean(ssim_vals))
    print("  PSNR ..."); psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_vals = []
    for g, gt_i in zip(gen, gt):
        gt_t = to_tensor(gt_i).unsqueeze(0).to(device)
        g_t = to_tensor(g.resize(gt_i.size)).unsqueeze(0).to(device)
        psnr_vals.append(psnr(g_t, gt_t).item())
    out["PSNR"] = float(np.mean(psnr_vals))
    print("  FID ..."); fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    bb = []
    for gt_i in gt:
        bb.append(to_tensor(gt_i.resize(TARGET_SIZE)))
        if len(bb) == 16: fid.update(torch.stack(bb).to(device), real=True); bb = []
    if bb: fid.update(torch.stack(bb).to(device), real=True); bb = []
    for g in gen:
        bb.append(to_tensor(g.resize(TARGET_SIZE)))
        if len(bb) == 16: fid.update(torch.stack(bb).to(device), real=False); bb = []
    if bb: fid.update(torch.stack(bb).to(device), real=False); bb = []
    try: out["FID"] = float(fid.compute().item())
    except Exception as e: print(f"  FID failed: {e}"); out["FID"] = float("nan")
    print("  MUSIQ ..."); musiq = pyiqa.create_metric("musiq", device=device)
    out["MUSIQ"] = float(np.mean([musiq(to_tensor(g.resize((512,512))).unsqueeze(0).to(device)).item() for g in gen]))
    print("  CLIP load ..."); clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("  CLIP-Text ..."); ct_vals = []
    for g, t in zip(gen, text_labels):
        inp = clip_processor(text=[t], images=g, return_tensors="pt", padding=True, truncation=True).to(device)
        o = clip_model(**inp); ct_vals.append((o.logits_per_image / 100.0).item())
    out["CLIP-Text"] = float(np.mean(ct_vals))
    print("  CLIP-Image ..."); ci_vals = []
    for g, gt_i in zip(gen, gt):
        gi = clip_processor(images=g,    return_tensors="pt").to(device)
        gti = clip_processor(images=gt_i, return_tensors="pt").to(device)
        gf  = F.normalize(clip_model.get_image_features(**gi),  dim=-1)
        gtf = F.normalize(clip_model.get_image_features(**gti), dim=-1)
        ci_vals.append((gf*gtf).sum(dim=-1).item())
    out["CLIP-Image"] = float(np.mean(ci_vals))
    print("  Region CLIP-T ..."); rt_vals = []
    for gc_, t in zip(gen_crops, text_labels[:len(gen_crops)]):
        gc_r = gc_.resize((224,224))
        inp = clip_processor(text=[t], images=gc_r, return_tensors="pt", padding=True, truncation=True).to(device)
        o = clip_model(**inp); rt_vals.append((o.logits_per_image / 100.0).item())
    out["Region CLIP-T"] = float(np.mean(rt_vals)) if rt_vals else float("nan")
    print("  Region CLIP-I ..."); ri_vals = []
    for gc_, gtc_ in zip(gen_crops, gt_crops):
        gc_r = gc_.resize((224,224)); gtc_r = gtc_.resize((224,224))
        gi = clip_processor(images=gc_r, return_tensors="pt").to(device)
        gti = clip_processor(images=gtc_r, return_tensors="pt").to(device)
        gf  = F.normalize(clip_model.get_image_features(**gi),  dim=-1)
        gtf = F.normalize(clip_model.get_image_features(**gti), dim=-1)
        ri_vals.append((gf*gtf).sum(dim=-1).item())
    out["Region CLIP-I"] = float(np.mean(ri_vals)) if ri_vals else float("nan")
    per_type = defaultdict(lambda: defaultdict(list))
    for i, t in enumerate(types):
        per_type[t]["SSIM"].append(ssim_vals[i])
        per_type[t]["PSNR"].append(psnr_vals[i])
        per_type[t]["CLIP-Image"].append(ci_vals[i])
        if i < len(ct_vals): per_type[t]["CLIP-Text"].append(ct_vals[i])
    out["per_type"] = {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in per_type.items()}
    del clip_model; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", default=OUTPUT_ROOT + "/eval_controlnet")
    ap.add_argument("--prompt_mode", choices=["class", "empty"], default="class",
                    help="'class' = 'mask contains <name>', 'empty' = single space ' '")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    print(f"prompt_mode: {args.prompt_mode}")

    pipe = load_controlnet_pipe(device)

    print("\n──── ControlNet single-box ────")
    records = load_eval_records()
    print(f"records: {len(records)}")
    gen, gt, gc, gtc, text, types = run_singlebox(pipe, records, args.out_dir, device, prompt_mode=args.prompt_mode)

    print("\n──── ControlNet multibox ────")
    mb = load_multibox_records()
    print(f"multibox records: {len(mb)}")
    mb_gen, mb_gt, mb_gc, mb_gtc, mb_text, mb_types = run_multibox(pipe, mb, args.out_dir + "_multibox", device, prompt_mode=args.prompt_mode)

    del pipe; torch.cuda.empty_cache()

    print("\n══ Single-box ══"); res_single = compute_metrics(gen, gt, gc, gtc, text, types, device)
    for k, v in res_single.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
    print("\n══ Multibox ══"); res_multi = compute_metrics(mb_gen, mb_gt, mb_gc, mb_gtc, mb_text, mb_types, device)
    for k, v in res_multi.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
    print("\n══ Combined ══")
    res_comb = compute_metrics(gen+mb_gen, gt+mb_gt, gc+mb_gc, gtc+mb_gtc,
                               text+mb_text, types+mb_types, device)
    for k, v in res_comb.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")

    save = {"single_box": res_single, "multibox": res_multi, "combined": res_comb,
            "method": "controlnet-flux-inpaint", "controlnet": CONTROLNET_REPO,
            "prompt_mode": args.prompt_mode}
    out_metrics = os.path.join(args.out_dir, "metrics.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved → {out_metrics}")


if __name__ == "__main__":
    main()
