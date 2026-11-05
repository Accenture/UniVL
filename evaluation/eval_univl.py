#!/usr/bin/env python3
"""
Evaluate a UniVL checkpoint on single-box and multi-box evaluation sets.

Usage (from the repository root):
    python -m evaluation.eval_univl --ckpt runs/<run>/ckpt/<step> --gpu 0

Metrics: FID↓, SSIM↑, PSNR↑, MUSIQ↑, CLIP-Text↑, CLIP-Image↑, Region CLIP↑, Region CLIP-I↑
"""
import argparse, json, os, sys, time, random
from univl.paths import DATA_ROOT, DEEPSEEK_OCR_WEIGHTS, OUTPUT_ROOT
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from collections import defaultdict
import torchvision.transforms as T

# ─── Config ───
FLUX_PATH = "black-forest-labs/FLUX.1-dev"
DEEPSEEK_W_PATH = DEEPSEEK_OCR_WEIGHTS
EVAL_JSONL = OUTPUT_ROOT + "/eval_sample_set_200.jsonl"
MULTIBOX_JSONL = OUTPUT_ROOT + "/multibox_eval_75.jsonl"
CONDITION_SIZE = (512, 512)
TARGET_SIZE = (512, 512)
SEED = 42
DTYPE = torch.bfloat16
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5

def _extract_objn(basename):
    stem = basename.split(".")[0]
    parts = stem.split("_")
    for part in reversed(parts):
        if part == "shrink" or part.isdigit(): continue
        return part
    return stem

def _render_text_in_bbox(img, x1, y1, x2, y2, text):
    if not text or x2 <= x1 + 4 or y2 <= y1 + 4:
        return img
    pw, ph = x2 - x1, y2 - y1
    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    font_size = max(4, min(18, ph // 3))
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    try: tw = draw.textlength(text, font=font)
    except: tw = font_size * len(text) * 0.6
    avail = pw - 4
    if tw > avail and tw > 0:
        font_size = max(4, int(font_size * avail / tw))
        try: font = ImageFont.truetype(FONT_PATH, font_size)
        except: font = ImageFont.load_default()
        try: tw = draw.textlength(text, font=font)
        except: tw = font_size * len(text) * 0.6
    tw = max(tw, 1)
    y_cur = 2
    while y_cur + font_size <= ph - 2:
        x_cur = 2
        while x_cur < pw - 2:
            draw.text((x_cur, y_cur), text, fill=(255, 255, 255), font=font)
            x_cur += int(tw) + 4
        y_cur += font_size + 3
    img = img.copy()
    img.paste(patch, (x1, y1))
    return img

def load_model(ckpt_dir, device, two_condition=False, sigclip_encoder=False, ab_vae=False):
    from diffusers import FluxPipeline
    from univl.deepencoder.sam_vary_sdpa import build_sam_vit_b
    from univl.deepencoder.clip_sdpa import build_clip_l
    from univl.deepencoder.build_linear import MlpProjector
    from addict import Dict as ADict

    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(device)
    target_dim = pipe.vae.config.latent_channels * 4
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()

    # OCR modules — SAM + (CLIP or SigLIP) + projector
    pipe.sam_model = build_sam_vit_b().to(device=device, dtype=DTYPE)
    pipe.sam_model.requires_grad_(False)
    if sigclip_encoder:
        from transformers import SiglipVisionModel
        siglip_name = "google/siglip-large-patch16-256"
        print(f"[SIGCLIP_ENCODER] Loading {siglip_name} (hidden=1024) in place of DeepSeek CLIP-L.")
        pipe.vision_model = SiglipVisionModel.from_pretrained(siglip_name).to(device=device, dtype=DTYPE)
    else:
        pipe.vision_model = build_clip_l().to(device=device, dtype=DTYPE)
    pipe.vision_model.requires_grad_(False)
    proj_cfg = ADict(projector_type="linear", input_dim=2048, n_embed=1280)
    pipe.projector = MlpProjector(proj_cfg).to(device=device, dtype=DTYPE)
    pipe.projector.requires_grad_(False)

    # Adapters
    hidden_dim = 256
    pipe.omini_adapter = torch.nn.Sequential(
        torch.nn.Linear(1024, hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, target_dim),
    ).to(device=device, dtype=DTYPE)
    pipe.omini_adapter2 = torch.nn.Linear(129, target_dim).to(device=device, dtype=DTYPE)
    pipe.conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False).to(device=device, dtype=DTYPE)
    pipe.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False).to(device=device, dtype=DTYPE)

    # Load DeepSeek weights (only into modules whose architecture matches)
    ds_state = load_file(DEEPSEEK_W_PATH)
    sam_state, vision_state, proj_state = {}, {}, {}
    for k, v in ds_state.items():
        clean = k.replace("model.", "", 1) if k.startswith("model.") else k
        if clean.startswith("sam_model."): sam_state[clean.replace("sam_model.", "")] = v
        elif clean.startswith("vision_model."): vision_state[clean.replace("vision_model.", "")] = v
        elif clean.startswith("projector."): proj_state[clean.replace("projector.", "")] = v
    pipe.sam_model.load_state_dict(sam_state, strict=False)
    if sigclip_encoder:
        # SigLIP came from from_pretrained; DeepSeek vision_model.* keys don't fit.
        print("[SIGCLIP_ENCODER] Skipping DeepSeek vision_model weight load (using SigLIP pretrained).")
    else:
        pipe.vision_model.load_state_dict(vision_state, strict=False)
    pipe.projector.load_state_dict(proj_state, strict=False)

    # Vision LoRA — different target_modules for SigLIP (split q/k/v) vs DeepSeek (fused qkv_proj)
    if sigclip_encoder:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    else:
        target_modules = ["qkv_proj", "out_proj", "fc1", "fc2"]
    vision_lora_config = LoraConfig(r=4, lora_alpha=4,
        target_modules=target_modules, lora_dropout=0.0, bias="none")
    pipe.vision_model = get_peft_model(pipe.vision_model, vision_lora_config)

    # Load adapter checkpoint
    asd = load_file(os.path.join(ckpt_dir, "omini_adapter.safetensors"))
    ad, ad2, c1, c2, vl = {}, {}, {}, {}, {}
    for n, t in asd.items():
        if n.startswith("omini_adapter2."): ad2[n.replace("omini_adapter2.", "")] = t
        elif n.startswith("omini_adapter."): ad[n.replace("omini_adapter.", "")] = t
        elif n.startswith("conv1."): c1[n.replace("conv1.", "")] = t
        elif n.startswith("conv2."): c2[n.replace("conv2.", "")] = t
        elif n.startswith("vision_lora."): vl[n.replace("vision_lora.", "")] = t
    pipe.omini_adapter.load_state_dict(ad, strict=True)
    pipe.omini_adapter2.load_state_dict(ad2, strict=True)
    if c1: pipe.conv1.load_state_dict(c1, strict=True)
    if c2: pipe.conv2.load_state_dict(c2, strict=True)
    set_peft_model_state_dict(pipe.vision_model, vl)
    print(f" - Adapter + vision_lora loaded from {ckpt_dir}")

    # Load transformer LoRA
    lora_path = os.path.join(ckpt_dir, "default.safetensors")
    pipe.load_lora_weights(lora_path, adapter_name="default")
    n_lora = sum(1 for n, _ in pipe.transformer.named_parameters() if "lora" in n.lower())
    print(f" - Transformer LoRA loaded ({n_lora} params)")

    for m in [pipe.sam_model, pipe.vision_model, pipe.projector,
              pipe.omini_adapter, pipe.omini_adapter2, pipe.conv1, pipe.conv2, pipe.transformer]:
        m.eval()

    return pipe

def _render_single_text_on_mask(img, x1, y1, x2, y2, text):
    """Single centered text on black rectangle, sized to fit within bbox."""
    if not text or x2 <= x1 + 4 or y2 <= y1 + 4:
        return img
    pw, ph = x2 - x1, y2 - y1
    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    font_size = max(6, min(36, ph // 3))
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    try: tw = draw.textlength(text, font=font)
    except: tw = font_size * len(text) * 0.6
    max_tw = pw * 0.9
    while tw > max_tw and font_size > 4:
        font_size -= 1
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
            tw = draw.textlength(text, font=font)
        except: tw = font_size * len(text) * 0.6
    tw_int = min(int(tw), pw)
    tx = max(0, (pw - tw_int) // 2)
    ty = max(0, (ph - font_size) // 2)
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    img = img.copy()
    img.paste(patch, (x1, y1))
    return img

def load_eval_records():
    records = []
    with open(EVAL_JSONL) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def load_multibox_records():
    """Load 75 multibox eval records from JSONL."""
    if not os.path.exists(MULTIBOX_JSONL):
        print(f"Multibox JSONL not found: {MULTIBOX_JSONL}")
        return []
    records = []
    with open(MULTIBOX_JSONL) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def _build_class_prompt_single(objn):
    return f"mask contains {objn.replace('-', ' ')}"


def _build_class_prompt_multi(class_names):
    return " ".join(f"mask{i+1} contains {n.replace('-', ' ')}"
                    for i, n in enumerate(class_names))


@torch.no_grad()
def run_inference(pipe, records, output_dir, device, use_text_on_mask=True, two_condition=False, prompt_mode="empty", seed=SEED):
    import univl.pipeline.flux_univl as flux_module
    from univl.pipeline.flux_univl import Condition, generate
    flux_module.TWO_CONDITION = two_condition

    os.makedirs(output_dir, exist_ok=True)
    gen_images = []
    gt_images = []
    gen_crops = []
    gt_crops = []
    text_labels = []
    record_types = []

    for idx, rec in enumerate(records):
        cond_path = rec["cond_path"]
        gt_path = rec["gt_path"]
        objn = rec["objn"]
        # Support both "bbox" and "coords" formats
        if "bbox" in rec:
            bbox = rec["bbox"]
            pt1, pt2 = [bbox[0], bbox[1]], [bbox[2], bbox[3]]
        else:
            pt1, pt2 = rec["coords"]

        if not os.path.exists(cond_path):
            print(f"  [{idx}] SKIP: {cond_path} not found")
            continue

        # Load condition image
        condition_img = Image.open(cond_path).convert("RGB")
        orig_w, orig_h = condition_img.size
        condition_img = condition_img.resize(CONDITION_SIZE)

        # Compute normalized bbox for mask
        pts = [pt1[0]/orig_w, pt1[1]/orig_h, pt2[0]/orig_w, pt2[1]/orig_h]
        pts = [max(0.0, min(1.0, p)) for p in pts]

        # Render text on mask if needed
        if use_text_on_mask:
            sx = CONDITION_SIZE[0] / orig_w
            sy = CONDITION_SIZE[1] / orig_h
            bx1, by1 = int(pt1[0]*sx), int(pt1[1]*sy)
            bx2, by2 = int(pt2[0]*sx), int(pt2[1]*sy)
            text = objn.replace("-", " ")
            condition_img = _render_text_in_bbox(condition_img, bx1, by1, bx2, by2, text)

        condition = Condition(condition_img, "default", [0, 0], 1.0)
        generator = torch.Generator(device=device).manual_seed(seed)

        prompt_str = _build_class_prompt_single(objn) if prompt_mode == "class" else ""
        result = generate(pipe, prompt=prompt_str, conditions=[condition],
            height=TARGET_SIZE[1], width=TARGET_SIZE[0],
            num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
            generator=generator, model_config={"independent_condition": False},
            kv_cache=False, pts=pts)
        gen_img = result.images[0]

        # Load GT
        gt_img = None
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert("RGB").resize(TARGET_SIZE)
        elif os.path.exists(gt_path.replace("_r.png", "_r.jpg")):
            gt_img = Image.open(gt_path.replace("_r.png", "_r.jpg")).convert("RGB").resize(TARGET_SIZE)

        # Save combined image
        basename = os.path.basename(cond_path).replace(".png", "").replace(".jpg", "")
        out_name = f"{idx:04d}_{rec['type']}_{basename}_{objn}.jpg"

        panels = [condition_img.resize(TARGET_SIZE), gen_img]
        if gt_img: panels.append(gt_img)
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        # Collect for metrics
        gen_images.append(gen_img)
        text_labels.append(objn)
        record_types.append(rec["type"])

        if gt_img is not None:
            gt_images.append(gt_img)
            # Crop bbox region for region-level metrics
            gw, gh = TARGET_SIZE
            cx1, cy1 = int(pts[0]*gw), int(pts[1]*gh)
            cx2, cy2 = int(pts[2]*gw), int(pts[3]*gh)
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(gw, cx2), min(gh, cy2)
            if cx2 > cx1 + 4 and cy2 > cy1 + 4:
                gen_crop = gen_img.crop((cx1, cy1, cx2, cy2))
                gt_crop = gt_img.crop((cx1, cy1, cx2, cy2))
                gen_crops.append(gen_crop)
                gt_crops.append(gt_crop)
        else:
            gt_images.append(None)

        if (idx+1) % 20 == 0 or idx == len(records)-1:
            print(f"  [{idx+1}/{len(records)}] {rec['type']:8s} {objn}")

    return gen_images, gt_images, gen_crops, gt_crops, text_labels, record_types

@torch.no_grad()
def run_multibox_inference(pipe, mb_records, output_dir, device, use_text_on_mask=True, two_condition=False, prompt_mode="empty", seed=SEED):
    """Run inference on multibox eval records (75 images with 2-5 masks each)."""
    import univl.pipeline.flux_univl as flux_module
    from univl.pipeline.flux_univl import Condition, generate
    flux_module.TWO_CONDITION = two_condition

    os.makedirs(output_dir, exist_ok=True)
    gen_images, gt_images, gen_crops, gt_crops = [], [], [], []
    text_labels, record_types = [], []

    LAION_ROOT = DATA_ROOT + "/laion-aes"

    for idx, rec in enumerate(mb_records):
        gt_path = rec["gt_path"]
        segments = rec["segments"]
        res = rec["resolution"]
        img_w, img_h = res["width"], res["height"]

        if not os.path.exists(gt_path):
            continue

        # Build condition image from GT with text on each mask
        condition_img = Image.open(gt_path).convert("RGB").resize(CONDITION_SIZE)

        # Build normalized bboxes and render text
        all_pts = []
        objn_parts = []
        for seg in segments:
            bbox = seg["bbox"]
            class_name = seg["class_name"].replace("-", " ")
            nx1 = max(0, min(1, bbox[0]/img_w))
            ny1 = max(0, min(1, bbox[1]/img_h))
            nx2 = max(0, min(1, bbox[2]/img_w))
            ny2 = max(0, min(1, bbox[3]/img_h))
            all_pts.append([nx1, ny1, nx2, ny2])
            objn_parts.append(class_name)

            cx1 = int(nx1 * CONDITION_SIZE[0])
            cy1 = int(ny1 * CONDITION_SIZE[1])
            cx2 = int(nx2 * CONDITION_SIZE[0])
            cy2 = int(ny2 * CONDITION_SIZE[1])
            if use_text_on_mask:
                condition_img = _render_text_in_bbox(condition_img, cx1, cy1, cx2, cy2, class_name)
            else:
                condition_img = _render_single_text_on_mask(condition_img, cx1, cy1, cx2, cy2, class_name)

        # pts for encode_images_cond: list of [x1,y1,x2,y2]
        pts = all_pts if len(all_pts) > 1 else all_pts[0]
        objn = ", ".join(objn_parts)

        # Sanitize objn for filesystem-safe filename (strip \', backslash, and other
        # disallowed-on-SMB chars). Resume support: skip if a panel for this idx exists.
        import re as _re, glob as _glob
        existing = _glob.glob(os.path.join(output_dir, f"mb_{idx:04d}_*.jpg"))
        if existing:
            # Reload the existing panel so downstream metric arrays line up.
            combined = Image.open(existing[0]).convert("RGB")
            cw = combined.width
            cond_img_loaded = combined.crop((0, 0, TARGET_SIZE[0], TARGET_SIZE[1]))
            gen_img = combined.crop((TARGET_SIZE[0], 0, 2*TARGET_SIZE[0], TARGET_SIZE[1]))
            gt_img  = combined.crop((2*TARGET_SIZE[0], 0, 3*TARGET_SIZE[0], TARGET_SIZE[1]))
        else:
            condition = Condition(condition_img, "default", [0, 0], 1.0)
            generator = torch.Generator(device=device).manual_seed(seed)
            mb_prompt_str = _build_class_prompt_multi(objn_parts) if prompt_mode == "class" else ""
            result = generate(pipe, prompt=mb_prompt_str, conditions=[condition],
                height=TARGET_SIZE[1], width=TARGET_SIZE[0],
                num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
                generator=generator, model_config={"independent_condition": False},
                kv_cache=False, pts=pts)
            gen_img = result.images[0]
            gt_img = Image.open(gt_path).convert("RGB").resize(TARGET_SIZE)

            safe_objn = objn[:30].replace("\\'", "'")
            safe_objn = _re.sub(r"[^A-Za-z0-9 ,._-]", "_", safe_objn)
            out_name = f"mb_{idx:04d}_{os.path.basename(gt_path).replace('.jpg','').replace('.png','')}_{safe_objn}.jpg"
            panels = [condition_img.resize(TARGET_SIZE), gen_img, gt_img]
            combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
            for i, p in enumerate(panels):
                combined.paste(p, (i*TARGET_SIZE[0], 0))
            combined.save(os.path.join(output_dir, out_name), quality=95)

        gen_images.append(gen_img)
        gt_images.append(gt_img)
        text_labels.append(objn)
        record_types.append("multibox")

        # Crop first bbox region for region-level metrics
        gw, gh = TARGET_SIZE
        p0 = all_pts[0]
        cx1, cy1 = int(p0[0]*gw), int(p0[1]*gh)
        cx2, cy2 = int(p0[2]*gw), int(p0[3]*gh)
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(gw, cx2), min(gh, cy2)
        if cx2 > cx1 + 4 and cy2 > cy1 + 4:
            gen_crops.append(gen_img.crop((cx1, cy1, cx2, cy2)))
            gt_crops.append(gt_img.crop((cx1, cy1, cx2, cy2)))

        if (idx+1) % 15 == 0 or idx == len(mb_records)-1:
            print(f"  [mb {idx+1}/{len(mb_records)}] {objn[:40]}")

    return gen_images, gt_images, gen_crops, gt_crops, text_labels, record_types

@torch.no_grad()
def compute_metrics(gen_images, gt_images, gen_crops, gt_crops, text_labels, record_types, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from transformers import CLIPModel, CLIPProcessor
    import pyiqa

    to_tensor = T.ToTensor()
    results = {}

    # Filter valid pairs (where GT exists)
    valid_pairs = [(g, gt) for g, gt in zip(gen_images, gt_images) if gt is not None]
    if not valid_pairs:
        print("No valid GT pairs found!")
        return {}

    gen_list = [p[0] for p in valid_pairs]
    gt_list = [p[1] for p in valid_pairs]

    print(f"\nComputing metrics on {len(valid_pairs)} valid pairs...")

    # ── 1. SSIM ──
    print("  Computing SSIM...")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_vals = []
    for g, gt in zip(gen_list, gt_list):
        gt_t = to_tensor(gt).unsqueeze(0).to(device)
        g_t = to_tensor(g.resize(gt.size)).unsqueeze(0).to(device)
        ssim_vals.append(ssim_metric(g_t, gt_t).item())
    results["SSIM"] = np.mean(ssim_vals)

    # ── 2. PSNR ──
    print("  Computing PSNR...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_vals = []
    for g, gt in zip(gen_list, gt_list):
        gt_t = to_tensor(gt).unsqueeze(0).to(device)
        g_t = to_tensor(g.resize(gt.size)).unsqueeze(0).to(device)
        psnr_vals.append(psnr_metric(g_t, gt_t).item())
    results["PSNR"] = np.mean(psnr_vals)

    # ── 3. FID ──
    print("  Computing FID...")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for gt in gt_list:
        gt_t = to_tensor(gt).unsqueeze(0).to(device)
        fid.update(gt_t, real=True)
    for g in gen_list:
        g_t = to_tensor(g.resize(TARGET_SIZE)).unsqueeze(0).to(device)
        fid.update(g_t, real=False)
    try:
        results["FID"] = fid.compute().item()
    except:
        results["FID"] = float("nan")

    # ── 4. MUSIQ ──
    print("  Computing MUSIQ...")
    musiq = pyiqa.create_metric("musiq", device=device)
    musiq_vals = []
    for g in gen_list:
        g_t = to_tensor(g.resize((512, 512))).unsqueeze(0).to(device)
        musiq_vals.append(musiq(g_t).item())
    results["MUSIQ"] = np.mean(musiq_vals)

    # ── 5. CLIP metrics ──
    print("  Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # CLIP-Text: generated image vs text label
    print("  Computing CLIP-Text...")
    clip_text_vals = []
    for g, text in zip(gen_list, [t for t, gt in zip(text_labels, gt_images) if gt is not None]):
        inputs = clip_processor(text=[text], images=g, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        sim = (outputs.logits_per_image / 100.0).item()
        clip_text_vals.append(sim)
    results["CLIP-Text"] = np.mean(clip_text_vals)

    # CLIP-Image: generated full image vs GT full image
    print("  Computing CLIP-Image...")
    clip_img_vals = []
    for g, gt in zip(gen_list, gt_list):
        g_inputs = clip_processor(images=g, return_tensors="pt").to(device)
        gt_inputs = clip_processor(images=gt, return_tensors="pt").to(device)
        g_feat = clip_model.get_image_features(**g_inputs)
        gt_feat = clip_model.get_image_features(**gt_inputs)
        g_feat = F.normalize(g_feat, dim=-1)
        gt_feat = F.normalize(gt_feat, dim=-1)
        sim = (g_feat * gt_feat).sum(dim=-1).item()
        clip_img_vals.append(sim)
    results["CLIP-Image"] = np.mean(clip_img_vals)

    # Region CLIP: generated crop vs text label (mask region only)
    print("  Computing Region CLIP-Text...")
    region_text_vals = []
    valid_text = [t for t, gt in zip(text_labels, gt_images) if gt is not None]
    for gc, text in zip(gen_crops, valid_text[:len(gen_crops)]):
        gc_resized = gc.resize((224, 224))
        inputs = clip_processor(text=[text], images=gc_resized, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        sim = (outputs.logits_per_image / 100.0).item()
        region_text_vals.append(sim)
    results["Region CLIP-T"] = np.mean(region_text_vals) if region_text_vals else float("nan")

    # Region CLIP-I: generated crop vs GT crop (mask region only)
    print("  Computing Region CLIP-Image...")
    region_img_vals = []
    for gc, gtc in zip(gen_crops, gt_crops):
        gc_resized = gc.resize((224, 224))
        gtc_resized = gtc.resize((224, 224))
        g_inputs = clip_processor(images=gc_resized, return_tensors="pt").to(device)
        gt_inputs = clip_processor(images=gtc_resized, return_tensors="pt").to(device)
        g_feat = clip_model.get_image_features(**g_inputs)
        gt_feat = clip_model.get_image_features(**gt_inputs)
        g_feat = F.normalize(g_feat, dim=-1)
        gt_feat = F.normalize(gt_feat, dim=-1)
        sim = (g_feat * gt_feat).sum(dim=-1).item()
        region_img_vals.append(sim)
    results["Region CLIP-I"] = np.mean(region_img_vals) if region_img_vals else float("nan")

    # ── Per-type breakdown ──
    valid_types = [t for t, gt in zip(record_types, gt_images) if gt is not None]
    per_type = defaultdict(lambda: defaultdict(list))

    for i, rtype in enumerate(valid_types):
        per_type[rtype]["SSIM"].append(ssim_vals[i])
        per_type[rtype]["PSNR"].append(psnr_vals[i])
        per_type[rtype]["CLIP-Image"].append(clip_img_vals[i])
        if i < len(clip_text_vals):
            per_type[rtype]["CLIP-Text"].append(clip_text_vals[i])

    results["per_type"] = {}
    for rtype in ["square", "add", "replace", "extract", "multibox"]:
        if rtype in per_type:
            results["per_type"][rtype] = {
                k: np.mean(v) for k, v in per_type[rtype].items()
            }

    del clip_model
    torch.cuda.empty_cache()
    return results

def print_results(results, model_name):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")
    arrows = {"FID": "↓", "SSIM": "↑", "PSNR": "↑", "MUSIQ": "↑",
              "CLIP-Text": "↑", "CLIP-Image": "↑", "Region CLIP-T": "↑", "Region CLIP-I": "↑"}
    for k in ["FID", "SSIM", "PSNR", "MUSIQ", "CLIP-Text", "CLIP-Image", "Region CLIP-T", "Region CLIP-I"]:
        if k in results:
            print(f"  {k}{arrows.get(k,'')}: {results[k]:.4f}")

    if "per_type" in results:
        print(f"\n  Per-type breakdown:")
        for rtype in ["square", "add", "replace", "extract", "multibox"]:
            if rtype in results["per_type"]:
                vals = results["per_type"][rtype]
                parts = [f"{k}={v:.4f}" for k, v in vals.items()]
                print(f"    {rtype:8s}: {', '.join(parts)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint directory containing default.safetensors and omini_adapter.safetensors")
    parser.add_argument("--name", type=str, default=None, help="Run name used for the output directory")
    parser.add_argument("--two-condition", action="store_true", help="Checkpoint trained with TWO_CONDITION=True")
    parser.add_argument("--attn-masking", action="store_true", help="Checkpoint trained with ATTN_MASKING=True")
    parser.add_argument("--sigclip-encoder", action="store_true", help="Checkpoint uses SigLIP instead of the OCR CLIP (ablation R2)")
    parser.add_argument("--vae-only", action="store_true", help="Checkpoint uses only the VAE condition encoder (AB_VAE, ablation R1)")
    parser.add_argument("--prompt-mode", choices=["empty", "class"], default="empty",
                        help="'empty': no text prompt (UniVL); 'class': also pass 'mask contains <label>' to T5 (UniVL+T5)")
    parser.add_argument("--multibox-jsonl", type=str, default=None, help="Override MULTIBOX_JSONL path")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: $UNIVL_OUTPUT_ROOT/eval_<name>_s<seed>)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed for image generation (default {SEED})")
    parser.add_argument("--metrics-only", action="store_true", help="Compute metrics from saved images, skip inference")
    parser.add_argument("--single-box-only", action="store_true", help="Skip multibox eval, only run 200 single-box")
    parser.add_argument("--eval-jsonl", type=str, default=None, help="Override EVAL_JSONL path for the single-box eval set")
    parser.add_argument("--out-dir-suffix", type=str, default="", help="Suffix added to the default output directory name (e.g., '_coco', '_freeform')")
    parser.add_argument("--no-rerender-text", action="store_true", help="Skip in-eval text rendering on cond image (use cond_path as-is)")
    args = parser.parse_args()
    if args.eval_jsonl:
        global EVAL_JSONL
        EVAL_JSONL = args.eval_jsonl
        print(f"EVAL_JSONL overridden to: {EVAL_JSONL}")
    if args.multibox_jsonl:
        global MULTIBOX_JSONL
        MULTIBOX_JSONL = args.multibox_jsonl

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    run_name = args.name or os.path.basename(os.path.normpath(args.ckpt))
    cfg = {
        "name": run_name,
        "ckpt": args.ckpt,
        "use_text_on_mask": True,
        "two_condition": args.two_condition,
        "attn_masking": args.attn_masking,
        "sigclip_encoder": args.sigclip_encoder,
        "ab_vae": args.vae_only,
        "prompt_mode": args.prompt_mode,
    }
    print(f"Model: {cfg['name']}")
    print(f"Checkpoint: {cfg['ckpt']}")
    print(f"Text on mask: {cfg['use_text_on_mask']}, Two condition: {cfg['two_condition']}")

    # Set TWO_CONDITION / ATTN_MASKING / SIGCLIP_ENCODER / AB_VAE flags in pipeline
    import univl.pipeline.flux_univl as flux_module
    flux_module.TWO_CONDITION = cfg["two_condition"]
    if hasattr(flux_module, "ATTN_MASKING"):
        flux_module.ATTN_MASKING = cfg.get("attn_masking", False)
    if hasattr(flux_module, "SIGCLIP_ENCODER"):
        flux_module.SIGCLIP_ENCODER = cfg.get("sigclip_encoder", False)
    if hasattr(flux_module, "AB_VAE"):
        flux_module.AB_VAE = cfg.get("ab_vae", False)
    print(f"Pipeline TWO_CONDITION={flux_module.TWO_CONDITION}, ATTN_MASKING={cfg.get('attn_masking', False)}, "
          f"SIGCLIP_ENCODER={cfg.get('sigclip_encoder', False)}, AB_VAE={cfg.get('ab_vae', False)}")

    records = load_eval_records()
    print(f"Loaded {len(records)} eval records")

    output_dir = args.out_dir or os.path.join(OUTPUT_ROOT, f"eval_{run_name}{args.out_dir_suffix}_s{args.seed}")
    # CLI override: turn off in-eval text re-rendering when cond is already pre-rendered.
    if args.no_rerender_text:
        cfg["use_text_on_mask"] = False
        print("--no-rerender-text: skipping in-eval text rendering on cond image (use cond_path as-is)")
    print(f"Seed: {args.seed}, output_dir: {output_dir}")

    if args.metrics_only:
        print(f"Metrics-only mode: loading images from {output_dir}")
        gen_images, gt_images, gen_crops, gt_crops, text_labels, record_types = [], [], [], [], [], []
        for idx, rec in enumerate(records):
            objn = rec["objn"]
            # Support both "bbox" and "coords" formats
            if "bbox" in rec:
                bbox = rec["bbox"]
                pt1, pt2 = [bbox[0], bbox[1]], [bbox[2], bbox[3]]
            else:
                pt1, pt2 = rec["coords"]
            cond_path = rec["cond_path"]
            gt_path = rec["gt_path"]
            orig_w, orig_h = Image.open(cond_path).size if os.path.exists(cond_path) else (512, 512)
            pts = [pt1[0]/orig_w, pt1[1]/orig_h, pt2[0]/orig_w, pt2[1]/orig_h]
            pts = [max(0.0, min(1.0, p)) for p in pts]

            basename = os.path.basename(cond_path).replace(".png", "").replace(".jpg", "")
            out_name = f"{idx:04d}_{rec['type']}_{basename}_{objn}.jpg"
            out_path = os.path.join(output_dir, out_name)
            if not os.path.exists(out_path):
                continue

            combined = Image.open(out_path).convert("RGB")
            w = combined.width
            panel_w = TARGET_SIZE[0]
            gen_img = combined.crop((panel_w, 0, panel_w*2, combined.height))

            gt_img = None
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert("RGB").resize(TARGET_SIZE)
            elif os.path.exists(gt_path.replace("_r.png", "_r.jpg")):
                gt_img = Image.open(gt_path.replace("_r.png", "_r.jpg")).convert("RGB").resize(TARGET_SIZE)

            gen_images.append(gen_img)
            text_labels.append(objn)
            record_types.append(rec["type"])
            if gt_img is not None:
                gt_images.append(gt_img)
                gw, gh = TARGET_SIZE
                cx1, cy1 = int(pts[0]*gw), int(pts[1]*gh)
                cx2, cy2 = int(pts[2]*gw), int(pts[3]*gh)
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(gw, cx2), min(gh, cy2)
                if cx2 > cx1 + 4 and cy2 > cy1 + 4:
                    gen_crops.append(gen_img.crop((cx1, cy1, cx2, cy2)))
                    gt_crops.append(gt_img.crop((cx1, cy1, cx2, cy2)))
            else:
                gt_images.append(None)
        print(f"Loaded {len(gen_images)} generated images")
    else:
        pipe = load_model(cfg["ckpt"], device,
                          two_condition=cfg["two_condition"],
                          sigclip_encoder=cfg.get("sigclip_encoder", False),
                          ab_vae=cfg.get("ab_vae", False))

        # ── Single-box eval (200 samples) ──
        print(f"\n{'─'*40}\nSingle-box inference ({len(records)} samples)\n{'─'*40}")
        gen_images, gt_images, gen_crops, gt_crops, text_labels, record_types = run_inference(
            pipe, records, output_dir, device,
            use_text_on_mask=cfg["use_text_on_mask"],
            two_condition=cfg["two_condition"],
            prompt_mode=cfg.get("prompt_mode", "empty"),
            seed=args.seed,
        )

        # ── Multibox eval (75 samples) ──
        mb_records = [] if args.single_box_only else load_multibox_records()
        mb_gen, mb_gt, mb_gen_crops, mb_gt_crops, mb_text, mb_types = [], [], [], [], [], []
        if mb_records:
            print(f"\n{'─'*40}\nMultibox inference ({len(mb_records)} samples)\n{'─'*40}")
            mb_gen, mb_gt, mb_gen_crops, mb_gt_crops, mb_text, mb_types = run_multibox_inference(
                pipe, mb_records, output_dir + "_multibox", device,
                use_text_on_mask=cfg["use_text_on_mask"],
                two_condition=cfg["two_condition"],
                prompt_mode=cfg.get("prompt_mode", "empty"),
                seed=args.seed,
            )

        # Free model memory before metrics
        del pipe
        torch.cuda.empty_cache()

    # ── Compute metrics: single-box ──
    print(f"\n{'='*70}\n  Single-box metrics\n{'='*70}")
    results = compute_metrics(gen_images, gt_images, gen_crops, gt_crops, text_labels, record_types, device)
    print_results(results, cfg["name"] + " [single-box]")

    # ── Compute metrics: multibox ──
    mb_results = {}
    if mb_gen:
        print(f"\n{'='*70}\n  Multibox metrics\n{'='*70}")
        mb_results = compute_metrics(mb_gen, mb_gt, mb_gen_crops, mb_gt_crops, mb_text, mb_types, device)
        print_results(mb_results, cfg["name"] + " [multibox]")

    # ── Compute metrics: combined ──
    all_gen = gen_images + mb_gen
    all_gt = gt_images + mb_gt
    all_crops_gen = gen_crops + mb_gen_crops
    all_crops_gt = gt_crops + mb_gt_crops
    all_text = text_labels + mb_text
    all_types = record_types + mb_types
    print(f"\n{'='*70}\n  Combined metrics ({len(all_gen)} total)\n{'='*70}")
    combined_results = compute_metrics(all_gen, all_gt, all_crops_gen, all_crops_gt, all_text, all_types, device)
    print_results(combined_results, cfg["name"] + " [combined]")

    # Save results
    save_all = {
        "single_box": {k: v for k, v in results.items()},
        "multibox": {k: v for k, v in mb_results.items()} if mb_results else {},
        "combined": {k: v for k, v in combined_results.items()},
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(save_all, f, indent=2)
    print(f"\nResults saved to {output_dir}/metrics.json")

if __name__ == "__main__":
    main()
