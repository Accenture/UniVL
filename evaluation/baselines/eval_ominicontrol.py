#!/usr/bin/env python3
"""
Baseline eval: vanilla OminiControl `fill_old` adapter on the same 200 single-box
+ 75 multi-box eval set used by evaluation/eval_univl.py. Two prompt scenarios:

  --prompt_mode empty   → prompt = ""
  --prompt_mode class   → prompt = "mask contains <class>"  (or compound for multi)

Output goes to:
  $UNIVL_OUTPUT_ROOT/eval_baseline_<prompt_mode>/           (single-box panels + metrics.json)
  $UNIVL_OUTPUT_ROOT/eval_baseline_<prompt_mode>_multibox/  (multibox panels)

Usage:
  python -m evaluation.baselines.eval_ominicontrol --prompt_mode empty --gpu 0
  python -m evaluation.baselines.eval_ominicontrol --prompt_mode class --gpu 1
"""
import argparse, json, os, sys
from univl.paths import OUTPUT_ROOT
from evaluation import eval_univl as r200

import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline
from univl.pipeline.flux_univl import Condition, generate, seed_everything

EVAL_JSONL     = r200.EVAL_JSONL
MULTIBOX_JSONL = r200.MULTIBOX_JSONL
CONDITION_SIZE = r200.CONDITION_SIZE
TARGET_SIZE    = r200.TARGET_SIZE
SEED           = r200.SEED
DTYPE          = r200.DTYPE
NUM_INFERENCE_STEPS = r200.NUM_INFERENCE_STEPS
GUIDANCE_SCALE      = r200.GUIDANCE_SCALE


def load_baseline(device):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                        torch_dtype=DTYPE).to(device)
    pipe.load_lora_weights("Yuanshi/OminiControl",
                           weight_name="experimental/fill.safetensors",
                           adapter_name="fill_old")
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.eval()
    return pipe


def _make_masked(gt_512, bboxes_512):
    masked = gt_512.copy()
    for x1, y1, x2, y2 in bboxes_512:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(TARGET_SIZE[0], x2), min(TARGET_SIZE[1], y2)
        if x2 > x1 and y2 > y1:
            masked.paste((0, 0, 0), (x1, y1, x2, y2))
    return masked


def _build_prompt_single(objn, mode):
    if mode == "empty":
        return ""
    return f"mask contains {objn.replace('-', ' ')}"


def _build_prompt_multi(class_names, mode):
    if mode == "empty":
        return ""
    return " ".join(f"mask{i+1} contains {n.replace('-', ' ')}"
                    for i, n in enumerate(class_names))


@torch.no_grad()
def run_singlebox_baseline(pipe, records, output_dir, device, prompt_mode):
    os.makedirs(output_dir, exist_ok=True)
    gen_imgs, gt_imgs, gen_crops, gt_crops = [], [], [], []
    text_labels, record_types = [], []

    for idx, rec in enumerate(records):
        gt_path = rec["gt_path"]
        objn    = rec["objn"]
        if "bbox" in rec:
            bbox = rec["bbox"]; pt1, pt2 = [bbox[0], bbox[1]], [bbox[2], bbox[3]]
        else:
            pt1, pt2 = rec["coords"]

        # Need a GT to mask out & to compare against
        if not os.path.exists(gt_path):
            alt = gt_path.replace("_r.png", "_r.jpg")
            if os.path.exists(alt):
                gt_path = alt
            else:
                # for square type, gt_path is laion-aes original; cond_path is text-mask version
                cond_path = rec.get("cond_path", "")
                if cond_path and os.path.exists(cond_path):
                    gt_path = cond_path
                else:
                    print(f"  [{idx}] SKIP missing gt {rec['gt_path']}")
                    continue

        gt_img_full = Image.open(gt_path).convert("RGB")
        orig_w, orig_h = gt_img_full.size
        gt_512 = gt_img_full.resize(TARGET_SIZE)

        # Build bbox in 512 coords
        sx = TARGET_SIZE[0] / orig_w
        sy = TARGET_SIZE[1] / orig_h
        bx1, by1 = int(pt1[0]*sx), int(pt1[1]*sy)
        bx2, by2 = int(pt2[0]*sx), int(pt2[1]*sy)

        masked_img = _make_masked(gt_512, [(bx1, by1, bx2, by2)])
        prompt = _build_prompt_single(objn, prompt_mode)

        condition = Condition(masked_img, "fill_old")
        generator = torch.Generator(device=device).manual_seed(SEED)
        result = generate(
            pipe, prompt=prompt, conditions=[condition],
            height=TARGET_SIZE[1], width=TARGET_SIZE[0],
            num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
            generator=generator, model_config={"independent_condition": False},
            kv_cache=False,
        )
        gen_img = result.images[0]

        # Save panel: [masked | gen | gt]
        basename = os.path.basename(gt_path).replace(".jpg", "").replace(".png", "")
        out_name = f"{idx:04d}_{rec['type']}_{basename}_{objn[:20]}.jpg"
        panels = [masked_img, gen_img, gt_512]
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        gen_imgs.append(gen_img)
        gt_imgs.append(gt_512)
        text_labels.append(objn)
        record_types.append(rec["type"])

        # Region crops
        cx1, cy1 = max(0, bx1), max(0, by1)
        cx2, cy2 = min(TARGET_SIZE[0], bx2), min(TARGET_SIZE[1], by2)
        if cx2 > cx1 + 4 and cy2 > cy1 + 4:
            gen_crops.append(gen_img.crop((cx1, cy1, cx2, cy2)))
            gt_crops.append(gt_512.crop((cx1, cy1, cx2, cy2)))

        if (idx+1) % 25 == 0 or idx == len(records)-1:
            print(f"  [{idx+1}/{len(records)}] {rec['type']:8s} {objn}")

    return gen_imgs, gt_imgs, gen_crops, gt_crops, text_labels, record_types


@torch.no_grad()
def run_multibox_baseline(pipe, mb_records, output_dir, device, prompt_mode):
    os.makedirs(output_dir, exist_ok=True)
    import re as _re
    gen_imgs, gt_imgs, gen_crops, gt_crops = [], [], [], []
    text_labels, record_types = [], []

    for idx, rec in enumerate(mb_records):
        gt_path = rec["gt_path"]
        segments = rec["segments"]
        res = rec["resolution"]
        img_w, img_h = res["width"], res["height"]
        if not os.path.exists(gt_path):
            continue

        gt_full = Image.open(gt_path).convert("RGB")
        gt_512 = gt_full.resize(TARGET_SIZE)
        sx = TARGET_SIZE[0] / img_w
        sy = TARGET_SIZE[1] / img_h

        bboxes_512 = []
        class_names = []
        for seg in segments:
            bb = seg["bbox"]
            bx1, by1 = int(bb[0]*sx), int(bb[1]*sy)
            bx2, by2 = int(bb[2]*sx), int(bb[3]*sy)
            bboxes_512.append((bx1, by1, bx2, by2))
            class_names.append(seg["class_name"])

        masked_img = _make_masked(gt_512, bboxes_512)
        prompt = _build_prompt_multi(class_names, prompt_mode)
        objn = ", ".join(c.replace("-", " ") for c in class_names)

        condition = Condition(masked_img, "fill_old")
        generator = torch.Generator(device=device).manual_seed(SEED)
        result = generate(
            pipe, prompt=prompt, conditions=[condition],
            height=TARGET_SIZE[1], width=TARGET_SIZE[0],
            num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
            generator=generator, model_config={"independent_condition": False},
            kv_cache=False,
        )
        gen_img = result.images[0]

        # SMB-safe filename (sanitize objn)
        safe_objn = objn[:30].replace("\\'", "'")
        safe_objn = _re.sub(r"[^A-Za-z0-9 ,._-]", "_", safe_objn)
        out_name = f"mb_{idx:04d}_{os.path.basename(gt_path).replace('.jpg','').replace('.png','')}_{safe_objn}.jpg"
        panels = [masked_img, gen_img, gt_512]
        combined = Image.new("RGB", (TARGET_SIZE[0]*len(panels), TARGET_SIZE[1]))
        for i, p in enumerate(panels):
            combined.paste(p, (i*TARGET_SIZE[0], 0))
        combined.save(os.path.join(output_dir, out_name), quality=95)

        gen_imgs.append(gen_img)
        gt_imgs.append(gt_512)
        text_labels.append(objn)
        record_types.append("multibox")

        # Region crop on first bbox
        cx1, cy1, cx2, cy2 = bboxes_512[0]
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(TARGET_SIZE[0], cx2), min(TARGET_SIZE[1], cy2)
        if cx2 > cx1 + 4 and cy2 > cy1 + 4:
            gen_crops.append(gen_img.crop((cx1, cy1, cx2, cy2)))
            gt_crops.append(gt_512.crop((cx1, cy1, cx2, cy2)))

        if (idx+1) % 15 == 0 or idx == len(mb_records)-1:
            print(f"  [mb {idx+1}/{len(mb_records)}] {objn[:40]}")

    return gen_imgs, gt_imgs, gen_crops, gt_crops, text_labels, record_types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_mode", choices=["empty", "class"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    # Force module-level flags to baseline (no OCR encoder, no two-cond, etc.)
    import univl.pipeline.flux_univl as flux_module
    flux_module.TWO_CONDITION   = False
    if hasattr(flux_module, "AB_VAE"):          flux_module.AB_VAE = False
    if hasattr(flux_module, "SIGCLIP_ENCODER"): flux_module.SIGCLIP_ENCODER = False
    if hasattr(flux_module, "ATTN_MASKING"):    flux_module.ATTN_MASKING = False
    print(f"flags: TWO_COND={flux_module.TWO_CONDITION}, AB_VAE={flux_module.AB_VAE}, "
          f"SIGCLIP_ENCODER={flux_module.SIGCLIP_ENCODER}")

    out_dir = OUTPUT_ROOT + f"/eval_baseline_{args.prompt_mode}"
    out_mb  = out_dir + "_multibox"

    print("Loading FLUX + fill_old LoRA ...")
    pipe = load_baseline(device)

    records = r200.load_eval_records()
    print(f"single-box records: {len(records)}")
    print(f"\n──── Single-box (prompt_mode={args.prompt_mode}) ────")
    gen, gt, gc, gtc, text, types = run_singlebox_baseline(pipe, records, out_dir, device, args.prompt_mode)

    mb_records = r200.load_multibox_records()
    print(f"multibox records: {len(mb_records)}")
    print(f"\n──── Multibox (prompt_mode={args.prompt_mode}) ────")
    mb_gen, mb_gt, mb_gc, mb_gtc, mb_text, mb_types = run_multibox_baseline(
        pipe, mb_records, out_mb, device, args.prompt_mode)

    # Free model memory
    del pipe; torch.cuda.empty_cache()

    print(f"\n══ Single-box metrics ══")
    res_single = r200.compute_metrics(gen, gt, gc, gtc, text, types, device)
    r200.print_results(res_single, f"baseline_{args.prompt_mode} [single-box]")

    print(f"\n══ Multibox metrics ══")
    res_multi = r200.compute_metrics(mb_gen, mb_gt, mb_gc, mb_gtc, mb_text, mb_types, device)
    r200.print_results(res_multi, f"baseline_{args.prompt_mode} [multibox]")

    print(f"\n══ Combined metrics ══")
    res_comb = r200.compute_metrics(
        gen + mb_gen, gt + mb_gt, gc + mb_gc, gtc + mb_gtc,
        text + mb_text, types + mb_types, device,
    )
    r200.print_results(res_comb, f"baseline_{args.prompt_mode} [combined]")

    save = {
        "single_box": res_single,
        "multibox":   res_multi,
        "combined":   res_comb,
        "prompt_mode": args.prompt_mode,
    }
    out_metrics = os.path.join(out_dir, "metrics.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved → {out_metrics}")


if __name__ == "__main__":
    main()
