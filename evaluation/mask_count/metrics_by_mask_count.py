#!/usr/bin/env python3
"""
Per-bin multibox metrics. Loads panels saved by evaluation/mask_count/infer_univl_by_mask_count.py
and the matching records JSONL, computes the 8-metric suite (FID, SSIM, PSNR,
MUSIQ, CLIP-Text, CLIP-Image, Region CLIP-T, Region CLIP-I).

Note: FID at N=200 with 2048 features is small-sample-biased (≈ +50–60 over
asymptote), but stable enough not to be NaN.
"""
import argparse, glob, json, os, re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict


TARGET_SIZE = (512, 512)
PANEL_W = 1536  # cond | gen | gt
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def split_panel(img):
    return (img.crop((0,    0, 512,  512)),
            img.crop((512,  0, 1024, 512)),
            img.crop((1024, 0, 1536, 512)))


def parse_idx(name):
    m = re.match(r"mb_(\d+)_", name)
    return int(m.group(1)) if m else None


def load_records(jsonl_path):
    out = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect_panels_with_records(panels_dir, records):
    files = sorted(glob.glob(os.path.join(panels_dir, "*.jpg")))
    print(f"  panels found: {len(files)} in {panels_dir}")
    gen_imgs, gt_imgs, gen_crops, gt_crops = [], [], [], []
    text_labels, types = [], []
    for f in files:
        idx = parse_idx(os.path.basename(f))
        if idx is None or idx >= len(records):
            continue
        rec = records[idx]
        img = Image.open(f).convert("RGB")
        if img.size[0] != PANEL_W:
            continue
        _, gen, gt = split_panel(img)
        gen_imgs.append(gen)
        gt_imgs.append(gt)
        # Build "objn" by joining class names (matches run_multibox_inference)
        segments = rec["segments"]
        objn_parts = [s["class_name"].replace("\\'", "'").replace("-", " ") for s in segments]
        text_labels.append(", ".join(objn_parts))
        types.append("multibox")
        # First bbox for region crop
        res = rec.get("resolution", {})
        img_w = res.get("width", TARGET_SIZE[0])
        img_h = res.get("height", TARGET_SIZE[1])
        bb = segments[0]["bbox"]
        nx1 = max(0, min(1, bb[0] / img_w))
        ny1 = max(0, min(1, bb[1] / img_h))
        nx2 = max(0, min(1, bb[2] / img_w))
        ny2 = max(0, min(1, bb[3] / img_h))
        gw, gh = TARGET_SIZE
        cx1, cy1 = int(nx1*gw), int(ny1*gh)
        cx2, cy2 = int(nx2*gw), int(ny2*gh)
        if cx2 > cx1 + 4 and cy2 > cy1 + 4:
            gen_crops.append(gen.crop((cx1, cy1, cx2, cy2)))
            gt_crops.append(gt.crop((cx1, cy1, cx2, cy2)))
    return gen_imgs, gt_imgs, gen_crops, gt_crops, text_labels, types


@torch.no_grad()
def compute_metrics(gen, gt, gen_crops, gt_crops, text_labels, types, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from transformers import CLIPModel, CLIPProcessor
    import pyiqa

    to_tensor = T.ToTensor()
    out = {}
    n = len(gen)
    print(f"  computing on {n} pairs")

    print("  SSIM ...")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_vals = []
    for g, gt_i in zip(gen, gt):
        gt_t = to_tensor(gt_i).unsqueeze(0).to(device)
        g_t = to_tensor(g.resize(gt_i.size)).unsqueeze(0).to(device)
        ssim_vals.append(ssim_metric(g_t, gt_t).item())
    out["SSIM"] = float(np.mean(ssim_vals))

    print("  PSNR ...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_vals = []
    for g, gt_i in zip(gen, gt):
        gt_t = to_tensor(gt_i).unsqueeze(0).to(device)
        g_t = to_tensor(g.resize(gt_i.size)).unsqueeze(0).to(device)
        psnr_vals.append(psnr_metric(g_t, gt_t).item())
    out["PSNR"] = float(np.mean(psnr_vals))

    print("  FID ...")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    bb = []
    for gt_i in gt:
        bb.append(to_tensor(gt_i.resize(TARGET_SIZE)))
        if len(bb) == 16:
            fid.update(torch.stack(bb).to(device), real=True); bb = []
    if bb: fid.update(torch.stack(bb).to(device), real=True); bb = []
    for g in gen:
        bb.append(to_tensor(g.resize(TARGET_SIZE)))
        if len(bb) == 16:
            fid.update(torch.stack(bb).to(device), real=False); bb = []
    if bb: fid.update(torch.stack(bb).to(device), real=False); bb = []
    try:
        out["FID"] = float(fid.compute().item())
    except Exception as e:
        print(f"  FID failed: {e}"); out["FID"] = float("nan")

    print("  MUSIQ ...")
    musiq = pyiqa.create_metric("musiq", device=device)
    out["MUSIQ"] = float(np.mean([musiq(to_tensor(g.resize((512, 512))).unsqueeze(0).to(device)).item() for g in gen]))

    print("  CLIP loading ...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    print("  CLIP-Text ...")
    clip_text_vals = []
    for g, text in zip(gen, text_labels):
        inputs = clip_processor(text=[text], images=g, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = clip_model(**inputs)
        clip_text_vals.append((outputs.logits_per_image / 100.0).item())
    out["CLIP-Text"] = float(np.mean(clip_text_vals))

    print("  CLIP-Image ...")
    clip_img_vals = []
    for g, gt_i in zip(gen, gt):
        gi  = clip_processor(images=g,    return_tensors="pt").to(device)
        gti = clip_processor(images=gt_i, return_tensors="pt").to(device)
        gf  = F.normalize(clip_model.get_image_features(**gi),  dim=-1)
        gtf = F.normalize(clip_model.get_image_features(**gti), dim=-1)
        clip_img_vals.append((gf * gtf).sum(dim=-1).item())
    out["CLIP-Image"] = float(np.mean(clip_img_vals))

    print("  Region CLIP-T ...")
    region_text_vals = []
    for gc, text in zip(gen_crops, text_labels[:len(gen_crops)]):
        gc_r = gc.resize((224, 224))
        inputs = clip_processor(text=[text], images=gc_r, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = clip_model(**inputs)
        region_text_vals.append((outputs.logits_per_image / 100.0).item())
    out["Region CLIP-T"] = float(np.mean(region_text_vals)) if region_text_vals else float("nan")

    print("  Region CLIP-I ...")
    region_img_vals = []
    for gc, gtc in zip(gen_crops, gt_crops):
        gc_r = gc.resize((224, 224))
        gtc_r = gtc.resize((224, 224))
        gi  = clip_processor(images=gc_r,  return_tensors="pt").to(device)
        gti = clip_processor(images=gtc_r, return_tensors="pt").to(device)
        gf  = F.normalize(clip_model.get_image_features(**gi),  dim=-1)
        gtf = F.normalize(clip_model.get_image_features(**gti), dim=-1)
        region_img_vals.append((gf * gtf).sum(dim=-1).item())
    out["Region CLIP-I"] = float(np.mean(region_img_vals)) if region_img_vals else float("nan")

    del clip_model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels_dir", required=True)
    ap.add_argument("--records",    required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    print(f"\n=== bin={args.bin}  gpu={args.gpu} ===")
    records = load_records(args.records)
    print(f"records: {len(records)}")

    gen, gt, gc, gtc, text, types = collect_panels_with_records(args.panels_dir, records)
    print(f"valid pairs: {len(gen)}, with-bbox-crop: {len(gc)}")
    if not gen:
        print("No panels — exiting.")
        return

    res = compute_metrics(gen, gt, gc, gtc, text, types, device)

    arrows = {"FID": "↓", "SSIM": "↑", "PSNR": "↑", "MUSIQ": "↑",
              "CLIP-Text": "↑", "CLIP-Image": "↑", "Region CLIP-T": "↑", "Region CLIP-I": "↑"}
    print(f"\n=== bin={args.bin} final metrics ===")
    for k in ["FID","SSIM","PSNR","MUSIQ","CLIP-Text","CLIP-Image","Region CLIP-T","Region CLIP-I"]:
        if k in res:
            v = res[k]
            v_str = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
            print(f"  {k}{arrows.get(k,'')}: {v_str}")

    out = {"bin": args.bin, "n_pairs": len(gen), "metrics": res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
