#!/usr/bin/env python3
"""Compute the 8-metric suite from saved panels.

Panels are 1536×512 = [masked | gen | gt]. Reads single + multi for one method
and writes metrics.json into the single-box output dir.

Usage:
  python \\
      evaluation/metrics_from_panels.py --method brushnet --gpu 0
  ... --method powerpaint --gpu 1
"""
import argparse, json, os, re, sys
from univl.paths import OUTPUT_ROOT
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

REPO_OUT       = OUTPUT_ROOT
EVAL_JSONL     = f"{REPO_OUT}/eval_sample_set_200.jsonl"
MULTIBOX_JSONL = f"{REPO_OUT}/multibox_eval_75.jsonl"
TARGET_SIZE = (512, 512)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def _resolve_gt_path(orig_path):
    if os.path.exists(orig_path): return orig_path
    m = re.match(r"(.*/laion-aes/)(\d+)(/.*)", orig_path)
    if not m: return None
    prefix, _shard, rest = m.group(1), m.group(2), m.group(3)
    for sh in range(0, 100):
        cand = f"{prefix}{sh:05d}{rest}"
        if os.path.exists(cand): return cand
    return None


def _split_panel(panel):
    """Panel is 3-up: [masked | gen | gt] each 512×512."""
    w, h = panel.size
    pw = w // 3
    masked = panel.crop((0, 0, pw, h)).resize(TARGET_SIZE)
    gen    = panel.crop((pw, 0, 2*pw, h)).resize(TARGET_SIZE)
    gt     = panel.crop((2*pw, 0, 3*pw, h)).resize(TARGET_SIZE)
    return masked, gen, gt


def load_singlebox_panels(panel_dir):
    """Iterate eval_sample_set_200.jsonl and load each panel + bbox/label."""
    out = []
    with open(EVAL_JSONL) as f:
        for line in f:
            r = json.loads(line.strip())
            r["__idx"] = len(out)
            out.append(r)
    gen_l, gt_l, gc, gtc, text_l, types_l = [], [], [], [], [], []
    for rec in out:
        idx = rec["__idx"]
        objn = rec["objn"]
        if "bbox" in rec: bbox = rec["bbox"]; pt1 = (bbox[0], bbox[1]); pt2 = (bbox[2], bbox[3])
        else: pt1, pt2 = rec["coords"]
        gt_path = rec["gt_path"]
        if not os.path.exists(gt_path):
            alt = gt_path.replace("_r.png", "_r.jpg")
            if os.path.exists(alt): gt_path = alt
        if not os.path.exists(gt_path):
            cond = rec.get("cond_path", "")
            if cond and os.path.exists(cond): gt_path = cond
            else: continue
        gt_full = Image.open(gt_path).convert("RGB")
        orig_w, orig_h = gt_full.size
        sx, sy = TARGET_SIZE[0]/orig_w, TARGET_SIZE[1]/orig_h
        bx1, by1 = int(pt1[0]*sx), int(pt1[1]*sy)
        bx2, by2 = int(pt2[0]*sx), int(pt2[1]*sy)

        basename = os.path.basename(gt_path).replace(".jpg", "").replace(".png", "")
        out_name = f"{idx:04d}_{rec['type']}_{basename}_{objn[:20]}.jpg"
        panel_p = os.path.join(panel_dir, out_name)
        if not os.path.exists(panel_p): continue
        _, gen, gt = _split_panel(Image.open(panel_p).convert("RGB"))

        gen_l.append(gen); gt_l.append(gt); text_l.append(objn); types_l.append(rec["type"])
        cx1, cy1 = max(0, bx1), max(0, by1)
        cx2, cy2 = min(TARGET_SIZE[0], bx2), min(TARGET_SIZE[1], by2)
        if cx2 > cx1+4 and cy2 > cy1+4:
            gc.append(gen.crop((cx1, cy1, cx2, cy2)))
            gtc.append(gt.crop((cx1, cy1, cx2, cy2)))
    return gen_l, gt_l, gc, gtc, text_l, types_l


def load_multibox_panels(panel_dir):
    out = []
    with open(MULTIBOX_JSONL) as f:
        for line in f:
            out.append(json.loads(line.strip()))
    gen_l, gt_l, gc, gtc, text_l, types_l = [], [], [], [], [], []
    for idx, rec in enumerate(out):
        gt_path = _resolve_gt_path(rec["gt_path"])
        if gt_path is None: continue
        segments = rec["segments"]
        res = rec["resolution"]; img_w, img_h = res["width"], res["height"]
        sx, sy = TARGET_SIZE[0]/img_w, TARGET_SIZE[1]/img_h
        bboxes_512 = []; class_names = []
        for seg in segments:
            bb = seg["bbox"]
            bx1, by1 = int(bb[0]*sx), int(bb[1]*sy)
            bx2, by2 = int(bb[2]*sx), int(bb[3]*sy)
            bboxes_512.append((bx1, by1, bx2, by2))
            class_names.append(seg["class_name"])
        objn = ", ".join(c.replace("-", " ") for c in class_names)
        safe_objn = re.sub(r"[^A-Za-z0-9 ,._-]", "_", objn[:30].replace("\\'", "'"))
        out_name = f"mb_{idx:04d}_{os.path.basename(gt_path).replace('.jpg','').replace('.png','')}_{safe_objn}.jpg"
        panel_p = os.path.join(panel_dir, out_name)
        if not os.path.exists(panel_p): continue
        _, gen, gt = _split_panel(Image.open(panel_p).convert("RGB"))
        gen_l.append(gen); gt_l.append(gt); text_l.append(objn); types_l.append("multibox")
        cx1, cy1, cx2, cy2 = bboxes_512[0]
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(TARGET_SIZE[0], cx2), min(TARGET_SIZE[1], cy2)
        if cx2 > cx1+4 and cy2 > cy1+4:
            gc.append(gen.crop((cx1, cy1, cx2, cy2)))
            gtc.append(gt.crop((cx1, cy1, cx2, cy2)))
    return gen_l, gt_l, gc, gtc, text_l, types_l


@torch.no_grad()
def compute_metrics(gen, gt, gen_crops, gt_crops, text_labels, types, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from transformers import CLIPModel, CLIPProcessor
    import pyiqa
    to_tensor = T.ToTensor()
    out = {}
    n = len(gen)
    if n == 0:
        return {"SSIM": float("nan"), "PSNR": float("nan"), "FID": float("nan"),
                "MUSIQ": float("nan"), "CLIP-Text": float("nan"), "CLIP-Image": float("nan"),
                "Region CLIP-T": float("nan"), "Region CLIP-I": float("nan")}
    print(f"\nComputing metrics on {n} pairs ...", flush=True)
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
        gi = clip_processor(images=g, return_tensors="pt").to(device)
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
    ap.add_argument("--method", required=True, choices=["brushnet", "powerpaint"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    panel_dir_single = f"{REPO_OUT}/eval_{args.method}"
    panel_dir_multi  = f"{REPO_OUT}/eval_{args.method}_multibox"
    print(f"Loading single-box panels from {panel_dir_single}")
    s_gen, s_gt, s_gc, s_gtc, s_text, s_types = load_singlebox_panels(panel_dir_single)
    print(f"  {len(s_gen)} pairs loaded")
    print(f"Loading multibox panels from {panel_dir_multi}")
    m_gen, m_gt, m_gc, m_gtc, m_text, m_types = load_multibox_panels(panel_dir_multi)
    print(f"  {len(m_gen)} pairs loaded")

    print("\n══ Single-box metrics ══", flush=True)
    res_s = compute_metrics(s_gen, s_gt, s_gc, s_gtc, s_text, s_types, device)
    print("\n══ Multibox metrics ══", flush=True)
    res_m = compute_metrics(m_gen, m_gt, m_gc, m_gtc, m_text, m_types, device)
    print("\n══ Combined metrics ══", flush=True)
    res_c = compute_metrics(s_gen+m_gen, s_gt+m_gt, s_gc+m_gc, s_gtc+m_gtc,
                            s_text+m_text, s_types+m_types, device)

    summary = {"single_box": res_s, "multibox": res_m, "combined": res_c,
               "method": args.method}
    out_p = f"{panel_dir_single}/metrics.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_p}")
    for split in ["single_box","multibox","combined"]:
        d = summary[split]
        print(f"\n[{split}]")
        for k in ["FID","SSIM","PSNR","MUSIQ","CLIP-Text","CLIP-Image","Region CLIP-T","Region CLIP-I"]:
            v = d.get(k, float("nan"))
            if isinstance(v, float) and v != v: print(f"  {k}: nan")
            else: print(f"  {k}: {v:.4f}" if abs(v) < 10 else f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
