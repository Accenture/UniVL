#!/usr/bin/env python3
"""Generate 300 irregular-mask test samples with BLANK (text-free) mask region
— for the OminiControl baseline that wasn't trained on tile-text-in-mask.

For each (image, object) pair:
  - Compute convex hull of the seg mask
  - Black out the irregular hull region (no text rendered)

Output dir:   <UNIVL_DATA_ROOT>/textmask_test_blank
Output JSONL: <UNIVL_DATA_ROOT>/textmask_test_blank_300.jsonl
"""
import argparse, json, os, random
from univl.paths import DATA_ROOT
import numpy as np
import cv2
import pycocotools.mask as mask_util
from pathlib import Path

JSONS_ROOT = f"{DATA_ROOT}/jsons"
LAION_ROOT = f"{DATA_ROOT}/laion-aes"
OUT_ROOT   = f"{DATA_ROOT}/textmask_test_blank"
OUT_JSONL  = f"{DATA_ROOT}/textmask_test_blank_300.jsonl"

KERNEL = np.ones((3, 3), np.uint8)


def rle_to_mask(rle_str, h, w):
    rle = {"counts": rle_str.encode("utf-8"), "size": [h, w]}
    return cv2.dilate(mask_util.decode(rle), KERNEL, iterations=3)


def get_convex_hull_mask(binary_mask):
    m8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros(m8.shape, dtype=np.uint8)
    for c in contours:
        h = cv2.convexHull(c)
        cv2.drawContours(hull, [h], -1, 255, thickness=cv2.FILLED)
    return (hull > 0).astype(np.uint8)


def render_blank(image_rgb, mask_bool):
    out = image_rgb.copy()
    out[mask_bool] = 0
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return out, None
    return out, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_parts", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    part_dirs = sorted(p for p in os.listdir(JSONS_ROOT)
                       if p.startswith("part") and os.path.isdir(os.path.join(JSONS_ROOT, p)))
    part_dirs = part_dirs[: args.max_parts]
    all_jsons = []
    for p in part_dirs:
        d = os.path.join(JSONS_ROOT, p)
        for f in os.listdir(d):
            if f.endswith(".json"): all_jsons.append(os.path.join(d, f))
    random.shuffle(all_jsons)

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    out_lines, written = [], 0
    for jpath in all_jsons:
        if written >= args.n: break
        try:
            with open(jpath) as f: jfile = json.load(f)
            relp = jfile["path"]
            img_path = os.path.join(LAION_ROOT, relp)
            if not os.path.exists(img_path): continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            h = jfile["resolution"]["height"]; w = jfile["resolution"]["width"]
            objs = jfile.get("segmentation", {}).get("object", [])
            if not objs: continue
            obj = random.choice(objs)
            cls = obj["class_name"]
            raw = rle_to_mask(obj["mask"], h, w)
            hull = get_convex_hull_mask(raw)
            mask_bool = hull.astype(bool)
            if mask_bool.sum() < 1000: continue

            out_rgb, bbox = render_blank(img_rgb, mask_bool)
            if bbox is None: continue

            stem, ext = os.path.splitext(relp)
            cond_rel = f"{stem}_{written:04d}_{cls.replace(' ', '-')}{ext}"
            cond_path = os.path.join(OUT_ROOT, cond_rel)
            Path(cond_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(cond_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

            out_lines.append(json.dumps({
                "cond_path": cond_path, "gt_path": img_path, "bbox": list(bbox),
                "resolution": {"height": h, "width": w}, "objn": cls,
                "type": "irregular_blank",
            }))
            written += 1
            if written % 50 == 0: print(f"  [{written}/{args.n}]")
        except Exception:
            continue
    with open(OUT_JSONL, "w") as f: f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {written} samples + JSONL → {OUT_JSONL}")


if __name__ == "__main__":
    main()
