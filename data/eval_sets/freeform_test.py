#!/usr/bin/env python3
"""Generate 300 irregular-mask test samples from laion-aes+jsons (data/eval_sets/build_freeform_masks.py logic).

For each of 300 random (image, object) pairs:
  - Compute convex hull of the segmentation mask.
  - Render the class name (single centered text) inside the hull.
  - Mask region is filled black with text on top in white.
  - Save to <UNIVL_DATA_ROOT>/textmask_test/ + jsonl.

Output JSONL (one line per sample):
  {"cond_path": "...", "gt_path": "...", "bbox": [x1,y1,x2,y2],
   "resolution": {"height": h, "width": w}, "objn": "...", "type": "irregular"}
"""
import argparse, json, os, random
from univl.paths import DATA_ROOT
import numpy as np
import cv2
import pycocotools.mask as mask_util
from pathlib import Path

JSONS_ROOT = f"{DATA_ROOT}/jsons"
LAION_ROOT = f"{DATA_ROOT}/laion-aes"
OUT_ROOT   = f"{DATA_ROOT}/textmask_test"
OUT_JSONL  = f"{DATA_ROOT}/textmask_test_300.jsonl"

KERNEL = np.ones((3, 3), np.uint8)


def rle_to_mask(rle_str, h, w):
    rle = {"counts": rle_str.encode("utf-8"), "size": [h, w]}
    m = mask_util.decode(rle)
    return cv2.dilate(m, KERNEL, iterations=3)


def get_convex_hull_mask(binary_mask):
    m8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros(m8.shape, dtype=np.uint8)
    for c in contours:
        h = cv2.convexHull(c)
        cv2.drawContours(hull, [h], -1, 255, thickness=cv2.FILLED)
    return (hull > 0).astype(np.uint8)


def render_text_in_mask(image_rgb, mask_bool, class_name):
    """Mask region filled black, single text drawn centered inside."""
    out = image_rgb.copy()
    out[mask_bool] = 0
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return out, None
    y_min, y_max, x_min, x_max = ys.min(), ys.max(), xs.min(), xs.max()
    obj_h, obj_w = y_max - y_min, x_max - x_min
    cx, cy = x_min + obj_w // 2, y_min + obj_h // 2

    font_scale = 3.0
    font_thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(class_name, font, font_scale, font_thickness)
    while tw > obj_w * 1.8 and font_scale > 0.5:
        font_scale -= 0.1
        (tw, th), baseline = cv2.getTextSize(class_name, font, font_scale, font_thickness)

    tx = int(cx - tw / 2); ty = int(cy + th / 2)
    pad = 1
    pt1 = (tx - pad, ty - th - pad); pt2 = (tx + tw + pad, ty + baseline + pad)
    cv2.rectangle(out, pt1, pt2, (0, 0, 0), -1)
    cv2.putText(out, class_name, (tx, ty), font, font_scale, (255, 255, 255), font_thickness)
    return out, (int(x_min), int(y_min), int(x_max), int(y_max))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_parts", type=int, default=4, help="how many part dirs to scan")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Collect candidate JSONs across part dirs
    part_dirs = sorted(p for p in os.listdir(JSONS_ROOT)
                       if p.startswith("part") and os.path.isdir(os.path.join(JSONS_ROOT, p)))
    part_dirs = part_dirs[: args.max_parts]
    all_jsons = []
    for p in part_dirs:
        d = os.path.join(JSONS_ROOT, p)
        for f in os.listdir(d):
            if f.endswith(".json"):
                all_jsons.append(os.path.join(d, f))
    random.shuffle(all_jsons)
    print(f"  {len(all_jsons)} candidate JSONs across {len(part_dirs)} part dirs")

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    out_lines = []
    written = 0

    for jpath in all_jsons:
        if written >= args.n: break
        try:
            with open(jpath) as f: jfile = json.load(f)
            relp = jfile["path"]            # e.g. "00000/00000/000000011.jpg"
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
            rle = obj["mask"]
            raw = rle_to_mask(rle, h, w)
            hull = get_convex_hull_mask(raw)
            mask_bool = hull.astype(bool)
            if mask_bool.sum() < 1000: continue   # skip tiny masks

            out_rgb, bbox = render_text_in_mask(img_rgb, mask_bool, cls)
            if bbox is None: continue

            stem, ext = os.path.splitext(relp)
            cond_rel = f"{stem}_{written:04d}_{cls.replace(' ', '-')}{ext}"
            cond_path = os.path.join(OUT_ROOT, cond_rel)
            Path(cond_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(cond_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

            rec = {
                "cond_path": cond_path,
                "gt_path":   img_path,
                "bbox":      list(bbox),
                "resolution": {"height": h, "width": w},
                "objn":      cls,
                "type":      "irregular",
            }
            out_lines.append(json.dumps(rec))
            written += 1
            if written % 50 == 0:
                print(f"  [{written}/{args.n}]")
        except Exception as e:
            continue

    with open(OUT_JSONL, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nWrote {written} samples + JSONL → {OUT_JSONL}")


if __name__ == "__main__":
    main()
