#!/usr/bin/env python3
"""Generate 200 COCO-val test samples with text rendered in instance-mask region.

For each of 200 random (image, instance) pairs from COCO val2017:
  - Decode COCO instance segmentation → binary mask
  - Compute convex hull of the mask
  - Render the COCO category name centered inside the hull
  - Mask region filled black with text on top in white
  - Save to <UNIVL_DATA_ROOT>/textmask_coco_test/ + JSONL

Output JSONL line:
  {"cond_path": "...", "gt_path": "...", "bbox": [x1,y1,x2,y2],
   "resolution": {...}, "objn": "<category>", "type": "coco"}
"""
import argparse, json, os, random
from univl.paths import COCO_ROOT, DATA_ROOT
import numpy as np
import cv2
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pathlib import Path

COCO_ROOT = COCO_ROOT
ANN_PATH  = f"{COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR   = f"{COCO_ROOT}/val2017"
OUT_ROOT  = f"{DATA_ROOT}/textmask_coco_test"
OUT_JSONL = f"{DATA_ROOT}/textmask_coco_test_200.jsonl"

KERNEL = np.ones((3, 3), np.uint8)


def get_convex_hull_mask(binary_mask):
    m8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros(m8.shape, dtype=np.uint8)
    for c in contours:
        h = cv2.convexHull(c)
        cv2.drawContours(hull, [h], -1, 255, thickness=cv2.FILLED)
    return (hull > 0).astype(np.uint8)


def render_text_in_mask(image_rgb, mask_bool, class_name):
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
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_area", type=int, default=2500, help="min mask area in pixels")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    coco = COCO(ANN_PATH)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    print(f"  {len(img_ids)} images in COCO val2017")

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    out_lines = []
    written = 0
    seen_imgs = set()

    for img_id in img_ids:
        if written >= args.n: break
        if img_id in seen_imgs: continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns: continue
        # Filter by area
        anns = [a for a in anns if a.get("area", 0) >= args.min_area]
        if not anns: continue
        ann = random.choice(anns)
        try:
            seg = ann["segmentation"]
            img_info = coco.imgs[img_id]
            h, w = img_info["height"], img_info["width"]
            if isinstance(seg, list):  # polygon
                rles = mask_util.frPyObjects(seg, h, w)
                rle = mask_util.merge(rles) if isinstance(rles, list) else rles
            else:
                rle = seg if isinstance(seg, dict) else mask_util.frPyObjects(seg, h, w)
            m = mask_util.decode(rle)
            if m.ndim == 3: m = m[..., 0]
            m = cv2.dilate(m, KERNEL, iterations=2)

            hull = get_convex_hull_mask(m)
            mask_bool = hull.astype(bool)
            if mask_bool.sum() < args.min_area: continue

            img_path = os.path.join(IMG_DIR, img_info["file_name"])
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            cls = cat_id_to_name[ann["category_id"]]
            out_rgb, bbox = render_text_in_mask(img_rgb, mask_bool, cls)
            if bbox is None: continue

            stem = os.path.splitext(img_info["file_name"])[0]
            cond_name = f"{stem}_{written:04d}_{cls.replace(' ', '-')}.jpg"
            cond_path = os.path.join(OUT_ROOT, cond_name)
            cv2.imwrite(cond_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

            rec = {
                "cond_path": cond_path,
                "gt_path":   img_path,
                "bbox":      list(bbox),
                "resolution": {"height": h, "width": w},
                "objn":      cls,
                "type":      "coco",
            }
            out_lines.append(json.dumps(rec))
            written += 1
            seen_imgs.add(img_id)
            if written % 25 == 0:
                print(f"  [{written}/{args.n}]")
        except Exception as e:
            print(f"  ERR img_id {img_id}: {type(e).__name__}: {e}")
            continue

    with open(OUT_JSONL, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nWrote {written} samples + JSONL → {OUT_JSONL}")


if __name__ == "__main__":
    main()
