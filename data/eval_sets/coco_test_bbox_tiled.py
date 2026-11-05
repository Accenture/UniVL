#!/usr/bin/env python3
"""Generate 200 COCO-val test samples with TILE-PATTERN text rendering inside
the COCO instance's *detection bbox* (rectangular, not segmentation hull).

Matches USE_TEXT_ON_MASK=True training distribution where the "mask" is the
square bbox region filled with tiled text.

Output dir:   <UNIVL_DATA_ROOT>/textmask_coco_test_full
Output JSONL: <UNIVL_DATA_ROOT>/textmask_coco_test_full_200.jsonl
"""
import argparse, json, os, random
from univl.paths import COCO_ROOT, DATA_ROOT
import numpy as np
import cv2
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COCO_ROOT = COCO_ROOT
ANN_PATH  = f"{COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR   = f"{COCO_ROOT}/val2017"
OUT_ROOT  = f"{DATA_ROOT}/textmask_coco_test_full"
OUT_JSONL = f"{DATA_ROOT}/textmask_coco_test_full_200.jsonl"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render_tile_text_in_bbox(image_rgb, bbox_xyxy, class_name):
    """Black out rectangular bbox; tile-pattern text drawn in it (no irregular clip)."""
    out = image_rgb.copy()
    x_min, y_min, x_max, y_max = bbox_xyxy
    pw = x_max - x_min + 1
    ph = y_max - y_min + 1
    if pw < 8 or ph < 8: return out

    out[y_min:y_max+1, x_min:x_max+1] = 0   # black out bbox

    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    font_size = max(4, min(36, ph // 3))
    try:    font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    try:    tw = draw.textlength(class_name, font=font)
    except: tw = font_size * len(class_name) * 0.6
    avail = max(1, pw - 4)
    if tw > avail and tw > 0:
        font_size = max(4, int(font_size * avail / tw))
        try:    font = ImageFont.truetype(FONT_PATH, font_size)
        except: font = ImageFont.load_default()
        try:    tw = draw.textlength(class_name, font=font)
        except: tw = font_size * len(class_name) * 0.6
    tw = max(int(tw), 1)
    y_cur = 2
    while y_cur + font_size <= ph - 2:
        x_cur = 2
        while x_cur < pw - 2:
            draw.text((x_cur, y_cur), class_name, fill=(255, 255, 255), font=font)
            x_cur += tw + 4
        y_cur += font_size + 3

    text_canvas = np.array(patch, dtype=np.uint8)
    text_present = text_canvas.any(axis=-1)
    out_sub = out[y_min:y_max+1, x_min:x_max+1]
    out_sub[text_present] = text_canvas[text_present]
    out[y_min:y_max+1, x_min:x_max+1] = out_sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_area", type=float, default=2500.0,
                    help="min bbox area (w*h) in pixels")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

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

        img_info = coco.imgs[img_id]
        h_img, w_img = img_info["height"], img_info["width"]

        # COCO bbox is [x, y, w, h] → convert to [x1, y1, x2, y2]
        good = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bw * bh < args.min_area: continue
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            x1 = max(0, min(w_img-1, x1)); x2 = max(0, min(w_img-1, x2))
            y1 = max(0, min(h_img-1, y1)); y2 = max(0, min(h_img-1, y2))
            if x2 - x1 < 8 or y2 - y1 < 8: continue
            good.append((a, x1, y1, x2, y2))
        if not good: continue
        ann, x1, y1, x2, y2 = random.choice(good)

        try:
            img_path = os.path.join(IMG_DIR, img_info["file_name"])
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            cls = cat_id_to_name[ann["category_id"]]
            out_rgb = render_tile_text_in_bbox(img_rgb, (x1, y1, x2, y2), cls)

            stem = os.path.splitext(img_info["file_name"])[0]
            cond_name = f"{stem}_{written:04d}_{cls.replace(' ', '-')}.jpg"
            cond_path = os.path.join(OUT_ROOT, cond_name)
            cv2.imwrite(cond_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

            rec = {
                "cond_path": cond_path,
                "gt_path":   img_path,
                "bbox":      [x1, y1, x2, y2],
                "resolution": {"height": h_img, "width": w_img},
                "objn":      cls,
                "type":      "coco_square",
            }
            out_lines.append(json.dumps(rec))
            written += 1
            seen_imgs.add(img_id)
            if written % 25 == 0: print(f"  [{written}/{args.n}]")
        except Exception as e:
            print(f"  ERR img_id {img_id}: {type(e).__name__}: {e}")
            continue

    with open(OUT_JSONL, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nWrote {written} samples + JSONL → {OUT_JSONL}")


if __name__ == "__main__":
    main()
