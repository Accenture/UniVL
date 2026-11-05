#!/usr/bin/env python3
"""Generate 300 irregular-mask test samples with TILE-PATTERN text rendering
(matches USE_TEXT_ON_MASK=True training distribution: text fills the mask region).

For each (image, object) pair:
  - Compute convex hull of segmentation mask
  - Black out the mask region in the cond image
  - Render a TILE pattern of class-name text inside the bbox of the hull,
    then clip the text-only pixels to the irregular hull mask so the text
    only appears inside the actual masked area.

Output dir:   <UNIVL_DATA_ROOT>/textmask_test_full
Output JSONL: <UNIVL_DATA_ROOT>/textmask_test_full_300.jsonl
"""
import argparse, json, os, random
from univl.paths import DATA_ROOT
import numpy as np
import cv2
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

JSONS_ROOT = f"{DATA_ROOT}/jsons"
LAION_ROOT = f"{DATA_ROOT}/laion-aes"
OUT_ROOT   = f"{DATA_ROOT}/textmask_test_full"
OUT_JSONL  = f"{DATA_ROOT}/textmask_test_full_300.jsonl"

KERNEL = np.ones((3, 3), np.uint8)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


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


def render_tile_text_in_irregular_mask(image_rgb, mask_bool, class_name):
    """
    Mask region zeroed; tile-pattern text drawn in the hull-bbox; text pixels
    clipped to the irregular mask so only the masked area carries the text.

    Returns (out_rgb, bbox=(x1,y1,x2,y2)) or (out_rgb, None) if mask too small.
    """
    out = image_rgb.copy()
    out[mask_bool] = 0

    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return out, None
    y_min, y_max, x_min, x_max = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    pw = x_max - x_min + 1
    ph = y_max - y_min + 1
    if pw < 8 or ph < 8: return out, None

    # Build a black PIL patch the size of the hull bbox and tile text on it.
    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    font_size = max(4, min(36, ph // 3))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        tw = draw.textlength(class_name, font=font)
    except Exception:
        tw = font_size * len(class_name) * 0.6
    avail = max(1, pw - 4)
    if tw > avail and tw > 0:
        font_size = max(4, int(font_size * avail / tw))
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()
        try:
            tw = draw.textlength(class_name, font=font)
        except Exception:
            tw = font_size * len(class_name) * 0.6
    tw = max(int(tw), 1)
    y_cur = 2
    while y_cur + font_size <= ph - 2:
        x_cur = 2
        while x_cur < pw - 2:
            draw.text((x_cur, y_cur), class_name, fill=(255, 255, 255), font=font)
            x_cur += tw + 4
        y_cur += font_size + 3

    # Clip the text canvas to the irregular hull mask, then paste into out.
    text_canvas = np.array(patch, dtype=np.uint8)                # ph x pw x 3
    text_present = text_canvas.any(axis=-1)                      # ph x pw bool
    submask = mask_bool[y_min:y_max+1, x_min:x_max+1]            # ph x pw bool
    pixels = text_present & submask
    out_sub = out[y_min:y_max+1, x_min:x_max+1]
    out_sub[pixels] = text_canvas[pixels]
    out[y_min:y_max+1, x_min:x_max+1] = out_sub
    return out, (x_min, y_min, x_max, y_max)


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
            rle = obj["mask"]
            raw = rle_to_mask(rle, h, w)
            hull = get_convex_hull_mask(raw)
            mask_bool = hull.astype(bool)
            if mask_bool.sum() < 1000: continue

            out_rgb, bbox = render_tile_text_in_irregular_mask(img_rgb, mask_bool, cls)
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
            if written % 50 == 0: print(f"  [{written}/{args.n}]")
        except Exception:
            continue

    with open(OUT_JSONL, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nWrote {written} samples + JSONL → {OUT_JSONL}")


if __name__ == "__main__":
    main()
