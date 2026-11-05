#!/usr/bin/env python3
"""200 COCO-val samples with BLANK (text-free) rectangular detection-bbox region
— for the OminiControl baseline.

Output dir:   <UNIVL_DATA_ROOT>/textmask_coco_test_blank
Output JSONL: <UNIVL_DATA_ROOT>/textmask_coco_test_blank_200.jsonl
"""
import argparse, json, os, random
from univl.paths import COCO_ROOT, DATA_ROOT
import numpy as np
import cv2
from pycocotools.coco import COCO
from pathlib import Path

COCO_ROOT = COCO_ROOT
ANN_PATH  = f"{COCO_ROOT}/annotations/instances_val2017.json"
IMG_DIR   = f"{COCO_ROOT}/val2017"
OUT_ROOT  = f"{DATA_ROOT}/textmask_coco_test_blank"
OUT_JSONL = f"{DATA_ROOT}/textmask_coco_test_blank_200.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_area", type=float, default=2500.0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    coco = COCO(ANN_PATH)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    out_lines, written = [], 0
    seen = set()
    for img_id in img_ids:
        if written >= args.n: break
        if img_id in seen: continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns: continue
        info = coco.imgs[img_id]
        h_img, w_img = info["height"], info["width"]
        good = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bw * bh < args.min_area: continue
            x1, y1 = max(0, int(bx)), max(0, int(by))
            x2, y2 = min(w_img - 1, int(bx + bw)), min(h_img - 1, int(by + bh))
            if x2 - x1 < 8 or y2 - y1 < 8: continue
            good.append((a, x1, y1, x2, y2))
        if not good: continue
        ann, x1, y1, x2, y2 = random.choice(good)
        try:
            img_path = os.path.join(IMG_DIR, info["file_name"])
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            out = img_bgr.copy()
            out[y1:y2 + 1, x1:x2 + 1] = 0   # blank black bbox
            cls = cat_id_to_name[ann["category_id"]]
            stem = os.path.splitext(info["file_name"])[0]
            cond_name = f"{stem}_{written:04d}_{cls.replace(' ', '-')}.jpg"
            cond_path = os.path.join(OUT_ROOT, cond_name)
            cv2.imwrite(cond_path, out)
            out_lines.append(json.dumps({
                "cond_path": cond_path, "gt_path": img_path,
                "bbox": [x1, y1, x2, y2],
                "resolution": {"height": h_img, "width": w_img},
                "objn": cls, "type": "coco_blank",
            }))
            written += 1
            seen.add(img_id)
            if written % 25 == 0: print(f"  [{written}/{args.n}]")
        except Exception as e:
            print(f"  ERR {img_id}: {e}")
            continue
    with open(OUT_JSONL, "w") as f: f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {written} samples + JSONL → {OUT_JSONL}")


if __name__ == "__main__":
    main()
