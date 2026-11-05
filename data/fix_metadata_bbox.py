#!/usr/bin/env python3
"""Rebuild metadata_coords JSONs with correct mask bbox.

The current JSONs have TEXT POSITION coordinates as bbox, but we need the
MASK REGION bbox. The mask region is the black area in the condition image
(rendered by the data pipeline). We detect this by finding all pixels < threshold.

Saves results as metadata_coords_fixed.json in each directory.
"""
import json, os, sys
from univl.paths import DATA_ROOT
import numpy as np
from PIL import Image
from collections import OrderedDict

CACHE = DATA_ROOT

JSONS = {
    "square":  f"{CACHE}/textmask_square/metadata_coords_new.json",
    "add":     f"{CACHE}/textmask_add_laion/metadata_coords_new.json",
    "replace": f"{CACHE}/textmask_replace_laion/metadata_coords_new.json",
    "extract": f"{CACHE}/textmask_extract_ref/metadata_coords_new.json",
}

THRESHOLD = 30  # pixels below this are considered "black" (mask region)


def detect_black_bbox(img_path):
    """Detect the bounding box of the black (masked) region in the condition image.
    Returns [x1, y1, x2, y2] in pixel coords, or None if no black region found.
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    gray = arr.mean(axis=2)
    black = gray < THRESHOLD
    rows = np.any(black, axis=1)
    cols = np.any(black, axis=0)
    if not rows.any() or not cols.any():
        return None, img.size
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1]) + 1
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1]) + 1
    return [cmin, rmin, cmax, rmax], img.size


def main():
    for name, json_path in JSONS.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {name}")
        print(f"{'='*60}")

        with open(json_path) as f:
            data = json.load(f)

        total = len(data)
        fixed = OrderedDict()
        skipped = 0
        no_black = 0

        for i, (cond_key, val) in enumerate(data.items()):
            if i % 5000 == 0:
                print(f"  [{i}/{total}]...", flush=True)

            # Resolve condition path
            if name == "square":
                cond_path = os.path.join(CACHE, cond_key)
            else:
                cond_path = cond_key

            if not os.path.exists(cond_path):
                skipped += 1
                continue

            # Detect black region
            bbox, (w, h) = detect_black_bbox(cond_path)
            if bbox is None:
                no_black += 1
                continue

            # Build new entry with correct bbox
            if name == "square":
                # Original format: [[pt1x,pt1y],[pt2x,pt2y]]
                # Keep original text coords as pt1/pt2, add bbox
                if isinstance(val, list):
                    new_val = {
                        "pt1": val[0],
                        "pt2": val[1],
                        "bbox": bbox,
                        "resolution": {"width": w, "height": h}
                    }
                else:
                    new_val = val
                    new_val["bbox"] = bbox
                    new_val["resolution"] = {"width": w, "height": h}
            elif name == "add":
                # Original format: [[[pt1x,pt1y],[pt2x,pt2y]]]
                if isinstance(val, list):
                    pt1 = val[0][0]
                    pt2 = val[0][1]
                    new_val = {
                        "pt1": pt1,
                        "pt2": pt2,
                        "bbox": bbox,
                        "resolution": {"width": w, "height": h}
                    }
                else:
                    new_val = val
                    new_val["bbox"] = bbox
            else:
                # replace/extract: already dict with pt1, pt2, resolution
                if isinstance(val, dict):
                    new_val = val.copy()
                    new_val["bbox"] = bbox
                    if "resolution" not in new_val:
                        new_val["resolution"] = {"width": w, "height": h}
                else:
                    new_val = {
                        "bbox": bbox,
                        "resolution": {"width": w, "height": h}
                    }

            fixed[cond_key] = new_val

        out_path = os.path.join(os.path.dirname(json_path), "metadata_coords_fixed.json")
        with open(out_path, "w") as f:
            json.dump(fixed, f)

        print(f"  Total: {total}, Fixed: {len(fixed)}, Skipped: {skipped}, No black: {no_black}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
