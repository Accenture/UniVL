#!/usr/bin/env python3
"""Regenerate multibox_filtered.jsonl by scanning part dirs and filtering.

Criteria:
- GT image must exist (with tar nesting fix: laion-aes/{first_dir}/{path})
- At least 2 valid segments with:
  - bbox area 4%-40% of image
  - clip_score > 0.01
"""
import os, json, sys
from univl.paths import DATA_ROOT

LAION_AES_ROOT = DATA_ROOT + "/laion-aes"
PART_DIRS = [
    DATA_ROOT + f"/jsons/part{i}_step4_aes_clip_withcount"
    for i in range(0, 8)
]
OUTPUT = DATA_ROOT + "/multibox_filtered.jsonl"

MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.40
MIN_CLIP_SCORE = 0.01
MIN_VALID_SEGMENTS = 2


def resolve_gt_path(rel_path):
    """Resolve GT path with tar nesting fix: laion-aes/{first_dir}/{path}."""
    first_dir = rel_path.split("/")[0]
    return os.path.join(LAION_AES_ROOT, first_dir, rel_path)


def count_valid_segments(data):
    """Count segments meeting area and clip_score criteria."""
    res = data.get("resolution", {})
    img_w = res.get("width", 1)
    img_h = res.get("height", 1)
    img_area = img_w * img_h
    if img_area <= 0:
        return 0

    count = 0
    for seg_type in ["object", "background"]:
        for seg in data.get("segmentation", {}).get(seg_type, []):
            bbox = seg.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            ratio = (bw * bh) / img_area
            if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
                continue
            cs = seg.get("clip_score")
            if cs is not None and cs < MIN_CLIP_SCORE:
                continue
            count += 1
    return count


def main():
    total_scanned = 0
    total_passed = 0
    missing_gt = 0

    with open(OUTPUT, "w") as out_f:
        for d in PART_DIRS:
            if not os.path.isdir(d):
                print(f"SKIP {d} (not found)")
                continue

            files = sorted(os.listdir(d))
            part_passed = 0

            for fn in files:
                total_scanned += 1
                fpath = os.path.join(d, fn)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                except:
                    continue

                # Check GT exists
                rel_path = data.get("path", "")
                gt_path = resolve_gt_path(rel_path)
                if not os.path.exists(gt_path):
                    missing_gt += 1
                    continue

                # Check enough valid segments
                n_valid = count_valid_segments(data)
                if n_valid < MIN_VALID_SEGMENTS:
                    continue

                # Strip heavy mask data — only keep bbox, class_name, clip_score, aes_score
                slim = {
                    "path": data["path"],
                    "resolution": data["resolution"],
                    "aes": data.get("aes"),
                    "cap": data.get("cap"),
                    "segmentation": {}
                }
                for seg_type in ["object", "background"]:
                    segs = data.get("segmentation", {}).get(seg_type, [])
                    slim["segmentation"][seg_type] = [
                        {k: s[k] for k in ["class_name", "bbox", "clip_score", "aes_score", "score"] if k in s}
                        for s in segs
                    ]
                out_f.write(json.dumps(slim) + "\n")
                total_passed += 1
                part_passed += 1

            print(f"  {os.path.basename(d)}: {len(files)} scanned, {part_passed} passed")

    print(f"\nTotal: {total_scanned} scanned, {total_passed} passed, {missing_gt} missing GT")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
