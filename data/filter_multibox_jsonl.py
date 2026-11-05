#!/usr/bin/env python3
"""Pre-filter multibox JSONs: remove masks with area < 4% of image, check GT exists.
Writes a single filtered JSONL file for fast loading during training.
"""
import os
from univl.paths import DATA_ROOT
import json
import sys
from collections import defaultdict

PART_DIRS = [
    DATA_ROOT + f"/jsons/part{i}_step4_aes_clip_withcount"
    for i in range(0, 8)
]
LAION_AES_ROOT = DATA_ROOT + "/laion-aes"
OUTPUT_JSONL = DATA_ROOT + "/multibox_filtered.jsonl"
MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.40
MIN_CLIP_SCORE = 0.01


def main():
    stats = defaultdict(int)
    written = 0

    with open(OUTPUT_JSONL, "w") as out_f:
        for part_dir in PART_DIRS:
            if not os.path.isdir(part_dir):
                print(f"SKIP: {part_dir}")
                continue
            part_name = os.path.basename(part_dir)
            files = sorted(os.listdir(part_dir))
            print(f"\nProcessing {part_name}: {len(files)} files...")
            stats["total_jsons"] += len(files)

            for fi, fn in enumerate(files):
                if fi % 10000 == 0 and fi > 0:
                    print(f"  {fi}/{len(files)}...")

                fpath = os.path.join(part_dir, fn)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                except:
                    stats["json_error"] += 1
                    continue

                # Check GT exists
                img_rel = data.get("path", "")
                gt_path = os.path.join(LAION_AES_ROOT, img_rel)
                if not os.path.exists(gt_path):
                    stats["gt_missing"] += 1
                    continue

                res = data.get("resolution", {})
                img_w = res.get("width", 0)
                img_h = res.get("height", 0)
                if img_w <= 0 or img_h <= 0:
                    stats["bad_resolution"] += 1
                    continue

                img_area = img_w * img_h

                # Collect valid segments
                valid_segs = []
                total_segs = 0
                for seg_type in ["object", "background"]:
                    for seg in data.get("segmentation", {}).get(seg_type, []):
                        total_segs += 1
                        bbox = seg.get("bbox")
                        if bbox is None or len(bbox) != 4:
                            stats["no_bbox"] += 1
                            continue

                        bw = max(0, bbox[2] - bbox[0])
                        bh = max(0, bbox[3] - bbox[1])
                        area_ratio = (bw * bh) / img_area

                        if area_ratio < MIN_AREA_RATIO:
                            stats["too_small"] += 1
                            continue
                        if area_ratio > MAX_AREA_RATIO:
                            stats["too_large"] += 1
                            continue

                        clip_score = seg.get("clip_score") or 0.0
                        if clip_score < MIN_CLIP_SCORE:
                            stats["low_clip"] += 1
                            continue

                        valid_segs.append({
                            "class_name": seg["class_name"],
                            "bbox": [round(x, 2) for x in bbox],
                            "clip_score": round(clip_score, 4),
                            "seg_type": seg_type,
                        })

                stats["total_segs"] += total_segs
                stats["valid_segs"] += len(valid_segs)

                # Need at least 4 valid segments (to sample 2-5 non-overlapping)
                if len(valid_segs) <= 3:
                    stats["too_few_segs"] += 1
                    continue

                # Write filtered record
                record = {
                    "path": img_rel,
                    "resolution": {"width": img_w, "height": img_h},
                    "segments": valid_segs,
                }
                out_f.write(json.dumps(record) + "\n")
                written += 1

                if len(valid_segs) >= 2:
                    stats["has_2plus"] += 1
                if len(valid_segs) >= 3:
                    stats["has_3plus"] += 1

    print(f"\n{'='*60}")
    print(f"DONE. Written {written} records to {OUTPUT_JSONL}")
    print(f"{'='*60}")
    print(f"  Total JSONs processed: {stats['total_jsons']}")
    print(f"  GT missing:            {stats['gt_missing']}")
    print(f"  Bad resolution:        {stats['bad_resolution']}")
    print(f"  JSON errors:           {stats['json_error']}")
    print(f"  Total segments:        {stats['total_segs']}")
    print(f"  Filtered (too small):  {stats['too_small']} (area < {MIN_AREA_RATIO*100:.0f}%)")
    print(f"  Filtered (too large):  {stats['too_large']} (area > {MAX_AREA_RATIO*100:.0f}%)")
    print(f"  Filtered (low CLIP):   {stats['low_clip']} (clip < {MIN_CLIP_SCORE})")
    print(f"  Valid segments kept:   {stats['valid_segs']}")
    print(f"  Images with 2+ segs:   {stats['has_2plus']}")
    print(f"  Images with 3+ segs:   {stats['has_3plus']}")
    print(f"  Too few segs (<=3):    {stats['too_few_segs']}")
    print(f"  Final images:          {written}")


if __name__ == "__main__":
    main()
