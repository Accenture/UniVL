#!/usr/bin/env python3
"""Update metadata_coords_new.json with correct mask bbox from segmentation data.

1. Build GT→condition_paths mapping from metadata_coords_new.json
2. For each part JSON, decode RLE mask → rectangular bbox
3. Match to condition paths via GT image + class_name + index
4. Save updated JSON as metadata_coords_bbox.json

Does NOT save images — only updates bbox coordinates.
"""
import os
from univl.paths import DATA_ROOT
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
from tqdm import tqdm

kernel = np.ones((3, 3), np.uint8)

CACHE = DATA_ROOT
INPUT_JSON = f"{CACHE}/textmask_square/metadata_coords_new.json"
OUTPUT_JSON = f"{CACHE}/textmask_square/metadata_coords_bbox.json"
OUTPUT_JSONL = f"{CACHE}/textmask_square/metadata_coords_bbox.jsonl"  # intermediate check


def rle_to_mask(rle_str, height, width):
    """Converts RLE string to a NumPy binary mask (with dilation)."""
    rle = {"counts": rle_str.encode("utf-8"), "size": [height, width]}
    mask = mask_util.decode(rle)
    # Match the original data_process_square2.py dilation
    np.random.seed(42)
    mask = cv2.dilate(mask, kernel, iterations=np.random.randint(0, 5))
    return mask


def get_rect_bbox(binary_mask):
    """Get [x_min, y_min, x_max, y_max] from binary mask."""
    ys, xs = np.where(binary_mask > 0)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    # 1. Load existing JSON
    print(f"Loading {INPUT_JSON}...")
    with open(INPUT_JSON) as f:
        data = json.load(f)
    print(f"  {len(data)} entries")

    # 2. Build mapping: GT_image_path → {(index, class_name): cond_key}
    # condition key format: "textmask_square/00000/00000/000000011_0_Ford-Mustang.jpg"
    # GT path derived from condition key (strip class and index)
    gt_to_conds = {}  # gt_rel_path → list of (index, class_name, cond_key)
    for cond_key in data.keys():
        rel_path = cond_key.replace("textmask_square/", "")
        bn = os.path.basename(rel_path)
        stem = bn.split(".")[0]  # "000000011_0_Ford-Mustang"
        parts = stem.split("_")

        # Extract: image_id, index, class_name
        # Format: {image_id}_{index}_{class_name}
        # image_id = parts[0], index = parts[1], class = "_".join(parts[2:])
        if len(parts) < 3:
            continue
        image_id = parts[0]
        try:
            obj_index = int(parts[1])
        except ValueError:
            continue
        class_name = "-".join(parts[2:])  # rejoin multi-part class names

        # GT relative path: dir_path/image_id.ext
        dir_path = os.path.dirname(rel_path)
        ext = bn.split(".")[-1]
        gt_rel = os.path.join(dir_path, f"{image_id}.{ext}")

        if gt_rel not in gt_to_conds:
            gt_to_conds[gt_rel] = []
        gt_to_conds[gt_rel].append((obj_index, class_name, cond_key))

    print(f"  {len(gt_to_conds)} unique GT images")

    # 3. Scan part JSONs and update bbox
    updated = 0
    not_found = 0
    f_jsonl = open(OUTPUT_JSONL, 'w')

    for part_id in range(8):
        part_dir = os.path.join(CACHE, f"jsons/part{part_id}_step4_aes_clip_withcount")
        if not os.path.isdir(part_dir):
            # Try alternate location
            part_dir = DATA_ROOT + f"/jsons/part{part_id}_step4_aes_clip_withcount"
            if not os.path.isdir(part_dir):
                print(f"  SKIP part{part_id} (not found)")
                continue

        filelist = sorted([f for f in os.listdir(part_dir) if f.endswith('.json')])
        print(f"  part{part_id}: {len(filelist)} files")

        for jfilename in tqdm(filelist, desc=f"part{part_id}"):
            try:
                with open(os.path.join(part_dir, jfilename)) as f:
                    jfile = json.load(f)

                img_rel = jfile['path']  # e.g. "00000/00000/000000011.jpg"

                # Check if any condition images map to this GT
                if img_rel not in gt_to_conds:
                    continue

                h = jfile['resolution']['height']
                w = jfile['resolution']['width']
                gt_path = os.path.join(CACHE, "laion-aes", img_rel)

                objects = jfile.get('segmentation', {}).get('object', [])

                # Build index→bbox mapping from segmentation
                np.random.seed(42)
                for i_, obj in enumerate(objects):
                    class_name = obj['class_name']
                    rle_str = obj.get('mask')
                    if rle_str is None:
                        continue

                    raw_mask = rle_to_mask(rle_str, h, w)
                    bbox = get_rect_bbox(raw_mask)
                    if bbox is None:
                        continue

                    x_min, y_min, x_max, y_max = bbox
                    # Skip tiny masks (same threshold as data_process_square2.py)
                    ratio = (x_max - x_min) * (y_max - y_min) / (h * w)
                    if ratio <= 0.004:
                        continue

                    # Match to condition entries
                    class_name_dash = class_name.replace(" ", "-")
                    for (obj_idx, cond_class, cond_key) in gt_to_conds[img_rel]:
                        if obj_idx == i_ and cond_class == class_name_dash:
                            # Update bbox in the JSON
                            old_val = data[cond_key]
                            if isinstance(old_val, list):
                                # Convert old format [[pt1],[pt2]] to dict
                                data[cond_key] = {
                                    "pt1": old_val[0],
                                    "pt2": old_val[1],
                                    "bbox": [x_min, y_min, x_max, y_max],
                                    "resolution": {"height": h, "width": w},
                                    "gt_path": gt_path,
                                }
                            elif isinstance(old_val, dict):
                                old_val["bbox"] = [x_min, y_min, x_max, y_max]
                                if "resolution" not in old_val:
                                    old_val["resolution"] = {"height": h, "width": w}
                                if "gt_path" not in old_val:
                                    old_val["gt_path"] = gt_path
                            updated += 1
                            # Write to JSONL for intermediate checking
                            entry = {cond_key: data[cond_key]}
                            f_jsonl.write(json.dumps(entry) + "\n")
                            if updated % 1000 == 0:
                                f_jsonl.flush()

            except Exception as e:
                print(f"  Error: {jfilename}: {e}")
                continue

    f_jsonl.close()
    print(f"JSONL saved: {OUTPUT_JSONL} ({updated} entries)")

    # 4. Count entries still without bbox
    no_bbox = sum(1 for v in data.values()
                  if (isinstance(v, list)) or (isinstance(v, dict) and "bbox" not in v))
    print(f"\nUpdated: {updated}, No bbox: {no_bbox}, Total: {len(data)}")

    # 5. Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f)
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
