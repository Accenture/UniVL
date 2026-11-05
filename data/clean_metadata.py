#!/usr/bin/env python3
"""Remove entries with missing cond or GT files from the 4 training JSONs.
Saves cleaned versions as metadata_coords_new.json in each directory.
"""
import json, os
from univl.paths import DATA_ROOT

LAION_BASE = DATA_ROOT + "/laion-aes"

JSONS = {
    "square":  DATA_ROOT + "/textmask_square/metadata_coords.json",
    "add":     DATA_ROOT + "/textmask_add_laion/metadata_coords.json",
    "replace": DATA_ROOT + "/textmask_replace_laion/metadata_coords.json",
    "extract": DATA_ROOT + "/textmask_extract_ref/metadata_coords.json",
}

for name, path in JSONS.items():
    with open(path) as f:
        data = json.load(f)

    json_dir = os.path.dirname(path)
    parent_dir = os.path.dirname(json_dir)
    total = len(data)
    cleaned = {}
    removed = 0

    for cond_path, meta in data.items():
        # Resolve condition path
        if not os.path.isabs(cond_path):
            full_cond = os.path.join(parent_dir, cond_path)
        else:
            full_cond = cond_path

        if not os.path.exists(full_cond):
            removed += 1
            continue

        # Resolve GT path
        gt_exists = True
        if name == "square":
            rel = cond_path.replace("textmask_square/", "")
            bn = os.path.basename(rel)
            img_id = bn.split(".")[0].split("_")[0]
            dir_path = os.path.dirname(rel)
            dir_parts = dir_path.split("/")
            gt_path = os.path.join(LAION_BASE, dir_parts[0], dir_path, f"{img_id}.jpg")
            if not os.path.exists(gt_path):
                gt_exists = False
        else:
            # add/replace/extract: GT = cond_path with _r before extension
            base, ext = os.path.splitext(full_cond)
            gt_path = f"{base}_r{ext}"
            if not os.path.exists(gt_path):
                gt_exists = False

        if not gt_exists:
            removed += 1
            continue

        cleaned[cond_path] = meta

    out_path = os.path.join(json_dir, "metadata_coords_new.json")
    with open(out_path, "w") as f:
        json.dump(cleaned, f)

    print(f"[{name}] {total} → {len(cleaned)} (removed {removed})")
    print(f"  Saved: {out_path}")

print("\nDone.")
