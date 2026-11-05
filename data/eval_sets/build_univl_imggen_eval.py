#!/usr/bin/env python3
"""Rebuild eval sets with correct mask bbox from updated JSONs.

Outputs:
  - eval_sample_set_200.jsonl  (200 single-box: 50 per type)
  - multibox_eval_75.jsonl     (75 multibox from part jsons)
"""
import json, os, random
from univl.paths import DATA_ROOT, OUTPUT_ROOT
from collections import defaultdict

CACHE = DATA_ROOT
OUT_DIR = OUTPUT_ROOT
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

# ── 1. Single-box (200 samples: 50 per type) ──────────────────────────────

SOURCES = {
    "square": {
        "path": f"{CACHE}/textmask_square/metadata_coords_bbox.jsonl",
        "fmt": "jsonl",
    },
    "add": {
        "path": f"{CACHE}/textmask_add_laion/metadata_coords2.json",
        "fmt": "json",
    },
    "replace": {
        "path": f"{CACHE}/textmask_replace_laion/metadata_coords2.json",
        "fmt": "json",
    },
    "extract": {
        "path": f"{CACHE}/textmask_extract_ref/metadata_coords2.json",
        "fmt": "json",
    },
}


def extract_objn(basename):
    stem = basename.split(".")[0]
    parts = stem.split("_")
    for part in reversed(parts):
        if part == "shrink" or part.isdigit():
            continue
        return part
    return stem


def load_source(name, src):
    """Load records from a source, return list of dicts with consistent format."""
    records = []

    if src["fmt"] == "jsonl":
        with open(src["path"], 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line[0] != '{':
                    continue
                try:
                    entry = json.loads(line)
                    k, v = list(entry.items())[0]
                except:
                    continue
                bbox = v.get("bbox")
                if not bbox:
                    continue
                cond_path = os.path.join(CACHE, k) if not os.path.isabs(k) else k
                gt_path = v.get("gt_path", "")
                res = v.get("resolution", {})
                if not os.path.exists(cond_path):
                    continue
                if gt_path and not os.path.exists(gt_path):
                    continue
                records.append({
                    "cond_path": cond_path,
                    "gt_path": gt_path,
                    "bbox": bbox,
                    "resolution": res,
                    "objn": extract_objn(os.path.basename(k)).replace("-", " "),
                    "type": name,
                })
    else:
        with open(src["path"]) as f:
            data = json.load(f)
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            bbox = v.get("bbox")
            if not bbox:
                continue
            res = v.get("resolution", {})
            if "width" not in res:
                continue
            cond_path = k if os.path.isabs(k) else os.path.join(CACHE, k)
            gt_path = v.get("gt_path")
            if gt_path is None:
                base, ext = os.path.splitext(cond_path)
                gt_path = f"{base}_r{ext}"
            if not os.path.exists(cond_path):
                continue
            if not os.path.exists(gt_path):
                continue
            records.append({
                "cond_path": cond_path,
                "gt_path": gt_path,
                "bbox": bbox,
                "resolution": res,
                "objn": extract_objn(os.path.basename(k)).replace("-", " "),
                "type": name,
            })

    return records


print("Building 200 single-box eval set...")
all_records = []
for name, src in SOURCES.items():
    records = load_source(name, src)
    sampled = random.sample(records, min(50, len(records)))
    all_records.extend(sampled)
    print(f"  {name}: {len(records)} valid → sampled {len(sampled)}")

out_200 = os.path.join(OUT_DIR, "eval_sample_set_200.jsonl")
with open(out_200, 'w') as f:
    for r in all_records:
        f.write(json.dumps(r) + "\n")
print(f"  Saved: {out_200} ({len(all_records)} records)")

# ── 2. Multibox (75 samples) ──────────────────────────────────────────────

MULTIBOX_JSONL = f"{CACHE}/multibox_filtered.jsonl"
LAION_ROOT = f"{CACHE}/laion-aes"

print("\nBuilding 75 multibox eval set...")
if os.path.exists(MULTIBOX_JSONL):
    mb_records = []
    with open(MULTIBOX_JSONL) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except:
                continue
            img_rel = data.get("path", "")
            # Resolve GT path (with tar nesting fix if needed)
            gt_path = os.path.join(LAION_ROOT, img_rel)
            if not os.path.exists(gt_path):
                # Try nesting fix
                first_dir = img_rel.split("/")[0]
                gt_path = os.path.join(LAION_ROOT, first_dir, img_rel)
            if not os.path.exists(gt_path):
                continue

            # Collect valid segments
            res = data.get("resolution", {})
            img_w = res.get("width", 1)
            img_h = res.get("height", 1)
            segments = []
            for seg_type in ["object", "background"]:
                for seg in data.get("segmentation", {}).get(seg_type, []):
                    bbox = seg.get("bbox")
                    if bbox is None or len(bbox) != 4:
                        continue
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    ratio = (bw * bh) / (img_w * img_h) if img_w * img_h > 0 else 0
                    if ratio < 0.04 or ratio > 0.40:
                        continue
                    cs = seg.get("clip_score")
                    if cs is not None and cs < 0.01:
                        continue
                    segments.append({
                        "class_name": seg["class_name"],
                        "bbox": [round(x, 1) for x in bbox],
                    })
            if len(segments) < 2:
                continue
            mb_records.append({
                "gt_path": gt_path,
                "resolution": res,
                "segments": segments[:5],  # cap at 5
            })

    sampled_mb = random.sample(mb_records, min(75, len(mb_records)))
    out_75 = os.path.join(OUT_DIR, "multibox_eval_75.jsonl")
    with open(out_75, 'w') as f:
        for r in sampled_mb:
            f.write(json.dumps(r) + "\n")
    print(f"  {len(mb_records)} valid → sampled {len(sampled_mb)}")
    print(f"  Saved: {out_75}")
else:
    print(f"  SKIP: {MULTIBOX_JSONL} not found")

# ── 3. Verify ──────────────────────────────────────────────────────────────

print("\nVerification:")
with open(out_200) as f:
    records = [json.loads(l) for l in f]
print(f"  Single-box: {len(records)} records")
for t in ["square", "add", "replace", "extract"]:
    n = sum(1 for r in records if r["type"] == t)
    has_bbox = sum(1 for r in records if r["type"] == t and r.get("bbox"))
    has_res = sum(1 for r in records if r["type"] == t and r.get("resolution"))
    print(f"    {t}: {n} records, {has_bbox} with bbox, {has_res} with resolution")

if os.path.exists(out_75):
    with open(out_75) as f:
        mb = [json.loads(l) for l in f]
    print(f"  Multibox: {len(mb)} records")
    seg_counts = [len(r["segments"]) for r in mb]
    print(f"    Segments per image: min={min(seg_counts)}, max={max(seg_counts)}, avg={sum(seg_counts)/len(seg_counts):.1f}")

print("\nDone.")
