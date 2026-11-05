#!/usr/bin/env python3
"""
Sample 4000 GT image paths from the training metadata files (1000 per scenario:
square / add / replace / extract). Output is a JSONL where each line is
  {"type": <scenario>, "gt_path": <abs path>}
suitable as a "real" reference set for FID against the eval8_twocond_40k or
eval200_Cn50 generations.

Usage:
    python3 evaluation/fid/select_reference_images.py --n_per_type 1000 --seed 42 \
        --out output_inference/train_real_4k.jsonl
"""
import argparse, json, os, random
from univl.paths import DATA_ROOT
from collections import Counter

CACHE = DATA_ROOT

# (type, metadata_file, key_format, gt_resolver)
#   key_format: "rel"  -> dict key is rel path under CACHE
#               "abs"  -> dict key is absolute cond path
#   gt_resolver tells how to obtain the gt_path from (key, value):
#               "value_field" : v["gt_path"]
#               "derive_r"    : <base>_r<ext> where (base, ext) = splitext(cond_path)
META = [
    ("square",  f"{CACHE}/textmask_square/metadata_coords_bbox.jsonl", "jsonl", "value_field"),
    ("add",     f"{CACHE}/textmask_add_laion/metadata_coords2.json",   "json",  "derive_r"),
    ("replace", f"{CACHE}/textmask_replace_laion/metadata_coords2.json", "json", "derive_r"),
    ("extract", f"{CACHE}/textmask_extract_ref/metadata_coords2.json",  "json",  "derive_r"),
]


def load_records(path, fmt):
    """Yield (cond_path, value_dict) for every record. Apostrophe-escape fix applied."""
    if fmt == "jsonl":
        with open(path, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                k, v = next(iter(e.items()))
                yield k.replace("\\'", "'"), v
    else:
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            yield k.replace("\\'", "'"), v


def resolve_gt(cond_path, value_dict, mode):
    if mode == "value_field":
        gt = value_dict.get("gt_path", "")
        return gt.replace("\\'", "'") if gt else ""
    elif mode == "derive_r":
        base, ext = os.path.splitext(cond_path)
        return f"{base}_r{ext}"
    return ""


def gt_abs(cond_path):
    """For square the cond_path is a rel key under CACHE; gt_path field in
    the record is already absolute, so we just return as-is from resolve_gt.
    For add/replace/extract, cond_path is already absolute."""
    return cond_path


def _extract_objn(cond_path):
    base = os.path.basename(cond_path).split(".")[0]
    parts = base.split("_")
    for part in reversed(parts):
        if part == "shrink" or part.isdigit():
            continue
        return part.replace("-", " ")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_type", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    selected = []

    for scenario, meta_path, fmt, gt_mode in META:
        if not os.path.exists(meta_path):
            print(f"  [{scenario}] SKIP: missing {meta_path}")
            continue
        print(f"  [{scenario}] scanning {meta_path} ...")
        # candidates: list of dicts {cond_path, gt_path, bbox, objn, type}
        candidates = []
        for raw_cond_path, v in load_records(meta_path, fmt):
            if not isinstance(v, dict):
                continue
            cond_path = raw_cond_path
            # square keys are relative -> absolute under CACHE
            if scenario == "square":
                cond_path = os.path.join(CACHE, cond_path)
            bbox = v.get("bbox")
            if bbox is None:
                continue
            gt = resolve_gt(raw_cond_path if scenario != "square" else os.path.join(CACHE, raw_cond_path),
                            v, gt_mode)
            if not gt:
                continue
            candidates.append({
                "type": scenario,
                "cond_path": cond_path,
                "gt_path": gt,
                "bbox": bbox,
                "objn": _extract_objn(cond_path),
            })
        print(f"  [{scenario}] candidates: {len(candidates)}")

        rng.shuffle(candidates)
        kept = []
        for rec in candidates:
            if len(kept) == args.n_per_type:
                break
            if os.path.exists(rec["cond_path"]) and os.path.exists(rec["gt_path"]):
                kept.append(rec)
        if len(kept) < args.n_per_type:
            print(f"  [{scenario}] WARNING: only found {len(kept)}/{args.n_per_type} existing files")
        selected.extend(kept)

    by_type = Counter(r["type"] for r in selected)
    print(f"\nTotal selected: {len(selected)}  by type: {dict(by_type)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
