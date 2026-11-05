#!/usr/bin/env python3
"""
Bin samples in multibox_filtered.jsonl by number of bboxes (n_box) and emit
one JSONL per bin in the multibox eval format used by evaluation.eval_univl.run_multibox_inference:

  {"gt_path": ..., "resolution": {"height":..., "width":...},
   "segments": [{"class_name":..., "bbox":[x1,y1,x2,y2]}, ...]}

Bins: n=2, n=3, n=4, n>=5 (capped per --n_per_bin, default 200).

Usage:
    python -m evaluation.mask_count.select_by_mask_count --n_per_bin 200 --seed 42 \\
        --out_dir output_inference/multibox_by_n
"""
import argparse, json, os, random
from univl.paths import DATA_ROOT
from collections import Counter, defaultdict

CACHE = DATA_ROOT
LAION_ROOT = f"{CACHE}/laion-aes"
SRC_JSONL  = f"{CACHE}/multibox_filtered.jsonl"


def load_records(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def to_eval_record(rec):
    """Convert multibox_filtered record to the eval-format expected by
    run_multibox_inference (same as multibox_eval_75.jsonl)."""
    rel = rec.get("path")
    if not rel:
        return None
    gt_path = os.path.join(LAION_ROOT, rel)
    segs = rec.get("segmentation", {}).get("object", [])
    eval_segs = []
    for s in segs:
        if "bbox" not in s or "class_name" not in s:
            continue
        eval_segs.append({"class_name": s["class_name"], "bbox": list(s["bbox"])})
    if not eval_segs:
        return None
    return {
        "gt_path":    gt_path,
        "resolution": rec.get("resolution", {}),
        "segments":   eval_segs,
    }


def n_bin(n):
    if n == 2: return "n2"
    if n == 3: return "n3"
    if n == 4: return "n4"
    if n >= 5: return "n5plus"
    return None  # n<2 ignored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_bin", type=int, default=200)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--check_files", action="store_true",
                    help="Only keep records whose gt_path actually exists on disk")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"Scanning {SRC_JSONL} ...")
    raw_counts = Counter()
    pools = defaultdict(list)
    n_total = 0
    for rec in load_records(SRC_JSONL):
        n_total += 1
        segs = rec.get("segmentation", {}).get("object", [])
        n = len(segs)
        raw_counts[n] += 1
        bin_label = n_bin(n)
        if bin_label is None:
            continue
        ev = to_eval_record(rec)
        if ev is None:
            continue
        pools[bin_label].append(ev)

    print(f"\nTotal records read: {n_total}")
    print(f"Raw n_box histogram (top 10):")
    for n, c in sorted(raw_counts.items()):
        print(f"  n={n:3d}  count={c}")

    print(f"\nBin pool sizes (after dropping records without bbox):")
    for b in ["n2", "n3", "n4", "n5plus"]:
        print(f"  {b:7s} pool={len(pools[b]):>7d}")

    os.makedirs(args.out_dir, exist_ok=True)
    final_counts = {}
    for b in ["n2", "n3", "n4", "n5plus"]:
        candidates = pools[b]
        rng.shuffle(candidates)
        kept = []
        for rec in candidates:
            if len(kept) == args.n_per_bin:
                break
            if (not args.check_files) or os.path.exists(rec["gt_path"]):
                kept.append(rec)
        out_path = os.path.join(args.out_dir, f"multibox_{b}.jsonl")
        with open(out_path, "w") as f:
            for rec in kept:
                f.write(json.dumps(rec) + "\n")
        final_counts[b] = len(kept)
        print(f"  {b:7s} → wrote {len(kept):>4d} records to {out_path}")

    summary_path = os.path.join(args.out_dir, "split_stats.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_total_scanned": n_total,
            "raw_n_box_histogram": dict(sorted(raw_counts.items())),
            "bin_pool_sizes":      {k: len(v) for k, v in pools.items()},
            "final_per_bin":       final_counts,
            "seed": args.seed,
            "n_per_bin_target":    args.n_per_bin,
        }, f, indent=2)
    print(f"\nSaved split summary → {summary_path}")


if __name__ == "__main__":
    main()
