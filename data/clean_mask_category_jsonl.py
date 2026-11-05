#!/usr/bin/env python3
"""Clean square JSONL: remove entries with missing cond or GT files, fix null bytes."""
import json, os
from univl.paths import DATA_ROOT

CACHE = DATA_ROOT
JSONL_PATH = DATA_ROOT + "/textmask_square/metadata_coords_bbox.jsonl"

total = 0
kept = 0
missing_cond = 0
missing_gt = 0
parse_err = 0

clean_lines = []

with open(JSONL_PATH, 'r', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line or line[0] != '{':
            continue
        total += 1
        try:
            entry = json.loads(line)
            k, v = list(entry.items())[0]
        except (json.JSONDecodeError, KeyError, IndexError):
            parse_err += 1
            continue

        # Check cond path
        cond_path = os.path.join(CACHE, k) if not os.path.isabs(k) else k
        if not os.path.exists(cond_path):
            missing_cond += 1
            continue

        # Check GT path
        gt_path = v.get("gt_path", "")
        if gt_path and not os.path.exists(gt_path):
            missing_gt += 1
            continue

        clean_lines.append(json.dumps(entry))
        kept += 1

# Overwrite
"""with open(JSONL_PATH+"1", 'w') as f:
    for line in clean_lines:
        f.write(line + "\n")"""

print(f"Total parsed: {total}")
print(f"Parse errors: {parse_err}")
print(f"Missing cond: {missing_cond}")
print(f"Missing GT:   {missing_gt}")
print(f"Kept:         {kept}")
print(f"Saved:        {JSONL_PATH}")
