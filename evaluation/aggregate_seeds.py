#!/usr/bin/env python3
"""Aggregate metrics.json files across seeds and report mean ± std.

Reads:  output_inference/eval200_<MODEL>{,_s<SEED>}/metrics.json
Splits: single_box, multibox, combined
Output: markdown table with per-config mean ± std across seeds.
"""
import json, os, sys, math, argparse
from univl.paths import OUTPUT_ROOT
from collections import defaultdict

REPO_OUT = OUTPUT_ROOT

# Configs to aggregate. (label, model_name, [seeds])
DEFAULT_CONFIGS = [
    ("OminiControl w/o text",  "VAE9k",       [42, 123, 7]),
    ("OminiControl w/ text",   "VAE9k_class", [42, 123, 7]),
    ("2cond 50k w/o text",     "Cn50",        [42, 123, 7]),
    ("2cond 50k w/ text",      "Cn50_class",  [42, 123, 7]),
]
METRICS = ["FID", "SSIM", "PSNR", "MUSIQ", "CLIP-Text", "CLIP-Image",
           "Region CLIP-T", "Region CLIP-I"]
SPLITS = ["single_box", "multibox", "combined"]
DEFAULT_SEED = 42  # eval200_<model> (no suffix) corresponds to this seed

def metrics_path(model, seed):
    """Prefer eval200_<model>_s<seed>, fall back to legacy eval200_<model> for seed=42."""
    suffixed = os.path.join(REPO_OUT, f"eval200_{model}_s{seed}", "metrics.json")
    if os.path.exists(suffixed):
        return suffixed
    if seed == DEFAULT_SEED:
        legacy = os.path.join(REPO_OUT, f"eval200_{model}", "metrics.json")
        if os.path.exists(legacy):
            return legacy
    return suffixed

def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, var ** 0.5, n

def fmt(mean, std, n):
    if math.isnan(mean):
        return "nan"
    if n <= 1:
        return f"{mean:.3f}"
    if abs(mean) < 10:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.2f}±{std:.2f}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="store_true", help="Output CSV instead of markdown")
    args = p.parse_args()

    # collected[label][split][metric] = [values across seeds]
    collected = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = defaultdict(list)  # label -> list of (seed, status)

    for label, model, seeds in DEFAULT_CONFIGS:
        for s in seeds:
            mp = metrics_path(model, s)
            if not os.path.exists(mp):
                seen[label].append((s, "MISSING"))
                continue
            seen[label].append((s, "OK"))
            data = json.load(open(mp))
            for split in SPLITS:
                d = data.get(split, {}) or {}
                for k in METRICS:
                    v = d.get(k)
                    if v is not None:
                        collected[label][split][k].append(v)

    print("# Status (which seeds were found)\n")
    for label in seen:
        items = ", ".join(f"s{s}={status}" for s, status in seen[label])
        print(f"- **{label}**: {items}")
    print()

    for split in SPLITS:
        print(f"\n## {split} (mean ± std across seeds)\n")
        if args.csv:
            print("Method," + ",".join(METRICS) + ",N")
        else:
            hdr = ["Method"] + METRICS + ["N"]
            print("| " + " | ".join(hdr) + " |")
            print("|" + "|".join(["---"] * len(hdr)) + "|")
        for label, _, _ in DEFAULT_CONFIGS:
            cells, n_used = [], 0
            for k in METRICS:
                vals = collected[label][split][k]
                m, s, n = mean_std(vals)
                cells.append(fmt(m, s, n))
                n_used = max(n_used, n)
            if args.csv:
                print(f"{label}," + ",".join(cells) + f",{n_used}")
            else:
                print(f"| {label} | " + " | ".join(cells) + f" | {n_used} |")

if __name__ == "__main__":
    main()
