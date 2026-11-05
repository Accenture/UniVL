"""Phrase-level vocabulary analysis for ImageConditionDataset.

Unlike analyze_vocab.py, this keeps the class name as a phrase
(`Ford-Mustang` -> `ford mustang`, kept as one 2-word entry).

Outputs:
  - per-word-count bucket counts
  - top phrases per 2+ word bucket
  - phrase wordcloud + top-N bar chart
"""

import json
from univl.paths import DATA_ROOT
import os
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


CACHE_DIR = DATA_ROOT
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_SQUARE_JSONL = f"{CACHE_DIR}/textmask_square/metadata_coords_bbox.jsonl"
JSON_ADD = f"{CACHE_DIR}/textmask_add_laion/metadata_coords2.json"
JSON_REPLACE = f"{CACHE_DIR}/textmask_replace_laion/metadata_coords2.json"
JSON_EXTRACT = f"{CACHE_DIR}/textmask_extract_ref/metadata_coords2.json"


def _extract_objn(basename: str) -> str:
    stem = basename.split(".")[0]
    parts = stem.split("_")
    for part in reversed(parts):
        if part == "shrink" or part.isdigit():
            continue
        return part
    return stem


def _to_phrase(objn: str) -> str:
    """Mirror the actual rendered text: objn.replace('-', ' '), lowercased,
    whitespace-collapsed."""
    if not objn:
        return ""
    p = objn.replace("-", " ").lower()
    p = re.sub(r"\s+", " ", p).strip()
    return p


def load_phrases():
    phrases = []  # one per record (= per box)

    if os.path.exists(JSON_SQUARE_JSONL):
        with open(JSON_SQUARE_JSONL, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    entry = json.loads(line)
                    k, v = next(iter(entry.items()))
                    if v.get("bbox") is None:
                        continue
                    phrases.append(_to_phrase(_extract_objn(os.path.basename(k))))
                except (json.JSONDecodeError, KeyError, StopIteration):
                    continue

    for json_path in [JSON_ADD, JSON_REPLACE, JSON_EXTRACT]:
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        for cond_path, v in data.items():
            if not isinstance(v, dict) or v.get("bbox") is None:
                continue
            phrases.append(_to_phrase(_extract_objn(os.path.basename(cond_path))))

    return [p for p in phrases if p]


def main():
    print("Loading phrases ...")
    phrases = load_phrases()
    total = len(phrases)
    print(f"Total phrase records: {total:,}")

    counter = Counter(phrases)
    print(f"Unique phrases: {len(counter):,}")

    # Bucket by word count
    buckets_records = Counter()       # records (instances) per #words
    buckets_unique = defaultdict(set) # unique phrases per #words
    for p, c in counter.items():
        n = len(p.split())
        buckets_records[n] += c
        buckets_unique[n].add(p)

    print("\n=== Phrase length distribution ===")
    print(f"{'#words':>6s} {'records':>12s} {'%records':>9s} "
          f"{'unique phrases':>15s} {'%unique':>9s}")
    for n in sorted(buckets_records):
        rec = buckets_records[n]
        uq = len(buckets_unique[n])
        print(f"{n:>6d} {rec:>12,d} {100*rec/total:>8.2f}% "
              f"{uq:>15,d} {100*uq/len(counter):>8.2f}%")

    # Top phrases per bucket (>=2 words)
    print("\n=== Top 5 phrases per bucket (#words >= 2) ===")
    for n in sorted(buckets_records):
        if n < 2:
            continue
        sub = Counter({p: c for p, c in counter.items() if len(p.split()) == n})
        top = sub.most_common(5)
        if not top:
            continue
        print(f"\n  -- {n}-word phrases "
              f"(unique={len(sub):,}, records={buckets_records[n]:,}) --")
        for p, c in top:
            print(f"    {c:>7,d}  {p}")

    # Phrase-level word cloud (keys are full phrases — wordcloud renders them
    # as single labels, no internal splitting).
    print("\nRendering phrase word cloud ...")
    wc = WordCloud(
        width=1600, height=900,
        background_color="white",
        collocations=False,
        prefer_horizontal=0.9,
        max_words=400,
        regexp=r"[^\s][^\t\n\r\f\v]*",  # keep spaces inside a token
    ).generate_from_frequencies(counter)
    out_wc = os.path.join(OUT_DIR, "vocab_wordcloud_phrase.png")
    wc.to_file(out_wc)
    print(f"  saved {out_wc}")

    # Top-N phrase bar chart (overall)
    topn = counter.most_common(30)
    fig, ax = plt.subplots(figsize=(12, 6))
    items = [w for w, _ in topn][::-1]
    counts = [c for _, c in topn][::-1]
    ax.barh(items, counts, color="seagreen")
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(topn)} phrase-level class names "
                 f"({len(counter):,} unique phrases)")
    fig.tight_layout()
    out_bar = os.path.join(OUT_DIR, "vocab_top40_phrase.png")
    fig.savefig(out_bar, dpi=120)
    plt.close(fig)
    print(f"  saved {out_bar}")

    # Word-count distribution chart
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = sorted(buckets_records)
    ax.bar(xs, [buckets_records[n] for n in xs], color="slateblue",
           label="records")
    ax.set_xlabel("# words in class name")
    ax.set_ylabel("# records (boxes)")
    ax.set_yscale("log")
    ax.set_title("Class-name length distribution (records per word count)")
    fig.tight_layout()
    out_len = os.path.join(OUT_DIR, "phrase_length_dist.png")
    fig.savefig(out_len, dpi=120)
    plt.close(fig)
    print(f"  saved {out_len}")


if __name__ == "__main__":
    main()
