"""
Multibox dataset: loads per-image segmentation JSONs and samples 2-5 non-overlapping
mask regions per image. Renders class-name text on each bbox and returns a single
condition image with multiple text-labeled regions + a list of bboxes as norm_bbox.

Supports both single-condition (USE_TEXT_ON_MASK=True) and TWO_CONDITION modes.
"""
import os
from ..paths import DATA_ROOT
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from ..pipeline.flux_univl import TWO_CONDITION

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LAION_AES_ROOT = DATA_ROOT + "/laion-aes"
PART_DIRS = [
    DATA_ROOT + f"/jsons/part{i}_step4_aes_clip_withcount"
    for i in range(0, 8)
]


def _resolve_gt_path(img_rel):
    """Resolve GT path with tar nesting fix: laion-aes/{first_dir}/{path}."""
    first_dir = img_rel.split("/")[0]
    return os.path.join(LAION_AES_ROOT, first_dir, img_rel)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _render_single_text_on_mask(img, x1, y1, x2, y2, text):
    """Render a single centered text on black rectangle.

    Fills bbox with black, draws one centered text label sized to fit within the mask.
    Used when USE_TEXT_ON_MASK=False for multibox: each mask gets its class name once.
    """
    if not text or x2 <= x1 + 4 or y2 <= y1 + 4:
        return img
    pw, ph = x2 - x1, y2 - y1
    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)

    # Start with font proportional to patch height, cap at reasonable size
    font_size = max(6, min(36, ph // 3))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        tw = draw.textlength(text, font=font)
    except Exception:
        tw = font_size * len(text) * 0.6

    # Shrink font until text fits within 90% of bbox width
    max_tw = pw * 0.9
    while tw > max_tw and font_size > 4:
        font_size -= 1
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
            tw = draw.textlength(text, font=font)
        except Exception:
            tw = font_size * len(text) * 0.6

    # Center the text within the patch (clamp to patch bounds)
    tw_int = min(int(tw), pw)
    tx = max(0, (pw - tw_int) // 2)
    ty = max(0, (ph - font_size) // 2)
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    img = img.copy()
    img.paste(patch, (x1, y1))
    return img


def _render_text_in_bbox(img, x1, y1, x2, y2, text):
    """Render tiled white text on black rectangle at (x1,y1)-(x2,y2)."""
    if not text or x2 <= x1 + 4 or y2 <= y1 + 4:
        return img
    pw, ph = x2 - x1, y2 - y1
    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    font_size = max(4, min(18, ph // 3))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    try:
        tw = draw.textlength(text, font=font)
    except:
        tw = font_size * len(text) * 0.6
    avail = pw - 4
    if tw > avail and tw > 0:
        font_size = max(4, int(font_size * avail / tw))
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()
        try:
            tw = draw.textlength(text, font=font)
        except:
            tw = font_size * len(text) * 0.6
    tw = max(tw, 1)
    y_cur = 2
    while y_cur + font_size <= ph - 2:
        x_cur = 2
        while x_cur < pw - 2:
            draw.text((x_cur, y_cur), text, fill=(255, 255, 255), font=font)
            x_cur += int(tw) + 4
        y_cur += font_size + 3
    img = img.copy()
    img.paste(patch, (x1, y1))
    return img


def _bbox_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] bboxes (pixel coords)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _has_overlap(selected, candidate, threshold=0.05):
    """Check if candidate bbox overlaps any already-selected bbox."""
    for s in selected:
        if _bbox_iou(s["bbox"], candidate["bbox"]) > threshold:
            return True
    return False


def _bbox_area_ratio(bbox, img_w, img_h):
    """Return bbox area as fraction of image area."""
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    return (bw * bh) / (img_w * img_h) if img_w * img_h > 0 else 0


def _sample_masks(segments, img_w, img_h, min_masks=2, max_masks=5, max_retries=5):
    """Sample 2-5 non-overlapping masks from segmentation list.

    Rules:
    - Each mask area must be 4%-40% of image area
    - No pair of masks should overlap (IoU > 0.05)
    - If can't find enough non-overlapping, fall back to 2, then 1
    """
    # Filter valid segments (reasonable size, has bbox)
    valid = []
    for seg in segments:
        bbox = seg.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        ratio = _bbox_area_ratio(bbox, img_w, img_h)
        if ratio < 0.04 or ratio > 0.40:
            continue
        # Skip very low CLIP score (likely misdetection)
        cs = seg.get("clip_score")
        if cs is not None and cs < 0.01:
            continue
        valid.append(seg)

    if not valid:
        return []

    target_n = random.randint(min_masks, min(max_masks, len(valid)))

    for retry in range(max_retries):
        random.shuffle(valid)
        selected = []
        for seg in valid:
            if not _has_overlap(selected, seg):
                selected.append(seg)
            if len(selected) >= target_n:
                break

        if len(selected) >= 2:
            return selected

        # Reduce target on retry
        target_n = 2

    # Final fallback: just pick 1
    return [random.choice(valid)]


# ── Dataset ──────────────────────────────────────────────────────────────────

FILTERED_JSONL = DATA_ROOT + "/multibox_filtered.jsonl"


class MultiboxDataset(Dataset):
    def __init__(
        self,
        filtered_jsonl=None,
        part_dirs=None,
        condition_size=(512, 512),
        target_size=(512, 512),
        drop_text_prob=0.1,
        drop_image_prob=0.1,
        position_scale=1.0,
        use_text_on_mask=True,
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.position_scale = position_scale
        self.use_text_on_mask = use_text_on_mask
        self.to_tensor = T.ToTensor()

        jsonl_path = filtered_jsonl or FILTERED_JSONL
        if os.path.exists(jsonl_path):
            # Fast path: load pre-filtered JSONL
            self.records = []
            with open(jsonl_path) as f:
                for line in f:
                    self.records.append(json.loads(line.strip()))
            print(f"MultiboxDataset: {len(self.records)} images from {jsonl_path}")
        else:
            # Fallback: scan part directories (slow)
            if part_dirs is None:
                part_dirs = PART_DIRS
            self.records = []
            for d in part_dirs:
                if not os.path.isdir(d):
                    continue
                for fn in sorted(os.listdir(d)):
                    fpath = os.path.join(d, fn)
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        self.records.append(data)
                    except:
                        pass
            print(f"MultiboxDataset: {len(self.records)} images from {len(part_dirs)} parts (unfiltered)")

    _fallback_depth = 0

    def _fallback_item(self):
        """Return a random different item when current one is invalid."""
        self._fallback_depth += 1
        if self._fallback_depth > 10:
            self._fallback_depth = 0
            # Return a blank dummy sample to avoid infinite recursion
            blank = Image.new("RGB", self.target_size, (0, 0, 0))
            return {
                "image": self.to_tensor(blank),
                "condition_0": self.to_tensor(blank),
                "condition_type_0": "ocr",
                "position_delta_0": np.array([0, 0]),
                "description": "",
                "norm_bbox": [[0.0, 0.0, 1.0, 1.0]],
                "objn": "",
            }
        result = self[random.randint(0, len(self) - 1)]
        self._fallback_depth = 0
        return result

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        data = self.records[idx]

        # Load GT image (with tar nesting fix)
        img_rel = data["path"]
        gt_path = _resolve_gt_path(img_rel)
        if not os.path.exists(gt_path):
            return self._fallback_item()
        try:
            image = Image.open(gt_path).convert("RGB")
        except Exception:
            return self._fallback_item()

        orig_w, orig_h = image.size
        res = data.get("resolution", {"width": orig_w, "height": orig_h})
        img_w, img_h = res["width"], res["height"]

        # Use pre-filtered segments (from JSONL) or raw segmentation
        all_segs = data.get("segments", [])
        if not all_segs:
            # Fallback for raw JSON format
            for seg_type in ["object", "background"]:
                for seg in data.get("segmentation", {}).get(seg_type, []):
                    all_segs.append(seg)

        # Sample non-overlapping masks
        selected = _sample_masks(all_segs, img_w, img_h)

        if not selected:
            # Fallback: return full image with no mask
            image = image.resize(self.target_size)
            condition_img = image.copy()
            return {
                "image": self.to_tensor(image),
                "condition_0": self.to_tensor(condition_img),
                "condition_type_0": "ocr",
                "position_delta_0": np.array([0, 0]),
                "description": "",
                "norm_bbox": [0.0, 0.0, 1.0, 1.0],
                "objn": "",
            }

        # Resize image
        image = image.resize(self.target_size)

        # Build condition image: start from GT, render text on each mask
        condition_img = Image.open(gt_path).convert("RGB").resize(self.condition_size)

        # Build normalized bboxes and render text
        norm_bboxes = []
        objn_list = []
        for seg in selected:
            bbox = seg["bbox"]  # pixel coords in original resolution
            class_name = seg["class_name"].replace("-", " ")

            # Normalize to [0, 1]
            nx1 = max(0.0, min(1.0, bbox[0] / img_w))
            ny1 = max(0.0, min(1.0, bbox[1] / img_h))
            nx2 = max(0.0, min(1.0, bbox[2] / img_w))
            ny2 = max(0.0, min(1.0, bbox[3] / img_h))
            norm_bboxes.append([nx1, ny1, nx2, ny2])
            objn_list.append(class_name)

            # Render text on condition image
            cx1 = int(nx1 * self.condition_size[0])
            cy1 = int(ny1 * self.condition_size[1])
            cx2 = int(nx2 * self.condition_size[0])
            cy2 = int(ny2 * self.condition_size[1])
            if self.use_text_on_mask:
                condition_img = _render_text_in_bbox(
                    condition_img, cx1, cy1, cx2, cy2, class_name
                )
            else:
                # Single centered text per mask (data_process_square.py style)
                condition_img = _render_single_text_on_mask(
                    condition_img, cx1, cy1, cx2, cy2, class_name
                )

        # Join object names for description
        description = ""  # USE_PROMPT is False

        # Dropout
        if random.random() < self.drop_text_prob:
            description = ""
        if random.random() < self.drop_image_prob:
            condition_img = Image.new("RGB", self.condition_size, (0, 0, 0))

        # Format norm_bbox as list of [x1,y1,x2,y2] for multibox support
        # Single box → [x1,y1,x2,y2], multiple → [[x1,y1,x2,y2], ...]
        if len(norm_bboxes) == 1:
            norm_bbox_out = norm_bboxes[0]
        else:
            norm_bbox_out = norm_bboxes

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": "ocr",
            "position_delta_0": np.array([0, 0]),
            "description": description,
            "norm_bbox": norm_bbox_out,
            "objn": "|".join(objn_list),  # pipe-separated for multi-label
            **({"position_scale_0": self.position_scale} if self.position_scale != 1.0 else {}),
        }
