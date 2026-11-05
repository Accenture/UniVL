import torch
from ..paths import DATA_ROOT
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from datasets import load_dataset
import json
from .trainer import UniVLModel, get_config, train
from ..pipeline.flux_univl import Condition, convert_to_condition, generate

USE_PROMPT = False
USE_TEXT_ON_MASK = True   # If True, fill bbox black and tile text label; if False, load image as-is
MASK_JITTER = True        # Randomly dilate/shrink bbox during training for robustness
MASK_JITTER_RANGE = (0.85, 1.3)  # Scale factor range (centered on bbox)
cache_dir = DATA_ROOT

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def _render_text_in_bbox(img: Image.Image, x1: int, y1: int, x2: int, y2: int, text: str) -> Image.Image:
    """Fill bbox with solid black then tile white text, clipped strictly inside bbox.
    Font size is scaled down if the text is wider than the bbox so full text always fits."""
    if not text or x2 <= x1 + 4 or y2 <= y1 + 4:
        return img
    pw, ph = x2 - x1, y2 - y1
    patch = Image.new("RGB", (pw, ph), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    font_size = max(4, min(18, ph // 3))
    try:
        font = ImageFont.truetype(_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    # Measure single-copy width; scale font down if it exceeds bbox width
    try:
        tw = draw.textlength(text, font=font)
    except Exception:
        tw = font_size * len(text) * 0.6
    avail = pw - 4  # 2px padding each side
    if tw > avail and tw > 0:
        scale = avail / tw
        font_size = max(4, int(font_size * scale))
        try:
            font = ImageFont.truetype(_FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()
        try:
            tw = draw.textlength(text, font=font)
        except Exception:
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

def _jitter_bbox(pt1, pt2, scale_range=(0.85, 1.3), img_w=None, img_h=None):
    """Randomly scale bbox around its center. Returns new (pt1, pt2).

    Also applies small random translation (up to 5% of bbox size) so the
    model doesn't overfit to perfectly centered masks.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return pt1, pt2

    # Random scale
    scale = random.uniform(*scale_range)
    nw, nh = w * scale, h * scale

    # Small random translation (up to 5% of bbox size)
    dx = random.uniform(-0.05, 0.05) * w
    dy = random.uniform(-0.05, 0.05) * h
    cx += dx
    cy += dy

    nx1, ny1 = cx - nw / 2, cy - nh / 2
    nx2, ny2 = cx + nw / 2, cy + nh / 2

    # Clamp to image bounds if provided
    if img_w is not None:
        nx1, nx2 = max(0, nx1), min(img_w, nx2)
    if img_h is not None:
        ny1, ny2 = max(0, ny1), min(img_h, ny2)

    return [nx1, ny1], [nx2, ny2]


def _extract_objn(basename: str) -> str:
    """Extract object class from filename, skipping trailing 'shrink' and numeric parts.
    E.g. '00111_..._rose_1_shrink.jpg' -> 'rose'
         '00042_..._person_0.jpg'       -> 'person'
    """
    stem = basename.split(".")[0]
    parts = stem.split("_")
    for part in reversed(parts):
        if part == "shrink" or part.isdigit():
            continue
        return part
    return stem


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset=None, # Ignoring original hf dataset as we use jsons directly now
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

        # --- LOAD AND PARSE ALL JSONS ---
        self.records = []
        
        # Paths to JSONs
        # Square uses bbox JSONL (has correct mask bbox from segmentation)
        json_square_jsonl = DATA_ROOT + "/textmask_square/metadata_coords_bbox.jsonl"
        # Add/replace/extract still use pt1/pt2 (text coords) — bbox TBD
        json_add = DATA_ROOT + "/textmask_add_laion/metadata_coords2.json"
        json_replace = DATA_ROOT + "/textmask_replace_laion/metadata_coords2.json"
        json_extract = DATA_ROOT + "/textmask_extract_ref/metadata_coords2.json"

        # 1. Square — load from JSONL with correct mask bbox
        if os.path.exists(json_square_jsonl):
            with open(json_square_jsonl, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line[0] != '{':
                        continue
                    try:
                        entry = json.loads(line)
                        k, v = list(entry.items())[0]
                        bbox = v.get("bbox")
                        if bbox is None:
                            continue
                        pt1 = [bbox[0], bbox[1]]
                        pt2 = [bbox[2], bbox[3]]
                        gt_path = v.get("gt_path", "")
                        cond_path = os.path.join(cache_dir, k)
                        self.records.append({
                            "gt_path": gt_path,
                            "cond_path": cond_path,
                            "coords": (pt1, pt2),
                            "type": "square",
                            "objn": _extract_objn(os.path.basename(k))
                        })
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        # Helper: load add/replace/extract JSON with file existence check
        def _load_bbox_json(json_path, dataset_type):
            if not os.path.exists(json_path):
                return
            skipped = 0
            with open(json_path, 'r') as f:
                data = json.load(f)
            for cond_path, v in data.items():
                bbox = v.get("bbox") if isinstance(v, dict) else None
                if bbox is None:
                    continue
                # Skip entries with missing files
                if not os.path.exists(cond_path):
                    skipped += 1
                    continue
                pt1 = [bbox[0], bbox[1]]
                pt2 = [bbox[2], bbox[3]]
                base, ext = os.path.splitext(cond_path)
                gt_path = v.get("gt_path", f"{base}_r{ext}")
                if not os.path.exists(gt_path):
                    skipped += 1
                    continue
                self.records.append({
                    "gt_path": gt_path,
                    "cond_path": cond_path,
                    "coords": (pt1, pt2),
                    "type": dataset_type,
                    "objn": _extract_objn(os.path.basename(cond_path))
                })
            if skipped > 0:
                print(f"  {dataset_type}: skipped {skipped} entries (missing files)")

        # 2. Add
        _load_bbox_json(json_add, "add")
        # 3. Replace
        _load_bbox_json(json_replace, "replace")
        # 4. Extract
        _load_bbox_json(json_extract, "extract")

        # Shuffle the combined dataset
        random.seed(42)
        random.shuffle(self.records)
        print(f"Loaded a total of {len(self.records)} records across all datasets.")
        for i in range(100):
            self.__getitem__(i)

    def __len__(self):
        return len(self.records)

    def __get_condition__(self, image, condition_type, record):
        cond_path = record["cond_path"]
        objn = record["objn"]
        condition_size = self.condition_size
        position_delta = np.array([0, 0])
        
        if condition_type in ["canny", "coloring", "deblurring", "depth"]:
            image, kwargs = image.resize(condition_size), {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            condition_img = convert_to_condition(condition_type, image, **kwargs)
        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize(condition_size)
            image = depth_img.resize(condition_size)
        elif condition_type == "fill":
            condition_img = image.resize(condition_size).convert("RGB")
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        # --- REQUESTED UPDATE FOR OCR ---
        elif condition_type == "ocr":
            # Load the condition image directly from path
            condition_img = Image.open(cond_path).convert("RGB")
            orig_cond_w, orig_cond_h = condition_img.size
            # Resize to match expected dimensions
            condition_img = condition_img.resize(condition_size)
            # Render text label onto the masked bbox region
            pt1, pt2 = record["coords"]
            scale_x = condition_size[0] / orig_cond_w
            scale_y = condition_size[1] / orig_cond_h
            x1 = int(pt1[0] * scale_x)
            y1 = int(pt1[1] * scale_y)
            x2 = int(pt2[0] * scale_x)
            y2 = int(pt2[1] * scale_y)
            text = objn.replace("-", " ")
            if USE_TEXT_ON_MASK:
                condition_img = _render_text_in_bbox(condition_img, x1, y1, x2, y2, text)
            else:
                from .multibox_dataset import _render_single_text_on_mask
                condition_img = _render_single_text_on_mask(condition_img, x1, y1, x2, y2, text)
        # -------------------------------
        elif condition_type == "sr":
            condition_img = image.resize(condition_size)
            position_delta = np.array([0, -condition_size[0] // 16])
        else:
            raise ValueError(f"Condition type {condition_type} is not implemented.")
        
        return condition_img, position_delta, objn

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Load the ground truth target image
        try:
            image = Image.open(record["gt_path"]).convert("RGB")
        except Exception as e:
            # Fallback for broken paths (return a black image or skip ideally)
            image = Image.new("RGB", self.target_size, (0,0,0))
            
        # Get original size for normalization
        try:
            orig_w, orig_h = Image.open(record["gt_path"].replace("_r.png", ".png")).convert("RGB").size
        except:
            orig_w, orig_h = image.size
        
        # Resize image for the model
        image = image.resize(self.target_size)

        condition_size = self.condition_size
        position_scale = self.position_scale

        # Apply bbox jitter before both text rendering and mask creation
        pt1_raw, pt2_raw = record["coords"]
        if MASK_JITTER and self.condition_type == "ocr":
            pt1_jit, pt2_jit = _jitter_bbox(pt1_raw, pt2_raw, MASK_JITTER_RANGE, img_w=orig_w, img_h=orig_h)
        else:
            pt1_jit, pt2_jit = list(pt1_raw), list(pt2_raw)

        # Pass jittered coords to condition builder (for text rendering)
        record_jit = {**record, "coords": (pt1_jit, pt2_jit)}
        condition_img, position_delta, objn = self.__get_condition__(
            image, self.condition_type, record_jit
        )

        description = objn if USE_PROMPT else ""

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new("RGB", condition_size, (0, 0, 0))

        # Use jittered coords for mask
        pt1, pt2 = pt1_jit, pt2_jit

        # Normalize coordinates
        norm_pt1 = [pt1[0] / orig_w, pt1[1] / orig_h]
        norm_pt2 = [pt2[0] / orig_w, pt2[1] / orig_h]
        #print(f"Record {idx}: Path {record['gt_path']} | Original coords {pt1}, {pt2} | original size {orig_w}, {orig_h} | Normalized coords {norm_pt1}, {norm_pt2}")

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            "norm_bbox": [norm_pt1[0], norm_pt1[1], norm_pt2[0], norm_pt2[1]],
            "objn": objn,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }

# Records used to render periodic validation samples during training (optional).
test_file_path = DATA_ROOT + "/test_masks/stest.jsonl"
test_lines = []
if os.path.exists(test_file_path):
    with open(test_file_path, 'r') as f:
        test_lines = f.readlines()

@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]

    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    adapter = model.adapter_names[2]
    condition_type = model.training_config["condition_type"]
    test_list = []

    if condition_type in ["canny", "coloring", "deblurring", "depth"]:
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition_img = convert_to_condition(condition_type, image, 5)
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "depth_pred":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "fill":
        condition_img = (
            Image.open("./assets/vase_hq.jpg").resize(condition_size).convert("RGB")
        )
        mask = Image.new("L", condition_img.size, 0)
        draw = ImageDraw.Draw(mask)
        a = condition_img.size[0] // 4
        b = a * 3
        draw.rectangle([a, a, b, b], fill=255)
        condition_img = Image.composite(
            condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
        )
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    # --- UPDATED OCR LOGIC START ---
    elif condition_type == "ocr":
        if not test_lines:
            print(f"Skipping validation samples: {test_file_path} not found.")
            return
        # 2. Parse the uniquely sampled entry (retry on bad lines)
        for _attempt in range(10):
            try:
                entry = json.loads(random.choice(test_lines).strip())
                break
            except json.JSONDecodeError:
                continue
        else:
            print("Warning: could not parse any test line")
            return []
        
        # In the new format, the entry is a dict with ONE key (the path) and the metadata as the value
        cpath, meta = list(entry.items())[0]
        
        # 3. Handle pathing (Check if the JSON path is already absolute)
        if os.path.isabs(cpath):
            cond_path = cpath
        else:
            cond_path = os.path.join(cache_dir, cpath)
            
        # 4. Extract and Normalize Bounding Box
        bbox = meta.get("bbox", None)
        res = meta.get("resolution", {})
        
        if bbox is not None and "width" in res and "height" in res:
            w, h = res["width"], res["height"]
            # Convert [x_min, y_min, x_max, y_max] to normalized [0, 1] range
            pts = [
                max(0.0, min(1.0, bbox[0] / w)),
                max(0.0, min(1.0, bbox[1] / h)),
                max(0.0, min(1.0, bbox[2] / w)),
                max(0.0, min(1.0, bbox[3] / h))
            ]
        else:
            pts = None
            
        # 5. Extract the prompt (object class) from the file name
        # Uses _extract_objn which skips trailing "shrink" and numeric suffixes
        prompt = _extract_objn(os.path.basename(cpath)).replace("-", " ")
        
        # 6. Load and resize the condition image
        condition_img = Image.open(cond_path).convert("RGB")
        orig_cond_w, orig_cond_h = condition_img.size
        condition_img = condition_img.resize(condition_size)
        # Render text on mask region
        if bbox is not None and prompt:
            x1 = int(bbox[0] * condition_size[0] / orig_cond_w)
            y1 = int(bbox[1] * condition_size[1] / orig_cond_h)
            x2 = int(bbox[2] * condition_size[0] / orig_cond_w)
            y2 = int(bbox[3] * condition_size[1] / orig_cond_h)
            if USE_TEXT_ON_MASK:
                condition_img = _render_text_in_bbox(condition_img, x1, y1, x2, y2, prompt)
            else:
                from .multibox_dataset import _render_single_text_on_mask
                condition_img = _render_single_text_on_mask(condition_img, x1, y1, x2, y2, prompt)

        # 7. Create the Condition object and add it to the test list
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        test_list.append((condition, prompt, pts, condition_img))
    # --- UPDATED OCR LOGIC END ---

    elif condition_type == "super_resolution":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    else:
        raise NotImplementedError
    os.makedirs(save_path, exist_ok=True)
    images = []
    for i, item in enumerate(test_list):
        if condition_type == "ocr":
            condition, prompt, pts, condition_img = item
        else:
            condition, prompt = item
            pts = None
            condition_img = condition.image # Assuming Condition object stores the PIL image
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)
        res = generate(
            model.flux_pipe,
            #prompt=prompt if random.random()>=0.6 else "",
            prompt=prompt if USE_PROMPT else "",
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
            pts=pts if condition_type == "ocr" else None, # Pass pts only for OCR condition
            #image_guidance_scale=3.0,
        )
        #file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        #res.images[0].save(file_path)
        # --- Post-processing ---
        generated_image = res.images[0]

        cond_image = condition_img
        
        if cond_image.size != generated_image.size:
            cond_image = cond_image.resize(generated_image.size)
        
        combined_width = cond_image.width + generated_image.width
        combined_height = max(cond_image.height, generated_image.height)
        
        combined_image = Image.new("RGB", (combined_width, combined_height))
        combined_image.paste(cond_image, (0, 0))
        combined_image.paste(generated_image, (cond_image.width, 0))

        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        combined_image.save(file_path)
        images.append(combined_image)
    return images

class CombinedDatasetWithMultibox(Dataset):
    """Wraps existing single-box dataset + multibox dataset.

    Samples from multibox with probability `multibox_ratio`, otherwise from base.
    Normalizes norm_bbox to always be a list-of-bboxes format.
    """
    def __init__(self, base_dataset, multibox_dataset=None, multibox_ratio=0.15):
        self.base = base_dataset
        self.multibox = multibox_dataset
        self.multibox_ratio = multibox_ratio if multibox_dataset and len(multibox_dataset) > 0 else 0.0
        total = len(self.base) + (len(self.multibox) if self.multibox else 0)
        print(f"CombinedDatasetWithMultibox: base={len(self.base)}, "
              f"multibox={len(self.multibox) if self.multibox else 0}, "
              f"ratio={self.multibox_ratio:.0%}, virtual_len={len(self.base)}")

    def __len__(self):
        return len(self.base)  # epoch size = base dataset size

    def __getitem__(self, idx):
        if self.multibox_ratio > 0 and random.random() < self.multibox_ratio:
            sample = self.multibox[random.randint(0, len(self.multibox) - 1)]
        else:
            sample = self.base[idx]

        # Normalize norm_bbox to list-of-bboxes format
        nb = sample.get("norm_bbox")
        if nb is not None:
            if isinstance(nb[0], (int, float)):
                # Single bbox [x1,y1,x2,y2] → [[x1,y1,x2,y2]]
                sample["norm_bbox"] = [nb]
            # else: already [[x1,y1,x2,y2], ...] from multibox
        return sample


def multibox_collate_fn(batch):
    """Custom collate that keeps norm_bbox and objn as plain lists (not tensors)."""
    from torch.utils.data.dataloader import default_collate

    # Separate fields that need special handling
    norm_bboxes = [item.pop("norm_bbox", None) for item in batch]
    objn_list = [item.pop("objn", "") for item in batch]

    # Default collate for the rest (image tensors, descriptions, etc.)
    collated = default_collate(batch)

    # Add back as plain lists
    collated["norm_bbox"] = norm_bboxes
    collated["objn"] = objn_list

    return collated


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Initialize custom dataset (It will now load its own JSON paths directly)
    base_dataset = ImageConditionDataset(
        base_dataset=None,
        condition_size=training_config["dataset"]["condition_size"],
        target_size=training_config["dataset"]["target_size"],
        condition_type=training_config["condition_type"],
        drop_text_prob=training_config["dataset"]["drop_text_prob"],
        drop_image_prob=training_config["dataset"]["drop_image_prob"],
        position_scale=training_config["dataset"].get("position_scale", 1.0),
    )

    # Add multibox dataset if configured
    multibox_ratio = training_config["dataset"].get("multibox_ratio", 0.30)
    multibox_ds = None
    if multibox_ratio > 0:
        from .multibox_dataset import MultiboxDataset
        multibox_ds = MultiboxDataset(
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            position_scale=training_config["dataset"].get("position_scale", 1.0),
        )

    dataset = CombinedDatasetWithMultibox(base_dataset, multibox_ds, multibox_ratio)

    # Initialize model
    trainable_model = UniVLModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()