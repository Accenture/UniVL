import os
from univl.paths import DATA_ROOT, IMGEDIT_ROOT
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
output_dir = DATA_ROOT + "/textmask_extract_ref"
metadata_path = os.path.join(output_dir, "metadata_coords2.json")

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# --- Initialize Metadata Dictionary ---
bbox_metadata = {}

# --- Main Logic ---
# The user specified parts 1 and 7
parts_to_process = [1, 7]

for _i in parts_to_process:
    root_dir = IMGEDIT_ROOT + f"/results_extract_ref_part{_i}"
    
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        continue

    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    subfolders.sort()

    print(f"Found {len(subfolders)} subfolders in {root_dir}")

    for subfolder in tqdm(subfolders):
        try:
            current_dir = os.path.join(root_dir, subfolder)
            
            # Define file paths
            result_json_path = os.path.join(current_dir, "result.json")
            original_png_path = os.path.join(current_dir, "original.png")
            result_2_png_path = os.path.join(current_dir, "result_2.png")
            
            # Check critical files exist
            if not (os.path.exists(result_json_path) and os.path.exists(original_png_path) and os.path.exists(result_2_png_path)):
                continue
                
            # 1. Load JSON Data
            with open(result_json_path, 'r') as f:
                result_data = json.load(f)
            
            # Extract relevant fields
            edit_obj = result_data.get('edit_obj')
            if not edit_obj:
                continue
                
            # --- Robust Score Filtering ---
            raw_score = edit_obj.get('score')
            final_score = 0.0

            if isinstance(raw_score, list):
                if len(raw_score) > 0:
                    final_score = raw_score[0]
            elif isinstance(raw_score, (float, int)):
                final_score = raw_score
            
            # Filter logic (Score > 0.9)
            if final_score <= 0.9:
                continue

            # Get BBox and Names
            bbox = edit_obj.get('bbox') # [min_x, min_y, max_x, max_y]
            class_name = edit_obj.get('class_name')
            edit_result_text = result_data.get('edit_result')

            # Validate data
            if not bbox or len(bbox) != 4 or not class_name or not edit_result_text:
                continue
            
            # Get Target Resolution
            target_h = result_data['resolution']['height']
            target_w = result_data['resolution']['width']

            # 2. Load and Resize Images
            # Load Original
            image_bgr = cv2.imread(original_png_path)
            if image_bgr is None: continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Load Result 2
            result2_bgr = cv2.imread(result_2_png_path)
            if result2_bgr is None: continue
            result2_rgb = cv2.cvtColor(result2_bgr, cv2.COLOR_BGR2RGB)

            # Resize Original to Target
            h_img, w_img = image_rgb.shape[:2]
            if (target_h != h_img) or (target_w != w_img):
                image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
            # Resize Result 2 to Target
            h_res, w_res = result2_rgb.shape[:2]
            if (target_h != h_res) or (target_w != w_res):
                result2_rgb = cv2.resize(result2_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # 3. Prepare Coordinates
            x_min, y_min, x_max, y_max = map(int, bbox)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(target_w, x_max)
            y_max = min(target_h, y_max)
            
            obj_h = y_max - y_min
            obj_w = x_max - x_min
            
            # Skip invalid bboxes
            if obj_h <= 0 or obj_w <= 0:
                continue

            center_x = x_min + obj_w // 2
            center_y = y_min + obj_h // 2
            
            # --- Helper Function to Create Masked Image with Text ---
            def create_masked_text_image(base_img, text, bx_min, bx_max, by_min, by_max, c_x, c_y, o_w, o_h):
                img_copy = base_img.copy()
                # Black out the bounding box
                img_copy[by_min:by_max, bx_min:bx_max] = 0
                
                # Text Settings
                font_scale = 3.0
                font_thickness = 3
                font = cv2.FONT_HERSHEY_SIMPLEX
                (t_w, t_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                # Shrink text to fit
                scale = font_scale
                while (t_w > o_w * 1.8) and scale > 0.5:
                    scale -= 0.1
                    (t_w, t_h), baseline = cv2.getTextSize(text, font, scale, font_thickness)
                
                t_x = int(c_x - t_w / 2)
                t_y = int(c_y + t_h / 2)
                
                # Draw Text Background (Black)
                pad = 1
                pt1 = (t_x - pad, t_y - t_h - pad)
                pt2 = (t_x + t_w + pad, t_y + baseline + pad)
                cv2.rectangle(img_copy, pt1, pt2, (0, 0, 0), -1)
                
                # Draw Text (White)
                cv2.putText(img_copy, text, (t_x, t_y), font, scale, (255, 255, 255), font_thickness)
                
                return img_copy, pt1, pt2

            # ---------------------------------------------------------
            # --- PAIR 1: Original + Mask (class_name) -> Original ---
            # ---------------------------------------------------------
            # Note: For the "reference" copy, we save the resized original image to match resolution
            img_0, pt1_0, pt2_0 = create_masked_text_image(
                image_rgb, class_name, x_min, x_max, y_min, y_max, center_x, center_y, obj_w, obj_h
            )
            
            safe_class_name = class_name.replace(" ", "-").replace("/", "-")
            fname_0 = f"{subfolder}_0_{safe_class_name}.png"
            fpath_0 = os.path.join(output_dir, fname_0)
            
            # Save Masked Image
            ###cv2.imwrite(fpath_0, cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR))
            
            # Save Reference (Resized Original)
            fname_0_r = f"{subfolder}_0_{safe_class_name}_r.png"
            fpath_0_r = os.path.join(output_dir, fname_0_r)
            ###cv2.imwrite(fpath_0_r, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
            # Metadata
            bbox_metadata[fpath_0] = {
                "pt1": pt1_0, 
                "pt2": pt2_0, 
                "bbox": [x_min, y_min, x_max, y_max],
                "resolution": {"height": target_h, "width": target_w}
            }

            # ---------------------------------------------------------
            # --- PAIR 2: Result_2 + Mask (edit_result) -> Original ---
            # ---------------------------------------------------------
            # Base image is now result_2 (resized)
            img_1, pt1_1, pt2_1 = create_masked_text_image(
                result2_rgb, edit_result_text, x_min, x_max, y_min, y_max, center_x, center_y, obj_w, obj_h
            )
            
            safe_edit_result = edit_result_text.replace(" ", "-").replace("/", "-")
            if len(safe_edit_result) > 100:
                safe_edit_result = safe_edit_result[:100]

            fname_1 = f"{subfolder}_1_{safe_edit_result}.png"
            fpath_1 = os.path.join(output_dir, fname_1)
            
            # Save Masked Image
            ###cv2.imwrite(fpath_1, cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR))
            
            # Save Reference (Resized Original)
            fname_1_r = f"{subfolder}_1_{safe_edit_result}_r.png"
            fpath_1_r = os.path.join(output_dir, fname_1_r)
            ###cv2.imwrite(fpath_1_r, cv2.cvtColor(result2_rgb, cv2.COLOR_RGB2BGR))
            
            # Metadata
            bbox_metadata[fpath_1] = {
                "pt1": pt1_1, 
                "pt2": pt2_1, 
                "bbox": [x_min, y_min, x_max, y_max],
                "resolution": {"height": target_h, "width": target_w}
            }

        except Exception as e:
            print(f"Error processing {subfolder}: {e}")
            continue

# --- Save Metadata to JSON ---
print(f"Saving metadata to {metadata_path}...")
with open(metadata_path, 'w') as f:
    json.dump(bbox_metadata, f, indent=4)

print("Processing complete.")