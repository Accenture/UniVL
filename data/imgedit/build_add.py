import os
from univl.paths import DATA_ROOT, IMGEDIT_ROOT
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import *

# --- Configuration ---
#root_dir_ = IMGEDIT_ROOT + "/results_add_laion_part"

output_dir = DATA_ROOT + "/textmask_add_laion"
metadata_path = os.path.join(output_dir, "metadata_coords2.json")

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# --- Initialize Metadata Dictionary ---
bbox_metadata = {}

# --- Main Logic ---
for _i in [5,4, 0,1]:
    root_dir = IMGEDIT_ROOT + f"/results_add_laion_part{_i}"
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    subfolders.sort()

    print(f"Found {len(subfolders)} subfolders in {root_dir}")


    for subfolder in tqdm(subfolders):
        try:
            current_dir = os.path.join(root_dir, subfolder)
            
            # Define file paths
            result_json_path = os.path.join(current_dir, "result.json")
            original_png_path = os.path.join(current_dir, "original.png")
            result_png_path = os.path.join(current_dir, "result.png")
            
            # Check critical files exist
            if not (os.path.exists(result_json_path) and os.path.exists(original_png_path)):
                continue
                
            # 1. Load JSON Data
            with open(result_json_path, 'r') as f:
                result_data = json.load(f)
                
            edit_result = result_data.get('edit_result')
            if edit_result is None:
                continue
                
            # --- Robust Score Filtering ---
            raw_score = edit_result.get('score')
            final_score = 0.0

            if isinstance(raw_score, list):
                if len(raw_score) > 0:
                    final_score = raw_score[0]
            elif isinstance(raw_score, (float, int)):
                final_score = raw_score
            
            # Filter logic (Score > 0.9)
            if final_score <= 0.9:
                continue

            class_name = edit_result['class_name']
            bbox = edit_result['bbox'] # [min_x, min_y, max_x, max_y]
            
            if not bbox or len(bbox) != 4:
                continue

            # 2. Load and Resize Image
            image_bgr = cv2.imread(original_png_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Ensure image matches JSON resolution
            target_h = result_data['resolution']['height']
            target_w = result_data['resolution']['width']
            
            h_img, w_img = image_rgb.shape[:2]
            if (target_h != h_img) or (target_w != w_img):
                image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # 3. Create Mask from Bounding Box
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            # Clamp coordinates
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(target_w, x_max)
            y_max = min(target_h, y_max)
            
            # Create the base masked image (Black box, No text)
            masked_img_no_text = image_rgb.copy()
            masked_img_no_text[y_min:y_max, x_min:x_max] = 0
            
            # ---------------------------------------------------------
            # --- SAVE SET 1: Masked (No Text) + Original Reference ---
            # ---------------------------------------------------------
            
            # Save Masked Image (No Text)
            """save_filename_1 = f"{subfolder}_1_.png"
            full_save_path_1 = os.path.join(output_dir, save_filename_1)
            
            masked_img_no_text_bgr = cv2.cvtColor(masked_img_no_text, cv2.COLOR_RGB2BGR)
            cv2.imwrite(full_save_path_1, masked_img_no_text_bgr)
            
            # Copy Original Image
            # Note: We copy the file from disk (unresized) as requested
            save_filename_1_r = f"{subfolder}_1__r.png"
            full_save_path_1_r = os.path.join(output_dir, save_filename_1_r)
            shutil.copy2(original_png_path, full_save_path_1_r)"""

            # ---------------------------------------------------------
            # --- SAVE SET 0: Masked (With Text) + Result Reference ---
            # ---------------------------------------------------------

            # Work on a copy for text addition
            masked_text_img = masked_img_no_text.copy() # Already has black box
            
            # --- Text Calculation ---
            obj_h = y_max - y_min
            obj_w = x_max - x_min
            center_x = x_min + obj_w // 2
            center_y = y_min + obj_h // 2
            
            font_scale = 3.0
            font_thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(class_name, font, font_scale, font_thickness)        
            
            # Shrink text
            while (text_w > obj_w * 1.8) and font_scale > 0.5:
                font_scale -= 0.1
                text_w, text_h = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
                
            text_x = int(center_x - text_w / 2)
            text_y = int(center_y + text_h / 2)
            
            # Draw Tight Rectangle Background for Text (Black)
            pad = 1 
            pt1 = (text_x - pad, text_y - text_h - pad)
            pt2 = (text_x + text_w + pad, text_y + baseline + pad)
            cv2.rectangle(masked_text_img, pt1, pt2, (0, 0, 0), -1)
            
            # Draw Text (White)
            cv2.putText(masked_text_img, class_name, (text_x, text_y), 
                        font, font_scale, (255, 255, 255), font_thickness)
            
            # Save Masked Image (With Text)
            safe_class_name = class_name.replace(" ", "-")
            save_filename_0 = f"{subfolder}_0_{safe_class_name}.png"
            full_save_path_0 = os.path.join(output_dir, save_filename_0)
            
            masked_text_img_bgr = cv2.cvtColor(masked_text_img, cv2.COLOR_RGB2BGR)
            ###cv2.imwrite(full_save_path_0, masked_text_img_bgr)
            
            # Copy Result Image
            if os.path.exists(result_png_path):
                save_filename_0_r = f"{subfolder}_0_{safe_class_name}_r.png"
                full_save_path_0_r = os.path.join(output_dir, save_filename_0_r)
                ###shutil.copy2(result_png_path, full_save_path_0_r)

            # --- Store Metadata (Keyed by the text mask file) ---
            bbox_metadata[full_save_path_0] = {
                "pt1": pt1, 
                "pt2": pt2, 
                "bbox": [x_min, y_min, x_max, y_max],
                "resolution": {"height": target_h, "width": target_w}
            }

            
            # Optional: Add metadata for the _1_ file as well if needed, 
            # but defaulting to linking it to the _0_ filename as the primary record.

        except Exception as e:
            print(f"Error processing {subfolder}: {e}")
            continue

# --- Save Metadata to JSON ---
print(f"Saving metadata to {metadata_path}...")
with open(metadata_path, 'w') as f:
    json.dump(bbox_metadata, f, indent=4)

print("Processing complete.")
