import os
from univl.paths import DATA_ROOT
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from pathlib import Path
from tqdm import *

kernel = np.ones((3, 3), np.uint8)

SAVEBBOX_ONLY = True

# --- Helper Function ---
def rle_to_mask(rle_str, height, width):
    """
    Converts RLE string to a NumPy binary mask.
    """
    rle = {"counts": rle_str.encode("utf-8"), "size": [height, width]}
    mask = mask_util.decode(rle)

    # --- NEW: Dilate the mask by 3 pixels ---
    # A 3x3 kernel expands boundaries by 1 pixel per iteration.
    # We use 3 iterations to get a 3-pixel expansion.
    mask = cv2.dilate(mask, kernel, iterations=np.random.randint(0, 5))
    return mask

# --- Helper Function: Get Rectangular Mask (EDITED) ---
def get_rectangular_mask(binary_mask):
    """
    Computes a rectangular mask defined by the upper-left and lower-right 
    boundaries of the input binary mask.
    Returns: A new binary mask (same shape) representing the bounding box.
    """
    # Create an empty mask of the same shape
    rect_mask = np.zeros_like(binary_mask)
    
    # Find the coordinates of all non-zero pixels
    ys, xs = np.where(binary_mask > 0)
    
    # If the mask is not empty, find the bounding box
    if len(ys) > 0:
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        # Fill the rectangle with 1s (or 255 if treated as uint8 later)
        # Note: +1 because numpy slicing is exclusive at the end
        rect_mask[y_min : y_max + 1, x_min : x_max + 1] = 1
        
    return rect_mask

output_jsonl_path = DATA_ROOT + "/textmask_square/metadata_coords.jsonl"
Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

# Open the JSONL file in write mode at the very beginning
with open(output_jsonl_path, 'w') as f_out:

    # --- Setup Paths ---
    for id_ in range(2): #range(8):
        path_ = DATA_ROOT
        jsonpath = os.path.join(path_, "jsons/part"+str(id_)+"_step4_aes_clip_withcount")

        filelist = [os.path.join(jsonpath, file) for file in os.listdir(jsonpath) if file.endswith('.json')]
        filelist.sort()

        # --- Main Logic ---
        for jfilename in tqdm(filelist):
            try:
                # 1. Load Data
                with open(jfilename) as f:
                    jfile = json.load(f)

                image_path = os.path.join(path_, "laion-aes", jfile['path'])
                image_bgr = cv2.imread(image_path)
                
                # Safety check in case image is corrupted or missing
                if image_bgr is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                h = jfile['resolution']['height']
                w = jfile['resolution']['width']

                np.random.seed(42) 

                objects = jfile['segmentation']['object']

                # 3. Iterate, Plot Three Columns, and Build Combined
                for i_, obj in enumerate(objects):
                    class_name = obj['class_name']
                    rle_str = obj['mask']
                    
                    # 1. Get Original Mask
                    raw_mask = rle_to_mask(rle_str, h, w)
                    
                    # 2. Compute Rectangular Mask
                    rect_mask = get_rectangular_mask(raw_mask)
                    mask_bool = rect_mask.astype(bool) 

                    masked_overlay = image_rgb.copy()
                    masked_overlay[~mask_bool] = 0 

                    masked_text_img = image_rgb.copy()
                    masked_text_img[mask_bool] = 0
                    
                    # --- Text Calculation ---
                    ys, xs = np.where(mask_bool)
                    if len(xs) > 0:
                        y_min, y_max, x_min, x_max = ys.min(), ys.max(), xs.min(), xs.max()
                        
                        # Convert numpy ints to native Python ints for JSON serialization
                        y_min, y_max = int(y_min), int(y_max)
                        x_min, x_max = int(x_min), int(x_max)
                        
                        obj_h, obj_w = y_max - y_min, x_max - x_min
                        center_x, center_y = x_min + obj_w // 2, y_min + obj_h // 2
                        
                        font_scale = 3.0
                        font_thickness = 3
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (text_w, text_h), baseline = cv2.getTextSize(class_name, font, font_scale, font_thickness)        
                        
                        # Shrink text to fit inside the hull bounding box
                        while (text_w > obj_w * 1.8) and font_scale > 0.5:
                            font_scale -= 0.1
                            text_w, text_h = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
                            
                        text_x = int(center_x - text_w / 2)
                        text_y = int(center_y + text_h / 2)
                        
                        # --- Draw Tight Rectangle Background ---
                        pad = 1  # Padding pixels
                        
                        pt1 = (text_x - pad, text_y - text_h - pad)
                        pt2 = (text_x + text_w + pad, text_y + baseline + pad)
                        name, name2 =  jfile['path'].split(".")
                        
                        # Create the folder(s) containing the file
                        fulls_path = os.path.join(DATA_ROOT + "/textmask_square/", name+"_"+str(i_)+"_"+class_name.replace(" ", "-")+"."+name2)
                        Path(fulls_path).parent.mkdir(parents=True, exist_ok=True)
                        
                        target_h, target_w = image_rgb.shape[:2]
                        target_h, target_w = int(target_h), int(target_w)
                        
                        ratio = (x_max - x_min) * (y_max - y_min ) / (target_h * target_w)
                        if ratio <= 0.004:
                            continue

                        cpath = os.path.join("textmask_square", name+"_"+str(i_)+"_"+class_name.replace(" ", "-")+"."+name2)
                        
                        # --- NEW: Write line-by-line to JSONL ---
                        entry = {
                            cpath: {
                                "gt_path": image_path, 
                                "pt1": pt1, 
                                "pt2": pt2, 
                                "bbox": [x_min, y_min, x_max, y_max],
                                "resolution": {"height": target_h, "width": target_w}
                            }
                        }
                        
                        # Dump exactly one JSON object per line and write it
                        f_out.write(json.dumps(entry) + "\n")
                        # Flush the buffer to physically write to disk immediately (crash-proof)
                        f_out.flush() 

                        # Save visual mask image if required
                        if id_ > 1:
                            cv2.rectangle(masked_text_img, pt1, pt2, (0, 0, 0), -1)
                            cv2.putText(masked_text_img, class_name, (text_x, text_y), 
                                        font, font_scale, (255, 255, 255), font_thickness)
                            
                            # Save the image
                            masked_text_img = cv2.cvtColor(masked_text_img, cv2.COLOR_RGB2BGR)
                            #cv2.imwrite(fulls_path, masked_text_img)

            except Exception as e:
                print(f"Error processing {jfilename}: {e}")
                continue

print("Processing complete. All bounding box entries saved to JSONL.")