import os
from univl.paths import DATA_ROOT
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from pathlib import Path

kernel = np.ones((3, 3), np.uint8)

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
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask

# --- Helper Function: Get Convex Hull Mask (FIXED) ---
def get_convex_hull_mask(binary_mask):
    """
    Computes the convex hull of a binary mask.
    Returns: A new binary mask (same shape) representing the convex hull.
    """
    # 1. Ensure the input is strictly contiguous and uint8
    mask_uint8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    
    # 2. Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Create an empty mask for the hull (ensure contiguous memory)
    hull_mask = np.zeros(mask_uint8.shape, dtype=np.uint8)
    
    # 4. Compute and draw hull
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
        
    # Convert back to binary (0 or 1)
    return (hull_mask > 0).astype(np.uint8)

# --- Setup Paths ---
for id_ in range(2): #range(8):
    path_ = DATA_ROOT
    #jsonpath = os.path.join(path_, "jsons/part0_step4_aes_clip_withcount")
    jsonpath = os.path.join(path_, "jsons/part"+str(id_)+"_step4_aes_clip_withcount")

    filelist = [os.path.join(jsonpath, file) for file in os.listdir(jsonpath) if file.endswith('.json')]
    filelist.sort()

    # --- Main Logic ---

    for jfilename in filelist:
        try:
            # 1. Load Data
            with open(jfilename) as f:
                jfile = json.load(f)

            image_path = os.path.join(path_, "laion-aes", jfile['path'])
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            h = jfile['resolution']['height']
            w = jfile['resolution']['width']

            np.random.seed(42) 

            print(f"Processing image: {jfile['path']}")
            objects = jfile['segmentation']['object']
            print(f"Found {len(objects)} objects.")

            # 3. Iterate, Plot Three Columns, and Build Combined

            for i_, obj in enumerate(objects):
                class_name = obj['class_name']
                rle_str = obj['mask']
                
                # 1. Get Original Mask
                raw_mask = rle_to_mask(rle_str, h, w)
                
                # 2. Compute Convex Hull Mask (Using fixed function)
                hull_mask = get_convex_hull_mask(raw_mask)
                mask_bool = hull_mask.astype(bool) 

                masked_overlay = image_rgb.copy()
                masked_overlay[~mask_bool] = 0 


                masked_text_img = image_rgb.copy()
                masked_text_img[mask_bool] = 0
                
                # --- Text Calculation ---
                ys, xs = np.where(mask_bool)
                if len(xs) > 0:
                    y_min, y_max, x_min, x_max = ys.min(), ys.max(), xs.min(), xs.max()
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
                    
                    # Coordinates for the background box
                    # Top-Left: (x, y - text_height)
                    # Bottom-Right: (x + width, y + baseline)
                    pt1 = (text_x - pad, text_y - text_h - pad)
                    pt2 = (text_x + text_w + pad, text_y + baseline + pad)
                    
                    # Draw filled rectangle (using the object's random color)
                    cv2.rectangle(masked_text_img, pt1, pt2, (0, 0, 0), -1)
                    cv2.putText(masked_text_img, class_name, (text_x, text_y), 
                                font, font_scale, (255, 255, 255), font_thickness)
                    
                    name, name2 =  jfile['path'].split(".")
                    fulls_path = os.path.join(DATA_ROOT + "/textmask_bg/", name+"_"+str(i_)+"_"+class_name.replace(" ", "-")+"."+name2)
                    # Create the folder(s) containing the file
                    Path(fulls_path).parent.mkdir(parents=True, exist_ok=True)

                    # Save the image
                    masked_text_img = cv2.cvtColor(masked_text_img, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(fulls_path, masked_text_img)

        except:
            continue
