import json
import numpy as np
import cv2
import os
import shutil
import argparse
from pycocotools import mask as mask_utils

def process_coco_json(input_path, output_path):
    print(f"🔍 Processing: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    new_annotations = []
    processed_count = 0
    # Create image lookup for fast dimension access
    image_dims = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    # Ensure categories are 1-based (Ultralytics converter uses category_id - 1)
    # If there's a category with ID 0, shift all IDs by 1
    cat_shift = 0
    if any(cat['id'] <= 0 for cat in data.get('categories', [])):
        print("ℹ️ Shifting category IDs to be 1-based...")
        cat_shift = 1
        for cat in data['categories']:
            cat['id'] += cat_shift
            
    for ann in data['annotations']:
        img_id = ann.get('image_id')
        if img_id not in image_dims:
            continue
        
        w, h = image_dims[img_id]
        
        # 1. Processing segmentation
        processed_seg = None
        
        if isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
            # Handle RLE
            seg = ann['segmentation']
            rle = mask_utils.frPyObjects(seg, h, w) if isinstance(seg['counts'], list) else seg
            try:
                binary_mask = mask_utils.decode(rle)
                Contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                polygons = [c.flatten().astype(float).tolist() for c in Contours if c.size >= 6]
                if polygons:
                    processed_seg = polygons
            except Exception as e:
                print(f"⚠️ Warning: RLE decode failed for ID {ann.get('id')}: {e}")

        elif isinstance(ann['segmentation'], list):
            # Handle existing polygons
            polygons = []
            for p in ann['segmentation']:
                if isinstance(p, list) and len(p) >= 6:
                    # Clean coordinates: ensure they are within bounds and are real numbers
                    pts = [float(x) for x in p if x is not None]
                    if len(pts) >= 6:
                        polygons.append(pts)
            if polygons:
                processed_seg = polygons
        
        # 2. If we have valid polygons, update and keep the annotation
        if processed_seg:
            ann['segmentation'] = processed_seg
            
            # Use shifted category ID
            raw_cat_id = ann.get('category_id')
            if raw_cat_id is None:
                continue
            ann['category_id'] = int(raw_cat_id) + cat_shift
            
            # Regenerate bbox [x, y, w, h] from polygons
            try:
                all_pts = np.concatenate([np.array(p).reshape(-1, 2) for p in processed_seg])
                x_min, y_min = np.min(all_pts, axis=0)
                x_max, y_max = np.max(all_pts, axis=0)
                
                # Sanity check for bbox
                if x_max <= x_min or y_max <= y_min:
                    continue
                    
                ann['bbox'] = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                # Area calculation
                ann['area'] = float(np.sum([mask_utils.area(mask_utils.frPyObjects([p], h, w)) for p in processed_seg]))
                
                new_annotations.append(ann)
                processed_count += 1
            except Exception:
                continue
        else:
            # Skip annotations without valid segmentation (polygons)
            continue

    # Update the data and save
    data['annotations'] = new_annotations
    
    # Create backup
    backup_path = input_path + ".bak"
    if not os.path.exists(backup_path): # Only backup once
        shutil.copy2(input_path, backup_path)
        print(f"💾 Backup created at: {backup_path}")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Processed {processed_count} annotations. File saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO RLE masks to polygons and remove bboxes.")
    parser.add_argument("--input", default='/home/rakib-ul-banna/projects/test-detection-system/main_data_sets/stout/train/_annotations.coco.json', help="Input COCO JSON file")
    parser.add_argument("--output", help="Output COCO JSON file (defaults to input path if not specified)")
    
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output if args.output else input_file
    
    process_coco_json(input_file, output_file)
 