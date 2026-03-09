import json
import os
import shutil
import random
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, images_dir, output_images_dir, output_labels_dir):
    """
    Converts COCO JSON to YOLO segmentation format.
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Map category IDs to names
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Map image IDs to image info
    images = {img['id']: img for img in data['images']}
    
    # Track annotations by image ID
    annotations_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Source image path
        src_img_path = images_dir / file_name
        if not src_img_path.exists():
            print(f"Warning: Image {src_img_path} not found. Skipping...")
            continue

        # Target image path
        target_img_path = output_images_dir / os.path.basename(file_name)
        target_label_path = output_labels_dir / f"{Path(file_name).stem}.txt"

        # Copy image
        shutil.copy(src_img_path, target_img_path)

        # Generate label file
        label_lines = []
        if img_id in annotations_by_img:
            for ann in annotations_by_img[img_id]:
                cat_id = ann['category_id']
                segmentation = ann.get('segmentation')
                if not segmentation:
                    continue
                
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... (normalized)
                for poly in segmentation:
                    if not isinstance(poly, list): continue
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        x = poly[i] / width
                        y = poly[i+1] / height
                        normalized_poly.extend([f"{x:.6f}", f"{y:.6f}"])
                    
                    label_lines.append(f"{cat_id} {' '.join(normalized_poly)}")

        if label_lines:
            target_label_path.write_text("\n".join(label_lines))

def setup_weeds_vs_crops(project_name="weedsVsCrops"):
    root_dir = Path.cwd()
    source_dir = root_dir / "main_data_sets" / project_name / "train"
    coco_json = source_dir / "_annotations.coco.json"
    
    target_base = root_dir / "datasets" / project_name
    if target_base.exists():
        shutil.rmtree(target_base)
    
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    image_ids = [img['id'] for img in data['images']]
    random.seed(42)
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * 0.8)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    def filter_coco(original_data, id_set):
        subset = original_data.copy()
        subset['images'] = [img for img in original_data['images'] if img['id'] in id_set]
        subset['annotations'] = [ann for ann in original_data['annotations'] if ann['image_id'] in id_set]
        return subset

    train_json = target_base / "train_temp.json"
    val_json = target_base / "val_temp.json"
    target_base.mkdir(parents=True, exist_ok=True)
    
    with open(train_json, 'w') as f:
        json.dump(filter_coco(data, train_ids), f)
    with open(val_json, 'w') as f:
        json.dump(filter_coco(data, val_ids), f)

    print("Converting Train split...")
    convert_coco_to_yolo(train_json, source_dir, target_base / "train" / "images", target_base / "train" / "labels")
    
    print("Converting Val split...")
    convert_coco_to_yolo(val_json, source_dir, target_base / "val" / "images", target_base / "val" / "labels")

    train_json.unlink()
    val_json.unlink()

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    names_dict = "\n".join([f"  {cid}: {name}" for cid, name in categories.items()])
    
    yaml_content = f"""path: {target_base.absolute()}
train: train/images
val: val/images
names:
{names_dict}
"""
    (target_base / "dataset.yaml").write_text(yaml_content)

    print(f"✅ Conversion finished! Dataset located at {target_base}")

if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "plants"
    setup_weeds_vs_crops(project)