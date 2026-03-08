import os
import shutil
import argparse
import yaml
import json
import random
from ultralytics.data.converter import convert_coco

def extract_categories(json_path):
    """Extracts category names and IDs from COCO JSON."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        categories = data.get('categories', [])
        return {cat['id']: cat['name'] for cat in categories}
    except Exception as e:
        print(f"⚠️ Warning: Could not parse categories from {json_path}: {e}")
        return {0: "crop"} # Fallback

def main(folder_name=None):
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format with dynamic categories and auto-splitting.")
    
    # Use folder_name to provide defaults for argparse
    default_src = f"main_data_sets/{folder_name}" if folder_name else None
    default_project = folder_name if folder_name else None

    parser.add_argument("--src", default=default_src, help=f"Source directory folder. Default: {default_src}")
    parser.add_argument("--project", default=default_project, help=f"Target project name. Default: {default_project}")
    
    args = parser.parse_args()
    
    if not args.src:
        parser.error("the following arguments are required: --src (or pass folder_name to main())")

    src_root = os.path.abspath(args.src)
    project_name = args.project if args.project else os.path.basename(src_root.rstrip(os.sep))
    project_dir = os.path.abspath(f"datasets/{project_name}")
    
    print(f"🚀 Starting dataset conversion: {src_root} -> {project_dir}")

    # Standard split detection
    src_train = os.path.join(src_root, "train")
    src_val = os.path.join(src_root, "val")
    if not os.path.exists(src_val):
        src_val = os.path.join(src_root, "valid")

    # Discover categories from train split (main data source)
    train_json = os.path.join(src_train, "_annotations.coco.json")
    category_names = extract_categories(train_json)
    print(f"🏷️ Extracted {len(category_names)} categories: {list(category_names.values())}")

    def process_split(src_path, split_name):
        if not os.path.exists(src_path):
            return 0
        
        dst_labels_dir = os.path.join(project_dir, split_name, "labels")
        dst_images_dir = os.path.join(project_dir, split_name, "images")
        os.makedirs(dst_labels_dir, exist_ok=True)
        os.makedirs(dst_images_dir, exist_ok=True)

        print(f"📂 Converting COCO for split: {split_name}...")
        # ultralytics convert_coco will handle the mapping to YOLO format
        convert_coco(labels_dir=src_path, save_dir=dst_labels_dir, use_segments=True)

        # Move and normalize labels from 'labels2/labels/_annotations.coco' folder (ultralytics quirk)
        labels_temp_root = dst_labels_dir + "2"
        if os.path.exists(labels_temp_root):
            for root, _, files in os.walk(labels_temp_root):
                for filename in files:
                    if filename.endswith(".txt"):
                        src_label_path = os.path.join(root, filename)
                        dst_label_path = os.path.join(dst_labels_dir, filename)
                        # We just move them. Note: We ARE NOT force-resetting to ID 0 here 
                        # to respect the multi-category discovery request.
                        shutil.move(src_label_path, dst_label_path)
            shutil.rmtree(labels_temp_root)

        # Copy images
        img_list = []
        for filename in os.listdir(src_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(src_path, filename), os.path.join(dst_images_dir, filename))
                img_list.append(filename)
        
        print(f"✅ Split '{split_name}' processed with {len(img_list)} images.")
        return img_list

    # Initial processing
    train_images = process_split(src_train, "train")
    val_images = process_split(src_val, "val")

    # Auto-split logic if val is empty (typical for 'a' dataset)
    if not val_images and train_images:
        print("⚖️ No validation set found. Performing 80/20 auto-split...")
        val_count = max(1, int(len(train_images) * 0.2))
            
        random.shuffle(train_images)
        to_move = train_images[:val_count]
        
        val_labels_dir = os.path.join(project_dir, "val", "labels")
        val_images_dir = os.path.join(project_dir, "val", "images")
        train_labels_dir = os.path.join(project_dir, "train", "labels")
        train_images_dir = os.path.join(project_dir, "train", "images")
        
        os.makedirs(val_labels_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        
        for img_name in to_move:
            # Move Image
            shutil.move(os.path.join(train_images_dir, img_name), os.path.join(val_images_dir, img_name))
            # Move Label (matching name, replace extension with .txt)
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            train_lbl_path = os.path.join(train_labels_dir, lbl_name)
            if os.path.exists(train_lbl_path):
                shutil.move(train_lbl_path, os.path.join(val_labels_dir, lbl_name))
        
        print(f"📦 Moved {len(to_move)} files to validation set.")
        val_images = to_move

    # Generate dataset.yaml
    if train_images:
        yaml_data = {
            "path": project_dir,
            "train": "train/images",
            "val": "val/images",
            "names": category_names
        }
        
        yaml_path = os.path.join(project_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        print(f"📄 Generated {yaml_path}")

    print(f"✨ All done! Dataset '{project_name}' is ready in 'datasets/'.")

if __name__ == "__main__":
    main(folder_name="a")