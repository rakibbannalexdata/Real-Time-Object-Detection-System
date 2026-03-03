import os
import shutil
from pathlib import Path

def setup_coco128_seg():
    source_base = Path("/home/rakib-ul-banna/projects/test-detection-system/main_data_sets/coco128-seg")
    target_base = Path("/home/rakib-ul-banna/projects/test-detection-system/datasets/coco128-seg")
    
    # Clean existing target
    if target_base.exists():
        print(f"Cleaning existing directory: {target_base}")
        shutil.rmtree(target_base)
    
    target_base.mkdir(parents=True, exist_ok=True)
    
    # Copy dataset.yaml
    if (source_base / "dataset.yaml").exists():
        shutil.copy(source_base / "dataset.yaml", target_base / "dataset.yaml")
        print(f"Copied dataset.yaml to {target_base}")
    
    # Copy train/val splits
    for split in ["train", "val"]:
        source_split = source_base / split
        target_split = target_base / split
        
        if source_split.exists():
            print(f"Copying {split} split...")
            shutil.copytree(source_split, target_split)
        else:
            # Fallback for nested structure if needed, or if images/labels are at root
            # Based on list_dir, they are in train/ and val/
            pass

    print(f"Dataset setup complete at {target_base}")

if __name__ == "__main__":
    setup_coco128_seg()
