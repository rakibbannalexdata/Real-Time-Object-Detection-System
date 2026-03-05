import os
import shutil
from pathlib import Path
import yaml

def setup_dataset(project_name):
    root_dir = Path.cwd()
    source_base = root_dir / "main_data_sets" / project_name
    target_base = root_dir / "datasets" / project_name
    
    # 1. Clean existing target
    if target_base.exists():
        print(f"Cleaning existing directory: {target_base}")
        shutil.rmtree(target_base)
    
    target_base.mkdir(parents=True, exist_ok=True)
    
    # 2. Copy and reformat dataset.yaml
    source_yaml = source_base / "data.yaml"
    if not source_yaml.exists():
        source_yaml = source_base / "dataset.yaml"
        
    target_yaml = target_base / "dataset.yaml"
    
    if source_yaml.exists():
        with open(source_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update paths to be absolute and match the new structure
        data['path'] = str(target_base.absolute())
        data['train'] = "train/images"
        data['val'] = "val/images"
        if 'test' in data:
            data['test'] = "test/images"
            
        with open(target_yaml, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"Reformated and copied {source_yaml.name} to {target_yaml}")
    else:
        print(f"Warning: No dataset config found at {source_yaml}")

    # 3. Copy train/val splits with directory mapping (valid -> val)
    for split in ["train", "val", "test"]:
        source_split_name = split
        # fallback for 'val' can be 'valid'
        if split == "val" and not (source_base / split).exists() and (source_base / "valid").exists():
            source_split_name = "valid"
            
        source_split = source_base / source_split_name
        target_split = target_base / split
        
        if source_split.exists():
            print(f"Copying {source_split_name} split to {split}...")
            shutil.copytree(source_split, target_split)
            
            # 4. Sanitize labels: remove bounding box annotations (exactly 5 values)
            labels_dir = target_split / "labels"
            if labels_dir.exists():
                sanitized_count = 0
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, "r") as f:
                        lines = f.readlines()
                    
                    # Keep only lines that are NOT bounding boxes (len != 5)
                    valid_lines = [line for line in lines if len(line.strip().split()) != 5]
                    
                    if len(valid_lines) != len(lines):
                        sanitized_count += len(lines) - len(valid_lines)
                        with open(label_file, "w") as f:
                            f.writelines(valid_lines)
                
                if sanitized_count > 0:
                    print(f"Sanitized {split}: removed {sanitized_count} bounding box annotations")

    print(f"Dataset setup complete at {target_base}")

if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "weedsVsCrops"
    setup_dataset(project)
