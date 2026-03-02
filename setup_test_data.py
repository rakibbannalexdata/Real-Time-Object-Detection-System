import os
from pathlib import Path

def create_dummy_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create a 640x640 black PNG
    from PIL import Image
    img = Image.new('RGB', (640, 640), color='black')
    img.save(path)

def setup_project(name, train_images, val_images, valid=True, error_type=None):
    base_dir = Path("datasets") / name
    for split in ["train", "val"]:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        count = train_images if split == "train" else val_images
        for i in range(count):
            img_path = img_dir / f"img_{i}.png"
            lbl_path = lbl_dir / f"img_{i}.txt"
            create_dummy_image(img_path)
            
            if error_type == "missing_label" and i == 0:
                continue
                
            with open(lbl_path, "w") as f:
                if error_type == "empty_label" and i == 0:
                    pass
                elif error_type == "invalid_polygon" and i == 0:
                    f.write("0 0.1 0.2 0.3\n") # 3 coords (odd)
                elif error_type == "out_of_bounds" and i == 0:
                    f.write("0 0.1 0.2 0.3 1.5\n") # coord 1.5 > 1.0
                elif error_type == "non_contiguous_classes":
                    f.write("0 0.1 0.2 0.3 0.4\n")
                    f.write("2 0.5 0.6 0.7 0.8\n") # Skip class 1
                else:
                    f.write("0 0.1 0.2 0.3 0.4 0.5 0.6\n")

if __name__ == "__main__":
    setup_project("valid_project", 2, 1)
    setup_project("missing_label_project", 2, 1, error_type="missing_label")
    setup_project("invalid_polygon_project", 2, 1, error_type="invalid_polygon")
    setup_project("out_of_bounds_project", 2, 1, error_type="out_of_bounds")
    setup_project("empty_label_project", 2, 1, error_type="empty_label")
    setup_project("non_contiguous_project", 2, 1, error_type="non_contiguous_classes")
    print("Test datasets created in 'datasets/'")
