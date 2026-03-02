import os
import shutil
from pathlib import Path

def create_yolo_segmentation_file(file_path, num_classes=3):
    """
    Creates a dummy YOLO segmentation label file.
    Format: class_id x1 y1 x2 y2 ... xn yn
    """
    with open(file_path, "w") as f:
        for i in range(num_classes):
            # A simple square polygon: (0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)
            f.write(f"{i} 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

def setup_test_data(project_name="valid_project"):
    base_path = Path("datasets") / project_name
    
    # Clean existing
    if base_path.exists():
        shutil.rmtree(base_path)
    
    for split in ["train", "val"]:
        img_dir = base_path / split / "images"
        lbl_dir = base_path / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 2 dummy images and labels per split
        for i in range(2):
            img_name = f"image_{i}.jpg"
            lbl_name = f"image_{i}.txt"
            
            # Create valid dummy image using numpy/cv2
            import cv2
            import numpy as np
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Test Image", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imwrite(str(img_dir / img_name), img)
            # Create valid label file
            create_yolo_segmentation_file(lbl_dir / lbl_name)

    print(f"Test data setup complete for {project_name}")

if __name__ == "__main__":
    setup_test_data()
    
    # Create an invalid project too
    # 1. Missing labels
    base_invalid = Path("datasets/missing_labels")
    (base_invalid / "train/images").mkdir(parents=True, exist_ok=True)
    (base_invalid / "train/labels").mkdir(parents=True, exist_ok=True)
    (base_invalid / "train/images/img1.jpg").touch()
    
    # 2. Invalid polygons
    base_poly = Path("datasets/invalid_poly")
    (base_poly / "train/images").mkdir(parents=True, exist_ok=True)
    (base_poly / "train/labels").mkdir(parents=True, exist_ok=True)
    (base_poly / "train/images/img1.jpg").touch()
    with open(base_poly / "train/labels/img1.txt", "w") as f:
        f.write("0 0.1 0.1 0.2\n") # Odd number of coordinates
