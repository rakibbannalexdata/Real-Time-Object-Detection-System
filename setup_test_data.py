import os
import shutil
from pathlib import Path
import cv2
import numpy as np

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

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
        
        # Create 20 dummy images to help convergence
        num_images = 20 if split == "train" else 2
        for img_idx in range(num_images):
            img_name = f"image_{img_idx}.jpg"
            lbl_name = f"image_{img_idx}.txt"
            
            # Create dummy image
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            
            with open(lbl_dir / lbl_name, "w") as f:
                # Use only first 3 classes (Person, Bicycle, Car) for the smoke test
                for class_id in range(3):
                    # Distinct positions for each class
                    offset = class_id * 0.2 + (img_idx * 0.01) # Slight jitter
                    x1, y1 = 0.1 + offset, 0.1 + offset
                    x2, y2 = 0.2 + offset, 0.2 + offset
                    
                    # Ensure within bounds
                    x1, y1 = max(0, min(1, x1)), max(0, min(1, y1))
                    x2, y2 = max(0, min(1, x2)), max(0, min(1, y2))
                    
                    p1 = (int(x1 * 640), int(y1 * 640))
                    p2 = (int(x2 * 640), int(y2 * 640))
                    
                    # Draw in image
                    cv2.rectangle(img, p1, p2, (255, 255, 255), -1)
                    cv2.putText(img, COCO_CLASSES[class_id], (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Write label
                    f.write(f"{class_id} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}\n")
            
            cv2.imwrite(str(img_dir / img_name), img)

    print(f"Test data setup complete for {project_name} (10 train images, 3 active classes)")

if __name__ == "__main__":
    setup_test_data()
