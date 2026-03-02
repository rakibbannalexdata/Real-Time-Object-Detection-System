# YOLOv8 Segmentation Training Workflow

This document provides a step-by-step guide on how to prepare data, train models, and perform inference using this system.

## 1. Prepare Your Dataset
Organize your data in the `datasets/` directory. Each project must follow this structure:

```text
datasets/
└── <your_project_name>/
    ├── train/
    │   ├── images/  (e.g., .jpg, .png)
    │   └── labels/  (e.g., .txt files in YOLO format)
    └── val/
        ├── images/
        └── labels/
```

**Label Format**: Each line in the `.txt` files should represent a polygon mask:
`<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`
*(Coordinates must be normalized between 0.0 and 1.0)*

## 2. Validate Dataset
Check if your dataset is correctly formatted by calling:
**Endpoint**: `GET /api/v1/training/dataset/info?project=<your_project_name>`

This will return image counts, detected classes, and any errors found (e.g., missing labels or out-of-bounds coordinates).

## 3. Start Training
Initiate training as a background task.
**Endpoint**: `POST /api/v1/training/start`

**Request Body**:
```json
{
  "project_name": "your_project_name",
  "epochs": 100,
  "imgsz": 640,
  "batch": 16,
  "class_names": {
    "0": "person",
    "1": "bicycle"
  }
}
```
*Note: `class_names` is optional but recommended for readable results.*

## 4. Monitor Progress
Training runs in the background. You can monitor the process in the server logs (`uvicorn` output). Once finished, the weights will be saved at:
`models/<your_project_name>/weights/best.pt`

## 5. Test Inference
Use your newly trained model without restarting the server.
**Endpoint**: `POST /api/v1/detect/image?model_path=models/<your_project_name>/weights/best.pt`

**Parameters**:
- `file`: Upload your image.
- `model_path`: Path to your `.pt` file.
- `confidence_threshold`: (Optional) e.g., `0.25`

The response will include a `segmentation` field with the polygon coordinates for each detected object.

---
*Created by Antigravity AI*
