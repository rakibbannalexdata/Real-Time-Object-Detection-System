# Segmentation Training Implementation Plan

This plan details the addition of a new segmentation model training feature to our YOLOv8 FastAPI backend.

## Proposed Changes

### Core Integration
- Add a new `app/schemas/training_schema.py` for training-related request and response models.
- Add a new `app/services/training_service.py` to handle the heavy lifting (validation, yaml generation, model training).
- Add a new `app/api/routes/training.py` with the new endpoints.
- Update [app/main.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/main.py) to include the new `training.py` router.

### [NEW] app/schemas/training_schema.py
- `TrainingStartRequest`
  - `project_name` (str)
  - `epochs` (int, default=100)
  - `imgsz` (int, default=640)
  - `batch` (int, default=16)
- `DatasetSummaryResponse`
  - `project` (str)
  - `train_images` (int)
  - `val_images` (int)
  - `total_classes` (int)
  - `classes_detected` (list[int])
  - `dataset_valid` (bool)

### [NEW] app/services/training_service.py
- `TrainingService` class
  - **Dataset Validation (`_validate_dataset`)**:
    - Ensures the directory structure `datasets/{project_name}/(train|val)/(images|labels)` exists.
    - Loops through every `.jpg`, `.jpeg`, `.png` in `images`.
    - Checks that a [.txt](file:///home/rakib-ul-banna/projects/test-detection-system/requirements.txt) file with the exact same base name exists in `labels`.
    - Reads the [.txt](file:///home/rakib-ul-banna/projects/test-detection-system/requirements.txt) file and ensures:
      - It is not empty.
      - Each line consists of an integer `class_id` followed by an even number of coordinates (all between `0` and `1`).
    - Collects unique `class_id`s to automatically compute the number of classes.
    - Will raise specific exceptions (e.g. `MissingLabelError`, `EmptyLabelError`, `InvalidPolygonError`) that bubble up.
  - **YAML Generation (`_generate_yaml`)**:
    - Takes the unique class IDs found during validation.
    - Creates `datasets/{project_name}/dataset.yaml` specifying `train: train/images`, `val: val/images`, and the dynamically generated `names` mapping.
  - **Dataset Summary (`get_dataset_info`)**:
    - Returns the `DatasetSummaryResponse`. Triggers the validation step automatically and aggregates the counts of train/val images and class stats.
  - **Model Training (`start_training`)**:
    - Calls `_validate_dataset`.
    - Calls `_generate_yaml`.
    - Triggers `YOLO("yolov8n-seg.pt").train(...)` correctly in a non-blocking thread or explicitly on the event loop so the endpoint can return quickly (or run it synchronously if required, but standard practice in FastAPI is to return a task ID. For simplicity if the prompt implies blocking, we will block, or run in a `ThreadPoolExecutor`).

### [NEW] app/api/routes/training.py
- **POST `/training/start`**
  - Accepts `TrainingStartRequest`.
  - Runs validation; if validation fails, catches custom errors and returns HTTP 400 with structured JSON format detailing the error reason (missing label, corrupt image, etc.)
  - If validation passes, starts training and returns where the best weights will be stored (`models/{project_name}/weights/best.pt`).
- **GET `/training/dataset/info`**
  - Accepts `project` query param.
  - Returns the `DatasetSummaryResponse` or throws 404 if the project dataset doesn't exist.

### [MODIFY] app/main.py
- Include the new `/training` router in the FastAPI app.

## Verification Plan

### Automated tests via CURL
- Missing labels: Create a dataset with an image but no [.txt](file:///home/rakib-ul-banna/projects/test-detection-system/requirements.txt) file -> Expect HTTP 400
- Invalid polygon: Add a [.txt](file:///home/rakib-ul-banna/projects/test-detection-system/requirements.txt) with odd number of coordinates -> Expect HTTP 400
- Out of bounds coordinate: Add a [.txt](file:///home/rakib-ul-banna/projects/test-detection-system/requirements.txt) with coordinate > 1 -> Expect HTTP 400
- Dataset info endpoint: Call `/dataset/info?project=valid_project` -> Expect correct count of images and classes.
- Valid training trigger: Should generate YAML and kick off the YOLO engine.
