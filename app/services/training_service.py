import logging
import os
import threading
import yaml
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status
from ultralytics import YOLO

from app.schemas.training_schema import DatasetSummaryResponse, TrainingStatusResponse

logger = logging.getLogger(__name__)

class TrainingService:
    """
    Handles training validation, dataset.yaml generation, and YOLOv8 segmentation model training.
    """

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, datasets_dir: str = "datasets", models_dir: str = "models"):
        self.datasets_dir = Path(datasets_dir)
        self.models_dir = Path(models_dir)
        self._training_states: dict[str, dict] = {}
        self._state_lock = threading.Lock()

    def _validate_polygon_line(self, line: str, line_no: int, filepath: Path) -> int:
        """
        Validates a single line of a YOLO segmentation label file.
        Returns the class ID.
        Raises ValueError if invalid.
        """
        parts = line.strip().split()
        if not parts:
            raise ValueError(f"Empty line {line_no} in {filepath}")

        # 1. First value must be an integer class_id
        try:
            class_id = int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid class ID '{parts[0]}' on line {line_no} in {filepath}")

        if class_id < 0:
            raise ValueError(f"Negative class ID {class_id} on line {line_no} in {filepath}")

        # 2. Remaining values must be an EVEN number of coordinates (x, y pairs)
        coords = parts[1:]
        if len(coords) == 0:
            raise ValueError(f"No coordinates found for class {class_id} on line {line_no} in {filepath}")
        if len(coords) % 2 != 0:
            raise ValueError(f"Odd number of coordinates ({len(coords)}) for polygon on line {line_no} in {filepath}")

        # 3. Coordinates must be floats between 0.0 and 1.0 (normalized)
        for i, coord_str in enumerate(coords):
            try:
                coord = float(coord_str)
            except ValueError:
                raise ValueError(f"Invalid coordinate '{coord_str}' on line {line_no} in {filepath}")
            if not (0.0 <= coord <= 1.0):
                raise ValueError(f"Coordinate {coord} out of bounds (0-1) on line {line_no} in {filepath}")

        return class_id

    def _validate_split(self, split_dir: Path) -> tuple[int, set[int]]:
        """
        Validates a specific split (train or val).
        Returns (image_count, unique_classes_set).
        """
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not images_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory not found: {images_dir}"
            )
        if not labels_dir.exists() or not labels_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory not found: {labels_dir}"
            )

        unique_classes = set()
        image_count = 0

        # Scan all images
        for entry in os.scandir(images_dir):
            if not entry.is_file():
                continue
            
            ext = Path(entry.name).suffix.lower()
            if ext not in self.ALLOWED_EXTENSIONS:
                continue
            
            image_count += 1
            base_name = Path(entry.name).stem

            # Check matching label file exists
            label_file = labels_dir / f"{base_name}.txt"
            if not label_file.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing label file '{label_file.name}' for image '{entry.name}' in {split_dir.name}"
                )

            # Validate label file contents
            try:
                content = label_file.read_text().strip()
                if not content:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Label file is empty: {label_file}"
                    )

                for line_no, line in enumerate(content.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    class_id = self._validate_polygon_line(line, line_no, label_file)
                    unique_classes.add(class_id)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                ) from e

        return image_count, unique_classes

    def validate_dataset(self, project_name: str) -> DatasetSummaryResponse:
        """
        Performs strict validation on the dataset format before training.
        Returns a summary containing image counts and detected classes.
        """
        project_dir = self.datasets_dir / project_name
        if not project_dir.exists() or not project_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project dataset directory not found: {project_dir}"
            )

        # Validate Train
        train_count, train_classes = self._validate_split(project_dir / "train")
        if train_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No images found in {project_dir / 'train' / 'images'}"
            )

        # Validate Val
        val_count, val_classes = self._validate_split(project_dir / "val")
        if val_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No images found in {project_dir / 'val' / 'images'}"
            )

        all_classes = train_classes.union(val_classes)
        valid_detected_classes = sorted(list(all_classes))
        # Support non-contiguous classes for COCO and other real datasets
        max_class_id = max(valid_detected_classes) if valid_detected_classes else 0

        return DatasetSummaryResponse(
            project=project_name,
            train_images=train_count,
            val_images=val_count,
            total_classes=len(valid_detected_classes),
            classes_detected=valid_detected_classes,
            dataset_valid=True
        )

    def get_training_status(self, project_name: str) -> TrainingStatusResponse:
        """
        Retrieves the current training status for a project.
        """
        with self._state_lock:
            state = self._training_states.get(project_name)

        if not state:
            # Check if training has already been completed in a previous backend run
            best_weights_path = self.models_dir / project_name / "weights" / "best.pt"
            if best_weights_path.exists():
                return TrainingStatusResponse(
                    project=project_name,
                    status="completed",
                    progress=100.0,
                    message="Training finished successfully in a previous session."
                )
            
            return TrainingStatusResponse(
                project=project_name,
                status="idle",
                message="No active training found for this project."
            )
            
        return TrainingStatusResponse(**state)

    def _update_state(self, project_name: str, **kwargs):
        with self._state_lock:
            if project_name not in self._training_states:
                self._training_states[project_name] = {
                    "project": project_name,
                    "status": "idle",
                    "current_epoch": 0,
                    "total_epochs": 0,
                    "progress": 0.0,
                    "message": ""
                }
            self._training_states[project_name].update(kwargs)

    def _generate_yaml(self, project_name: str, max_class_id: int, class_names: dict[int, str] = None) -> Path:
        """
        Generates the dataset.yaml dynamically for YOLO training.
        Attempts to preserve existing class names if none are provided.
        """
        project_dir = self.datasets_dir / project_name
        yaml_path = project_dir / "dataset.yaml"

        # 1. Start with provided or empty dict
        final_class_names = class_names or {}

        # 2. If no names provided, try to load from existing YAML
        if not final_class_names and yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'names' in data:
                        # Convert keys to int if they are strings (YAML loader might return strings or ints)
                        raw_names = data['names']
                        if isinstance(raw_names, dict):
                            final_class_names = {int(k): v for k, v in raw_names.items()}
                        elif isinstance(raw_names, list):
                            final_class_names = {i: v for i, v in enumerate(raw_names)}
            except Exception as e:
                logger.warning(f"Could not parse existing dataset.yaml for names: {e}")

        # 3. Format class names dynamically (up to max_class_id)
        # Use provided/existing names, default to class_i if still missing
        names_dict = "\n".join([f"  {i}: {final_class_names.get(i, f'class_{i}')}" for i in range(max_class_id + 1)])

        # Path must be absolute for YOLO to resolve correctly
        abs_project_dir = project_dir.absolute()

        yaml_content = f"""path: {abs_project_dir}
train: train/images
val: val/images
names:
{names_dict}
"""
        yaml_path.write_text(yaml_content)
        logger.info(f"Generated dataset.yaml at {yaml_path}")
        return yaml_path

    def start_training_sync(self, project_name: str, epochs: int, imgsz: int, batch: int, class_names: dict[int, str] = None) -> str:
        """
        Synchronously starts YOLO training.
        Intended to be run in a background thread via FastAPI BackgroundTasks.
        """
        try:
            # 1. Validate dataset (re-run here to be safe, though endpoint already ran it to get summary)
            summary = self.validate_dataset(project_name)
            
            logger.info("====================================")
            logger.info("DATASET VALIDATION PASSED")
            logger.info(f"Project: {summary.project}")
            logger.info(f"Train images: {summary.train_images}")
            logger.info(f"Val images: {summary.val_images}")
            # Support non-contiguous classes
            all_classes = summary.classes_detected
            max_class_id = max(all_classes) if all_classes else 0

            # 2. Generate YAML configuration
            yaml_path = self._generate_yaml(project_name, max_class_id, class_names)

            # 3. Train using YOLO engine
            # yolov8n-seg.pt -> nanoseg model
            logger.info(f"Starting YOLO subset-segmentation training for project: {project_name}")
            self._update_state(
                project_name, 
                status="training", 
                total_epochs=epochs, 
                current_epoch=0, 
                progress=0.0,
                message="Initializing YOLO engine..."
            )
            
            model = YOLO("yolov8n-seg.pt")

            # Define callbacks for status tracking
            def on_train_epoch_end(trainer):
                current = trainer.epoch + 1
                progress = round((current / epochs) * 100, 2)
                self._update_state(
                    project_name,
                    current_epoch=current,
                    progress=progress,
                    message=f"Training epoch {current}/{epochs}"
                )

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # Use absolute path for project to prevent YOLO from prepending 'runs/segment'
            abs_models_dir = self.models_dir.absolute()

            model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=str(abs_models_dir),
                name=project_name,
                exist_ok=True # Enable resuming/overwriting
            )

            # Construct the returned path relative to the current working directory
            # or as the user expects it (usually relative to PROJECT_ROOT)
            best_weights = self.models_dir / project_name / "weights" / "best.pt"
            logger.info(f"Training completed successfully! Best weights saved to: {best_weights}")
            
            self._update_state(
                project_name,
                status="completed",
                progress=100.0,
                message="Training finished successfully."
            )
            return str(best_weights)

        except Exception as e:
            logger.exception("Training failed due to unexpected error.")
            self._update_state(
                project_name,
                status="failed",
                message=f"Error: {str(e)}"
            )
            raise e
