import logging
import os
from pathlib import Path

from fastapi import HTTPException, status
from ultralytics import YOLO

from app.schemas.training_schema import DatasetSummaryResponse

logger = logging.getLogger(__name__)

class TrainingService:
    """
    Handles training validation, dataset.yaml generation, and YOLOv8 segmentation model training.
    """

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, datasets_dir: str = "datasets", models_dir: str = "models"):
        self.datasets_dir = Path(datasets_dir)
        self.models_dir = Path(models_dir)

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
        # Verify classes are contiguous from 0 to N-1
        expected_classes = set(range(len(all_classes)))
        if all_classes != expected_classes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Class IDs must be contiguous starting from 0. Found classes: {sorted(all_classes)}"
            )

        valid_detected_classes = sorted(list(all_classes))

        return DatasetSummaryResponse(
            project=project_name,
            train_images=train_count,
            val_images=val_count,
            total_classes=len(valid_detected_classes),
            classes_detected=valid_detected_classes,
            dataset_valid=True
        )

    def _generate_yaml(self, project_name: str, num_classes: int) -> Path:
        """
        Generates the dataset.yaml dynamically for YOLO training.
        """
        project_dir = self.datasets_dir / project_name
        yaml_path = project_dir / "dataset.yaml"

        # Format class names dynamically (e.g., 0: class_0, 1: class_1)
        names_dict = "\n".join([f"  {i}: class_{i}" for i in range(num_classes)])

        # Path must be absolute for YOLO to resolve correctly, or relative to the execution dict.
        # Ultralytics recommends writing absolute paths to avoid confusion.
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

    def start_training_sync(self, project_name: str, epochs: int, imgsz: int, batch: int) -> str:
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
            logger.info(f"Total classes detected: {summary.total_classes}")
            logger.info("====================================")

            # 2. Generate YAML configuration
            yaml_path = self._generate_yaml(project_name, summary.total_classes)

            # 3. Train using YOLO engine
            # yolov8n-seg.pt -> nanoseg model
            logger.info(f"Starting YOLO subset-segmentation training for project: {project_name}")
            model = YOLO("yolov8n-seg.pt")

            model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=str(self.models_dir),
                name=project_name,
                exist_ok=True # Enable resuming/overwriting
            )

            best_weights = self.models_dir / project_name / "weights" / "best.pt"
            logger.info(f"Training completed successfully! Best weights saved to: {best_weights}")
            return str(best_weights)

        except Exception as e:
            logger.exception("Training failed due to unexpected error.")
            raise e
