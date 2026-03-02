from pydantic import BaseModel, Field

class TrainingStartRequest(BaseModel):
    project_name: str = Field(..., description="Name of the dataset project folder inside 'datasets/'")
    epochs: int = Field(default=100, gt=0, description="Number of training epochs")
    imgsz: int = Field(default=640, gt=0, description="Image size for training")
    batch: int = Field(default=16, gt=0, description="Batch size")
    class_names: dict[int, str] = Field(default_factory=dict, description="Optional mapping of class IDs to names")

class DatasetSummaryResponse(BaseModel):
    project: str = Field(..., description="Project name")
    train_images: int = Field(..., description="Total training images found")
    val_images: int = Field(..., description="Total validation images found")
    total_classes: int = Field(..., description="Number of unique classes detected")
    classes_detected: list[int] = Field(..., description="List of unique class IDs detected")
    dataset_valid: bool = Field(..., description="Whether the dataset passed validation")

class TrainingStartResponse(BaseModel):
    status: str = Field(default="started")
    project: str
    message: str
    best_weights_path: str

class TrainingStatusResponse(BaseModel):
    project: str
    status: str = Field(..., description="idle, training, completed, failed")
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    message: str = ""
