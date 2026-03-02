from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import ValidationError

from app.schemas.training_schema import DatasetSummaryResponse, TrainingStartRequest, TrainingStartResponse
from app.services.training_service import TrainingService
from app.schemas.detection_schema import ErrorResponse

router = APIRouter(prefix="/training", tags=["Training"])

def get_training_service() -> TrainingService:
    return TrainingService()

@router.get(
    "/dataset/info",
    response_model=DatasetSummaryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid dataset structure or formatting"},
        404: {"model": ErrorResponse, "description": "Project dataset not found"}
    },
    summary="Get dataset validation summary",
    description="Runs strict validation on the dataset format and returns a summary of images and classes detected."
)
def get_dataset_info(
    project: str = Query(..., description="Name of the project folder in datasets/"),
    service: TrainingService = Depends(get_training_service)
) -> DatasetSummaryResponse:
    """
    Returns a summary of the dataset before training, ensuring all labels are 
    valid and matching images exist.
    """
    try:
        summary = service.validate_dataset(project)
        return summary
    except HTTPException as e:
        # Re-raise HTTPExceptions raised during validation
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error validating dataset: {e}"
        ) from e


@router.post(
    "/start",
    response_model=TrainingStartResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Dataset validation failed"},
        404: {"model": ErrorResponse, "description": "Dataset not found"}
    },
    summary="Start YOLOv8 segmentation training",
    description="Validates the dataset format, auto-generates dataset.yaml, and starts training in the background."
)
def start_training(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    service: TrainingService = Depends(get_training_service)
) -> TrainingStartResponse:
    """
    Starts the training pipeline for the given project.
    """
    # 1. Validate BEFORE starting background task
    try:
        service.validate_dataset(request.project_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error validating dataset: {e}"
        ) from e

    # 2. Trigger training in the background
    background_tasks.add_task(
        service.start_training_sync,
        project_name=request.project_name,
        epochs=request.epochs,
        imgsz=request.imgsz,
        batch=request.batch,
        class_names=request.class_names
    )

    best_weights_path = f"models/{request.project_name}/weights/best.pt"

    return TrainingStartResponse(
        status="started",
        project=request.project_name,
        message="Dataset validated successfully. Training started in the background.",
        best_weights_path=best_weights_path
    )
