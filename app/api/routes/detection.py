"""
API routes for object detection endpoints.

Endpoints
---------
POST /detect/image   — detect objects in an uploaded image
POST /detect/video   — detect objects in an uploaded video (frame-by-frame)
POST /detect/base64  — detect objects from a base64-encoded image
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Annotated, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.model_loader import ModelLoader
from app.schemas.detection_schema import (
    Base64DetectionRequest,
    ErrorResponse,
    FrameDetection,
    ImageDetectionResponse,
    VideoDetectionResponse,
)
from app.services.detection_service import DetectionService
from app.utils.image_utils import base64_to_image, decode_image, get_image_dimensions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["Detection"])


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_model_loader() -> ModelLoader:
    """FastAPI dependency: return the singleton ModelLoader."""
    return ModelLoader.get_instance()


def get_detection_service(
    loader: Annotated[ModelLoader, Depends(get_model_loader)],
) -> DetectionService:
    """FastAPI dependency: build DetectionService with the model loader."""
    return DetectionService(loader)


def validate_image_file(
    file: UploadFile,
    settings: Settings,
) -> None:
    """Raise HTTP 400 if the uploaded file is not an allowed image type."""
    content_type = file.content_type or ""
    if content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image type '{content_type}'. "
                f"Allowed types: {settings.allowed_image_types}"
            ),
        )


def validate_video_file(
    file: UploadFile,
    settings: Settings,
) -> None:
    """Raise HTTP 400 if the uploaded file is not an allowed video type."""
    content_type = file.content_type or ""
    if content_type not in settings.allowed_video_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported video type '{content_type}'. "
                f"Allowed types: {settings.allowed_video_types}"
            ),
        )


# ---------------------------------------------------------------------------
# Image detection endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/image",
    response_model=ImageDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or corrupt image"},
        422: {"model": ErrorResponse, "description": "Image decode failure"},
        500: {"model": ErrorResponse, "description": "Model inference failure"},
    },
    summary="Detect objects in an uploaded image",
    description=(
        "Upload a JPEG, PNG, BMP, WEBP, or TIFF image and receive a list of "
        "detected objects with class names, confidence scores, and bounding boxes."
    ),
)
async def detect_image(
    file: Annotated[UploadFile, File(description="Image file to analyse")],
    confidence_threshold: Annotated[
        Optional[float],
        Query(ge=0.0, le=1.0, description="Minimum confidence score (overrides default)"),
    ] = None,
    model_path: Annotated[
        Optional[str],
        Query(description="Path to a specific YOLO model (.pt) to use for this request"),
    ] = None,
    settings: Settings = Depends(get_settings),
    service: DetectionService = Depends(get_detection_service),
) -> ImageDetectionResponse:
    """Detect objects in a single uploaded image."""

    # --- Validate file type ---
    validate_image_file(file, settings)
    # --- Read file bytes ---
    try:
        image_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        ) from exc

    # --- Decode image ---
    try:
        image = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    width, height = get_image_dimensions(image)
    conf = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold

    # --- Run inference ---
    try:
        detections, inference_ms = service.detect_image(
            image=image,
            confidence_threshold=conf,
            iou_threshold=settings.iou_threshold,
            max_size=settings.max_image_size,
            model_path=model_path,
        )
    except RuntimeError as exc:
        logger.exception("Model inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference error: {exc}",
        ) from exc

    return ImageDetectionResponse(
        detections=detections,
        total_detections=len(detections),
        image_width=width,
        image_height=height,
        confidence_threshold=conf,
        inference_time_ms=round(inference_ms, 2),
        model_device=ModelLoader.get_instance().device,
    )


# ---------------------------------------------------------------------------
# Video detection endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/video",
    response_model=VideoDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Video processing failure"},
    },
    summary="Detect objects in an uploaded video",
    description=(
        "Upload a video file (MP4, AVI, MOV, MKV, WEBM) and receive per-frame "
        "detection results. Large videos are automatically frame-skipped to "
        "prevent memory exhaustion."
    ),
)
async def detect_video(
    file: Annotated[UploadFile, File(description="Video file to analyse")],
    confidence_threshold: Annotated[
        Optional[float],
        Query(ge=0.0, le=1.0, description="Minimum confidence score (overrides default)"),
    ] = None,
    model_path: Annotated[
        Optional[str],
        Query(description="Path to a specific YOLO model (.pt) to use for this request"),
    ] = None,
    settings: Settings = Depends(get_settings),
    service: DetectionService = Depends(get_detection_service),
) -> VideoDetectionResponse:
    """Detect objects frame-by-frame in an uploaded video."""

    # --- Validate file type ---
    validate_video_file(file, settings)

    conf = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold

    # --- Save to temp file (OpenCV requires a file path) ---
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        # Async write to the temp file in chunks to avoid blocking the event loop
        async with aiofiles.open(tmp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                await f.write(chunk)

        logger.info("Video saved to temp file: %s", tmp_path)

        # --- Run inference ---
        try:
            result = service.detect_video(
                video_path=tmp_path,
                confidence_threshold=conf,
                iou_threshold=settings.iou_threshold,
                frame_skip=settings.video_frame_skip,
                max_frames=settings.max_video_frames,
                max_size=settings.max_image_size,
                model_path=model_path,
            )
        except (ValueError, RuntimeError) as exc:
            logger.exception("Video detection failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Video processing error: {exc}",
            ) from exc

    finally:
        # Clean up the temp file regardless of success or failure
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Removed temp file: %s", tmp_path)

    return result


# ---------------------------------------------------------------------------
# Base64 image detection endpoint (bonus)
# ---------------------------------------------------------------------------


@router.post(
    "/base64",
    response_model=ImageDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid base64 data"},
        422: {"model": ErrorResponse, "description": "Image decode failure"},
        500: {"model": ErrorResponse, "description": "Model inference failure"},
    },
    summary="Detect objects from a base64-encoded image",
    description=(
        "Send a JSON body containing a base64-encoded image "
        "(with or without a data-URI prefix) and receive detection results."
    ),
)
async def detect_base64(
    payload: Base64DetectionRequest,
    model_path: Annotated[
        Optional[str],
        Query(description="Path to a specific YOLO model (.pt) to use for this request"),
    ] = None,
    settings: Settings = Depends(get_settings),
    service: DetectionService = Depends(get_detection_service),
) -> ImageDetectionResponse:
    """Detect objects in an image provided as a base64 string."""

    # --- Decode base64 → image ---
    try:
        image = base64_to_image(payload.base64_image)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    width, height = get_image_dimensions(image)
    conf = (
        payload.confidence_threshold
        if payload.confidence_threshold is not None
        else settings.confidence_threshold
    )

    # --- Run inference ---
    try:
        detections, inference_ms = service.detect_image(
            image=image,
            confidence_threshold=conf,
            iou_threshold=settings.iou_threshold,
            max_size=settings.max_image_size,
            model_path=model_path,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference error: {exc}",
        ) from exc

    return ImageDetectionResponse(
        detections=detections,
        total_detections=len(detections),
        image_width=width,
        image_height=height,
        confidence_threshold=conf,
        inference_time_ms=round(inference_ms, 2),
        model_device=ModelLoader.get_instance().device,
    )
