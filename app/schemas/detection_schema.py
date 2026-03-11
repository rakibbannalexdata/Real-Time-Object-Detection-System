"""
Pydantic response models for the Object Detection API.

These models enforce strict typing on all API responses and are used for
automatic OpenAPI documentation generation.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates [x1, y1, x2, y2]."""

    x1: float = Field(..., description="Left edge of the bounding box")
    y1: float = Field(..., description="Top edge of the bounding box")
    x2: float = Field(..., description="Right edge of the bounding box")
    y2: float = Field(..., description="Bottom edge of the bounding box")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class Detection(BaseModel):
    """A single detected object."""

    class_name: str = Field(..., alias="class", description="Detected object class name")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0–1)",
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in normalized coordinates (0-1)",
    )
    class_id: int = Field(..., description="Numeric class identifier")
    segmentation: Optional[list[list[float]]] = Field(
        default=None,
        description="Optional list of normalized polygon coordinates [x1, y1, x2, y2, ...]",
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Image detection response
# ---------------------------------------------------------------------------


class ImageDetectionResponse(BaseModel):
    """Response schema for POST /detect/image."""

    status: str = Field(default="success")
    detections: list[Detection] = Field(
        default_factory=list,
        description="List of detected objects",
    )
    total_detections: int = Field(..., description="Number of objects detected")
    image_width: int = Field(..., description="Width of the processed image in pixels")
    image_height: int = Field(..., description="Height of the processed image in pixels")
    confidence_threshold: float = Field(
        ..., description="Confidence threshold applied during inference"
    )
    inference_time_ms: float = Field(
        ..., description="Model inference time in milliseconds"
    )
    model_device: str = Field(..., description="Device used for inference (cpu/cuda/mps)")


# ---------------------------------------------------------------------------
# Video detection response
# ---------------------------------------------------------------------------


class FrameDetection(BaseModel):
    """Detection results for a single video frame."""

    frame_index: int = Field(..., description="Zero-based frame index")
    timestamp_ms: float = Field(..., description="Frame timestamp in milliseconds")
    detections: list[Detection] = Field(default_factory=list)
    total_detections: int = Field(..., description="Objects detected in this frame")


class VideoDetectionResponse(BaseModel):
    """Response schema for POST /detect/video."""

    status: str = Field(default="success")
    total_frames: int = Field(..., description="Total frames in the video")
    processed_frames: int = Field(..., description="Frames actually processed (after skip)")
    frame_detections: list[FrameDetection] = Field(
        default_factory=list,
        description="Per-frame detection results",
    )
    confidence_threshold: float = Field(..., description="Confidence threshold applied")
    processing_time_ms: float = Field(
        ..., description="Total video processing time in milliseconds"
    )
    video_fps: float = Field(..., description="Source video frames per second")
    video_width: int = Field(..., description="Source video width in pixels")
    video_height: int = Field(..., description="Source video height in pixels")


# ---------------------------------------------------------------------------
# Base64 detection request/response
# ---------------------------------------------------------------------------


class Base64DetectionRequest(BaseModel):
    """Request schema for POST /detect/base64."""

    base64_image: str = Field(
        ...,
        description="Base64-encoded image string (with or without data-URI prefix)",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the default confidence threshold for this request",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(default="healthy")
    model_loaded: bool
    model_path: str
    model_device: str
    cuda_available: bool
    app_version: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Generic error response body."""

    status: str = Field(default="error")
    detail: str
    error_code: Optional[str] = None
