"""
Application configuration using Pydantic Settings.
All values are configurable via environment variables or a .env file.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Object Detection System.
    Values can be overridden by environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "Real-Time Object Detection System"
    app_version: str = "1.0.0"
    debug: bool = False

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["*"]

    # Rate limiting (requests per minute per IP)
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    # -------------------------------------------------------------------------
    # YOLO Model
    # -------------------------------------------------------------------------
    model_path: str = "yolov8n.pt"
    # Default confidence threshold (0–1). Can be overridden per-request.
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    # NMS IoU threshold
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # Image processing
    # -------------------------------------------------------------------------
    # Resize any dimension above this before inference to limit memory usage
    max_image_size: int = Field(default=1280, ge=320)

    # -------------------------------------------------------------------------
    # Video processing
    # -------------------------------------------------------------------------
    # Process every N-th frame (1 = all frames, 2 = every other frame, …)
    video_frame_skip: int = Field(default=1, ge=1)
    # Maximum frames to process per video upload (safety cap)
    max_video_frames: int = Field(default=500, ge=1)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # -------------------------------------------------------------------------
    # Supported file types
    # -------------------------------------------------------------------------
    allowed_image_types: list[str] = [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
        "image/tiff",
    ]
    allowed_video_types: list[str] = [
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    ]

    @field_validator("confidence_threshold", "iou_threshold", mode="before")
    @classmethod
    def clamp_float(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Use as a FastAPI dependency: `settings: Settings = Depends(get_settings)`.
    """
    return Settings()
