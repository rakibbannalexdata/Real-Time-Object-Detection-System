"""
Detection service — core business logic layer.

This module is responsible for running YOLOv8 inference on images and
videos. It is intentionally decoupled from HTTP concerns (FastAPI) so it
can be tested independently or reused in other contexts (CLI, background
workers, etc.).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

import cv2
import numpy as np

from app.core.model_loader import ModelLoader
from app.schemas.detection_schema import (
    Detection,
    FrameDetection,
    ImageDetectionResponse,
    VideoDetectionResponse,
)
from app.utils.image_utils import resize_if_large

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Provides image and video detection capabilities using a pre-loaded
    YOLO model.

    Designed to be used as a FastAPI dependency (injected via Depends).
    """

    def __init__(self, model_loader: ModelLoader) -> None:
        self._loader = model_loader

    # ------------------------------------------------------------------
    # Image detection
    # ------------------------------------------------------------------

    def detect_image(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float = 0.45,
        max_size: int = 1280,
        model_path: Optional[str] = None,
    ) -> tuple[list[Detection], float]:
        """
        Run YOLO detection on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image array.
        confidence_threshold : float
            Minimum confidence score to include a detection.
        iou_threshold : float
            IoU threshold for non-maximum suppression.
        max_size : int
            Resize the largest dimension to this value before inference.

        Returns
        -------
        tuple[list[Detection], float]
            (detections, inference_time_ms)
        """
        if not self._loader.is_loaded:
            raise RuntimeError("Model is not loaded. Cannot perform detection.")

        # Resize to avoid OOM on very large images
        image = resize_if_large(image, max_size)

        logger.debug(
            "Running image inference — size=%s conf=%.2f iou=%.2f",
            image.shape[:2],
            confidence_threshold,
            iou_threshold,
        )

        t_start = time.perf_counter()

        # Get model from path if provided, otherwise default
        model = self._loader.get_model(model_path) if model_path else self._loader.model

        results = model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t_start) * 1000

        detections = self._parse_results(results)

        logger.info(
            "Image detection complete — %d objects found in %.1f ms",
            len(detections),
            inference_ms,
        )

        return detections, inference_ms

    # ------------------------------------------------------------------
    # Video detection
    # ------------------------------------------------------------------

    def detect_video(
        self,
        video_path: str,
        confidence_threshold: float,
        iou_threshold: float = 0.45,
        frame_skip: int = 1,
        max_frames: int = 500,
        max_size: int = 1280,
        model_path: Optional[str] = None,
    ) -> VideoDetectionResponse:
        """
        Run YOLO detection frame-by-frame on a video file.

        Resources (cv2.VideoCapture) are always released, even if an
        exception occurs during processing.

        Parameters
        ----------
        video_path : str
            Absolute path to the video file.
        confidence_threshold : float
            Minimum detection confidence.
        iou_threshold : float
            NMS IoU threshold.
        frame_skip : int
            Process every N-th frame (1 = all frames).
        max_frames : int
            Hard cap on how many frames to process.
        max_size : int
            Maximum frame dimension before resizing.

        Returns
        -------
        VideoDetectionResponse
        """
        if not self._loader.is_loaded:
            raise RuntimeError("Model is not loaded. Cannot perform detection.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: '{video_path}'")

        total_time_start = time.perf_counter()

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                "Processing video — frames=%d fps=%.1f size=%dx%d "
                "skip=%d max_frames=%d",
                total_frames, fps, vid_width, vid_height, frame_skip, max_frames,
            )

            frame_detections: list[FrameDetection] = []
            frame_index = 0
            processed_count = 0

            # Get model from path if provided
            model = self._loader.get_model(model_path) if model_path else self._loader.model

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply frame skip — only process every N-th frame
                if frame_index % frame_skip == 0 and processed_count < max_frames:
                    frame_resized = resize_if_large(frame, max_size)

                    results = model.predict(
                        source=frame_resized,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )

                    detections = self._parse_results(results)
                    timestamp_ms = (frame_index / fps) * 1000

                    frame_detections.append(
                        FrameDetection(
                            frame_index=frame_index,
                            timestamp_ms=round(timestamp_ms, 2),
                            detections=detections,
                            total_detections=len(detections),
                        )
                    )
                    processed_count += 1

                frame_index += 1

            total_ms = (time.perf_counter() - total_time_start) * 1000

            logger.info(
                "Video processing complete — %d/%d frames processed in %.1f ms",
                processed_count,
                total_frames,
                total_ms,
            )

            return VideoDetectionResponse(
                total_frames=total_frames,
                processed_frames=processed_count,
                frame_detections=frame_detections,
                confidence_threshold=confidence_threshold,
                processing_time_ms=round(total_ms, 2),
                video_fps=fps,
                video_width=vid_width,
                video_height=vid_height,
            )

        finally:
            # Always release the VideoCapture to prevent resource leaks
            cap.release()
            logger.debug("VideoCapture released for '%s'.", video_path)

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_results(self, results) -> list[Detection]:
        """
        Convert raw Ultralytics Results objects into Detection schema instances.

        Parameters
        ----------
        results : list[ultralytics.engine.results.Results]
            Raw YOLO prediction output.

        Returns
        -------
        list[Detection]
        """
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            names: dict[int, str] = result.names  # {class_id: class_name}

            boxes = result.boxes
            for i in range(len(boxes)):
                # xyxyn format: [x1, y1, x2, y2] normalized
                xyxyn = boxes.xyxyn[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = names.get(cls_id, str(cls_id))

                detections.append(
                    Detection(
                        **{
                            "class": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(v, 6) for v in xyxyn],
                            "class_id": cls_id,
                            "segmentation": self._extract_mask(result, i)
                        }
                    )
                )

        # Sort by confidence descending for consistent output
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _extract_mask(result, index: int) -> Optional[list[list[float]]]:
        """
        Extract normalized segmentation mask for a specific detection index.
        Uses pixel-level coordinates normalized by image size for maximum precision.
        """
        if not hasattr(result, "masks") or result.masks is None:
            return None
        
        try:
            # result.masks.xy returns coordinates in pixels [x, y, x, y, ...]
            masks_xy = result.masks.xy
            if index < len(masks_xy):
                mask_pixels = masks_xy[index]
                if len(mask_pixels) == 0:
                    return None
                
                # Manual normalization for higher precision
                h, w = result.orig_shape
                mask_normalized = []
                for pt in mask_pixels:
                    mask_normalized.append([round(float(pt[0]) / w, 6), round(float(pt[1]) / h, 6)])
                
                return mask_normalized
        except Exception as exc:
            logger.warning("Failed to extract mask for detection %d: %s", index, exc)
            
        return None
