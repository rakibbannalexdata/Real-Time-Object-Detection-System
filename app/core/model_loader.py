"""
YOLOv8 model loader — singleton pattern.

The model is loaded **once** at application startup and reused for every
inference request, avoiding repeated I/O and GPU initialisation overhead.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Thread-safe singleton wrapper around an Ultralytics YOLO model.

    Usage
    -----
    loader = ModelLoader.get_instance(model_path="yolov8n.pt")
    results = loader.model(image)
    """

    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: str) -> None:
        # Prevent direct instantiation — use get_instance()
        if ModelLoader._instance is not None:
            raise RuntimeError(
                "ModelLoader is a singleton. Use ModelLoader.get_instance()."
            )

        self._model_path = model_path
        self._model = None
        self._device: str = self._detect_device()
        self._load_model()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, model_path: str = "yolov8n.pt") -> "ModelLoader":
        """
        Return the existing singleton, or create it on first call.
        Thread-safe via a double-checked lock.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info(
                        "Initialising ModelLoader — path=%s", model_path
                    )
                    instance = object.__new__(cls)
                    instance._model_path = model_path
                    instance._model = None
                    instance._device = cls._detect_device()
                    instance._load_model()
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Destroy the singleton (useful for testing or reloading models).
        """
        with cls._lock:
            cls._instance = None

    @property
    def model(self):
        """The loaded YOLO model object."""
        return self._model

    @property
    def device(self) -> str:
        """The device the model is running on ('cuda' | 'mps' | 'cpu')."""
        return self._device

    @property
    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the YOLO model from *model_path*.
        The model file is downloaded automatically by Ultralytics if it is
        not present locally (requires internet access on first run).
        """
        from ultralytics import YOLO  # deferred import keeps startup fast

        model_file = Path(self._model_path)

        # If the path is absolute/relative and does NOT exist, fall back to
        # letting Ultralytics download it from the official registry.
        if model_file.is_absolute() and not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self._model_path}'. "
                "Provide a valid path or use a standard model name such as "
                "'yolov8n.pt' so Ultralytics can download it automatically."
            )

        try:
            logger.info(
                "Loading YOLO model '%s' on device '%s' …",
                self._model_path,
                self._device,
            )
            self._model = YOLO(self._model_path)
            # Move model to the selected device
            self._model.to(self._device)
            logger.info("YOLO model loaded successfully ✔")
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", exc)
            raise RuntimeError(
                f"Could not load YOLO model '{self._model_path}': {exc}"
            ) from exc

    @staticmethod
    def _detect_device() -> str:
        """
        Auto-select the best available compute device:
        CUDA GPU  →  Apple MPS  →  CPU
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(
                "CUDA available — using GPU: %s",
                torch.cuda.get_device_name(0),
            )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple MPS available — using MPS device.")
        else:
            device = "cpu"
            logger.info("No GPU detected — running on CPU.")
        return device
