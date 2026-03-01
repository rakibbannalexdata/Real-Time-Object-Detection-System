"""
Image utility helpers for pre- and post-processing.

Designed to be pure functions (no side effects) so they are easy to
test in isolation.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def decode_image(data: bytes) -> np.ndarray:
    """
    Decode raw image bytes into an OpenCV BGR ndarray.

    Parameters
    ----------
    data : bytes
        Raw bytes of an image file (JPEG, PNG, BMP, WEBP, …).

    Returns
    -------
    np.ndarray
        BGR image array of shape (H, W, 3).

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as an image.
    """
    if not data:
        raise ValueError("Image data is empty.")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(
            "Failed to decode image. The file may be corrupted or in an "
            "unsupported format."
        )

    logger.debug(
        "Decoded image: shape=%s dtype=%s", img.shape, img.dtype
    )
    return img


def resize_if_large(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Proportionally resize *image* if its largest dimension exceeds *max_size*.

    This prevents out-of-memory errors on very high-resolution inputs while
    preserving the aspect ratio for accurate detection.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    max_size : int
        Maximum allowed dimension (width or height).

    Returns
    -------
    np.ndarray
        Possibly resized BGR image.
    """
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest <= max_size:
        return image

    scale = max_size / largest
    new_w = int(w * scale)
    new_h = int(h * scale)

    logger.debug(
        "Resizing image from (%d×%d) to (%d×%d) [scale=%.3f]",
        w, h, new_w, new_h, scale,
    )
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def base64_to_image(b64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string into an OpenCV BGR ndarray.

    Accepts both raw base64 and data-URI strings
    (e.g. ``data:image/jpeg;base64,<data>``).

    Parameters
    ----------
    b64_string : str
        Base64-encoded image data.

    Returns
    -------
    np.ndarray
        BGR image array.

    Raises
    ------
    ValueError
        If the string is not valid base64 or cannot be decoded as an image.
    """
    # Strip optional data-URI prefix: "data:image/jpeg;base64,<data>"
    b64_string = re.sub(r"^data:image/[^;]+;base64,", "", b64_string.strip())

    try:
        image_bytes = base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError(f"Invalid base64 string: {exc}") from exc

    return decode_image(image_bytes)


def image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """
    Encode an OpenCV BGR image to a base64 string.

    Useful for streaming annotated frames in the base64 endpoint.

    Parameters
    ----------
    image : np.ndarray
        BGR image to encode.
    ext : str
        Target encoding extension ('.jpg', '.png', …).

    Returns
    -------
    str
        Base64-encoded image string (no data-URI prefix).
    """
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image to '{ext}'.")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """
    Return (width, height) tuple from an OpenCV image.

    Parameters
    ----------
    image : np.ndarray
        BGR image array.

    Returns
    -------
    Tuple[int, int]
        (width, height)
    """
    h, w = image.shape[:2]
    return w, h
