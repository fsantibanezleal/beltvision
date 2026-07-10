"""Format helpers for ingestion.

Kept dependency-light: OpenCV and NumPy only (both are in the slim classical stack),
so importing ``beltvision.io`` never drags a heavy engine into the import path.
"""
from __future__ import annotations

import numpy as np

from .schema import IMAGE_FORMATS, VIDEO_FORMATS


def normalize_ext(name_or_ext: str | None) -> str | None:
    """Return a lower-case extension without a leading dot, or None."""
    if not name_or_ext:
        return None
    ext = name_or_ext.rsplit(".", 1)[-1].lower().strip()
    return ext or None


def is_image_format(ext: str | None) -> bool:
    return normalize_ext(ext) in IMAGE_FORMATS


def is_video_format(ext: str | None) -> bool:
    return normalize_ext(ext) in VIDEO_FORMATS


def decode_image(raw: bytes) -> np.ndarray:
    """Decode image bytes into a BGR uint8 array.

    Raises ``ValueError`` if the bytes are not a decodable image (Contract 1 rejects
    corrupt input rather than silently coercing it).
    """
    import cv2  # local import keeps module import cheap and runtime-boundary clean

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def contrast_std(img_bgr: np.ndarray) -> float:
    """Standard deviation of the LAB L-channel: the dust/haze severity proxy.

    Low values indicate a washed-out, hazy frame (the COLA 34 dust regime).
    """
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].std())
