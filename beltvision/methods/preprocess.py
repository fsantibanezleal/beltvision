"""Capability 0 (cross-cutting, mandatory first stage): preprocessing.

CLAHE-first for the dusty COLA 34 regime, then edge-preserving denoise, then an honest
dust/haze severity score so downstream confidences can be down-weighted. Both the LAB
L-channel CLAHE and the severity score are pure classical OpenCV/NumPy, so this runs
LIVE in the browser (opencv.js/WASM) and on the server.

The other capabilities call :func:`apply_clahe_lab` before their own work, so the
CLAHE-first rule holds uniformly and train/serve use the identical operation.

Reference: CLAHE clip 2.0 / 8x8 on LAB-L is the domain-standard dust remedy
(IET Image Processing DOI 10.1049/iet-ipr.2019.0992; Springer DOI 10.1007/s44163-025-00663-5).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_bgr, result, timed

REFERENCE = "CLAHE clip2.0/8x8 LAB-L; He et al. 2009 dark channel; IET-IPR 10.1049/iet-ipr.2019.0992"


def apply_clahe_lab(bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """LAB L-channel CLAHE (the mandatory first operation). Returns a BGR uint8 image."""
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile))).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def bilateral_denoise(
    bgr: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0
) -> np.ndarray:
    """Edge-preserving denoise applied before edge operations."""
    import cv2

    return cv2.bilateralFilter(bgr, int(d), float(sigma_color), float(sigma_space))


def dark_channel_haze(bgr: np.ndarray, patch: int = 15) -> float:
    """Dark-channel haze estimate in [0, 1]; higher means hazier/dustier.

    The dark channel (per-pixel channel minimum, min-filtered over a patch) is bright in
    hazy regions and dark in clear ones (He, Sun & Tang, CVPR 2009).
    """
    import cv2

    dark = bgr.min(axis=2).astype(np.uint8)
    k = max(3, int(patch) | 1)
    eroded = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return float(np.clip(eroded.mean() / 255.0, 0.0, 1.0))


def global_contrast(bgr: np.ndarray) -> float:
    """Standard deviation of the LAB L-channel: low means washed-out/hazy."""
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].std())


def haze_severity(bgr: np.ndarray) -> dict[str, float]:
    """Fuse the dark-channel and global-contrast cues into a [0, 1] severity score."""
    dark = dark_channel_haze(bgr)
    contrast = global_contrast(bgr)
    # Contrast term: an L-std of ~12 is the ingestion haze floor; normalize around it.
    contrast_term = float(np.clip(1.0 - contrast / 60.0, 0.0, 1.0))
    severity = float(np.clip(0.5 * dark + 0.5 * contrast_term, 0.0, 1.0))
    return {
        "severity": round(severity, 4),
        "dark_channel": round(dark, 4),
        "global_contrast_lstd": round(contrast, 3),
    }


def clahe_lab(
    image: Any,
    *,
    clip: float = 2.0,
    tile: int = 8,
    denoise: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Method ``preprocess.clahe_lab``: CLAHE + bilateral denoise + dust/haze severity."""
    bgr = as_bgr(image)
    with timed() as t:
        pre = apply_clahe_lab(bgr, clip=clip, tile=tile)
        if denoise:
            pre = bilateral_denoise(pre)
        before = haze_severity(bgr)
        after = haze_severity(pre)
    h, w = bgr.shape[:2]
    payload = {
        "shape": [int(h), int(w)],
        "clahe": {"clip_limit": float(clip), "tile": int(tile), "color_space": "LAB-L"},
        "denoise": {"applied": bool(denoise), "method": "bilateral"},
        "haze_before": before,
        "haze_after": after,
        "haze_reduction": round(before["severity"] - after["severity"], 4),
        "flagged_hazy": bool(after["severity"] >= 0.5),
    }
    return result(
        "preprocess.clahe_lab",
        "preprocess",
        "classical",
        REFERENCE,
        payload=payload,
        model_bytes=0,
        infer_ms=t.ms,
        web_drivable=True,
    )
