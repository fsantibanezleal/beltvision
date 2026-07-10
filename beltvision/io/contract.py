"""Contract 1: ingestion (raw -> pipeline).

This is the bring-your-own-data gate. Bad data is REJECTED, not silently coerced;
borderline data is FLAGGED so downstream confidences can be honestly down-weighted;
clean data is ACCEPTED. The policy is explicit and lives in one place so the web
app, the pipeline, and ``data/README.md`` cannot disagree about what "valid input"
means.

Outlier policy
--------------
- REJECT: undecodable/corrupt, wrong channel count, out-of-range resolution, or a
  payload larger than the media size cap.
- FLAG:   decodable and in range, but low-contrast (dusty/hazy) below the contrast
  floor. The frame is still processed; the flag rides along in ``measured``.
- ACCEPT: decodable, in range, adequate contrast.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .formats import contrast_std, is_image_format, is_video_format, normalize_ext
from .schema import (
    ACCEPT,
    CHANNELS,
    FLAG,
    MAX_IMAGE_BYTES,
    MAX_SIDE_PX,
    MAX_VIDEO_BYTES,
    MIN_CONTRAST_STD,
    MIN_SIDE_PX,
    REJECT,
    IngestionParams,
    IngestionResult,
)


def validate_format(params: IngestionParams) -> list[str]:
    """Return a list of format-level rejection reasons (empty means the format is ok)."""
    reasons: list[str] = []
    ext = normalize_ext(params.declared_format)
    if ext is None:
        return reasons  # a missing declared format is allowed; the pixels are checked
    if params.media_type == "video":
        if not is_video_format(ext):
            reasons.append(f"unsupported video format: {ext}")
    elif not is_image_format(ext):
        reasons.append(f"unsupported image format: {ext}")
    return reasons


def validate_image(img_bgr: np.ndarray, params: IngestionParams | None = None) -> IngestionResult:
    """Validate a decoded BGR frame against Contract 1.

    Parameters
    ----------
    img_bgr:
        Decoded image as an (H, W, 3) uint8 BGR array.
    params:
        Optional declared ingestion parameters.
    """
    params = params or IngestionParams()
    reasons: list[str] = []
    measured: dict[str, Any] = {}

    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return IngestionResult(REJECT, ["not a decodable image array"], measured)
    if img_bgr.ndim != 3 or img_bgr.shape[2] != CHANNELS:
        return IngestionResult(
            REJECT, [f"expected 3-channel image, got shape {tuple(img_bgr.shape)}"], measured
        )

    h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    short_side, long_side = min(h, w), max(h, w)
    measured.update({"height": h, "width": w, "dtype": str(img_bgr.dtype)})

    reasons.extend(validate_format(params))

    if img_bgr.dtype != np.uint8:
        reasons.append(f"expected uint8 pixels, got {img_bgr.dtype}")
    if short_side < MIN_SIDE_PX:
        reasons.append(f"short side {short_side}px < min {MIN_SIDE_PX}px")
    if long_side > MAX_SIDE_PX:
        reasons.append(f"long side {long_side}px > max {MAX_SIDE_PX}px")

    if reasons:
        return IngestionResult(REJECT, reasons, measured)

    # In range and decodable: apply the outlier (haze) policy as a non-fatal flag.
    std = contrast_std(img_bgr)
    measured["contrast_std"] = round(std, 3)
    if std < MIN_CONTRAST_STD:
        return IngestionResult(
            FLAG,
            [f"low-contrast/haze: L-std {std:.1f} < {MIN_CONTRAST_STD} (down-weight confidence)"],
            measured,
        )
    return IngestionResult(ACCEPT, [], measured)


def validate_bytes(
    raw: bytes, params: IngestionParams | None = None
) -> IngestionResult:
    """Validate raw uploaded bytes: size cap first, then decode, then pixel checks."""
    params = params or IngestionParams()
    cap = MAX_VIDEO_BYTES if params.media_type == "video" else MAX_IMAGE_BYTES
    if len(raw) > cap:
        return IngestionResult(
            REJECT,
            [f"payload {len(raw)} bytes > cap {cap} bytes for {params.media_type}"],
            {"bytes": len(raw)},
        )
    if params.media_type == "video":
        # Video frame-by-frame decoding happens in the pipeline; here we only bound
        # size and format. A zero-length payload is rejected.
        if not raw:
            return IngestionResult(REJECT, ["empty payload"], {"bytes": 0})
        fmt_reasons = validate_format(params)
        verdict = REJECT if fmt_reasons else ACCEPT
        return IngestionResult(verdict, fmt_reasons, {"bytes": len(raw)})

    from .formats import decode_image

    try:
        img = decode_image(raw)
    except ValueError as exc:
        return IngestionResult(REJECT, [str(exc)], {"bytes": len(raw)})
    result = validate_image(img, params)
    result.measured["bytes"] = len(raw)
    return result
