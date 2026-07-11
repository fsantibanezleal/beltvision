"""Pure geometric measurement (the ImageJ "Analyze" model, classical).

Small, exact, image-free primitives that close the measurement loop of the guided
pipeline: the angle between two sketched lines, a segment length, a polygon area, an
object count + density over a mask, and a pixel<->millimetre scale calibrated from a
known-length line. Everything here is deterministic and unit-tested against exact numbers
on known inputs; there is no drawing and no learned model.

A ``line`` / ``seg`` is accepted either as a flat ``[x1, y1, x2, y2]`` or as a pair of
points ``[[x1, y1], [x2, y2]]``. A ``polygon`` is an ``(N, 2)`` list/array of vertices.

Reference: ImageJ Analyze menu (length/angle/area + set-scale calibration),
https://imagej.net/ij/docs/menus/analyze.html.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_seg(line: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a line into two 2-D endpoints ``(p0, p1)`` as float arrays."""
    arr = np.asarray(line, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"a line needs 4 numbers (x1,y1,x2,y2) or 2 points, got {arr.size}")
    return arr[:2].copy(), arr[2:].copy()


def _direction(line: Any) -> np.ndarray:
    p0, p1 = _as_seg(line)
    d = p1 - p0
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-12:
        raise ValueError("degenerate line: the two endpoints coincide")
    return d / n


def angle_between(line1: Any, line2: Any) -> float:
    """Angle between the direction vectors of two segments, in degrees within ``[0, 180]``.

    Two segments drawn from a shared vertex give the angle at that vertex (the ImageJ
    angle tool). Collinear same-direction segments -> ``0``; perpendicular -> ``90``;
    opposite directions -> ``180``.
    """
    d1, d2 = _direction(line1), _direction(line2)
    cos = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    return round(float(np.degrees(np.arccos(cos))), 6)


def line_angle(line: Any) -> float:
    """Absolute orientation of a segment from the x-axis, in degrees within ``[0, 180)``."""
    d = _direction(line)
    return round(float(np.degrees(np.arctan2(d[1], d[0])) % 180.0), 6)


def segment_length(seg: Any) -> float:
    """Euclidean length of a segment, in pixels."""
    p0, p1 = _as_seg(seg)
    return round(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])), 6)


def polygon_area(points: Any) -> float:
    """Area of a simple polygon (shoelace formula), in square pixels (always non-negative)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"polygon needs an (N>=3, 2) point array, got shape {pts.shape}")
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return round(area, 6)


def polygon_perimeter(points: Any) -> float:
    """Perimeter of a closed polygon, in pixels."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError(f"polygon needs an (N>=2, 2) point array, got shape {pts.shape}")
    d = pts - np.roll(pts, -1, axis=0)
    return round(float(np.sum(np.hypot(d[:, 0], d[:, 1]))), 6)


def count_objects(
    mask: Any, area_range: tuple[float, float] | None = None
) -> dict[str, Any]:
    """Count connected components in a boolean/uint8 ``mask`` (4/8-connectivity via OpenCV).

    ``area_range`` is an inclusive ``(min_px, max_px)`` filter on component area; ``None``
    keeps every component. Returns ``{count, mean_area, total_area, areas}`` where ``areas``
    is the per-object pixel area list (background label 0 excluded).
    """
    import cv2

    m = np.ascontiguousarray(np.asarray(mask) > 0).astype(np.uint8)
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas: list[float] = []
    for label in range(1, n_labels):  # 0 is background
        a = float(stats[label, cv2.CC_STAT_AREA])
        if area_range is not None and not (area_range[0] <= a <= area_range[1]):
            continue
        areas.append(a)
    total = float(sum(areas))
    mean = float(total / len(areas)) if areas else 0.0
    return {
        "count": int(len(areas)),
        "mean_area": round(mean, 6),
        "total_area": round(total, 6),
        "areas": [round(a, 6) for a in areas],
    }


def density(count: int, roi_area_px: float) -> float:
    """Objects per pixel of ROI area (``count / roi_area_px``). ``0`` for a zero-area ROI."""
    if roi_area_px <= 0:
        return 0.0
    return round(float(count) / float(roi_area_px), 9)


def calibrate_scale(known_len_px: float, known_len_mm: float) -> float:
    """Pixels-per-millimetre from a drawn line of known real length (ImageJ set-scale)."""
    if known_len_mm <= 0:
        raise ValueError("known_len_mm must be positive")
    return round(float(known_len_px) / float(known_len_mm), 9)


def px_to_mm(px: float, px_per_mm: float) -> float:
    """Convert a pixel length to millimetres given a calibration."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    return round(float(px) / float(px_per_mm), 6)


def mm_to_px(mm: float, px_per_mm: float) -> float:
    """Convert a millimetre length to pixels given a calibration."""
    return round(float(mm) * float(px_per_mm), 6)


def px2_to_mm2(px2: float, px_per_mm: float) -> float:
    """Convert a pixel area to square millimetres given a calibration."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    return round(float(px2) / (float(px_per_mm) ** 2), 6)


__all__ = [
    "angle_between",
    "line_angle",
    "segment_length",
    "polygon_area",
    "polygon_perimeter",
    "count_objects",
    "density",
    "calibrate_scale",
    "px_to_mm",
    "mm_to_px",
    "px2_to_mm2",
]
