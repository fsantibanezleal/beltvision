"""Capability 1: low-level belt-structure geometry primitives (all classical, LIVE).

Straight-line and orientation primitives only. There is NO forced parametric edge model
here: the belt EDGES / CENTRELINE / ALIGNMENT are derived from the segmented belt-mask
shape in :mod:`beltvision.methods.beltline` (mask boundary + medial axis), which works at
any orientation and any shape. The old degree-2 polynomial ("parabola") edge fit and the
axis-assuming misalignment method have been removed - a belt edge is never modelled as a
forced curve here.

- ``geometry.hough_edges``       - Canny + HoughLinesP straight-line edge candidates (M1).
- ``geometry.radon_orientation`` - Radon dominant belt orientation (M3).
- ``geometry.kalman_edge``       - per-camera constant-velocity Kalman edge tracker (M27).
- ``geometry.obb``               - cv2.minAreaRect oriented boxes, belt + per region (M28).

References: Canny 1986; Radon transform (skimage ``radon``); Kalman 1960; OpenCV
``minAreaRect``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._common import as_bgr, cap, result, timed
from .preprocess import apply_clahe_lab

_CANNY_LO, _CANNY_HI = 50, 150
_MAX_SEGMENTS = 40


def _clahe_gray(image: Any) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    bgr = apply_clahe_lab(as_bgr(image))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _canny(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), _CANNY_LO, _CANNY_HI)


# --- M1: Canny + HoughLinesP near-vertical edge candidates -----------------------------
def hough_edges(image: Any, *, angle_tol_deg: float = 35.0, **_: Any) -> dict[str, Any]:
    """Near-vertical straight-line edge candidates, split left/right of the frame centre."""
    import cv2

    _bgr, gray = _clahe_gray(image)
    h, w = gray.shape
    with timed() as t:
        edges = _canny(gray)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40, minLineLength=int(0.25 * h), maxLineGap=20
        )
        left: list[list[int]] = []
        right: list[list[int]] = []
        angles: list[float] = []
        if lines is not None:
            for seg in np.asarray(lines).reshape(-1, 4):
                x1, y1, x2, y2 = (int(v) for v in seg)
                dx, dy = x2 - x1, y2 - y1
                ang = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))  # 0 = vertical
                if ang > angle_tol_deg:
                    continue
                angles.append(ang)
                (left if (x1 + x2) * 0.5 < w * 0.5 else right).append([x1, y1, x2, y2])
    payload = {
        "shape": [int(h), int(w)],
        "n_candidates": len(left) + len(right),
        "left_segments": cap(left, _MAX_SEGMENTS),
        "right_segments": cap(right, _MAX_SEGMENTS),
        "mean_angle_from_vertical_deg": round(float(np.mean(angles)), 3) if angles else None,
        "angle_tol_deg": float(angle_tol_deg),
    }
    return result(
        "geometry.hough_edges", "geometry", "classical",
        "Canny 1986; OpenCV HoughLinesP",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True,
    )


# --- M3: Radon dominant orientation ----------------------------------------------------
def radon_orientation(image: Any, *, max_side: int = 128, **_: Any) -> dict[str, Any]:
    """Dominant structural orientation via the Radon transform (noise-robust vs Hough)."""
    import cv2
    from skimage.transform import radon

    _bgr, gray = _clahe_gray(image)
    h, w = gray.shape
    with timed() as t:
        side = min(max_side, min(h, w))
        roi = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float64)
        roi -= roi.mean()
        theta = np.arange(0.0, 180.0, 1.0)
        sino = radon(roi, theta=theta, circle=False)
        col_var = sino.var(axis=0)
        dom_idx = int(np.argmax(col_var))
        dom_proj_angle = float(theta[dom_idx])
        # A projection angle t peaks when lines are perpendicular to it, i.e. structures
        # run at (t - 90) deg from the x-axis; report deviation from a vertical belt.
        structure_angle = (dom_proj_angle - 90.0) % 180.0
        dev_from_vertical = float(min(abs(structure_angle - 90.0), 180.0 - abs(structure_angle - 90.0)))
        strength = float(col_var[dom_idx] / (col_var.mean() + 1e-9))
    payload = {
        "shape": [int(h), int(w)],
        "roi_side": int(side),
        "dominant_projection_angle_deg": round(dom_proj_angle, 3),
        "structure_orientation_deg": round(structure_angle, 3),
        "deviation_from_vertical_deg": round(dev_from_vertical, 3),
        "orientation_strength": round(strength, 4),
    }
    return result(
        "geometry.radon_orientation", "geometry", "classical",
        "Radon transform; skimage.transform.radon",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=False,
    )


# --- M27: per-camera constant-velocity Kalman edge tracker -----------------------------
@dataclass
class _CVKalman:
    """1-D constant-velocity Kalman filter over an edge x-position."""

    pos: float
    vel: float = 0.0
    P: np.ndarray = field(default_factory=lambda: np.diag([16.0, 16.0]).astype(np.float64))
    n_updates: int = 0
    n_predicted: int = 0

    _F = np.array([[1.0, 1.0], [0.0, 1.0]])
    _H = np.array([[1.0, 0.0]])
    _Q = np.diag([0.5, 0.5])
    _R = np.array([[9.0]])

    def step(self, z: float | None) -> None:
        x = np.array([self.pos, self.vel])
        x = self._F @ x
        self.P = self._F @ self.P @ self._F.T + self._Q
        if z is not None and np.isfinite(z):
            y = np.array([float(z)]) - self._H @ x
            s_innov = self._H @ self.P @ self._H.T + self._R
            k = self.P @ self._H.T @ np.linalg.inv(s_innov)
            x = x + (k @ y)
            self.P = (np.eye(2) - k @ self._H) @ self.P
            self.n_updates += 1
        else:
            self.n_predicted += 1
        self.pos, self.vel = float(x[0]), float(x[1])


# Per-camera persistent state: {camera_id: {"left": _CVKalman, "right": _CVKalman}}.
_KALMAN_STATE: dict[str, dict[str, _CVKalman]] = {}


def reset_kalman(camera_id: str | None = None) -> None:
    """Clear the persistent Kalman state (all cameras, or one)."""
    if camera_id is None:
        _KALMAN_STATE.clear()
    else:
        _KALMAN_STATE.pop(camera_id, None)


def _measure_edges(gray: np.ndarray) -> tuple[float | None, float | None]:
    """Measure left/right edge x at mid-height from the Canny edge map (None if occluded)."""
    edges = _canny(gray)
    h, w = gray.shape
    band = edges[int(0.4 * h) : int(0.6 * h) + 1, :]
    col = band.sum(axis=0)
    centre = w // 2
    left_cols = np.nonzero(col[:centre])[0]
    right_cols = np.nonzero(col[centre:])[0]
    left = float(left_cols[np.argmax(col[left_cols])]) if left_cols.size else None
    right = float(centre + right_cols[np.argmax(col[centre:][right_cols])]) if right_cols.size else None
    return left, right


def kalman_edge(image: Any, *, camera_id: str = "default", **_: Any) -> dict[str, Any]:
    """Constant-velocity Kalman smoothing of edge x-position; persists across calls."""
    _bgr, gray = _clahe_gray(image)
    h, w = gray.shape
    with timed() as t:
        left_z, right_z = _measure_edges(gray)
        cam = _KALMAN_STATE.setdefault(camera_id, {})
        if "left" not in cam:
            cam["left"] = _CVKalman(pos=left_z if left_z is not None else 0.18 * w)
            cam["right"] = _CVKalman(pos=right_z if right_z is not None else 0.82 * w)
        cam["left"].step(left_z)
        cam["right"].step(right_z)
    payload = {
        "shape": [int(h), int(w)],
        "camera_id": camera_id,
        "measured": {
            "left_x": round(left_z, 2) if left_z is not None else None,
            "right_x": round(right_z, 2) if right_z is not None else None,
        },
        "smoothed": {
            "left_x": round(cam["left"].pos, 2),
            "right_x": round(cam["right"].pos, 2),
            "left_velocity": round(cam["left"].vel, 4),
            "right_velocity": round(cam["right"].vel, 4),
        },
        "n_updates": int(cam["left"].n_updates),
        "n_predicted_only": int(cam["left"].n_predicted),
        "occluded_this_frame": bool(left_z is None or right_z is None),
    }
    return result(
        "geometry.kalman_edge", "geometry", "classical",
        "Kalman 1960 (constant-velocity edge smoothing / wander trend)",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True,
    )


# --- M28: oriented bounding boxes (belt + per region) ----------------------------------
def _rect_to_dict(rect: Any) -> dict[str, Any]:
    import cv2

    (cx, cy), (rw, rh), ang = rect
    box = cv2.boxPoints(rect)
    return {
        "center": [round(float(cx), 2), round(float(cy), 2)],
        "size": [round(float(rw), 2), round(float(rh), 2)],
        "angle_deg": round(float(ang), 3),
        "box_points": [[round(float(x), 2), round(float(y), 2)] for x, y in box],
    }


def obb(image: Any, *, min_region_area_frac: float = 0.001, max_regions: int = 24, **_: Any) -> dict[str, Any]:
    """Oriented bounding boxes via cv2.minAreaRect: whole belt plus per-region boxes."""
    import cv2

    _bgr, gray = _clahe_gray(image)
    h, w = gray.shape
    with timed() as t:
        _thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg = np.column_stack(np.nonzero(mask)[::-1]).astype(np.int32)  # (x, y)
        belt = _rect_to_dict(cv2.minAreaRect(fg)) if fg.shape[0] >= 3 else None
        min_area = min_region_area_frac * h * w
        regions = sorted(
            (c for c in contours if cv2.contourArea(c) >= min_area),
            key=cv2.contourArea,
            reverse=True,
        )
        region_obbs = [_rect_to_dict(cv2.minAreaRect(c)) for c in regions[:max_regions]]
    payload = {
        "shape": [int(h), int(w)],
        "belt_obb": belt,
        "n_regions": len(region_obbs),
        "region_obbs": region_obbs,
        "foreground_frac": round(float(np.count_nonzero(mask)) / (h * w), 4),
    }
    return result(
        "geometry.obb", "geometry", "classical",
        "OpenCV minAreaRect (oriented bounding box)",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True,
    )
