"""Capability 1 + 6: belt-structure geometry and misalignment metrics (all classical, LIVE).

Six methods, each CLAHE-first:
- ``geometry.hough_edges``   - Canny + HoughLinesP near-vertical edge candidates (M1).
- ``geometry.ransac_edges``  - RANSAC deg-2 polynomial left/right edge fit (M2).
- ``geometry.radon_orientation`` - Radon dominant belt orientation (M3).
- ``geometry.misalignment``  - centreline deviation, width profile, skew, flags (M26).
- ``geometry.kalman_edge``   - per-camera constant-velocity Kalman edge tracker (M27).
- ``geometry.obb``           - cv2.minAreaRect oriented boxes, belt + per region (M28).

References: Canny 1986; Fischler & Bolles 1981 (RANSAC), Comm. ACM 24(6); Radon transform
(skimage ``radon``); Kalman 1960; OpenCV ``minAreaRect``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._common import as_bgr, cap, result, timed
from .preprocess import apply_clahe_lab

_CANNY_LO, _CANNY_HI = 50, 150
_MAX_SEGMENTS = 40
_MAX_POLYLINE = 24


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


# --- M2: RANSAC + deg-2 polynomial edge fit --------------------------------------------
def _ransac_poly(
    ys: np.ndarray, xs: np.ndarray, deg: int, iters: int, thresh: float, rng: np.random.Generator
) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = ys.shape[0]
    if n < deg + 1:
        return None, None
    best_coeffs: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # poorly-conditioned fits are expected and handled
        for _ in range(iters):
            idx = rng.choice(n, deg + 1, replace=False)
            try:
                coeffs = np.polyfit(ys[idx], xs[idx], deg)
            except (np.linalg.LinAlgError, ValueError):
                continue
            res = np.abs(np.polyval(coeffs, ys) - xs)
            inliers = res < thresh
            if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
                best_coeffs, best_inliers = coeffs, inliers
        if best_coeffs is not None and best_inliers is not None and int(best_inliers.sum()) >= deg + 1:
            best_coeffs = np.polyfit(ys[best_inliers], xs[best_inliers], deg)
    return best_coeffs, best_inliers


def _fit_side(
    pts: np.ndarray, deg: int, h: int, rng: np.random.Generator
) -> dict[str, Any] | None:
    if pts.shape[0] < deg + 1:
        return None
    ys, xs = pts[:, 1].astype(np.float64), pts[:, 0].astype(np.float64)
    coeffs, inliers = _ransac_poly(ys, xs, deg=deg, iters=120, thresh=3.0, rng=rng)
    if coeffs is None:
        return None
    sample_y = np.linspace(0, h - 1, _MAX_POLYLINE)
    poly_x = np.polyval(coeffs, sample_y)
    n_in = int(inliers.sum()) if inliers is not None else 0
    return {
        "coeffs": [round(float(c), 6) for c in coeffs],
        "inliers": n_in,
        "inlier_ratio": round(n_in / max(pts.shape[0], 1), 4),
        "polyline": [
            [round(float(x), 2), round(float(y), 2)]
            for x, y in zip(poly_x, sample_y, strict=False)
        ],
    }


def ransac_edges(image: Any, *, degree: int = 2, seed: int = 34, **_: Any) -> dict[str, Any]:
    """Robust deg-2 polynomial fit of the left/right belt edges from Canny edge points."""
    _bgr, gray = _clahe_gray(image)
    h, w = gray.shape
    rng = np.random.default_rng(int(seed))
    with timed() as t:
        edges = _canny(gray)
        ys, xs = np.nonzero(edges)
        pts = np.stack([xs, ys], axis=1)
        centre = w * 0.5
        left = _fit_side(pts[pts[:, 0] < centre], degree, h, rng)
        right = _fit_side(pts[pts[:, 0] >= centre], degree, h, rng)
    payload = {
        "shape": [int(h), int(w)],
        "degree": int(degree),
        "n_edge_points": int(pts.shape[0]),
        "left_edge": left,
        "right_edge": right,
        "both_edges_found": bool(left is not None and right is not None),
    }
    return result(
        "geometry.ransac_edges", "geometry", "classical",
        "Fischler & Bolles 1981 (RANSAC), Comm. ACM 24(6); polynomial edge fit",
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


# --- M26: misalignment metrics ---------------------------------------------------------
def misalignment(
    image: Any, *, degree: int = 2, seed: int = 34, px_per_mm: float | None = None,
    tolerance_frac: float = 0.05, **_: Any,
) -> dict[str, Any]:
    """Centreline deviation, width profile, skew and flags derived from fitted edges."""
    _bgr, gray = _clahe_gray(image)
    h, w = gray.shape
    with timed() as t:
        edges = _canny(gray)
        ys, xs = np.nonzero(edges)
        pts = np.stack([xs, ys], axis=1)
        rng = np.random.default_rng(int(seed))
        centre = w * 0.5
        left = _fit_side(pts[pts[:, 0] < centre], degree, h, rng)
        right = _fit_side(pts[pts[:, 0] >= centre], degree, h, rng)

        metrics: dict[str, Any] = {"both_edges_found": bool(left and right)}
        if left and right:
            sample_y = np.linspace(0, h - 1, _MAX_POLYLINE)
            lx = np.polyval(np.array(left["coeffs"]), sample_y)
            rx = np.polyval(np.array(right["coeffs"]), sample_y)
            width = rx - lx
            centreline = (lx + rx) * 0.5
            dev_px = float(np.mean(centreline) - centre)
            # Skew: angle of each edge over the sampled span (deg from vertical).
            l_ang = float(np.degrees(np.arctan2(lx[-1] - lx[0], h)))
            r_ang = float(np.degrees(np.arctan2(rx[-1] - rx[0], h)))
            metrics.update(
                {
                    "mean_width_px": round(float(np.mean(width)), 2),
                    "width_std_px": round(float(np.std(width)), 2),
                    "width_profile_px": [round(float(v), 2) for v in width],
                    "centreline_deviation_px": round(dev_px, 2),
                    "centreline_deviation_frac": round(dev_px / max(w, 1), 4),
                    "left_edge_angle_deg": round(l_ang, 3),
                    "right_edge_angle_deg": round(r_ang, 3),
                    "edge_skew_deg": round(abs(l_ang - r_ang), 3),
                    "misaligned": bool(abs(dev_px) > tolerance_frac * w),
                    "crooked": bool(np.std(width) > 0.1 * max(np.mean(width), 1.0)),
                }
            )
            if px_per_mm and px_per_mm > 0:
                metrics["centreline_deviation_mm"] = round(dev_px / px_per_mm, 3)
                metrics["mean_width_mm"] = round(float(np.mean(width)) / px_per_mm, 3)
                metrics["calibration"] = "absolute-mm"
            else:
                metrics["calibration"] = "relative-px-only (no px_per_mm; mm not fabricated)"
    payload = {"shape": [int(h), int(w)], **metrics}
    return result(
        "geometry.misalignment", "geometry", "classical",
        "Derived from RANSAC-polynomial edges (Fischler & Bolles 1981)",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True,
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
