"""Orientation- and ROI-constrained line detectors (THE fix).

The defect this repairs: firing Hough / RANSAC on the raw frame makes point-detectors lock
onto noise and "find lines in all directions". The classical discipline is a staged
pipeline - preprocess -> denoise -> edge -> threshold to a BINARY edge map -> detect - and
the detector is CONSTRAINED to a region of interest and an orientation band derived from the
user's view/ROI prior.

This module implements exactly that:

- :func:`preprocess_for_lines` builds the binary edge map (CLAHE -> denoise -> edge ->
  restrict to the ROI mask).
- :func:`gradient_orientation_gate` keeps only edge pixels whose Sobel gradient orientation
  is within a band of the belt normal (the gradient/oriented-Hough clutter reducer).
- :func:`hough_constrained` runs ``skimage.transform.hough_line`` with a THETA VECTOR limited
  to the orientation band, so it is mathematically impossible for it to return a line outside
  the band (no spurious perpendicular lines).
- :func:`ransac_line_constrained` runs RANSAC over ROI edge points and REJECTS any model whose
  angle leaves the band (``is_model_valid``), returning straight in-band lines only.

Angle convention: an AXIS angle is the line's direction from the x-axis in ``[0, 180)``.
scikit-image's Hough ``theta`` is the line NORMAL in ``[-pi/2, pi/2)``; a line with normal
``theta`` has axis ``theta + 90``. All the band bookkeeping is done in axis degrees.

References: Canny 1986; Matas et al. 2000 (progressive probabilistic Hough); gradient-
orientation Hough clutter reduction (arXiv:1510.04863); Fischler & Bolles 1981 (RANSAC);
staged edge->threshold->Hough pipeline (WTTech; ACM line-detection review 2025).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..render import draw_legend, draw_summary, to_png_b64
from ._common import as_bgr, result, timed
from .preprocess import apply_clahe_lab

_CAP = "geometry"
_TIER = "classical"
FAM_CONSTRAINED = "constrained_lines"

_HOUGH_REF = ("Matas et al. 2000 (progressive probabilistic Hough); skimage hough_line + "
              "gradient-orientation band (arXiv:1510.04863); Canny 1986 edge->threshold pipeline")
_RANSAC_REF = ("Fischler & Bolles 1981 (RANSAC); skimage LineModelND with an orientation-band "
               "is_model_valid reject; staged edge->threshold->fit pipeline")


# --- angle helpers ----------------------------------------------------------------------
def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two AXIS angles in ``[0, 180)``."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _theta_vector(center_deg: float, band_deg: float, step_deg: float = 0.5) -> np.ndarray:
    """A scikit-image Hough ``theta`` array (line normals, radians) for an AXIS band.

    Axis in ``[center-band, center+band]`` -> normal = axis - 90, wrapped into
    ``[-90, 90)`` and returned in radians. Restricting theta to this vector is what makes an
    out-of-band line un-representable in the accumulator.
    """
    band = float(max(band_deg, step_deg))
    axes = np.arange(center_deg - band, center_deg + band + step_deg, step_deg)
    normals = (axes - 90.0) % 180.0
    normals = np.where(normals >= 90.0, normals - 180.0, normals)  # -> [-90, 90)
    return np.radians(normals)


def _axis_of_theta(theta_rad: float) -> float:
    """Axis angle (deg, ``[0, 180)``) of a scikit-image Hough normal ``theta`` (radians)."""
    return float((np.degrees(theta_rad) + 90.0) % 180.0)


# --- stage 1: the binary edge map -------------------------------------------------------
def _gray_clahe(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY)


def preprocess_for_lines(
    bgr: np.ndarray,
    roi_mask: np.ndarray | None = None,
    denoise: str = "gaussian",
    edge: str = "canny",
) -> np.ndarray:
    """Build a BINARY edge map: gray -> CLAHE -> denoise -> edge -> restrict to ``roi_mask``.

    ``denoise`` in {"gaussian", "median", "bilateral", "none"}; ``edge`` in
    {"canny", "log", "sobel"}. Returns a boolean ``(H, W)`` edge map.
    """
    import cv2

    gray = _gray_clahe(as_bgr(bgr))
    d = str(denoise).lower()
    if d == "median":
        den = cv2.medianBlur(gray, 5)
    elif d == "bilateral":
        den = cv2.bilateralFilter(gray, 5, 50.0, 50.0)
    elif d in ("none", ""):
        den = gray
    else:  # gaussian (default)
        den = cv2.GaussianBlur(gray, (5, 5), 0)

    e = str(edge).lower()
    if e == "log":
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(den, (0, 0), 1.6), cv2.CV_32F, ksize=3))
        thr = float(np.percentile(lap, 92.0))
        binary = lap >= max(thr, 1e-6)
    elif e in ("sobel", "sobel-threshold", "sobel_threshold"):
        gx = cv2.Sobel(den, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(den, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(np.percentile(mag, 90.0))
        binary = mag >= max(thr, 1e-6)
    else:  # canny (default)
        binary = cv2.Canny(den, 50, 150) > 0

    if roi_mask is not None:
        binary = binary & (np.asarray(roi_mask) > 0)
    return binary


# --- stage 2: gradient-orientation gate -------------------------------------------------
def gradient_orientation_gate(
    binary_edge: np.ndarray, gray: np.ndarray,
    theta_center_deg: float, theta_band_deg: float,
) -> np.ndarray:
    """Keep edge pixels whose Sobel gradient orientation is within the band of the belt normal.

    A true belt edge has its intensity gradient along the belt NORMAL (``axis + 90``). Keeping
    only pixels whose gradient points that way de-clutters the accumulator and removes noise
    edges that would vote for spurious lines.
    """
    import cv2

    g = np.asarray(gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_ori = np.degrees(np.arctan2(gy, gx)) % 180.0
    normal = (float(theta_center_deg) + 90.0) % 180.0
    diff = np.abs(grad_ori - normal) % 180.0
    diff = np.minimum(diff, 180.0 - diff)
    keep = diff <= float(theta_band_deg)
    return (np.asarray(binary_edge) > 0) & keep


# --- overlay plumbing -------------------------------------------------------------------
def _canvas(binary_edge: np.ndarray, bgr: np.ndarray | None) -> np.ndarray:
    import cv2

    h, w = np.asarray(binary_edge).shape[:2]
    if bgr is not None:
        img = as_bgr(bgr)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.addWeighted(img, 0.5, np.zeros_like(img), 0.5, 0)
    edge = (np.asarray(binary_edge) > 0).astype(np.uint8) * 90
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)


def _record(
    method_id: str, name: str, reference: str, *, metric_name: str, metric_value: float,
    overlay: np.ndarray, infer_ms: float, summary: str, family: str = FAM_CONSTRAINED,
    extra: dict[str, Any] | None = None, capability: str = _CAP, web_drivable: bool = True,
) -> dict[str, Any]:
    """Assemble the uniform method record (envelope + payload) and attach the overlay."""
    payload: dict[str, Any] = {
        "name": name, "family": family, "summary": summary,
        "metric_name": metric_name, "metric_value": round(float(metric_value), 5),
    }
    if extra:
        payload.update(extra)
    res = result(method_id, capability, _TIER, reference, payload=payload,
                 model_bytes=0, infer_ms=infer_ms, web_drivable=web_drivable)
    res["overlay_b64"] = to_png_b64(overlay)
    res["id"] = res["method"]
    return res


# --- stage 3a: constrained Hough --------------------------------------------------------
def _segment_from_line(
    theta_rad: float, dist: float, support: np.ndarray, min_len_px: float
) -> tuple[tuple[float, float], tuple[float, float], int] | None:
    """Clip an infinite Hough line to the pixels supported by the edge map; return endpoints.

    Samples the line inside the frame and keeps the span between the first and last edge-
    supported sample (so a real segment, not an image-spanning line). ``None`` if the
    supported span is shorter than ``min_len_px``.
    """
    h, w = support.shape[:2]
    a, b = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    x0, y0 = a * dist, b * dist
    diag = float(np.hypot(h, w))
    t = np.arange(-diag, diag + 1.0)
    xs = np.round(x0 - t * b).astype(np.int64)  # direction along line = (-b, a)
    ys = np.round(y0 + t * a).astype(np.int64)
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    t, xs, ys = t[inb], xs[inb], ys[inb]
    if t.size == 0:
        return None
    hit = np.asarray(support)[ys, xs] > 0
    tv = t[hit]
    if tv.size < 2:
        return None
    t_lo, t_hi = float(tv.min()), float(tv.max())
    if (t_hi - t_lo) < float(min_len_px):
        return None
    p0 = (x0 - t_lo * b, y0 + t_lo * a)
    p1 = (x0 - t_hi * b, y0 + t_hi * a)
    return p0, p1, int(hit.sum())


def hough_constrained(
    binary_edge: np.ndarray,
    theta_center_deg: float,
    theta_band_deg: float,
    min_len_px: float | None = None,
    roi_mask: np.ndarray | None = None,
    bgr: np.ndarray | None = None,
    max_lines: int = 6,
) -> dict[str, Any]:
    """Hough over a binary edge map with a THETA VECTOR limited to the orientation band.

    Returns the uniform method record. Because ``theta`` only spans the band, no line outside
    ``theta_center_deg +/- theta_band_deg`` can appear (the perpendicular-noise fix).
    """
    import cv2
    from skimage.transform import hough_line, hough_line_peaks

    be = np.asarray(binary_edge) > 0
    if roi_mask is not None:
        be = be & (np.asarray(roi_mask) > 0)
    h, w = be.shape[:2]
    min_len = float(min_len_px) if min_len_px is not None else 0.15 * float(min(h, w))
    support = cv2.dilate(be.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    segments: list[dict[str, Any]] = []
    with timed() as t:
        theta = _theta_vector(theta_center_deg, theta_band_deg)
        hspace, angles, dists = hough_line(be, theta=theta)
        if hspace.max() > 0:
            thresh = 0.33 * float(hspace.max())
            peaks = hough_line_peaks(hspace, angles, dists, num_peaks=int(max_lines),
                                     threshold=thresh, min_distance=9, min_angle=6)
            for _accum, ang, dist in zip(*peaks, strict=False):
                seg = _segment_from_line(float(ang), float(dist), support, min_len)
                if seg is None:
                    continue
                p0, p1, sup = seg
                segments.append({
                    "p0": [round(p0[0], 1), round(p0[1], 1)],
                    "p1": [round(p1[0], 1), round(p1[1], 1)],
                    "angle_deg": round(_axis_of_theta(float(ang)), 2),
                    "support_px": int(sup),
                })
    axes = [s["angle_deg"] for s in segments]
    mean_axis = round(float(np.mean(axes)), 2) if axes else None

    img = _canvas(be, bgr)
    for s in segments:
        p0 = (int(round(s["p0"][0])), int(round(s["p0"][1])))
        p1 = (int(round(s["p1"][0])), int(round(s["p1"][1])))
        cv2.line(img, p0, p1, (70, 230, 70), 2, cv2.LINE_AA)
    lo = round(theta_center_deg - theta_band_deg, 1)
    hi = round(theta_center_deg + theta_band_deg, 1)
    draw_legend(img, [((70, 230, 70), f"in-band line [{lo},{hi}]deg")])
    draw_summary(img, f"Constrained Hough: {len(segments)} straight line(s), all within the "
                      f"belt band {theta_center_deg:.0f}+/-{theta_band_deg:.0f}deg "
                      f"(mean axis {mean_axis if mean_axis is not None else 'n/a'}deg). "
                      "Runs on the ROI edge map with a theta vector limited to the band, so "
                      "no perpendicular noise lines can appear.")
    return _record(
        "geometry.hough_constrained", "Constrained Hough lines", _HOUGH_REF,
        metric_name="n_lines", metric_value=float(len(segments)), overlay=img, infer_ms=t.ms,
        summary=f"{len(segments)} in-band straight line(s)",
        extra={"theta_center_deg": round(float(theta_center_deg), 2),
               "theta_band_deg": round(float(theta_band_deg), 2),
               "mean_axis_deg": mean_axis, "angles_deg": axes,
               "min_len_px": round(min_len, 1), "roi_used": bool(roi_mask is not None),
               "segments": segments})


# --- stage 3b: constrained RANSAC -------------------------------------------------------
def _edge_points(points_or_edge: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    """Coerce an ``(N, 2)`` point set or a binary edge image into ``(N, 2)`` [x, y] points."""
    arr = np.asarray(points_or_edge)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.astype(np.float64)
    be = arr > 0
    if roi_mask is not None:
        be = be & (np.asarray(roi_mask) > 0)
    ys, xs = np.nonzero(be)
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)


def _line_model_axis(model: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(origin, direction, axis_deg)`` from a skimage LineModelND, version-robust."""
    origin = getattr(model, "origin", None)
    direction = getattr(model, "direction", None)
    if origin is None or direction is None:
        origin, direction = model.params
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    axis = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    return origin, direction, axis


def ransac_line_constrained(
    points_or_edge: np.ndarray,
    theta_center_deg: float,
    theta_band_deg: float,
    roi_mask: np.ndarray | None = None,
    bgr: np.ndarray | None = None,
    max_lines: int = 3,
    residual_px: float = 2.5,
    min_inliers: int = 25,
) -> dict[str, Any]:
    """RANSAC straight lines over ROI edge points, REJECTING any out-of-band model.

    ``is_model_valid`` vetoes every candidate whose axis leaves the band, so the returned
    lines are guaranteed in-band (no cross-direction junk). Returns the uniform method record.
    """
    import cv2
    from skimage.measure import LineModelND, ransac

    pts_all = _edge_points(points_or_edge, roi_mask)
    edge_shape = np.asarray(points_or_edge).shape[:2] if np.asarray(points_or_edge).ndim == 2 \
        and np.asarray(points_or_edge).shape[1] != 2 else None

    def _valid(model: Any, *_data: Any) -> bool:
        _o, _d, axis = _line_model_axis(model)
        return _ang_diff(axis, theta_center_deg) <= float(theta_band_deg)

    lines: list[dict[str, Any]] = []
    with timed() as t:
        pts = pts_all.copy()
        for _ in range(int(max_lines)):
            if pts.shape[0] < max(min_inliers, 2):
                break
            try:
                model, inliers = ransac(pts, LineModelND, min_samples=2,
                                        residual_threshold=float(residual_px), max_trials=500,
                                        is_model_valid=_valid)
            except Exception:  # noqa: BLE001 - a degenerate cloud just ends the search
                break
            if model is None or inliers is None or int(inliers.sum()) < int(min_inliers):
                break
            origin, direction, axis = _line_model_axis(model)
            inlier_pts = pts[inliers]
            tt = (inlier_pts - origin) @ direction
            p0 = origin + float(tt.min()) * direction
            p1 = origin + float(tt.max()) * direction
            lines.append({
                "p0": [round(float(p0[0]), 1), round(float(p0[1]), 1)],
                "p1": [round(float(p1[0]), 1), round(float(p1[1]), 1)],
                "angle_deg": round(axis, 2),
                "inlier_frac": round(float(inliers.sum()) / float(pts.shape[0]), 3),
                "n_inliers": int(inliers.sum()),
            })
            pts = pts[~inliers]
    mean_frac = round(float(np.mean([ln["inlier_frac"] for ln in lines])), 3) if lines else 0.0
    axes = [ln["angle_deg"] for ln in lines]

    canvas_shape = edge_shape if edge_shape is not None else (
        (int(pts_all[:, 1].max()) + 2, int(pts_all[:, 0].max()) + 2) if pts_all.size else (2, 2))
    base_edge = np.zeros(canvas_shape, dtype=np.uint8)
    img = _canvas(base_edge, bgr)
    for col, ln in zip([(60, 200, 60), (60, 220, 220), (200, 120, 40)], lines, strict=False):
        p0 = (int(round(ln["p0"][0])), int(round(ln["p0"][1])))
        p1 = (int(round(ln["p1"][0])), int(round(ln["p1"][1])))
        cv2.line(img, p0, p1, col, 3, cv2.LINE_AA)
    lo = round(theta_center_deg - theta_band_deg, 1)
    hi = round(theta_center_deg + theta_band_deg, 1)
    draw_legend(img, [((60, 200, 60), f"in-band RANSAC line [{lo},{hi}]deg")])
    draw_summary(img, f"Constrained RANSAC: {len(lines)} straight line(s), mean inlier "
                      f"fraction {mean_frac*100:.0f}%, every model rejected unless its axis is "
                      f"within {theta_center_deg:.0f}+/-{theta_band_deg:.0f}deg. No cross-"
                      "direction lines survive the band veto.")
    return _record(
        "geometry.ransac_line_constrained", "Constrained RANSAC lines", _RANSAC_REF,
        metric_name="inlier_frac", metric_value=mean_frac, overlay=img, infer_ms=t.ms,
        summary=f"{len(lines)} in-band RANSAC line(s), mean inlier {mean_frac*100:.0f}%",
        extra={"theta_center_deg": round(float(theta_center_deg), 2),
               "theta_band_deg": round(float(theta_band_deg), 2),
               "n_lines": len(lines), "angles_deg": axes,
               "roi_used": bool(roi_mask is not None), "lines": lines})


# --- image-first registry wrappers ------------------------------------------------------
def _band_for(image: np.ndarray, view_type: str | None, annotations: list | None,
              theta_center_deg: float | None, theta_band_deg: float | None) -> tuple[float, float]:
    if theta_center_deg is not None and theta_band_deg is not None:
        return float(theta_center_deg), float(theta_band_deg)
    from .roi import orientation_band

    c, b = orientation_band(view_type, annotations, np.asarray(image).shape, np.asarray(image))
    return (float(theta_center_deg) if theta_center_deg is not None else c,
            float(theta_band_deg) if theta_band_deg is not None else b)


def _roi_from(image: np.ndarray, annotations: list | None, roi_label: Any) -> np.ndarray | None:
    if not annotations or roi_label is None:
        return None
    from .roi import combine_by_label

    mask = combine_by_label(annotations, np.asarray(image).shape, roi_label)
    return mask if mask.any() else None


def hough_constrained_method(
    image: Any, *, view_type: str | None = None, annotations: list | None = None,
    roi_label: Any = None, theta_center_deg: float | None = None,
    theta_band_deg: float | None = None, denoise: str = "gaussian", edge: str = "canny",
    min_len_px: float | None = None, use_gate: bool = True, **_: Any,
) -> dict[str, Any]:
    """Registry entry: build the ROI edge map + band from the prior, then constrained Hough."""
    bgr = as_bgr(image)
    roi_mask = _roi_from(bgr, annotations, roi_label)
    center, band = _band_for(bgr, view_type, annotations, theta_center_deg, theta_band_deg)
    be = preprocess_for_lines(bgr, roi_mask=roi_mask, denoise=denoise, edge=edge)
    if use_gate:
        be = gradient_orientation_gate(be, _gray_clahe(bgr), center, band)
    return hough_constrained(be, center, band, min_len_px=min_len_px,
                             roi_mask=roi_mask, bgr=bgr)


def ransac_line_constrained_method(
    image: Any, *, view_type: str | None = None, annotations: list | None = None,
    roi_label: Any = None, theta_center_deg: float | None = None,
    theta_band_deg: float | None = None, denoise: str = "gaussian", edge: str = "canny",
    use_gate: bool = True, **_: Any,
) -> dict[str, Any]:
    """Registry entry: build the ROI edge map + band from the prior, then constrained RANSAC."""
    bgr = as_bgr(image)
    roi_mask = _roi_from(bgr, annotations, roi_label)
    center, band = _band_for(bgr, view_type, annotations, theta_center_deg, theta_band_deg)
    be = preprocess_for_lines(bgr, roi_mask=roi_mask, denoise=denoise, edge=edge)
    if use_gate:
        be = gradient_orientation_gate(be, _gray_clahe(bgr), center, band)
    return ransac_line_constrained(be, center, band, roi_mask=roi_mask, bgr=bgr)


__all__ = [
    "FAM_CONSTRAINED",
    "preprocess_for_lines",
    "gradient_orientation_gate",
    "hough_constrained",
    "ransac_line_constrained",
    "hough_constrained_method",
    "ransac_line_constrained_method",
]
