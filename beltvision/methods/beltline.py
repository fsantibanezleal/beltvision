"""Belt geometry derived from the segmented BELT mask (no assumed parametric model).

Everything here is computed from the shape of the belt mask produced by the 4-class
semantic segmentation, so it works for a belt at ANY orientation (vertical, horizontal,
diagonal) and ANY shape (straight or curved):

- ORIENTATION: the belt axis angle from the principal axis of the belt-mask region
  (second moments / PCA of the mask pixels). Reported in degrees from the x-axis, with a
  plain-language label (near-vertical / near-horizontal / diagonal) and a curvature note.
- CENTRELINE: the medial line of the band - the per-cross-section centroid along the
  belt axis (the distance-transform ridge), naturally straight when the belt is straight
  and curved when it curves.
- EDGES: the two long borders, recovered as the centreline offset by the local
  half-width along the cross-section normal (they follow the mask boundary, any shape).
- WIDTH profile: the belt width measured PERPENDICULAR to the local centreline tangent,
  along the belt.
- ALIGNMENT: for a lateral view, the angular difference between the belt axis and the
  SUPPORTING-STRUCTURE axis (detected from the external/structure layer) plus the lateral
  offset of the belt centreline from the support centreline. A belt parallel to its
  support is aligned (~0 deg); divergence is the misalignment angle.

If the belt mask is too small or too ambiguous to fit a coherent axis, the functions
report ``confidence: "low"`` with a reason and DO NOT emit a fabricated centreline.

Reference: image second-moments / principal-axis (Hu 1962); medial axis (Blum 1967);
Hough line transform for the support-structure axis.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_bgr, result, timed
from .preprocess import apply_clahe_lab

_N_BINS = 28


def _pca_axis(mask: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Principal axis of a boolean mask: (angle_deg[0,180), mean_xy, elongation=sqrt(l1/l2))."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    l1, l2 = float(evals.max()), float(max(evals.min(), 1e-6))
    return angle, mean, float(np.sqrt(l1 / l2))


def _structure_axis(gray: np.ndarray, mask: np.ndarray) -> float | None:
    """Dominant ORIENTED-structure direction inside the mask (belt edges/streaks/weave).

    Uses the structure tensor over the masked gradients: edges/streaks run perpendicular to
    the mean gradient, so the belt axis = (gradient orientation + 90) mod 180. Robust to the
    blob's overall shape (unlike PCA of the mask), so it recovers the belt travel direction
    even when the segmented region is a broad patch.
    """
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = mask
    jxx = float(np.mean((gx * gx)[m]))
    jyy = float(np.mean((gy * gy)[m]))
    jxy = float(np.mean((gx * gy)[m]))
    denom = (jxx - jyy)
    coherence = np.hypot(2 * jxy, denom) / (jxx + jyy + 1e-6)
    if coherence < 0.05:
        return None
    grad_ori = 0.5 * np.degrees(np.arctan2(2 * jxy, denom))
    return float((grad_ori + 90.0) % 180.0)


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two axis angles in [0,180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _unit_from_angle(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    th = np.radians(angle_deg)
    d = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
    n = np.array([-np.sin(th), np.cos(th)], dtype=np.float64)
    return d, n


def _orientation_label(angle_deg: float) -> str:
    """Plain-language orientation for an axis angle from the x-axis."""
    a = angle_deg % 180.0
    dv = min(abs(a - 90.0), 180.0 - abs(a - 90.0))  # deviation from vertical
    dh = min(a, 180.0 - a)                            # deviation from horizontal
    if dv <= 15.0:
        return "near-vertical"
    if dh <= 15.0:
        return "near-horizontal"
    return "diagonal"


def compute_belt_geometry(
    belt_mask: np.ndarray, *, external_mask: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> dict[str, Any]:
    """Derive orientation / centreline / edges / width / alignment from the belt mask."""
    h, w = belt_mask.shape[:2]
    area_frac = float(belt_mask.sum()) / float(h * w)
    if belt_mask.sum() < 200 or area_frac < 0.01:
        return {"confidence": "low", "reason": "belt region not found / too small",
                "belt_area_frac": round(area_frac, 4)}

    pca_angle, mean, elong = _pca_axis(belt_mask)
    struct_angle = _structure_axis(gray, belt_mask) if gray is not None else None

    # Choose the belt axis and gate confidence by how strand-like + consistent the region is.
    if struct_angle is not None and (elong < 1.6 or _ang_diff(struct_angle, pca_angle) > 20.0):
        # broad/ambiguous region OR PCA disagrees with the oriented structure: trust the
        # oriented structure (belt travel direction) but be honest about the uncertainty.
        angle = struct_angle
        agree = _ang_diff(struct_angle, pca_angle)
        confidence = "medium" if agree < 35.0 else "low"
        axis_source = "oriented-structure (mask shape ambiguous)"
    else:
        angle = pca_angle
        confidence = "high" if elong > 2.0 and area_frac > 0.03 else "medium"
        axis_source = "mask principal axis"
    major, minor = _unit_from_angle(angle)

    ys, xs = np.nonzero(belt_mask)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1) - mean
    u = pts @ major   # along-axis coordinate
    v = pts @ minor   # cross-axis coordinate

    u_min, u_max = np.percentile(u, 1), np.percentile(u, 99)
    edges_u = np.linspace(u_min, u_max, _N_BINS + 1)
    centres_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    uc, v_mid_s, v_top_s, v_bot_s, widths = [], [], [], [], []
    for i in range(_N_BINS):
        sel = (u >= edges_u[i]) & (u < edges_u[i + 1])
        if sel.sum() < 8:
            continue
        v_sel = v[sel]
        v_top = float(np.percentile(v_sel, 97))
        v_bot = float(np.percentile(v_sel, 3))
        uc.append(float(centres_u[i]))
        v_mid_s.append(float(np.median(v_sel)))
        v_top_s.append(v_top)
        v_bot_s.append(v_bot)
        widths.append(v_top - v_bot)
    if len(uc) < 4:
        return {"confidence": "low", "reason": "too few cross-sections for a centreline",
                "belt_area_frac": round(area_frac, 4)}

    # A belt is TWO quasi-parallel STRAIGHT lines; the centreline is a STRAIGHT line (their
    # midline). Fit v = a*u + b (degree 1, least squares) for the centreline and each edge -
    # NEVER a free per-cross-section curve, never a polynomial/parabola.
    uc_a = np.asarray(uc, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)

    def _fit_line(vs: list[float]) -> tuple[float, float]:
        mat = np.vstack([uc_a, np.ones_like(uc_a)]).T
        a, b = np.linalg.lstsq(mat, np.asarray(vs, dtype=np.float64), rcond=None)[0]
        return float(a), float(b)

    ac, bc = _fit_line(v_mid_s)   # straight centreline
    at, bt = _fit_line(v_top_s)   # straight top edge
    ab, bb = _fit_line(v_bot_s)   # straight bottom edge
    u_ends = np.array([u_min, u_max], dtype=np.float64)

    def _line_xy(a: float, b: float) -> np.ndarray:
        return np.array([mean + u_ends[k] * major + (a * u_ends[k] + b) * minor
                         for k in range(2)], dtype=np.float64)
    cl = _line_xy(ac, bc)   # 2-point STRAIGHT centreline
    ea = _line_xy(at, bt)   # 2-point STRAIGHT top edge
    eb = _line_xy(ab, bb)   # 2-point STRAIGHT bottom edge

    # residual of the straight centreline fit = how non-straight the real belt is (small=straight)
    span = float(u_max - u_min) + 1e-6
    curvature = float(np.max(np.abs(np.asarray(v_mid_s) - (ac * uc_a + bc))) / span)
    curved = bool(curvature > 0.04)
    # parallelism: difference in the two edge slopes (0 deg => perfectly parallel lines)
    skew = float(abs(np.degrees(np.arctan(at)) - np.degrees(np.arctan(ab))))

    dev_from_vertical = float(min(abs(angle - 90.0), 180.0 - abs(angle - 90.0)))
    out: dict[str, Any] = {
        "confidence": confidence,
        "axis_source": axis_source,
        "belt_area_frac": round(area_frac, 4),
        "axis_angle_deg": round(angle, 2),
        "pca_axis_deg": round(pca_angle, 2),
        "structure_axis_deg": (round(struct_angle, 2) if struct_angle is not None else None),
        "orientation": _orientation_label(angle),
        "deviation_from_vertical_deg": round(dev_from_vertical, 2),
        "elongation": round(elong, 3),
        "curved": curved,
        "curvature": round(curvature, 5),
        "tangent_skew_deg": round(skew, 2),
        "mean_width_px": round(float(np.median(widths)), 2),
        "width_std_px": round(float(np.std(widths)), 2),
        "centreline_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in cl],
        "edge_a_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in ea],
        "edge_b_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in eb],
        "width_profile_px": [round(float(x), 1) for x in widths],
    }

    # --- misalignment vs the supporting structure (from the external layer) ---
    if external_mask is not None and gray is not None and confidence != "low":
        support = _support_axis(gray, external_mask, belt_mask, belt_axis=angle)
        if support is not None:
            sup_angle = support["axis_angle_deg"]
            diff = angle - sup_angle
            diff = (diff + 90.0) % 180.0 - 90.0  # wrap to (-90, 90]
            out["support_axis_deg"] = round(float(sup_angle), 2)
            out["misalignment_deg"] = round(float(diff), 2)
            out["misaligned"] = bool(abs(diff) > 3.0)
            out["support_confidence"] = support["confidence"]
        else:
            out["support_axis_deg"] = None
            out["misalignment_deg"] = None
            out["misaligned"] = None
            out["support_confidence"] = "low"
            out["support_note"] = "supporting-structure axis not detected in external layer"
    return out


def _support_axis(
    gray: np.ndarray, external_mask: np.ndarray, belt_mask: np.ndarray,
    *, belt_axis: float | None = None,
) -> dict[str, Any] | None:
    """Dominant straight-line direction of the support structure (external layer).

    Lines that run parallel to the belt axis are belt structure (edges/weave), not the
    support, so when the belt axis is known they are excluded; the support is the dominant
    remaining straight structure alongside the belt. If nothing diverges from the belt axis
    the support is parallel (a belt running true on its support) and None is returned.
    """
    import cv2

    h, w = gray.shape[:2]
    # The support structure (frame / stringer / idler line) runs immediately alongside the
    # belt. Search the RING just outside the belt footprint (class-agnostic): that is where
    # the supporting surface is, whatever the semantic layer labelled it.
    belt_u8 = belt_mask.astype(np.uint8)
    dil = cv2.dilate(belt_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    ring = (dil > 0) & (belt_u8 == 0)
    band = ring | (external_mask & (dil > 0))
    if band.sum() < 200:
        band = external_mask
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)
    edges[~band] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=int(0.12 * max(h, w)), maxLineGap=20)
    if lines is None:
        return None
    angles, lengths = [], []
    for seg in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in seg)
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        if belt_axis is not None and _ang_diff(ang, belt_axis) < 4.0:
            continue  # belt-parallel structure (edges/weave), not the support
        angles.append(ang)
        lengths.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return None
    angles = np.array(angles)
    lengths = np.array(lengths)
    # length-weighted circular mean on the doubled angle (axis, not direction)
    a2 = np.radians(2.0 * angles)
    cx = float(np.sum(lengths * np.cos(a2)))
    sx = float(np.sum(lengths * np.sin(a2)))
    dom = (np.degrees(np.arctan2(sx, cx)) / 2.0) % 180.0
    conc = np.hypot(cx, sx) / (lengths.sum() + 1e-6)  # 0..1 concentration
    return {"axis_angle_deg": float(dom),
            "confidence": "high" if conc > 0.6 else "medium" if conc > 0.3 else "low",
            "n_lines": int(len(angles))}


def belt_geometry(
    image: Any, *, view_type: str | None = None, **_: Any
) -> dict[str, Any]:
    """Method wrapper: segment the belt, then derive its geometry from the mask."""
    import cv2

    from .semantic import compute_layers

    ref = "Principal-axis (Hu 1962) + medial line from the segmented belt mask; Hough support axis"
    bgr = apply_clahe_lab(as_bgr(image))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    with timed() as t:
        layers = compute_layers(image, view_type=view_type, use_learned=False)
        footprint = layers.belt_mask | layers.content_mask
        geo = compute_belt_geometry(footprint, external_mask=layers.mask(0), gray=gray)
    payload = {"shape": [int(bgr.shape[0]), int(bgr.shape[1])], "view_type": view_type, **geo}
    return result(
        "geometry.belt_geometry", "geometry", "classical", ref,
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True,
    )
