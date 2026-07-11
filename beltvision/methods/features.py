"""Capability 6: the classical feature / edge / keypoint / texture toolbox (all LIVE).

A broad bench of the standard, textbook classical operators, each implemented for real on
opencv / scikit-image / numpy and each producing THREE things a viewer can trust:

- a DRAWN overlay (base64 PNG data URL) that is visually distinct from every other method,
- a single scalar / count METRIC (with an explicit ``metric_name``), and
- a family + reference so the front end can group and cite it.

Every operator is CLAHE-first (the mandatory preprocess) and, where a belt region is
relevant (edge density, corners, texture, boundary line fits, shape), it runs INSIDE the
belt footprint mask produced by :mod:`beltvision.methods.semantic`. When no mask is passed
the mask-relevant operators segment the belt themselves; :func:`run_all` segments ONCE and
shares the mask across the bench so it stays cheap.

Families
--------
- ``edge_operator``      Canny, Sobel, Scharr, Laplacian, LoG, Prewitt, Roberts, morph-grad.
- ``lines_boundaries``   HoughLinesP, RANSAC straight-LINE boundary fit, Radon orientation.
- ``superpixels``        SLIC over-segmentation.
- ``shape``              OBB (minAreaRect), external contours.
- ``corners_keypoints``  Harris, Shi-Tomasi, ORB.
- ``texture``            Gabor bank, Local Binary Pattern.

Plus :func:`geometry_analysis`, a single consolidated straight-line geometry read that
draws the corrected centreline + two straight edges + OBB and cross-checks the belt axis
against Hough, a RANSAC boundary-line fit and Radon on one legible overlay.

References: Canny 1986; Sobel-Feldman; Scharr 2000; Marr-Hildreth 1980 (LoG); Prewitt 1970;
Roberts 1963; Harris & Stephens 1988; Shi & Tomasi 1994; Rublee et al. 2011 (ORB); Achanta
et al. 2012 (SLIC); Fischler & Bolles 1981 (RANSAC) + skimage ``LineModelND``; Radon
transform; Gabor 1946 filter bank; Ojala et al. 2002 (LBP).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_bgr, result, timed
from .preprocess import apply_clahe_lab

# --- families ---------------------------------------------------------------------------
FAM_EDGE = "edge_operator"
FAM_LINES = "lines_boundaries"
FAM_SUPERPIXEL = "superpixels"
FAM_SHAPE = "shape"
FAM_CORNERS = "corners_keypoints"
FAM_TEXTURE = "texture"

_CAP = "features"          # capability axis
_TIER = "classical"        # compute tier (manifest axis): all classical, no learned weight


# --- small shared plumbing --------------------------------------------------------------
def _clahe(image: Any) -> np.ndarray:
    return apply_clahe_lab(as_bgr(image))


def _gray(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _belt_mask(image: Any) -> np.ndarray:
    """The belt FOOTPRINT (exposed belt + its load) from the 4-class segmentation."""
    from .semantic import compute_layers

    layers = compute_layers(image, use_learned=False)
    return layers.belt_mask | layers.content_mask


def _darken(bgr: np.ndarray, factor: float = 0.5) -> np.ndarray:
    import cv2

    return cv2.addWeighted(bgr, 1.0 - factor, np.zeros_like(bgr), factor, 0)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _density(binary: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is not None and bool(mask.any()):
        num = float(np.count_nonzero(binary & mask))
        den = float(np.count_nonzero(mask))
    else:
        num = float(np.count_nonzero(binary))
        den = float(binary.size)
    return round(num / max(den, 1.0), 5)


def _to_b64(bgr: np.ndarray) -> str:
    from ..render import to_png_b64

    return to_png_b64(bgr)


def _legend(img: np.ndarray, entries: list[tuple[tuple[int, int, int], str]]) -> None:
    from ..render import draw_legend

    draw_legend(img, entries)


def _summary(img: np.ndarray, text: str) -> None:
    from ..render import draw_summary

    draw_summary(img, text)


def _finish(
    method_id: str,
    family: str,
    name: str,
    reference: str,
    *,
    metric_name: str,
    metric_value: float,
    overlay: np.ndarray,
    infer_ms: float,
    web_drivable: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the gate-measured envelope on the COMPACT metric payload, then attach the
    (heavy) overlay outside the trace so the lane stays honest and manifests stay slim."""
    payload: dict[str, Any] = {
        "name": name,
        "family": family,
        "metric_name": metric_name,
        "metric_value": round(float(metric_value), 5),
    }
    if extra:
        payload.update(extra)
    res = result(
        method_id, _CAP, _TIER, reference,
        payload=payload, model_bytes=0, infer_ms=infer_ms, web_drivable=web_drivable,
    )
    res["overlay_b64"] = _to_b64(overlay)
    return res


# --- EDGE OPERATORS (metric: edge density) ----------------------------------------------
def _edge_method(
    method_id: str, name: str, reference: str, color: tuple[int, int, int],
    image: Any, mask: np.ndarray | None, compute, thr: float,
) -> dict[str, Any]:
    import cv2

    bgr = _clahe(image)
    gray = _gray(bgr)
    with timed() as t:
        mag = compute(gray, cv2)
        mag01 = _norm01(mag)
        binary = mag01 >= thr
        dens = _density(binary, mask)
    # overlay: darkened frame + additive coloured edge magnitude
    img = _darken(bgr, 0.55)
    layer = (mag01[..., None] * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    img = cv2.add(img, layer)
    _legend(img, [(color, f"{name} (>= {thr:.2f})")])
    scope = "belt mask" if mask is not None else "full frame"
    _summary(img, f"{name}: edge density {dens*100:.1f}% of {scope} "
                  f"(fraction of pixels above the {thr:.2f} normalized-gradient threshold).")
    return _finish(method_id, FAM_EDGE, name, reference,
                   metric_name="edge_density", metric_value=dens, overlay=img,
                   infer_ms=t.ms, extra={"threshold": float(thr),
                                         "within_mask": bool(mask is not None)})


def canny(image: Any, *, mask: np.ndarray | None = None, lo: int = 50, hi: int = 150, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), int(lo), int(hi)).astype(np.float32)
    return _edge_method("features.canny", "Canny", "Canny 1986 (hysteresis edge detector)",
                        (255, 245, 60), image, mask, compute, 0.5)


def sobel_magnitude(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    return _edge_method("features.sobel", "Sobel magnitude", "Sobel-Feldman gradient operator",
                        (80, 230, 80), image, mask, compute, 0.28)


def scharr(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        return np.sqrt(gx * gx + gy * gy)
    return _edge_method("features.scharr", "Scharr magnitude", "Scharr 2000 (rotation-optimal 3x3)",
                        (40, 170, 255), image, mask, compute, 0.28)


def laplacian(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        return np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return _edge_method("features.laplacian", "Laplacian", "Laplacian second-derivative operator",
                        (210, 70, 210), image, mask, compute, 0.3)


def laplacian_of_gaussian(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        blur = cv2.GaussianBlur(gray, (0, 0), 1.6)
        return np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
    return _edge_method("features.log", "Laplacian-of-Gaussian",
                        "Marr & Hildreth 1980 (LoG blob/edge)", (255, 140, 0),
                        image, mask, compute, 0.3)


def prewitt(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        from skimage.filters import prewitt as sk_prewitt

        return sk_prewitt(gray.astype(np.float32))
    return _edge_method("features.prewitt", "Prewitt", "Prewitt 1970 gradient operator",
                        (0, 210, 210), image, mask, compute, 0.25)


def roberts_cross(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        from skimage.filters import roberts as sk_roberts

        return sk_roberts(gray.astype(np.float32))
    return _edge_method("features.roberts", "Roberts cross", "Roberts 1963 cross-gradient",
                        (150, 255, 150), image, mask, compute, 0.22)


def morphological_gradient(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    def compute(gray, cv2):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k).astype(np.float32)
    return _edge_method("features.morph_gradient", "Morphological gradient",
                        "Morphological gradient (dilation - erosion)", (255, 80, 180),
                        image, mask, compute, 0.3)


# --- LINES / BOUNDARIES -----------------------------------------------------------------
def _axis_circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Length-weighted circular mean of axis angles (mod 180), on the doubled angle."""
    a2 = np.radians(2.0 * angles)
    cx = float(np.sum(weights * np.cos(a2)))
    sx = float(np.sum(weights * np.sin(a2)))
    return float((np.degrees(np.arctan2(sx, cx)) / 2.0) % 180.0)


def hough_lines_p(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """Probabilistic Hough straight segments; metric = number of lines (+ dominant angle)."""
    import cv2

    bgr = _clahe(image)
    gray = _gray(bgr)
    h, w = gray.shape
    with timed() as t:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        if mask is not None:
            dil = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            edges[dil == 0] = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=int(0.18 * min(h, w)), maxLineGap=16)
        segs: list[tuple[int, int, int, int]] = []
        angles: list[float] = []
        lengths: list[float] = []
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                segs.append((int(x1), int(y1), int(x2), int(y2)))
                angles.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0))
                lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
        n = len(segs)
        dom = _axis_circular_mean(np.asarray(angles), np.asarray(lengths)) if n else None
    img = _darken(bgr, 0.5)
    for x1, y1, x2, y2 in segs:
        cv2.line(img, (x1, y1), (x2, y2), (70, 230, 70), 2, cv2.LINE_AA)
    _legend(img, [((70, 230, 70), "Hough straight segment")])
    dom_s = "n/a" if dom is None else f"{dom:.1f}deg"
    _summary(img, f"HoughLinesP: {n} straight segment(s); dominant axis {dom_s} "
                  f"(from the x-axis). Straight-edge / line candidate finder.")
    return _finish("features.hough_lines_p", FAM_LINES, "HoughLinesP",
                   "Matas et al. 2000 (progressive probabilistic Hough transform)",
                   metric_name="n_lines", metric_value=float(n), overlay=img, infer_ms=t.ms,
                   extra={"dominant_angle_deg": (round(dom, 2) if dom is not None else None)})


def _fit_boundary_lines(mask: np.ndarray) -> dict[str, Any] | None:
    """Split the belt-mask boundary into its two long sides and RANSAC-fit a STRAIGHT line
    to each (skimage ``LineModelND``). Returns per-side angle + inlier fraction + endpoints."""
    import cv2
    from skimage.measure import LineModelND, ransac

    m8 = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m8)
    if xs.size < 80:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = np.array([-major[1], major[0]], dtype=np.float64)

    er = cv2.erode(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bnd = (m8 > 0) & (er == 0)
    by, bx = np.nonzero(bnd)
    if bx.size < 40:
        return None
    bp = np.stack([bx.astype(np.float64), by.astype(np.float64)], axis=1)
    u = (bp - mean) @ major
    v = (bp - mean) @ minor
    u_lo, u_hi = np.percentile(u, 2), np.percentile(u, 98)
    pad = 0.06 * (u_hi - u_lo)
    core = (u >= u_lo + pad) & (u <= u_hi - pad)  # drop the two end caps (keep the long sides)
    side_a = bp[core & (v > 0)]
    side_b = bp[core & (v < 0)]

    def _fit(side: np.ndarray) -> dict[str, Any] | None:
        if side.shape[0] < 12:
            return None
        try:
            model, inliers = ransac(side, LineModelND, min_samples=2,
                                    residual_threshold=2.5, max_trials=250)
        except Exception:  # noqa: BLE001 - a degenerate side degrades to None
            return None
        if model is None or inliers is None:
            return None
        origin = getattr(model, "origin", None)
        direction = getattr(model, "direction", None)
        if origin is None or direction is None:
            origin, direction = model.params  # skimage < 0.26 fallback
        ang = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
        tt = (side - origin) @ direction
        p0 = origin + float(tt.min()) * direction
        p1 = origin + float(tt.max()) * direction
        return {"angle_deg": round(ang, 2),
                "inlier_frac": round(float(np.mean(inliers)), 3),
                "n_points": int(side.shape[0]),
                "p0": [round(float(p0[0]), 1), round(float(p0[1]), 1)],
                "p1": [round(float(p1[0]), 1), round(float(p1[1]), 1)]}

    a = _fit(side_a)
    b = _fit(side_b)
    if a is None and b is None:
        return None
    return {"line_a": a, "line_b": b,
            "mean": [round(float(mean[0]), 1), round(float(mean[1]), 1)]}


def ransac_lines(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """RANSAC straight-LINE fit of the two belt-mask boundaries; metric = mean inlier frac."""
    import cv2

    bgr = _clahe(image)
    m = mask if mask is not None else _belt_mask(image)
    with timed() as t:
        fit = _fit_boundary_lines(m)
    img = _darken(bgr, 0.5)
    fracs: list[float] = []
    angs: list[float] = []
    legend: list[tuple[tuple[int, int, int], str]] = []
    if fit:
        for side, col, lbl in (("line_a", (60, 200, 60), "RANSAC edge A"),
                               ("line_b", (60, 220, 220), "RANSAC edge B")):
            ln = fit.get(side)
            if ln:
                p0 = tuple(int(round(c)) for c in ln["p0"])
                p1 = tuple(int(round(c)) for c in ln["p1"])
                cv2.line(img, p0, p1, col, 3, cv2.LINE_AA)
                fracs.append(ln["inlier_frac"])
                angs.append(ln["angle_deg"])
                legend.append((col, f"{lbl}  {ln['angle_deg']:.1f}deg"))
    mean_frac = float(np.mean(fracs)) if fracs else 0.0
    if legend:
        _legend(img, legend)
    par = (f"parallelism {abs(angs[0]-angs[1]):.1f}deg" if len(angs) == 2 else "one side only")
    _summary(img, f"RANSAC straight-LINE boundary fit: {len(angs)} belt edge(s), "
                  f"mean inlier fraction {mean_frac*100:.0f}%, {par}. "
                  "Robust straight lines fit to the segmented belt borders.")
    return _finish("features.ransac_lines", FAM_LINES, "RANSAC straight-line boundary fit",
                   "Fischler & Bolles 1981 (RANSAC); skimage LineModelND",
                   metric_name="inlier_frac", metric_value=mean_frac, overlay=img, infer_ms=t.ms,
                   extra={"line_a": (fit or {}).get("line_a"),
                          "line_b": (fit or {}).get("line_b"),
                          "edge_angles_deg": [round(a, 2) for a in angs]})


def _radon_orientation(gray: np.ndarray, max_side: int = 160) -> tuple[float, float]:
    """Dominant structural orientation (deg from x-axis) + strength via the Radon transform."""
    import cv2
    from skimage.transform import radon

    h, w = gray.shape
    side = min(max_side, min(h, w))
    roi = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float64)
    roi -= roi.mean()
    theta = np.arange(0.0, 180.0, 1.0)
    sino = radon(roi, theta=theta, circle=False)
    col_var = sino.var(axis=0)
    dom_idx = int(np.argmax(col_var))
    structure_angle = float((theta[dom_idx] - 90.0) % 180.0)
    strength = float(col_var[dom_idx] / (col_var.mean() + 1e-9))
    return structure_angle, strength


def radon_orientation(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """Radon dominant orientation; metric = orientation (deg). Overlay = orientation arrow."""
    import cv2

    bgr = _clahe(image)
    gray = _gray(bgr)
    h, w = gray.shape
    with timed() as t:
        angle, strength = _radon_orientation(gray)
    img = _darken(bgr, 0.5)
    cx, cy = w / 2.0, h / 2.0
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    arm = 0.4 * min(h, w)
    p0 = (int(cx - arm * d[0]), int(cy - arm * d[1]))
    p1 = (int(cx + arm * d[0]), int(cy + arm * d[1]))
    cv2.arrowedLine(img, p0, p1, (235, 120, 40), 3, cv2.LINE_AA, tipLength=0.06)
    cv2.arrowedLine(img, p1, p0, (235, 120, 40), 3, cv2.LINE_AA, tipLength=0.06)
    _legend(img, [((235, 120, 40), f"Radon dominant orientation {angle:.1f}deg")])
    _summary(img, f"Radon orientation {angle:.1f}deg from the x-axis (strength "
                  f"{strength:.1f}x mean). Noise-robust global structure direction.")
    return _finish("features.radon_orientation", FAM_LINES, "Radon orientation",
                   "Radon transform (skimage.transform.radon)",
                   metric_name="orientation_deg", metric_value=angle, overlay=img,
                   infer_ms=t.ms, web_drivable=False,
                   extra={"orientation_strength": round(strength, 3)})


# --- SUPERPIXELS ------------------------------------------------------------------------
def slic_superpixels(image: Any, *, mask: np.ndarray | None = None, n_segments: int = 220,
                     compactness: float = 12.0, **_: Any) -> dict[str, Any]:
    """SLIC over-segmentation; metric = number of superpixels. Overlay = boundaries."""
    from skimage.color import rgb2lab
    from skimage.segmentation import find_boundaries
    from skimage.segmentation import slic as sk_slic

    bgr = _clahe(image)
    rgb = bgr[:, :, ::-1]
    with timed() as t:
        lab = rgb2lab(rgb / 255.0)
        labels = sk_slic(lab, n_segments=int(n_segments), compactness=float(compactness),
                         sigma=1.0, start_label=0, channel_axis=-1)
        n_actual = int(np.unique(labels).size)
        bnd = find_boundaries(labels, mode="outer")
    img = _darken(bgr, 0.35)
    img[bnd] = (60, 255, 255)
    _legend(img, [((60, 255, 255), "superpixel boundary")])
    _summary(img, f"SLIC: {n_actual} superpixels (compactness {compactness:.0f}). "
                  "Cheap, region-preserving over-segmentation.")
    return _finish("features.slic", FAM_SUPERPIXEL, "SLIC superpixels",
                   "Achanta et al. 2012 (SLIC), TPAMI 34(11)",
                   metric_name="n_superpixels", metric_value=float(n_actual), overlay=img,
                   infer_ms=t.ms, web_drivable=False,
                   extra={"n_segments_requested": int(n_segments)})


# --- SHAPE ------------------------------------------------------------------------------
def obb(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """Oriented bounding box of the belt mask (cv2.minAreaRect); metric = OBB angle."""
    import cv2

    bgr = _clahe(image)
    m = mask if mask is not None else _belt_mask(image)
    h, w = bgr.shape[:2]
    with timed() as t:
        ys, xs = np.nonzero(m)
        rect = None
        if xs.size >= 3:
            fg = np.stack([xs, ys], axis=1).astype(np.int32)
            rect = cv2.minAreaRect(fg)
    img = _darken(bgr, 0.45)
    if rect is not None:
        (cxr, cyr), (rw, rh), ang = rect
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(img, [box], True, (60, 200, 255), 3, cv2.LINE_AA)
        cv2.circle(img, (int(cxr), int(cyr)), 4, (60, 200, 255), -1, cv2.LINE_AA)
        angle = float(ang)
        width, height = float(rw), float(rh)
        _legend(img, [((60, 200, 255), f"OBB {angle:.1f}deg  {width:.0f}x{height:.0f}px")])
        _summary(img, f"Oriented bounding box (minAreaRect): angle {angle:.1f}deg, "
                      f"size {width:.0f}x{height:.0f}px, area {width*height/ (h*w) * 100:.0f}% "
                      "of frame.")
        extra = {"obb_angle_deg": round(angle, 2), "width_px": round(width, 1),
                 "height_px": round(height, 1),
                 "center": [round(float(cxr), 1), round(float(cyr), 1)],
                 "box_points": [[int(x), int(y)] for x, y in box]}
        metric = angle
    else:
        _legend(img, [((60, 200, 255), "OBB (belt not found)")])
        _summary(img, "OBB: no belt region to bound.")
        extra = {"obb_angle_deg": None}
        metric = 0.0
    return _finish("features.obb", FAM_SHAPE, "Oriented bounding box (minAreaRect)",
                   "OpenCV minAreaRect (rotating-calipers OBB)",
                   metric_name="obb_angle_deg", metric_value=metric, overlay=img,
                   infer_ms=t.ms, extra=extra)


def contours(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """External contours of the belt mask; metric = number of contours (+ total perimeter)."""
    import cv2

    bgr = _clahe(image)
    m = (mask if mask is not None else _belt_mask(image)).astype(np.uint8)
    with timed() as t:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= 25.0]
        perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    img = _darken(bgr, 0.45)
    cv2.drawContours(img, cnts, -1, (80, 255, 140), 2, cv2.LINE_AA)
    _legend(img, [((80, 255, 140), "belt-region contour")])
    _summary(img, f"Contours: {len(cnts)} external contour(s), total perimeter "
                  f"{perim:.0f}px. Region outline of the segmented belt footprint.")
    return _finish("features.contours", FAM_SHAPE, "External contours",
                   "Suzuki & Abe 1985 (contour tracing); OpenCV findContours",
                   metric_name="n_contours", metric_value=float(len(cnts)), overlay=img,
                   infer_ms=t.ms, extra={"total_perimeter_px": round(perim, 1)})


# --- CORNERS / KEYPOINTS ----------------------------------------------------------------
def _keypoint_overlay(bgr: np.ndarray, pts: np.ndarray, color: tuple[int, int, int],
                      name: str, reference_note: str) -> np.ndarray:
    import cv2

    img = _darken(bgr, 0.5)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    _legend(img, [(color, f"{name}  n={len(pts)}")])
    _summary(img, f"{name}: {len(pts)} point(s) detected. {reference_note}")
    return img


def harris(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """Harris corner response; metric = number of corners."""
    import cv2

    bgr = _clahe(image)
    gray = np.float32(_gray(bgr))
    with timed() as t:
        resp = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        resp = cv2.dilate(resp, None)
        # Threshold relative to the response WITHIN the region of interest, so belt corners
        # are not drowned out by stronger structural corners outside the mask.
        ref = resp[mask] if (mask is not None and mask.any()) else resp
        peak = float(ref.max()) if ref.size else 0.0
        thr = 0.01 * peak if peak > 0 else 1e18
        corner = resp > thr
        if mask is not None:
            corner &= mask
        ys, xs = np.nonzero(corner)
        pts = np.stack([xs, ys], axis=1)
    img = _keypoint_overlay(bgr, pts, (60, 80, 255), "Harris corners",
                            "Corner (two strong gradient directions) response.")
    return _finish("features.harris", FAM_CORNERS, "Harris corners",
                   "Harris & Stephens 1988 (combined corner/edge detector)",
                   metric_name="n_corners", metric_value=float(len(pts)), overlay=img,
                   infer_ms=t.ms)


def shi_tomasi(image: Any, *, mask: np.ndarray | None = None, max_corners: int = 400, **_: Any) -> dict[str, Any]:
    """Shi-Tomasi good features to track; metric = number of features."""
    import cv2

    bgr = _clahe(image)
    gray = _gray(bgr)
    roi_mask = mask.astype(np.uint8) if mask is not None else None
    with timed() as t:
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=int(max_corners), qualityLevel=0.01,
                                          minDistance=7, mask=roi_mask, blockSize=7)
        pts = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2), np.float32)
    img = _keypoint_overlay(bgr, pts, (255, 180, 40), "Shi-Tomasi features",
                            "Good-features-to-track (min-eigenvalue) corners.")
    return _finish("features.shi_tomasi", FAM_CORNERS, "Shi-Tomasi good features",
                   "Shi & Tomasi 1994 (Good Features to Track), CVPR",
                   metric_name="n_features", metric_value=float(len(pts)), overlay=img,
                   infer_ms=t.ms)


def orb(image: Any, *, mask: np.ndarray | None = None, n_features: int = 500, **_: Any) -> dict[str, Any]:
    """ORB (Oriented FAST + rotated BRIEF) keypoints; metric = number of keypoints."""
    import cv2

    bgr = _clahe(image)
    gray = _gray(bgr)
    roi_mask = mask.astype(np.uint8) if mask is not None else None
    with timed() as t:
        det = cv2.ORB_create(nfeatures=int(n_features))
        kps = det.detect(gray, roi_mask)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else np.empty((0, 2), np.float32)
    img = _darken(bgr, 0.5)
    kp_img = cv2.drawKeypoints(img, kps, None, color=(0, 200, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    _legend(kp_img, [((0, 200, 255), f"ORB keypoint  n={len(pts)}")])
    _summary(kp_img, f"ORB: {len(pts)} keypoint(s) (oriented FAST corners + BRIEF). "
                     "Scale/rotation-aware, patent-free.")
    return _finish("features.orb", FAM_CORNERS, "ORB keypoints",
                   "Rublee et al. 2011 (ORB), ICCV",
                   metric_name="n_keypoints", metric_value=float(len(pts)), overlay=kp_img,
                   infer_ms=t.ms)


# --- TEXTURE ----------------------------------------------------------------------------
def gabor_bank(image: Any, *, mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """Gabor filter bank at 4 orientations; metric = dominant orientation (deg)."""
    import cv2

    bgr = _clahe(image)
    gray = _gray(bgr).astype(np.float32)
    thetas = [0.0, 45.0, 90.0, 135.0]
    with timed() as t:
        responses = []
        means = []
        for th in thetas:
            k = cv2.getGaborKernel((21, 21), 4.0, np.radians(th), 10.0, 0.5, 0.0, ktype=cv2.CV_32F)
            r = np.abs(cv2.filter2D(gray, cv2.CV_32F, k))
            responses.append(r)
            vals = r[mask] if (mask is not None and mask.any()) else r
            means.append(float(vals.mean()))
        stack = np.stack(responses, axis=0)
        maxresp = stack.max(axis=0)
        dom_idx = int(np.argmax(means))
        dom = float(thetas[dom_idx])
    heat = cv2.applyColorMap((_norm01(maxresp) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img = cv2.addWeighted(_darken(bgr, 0.4), 0.45, heat, 0.55, 0)
    _legend(img, [((0, 255, 255), f"Gabor max response; dominant {dom:.0f}deg")])
    _summary(img, f"Gabor bank (0/45/90/135deg): dominant texture orientation {dom:.0f}deg. "
                  "Max-response energy map over the belt texture.")
    return _finish("features.gabor", FAM_TEXTURE, "Gabor filter bank",
                   "Gabor 1946; Daugman 1985 (Gabor texture energy)",
                   metric_name="dominant_orientation_deg", metric_value=dom, overlay=img,
                   infer_ms=t.ms, web_drivable=False,
                   extra={"orientation_response_mean": [round(m, 3) for m in means],
                          "orientations_deg": thetas})


def lbp(image: Any, *, mask: np.ndarray | None = None, points: int = 8, radius: int = 1, **_: Any) -> dict[str, Any]:
    """Local Binary Pattern texture map; metric = Shannon entropy of the LBP histogram."""
    import cv2
    from skimage.feature import local_binary_pattern

    bgr = _clahe(image)
    gray = _gray(bgr)
    with timed() as t:
        codes = local_binary_pattern(gray, int(points), int(radius), method="uniform")
        n_bins = int(points) + 2
        vals = codes[mask] if (mask is not None and mask.any()) else codes.ravel()
        hist, _ = np.histogram(vals, bins=n_bins, range=(0, n_bins), density=True)
        nz = hist[hist > 0]
        entropy = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
    lbp8 = (codes / max(float(codes.max()), 1.0) * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(lbp8, cv2.COLORMAP_JET)
    img = cv2.addWeighted(_darken(bgr, 0.35), 0.4, heat, 0.6, 0)
    _legend(img, [((0, 255, 255), f"LBP code map (uniform P={points},R={radius})")])
    _summary(img, f"Local Binary Pattern: texture entropy {entropy:.2f} bits "
                  f"(uniform P={points}, R={radius}). Higher = more varied micro-texture.")
    return _finish("features.lbp", FAM_TEXTURE, "Local Binary Pattern",
                   "Ojala et al. 2002 (uniform LBP), TPAMI 24(7)",
                   metric_name="lbp_entropy", metric_value=entropy, overlay=img, infer_ms=t.ms,
                   web_drivable=False, extra={"points": int(points), "radius": int(radius)})


# --- CONSOLIDATED STRAIGHT-LINE GEOMETRY ANALYSIS ---------------------------------------
def geometry_analysis(image: Any, *, view_type: str | None = None,
                      mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """One consolidated straight-line geometry read: the corrected centreline + two straight
    edges + OBB, cross-checked against Hough, a RANSAC boundary-line fit and Radon, drawn on
    a single legible overlay with numeric labels. Never withholds under low confidence - it
    shows the estimate with a confidence note."""
    import cv2

    from ..render import geometry_analysis_overlay
    from .beltline import compute_belt_geometry
    from .semantic import compute_layers

    bgr = _clahe(image)
    gray = _gray(bgr)
    h, w = gray.shape
    with timed() as t:
        layers = compute_layers(image, view_type=view_type, use_learned=False)
        footprint = (layers.belt_mask | layers.content_mask) if mask is None else mask
        geo = compute_belt_geometry(footprint, external_mask=layers.mask(0), gray=gray)

        # OBB of the footprint
        obb_d: dict[str, Any] | None = None
        ys, xs = np.nonzero(footprint)
        if xs.size >= 3:
            rect = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.int32))
            (cxr, cyr), (rw, rh), ang = rect
            obb_d = {"angle_deg": round(float(ang), 2), "width_px": round(float(rw), 1),
                     "height_px": round(float(rh), 1),
                     "box_points": [[int(x), int(y)] for x, y in cv2.boxPoints(rect)]}

        # Hough dominant axis on masked Canny
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        dil = cv2.dilate(footprint.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        edges[dil == 0] = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=int(0.18 * min(h, w)), maxLineGap=16)
        h_ang: float | None = None
        n_lines = 0
        if lines is not None:
            arr = np.asarray(lines).reshape(-1, 4)
            angs = np.degrees(np.arctan2(arr[:, 3] - arr[:, 1], arr[:, 2] - arr[:, 0])) % 180.0
            lens = np.hypot(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])
            n_lines = int(arr.shape[0])
            h_ang = _axis_circular_mean(angs, lens)

        # RANSAC boundary-line fit + Radon
        ransac_fit = _fit_boundary_lines(footprint)
        radon_ang, radon_strength = _radon_orientation(gray)

    hough = {"axis_deg": (round(h_ang, 2) if h_ang is not None else None), "n_lines": n_lines}
    r_a = (ransac_fit or {}).get("line_a")
    r_b = (ransac_fit or {}).get("line_b")
    ransac_line = {
        "line_a_deg": (r_a["angle_deg"] if r_a else None),
        "line_b_deg": (r_b["angle_deg"] if r_b else None),
        "inlier_frac": round(float(np.mean([x["inlier_frac"] for x in (r_a, r_b) if x])), 3)
        if (r_a or r_b) else None,
    }
    radon = {"orientation_deg": round(radon_ang, 2), "strength": round(radon_strength, 3)}

    orientation = geo.get("axis_angle_deg")
    confidence = geo.get("confidence", "low")
    payload_geo: dict[str, Any] = {
        "name": "Consolidated belt geometry",
        "family": "geometry",
        "confidence": confidence,
        "axis_source": geo.get("axis_source"),
        "orientation_deg": orientation,
        "orientation_label": geo.get("orientation"),
        "deviation_from_vertical_deg": geo.get("deviation_from_vertical_deg"),
        "centreline_xy": geo.get("centreline_xy"),
        "edge_a_xy": geo.get("edge_a_xy"),
        "edge_b_xy": geo.get("edge_b_xy"),
        "belt_width_px": geo.get("mean_width_px"),
        "width_std_px": geo.get("width_std_px"),
        "edge_parallelism_deg": geo.get("tangent_skew_deg"),
        "curved": geo.get("curved"),
        "curvature": geo.get("curvature"),
        "straight": (None if geo.get("curved") is None else (not geo.get("curved"))),
        "obb": obb_d,
        "hough": hough,
        "ransac_line": ransac_line,
        "ransac_a": r_a,
        "ransac_b": r_b,
        "radon": radon,
        "support_axis_deg": geo.get("support_axis_deg"),
        "misalignment_deg": geo.get("misalignment_deg"),
    }
    overlay = geometry_analysis_overlay(bgr, payload_geo)

    metric_val = float(orientation) if orientation is not None else 0.0
    res = result("geometry.analysis", "geometry", _TIER,
                 "Consolidated straight-line belt geometry (principal-axis centreline + "
                 "least-squares straight edges) cross-checked with Hough / RANSAC-line / Radon",
                 payload={**payload_geo, "metric_name": "orientation_deg",
                          "metric_value": round(metric_val, 3)},
                 model_bytes=0, infer_ms=t.ms, web_drivable=True)
    res["overlay_b64"] = _to_b64(overlay)
    return res


# --- registry-facing method table + run_all ---------------------------------------------
# id -> (callable, family, human name). The order is the display order in the toolbox.
_FEATURE_FUNCS: dict[str, tuple[Any, str, str]] = {
    "features.canny": (canny, FAM_EDGE, "Canny"),
    "features.sobel": (sobel_magnitude, FAM_EDGE, "Sobel magnitude"),
    "features.scharr": (scharr, FAM_EDGE, "Scharr magnitude"),
    "features.laplacian": (laplacian, FAM_EDGE, "Laplacian"),
    "features.log": (laplacian_of_gaussian, FAM_EDGE, "Laplacian-of-Gaussian"),
    "features.prewitt": (prewitt, FAM_EDGE, "Prewitt"),
    "features.roberts": (roberts_cross, FAM_EDGE, "Roberts cross"),
    "features.morph_gradient": (morphological_gradient, FAM_EDGE, "Morphological gradient"),
    "features.hough_lines_p": (hough_lines_p, FAM_LINES, "HoughLinesP"),
    "features.ransac_lines": (ransac_lines, FAM_LINES, "RANSAC straight-line boundary fit"),
    "features.radon_orientation": (radon_orientation, FAM_LINES, "Radon orientation"),
    "features.slic": (slic_superpixels, FAM_SUPERPIXEL, "SLIC superpixels"),
    "features.obb": (obb, FAM_SHAPE, "Oriented bounding box"),
    "features.contours": (contours, FAM_SHAPE, "External contours"),
    "features.harris": (harris, FAM_CORNERS, "Harris corners"),
    "features.shi_tomasi": (shi_tomasi, FAM_CORNERS, "Shi-Tomasi good features"),
    "features.orb": (orb, FAM_CORNERS, "ORB keypoints"),
    "features.gabor": (gabor_bank, FAM_TEXTURE, "Gabor filter bank"),
    "features.lbp": (lbp, FAM_TEXTURE, "Local Binary Pattern"),
}


def feature_ids() -> list[str]:
    """The feature-toolbox method ids, in display order."""
    return list(_FEATURE_FUNCS)


def run_all(image: Any, *, mask: np.ndarray | None = None,
            methods: list[str] | None = None) -> dict[str, Any]:
    """Run the whole classical feature/edge bench once and return per-method overlays+metrics.

    The belt footprint mask is segmented ONCE and shared across every mask-relevant operator.
    Returns ``{"methods": [{id, name, family, tier, reference, metric_name, metric_value,
    overlay_b64}, ...]}``.
    """
    ids = methods if methods is not None else feature_ids()
    unknown = [m for m in ids if m not in _FEATURE_FUNCS]
    if unknown:
        raise KeyError(f"unknown feature method(s) {unknown}; known: {feature_ids()}")
    shared_mask = mask if mask is not None else _belt_mask(image)

    out: list[dict[str, Any]] = []
    for mid in ids:
        fn, _fam, _name = _FEATURE_FUNCS[mid]
        res = fn(image, mask=shared_mask)
        out.append({
            "id": res["method"],
            "name": res["name"],
            "family": res["family"],
            "tier": res["tier"],
            "reference": res["reference"],
            "metric_name": res["metric_name"],
            "metric_value": res["metric_value"],
            "overlay_b64": res.get("overlay_b64"),
        })
    return {"methods": out, "n_methods": len(out)}


__all__ = [
    "FAM_EDGE", "FAM_LINES", "FAM_SUPERPIXEL", "FAM_SHAPE", "FAM_CORNERS", "FAM_TEXTURE",
    "canny", "sobel_magnitude", "scharr", "laplacian", "laplacian_of_gaussian",
    "prewitt", "roberts_cross", "morphological_gradient",
    "hough_lines_p", "ransac_lines", "radon_orientation",
    "slic_superpixels", "obb", "contours",
    "harris", "shi_tomasi", "orb", "gabor_bank", "lbp",
    "geometry_analysis", "run_all", "feature_ids",
]
