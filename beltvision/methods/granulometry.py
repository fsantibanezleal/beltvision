"""Capability 4: granulometry (particle size distribution), classical, LIVE-SERVER.

``granulometry.watershed_psd`` (M20) is the industry-standard Split/WipFrag-lineage
pipeline: CLAHE -> Otsu -> distance transform -> watershed delineation -> regionprops ->
equivalent diameters -> D10/D50/D80 + oversize% + a Rosin-Rammler (Weibull) fit, returning
the cumulative PSD curve. It reports RELATIVE units (px) when no ``px_per_mm`` calibration
is given and never fabricates millimetres (honesty rule).

References: Gonzalez & Woods (morphological granulometry / watershed); Rosin-Rammler
ISO 9276-1; the Split-Online / WipFrag watershed principle.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_bgr, cap, result, timed
from .preprocess import apply_clahe_lab

_MAX_CURVE = 32


def _rosin_rammler(diam: np.ndarray) -> dict[str, Any]:
    """Fit the Rosin-Rammler CDF P(d) = 1 - exp(-(d/dc)^n) to the diameters."""
    from scipy.optimize import curve_fit

    diam = np.sort(diam[diam > 0])
    if diam.size < 4:
        return {"fitted": False, "reason": "too few particles for a stable fit"}
    cdf = (np.arange(1, diam.size + 1) - 0.5) / diam.size

    def model(d: np.ndarray, dc: float, n: float) -> np.ndarray:
        return 1.0 - np.exp(-((d / dc) ** n))

    try:
        p0 = [float(np.median(diam)), 1.5]
        popt, _ = curve_fit(
            model, diam, cdf, p0=p0, bounds=([1e-3, 0.2], [np.inf, 12.0]), maxfev=8000
        )
        pred = model(diam, *popt)
        ss_res = float(np.sum((cdf - pred) ** 2))
        ss_tot = float(np.sum((cdf - cdf.mean()) ** 2)) + 1e-12
        return {
            "fitted": True,
            "characteristic_size_dc": round(float(popt[0]), 3),
            "uniformity_index_n": round(float(popt[1]), 3),
            "r_squared": round(1.0 - ss_res / ss_tot, 4),
        }
    except (RuntimeError, ValueError):
        return {"fitted": False, "reason": "curve_fit did not converge"}


def psd_from_mask(
    bgr: np.ndarray,
    foreground: np.ndarray,
    *,
    px_per_mm: float | None = None,
    oversize_px: float = 30.0,
    min_area_px: int = 12,
) -> dict[str, Any]:
    """Watershed PSD computed ONLY inside a supplied foreground mask (e.g. the mineral layer).

    Same delineation pipeline as :func:`watershed_psd` but the binary foreground is the
    given mask rather than a whole-frame Otsu, so granulometry is measured on the
    segmented material only (never the belt or the background).
    """
    import cv2
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed

    gray = cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = cv2.morphologyEx(
        (foreground > 0).astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    dist = ndi.distance_transform_edt(mask)
    min_dist = max(3, int(0.01 * min(h, w)))
    coords = peak_local_max(dist, min_distance=min_dist, labels=mask)
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    markers, _ = ndi.label(markers > 0)
    labels = watershed(-dist, markers, mask=mask)
    diam_px = [
        float(np.sqrt(4.0 * prop.area / np.pi))
        for prop in regionprops(labels)
        if prop.area >= min_area_px
    ]
    diam = np.asarray(diam_px, dtype=np.float64)
    if diam.size:
        d10, d50, d80 = (float(np.percentile(diam, p)) for p in (10, 50, 80))
        oversize_frac = float(np.mean(diam > oversize_px))
        sorted_d = np.sort(diam)
        cum = (np.arange(1, sorted_d.size + 1)) / sorted_d.size
        step = max(1, sorted_d.size // _MAX_CURVE)
        curve = [[round(float(sorted_d[i]), 3), round(float(cum[i]), 4)]
                 for i in range(0, sorted_d.size, step)]
        rr = _rosin_rammler(diam)
    else:
        d10 = d50 = d80 = 0.0
        oversize_frac = 0.0
        curve = []
        rr = {"fitted": False, "reason": "no particles segmented in mineral mask"}
    unit = "mm" if (px_per_mm and px_per_mm > 0) else "px"
    scale = (1.0 / px_per_mm) if (px_per_mm and px_per_mm > 0) else 1.0
    return {
        "n_particles": int(diam.size),
        "unit": unit,
        "calibration": ("absolute-mm" if unit == "mm"
                        else "relative-px-only (no px_per_mm; mm not fabricated)"),
        "D10": round(d10 * scale, 3),
        "D50": round(d50 * scale, 3),
        "D80": round(d80 * scale, 3),
        "oversize_frac": round(oversize_frac, 4),
        "oversize_threshold_px": float(oversize_px),
        "psd_curve": cap(curve, _MAX_CURVE),
        "rosin_rammler": rr,
    }


def watershed_psd(
    image: Any,
    *,
    px_per_mm: float | None = None,
    oversize_px: float = 30.0,
    min_area_px: int = 12,
    **_: Any,
) -> dict[str, Any]:
    """Watershed granulometry -> D10/D50/D80, oversize%, PSD curve and Rosin-Rammler fit."""
    import cv2
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed

    bgr = apply_clahe_lab(as_bgr(image))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    with timed() as t:
        _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        mask = binary > 0
        dist = ndi.distance_transform_edt(mask)
        min_dist = max(3, int(0.01 * min(h, w)))
        coords = peak_local_max(dist, min_distance=min_dist, labels=mask)
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        markers, _ = ndi.label(markers > 0)
        labels = watershed(-dist, markers, mask=mask)

        diam_px: list[float] = []
        for prop in regionprops(labels):
            if prop.area < min_area_px:
                continue
            # equivalent_diameter_area: diameter of a circle with the region's area.
            diam_px.append(float(np.sqrt(4.0 * prop.area / np.pi)))
        diam = np.asarray(diam_px, dtype=np.float64)

        if diam.size:
            d10, d50, d80 = (float(np.percentile(diam, p)) for p in (10, 50, 80))
            oversize_frac = float(np.mean(diam > oversize_px))
            sorted_d = np.sort(diam)
            cum = (np.arange(1, sorted_d.size + 1)) / sorted_d.size
            step = max(1, sorted_d.size // _MAX_CURVE)
            curve = [
                [round(float(sorted_d[i]), 3), round(float(cum[i]), 4)]
                for i in range(0, sorted_d.size, step)
            ]
            rr = _rosin_rammler(diam)
        else:
            d10 = d50 = d80 = 0.0
            oversize_frac = 0.0
            curve = []
            rr = {"fitted": False, "reason": "no particles segmented"}

    unit = "mm" if (px_per_mm and px_per_mm > 0) else "px"
    scale = (1.0 / px_per_mm) if (px_per_mm and px_per_mm > 0) else 1.0
    payload = {
        "shape": [int(h), int(w)],
        "n_particles": int(diam.size),
        "unit": unit,
        "calibration": (
            "absolute-mm" if unit == "mm" else "relative-px-only (no px_per_mm; mm not fabricated)"
        ),
        "D10": round(d10 * scale, 3),
        "D50": round(d50 * scale, 3),
        "D80": round(d80 * scale, 3),
        "oversize_frac": round(oversize_frac, 4),
        "oversize_threshold_px": float(oversize_px),
        "psd_curve": cap(curve, _MAX_CURVE),
        "rosin_rammler": rr,
    }
    return result(
        "granulometry.watershed_psd", "granulometry", "classical",
        "Watershed granulometry (Gonzalez & Woods; Split/WipFrag); Rosin-Rammler ISO 9276-1",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=False,
    )
