"""View-derived analyses computed FROM the 4-class semantic layers.

Each function takes the CLAHE-first frame plus the relevant class mask(s) from
:mod:`beltvision.methods.semantic` and returns a JSON-safe metrics dict (plus the
geometry an overlay needs). They never invent a result: an analysis that does not
apply to a mask (e.g. mineral on an empty return strand) reports zero/na honestly.

- :func:`belt_damage`        - cuts / rips / tears / holes / wear inside the belt mask.
- :func:`edge_condition`     - fraying / missing chunks along the belt-mask borders.
- :func:`surface_state`      - surface irregularity + dust/haze on the belt.
- :func:`content_quantity`    - coverage %, load and granulometry INSIDE the mineral mask.
- :func:`foreign_objects`    - tramp objects from the FOREIGN class (+ optional detector).
"""
from __future__ import annotations

from typing import Any

import numpy as np

_MB = 1024 * 1024


def _restrict(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = gray.copy()
    out[~mask] = 0
    return out


def _boxes_from_mask(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    import cv2

    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        out.append({"bbox_xywh": [x, y, w, h], "area_px": area,
                    "elongation": round(float(max(w, h) / max(min(w, h), 1)), 2)})
    out.sort(key=lambda d: d["area_px"], reverse=True)
    return out


def belt_damage(bgr: np.ndarray, belt_mask: np.ndarray, **_: Any) -> dict[str, Any]:
    """Detect rips/tears/holes/wear inside the belt region (morphology + gradient + residual)."""
    import cv2

    h, w = bgr.shape[:2]
    belt_area = int(belt_mask.sum())
    if belt_area < 0.01 * h * w:
        return {"applicable": False, "status": "na",
                "reason": "belt region not segmented", "severity": 0.0}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # dark thin structures (rips/cuts) via blackhat; bright scratches via tophat
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    resp = cv2.max(blackhat, tophat).astype(np.float32)
    resp[~belt_mask] = 0.0
    vals = resp[belt_mask]
    thr = float(np.percentile(vals, 99.0)) if vals.size else 0.0
    thr = max(thr, 18.0)
    dmg = (resp >= thr) & belt_mask
    dmg = cv2.morphologyEx(dmg.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0

    min_area = max(40, int(0.0012 * belt_area))
    boxes = _boxes_from_mask(dmg, min_area)
    # severity is driven by the FLAGGED REGIONS (localized rips/holes), not diffuse dust
    # texture: with no significant region the belt is intact (severity ~ 0), never a
    # contradictory "0 regions but moderate".
    flagged_area = int(sum(b["area_px"] for b in boxes))
    damaged_frac = flagged_area / max(belt_area, 1)
    elong = max((b["elongation"] for b in boxes), default=0.0)
    if not boxes:
        severity = 0.0
    else:
        severity = float(np.clip(0.7 * min(damaged_frac * 60.0, 1.0)
                                 + 0.3 * min(elong / 6.0, 1.0), 0.0, 1.0))
    label = ("none" if severity < 0.1 else "minor" if severity < 0.4
             else "moderate" if severity < 0.7 else "severe")
    return {
        "applicable": True, "status": "ok",
        "belt_area_px": belt_area,
        "damaged_area_px": flagged_area,
        "damaged_frac_of_belt": round(damaged_frac, 4),
        "n_damage_regions": len(boxes),
        "regions": boxes[:24],
        "severity": round(severity, 3),
        "severity_label": label,
    }


def edge_condition(
    belt_mask: np.ndarray, edge_a: list | None, edge_b: list | None, **_: Any
) -> dict[str, Any]:
    """Border condition: roughness / notches along each belt edge vs a smooth edge."""
    if not edge_a or not edge_b or len(edge_a) < 5:
        return {"applicable": False, "status": "na",
                "reason": "belt edges not available (geometry low confidence)"}

    def _roughness(edge: list) -> dict[str, Any]:
        e = np.asarray(edge, dtype=np.float64)
        # residual of edge points from a smoothed (moving-average) edge
        sx = np.convolve(e[:, 0], np.ones(5) / 5, mode="same")
        sy = np.convolve(e[:, 1], np.ones(5) / 5, mode="same")
        res = np.hypot(e[:, 0] - sx, e[:, 1] - sy)
        res[:2] = res[-2:] = 0.0  # ignore convolution edge effects
        rough = float(np.std(res))
        notches = int(np.sum(res > (np.mean(res) + 3.0 * np.std(res) + 1.5)))
        return {"roughness_px": round(rough, 2), "notches": notches,
                "max_deviation_px": round(float(res.max()), 2)}

    ra, rb = _roughness(edge_a), _roughness(edge_b)
    worst = max(ra["roughness_px"], rb["roughness_px"])
    frayed = bool(worst > 3.0 or ra["notches"] + rb["notches"] > 2)
    return {
        "applicable": True, "status": "ok",
        "edge_a": ra, "edge_b": rb,
        "worst_roughness_px": round(worst, 2),
        "frayed_or_chunks": frayed,
        "verdict": "border damage / fraying" if frayed else "borders smooth",
    }


def surface_state(bgr: np.ndarray, belt_mask: np.ndarray, **_: Any) -> dict[str, Any]:
    """Surface irregularity (texture non-uniformity) + dust/haze on the belt."""
    import cv2

    from .preprocess import haze_severity

    haze = haze_severity(bgr)
    h, w = bgr.shape[:2]
    if belt_mask.sum() < 0.01 * h * w:
        return {"applicable": True, "status": "ok", "haze": haze,
                "surface_uniformity": None,
                "reason_surface": "belt region too small for a surface reading"}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    belt_vals = lap[belt_mask]
    irregularity = float(np.std(belt_vals))
    uniformity = float(np.clip(1.0 - irregularity / 40.0, 0.0, 1.0))
    return {
        "applicable": True, "status": "ok",
        "haze": haze,
        "surface_irregularity_lapstd": round(irregularity, 2),
        "surface_uniformity": round(uniformity, 3),
        "flagged_irregular": bool(uniformity < 0.4),
    }


def content_quantity(
    bgr: np.ndarray, belt_mask: np.ndarray, content_mask: np.ndarray,
    *, px_per_mm: float | None = None, **_: Any,
) -> dict[str, Any]:
    """Coverage %, load and granulometry (PSD) INSIDE the mineral mask only."""
    from .granulometry import psd_from_mask

    h, w = bgr.shape[:2]
    content_area = int(content_mask.sum())
    # the belt FOOTPRINT is exposed belt + whatever content sits on top of it; content
    # coverage is content over that footprint (so a fully-loaded belt reads ~100%, not 0
    # when little bare belt shows).
    footprint_area = int((belt_mask | content_mask).sum())
    belt_area = footprint_area
    coverage = (content_area / footprint_area) if footprint_area > 0 else 0.0
    out: dict[str, Any] = {
        "applicable": True, "status": "ok",
        "content_area_px": content_area,
        "belt_area_px": belt_area,
        "coverage_frac_of_belt": round(coverage, 4),
        "coverage_pct": round(100.0 * coverage, 1),
        "load_label": ("empty" if coverage < 0.05 else "light" if coverage < 0.3
                       else "moderate" if coverage < 0.7 else "heavy"),
    }
    if content_area >= 0.01 * h * w:
        out["granulometry"] = psd_from_mask(bgr, content_mask, px_per_mm=px_per_mm)
    else:
        out["granulometry"] = {"n_particles": 0,
                               "note": "content coverage ~0 (empty belt) - PSD not computed"}
    return out


def foreign_objects(
    bgr: np.ndarray, foreign_mask: np.ndarray, belt_mask: np.ndarray,
    *, detections: list | None = None, **_: Any,
) -> dict[str, Any]:
    """Foreign / unexpected objects from the FOREIGN class (+ optional detector boxes)."""
    h, w = bgr.shape[:2]
    min_area = max(30, int(0.0006 * h * w))
    boxes = _boxes_from_mask(foreign_mask, min_area)
    on_belt = 0
    for b in boxes:
        x, y, bw, bh = b["bbox_xywh"]
        cx, cy = x + bw // 2, y + bh // 2
        b["on_belt"] = bool(belt_mask[min(cy, h - 1), min(cx, w - 1)])
        on_belt += int(b["on_belt"])
    det = detections or []
    return {
        "applicable": True, "status": "ok",
        "n_foreign_regions": len(boxes),
        "n_on_belt": on_belt,
        "regions": boxes[:24],
        "detector_boxes": det[:24],
        "verdict": ("foreign object(s) present" if boxes else "no foreign object detected"),
    }
