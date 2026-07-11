"""Stage 1 + 2 of the view-aware pipeline: recognise the view, then map it to analyses.

A belt is monitored from several VIEW TYPES; each enables a different set of analyses.
The tool is view-aware, so the entry point is automatic view recognition:

- Stage 1 - :func:`recognize_view` predicts the ``view_type`` of any frame from classical
  scene features (material fill in the centre, warmth/texture, dust/haze, dominant
  orientation) and returns a confidence and the per-view scores. It is a scored classical
  classifier (no labels are needed); it must recognise the real COLA 34 frame as
  ``end_return`` with a sensible confidence.
- Stage 2 - :data:`VIEW_ANALYSES` maps a ``view_type`` to the analysis ids that provide
  information for that view (only those are offered/shown for a case).

VIEW TYPES:
- ``end_return``   - lateral/end view of the empty RETURN strand: inspect the belt itself
  (damage, edges, alignment, surface, dust). No mineral/content (the strand is empty).
- ``top_carrying`` - top-down of the loaded belt: analyse the CONTENT (granulometry, load,
  coverage, foreign objects, dust). No belt-cut inspection (material covers the belt).
- ``side_profile`` - side view: profile/sag, alignment, content.
- ``oblique_cctv`` - degraded oblique CCTV: damage + alignment + foreign + dust.
"""
from __future__ import annotations

from typing import Any

import numpy as np

VIEW_TYPES = ("end_return", "top_carrying", "side_profile", "oblique_cctv")

VIEW_LABELS = {
    "end_return": "End / return strand",
    "top_carrying": "Top / carrying (loaded)",
    "side_profile": "Side profile",
    "oblique_cctv": "Oblique CCTV",
}

# Stage 2: view_type -> ordered analysis ids that apply to it.
VIEW_ANALYSES: dict[str, list[str]] = {
    "end_return": ["semantic", "belt_geometry", "damage", "edges", "surface", "dust"],
    "top_carrying": ["semantic", "content", "foreign", "dust"],
    "side_profile": ["semantic", "belt_geometry", "content", "surface"],
    "oblique_cctv": ["semantic", "belt_geometry", "damage", "foreign", "dust"],
}

# Analysis metadata (id -> title + which semantic layer it consumes + one-liner).
ANALYSIS_META: dict[str, dict[str, str]] = {
    "semantic": {"title": "Semantic layers", "layer": "all",
                 "about": "Segments every pixel into belt / content / foreign / external."},
    # (content = the transported material: ore, aggregate, food, packages, recycling...)
    "belt_geometry": {"title": "Belt geometry & alignment", "layer": "belt",
                      "about": "Belt axis, centreline and alignment vs the support structure."},
    "damage": {"title": "Belt damage", "layer": "belt",
               "about": "Cuts, rips, tears, holes and wear inside the belt surface."},
    "edges": {"title": "Edges / borders", "layer": "belt",
              "about": "Fraying and missing chunks along the belt borders."},
    "surface": {"title": "Surface", "layer": "belt",
                "about": "Surface irregularity and dust/haze on the belt."},
    "dust": {"title": "Dust / haze", "layer": "all",
             "about": "Airborne dust and haze severity affecting the view."},
    "content": {"title": "Transported content", "layer": "content",
                "about": "Load coverage and granulometry (PSD) of the material on the belt."},
    "foreign": {"title": "Foreign objects", "layer": "foreign",
                "about": "Unexpected objects (metal, wood, tools, debris) that are not belt or content."},
    "dynamic": {"title": "Dynamic / temporal", "layer": "all",
                "about": "Per-frame tracking, belt speed, drift and events over time."},
}


def analyses_for_view(view_type: str) -> list[str]:
    """The analysis ids applicable to a view (Stage 2)."""
    return list(VIEW_ANALYSES.get(view_type, VIEW_ANALYSES["oblique_cctv"]))


def _center_features(bgr: np.ndarray) -> dict[str, float]:
    import cv2

    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_ch, b_ch = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    warmth = a_ch + 0.4 * b_ch
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # structure-tensor coherence: oriented (belt streaks) vs isotropic (a pile of material)
    kk = 11
    jxx = cv2.blur(gx * gx, (kk, kk))
    jyy = cv2.blur(gy * gy, (kk, kk))
    jxy = cv2.blur(gx * gy, (kk, kk))
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-3)
    y0, y1 = int(0.25 * h), int(0.75 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    c = (slice(y0, y1), slice(x0, x1))
    warm_c = float(np.clip(warmth[c].mean() / 30.0, -1.0, 1.0))
    tex_c = float(np.clip(grad[c].mean() / 40.0, 0.0, 1.0))
    coh_c = float(np.clip(coh[c].mean(), 0.0, 1.0))
    # isotropic textured fill in the centre = a pile of transported material (colour-agnostic,
    # so it works for dark ore, pale aggregate, boxes...). Belt streaks are coherent, not this.
    iso_c = float(np.clip(tex_c * (1.0 - coh_c), 0.0, 1.0))
    vstruct = float(np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-6))
    return {"warm_center": warm_c, "tex_center": tex_c, "coh_center": coh_c,
            "iso_center": iso_c, "vstruct": vstruct}


def recognize_view(image: Any) -> dict[str, Any]:
    """Stage 1: predict the ``view_type`` of a frame (classical scene classifier)."""
    import cv2

    from .methods._common import as_bgr
    from .methods.preprocess import apply_clahe_lab, haze_severity

    raw = as_bgr(image)
    bgr = apply_clahe_lab(raw)
    f = _center_features(bgr)
    # haze is a property of the RAW frame; measuring it post-CLAHE would erase the signal.
    haze = haze_severity(raw)["severity"]
    warm = f["warm_center"]
    tex = f["tex_center"]
    vstruct = f["vstruct"]
    iso = f["iso_center"]     # isotropic material fill in the centre (colour-agnostic)
    coh = f["coh_center"]     # oriented belt streaks in the centre
    # loaded content = the centre is filled with isotropically-textured material (any colour),
    # with a bonus for a warm material tone. Oriented streaks (a bare belt) are NOT content.
    content_fill = iso + 0.3 * max(warm, 0.0)

    # dominant orientation of the scene (structure tensor over the frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ori = float((np.degrees(0.5 * np.arctan2(2 * np.mean(gx * gy),
                 np.mean(gx * gx) - np.mean(gy * gy)))) % 180.0)
    near_horizontal = 1.0 if min(ori, 180 - ori) < 25 else 0.0

    logits = {
        # loaded top view: isotropic material fills the centre (bare belt is coherent, not this)
        "top_carrying": 4.0 * content_fill - 1.0 * coh,
        # empty return strand: oriented streaks / bare belt in the centre, hazy, not filled
        "end_return": 1.5 * coh + 1.3 * haze - 2.5 * content_fill + 0.6,
        # side profile: near-horizontal band, some material, less haze
        "side_profile": 0.9 * near_horizontal + 0.3 * max(warm, 0.0) - 1.0 * content_fill,
        # oblique cctv: baseline fallback
        "oblique_cctv": 0.45,
    }
    keys = list(logits)
    arr = np.array([logits[k] for k in keys], dtype=np.float64)
    ex = np.exp(arr - arr.max())
    probs = ex / ex.sum()
    order = np.argsort(-probs)
    best = keys[int(order[0])]
    conf = float(probs[int(order[0])])
    return {
        "view_type": best,
        "view_label": VIEW_LABELS[best],
        "confidence": round(conf, 3),
        "scores": {k: round(float(p), 3) for k, p in zip(keys, probs, strict=False)},
        "features": {"haze": round(haze, 3), "warm_center": round(warm, 3),
                     "tex_center": round(tex, 3), "vertical_structure": round(vstruct, 3),
                     "dominant_orientation_deg": round(ori, 1)},
    }
