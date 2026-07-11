"""The LIVE-tier computer-vision method ladder.

A clean method API over the six capabilities plus mandatory preprocessing. Each method is
a typed function ``fn(image, **params) -> dict`` returning a JSON-safe result whose
envelope carries the measured gate inputs and the lane the gate assigned (see
``methods._common``). Learned methods degrade to ``{"status": "weights_absent", ...}`` when
an optional weight is missing, never raising.

Public surface:
- ``REGISTRY``: method-id -> :class:`MethodSpec` (callable + tier + capability + reference).
- ``run(method_id, image, **params)``: dispatch a single method by id.
- ``run_ladder(image, **params)``: run every registered method once (the full live ladder).
- ``list_methods`` / ``methods_by_capability`` / ``learned_methods``: registry queries.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.gate import classify_lane
from ..core.manifest import build_method_result
from . import (
    anomaly,
    beltline,
    detection,
    features,
    geometry,
    granulometry,
    preprocess,
    segmentation,
    semantic,
    tracking,
)
from ._common import ENVELOPE_KEYS

# The maturity TIER vocabulary the front end groups by (distinct from the per-method
# COMPUTE tier {classical,learned,foundation} that each result envelope carries for the
# lane/manifest, and distinct from the measured lane). A method's maturity tier answers
# "how advanced is the technique?":
#   - "classical"    hand-engineered CV / statistics (edges, lines, shape, texture, PSD,
#                    optical flow, Kalman): no learned weight, decades-proven.
#   - "sota"         modern learned / deep methods that are the current production standard
#                    (the trained belt segmenter, MobileSAM, PaDiM/Conv-AE anomaly, ONNX
#                    detector).
#   - "beyond_sota"  open-vocabulary foundation-model frontier (GroundedSAM open-vocab
#                    relabelling, DINOv2 / AnomalyDINO) - registered only when present.
TIERS = ("classical", "sota", "beyond_sota")


@dataclass(frozen=True)
class MethodSpec:
    """A registered method: its callable and its documentation axes."""

    method_id: str
    capability: str
    tier: str  # maturity tier: "classical" | "sota" | "beyond_sota" (see TIERS)
    fn: Callable[..., dict[str, Any]]
    reference: str
    summary: str = ""
    family: str = ""  # fine-grained sub-group within a capability (for the toolbox UI)


def _spec(method_id, capability, tier, fn, reference, summary="", family="") -> MethodSpec:
    return MethodSpec(method_id, capability, tier, fn, reference, summary, family)


# The ladder, in capability order. Preprocess first (mandatory), then the capabilities.
# The 3rd argument is the MATURITY tier (classical | sota | beyond_sota) the UI groups by.
_SPECS: tuple[MethodSpec, ...] = (
    _spec("preprocess.clahe_lab", "preprocess", "classical", preprocess.clahe_lab,
          "CLAHE LAB-L clip2.0/8x8 + bilateral + dark-channel haze",
          "Mandatory first stage: dust-robust CLAHE + denoise + haze severity.",
          family="preprocess"),
    _spec("geometry.hough_edges", "geometry", "classical", geometry.hough_edges,
          "Canny 1986; OpenCV HoughLinesP", "Straight-line edge candidates.",
          family="lines_boundaries"),
    _spec("geometry.belt_geometry", "geometry", "classical", beltline.belt_geometry,
          "Principal-axis + medial line from the segmented belt mask; Hough support axis",
          "Orientation-agnostic belt axis, centreline, edges, width and alignment from the mask.",
          family="belt_geometry"),
    _spec("geometry.analysis", "geometry", "classical", features.geometry_analysis,
          "Consolidated straight-line belt geometry (PCA centreline + least-squares straight "
          "edges) cross-checked with Hough / RANSAC-line / Radon",
          "One legible geometry read: orientation, straight centreline + two straight edges, "
          "OBB, width, parallelism, and the Hough/RANSAC-line/Radon angle cross-check.",
          family="belt_geometry"),
    _spec("geometry.radon_orientation", "geometry", "classical", geometry.radon_orientation,
          "Radon transform; skimage.radon", "Noise-robust dominant belt orientation.",
          family="lines_boundaries"),
    _spec("geometry.kalman_edge", "geometry", "classical", geometry.kalman_edge,
          "Kalman 1960", "Per-camera constant-velocity edge smoothing (wander trend).",
          family="belt_geometry"),
    _spec("geometry.obb", "geometry", "classical", geometry.obb,
          "OpenCV minAreaRect", "Oriented bounding boxes: belt plus per region.",
          family="shape"),
    _spec("granulometry.watershed_psd", "granulometry", "classical", granulometry.watershed_psd,
          "Watershed granulometry; Rosin-Rammler ISO 9276-1",
          "Watershed PSD -> D10/D50/D80, oversize%, Rosin-Rammler fit.",
          family="granulometry"),
    _spec("segmentation.semantic_layers", "segmentation", "sota", semantic.semantic_layers,
          "SAM/MobileSAM (Apache-2.0) + CLIP open-vocab; trained SegFormer-B0 segmenter; "
          "classical colour/texture prior fallback",
          "4-class semantic backbone: belt / content / foreign / external (never weights_absent).",
          family="segmentation"),
    _spec("segmentation.slic", "segmentation", "classical", segmentation.slic,
          "Achanta et al. 2012 (SLIC), TPAMI 34(11)", "SLIC superpixel over-segmentation.",
          family="superpixels"),
    _spec("segmentation.mobile_sam", "segmentation", "sota", segmentation.mobile_sam,
          "MobileSAM (Tiny-ViT 5M, Apache-2.0); FastSAM",
          "Automatic mask generation (learned, [dl]); graceful weights_absent.",
          family="segmentation"),
    _spec("anomaly.padim_lite", "anomaly", "sota", anomaly.padim_lite,
          "PaDiM arXiv:2011.08785", "Per-patch Gaussian + Mahalanobis heatmap (live CPU).",
          family="anomaly"),
    _spec("anomaly.conv_ae", "anomaly", "sota", anomaly.conv_ae,
          "Bergmann et al. 2019 (MVTec AD AE baseline), CVPR 2019",
          "Conv-AE reconstruction anomaly: arch + ONNX/torch inference; weights_absent.",
          family="anomaly"),
    _spec("detection.onnx_detector", "detection", "sota", detection.onnx_detector,
          "RT-DETR arXiv:2304.08069 (Apache-2.0)",
          "ONNX Runtime CPU detector -> boxes+scores+labels; graceful weights_absent.",
          family="detection"),
    _spec("tracking.optical_flow", "tracking", "classical", tracking.optical_flow,
          "Farneback 2003 (SCIA)", "Dense optical-flow belt speed + motion direction.",
          family="tracking"),
    _spec("tracking.bytetrack_associate", "tracking", "classical", tracking.bytetrack_associate,
          "ByteTrack arXiv:2110.06864 (MIT)",
          "ByteTrack-style associator over detector boxes (detector is the cost).",
          family="tracking"),
    # --- the classical feature/edge/keypoint/texture toolbox (all classical maturity) ---
    _spec("features.canny", "features", "classical", features.canny,
          "Canny 1986 (hysteresis edge detector)", "Canny edges; metric edge density.",
          family=features.FAM_EDGE),
    _spec("features.sobel", "features", "classical", features.sobel_magnitude,
          "Sobel-Feldman gradient operator", "Sobel gradient magnitude; edge density.",
          family=features.FAM_EDGE),
    _spec("features.scharr", "features", "classical", features.scharr,
          "Scharr 2000 (rotation-optimal 3x3)", "Scharr gradient magnitude; edge density.",
          family=features.FAM_EDGE),
    _spec("features.laplacian", "features", "classical", features.laplacian,
          "Laplacian second-derivative operator", "Laplacian response; edge density.",
          family=features.FAM_EDGE),
    _spec("features.log", "features", "classical", features.laplacian_of_gaussian,
          "Marr & Hildreth 1980 (LoG)", "Laplacian-of-Gaussian; edge density.",
          family=features.FAM_EDGE),
    _spec("features.prewitt", "features", "classical", features.prewitt,
          "Prewitt 1970 gradient operator", "Prewitt gradient magnitude; edge density.",
          family=features.FAM_EDGE),
    _spec("features.roberts", "features", "classical", features.roberts_cross,
          "Roberts 1963 cross-gradient", "Roberts cross gradient; edge density.",
          family=features.FAM_EDGE),
    _spec("features.morph_gradient", "features", "classical", features.morphological_gradient,
          "Morphological gradient (dilation - erosion)", "Morphological gradient; edge density.",
          family=features.FAM_EDGE),
    _spec("features.hough_lines_p", "features", "classical", features.hough_lines_p,
          "Matas et al. 2000 (progressive probabilistic Hough)",
          "HoughLinesP straight segments; metric #lines + dominant angle.",
          family=features.FAM_LINES),
    _spec("features.ransac_lines", "features", "classical", features.ransac_lines,
          "Fischler & Bolles 1981 (RANSAC); skimage LineModelND",
          "RANSAC straight-line fit of the two belt-mask boundaries; angles + inlier frac.",
          family=features.FAM_LINES),
    _spec("features.radon_orientation", "features", "classical", features.radon_orientation,
          "Radon transform (skimage.transform.radon)",
          "Radon dominant orientation (angle + strength).", family=features.FAM_LINES),
    _spec("features.slic", "features", "classical", features.slic_superpixels,
          "Achanta et al. 2012 (SLIC), TPAMI 34(11)",
          "SLIC superpixels; metric #superpixels.", family=features.FAM_SUPERPIXEL),
    _spec("features.obb", "features", "classical", features.obb,
          "OpenCV minAreaRect (rotating-calipers OBB)",
          "Oriented bounding box of the belt mask; angle/w/h.", family=features.FAM_SHAPE),
    _spec("features.contours", "features", "classical", features.contours,
          "Suzuki & Abe 1985; OpenCV findContours",
          "External belt-region contours; metric #contours.", family=features.FAM_SHAPE),
    _spec("features.harris", "features", "classical", features.harris,
          "Harris & Stephens 1988", "Harris corners; metric #corners.",
          family=features.FAM_CORNERS),
    _spec("features.shi_tomasi", "features", "classical", features.shi_tomasi,
          "Shi & Tomasi 1994 (Good Features to Track), CVPR",
          "Shi-Tomasi good features; metric #features.", family=features.FAM_CORNERS),
    _spec("features.orb", "features", "classical", features.orb,
          "Rublee et al. 2011 (ORB), ICCV", "ORB keypoints; metric #keypoints.",
          family=features.FAM_CORNERS),
    _spec("features.gabor", "features", "classical", features.gabor_bank,
          "Gabor 1946; Daugman 1985 (Gabor texture energy)",
          "Gabor filter bank; metric dominant orientation.", family=features.FAM_TEXTURE),
    _spec("features.lbp", "features", "classical", features.lbp,
          "Ojala et al. 2002 (uniform LBP), TPAMI 24(7)",
          "Local Binary Pattern map; metric texture entropy.", family=features.FAM_TEXTURE),
)

REGISTRY: dict[str, MethodSpec] = {s.method_id: s for s in _SPECS}


def list_methods() -> list[str]:
    """Registered method ids, in ladder order."""
    return [s.method_id for s in _SPECS]


def methods_by_capability() -> dict[str, list[str]]:
    """Method ids grouped by capability."""
    out: dict[str, list[str]] = {}
    for s in _SPECS:
        out.setdefault(s.capability, []).append(s.method_id)
    return out


def methods_by_tier() -> dict[str, list[str]]:
    """Method ids grouped by MATURITY tier (classical / sota / beyond_sota)."""
    out: dict[str, list[str]] = {t: [] for t in TIERS}
    for s in _SPECS:
        out.setdefault(s.tier, []).append(s.method_id)
    return out


def families() -> dict[str, list[str]]:
    """Method ids grouped by fine-grained family (the toolbox sub-groups)."""
    out: dict[str, list[str]] = {}
    for s in _SPECS:
        out.setdefault(s.family or s.capability, []).append(s.method_id)
    return out


def method_index() -> list[dict[str, str]]:
    """A flat, JSON-safe catalogue of every registered method for the front end to group by.

    Each entry: ``{id, capability, tier, family, reference, summary}``. The UI groups by
    ``tier`` (classical / sota / beyond_sota) and/or ``family`` to build the method toolbox.
    """
    return [
        {"id": s.method_id, "capability": s.capability, "tier": s.tier,
         "family": s.family or s.capability, "reference": s.reference, "summary": s.summary}
        for s in _SPECS
    ]


def learned_methods() -> list[str]:
    """Method ids that use a learned/deep model (maturity tier sota or beyond_sota)."""
    return [s.method_id for s in _SPECS if s.tier in ("sota", "beyond_sota")]


def run(method_id: str, image: np.ndarray, **params: Any) -> dict[str, Any]:
    """Dispatch one method by id on a BGR frame. Returns its JSON-safe result dict."""
    if method_id not in REGISTRY:
        raise KeyError(f"unknown method {method_id!r}; known: {list_methods()}")
    return REGISTRY[method_id].fn(image, **params)


def run_ladder(image: np.ndarray, **params: Any) -> dict[str, dict[str, Any]]:
    """Run every registered method once on ``image``; return {method_id: result}."""
    return {mid: run(mid, image, **params) for mid in list_methods()}


def to_manifest_method(res: dict[str, Any]) -> dict[str, Any]:
    """Fold a ladder result into a Contract 2 method-result (re-runs the gate on its numbers)."""
    verdict = classify_lane(
        model_bytes=int(res["model_bytes"]),
        infer_ms=float(res["infer_ms"]),
        trace_bytes=int(res["trace_bytes"]),
        web_drivable=bool(res["web_drivable"]),
    )
    metrics = {k: v for k, v in res.items() if k not in ENVELOPE_KEYS}
    metrics["status"] = res["status"]
    return build_method_result(
        method=res["method"],
        capability=res["capability"],
        tier=res["tier"],
        verdict=verdict,
        metrics=metrics,
        reference=res.get("reference"),
        notes=res.get("notes"),
    )


__all__ = [
    "MethodSpec",
    "REGISTRY",
    "TIERS",
    "list_methods",
    "methods_by_capability",
    "methods_by_tier",
    "families",
    "method_index",
    "learned_methods",
    "run",
    "run_ladder",
    "to_manifest_method",
    "features",
]
