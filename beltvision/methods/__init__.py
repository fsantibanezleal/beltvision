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
    geometry,
    granulometry,
    preprocess,
    segmentation,
    semantic,
    tracking,
)
from ._common import ENVELOPE_KEYS


@dataclass(frozen=True)
class MethodSpec:
    """A registered method: its callable and its documentation axes."""

    method_id: str
    capability: str
    tier: str  # "classical" | "learned" | "foundation"
    fn: Callable[..., dict[str, Any]]
    reference: str
    summary: str = ""


def _spec(method_id, capability, tier, fn, reference, summary="") -> MethodSpec:
    return MethodSpec(method_id, capability, tier, fn, reference, summary)


# The ladder, in capability order. Preprocess first (mandatory), then the six capabilities.
_SPECS: tuple[MethodSpec, ...] = (
    _spec("preprocess.clahe_lab", "preprocess", "classical", preprocess.clahe_lab,
          "CLAHE LAB-L clip2.0/8x8 + bilateral + dark-channel haze",
          "Mandatory first stage: dust-robust CLAHE + denoise + haze severity."),
    _spec("geometry.hough_edges", "geometry", "classical", geometry.hough_edges,
          "Canny 1986; OpenCV HoughLinesP", "Straight-line edge candidates."),
    _spec("geometry.belt_geometry", "geometry", "classical", beltline.belt_geometry,
          "Principal-axis + medial line from the segmented belt mask; Hough support axis",
          "Orientation-agnostic belt axis, centreline, edges, width and alignment from the mask."),
    _spec("geometry.radon_orientation", "geometry", "classical", geometry.radon_orientation,
          "Radon transform; skimage.radon", "Noise-robust dominant belt orientation."),
    _spec("geometry.kalman_edge", "geometry", "classical", geometry.kalman_edge,
          "Kalman 1960", "Per-camera constant-velocity edge smoothing (wander trend)."),
    _spec("geometry.obb", "geometry", "classical", geometry.obb,
          "OpenCV minAreaRect", "Oriented bounding boxes: belt plus per region."),
    _spec("granulometry.watershed_psd", "granulometry", "classical", granulometry.watershed_psd,
          "Watershed granulometry; Rosin-Rammler ISO 9276-1",
          "Watershed PSD -> D10/D50/D80, oversize%, Rosin-Rammler fit."),
    _spec("segmentation.semantic_layers", "segmentation", "learned", semantic.semantic_layers,
          "SAM/MobileSAM (Apache-2.0) + CLIP open-vocab; classical colour/texture prior fallback",
          "4-class semantic backbone: belt / content / foreign / external (never weights_absent)."),
    _spec("segmentation.slic", "segmentation", "classical", segmentation.slic,
          "Achanta et al. 2012 (SLIC), TPAMI 34(11)", "SLIC superpixel over-segmentation."),
    _spec("segmentation.mobile_sam", "segmentation", "learned", segmentation.mobile_sam,
          "MobileSAM (Tiny-ViT 5M, Apache-2.0); FastSAM",
          "Automatic mask generation (learned, [dl]); graceful weights_absent."),
    _spec("anomaly.padim_lite", "anomaly", "learned", anomaly.padim_lite,
          "PaDiM arXiv:2011.08785", "Per-patch Gaussian + Mahalanobis heatmap (live CPU)."),
    _spec("anomaly.conv_ae", "anomaly", "learned", anomaly.conv_ae,
          "Bergmann et al. 2019 (MVTec AD AE baseline), CVPR 2019",
          "Conv-AE reconstruction anomaly: arch + ONNX/torch inference; weights_absent."),
    _spec("detection.onnx_detector", "detection", "learned", detection.onnx_detector,
          "RT-DETR arXiv:2304.08069 (Apache-2.0)",
          "ONNX Runtime CPU detector -> boxes+scores+labels; graceful weights_absent."),
    _spec("tracking.optical_flow", "tracking", "classical", tracking.optical_flow,
          "Farneback 2003 (SCIA)", "Dense optical-flow belt speed + motion direction."),
    _spec("tracking.bytetrack_associate", "tracking", "classical", tracking.bytetrack_associate,
          "ByteTrack arXiv:2110.06864 (MIT)",
          "ByteTrack-style associator over detector boxes (detector is the cost)."),
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


def learned_methods() -> list[str]:
    """Method ids whose tier is learned/foundation."""
    return [s.method_id for s in _SPECS if s.tier in ("learned", "foundation")]


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
    "list_methods",
    "methods_by_capability",
    "learned_methods",
    "run",
    "run_ladder",
    "to_manifest_method",
]
