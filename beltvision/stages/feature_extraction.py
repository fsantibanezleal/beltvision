"""Stage 2: feature_extraction.

Classical, transparent features on the CLAHE frame: Canny edges (belt-structure
geometry) and a grid of per-patch intensity/gradient statistics that serve as the
"normal" representation the anomaly baseline is fit to.

Rework surface: DINOv2 patch embeddings, SLIC superpixels, and learned backbones
attach here in the precompute lane. The frozen part is that features are computed
once, deterministically, and cached on the context for the later stages.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..context import StageContext
from ..core.trace import stage_timer

PATCH = 32  # feature grid patch size in pixels


def _edge_map(gray: np.ndarray) -> np.ndarray:
    import cv2

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 50, 150)


def _patch_features(gray: np.ndarray) -> np.ndarray:
    """Return an (n_patches, 4) array: mean, std, grad-mean, grad-std per patch."""
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    h, w = gray.shape
    feats = []
    for y in range(0, h - PATCH + 1, PATCH):
        for x in range(0, w - PATCH + 1, PATCH):
            g = gray[y : y + PATCH, x : x + PATCH].astype(np.float32)
            gm = grad[y : y + PATCH, x : x + PATCH]
            feats.append([g.mean(), g.std(), gm.mean(), gm.std()])
    return np.asarray(feats, dtype=np.float32) if feats else np.zeros((1, 4), np.float32)


def feature_extraction(ctx: StageContext) -> dict[str, Any]:
    import cv2

    with stage_timer(ctx.trace, "feature_extraction") as t:
        clahe = ctx.state["clahe_bgr"]
        gray = cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)
        edges = _edge_map(gray)
        feats = _patch_features(gray)

        ctx.state["gray"] = gray
        ctx.state["edges"] = edges
        ctx.state["patch_features"] = feats

        edge_density = float(np.count_nonzero(edges)) / edges.size
        summary = {
            "edge_density": round(edge_density, 5),
            "n_patches": int(feats.shape[0]),
            "feature_dim": int(feats.shape[1]),
            "patch_px": PATCH,
        }
        ctx.state["feature_summary"] = summary
        t.note(n_patches=int(feats.shape[0]), edge_density=round(edge_density, 4))
    return summary
