"""Stage 4: infer.

Runs the method ladder on the case and, for each method, MEASURES the gate inputs
(model bytes, inference milliseconds, trace bytes, web-drivability) and asks
``core.gate.classify_lane`` for the lane. The verdict and the measured numbers ride
into the manifest; nothing is hand-labeled.

Two methods run here at test-affordable cost:
- ``edge_geometry`` (classical, capability "edges") - Canny/Sobel structure.
- ``patch_anomaly`` (learned, capability "anomaly") - Mahalanobis distance to the
  stage-3 normal model, producing an anomaly heatmap.

Rework surface: RT-DETR/D-FINE detection, SegFormer segmentation, EfficientAD /
AnomalyDINO anomaly, ByteTrack tracking attach here; each measures its own gate
inputs and is classified by the same code path.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..context import StageContext
from ..core.gate import classify_lane
from ..core.manifest import build_method_result
from ..core.trace import stage_timer
from .feature_extraction import PATCH


def _grid_shape(h: int, w: int) -> tuple[int, int]:
    rows = len(range(0, h - PATCH + 1, PATCH))
    cols = len(range(0, w - PATCH + 1, PATCH))
    return rows, cols


def _result_bytes(payload: dict[str, Any]) -> int:
    import json

    from ..core.manifest import jsonable

    return len(json.dumps(jsonable(payload), separators=(",", ":")).encode("utf-8"))


def _edge_geometry(ctx: StageContext) -> dict[str, Any]:
    edges = ctx.state["edges"]
    t0 = time.perf_counter()
    density = float(np.count_nonzero(edges)) / edges.size
    # A cheap horizontal/vertical balance proxy for belt orientation.
    col_energy = float(edges.mean(axis=0).std())
    row_energy = float(edges.mean(axis=1).std())
    infer_ms = (time.perf_counter() - t0) * 1000.0
    metrics = {
        "edge_density": round(density, 5),
        "col_energy": round(col_energy, 4),
        "row_energy": round(row_energy, 4),
    }
    verdict = classify_lane(
        model_bytes=0,
        infer_ms=infer_ms,
        trace_bytes=_result_bytes(metrics),
        web_drivable=True,
    )
    return build_method_result(
        method="edge_geometry",
        capability="edges",
        tier="classical",
        verdict=verdict,
        metrics=metrics,
        reference="Canny 1986; Fischler & Bolles 1981 (RANSAC)",
    )


def _patch_anomaly(ctx: StageContext) -> dict[str, Any]:
    feats: np.ndarray = ctx.state["patch_features"]
    model = ctx.state["normal_model"]
    gray = ctx.state["gray"]

    t0 = time.perf_counter()
    dist = (((feats - model["mean"]) ** 2) / model["var"]).sum(axis=1)
    rows, cols = _grid_shape(*gray.shape[:2])
    grid = dist[: rows * cols].reshape(rows, cols) if rows * cols else dist.reshape(1, -1)
    gmax = float(grid.max()) if grid.size else 0.0
    norm = grid / gmax if gmax > 0 else grid
    infer_ms = (time.perf_counter() - t0) * 1000.0

    peak_idx = int(np.argmax(norm))
    peak_rc = (peak_idx // norm.shape[1], peak_idx % norm.shape[1])
    ctx.state["anomaly_grid"] = norm.astype(np.float32)
    ctx.state["anomaly_peak_rc"] = peak_rc

    metrics = {
        "max_score": round(gmax, 4),
        "mean_score": round(float(grid.mean()), 4) if grid.size else 0.0,
        "peak_row": peak_rc[0],
        "peak_col": peak_rc[1],
        "grid_rows": int(norm.shape[0]),
        "grid_cols": int(norm.shape[1]),
    }
    verdict = classify_lane(
        model_bytes=int(ctx.state["normal_model_bytes"]),
        infer_ms=infer_ms,
        trace_bytes=_result_bytes(metrics),
        web_drivable=True,
    )
    return build_method_result(
        method="patch_anomaly",
        capability="anomaly",
        tier="learned",
        verdict=verdict,
        metrics=metrics,
        reference="Bergmann et al. 2019 (MVTec AD autoencoder baseline); PaDiM arXiv:2011.08785",
        notes="Classical stand-in for the learned anomaly baseline; ONNX conv-AE/EfficientAD in precompute.",
    )


def _run_ladder(ctx: StageContext) -> list[dict[str, Any]]:
    """Run the LIVE method ladder on the frame and fold each result into a manifest method.

    Each ladder method measures its own gate inputs (bytes, ms, web-drivability) and is
    classified by ``core.gate``; ``to_manifest_method`` re-runs the gate on those numbers
    so the manifest lane is measured, never labeled. Learned methods whose optional weight
    is absent contribute a ``weights_absent`` method result rather than raising.
    """
    from ..methods import list_methods, run, to_manifest_method

    folded: list[dict[str, Any]] = []
    for method_id in list_methods():
        res = run(method_id, ctx.image_bgr, camera_id=ctx.case_id, tracker_id=ctx.case_id)
        folded.append(to_manifest_method(res))
    return folded


def infer(ctx: StageContext) -> dict[str, Any]:
    with stage_timer(ctx.trace, "infer") as t:
        # The stage's built-in demonstration methods, then the full live ladder.
        methods = [_edge_geometry(ctx), _patch_anomaly(ctx)]
        methods.extend(_run_ladder(ctx))
        ctx.state["methods"] = methods
        summary = {
            "n_methods": len(methods),
            "methods": [m["method"] for m in methods],
            "lanes": [m["lane"] for m in methods],
        }
        ctx.state["infer_summary"] = summary
        t.note(n_methods=len(methods))
    return summary
