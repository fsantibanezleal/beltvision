"""Offline PRECOMPUTE of the FULL per-method toolbox for one still frame.

For a single catalogue frame this runs every method a precomputed case should be able to
REPLAY with no live gaps, and returns each as a uniform, JSON-safe record carrying a drawn
overlay (base64 PNG data URL), a scalar metric, and its maturity ``tier`` + ``family`` so
the serving layer can group the toolbox:

- the classical feature / edge / keypoint / texture bench (``features.run_all`` - 19
  operators, each its own overlay + metric),
- the consolidated straight-line geometry read (``features.geometry_analysis``),
- the 4-class semantic segmentation map (trained ONNX segmenter + optional open-vocab
  foreign),
- unsupervised anomaly (``padim_lite`` self-reference, the trained conv-AE reconstruction
  residual, and the PaDiM + PatchCore-lite frozen-backbone banks) - each a colour heatmap,
- granulometry (watershed PSD) drawn on the segmented content,
- dense optical-flow motion,
- MobileSAM automatic masks (best-effort).

Overlays for the methods that do not draw their own (anomaly / granulometry / flow /
semantic / SAM) are rendered here via :mod:`beltvision.render`. Torch, onnxruntime and
ultralytics are imported lazily, only for the method that needs them; a method that cannot
run (missing weight, runtime error) is recorded with a ``status`` and skipped, never raising.
GPU is used for the frozen-backbone banks and MobileSAM when ``device='cuda'``.

This module lives in the precompute lane (heavy extras allowed); the slim runtime never
imports it. The serving product persists the returned overlays as compact JPEGs and replays
them (ADR-0014).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .. import render
from ..methods import anomaly, features, granulometry, segmentation, tracking
from ..methods._common import as_bgr
from ..methods.preprocess import apply_clahe_lab
from ..methods.semantic import compute_layers

# grid geometry the committed PaDiM / PatchCore banks were fitted on (see learned_artifacts).
_BANK_INPUT = 256
_BANK_GRID = 8

_CONV_AE_NAME = "conv_ae.onnx"
_PADIM_NAME = "padim_ironore.npz"
_PATCHCORE_NAME = "patchcore_ironore.npz"
_MOBILE_SAM_NAME = "mobile_sam.pt"


def _weights_dir(weights_dir: str | Path | None) -> Path:
    if weights_dir:
        return Path(weights_dir)
    return Path(os.environ.get("BELTVISION_WEIGHTS_DIR", "."))


def _entry(
    *,
    method_id: str,
    capability: str,
    tier: str,
    family: str,
    name: str,
    reference: str,
    metric_name: str,
    metric_value: float | None,
    summary: str,
    overlay_b64: str | None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One uniform per-method record for the serving layer to persist + group."""
    return {
        "id": method_id,
        "capability": capability,
        "tier": tier,
        "family": family,
        "name": name,
        "reference": reference,
        "metric_name": metric_name,
        "metric_value": (None if metric_value is None else round(float(metric_value), 5)),
        "summary": summary,
        "overlay_b64": overlay_b64,
        "status": status,
        "extra": extra or {},
    }


# --- classical bench ---------------------------------------------------------------------
def _classical_features(bgr: np.ndarray, footprint: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    fa = features.run_all(bgr, mask=footprint)
    for m in fa["methods"]:
        out.append(_entry(
            method_id=m["id"], capability="features", tier=m.get("tier", "classical"),
            family=m.get("family", "features"), name=m["name"], reference=m["reference"],
            metric_name=m["metric_name"], metric_value=m["metric_value"],
            summary=f"{m['name']}: {m['metric_name']} = {m['metric_value']}.",
            overlay_b64=m.get("overlay_b64"),
        ))
    return out


def _geometry_analysis(bgr: np.ndarray, footprint: np.ndarray,
                       view_type: str | None) -> dict[str, Any]:
    res = features.geometry_analysis(bgr, view_type=view_type, mask=footprint)
    ori = res.get("orientation_deg")
    return _entry(
        method_id="geometry.analysis", capability="geometry", tier=res.get("tier", "classical"),
        family="belt_geometry", name="Consolidated belt geometry", reference=res["reference"],
        metric_name="orientation_deg", metric_value=res.get("metric_value"),
        summary=(f"Straight-line belt geometry: axis {ori:.1f}deg, width "
                 f"~{res.get('belt_width_px') or 0:.0f}px; Hough/RANSAC-line/Radon cross-check."
                 if ori is not None else "Belt geometry estimate (low confidence)."),
        overlay_b64=res.get("overlay_b64"),
        extra={"confidence": res.get("confidence"),
               "hough": res.get("hough"), "ransac_line": res.get("ransac_line"),
               "radon": res.get("radon"), "obb": res.get("obb"),
               "belt_width_px": res.get("belt_width_px")},
    )


# --- semantic ----------------------------------------------------------------------------
def _semantic(bgr_clahe: np.ndarray, layers) -> dict[str, Any]:
    ov = render.semantic_overlay(bgr_clahe, layers.label_map, layers.coverage, layers.engine)
    pct = {k: round(layers.coverage.get(k, 0.0) * 100, 1)
           for k in ("belt", "content", "foreign", "external")}
    return _entry(
        method_id="segmentation.semantic_layers", capability="segmentation", tier="sota",
        family="segmentation", name="4-class semantic layers",
        reference="Trained ONNX belt segmenter + open-vocab MobileSAM/CLIP foreign; classical prior",
        metric_name="belt_coverage_frac", metric_value=layers.coverage.get("belt", 0.0),
        summary=(f"4-class semantic map ({layers.engine}): belt {pct['belt']}%, content "
                 f"{pct['content']}%, foreign {pct['foreign']}%, external {pct['external']}%."),
        overlay_b64=render.to_png_b64(ov),
        extra={"engine": layers.engine, "coverage": layers.coverage,
               "n_regions": layers.n_regions},
    )


# --- anomaly: padim_lite (self-reference, no external bank) -------------------------------
def _padim_lite(bgr: np.ndarray) -> dict[str, Any]:
    res = anomaly.padim_lite(bgr)
    grid = np.asarray(res.get("residual_heatmap") or [[0.0]], dtype=np.float32)
    gr, gc = grid.shape
    h, w = bgr.shape[:2]
    peak = (int((res.get("peak_col", 0) + 0.5) / max(gc, 1) * w),
            int((res.get("peak_row", 0) + 0.5) / max(gr, 1) * h))
    score = res.get("image_score", 0.0)
    ov = render.heatmap_overlay(
        bgr, grid, legend_label="PaDiM-lite residual", peak_xy=peak, title="PaDiM-lite anomaly",
        summary=(f"PaDiM-lite (per-patch Gaussian, Mahalanobis, {res.get('fit_source')}): image "
                 f"score {score:.3f}, peak at grid cell ({res.get('peak_row')},{res.get('peak_col')})."),
    )
    return _entry(
        method_id="anomaly.padim_lite", capability="anomaly", tier="sota", family="anomaly",
        name="PaDiM-lite (per-patch Gaussian)", reference=res.get("reference", "PaDiM arXiv:2011.08785"),
        metric_name="image_score", metric_value=score,
        summary=(f"PaDiM-lite anomaly: image score {score:.3f} "
                 f"(higher = more anomalous vs the frame's own patch statistics)."),
        overlay_b64=render.to_png_b64(ov),
        extra={"fit_source": res.get("fit_source"), "grid": [gr, gc]},
    )


# --- anomaly: conv-AE reconstruction residual (trained ONNX) -----------------------------
def _conv_ae_residual(bgr: np.ndarray, onnx_path: Path,
                      input_size: int = 256) -> tuple[np.ndarray, float]:
    import cv2
    import onnxruntime as ort

    gray = cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    x = (x.astype(np.float32) / 255.0)[None, None, :, :]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    recon = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    residual = np.abs(x - recon)[0, 0]
    return residual, float(residual.mean())


def _conv_ae(bgr: np.ndarray, weights_dir: Path) -> dict[str, Any] | None:
    onnx_path = weights_dir / _CONV_AE_NAME
    ref = "Bergmann et al. 2019 (MVTec AD autoencoder baseline), CVPR 2019"
    if not onnx_path.is_file():
        return _entry(
            method_id="anomaly.conv_ae", capability="anomaly", tier="sota", family="anomaly",
            name="Conv-AE reconstruction", reference=ref, metric_name="image_score",
            metric_value=None, summary="Conv-AE weight absent (train + export conv_ae.onnx offline).",
            overlay_b64=None, status="weights_absent",
        )
    residual, score = _conv_ae_residual(bgr, onnx_path)
    h, w = bgr.shape[:2]
    py, px = np.unravel_index(int(np.argmax(residual)), residual.shape)
    peak = (int(px / residual.shape[1] * w), int(py / residual.shape[0] * h))
    ov = render.heatmap_overlay(
        bgr, residual, legend_label="AE reconstruction residual", peak_xy=peak,
        colormap=None, title="Conv-AE anomaly",
        summary=(f"Conv-AE reconstruction anomaly: mean residual {score:.4f}, peak residual "
                 f"{float(residual.max()):.4f}. High residual = poorly reconstructed (anomalous)."),
    )
    return _entry(
        method_id="anomaly.conv_ae", capability="anomaly", tier="sota", family="anomaly",
        name="Conv-AE reconstruction", reference=ref, metric_name="image_score",
        metric_value=score,
        summary=(f"Conv-AE reconstruction anomaly: mean absolute residual {score:.4f} "
                 "(trained on normal belt frames, higher = more anomalous)."),
        overlay_b64=render.to_png_b64(ov), extra={"max_residual": round(float(residual.max()), 6)},
    )


# --- anomaly: PaDiM + PatchCore frozen-backbone banks (GPU) -------------------------------
def _bank_features(bgr: np.ndarray, device: str) -> np.ndarray | None:
    try:
        from .backbone import ResNetPatchFeatures
    except Exception:
        return None
    ext = ResNetPatchFeatures(input_size=_BANK_INPUT, grid=_BANK_GRID, device=device)
    feats = ext.extract([bgr])  # (1, P, D)
    return feats[0] if feats.shape[0] else None


def _padim_bank(bgr: np.ndarray, feat_pd: np.ndarray, weights_dir: Path) -> dict[str, Any] | None:
    path = weights_dir / _PADIM_NAME
    if not path.is_file() or feat_pd is None:
        return None
    z = np.load(path)
    means, inv_covs, sel = z["means"].astype(np.float64), z["inv_covs"].astype(np.float64), z["sel_idx"]
    grid = int(z["grid"]) if "grid" in z else _BANK_GRID
    f = feat_pd[:, sel].astype(np.float64)                 # (P, d_sel)
    centered = f - means                                   # (P, d_sel)
    tmp = np.einsum("pij,pj->pi", inv_covs, centered)      # (P, d_sel)
    dist = np.sqrt(np.einsum("pi,pi->p", centered, tmp).clip(min=0.0))  # (P,)
    gmap = dist.reshape(grid, grid)
    h, w = bgr.shape[:2]
    pi = int(np.argmax(dist))
    pr, pc = pi // grid, pi % grid
    peak = (int((pc + 0.5) / grid * w), int((pr + 0.5) / grid * h))
    score = float(dist.max())
    ov = render.heatmap_overlay(
        bgr, gmap, legend_label="PaDiM Mahalanobis", peak_xy=peak, title="PaDiM anomaly",
        summary=(f"PaDiM (frozen ResNet-18 layer2+3, per-position Gaussian): max Mahalanobis "
                 f"distance {score:.2f} vs the iron-ore normal bank; hot = off-distribution."),
    )
    return _entry(
        method_id="anomaly.padim", capability="anomaly", tier="sota", family="anomaly",
        name="PaDiM (frozen-backbone bank)", reference="PaDiM arXiv:2011.08785",
        metric_name="max_mahalanobis", metric_value=score,
        summary=(f"PaDiM bank anomaly: max Mahalanobis {score:.2f} (fit on iron-ore normals; "
                 "an out-of-domain frame reads uniformly high)."),
        overlay_b64=render.to_png_b64(ov), extra={"grid": [grid, grid]},
    )


def _patchcore_bank(bgr: np.ndarray, feat_pd: np.ndarray, weights_dir: Path) -> dict[str, Any] | None:
    path = weights_dir / _PATCHCORE_NAME
    if not path.is_file() or feat_pd is None:
        return None
    z = np.load(path)
    coreset = z["coreset"].astype(np.float32)
    grid = int(z["grid"]) if "grid" in z else _BANK_GRID
    q = feat_pd.astype(np.float32)                         # (P, D)
    q2 = (q * q).sum(axis=1)[:, None]
    c2 = (coreset * coreset).sum(axis=1)[None, :]
    d2 = np.clip(q2 + c2 - 2.0 * (q @ coreset.T), 0.0, None)
    nn = np.sqrt(d2.min(axis=1))                           # (P,)
    gmap = nn.reshape(grid, grid)
    h, w = bgr.shape[:2]
    pi = int(np.argmax(nn))
    pr, pc = pi // grid, pi % grid
    peak = (int((pc + 0.5) / grid * w), int((pr + 0.5) / grid * h))
    score = float(nn.max())
    ov = render.heatmap_overlay(
        bgr, gmap, legend_label="PatchCore kNN distance", peak_xy=peak, title="PatchCore anomaly",
        summary=(f"PatchCore-lite (coreset memory + nearest-neighbour): max patch distance "
                 f"{score:.2f} to the iron-ore normal bank; hot = far from any normal patch."),
    )
    return _entry(
        method_id="anomaly.patchcore", capability="anomaly", tier="sota", family="anomaly",
        name="PatchCore-lite (coreset memory)",
        reference="PatchCore arXiv:2106.08265 (coreset memory + kNN); lite random coreset",
        metric_name="max_nn_distance", metric_value=score,
        summary=(f"PatchCore-lite anomaly: max nearest-neighbour distance {score:.2f} to the "
                 "iron-ore normal coreset (higher = more anomalous)."),
        overlay_b64=render.to_png_b64(ov), extra={"grid": [grid, grid], "coreset_size": int(coreset.shape[0])},
    )


# --- granulometry (on the segmented content) ---------------------------------------------
def _watershed_labels(bgr: np.ndarray, foreground: np.ndarray, *, min_area_px: int = 12):
    import cv2
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed

    gray = cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = cv2.morphologyEx((foreground > 0).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=max(3, int(0.01 * min(h, w))), labels=mask)
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    markers, _ = ndi.label(markers > 0)
    labels = watershed(-dist, markers, mask=mask)
    diam = np.asarray([float(np.sqrt(4.0 * p.area / np.pi)) for p in regionprops(labels)
                       if p.area >= min_area_px], dtype=np.float64)
    return labels, diam


def _granulometry(bgr: np.ndarray, content_mask: np.ndarray,
                  footprint: np.ndarray, px_per_mm: float | None) -> dict[str, Any]:
    # Prefer the segmented content; fall back to the belt footprint, then whole-frame Otsu.
    fg = content_mask if int(content_mask.sum()) > 400 else footprint
    where = "content" if int(content_mask.sum()) > 400 else "belt footprint"
    res = granulometry.psd_from_mask(bgr, fg, px_per_mm=px_per_mm)
    labels, _diam = _watershed_labels(bgr, fg)
    d50, unit, n = res.get("D50", 0.0), res.get("unit", "px"), res.get("n_particles", 0)
    summ = (f"Granulometry on the {where}: {n} particles, D50 {d50:.0f} {unit}, D80 "
            f"{res.get('D80', 0):.0f} {unit}, oversize {res.get('oversize_frac', 0)*100:.0f}%.")
    ov = render.granulometry_overlay(bgr, labels, summary=summ)
    return _entry(
        method_id="granulometry.watershed_psd", capability="granulometry", tier="classical",
        family="granulometry", name="Watershed granulometry (PSD)",
        reference="Watershed granulometry (Gonzalez & Woods; Split/WipFrag); Rosin-Rammler ISO 9276-1",
        metric_name="D50", metric_value=d50, summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"unit": unit, "n_particles": n, "D10": res.get("D10"), "D80": res.get("D80"),
               "oversize_frac": res.get("oversize_frac"), "measured_on": where},
    )


# --- tracking: dense optical flow --------------------------------------------------------
def _optical_flow(bgr: np.ndarray) -> dict[str, Any]:
    import cv2

    res = tracking.optical_flow(bgr)
    curr = cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY)
    prev = np.roll(curr, -4, axis=0)  # same self-shift demo the live method labels honestly
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    speed = res.get("belt_speed_px_per_frame", 0.0)
    summ = (f"Dense Farneback optical flow ({res.get('source')}): belt speed {speed:.2f} px/frame, "
            f"direction {res.get('flow_direction_deg', 0):.0f}deg, "
            f"{'moving' if res.get('moving') else 'stopped'}.")
    ov = render.flow_overlay(bgr, flow, summary=summ)
    return _entry(
        method_id="tracking.optical_flow", capability="tracking", tier="classical",
        family="tracking", name="Dense optical flow (Farneback)",
        reference="Farneback 2003 (SCIA) dense optical flow", metric_name="belt_speed_px_per_frame",
        metric_value=speed, summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"source": res.get("source"), "flow_direction_deg": res.get("flow_direction_deg"),
               "moving": res.get("moving")},
    )


# --- segmentation: MobileSAM automatic masks (best-effort, GPU) ---------------------------
def _mobile_sam(bgr: np.ndarray, weights_dir: Path) -> dict[str, Any] | None:
    path = weights_dir / _MOBILE_SAM_NAME
    if not path.is_file():
        return None
    res = segmentation.mobile_sam(bgr, weights=str(path))
    if res.get("status") != "ok":
        return None
    masks = res.get("masks", [])
    boxes = [tuple(m["bbox_xywh"]) for m in masks[:40] if m.get("bbox_xywh")]
    n = res.get("n_masks", len(boxes))
    summ = (f"MobileSAM automatic mask generation: {n} masks, "
            f"{res.get('coverage_frac', 0)*100:.0f}% frame coverage (learned, class-agnostic).")
    ov = render.masks_overlay(bgr, boxes, summary=summ)
    return _entry(
        method_id="segmentation.mobile_sam", capability="segmentation", tier="sota",
        family="segmentation", name="MobileSAM automatic masks",
        reference="MobileSAM (Tiny-ViT 5M), Apache-2.0; FastSAM", metric_name="n_masks",
        metric_value=float(n), summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"coverage_frac": res.get("coverage_frac")},
    )


def precompute_methods(
    image: Any,
    *,
    view_type: str | None = None,
    px_per_mm: float | None = None,
    device: str = "cpu",
    use_learned: bool = True,
    weights_dir: str | Path | None = None,
    include_mobile_sam: bool = True,
) -> dict[str, Any]:
    """Run the FULL per-method toolbox on one frame and return uniform, overlay-carrying records.

    Returns ``{"device", "n_methods", "methods": [record, ...], "errors": [...]}`` where each
    record is ``{id, capability, tier, family, name, reference, metric_name, metric_value,
    summary, overlay_b64, status, extra}``. Every method is wrapped so one failure is recorded
    (in ``errors``) and skipped, never aborting the batch.
    """
    bgr = as_bgr(image)
    clahe = apply_clahe_lab(bgr)
    wd = _weights_dir(weights_dir)
    methods: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    # segment the 4 semantic layers ONCE; share the belt footprint across the bench.
    layers = compute_layers(bgr, view_type=view_type, use_learned=use_learned)
    footprint = layers.belt_mask | layers.content_mask

    def _try(label: str, fn):
        try:
            r = fn()
            if r is None:
                return
            methods.extend(r) if isinstance(r, list) else methods.append(r)
        except Exception as exc:  # noqa: BLE001 - one broken method must not abort the batch
            errors.append({"method": label, "error": f"{type(exc).__name__}: {exc}"})

    # classical bench (19) + consolidated geometry (1)
    _try("features.run_all", lambda: _classical_features(bgr, footprint))
    _try("geometry.analysis", lambda: _geometry_analysis(bgr, footprint, view_type))
    # semantic (sota)
    _try("segmentation.semantic_layers", lambda: _semantic(clahe, layers))
    # anomaly (sota): self-ref padim-lite, trained conv-AE, frozen-backbone banks
    _try("anomaly.padim_lite", lambda: _padim_lite(bgr))
    _try("anomaly.conv_ae", lambda: _conv_ae(bgr, wd))
    feat_pd = None
    try:
        feat_pd = _bank_features(bgr, device)
    except Exception as exc:  # noqa: BLE001 - banks are optional (torch/torchvision)
        errors.append({"method": "backbone.extract", "error": f"{type(exc).__name__}: {exc}"})
    _try("anomaly.padim", lambda: _padim_bank(bgr, feat_pd, wd))
    _try("anomaly.patchcore", lambda: _patchcore_bank(bgr, feat_pd, wd))
    # granulometry (classical) on the segmented content
    _try("granulometry.watershed_psd",
         lambda: _granulometry(bgr, layers.content_mask, footprint, px_per_mm))
    # tracking (classical): dense optical flow
    _try("tracking.optical_flow", lambda: _optical_flow(bgr))
    # MobileSAM automatic masks (best-effort, sota)
    if include_mobile_sam:
        _try("segmentation.mobile_sam", lambda: _mobile_sam(bgr, wd))

    return {
        "device": device,
        "layers_engine": layers.engine,
        "n_methods": len(methods),
        "methods": methods,
        "errors": errors,
    }


__all__ = ["precompute_methods"]
