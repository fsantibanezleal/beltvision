"""The Pipeline Studio node DAG engine.

A classical-CV pipeline is a directed acyclic graph of small operators - source/ROI ->
preprocess -> transform -> binarize/morphology -> constrained detect -> measure - where the
image threads from a node to its consumers and EVERY node exposes its own step result
(overlay + metrics) so a user can inspect and work on any intermediate stage. This is the
engine behind the guided-analysis defect fix: detectors run on a preprocessed, thresholded,
ROI-masked, orientation-constrained input, never the raw frame.

- :data:`OP_REGISTRY` maps ``op_id -> {fn(image, params, ctx) -> {image, overlay_b64,
  metrics, status}, category, params_schema, reference}``. Ops reuse the real
  :mod:`beltvision.methods` implementations (features / constrained / transforms / measure /
  granulometry / roi); nothing is reimplemented here.
- :func:`run_pipeline` executes a spec ``{"nodes":[{"id","op","params","inputs":[ids]}]}``
  topologically, threads each node's output image to its consumers and captures every node's
  overlay + metrics. One node failing is recorded in ``errors`` and never aborts the run.
- :data:`TEMPLATES` ships the correctly-staged pipelines (``belt_detection``,
  ``belt_condition``, ``material_on_belt``, and ``robust_cascade`` — the auto multi-pipeline
  cascade driven by your ROI); :func:`list_templates` / :func:`get_template` expose them.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from .methods import constrained, features, granulometry
from .methods import measure as measure_mod
from .methods import robust as robust_mod
from .methods import roi as roi_mod
from .methods import transforms as tf
from .methods._common import ENVELOPE_KEYS, as_bgr
from .methods.preprocess import apply_clahe_lab
from .render import draw_legend, draw_summary, to_png_b64

_DROP = set(ENVELOPE_KEYS) | {"overlay_b64"}


# --- small image helpers ----------------------------------------------------------------
def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _is_binary(a: np.ndarray) -> bool:
    a = np.asarray(a)
    if a.ndim != 2:
        return False
    if a.dtype == bool:
        return True
    u = np.unique(a)
    return u.size <= 2 and set(int(v) for v in u.tolist()) <= {0, 1, 255}


def _gray_u8(image: Any) -> np.ndarray:
    import cv2

    a = np.asarray(image)
    if a.ndim == 3:
        return cv2.cvtColor(as_bgr(a), cv2.COLOR_BGR2GRAY)
    if a.dtype == np.uint8:
        return a
    if _is_binary(a):
        return (a > 0).astype(np.uint8) * 255
    return (_norm01(a) * 255.0).astype(np.uint8)


def _binary_bool(image: Any) -> np.ndarray:
    import cv2

    a = np.asarray(image)
    if a.ndim == 2 and _is_binary(a):
        return a > 0
    gray = _gray_u8(a)
    _thr, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b > 0


def _display_bgr(image: Any) -> np.ndarray:
    import cv2

    a = np.asarray(image)
    if a.ndim == 3:
        return as_bgr(a).copy()
    return cv2.cvtColor(_gray_u8(a), cv2.COLOR_GRAY2BGR)


def _step(display_bgr: np.ndarray, label: str, summary: str) -> str:
    img = display_bgr.copy()
    draw_legend(img, [((0, 200, 255), label)])
    draw_summary(img, summary)
    return to_png_b64(img)


def _metrics_of(rec: dict[str, Any]) -> dict[str, Any]:
    """Strip the gate/envelope keys off a method record, leaving the inspectable metrics."""
    return {k: v for k, v in rec.items() if k not in _DROP}


def _ok(image: Any, overlay_b64: str | None, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"image": image, "overlay_b64": overlay_b64, "metrics": metrics, "status": "ok"}


# --- context helpers --------------------------------------------------------------------
def _roi_mask(ctx: dict[str, Any]) -> np.ndarray | None:
    return ctx.get("roi_mask")


def _band(ctx: dict[str, Any], params: dict[str, Any]) -> tuple[float, float]:
    pr = ctx.get("priors") or {}
    center = params.get("theta_center_deg", pr.get("theta_center_deg"))
    band = params.get("theta_band_deg", pr.get("theta_band_deg"))
    if center is not None and band is not None:
        return float(center), float(band)
    c, b = roi_mod.orientation_band(pr.get("view") or pr.get("view_type"),
                                    ctx.get("rois"), ctx["shape"], ctx["frame"])
    return (float(center) if center is not None else c,
            float(band) if band is not None else b)


def _edge_map(image: Any, ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    import cv2

    a = np.asarray(image)
    if a.ndim == 2 and _is_binary(a):
        return a > 0
    # Use the actual node's input image, not ctx["frame"]: upstream preprocessing must thread
    # correctly through the DAG.  Falling back to ctx["frame"] would silently discard every
    # upstream CLAHE / denoise / ROI step whenever the input to a detect node is non-binary.
    bgr_in = as_bgr(a) if a.ndim == 3 else cv2.cvtColor(_gray_u8(a), cv2.COLOR_GRAY2BGR)
    return constrained.preprocess_for_lines(
        bgr_in, roi_mask=_roi_mask(ctx),
        denoise=params.get("denoise", "gaussian"), edge=params.get("edge", "canny"))


# --- SOURCE / ROI -----------------------------------------------------------------------
def _op_apply_roi(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    label = params.get("label") or params.get("labels")
    anns = ctx.get("rois") or []

    # Select annotations that match the label (str or list); if no label, take all.
    if label:
        wanted = {label} if isinstance(label, str) else set(label)
        selected = [a for a in anns if str(a.get("label", "")) in wanted]
    else:
        selected = list(anns)

    if selected:
        # Rasterise each annotation separately, then union them.
        # ctx["roi_masks"] is the per-annotation list so detect ops can run once per ROI.
        per_ann = [roi_mod.rasterize([a], ctx["shape"]) for a in selected]
        mask = np.zeros(ctx["shape"][:2], dtype=bool)
        valid_masks: list[np.ndarray] = []
        for m in per_ann:
            mask |= m
            if m.any():
                valid_masks.append(m)
        ctx["roi_masks"] = valid_masks
    else:
        mask = np.ones(ctx["shape"][:2], dtype=bool)
        ctx["roi_masks"] = []

    ctx["roi_mask"] = mask if mask.any() else None
    disp = _display_bgr(ctx["frame"])
    if mask.any() and not mask.all():
        color = np.zeros_like(disp)
        color[mask] = (60, 200, 255)
        disp = cv2.addWeighted(disp, 0.75, color, 0.25, 0)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, (60, 200, 255), 2, cv2.LINE_AA)
    frac = float(mask.mean())
    n_rois = len(ctx["roi_masks"])
    b64 = _step(disp, "region of interest",
                f"ROI '{label}': {int(mask.sum())} px ({frac*100:.0f}% of frame), "
                f"{n_rois} annotation(s). Line detectors will run once per ROI.")
    return _ok(image, b64, {"roi_label": label, "roi_area_px": int(mask.sum()),
                            "roi_frac": round(frac, 4), "n_rois": n_rois})


# --- PREPROCESS -------------------------------------------------------------------------
def _op_to_gray(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    gray = _gray_u8(image)
    return _ok(gray, _step(_display_bgr(gray), "grayscale", "Single-channel intensity image."),
               {"mean_intensity": round(float(gray.mean()), 2)})


def _op_clahe(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    a = np.asarray(image)
    bgr = as_bgr(a) if a.ndim == 3 else cv2.cvtColor(_gray_u8(a), cv2.COLOR_GRAY2BGR)
    out = apply_clahe_lab(bgr, clip=float(params.get("clip", 2.0)), tile=int(params.get("tile", 8)))
    return _ok(out, _step(out, "CLAHE (LAB-L)",
                          "Contrast-limited adaptive histogram equalisation (dust-robust "
                          "first stage)."), {"clip": float(params.get("clip", 2.0))})


def _op_gaussian(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    k = int(params.get("ksize", 5)) | 1
    out = cv2.GaussianBlur(np.asarray(image), (k, k), 0)
    return _ok(out, _step(_display_bgr(out), "gaussian denoise",
                          f"Gaussian blur (k={k}) - smooth noise before edges."), {"ksize": k})


def _op_median(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    k = int(params.get("ksize", 5)) | 1
    out = cv2.medianBlur(_gray_u8(image) if np.asarray(image).ndim == 2 else as_bgr(image), k)
    return _ok(out, _step(_display_bgr(out), "median denoise",
                          f"Median filter (k={k}) - remove speckle, keep edges."), {"ksize": k})


def _op_bilateral(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    out = cv2.bilateralFilter(np.asarray(image), int(params.get("d", 5)),
                              float(params.get("sigma_color", 50.0)),
                              float(params.get("sigma_space", 50.0)))
    return _ok(out, _step(_display_bgr(out), "bilateral denoise",
                          "Edge-preserving bilateral denoise."), {})


def _op_illumination_normalize(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    gray = _gray_u8(image).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 16.0)
    norm = gray / (bg + 1e-3)
    out = (_norm01(norm) * 255.0).astype(np.uint8)
    return _ok(out, _step(_display_bgr(out), "illumination normalize",
                          "Divide by a large-scale blur to flatten uneven lighting/shading."),
               {"kernel_sigma": round(max(gray.shape) / 16.0, 1)})


# --- TRANSFORM (gradient / edge) --------------------------------------------------------
def _grad(image: Any, mag_fn: Callable[[np.ndarray, Any], np.ndarray], label: str,
          desc: str, thr: float = 0.28) -> dict[str, Any]:
    import cv2

    gray = _gray_u8(image)
    mag = _norm01(mag_fn(gray, cv2))
    out = (mag * 255.0).astype(np.uint8)
    dens = round(float(np.mean(mag >= thr)), 5)
    return _ok(out, _step(_display_bgr(out), label,
                          f"{desc} Edge density {dens*100:.1f}% above {thr:.2f}."),
               {"edge_density": dens})


def _op_sobel(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    def m(g, cv2):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    return _grad(image, m, "Sobel magnitude", "Sobel gradient magnitude.")


def _op_scharr(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    def m(g, cv2):
        gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
        return np.sqrt(gx * gx + gy * gy)
    return _grad(image, m, "Scharr magnitude", "Scharr rotation-optimal gradient magnitude.")


def _op_log(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    def m(g, cv2):
        return np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.6), cv2.CV_32F, ksize=3))
    return _grad(image, m, "Laplacian-of-Gaussian", "Marr-Hildreth LoG response.", thr=0.3)


def _op_morph_gradient(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    def m(g, cv2):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k).astype(np.float32)
    return _grad(image, m, "Morphological gradient", "Dilation-minus-erosion gradient.", thr=0.3)


def _op_canny(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    gray = _gray_u8(image)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0),
                      int(params.get("lo", 50)), int(params.get("hi", 150)))
    dens = round(float(np.mean(edges > 0)), 5)
    return _ok(edges, _step(_display_bgr(edges), "Canny edges",
                            f"Binary Canny edge map (density {dens*100:.1f}%). The thresholded "
                            "input a Hough/RANSAC line detector requires."),
               {"edge_density": dens})


def _op_gabor(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    gray = _gray_u8(image).astype(np.float32)
    responses = []
    for th in (0.0, 45.0, 90.0, 135.0):
        k = cv2.getGaborKernel((21, 21), 4.0, np.radians(th), 10.0, 0.5, 0.0, ktype=cv2.CV_32F)
        responses.append(np.abs(cv2.filter2D(gray, cv2.CV_32F, k)))
    out = (_norm01(np.stack(responses, 0).max(0)) * 255.0).astype(np.uint8)
    return _ok(out, _step(_display_bgr(out), "Gabor energy",
                          "Max Gabor filter-bank response over 0/45/90/135deg."), {})


# --- TRANSFORM (frequency / wavelet) reused from transforms.py --------------------------
def _op_fft_spectrum(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    rec = tf.fft_spectrum(image)
    return _ok(image, rec.get("overlay_b64"), _metrics_of(rec))


def _op_fft_filter(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    rec = tf.fft_filter(image, **params)
    recon, _ret = tf.fft_reconstruct_array(_gray_u8(image), **{
        k: params[k] for k in ("kind", "orientation_deg", "width_deg", "r_low", "r_high")
        if k in params})
    out = (_norm01(np.abs(recon)) * 255.0).astype(np.uint8)
    return _ok(out, rec.get("overlay_b64"), _metrics_of(rec))


def _op_dwt_reconstruct(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    rec = tf.dwt_reconstruct(image, **params)
    recon, _rf, _lvl = tf.dwt_reconstruct_array(_gray_u8(image), **{
        k: params[k] for k in ("wavelet", "level", "keep") if k in params})
    out = (_norm01(np.abs(recon)) * 255.0).astype(np.uint8)
    return _ok(out, rec.get("overlay_b64"), _metrics_of(rec))


def _op_wavelet_denoise(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    rec = tf.wavelet_denoise(image, **params)
    den, _removed = tf.wavelet_denoise_array(_gray_u8(image),
                                             **{k: params[k] for k in ("wavelet",) if k in params})
    return _ok(den, rec.get("overlay_b64"), _metrics_of(rec))


# --- BINARIZE / MORPHOLOGY --------------------------------------------------------------
def _op_otsu(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    gray = _gray_u8(image)
    thr, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _ok(b, _step(_display_bgr(b), "Otsu threshold",
                        f"Global Otsu binarisation at {int(thr)}."), {"threshold": int(thr),
                                                                      "fg_frac": round(float(np.mean(b > 0)), 4)})


def _op_adaptive(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    gray = _gray_u8(image)
    block = int(params.get("block", 31)) | 1
    b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                              block, float(params.get("C", 5.0)))
    return _ok(b, _step(_display_bgr(b), "adaptive threshold",
                        f"Adaptive Gaussian threshold (block {block})."), {"block": block})


def _morph(image: Any, opname: str, cv_op: int, ksize: int) -> dict[str, Any]:
    import cv2

    m = (_binary_bool(image)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.morphologyEx(m, cv_op, k) if cv_op is not None else m
    return _ok(out, _step(_display_bgr(out), f"morphology: {opname}",
                          f"Binary {opname} (k={ksize})."), {"fg_frac": round(float(np.mean(out > 0)), 4)})


def _op_open(image, params, ctx):
    import cv2
    return _morph(image, "open", cv2.MORPH_OPEN, int(params.get("ksize", 3)))


def _op_close(image, params, ctx):
    import cv2
    return _morph(image, "close", cv2.MORPH_CLOSE, int(params.get("ksize", 5)))


def _op_dilate(image, params, ctx):
    import cv2
    return _morph(image, "dilate", cv2.MORPH_DILATE, int(params.get("ksize", 3)))


def _op_erode(image, params, ctx):
    import cv2
    return _morph(image, "erode", cv2.MORPH_ERODE, int(params.get("ksize", 3)))


def _op_skeletonize(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from skimage.morphology import skeletonize

    sk = skeletonize(_binary_bool(image))
    out = sk.astype(np.uint8) * 255
    return _ok(out, _step(_display_bgr(out), "skeleton",
                          "Morphological skeleton (medial axis of the binary shape)."),
               {"skeleton_px": int(sk.sum())})


def _op_thin(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from skimage.morphology import thin

    th = thin(_binary_bool(image))
    out = th.astype(np.uint8) * 255
    return _ok(out, _step(_display_bgr(out), "thinned",
                          "Iterative thinning to unit-width strokes."), {"thin_px": int(th.sum())})


# --- DETECT -----------------------------------------------------------------------------
def _op_hough_constrained(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    center, band = _band(ctx, params)
    roi_masks = ctx.get("roi_masks") or []

    if len(roi_masks) > 1:
        # Per-ROI: the user drew one annotation per belt region (e.g., left-edge strip and
        # right-edge strip).  Run constrained Hough separately within each ROI and merge.
        # This is the correct behaviour: union-then-detect can confuse lines across regions.
        be_full = _edge_map(image, ctx, params)
        all_segments: list[dict[str, Any]] = []
        for per_mask in roi_masks:
            be_per = be_full & per_mask
            if params.get("use_gate", True):
                be_per = constrained.gradient_orientation_gate(be_per, ctx["clahe_gray"],
                                                               center, band)
            rec = constrained.hough_constrained(be_per, center, band,
                                                min_len_px=params.get("min_len_px"),
                                                roi_mask=per_mask, bgr=None)
            all_segments.extend(rec.get("segments") or [])

        canvas = _display_bgr(ctx["frame"])
        for s in all_segments:
            p0 = (int(round(s["p0"][0])), int(round(s["p0"][1])))
            p1 = (int(round(s["p1"][0])), int(round(s["p1"][1])))
            cv2.line(canvas, p0, p1, (70, 230, 70), 2, cv2.LINE_AA)
        lo, hi = round(center - band, 1), round(center + band, 1)
        draw_legend(canvas, [((70, 230, 70), f"per-ROI in-band lines [{lo}, {hi}]deg")])
        draw_summary(canvas, f"Constrained Hough (per-ROI): {len(all_segments)} line(s) across "
                             f"{len(roi_masks)} ROIs, all within {center:.0f}+/-{band:.0f}deg.")
        metrics: dict[str, Any] = {
            "n_lines": len(all_segments), "segments": all_segments,
            "theta_center_deg": round(center, 2), "theta_band_deg": round(band, 2),
            "n_rois": len(roi_masks), "per_roi": True,
        }
        return _ok(image, to_png_b64(canvas), metrics)

    # Single mask (or no mask): standard constrained Hough.
    be = _edge_map(image, ctx, params)
    if params.get("use_gate", True):
        be = constrained.gradient_orientation_gate(be, ctx["clahe_gray"], center, band)
    rec = constrained.hough_constrained(be, center, band, min_len_px=params.get("min_len_px"),
                                        roi_mask=_roi_mask(ctx), bgr=ctx["frame"])
    return _ok(image, rec.get("overlay_b64"), _metrics_of(rec))


def _op_ransac_line_constrained(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    center, band = _band(ctx, params)
    roi_masks = ctx.get("roi_masks") or []

    if len(roi_masks) > 1:
        be_full = _edge_map(image, ctx, params)
        all_lines: list[dict[str, Any]] = []
        for per_mask in roi_masks:
            be_per = be_full & per_mask
            if params.get("use_gate", True):
                be_per = constrained.gradient_orientation_gate(be_per, ctx["clahe_gray"],
                                                               center, band)
            rec = constrained.ransac_line_constrained(be_per, center, band,
                                                      roi_mask=per_mask, bgr=None)
            all_lines.extend(rec.get("lines") or [])

        canvas = _display_bgr(ctx["frame"])
        for ln in all_lines:
            p0 = (int(round(ln["p0"][0])), int(round(ln["p0"][1])))
            p1 = (int(round(ln["p1"][0])), int(round(ln["p1"][1])))
            cv2.line(canvas, p0, p1, (60, 200, 60), 3, cv2.LINE_AA)
        lo, hi = round(center - band, 1), round(center + band, 1)
        draw_legend(canvas, [((60, 200, 60), f"per-ROI RANSAC lines [{lo}, {hi}]deg")])
        draw_summary(canvas, f"Constrained RANSAC (per-ROI): {len(all_lines)} line(s) across "
                             f"{len(roi_masks)} ROIs, all within {center:.0f}+/-{band:.0f}deg.")
        metrics: dict[str, Any] = {
            "n_lines": len(all_lines), "lines": all_lines,
            "segments": all_lines,  # alias so measure_lines / belt_edges can consume either key
            "theta_center_deg": round(center, 2), "theta_band_deg": round(band, 2),
            "n_rois": len(roi_masks), "per_roi": True,
        }
        return _ok(image, to_png_b64(canvas), metrics)

    be = _edge_map(image, ctx, params)
    if params.get("use_gate", True):
        be = constrained.gradient_orientation_gate(be, ctx["clahe_gray"], center, band)
    rec = constrained.ransac_line_constrained(be, center, band, roi_mask=_roi_mask(ctx),
                                              bgr=ctx["frame"])
    m = _metrics_of(rec)
    m.setdefault("segments", m.get("lines") or [])  # expose under both keys
    return _ok(image, rec.get("overlay_b64"), m)


def _op_find_contours(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    b = _binary_bool(image).astype(np.uint8)
    if _roi_mask(ctx) is not None:
        b = b & _roi_mask(ctx).astype(np.uint8)
    min_area = float(params.get("min_area", 25.0))
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    areas = [round(float(cv2.contourArea(c)), 1) for c in cnts]
    disp = _display_bgr(ctx["frame"])
    cv2.drawContours(disp, cnts, -1, (80, 255, 140), 2, cv2.LINE_AA)
    b64 = _step(disp, "contours", f"{len(cnts)} external contour(s) with area >= {min_area:.0f}px.")
    return _ok(image, b64, {"n_contours": len(cnts), "areas": areas,
                            "mean_area": round(float(np.mean(areas)), 1) if areas else 0.0})


def _op_blob_detect(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2
    from skimage.feature import blob_log

    gray = _gray_u8(image).astype(np.float32) / 255.0
    blobs = blob_log(gray, max_sigma=float(params.get("max_sigma", 24.0)), num_sigma=8,
                     threshold=float(params.get("threshold", 0.08)))
    disp = _display_bgr(ctx["frame"])
    for y, x, s in blobs:
        cv2.circle(disp, (int(x), int(y)), int(s * np.sqrt(2)), (0, 200, 255), 2, cv2.LINE_AA)
    b64 = _step(disp, "LoG blobs", f"{len(blobs)} blob(s) via Laplacian-of-Gaussian scale space.")
    radii = [round(float(s * np.sqrt(2)), 1) for _y, _x, s in blobs]
    return _ok(image, b64, {"n_blobs": int(len(blobs)),
                            "mean_radius_px": round(float(np.mean(radii)), 1) if radii else 0.0})


def _op_harris(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    rec = features.harris(ctx["frame"], mask=_roi_mask(ctx))
    return _ok(image, rec.get("overlay_b64"), _metrics_of(rec))


def _op_shi_tomasi(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    rec = features.shi_tomasi(ctx["frame"], mask=_roi_mask(ctx))
    return _ok(image, rec.get("overlay_b64"), _metrics_of(rec))


def _op_belt_edges(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Extract the two belt-edge lines (edge_a / edge_b) from upstream Hough / RANSAC results.

    Reads the ``segments`` or ``lines`` key from every upstream detect node, projects each
    segment's midpoint onto the belt normal direction, applies a 2-cluster split that maximises
    the inter-cluster gap, and returns the longest representative from each cluster.  Reports
    belt width in pixels (and mm if ``px_per_mm`` is set) and the belt centreline.
    """
    import cv2

    # Gather all segments / lines from upstream detect nodes.
    all_segments: list[dict[str, Any]] = []
    for in_res in ctx.get("inputs", []):
        m = in_res.get("metrics") or {}
        for key in ("segments", "lines"):
            segs = m.get(key)
            if segs:
                all_segments.extend(segs)
                break

    center, band = _band(ctx, params)
    result_dict = constrained.extract_belt_edges(
        all_segments, center, frame_shape=ctx["shape"][:2])

    canvas = _display_bgr(ctx["frame"])
    label_pairs: list[tuple] = []
    if result_dict["found"]:
        edge_a = result_dict.get("edge_a")
        edge_b = result_dict.get("edge_b")
        for edge, col, tag in [
            (edge_a, (60, 230, 80), "edge A"),
            (edge_b, (80, 180, 255), "edge B"),
        ]:
            if edge:
                p0 = (int(round(edge["p0"][0])), int(round(edge["p0"][1])))
                p1 = (int(round(edge["p1"][0])), int(round(edge["p1"][1])))
                cv2.line(canvas, p0, p1, col, 3, cv2.LINE_AA)
                label_pairs.append((col, f"belt {tag}: {edge.get('angle_deg', '?'):.1f}deg"))
        cl = result_dict.get("centreline")
        if cl:
            cen_col = (200, 200, 60)
            cv2.line(canvas,
                     (int(round(cl["p0"][0])), int(round(cl["p0"][1]))),
                     (int(round(cl["p1"][0])), int(round(cl["p1"][1]))),
                     cen_col, 2, cv2.LINE_AA)
            px_per_mm = (ctx.get("priors") or {}).get("px_per_mm")
            w_note = (f" = {result_dict['width_px'] / px_per_mm:.1f}mm"
                      if px_per_mm and px_per_mm > 0 else "")
            label_pairs.append((cen_col,
                                 f"centreline, belt width={result_dict['width_px']:.0f}px{w_note}"))

    draw_legend(canvas, label_pairs)
    if result_dict["found"]:
        draw_summary(canvas, f"Belt edges FOUND: A and B are the most-separated in-band line pair. "
                             f"Width={result_dict['width_px']:.0f}px.")
    else:
        draw_summary(canvas, f"Belt edges NOT FOUND: {result_dict.get('reason', 'unknown reason')}. "
                             "Draw ROI annotations on each belt edge and re-run belt_detection.")

    metrics: dict[str, Any] = {k: v for k, v in result_dict.items()}
    # Expose the two belt edges as 'segments' so downstream measure_lines picks them up.
    if result_dict["found"]:
        metrics["segments"] = [s for s in [result_dict.get("edge_a"),
                                            result_dict.get("edge_b")] if s]
    px_per_mm = (ctx.get("priors") or {}).get("px_per_mm")
    if px_per_mm and px_per_mm > 0 and result_dict.get("width_px"):
        metrics["width_mm"] = round(result_dict["width_px"] / float(px_per_mm), 2)

    return _ok(image, to_png_b64(canvas), metrics)


# --- ROBUST CASCADE ops (the auto multi-pipeline analyses, drivable with ROIs) ----------
def _op_robust_belt_band(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Robust belt-band detector (orientation consensus + normal-projection two-limits + Hough
    cross-check). Honours the drawn ROI mask, so a user's ROI focuses/boosts the estimate."""
    rec = robust_mod.belt_band(ctx["frame"], roi_mask=_roi_mask(ctx))
    return _ok(image, rec.get("overlay_b64"), _metrics_of(rec))


def _op_robust_damage(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Robust damage: RGB anomaly ensemble inside the belt band (belt/content split when a
    'content' ROI is drawn). Runs the robust belt_band first, then damage within it."""
    band = robust_mod.belt_band(ctx["frame"], roi_mask=_roi_mask(ctx))
    content = roi_mod.combine_by_label(ctx.get("rois") or [], ctx["shape"], "content")
    rec = robust_mod.damage(ctx["frame"], band=band,
                            content_mask=(content if content.any() else None))
    return _ok(image, rec.get("overlay_b64"), _metrics_of(rec))


# --- MEASURE ----------------------------------------------------------------------------
def _first_geometry(ctx: dict[str, Any]) -> list[list[list[float]]]:
    """Pull line segments from the first upstream detect node that produced any."""
    for r in ctx.get("inputs", []):
        m = r.get("metrics") or {}
        for key in ("segments", "lines"):
            geom = m.get(key)
            if geom:
                return [[g["p0"], g["p1"]] for g in geom]
    return []


def _op_measure_lines(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    lines = _first_geometry(ctx)
    lengths = [measure_mod.segment_length(ln) for ln in lines]
    metrics: dict[str, Any] = {"n_lines": len(lines),
                               "lengths_px": [round(x, 1) for x in lengths],
                               "mean_length_px": round(float(np.mean(lengths)), 1) if lengths else 0.0}
    if len(lines) >= 2:
        order = np.argsort(lengths)[::-1]
        l1, l2 = lines[int(order[0])], lines[int(order[1])]
        metrics["angle_between_deg"] = measure_mod.angle_between(l1, l2)
    px_per_mm = (ctx.get("priors") or {}).get("px_per_mm")
    if px_per_mm:
        metrics["lengths_mm"] = [measure_mod.px_to_mm(x, px_per_mm) for x in lengths]
    disp = _display_bgr(ctx["frame"])
    for ln in lines:
        p0 = (int(round(ln[0][0])), int(round(ln[0][1])))
        p1 = (int(round(ln[1][0])), int(round(ln[1][1])))
        cv2.line(disp, p0, p1, (70, 230, 70), 2, cv2.LINE_AA)
    ang = metrics.get("angle_between_deg")
    b64 = _step(disp, "measured lines",
                f"{len(lines)} line(s), mean length {metrics['mean_length_px']:.0f}px"
                + (f", angle between the two longest {ang:.1f}deg." if ang is not None else "."))
    return _ok(image, b64, metrics)


def _op_measure_objects(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    mask = _binary_bool(image)
    if _roi_mask(ctx) is not None:
        mask = mask & _roi_mask(ctx)
    area_range = params.get("area_range")
    if area_range is not None:
        area_range = (float(area_range[0]), float(area_range[1]))
    res = measure_mod.count_objects(mask, area_range)
    roi_area = float(_roi_mask(ctx).sum()) if _roi_mask(ctx) is not None else float(mask.size)
    res["density"] = measure_mod.density(res["count"], roi_area)
    px_per_mm = (ctx.get("priors") or {}).get("px_per_mm")
    if px_per_mm:
        res["mean_area_mm2"] = measure_mod.px2_to_mm2(res["mean_area"], px_per_mm)
    disp = _display_bgr(ctx["frame"])
    n_lab, labels, stats, _cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for label in range(1, n_lab):
        a = float(stats[label, cv2.CC_STAT_AREA])
        if area_range is not None and not (area_range[0] <= a <= area_range[1]):
            continue
        x, y, w, h = (int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]),
                      int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]))
        cv2.rectangle(disp, (x, y), (x + w, y + h), (200, 60, 200), 2, cv2.LINE_AA)
    b64 = _step(disp, "measured objects",
                f"{res['count']} object(s), mean area {res['mean_area']:.0f}px2, density "
                f"{res['density']:.2e}/px2.")
    return _ok(image, b64, res)


def _op_granulometry(image: Any, params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2

    fg = _binary_bool(image)
    if _roi_mask(ctx) is not None:
        fg = fg & _roi_mask(ctx)
    px_per_mm = (ctx.get("priors") or {}).get("px_per_mm")
    res = granulometry.psd_from_mask(as_bgr(ctx["frame"]), fg, px_per_mm=px_per_mm)
    disp = _display_bgr(ctx["frame"])
    cnts, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp, cnts, -1, (0, 200, 255), 1, cv2.LINE_AA)
    b64 = _step(disp, "granulometry (watershed PSD)",
                f"{res['n_particles']} fragment(s); D50 {res['D50']:.1f}{res['unit']}, "
                f"D80 {res['D80']:.1f}{res['unit']}, oversize {res['oversize_frac']*100:.0f}%.")
    return _ok(image, b64, res)


# --- OP REGISTRY ------------------------------------------------------------------------
def _entry(fn: Callable, category: str, reference: str,
           params_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"fn": fn, "category": category, "reference": reference,
            "params_schema": params_schema or {}}


OP_REGISTRY: dict[str, dict[str, Any]] = {
    # source / roi
    "apply_roi": _entry(_op_apply_roi, "source_roi", "User ROI rasterisation (beltvision.methods.roi)",
                        {"label": {"type": "str|list", "default": None}}),
    # preprocess
    "to_gray": _entry(_op_to_gray, "preprocess", "Grayscale conversion"),
    "clahe": _entry(_op_clahe, "preprocess", "CLAHE clip2.0/8x8 LAB-L (IET-IPR 10.1049/iet-ipr.2019.0992)",
                    {"clip": {"type": "float", "default": 2.0}, "tile": {"type": "int", "default": 8}}),
    "gaussian_denoise": _entry(_op_gaussian, "preprocess", "Gaussian blur",
                               {"ksize": {"type": "int", "default": 5}}),
    "median_denoise": _entry(_op_median, "preprocess", "Median filter",
                             {"ksize": {"type": "int", "default": 5}}),
    "bilateral_denoise": _entry(_op_bilateral, "preprocess", "Edge-preserving bilateral filter"),
    "illumination_normalize": _entry(_op_illumination_normalize, "preprocess",
                                     "Large-scale-blur division (homomorphic-style shading flatten)"),
    # transform
    "sobel": _entry(_op_sobel, "transform", "Sobel-Feldman gradient"),
    "scharr": _entry(_op_scharr, "transform", "Scharr 2000 gradient"),
    "canny": _entry(_op_canny, "transform", "Canny 1986 hysteresis edges",
                    {"lo": {"type": "int", "default": 50}, "hi": {"type": "int", "default": 150}}),
    "log": _entry(_op_log, "transform", "Marr-Hildreth Laplacian-of-Gaussian"),
    "morph_gradient": _entry(_op_morph_gradient, "transform", "Morphological gradient"),
    "gabor": _entry(_op_gabor, "transform", "Gabor filter bank (0/45/90/135deg)"),
    "fft_spectrum": _entry(_op_fft_spectrum, "transform", tf._FFT_REF),
    "fft_filter": _entry(_op_fft_filter, "transform", tf._FFT_REF,
                         {"kind": {"type": "str", "default": "directional"},
                          "orientation_deg": {"type": "float", "default": 0.0}}),
    "dwt_reconstruct": _entry(_op_dwt_reconstruct, "transform", tf._DWT_REF,
                              {"keep": {"type": "list", "default": ["detail"]},
                               "wavelet": {"type": "str", "default": "db2"},
                               "level": {"type": "int", "default": 2}}),
    "wavelet_denoise": _entry(_op_wavelet_denoise, "transform", tf._DENOISE_REF,
                              {"wavelet": {"type": "str", "default": "db2"}}),
    # binarize / morphology
    "otsu": _entry(_op_otsu, "binarize_morphology", "Otsu 1979 global threshold"),
    "adaptive_threshold": _entry(_op_adaptive, "binarize_morphology", "Adaptive Gaussian threshold",
                                 {"block": {"type": "int", "default": 31}, "C": {"type": "float", "default": 5.0}}),
    "skeletonize": _entry(_op_skeletonize, "binarize_morphology", "Zhang-Suen skeletonization"),
    "thin": _entry(_op_thin, "binarize_morphology", "Morphological thinning"),
    "open": _entry(_op_open, "binarize_morphology", "Morphological opening",
                   {"ksize": {"type": "int", "default": 3}}),
    "close": _entry(_op_close, "binarize_morphology", "Morphological closing",
                    {"ksize": {"type": "int", "default": 5}}),
    "dilate": _entry(_op_dilate, "binarize_morphology", "Morphological dilation",
                     {"ksize": {"type": "int", "default": 3}}),
    "erode": _entry(_op_erode, "binarize_morphology", "Morphological erosion",
                    {"ksize": {"type": "int", "default": 3}}),
    # detect
    "hough_constrained": _entry(_op_hough_constrained, "detect",
                                "Constrained Hough (skimage hough_line, band-limited theta)",
                                {"theta_center_deg": {"type": "float", "default": None},
                                 "theta_band_deg": {"type": "float", "default": None},
                                 "use_gate": {"type": "bool", "default": True},
                                 "min_len_px": {"type": "float", "default": None}}),
    "ransac_line_constrained": _entry(_op_ransac_line_constrained, "detect",
                                      "Constrained RANSAC (orientation-band is_model_valid reject)",
                                      {"theta_center_deg": {"type": "float", "default": None},
                                       "theta_band_deg": {"type": "float", "default": None}}),
    "find_contours": _entry(_op_find_contours, "detect", "Suzuki & Abe 1985 contour tracing",
                            {"min_area": {"type": "float", "default": 25.0}}),
    "blob_detect": _entry(_op_blob_detect, "detect", "Laplacian-of-Gaussian blob scale space",
                          {"max_sigma": {"type": "float", "default": 24.0},
                           "threshold": {"type": "float", "default": 0.08}}),
    "harris": _entry(_op_harris, "detect", "Harris & Stephens 1988 corners"),
    "shi_tomasi": _entry(_op_shi_tomasi, "detect", "Shi & Tomasi 1994 good features"),
    "belt_edges": _entry(_op_belt_edges, "detect",
                         "Belt-edge pair extraction: 2-cluster split of upstream in-band lines "
                         "→ edge_a / edge_b / width_px / centreline "
                         "(beltvision.methods.constrained.extract_belt_edges)"),
    # robust cascade (auto multi-pipeline, ROI-drivable)
    "robust_belt_band": _entry(_op_robust_belt_band, "detect",
                               "Robust belt band: orientation consensus (Radon/FFT/structure-tensor) "
                               "+ normal-projection two-limits + Hough cross-check, fused with an "
                               "agreement confidence; centreline = midline of the limits "
                               "(beltvision.methods.robust.belt_band)"),
    "robust_damage": _entry(_op_robust_damage, "detect",
                            "Robust damage: RGB anomaly ensemble (illum+wavelet+FFT+morph) inside "
                            "the belt band, with belt/content split (beltvision.methods.robust.damage)"),
    # measure
    "measure_lines": _entry(_op_measure_lines, "measure",
                            "Angle/length over detected lines (beltvision.methods.measure)"),
    "measure_objects": _entry(_op_measure_objects, "measure",
                              "Count/area/density over a mask (beltvision.methods.measure)",
                              {"area_range": {"type": "list", "default": None}}),
    "granulometry": _entry(_op_granulometry, "measure",
                           "Watershed PSD over the segmented content (Rosin-Rammler ISO 9276-1)"),
}


def list_ops() -> list[str]:
    """Every registered op id, in registration (category) order."""
    return list(OP_REGISTRY)


def op_catalog() -> list[dict[str, Any]]:
    """A flat JSON-safe catalogue of the op registry for the front end (id/category/refs/params)."""
    return [{"id": op_id, "category": spec["category"], "reference": spec["reference"],
             "params_schema": spec["params_schema"]} for op_id, spec in OP_REGISTRY.items()]


# --- topological execution --------------------------------------------------------------
def _toposort(nodes: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Kahn topological order over node ids; report any cyclic / unorderable nodes as errors."""
    ids = [n["id"] for n in nodes]
    indeg = {i: 0 for i in ids}
    adj: dict[str, list[str]] = {i: [] for i in ids}
    id_set = set(ids)
    for n in nodes:
        for dep in (n.get("inputs") or []):
            if dep in id_set:
                adj[dep].append(n["id"])
                indeg[n["id"]] += 1
    queue = [i for i in ids if indeg[i] == 0]
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    errors: list[dict[str, Any]] = []
    if len(order) != len(ids):
        cyclic = [i for i in ids if i not in order]
        errors.append({"error": "cycle or unorderable nodes", "nodes": cyclic})
    return order, errors


def run_pipeline(
    spec: dict[str, Any], image: np.ndarray,
    rois: list[dict[str, Any]] | None = None, priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a pipeline spec on a frame; return each node's step result + any errors.

    ``spec = {"nodes": [{"id", "op", "params", "inputs": [node_ids]}]}``. Executes
    topologically, threading each node's output image to its consumers. Every node's overlay
    and metrics are captured; one node raising is recorded in ``errors`` and never aborts.
    Returns ``{"nodes": [{id, op, metrics, overlay_b64, status}], "errors": [...]}``.
    """
    import cv2

    frame = as_bgr(image)
    clahe_bgr = apply_clahe_lab(frame)
    ctx: dict[str, Any] = {
        "frame": frame, "shape": frame.shape, "clahe_bgr": clahe_bgr,
        "clahe_gray": cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY),
        "rois": rois or [], "priors": priors or {},
        "roi_mask": None, "roi_masks": [], "inputs": [],
    }
    nodes = spec.get("nodes", []) if isinstance(spec, dict) else []
    spec_by_id = {n["id"]: n for n in nodes}
    order, errors = _toposort(nodes)

    results_by_id: dict[str, dict[str, Any]] = {}
    nodes_out: list[dict[str, Any]] = []
    for nid in order:
        n = spec_by_id[nid]
        op = n.get("op")
        params = n.get("params") or {}
        inputs = [i for i in (n.get("inputs") or []) if i in results_by_id]
        in_results = [results_by_id[i] for i in inputs]
        primary = in_results[0]["image"] if in_results else frame
        ctx["inputs"] = in_results
        if op not in OP_REGISTRY:
            errors.append({"id": nid, "op": op, "error": "unknown op"})
            results_by_id[nid] = {"image": primary, "metrics": {}, "op": op}
            nodes_out.append({"id": nid, "op": op, "metrics": {}, "overlay_b64": None,
                              "status": "error", "error": "unknown op"})
            continue
        try:
            out = OP_REGISTRY[op]["fn"](primary, params, ctx)
            img_out = out.get("image")
            if img_out is None:
                img_out = primary
            metrics = out.get("metrics", {})
            results_by_id[nid] = {"image": img_out, "metrics": metrics, "op": op}
            nodes_out.append({"id": nid, "op": op, "metrics": metrics,
                              "overlay_b64": out.get("overlay_b64"),
                              "status": out.get("status", "ok")})
        except Exception as exc:  # noqa: BLE001 - one node failing must never abort the run
            errors.append({"id": nid, "op": op, "error": str(exc)})
            results_by_id[nid] = {"image": primary, "metrics": {}, "op": op}
            nodes_out.append({"id": nid, "op": op, "metrics": {}, "overlay_b64": None,
                              "status": "error", "error": str(exc)})
    return {"nodes": nodes_out, "errors": errors}


# --- TEMPLATES --------------------------------------------------------------------------
TEMPLATES: dict[str, dict[str, Any]] = {
    # gray -> clahe -> denoise -> edge -> roi -> constrained detect -> belt_edges -> measure
    "belt_detection": {
        "focus": "belt_detection",
        "description": (
            "Straight belt edges + centreline via constrained Hough (per-ROI when multiple "
            "ROIs are drawn); extract_belt_edges pairs the two most-separated in-band lines "
            "as the left/right belt edges and reports belt width + centreline."
        ),
        "nodes": [
            {"id": "clahe", "op": "clahe", "params": {}, "inputs": []},
            {"id": "gray", "op": "to_gray", "params": {}, "inputs": ["clahe"]},
            {"id": "denoise", "op": "gaussian_denoise", "params": {}, "inputs": ["gray"]},
            {"id": "edges", "op": "canny", "params": {}, "inputs": ["denoise"]},
            {"id": "roi", "op": "apply_roi",
             "params": {"label": ["expected-belt-limits", "belt-section", "belt"]},
             "inputs": ["edges"]},
            {"id": "lines", "op": "hough_constrained", "params": {}, "inputs": ["roi"]},
            {"id": "belt_edges", "op": "belt_edges", "params": {}, "inputs": ["lines"]},
            {"id": "measure", "op": "measure_lines", "params": {}, "inputs": ["belt_edges"]},
        ],
    },
    # roi=belt -> clahe -> illumination-normalise -> wavelet texture removal -> morphology -> anomaly
    "belt_condition": {
        "focus": "belt_condition",
        "description": "Tear/crack anomaly map along the belt via wavelet texture removal + morphology.",
        "nodes": [
            {"id": "roi", "op": "apply_roi",
             "params": {"label": ["belt-section", "belt", "expected-belt-limits"]}, "inputs": []},
            {"id": "clahe", "op": "clahe", "params": {}, "inputs": ["roi"]},
            {"id": "illum", "op": "illumination_normalize", "params": {}, "inputs": ["clahe"]},
            {"id": "wavelet", "op": "dwt_reconstruct", "params": {"keep": ["detail"]},
             "inputs": ["illum"]},
            {"id": "thresh", "op": "otsu", "params": {}, "inputs": ["wavelet"]},
            {"id": "morph", "op": "close", "params": {"ksize": 5}, "inputs": ["thresh"]},
            {"id": "anomaly", "op": "measure_objects", "params": {}, "inputs": ["morph"]},
        ],
    },
    # the robust auto cascade, drivable with ROIs — the "refine in Studio" target: draw an ROI
    # on the belt (and optionally a 'content' ROI) and re-run the same robust analysis the
    # Precomputed Analysis tab shows, with your ROI focusing the estimate.
    "robust_cascade": {
        "focus": "belt_detection",
        "description": "Robust belt band (orientation consensus + two-limits + Hough) then RGB "
                       "anomaly damage inside it — the auto cascade, focused by your ROI.",
        "nodes": [
            {"id": "roi", "op": "apply_roi",
             "params": {"label": ["expected-belt-limits", "belt-section", "belt"]}, "inputs": []},
            {"id": "band", "op": "robust_belt_band", "params": {}, "inputs": ["roi"]},
            {"id": "damage", "op": "robust_damage", "params": {}, "inputs": ["band"]},
        ],
    },
    # roi=content -> segment -> watershed -> granulometry
    "material_on_belt": {
        "focus": "material_on_belt",
        "description": "Content granulometry (PSD / coverage / count) over the segmented material.",
        "nodes": [
            {"id": "roi", "op": "apply_roi", "params": {"label": ["content"]}, "inputs": []},
            {"id": "clahe", "op": "clahe", "params": {}, "inputs": ["roi"]},
            {"id": "gray", "op": "to_gray", "params": {}, "inputs": ["clahe"]},
            {"id": "segment", "op": "otsu", "params": {}, "inputs": ["gray"]},
            {"id": "clean", "op": "open", "params": {"ksize": 3}, "inputs": ["segment"]},
            {"id": "granulometry", "op": "granulometry", "params": {}, "inputs": ["clean"]},
        ],
    },
}


def list_templates() -> list[str]:
    """The ready-made pipeline template names."""
    return list(TEMPLATES)


def get_template(name: str) -> dict[str, Any]:
    """Return a deep copy of a template spec (safe to mutate)."""
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; known: {list_templates()}")
    return copy.deepcopy(TEMPLATES[name])


__all__ = [
    "OP_REGISTRY", "TEMPLATES", "run_pipeline", "list_ops", "op_catalog",
    "list_templates", "get_template",
]
