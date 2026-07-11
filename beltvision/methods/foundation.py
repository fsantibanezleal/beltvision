"""Capability tier ``beyond_sota``: the open-vocabulary / foundation-model frontier.

These methods run REAL foundation models (DINOv2, Depth-Anything-V2, OWLv2, GroundingDINO
+ SAM, SAM2) on the GPU in the offline PRECOMPUTE lane. Each produces a legible drawn
overlay + a scalar metric and is returned as the same uniform record the classical / sota
methods are (``id, capability, tier, family, name, reference, metric_name, metric_value,
summary, overlay_b64, status, extra``), so the serving product persists the overlay as a
compact JPEG and REPLAYS it (ADR-0014) with zero live compute.

Two surfaces live here:

- ``*_record(bgr, *, device, ...)`` - the heavy producers the precompute lane calls. They
  lazily import torch / transformers / ultralytics INSIDE the function, cache each loaded
  model module-wide, cap the input long-side for VRAM, and run on ``device='cuda'``. A
  genuine failure (a model that will not install / run) RAISES so the precompute wrapper
  records it in ``errors`` and skips it - never a fabricated or empty-overlay entry.
- ``grounded_sam`` / ``dinov2`` / ``sam2`` / ``owlv2`` / ``depth_anything_v2`` /
  ``dinov2_knn`` - the LIVE-ladder registry callables. The slim CPU/VPS runtime does NOT
  host these foundation models, so live they return a graceful ``weights_absent`` envelope
  (never raise, never download); catalogue cases replay the precomputed overlay instead.

References: Oquab et al. 2023 (DINOv2, arXiv:2304.07193); Yang et al. 2024 (Depth-Anything-V2,
arXiv:2406.09414); Minderer et al. 2023 (OWLv2, arXiv:2306.09683); Liu et al. 2023
(Grounding-DINO, arXiv:2303.05499); Kirillov et al. 2023 (SAM); Ravi et al. 2024 (SAM 2,
arXiv:2408.00714).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._common import weights_absent

# --- module-wide model cache (loaded once, reused across the 22-case precompute) ----------
_MODELS: dict[str, Any] = {}

_MB = 1024 * 1024

# belt-domain open-vocabulary prompts (lowercase phrases for the text-conditioned models).
_OPENVOCAB_PHRASES = (
    "conveyor belt",
    "rock ore mineral pile",
    "a piece of wood",
    "a metal tool or wrench",
    "a foreign object",
    "a person",
)

# a stable BGR palette for per-instance masks / boxes.
_PALETTE_BGR = (
    (60, 180, 75), (40, 120, 235), (200, 60, 200), (60, 255, 255),
    (235, 120, 40), (75, 75, 255), (200, 200, 60), (140, 60, 220),
    (60, 220, 140), (0, 165, 255),
)


def _record(
    *,
    method_id: str,
    capability: str,
    family: str,
    name: str,
    reference: str,
    metric_name: str,
    metric_value: float | None,
    summary: str,
    overlay_b64: str | None,
    tier: str = "beyond_sota",
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One uniform per-method record (same shape the precompute ``_entry`` emits)."""
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


def _cap_long_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale a BGR frame so its long side is at most ``max_side`` (VRAM cap)."""
    import cv2

    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / float(max(h, w))
    return cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _to_pil(bgr: np.ndarray):
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# =========================================================================================
# DINOv2 dense patch features (shared by the feature overlay + the kNN anomaly)
# =========================================================================================
_DINOV2_NAME = "facebook/dinov2-base"
_DINOV2_PATCH = 14


def _load_dinov2(device: str):
    if "dinov2" not in _MODELS:
        from transformers import AutoModel

        _MODELS["dinov2"] = AutoModel.from_pretrained(_DINOV2_NAME).to(device).eval()
    return _MODELS["dinov2"]


def _dinov2_grid(bgr: np.ndarray, device: str, side: int = 448) -> tuple[np.ndarray, int, int]:
    """Return DINOv2 dense patch features as a ``(gh, gw, D)`` float32 grid.

    Manual ImageNet preprocessing to a fixed ``side`` (multiple of the patch size) with
    ``interpolate_pos_encoding`` so the patch grid is predictable and reasonably dense.
    """
    import cv2
    import torch

    model = _load_dinov2(device)
    side = (side // _DINOV2_PATCH) * _DINOV2_PATCH
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (resized - mean) / std
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
    with torch.no_grad():
        out = model(pixel_values=x, interpolate_pos_encoding=True)
    tokens = out.last_hidden_state[0, 1:, :].float().cpu().numpy()  # drop CLS -> (P, D)
    g = side // _DINOV2_PATCH
    feat = tokens.reshape(g, g, tokens.shape[-1])
    _empty_cache()
    return feat.astype(np.float32), g, g


def _pca_rgb(feat: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a ``(gh, gw, D)`` feature grid to a ``(gh, gw, 3)`` uint8 PCA-RGB map.

    Returns the map and the fraction of variance the first 3 components explain.
    """
    gh, gw, d = feat.shape
    flat = feat.reshape(gh * gw, d).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # economy SVD; columns of Vt are the principal directions.
    _u, s, vt = np.linalg.svd(flat, full_matrices=False)
    comps = flat @ vt[:3].T  # (P, 3)
    ev = float((s[:3] ** 2).sum() / (s ** 2).sum()) if s.size else 0.0
    rgb = np.zeros_like(comps)
    for c in range(3):
        ch = comps[:, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[:, c] = np.clip((ch - lo) / (hi - lo + 1e-9), 0.0, 1.0) if hi > lo else 0.0
    img = (rgb.reshape(gh, gw, 3) * 255.0).astype(np.uint8)
    return img, ev


def dinov2_record(bgr: np.ndarray, *, device: str = "cuda", max_side: int = 896) -> dict[str, Any]:
    """DINOv2 dense patch features -> PCA(1-3) -> RGB feature overlay upsampled to the frame."""
    import cv2

    from .. import render

    frame = _cap_long_side(bgr, max_side)
    feat, gh, gw = _dinov2_grid(frame, device)
    pca_rgb, ev = _pca_rgb(feat)
    h, w = frame.shape[:2]
    up = cv2.resize(pca_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    # BGR for the pipeline; the PCA channels are arbitrary-signed so treat as pseudo-colour.
    up_bgr = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)
    ov = cv2.addWeighted(frame, 0.4, up_bgr, 0.6, 0)
    summ = (f"DINOv2 (ViT-B/14, self-supervised) dense patch features: {feat.shape[-1]}-d per "
            f"{gh}x{gw} patch, PCA(1-3)->RGB. Top-3 components explain {ev*100:.0f}% of the "
            "feature variance; similar colours = semantically similar surface.")
    render.draw_legend(ov, [((180, 180, 180), "DINOv2 PCA(1-3) -> RGB feature map")])
    render.draw_summary(ov, summ)
    return _record(
        method_id="features.dinov2", capability="features", family="foundation_feature",
        name="DINOv2 dense patch features (PCA-RGB)",
        reference="DINOv2, Oquab et al. 2023 (arXiv:2304.07193); facebook/dinov2-base",
        metric_name="explained_variance", metric_value=ev,
        summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"feature_dim": int(feat.shape[-1]), "patch_grid": [gh, gw],
               "model": _DINOV2_NAME},
    )


def dinov2_knn_record(
    bgr: np.ndarray, *, device: str = "cuda", footprint: np.ndarray | None = None,
    max_side: int = 896, k: int = 8,
) -> dict[str, Any]:
    """DINOv2 patch features + per-patch kNN cosine distance -> foundation anomaly heatmap.

    Each patch is scored by its mean cosine distance to its ``k`` nearest neighbours among
    the frame's own patches: patches in the dominant surface cluster read low, an off-manifold
    region (a foreign object, a defect) reads high. The foundation-feature analogue of PatchCore.
    """
    import cv2

    from .. import render

    frame = _cap_long_side(bgr, max_side)
    feat, gh, gw = _dinov2_grid(frame, device)
    h, w = frame.shape[:2]
    x = feat.reshape(gh * gw, feat.shape[-1]).astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    sim = x @ x.T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, np.inf)
    kk = int(max(1, min(k, dist.shape[0] - 1)))
    part = np.partition(dist, kk, axis=1)[:, :kk]
    score = part.mean(axis=1).reshape(gh, gw)  # (gh, gw) anomaly

    # restrict the PEAK to the belt footprint when it is available (relevance), but always
    # display the full heatmap so nothing is fabricated or hidden.
    peak_score = score.copy()
    if footprint is not None and int(np.asarray(footprint).sum()) > 0:
        fp = cv2.resize(np.asarray(footprint).astype(np.uint8), (gw, gh),
                        interpolation=cv2.INTER_NEAREST) > 0
        if fp.any():
            peak_score = np.where(fp, score, score.min())
    pi = int(np.argmax(peak_score))
    pr, pc = pi // gw, pi % gw
    peak = (int((pc + 0.5) / gw * w), int((pr + 0.5) / gh * h))
    smax = float(score.max())
    summ = (f"DINOv2 kNN patch anomaly (k={kk}, cosine): max nearest-neighbour distance "
            f"{smax:.3f} over the {gh}x{gw} patch grid; hot = off the frame's own feature "
            "manifold (a foundation-feature analogue of PatchCore).")
    ov = render.heatmap_overlay(
        frame, score, legend_label="DINOv2 kNN distance", peak_xy=peak,
        title="DINOv2 anomaly", summary=summ,
    )
    return _record(
        method_id="anomaly.dinov2_knn", capability="anomaly", family="foundation_anomaly",
        name="DINOv2 kNN patch anomaly (AnomalyDINO-style)",
        reference="AnomalyDINO (Damm et al. 2024) over DINOv2 features (Oquab et al. 2023); PatchCore analogue",
        metric_name="max_nn_distance", metric_value=smax,
        summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"patch_grid": [gh, gw], "k": kk, "model": _DINOV2_NAME},
    )


# =========================================================================================
# Depth-Anything-V2 monocular depth
# =========================================================================================
_DEPTH_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


def _load_depth(device: str):
    if "depth" not in _MODELS:
        from transformers import pipeline

        _MODELS["depth"] = pipeline(
            "depth-estimation", model=_DEPTH_NAME, device=(0 if device.startswith("cuda") else -1),
        )
    return _MODELS["depth"]


def depth_anything_record(
    bgr: np.ndarray, *, device: str = "cuda", max_side: int = 896
) -> dict[str, Any]:
    """Depth-Anything-V2-Small monocular relative depth -> colorised depth-relief overlay."""
    import cv2

    from .. import render

    frame = _cap_long_side(bgr, max_side)
    pipe = _load_depth(device)
    out = pipe(_to_pil(frame))
    pred = out.get("predicted_depth")
    if pred is not None:
        depth = pred.squeeze().float().cpu().numpy()
    else:
        depth = np.asarray(out["depth"], dtype=np.float32)
    depth = cv2.resize(depth.astype(np.float32), (frame.shape[1], frame.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    dmin, dmax = float(depth.min()), float(depth.max())
    drange = dmax - dmin
    summ = (f"Depth-Anything-V2 (ViT-S) monocular relative depth: relative range {drange:.1f} "
            "(near = bright). Reads the belt-surface + load relief from a single RGB frame.")
    ov = render.heatmap_overlay(
        frame, depth, legend_label="relative depth (near->far)", summary=summ,
        title="Monocular depth", colormap=cv2.COLORMAP_MAGMA, alpha=0.6,
    )
    _empty_cache()
    return _record(
        method_id="depth.depth_anything_v2", capability="depth", family="monocular_depth",
        name="Depth-Anything-V2 monocular depth",
        reference="Depth-Anything-V2, Yang et al. 2024 (arXiv:2406.09414); depth-anything/Depth-Anything-V2-Small-hf",
        metric_name="relative_depth_range", metric_value=drange,
        summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"depth_min": round(dmin, 3), "depth_max": round(dmax, 3), "model": _DEPTH_NAME},
    )


# =========================================================================================
# OWLv2 open-vocabulary detection
# =========================================================================================
_OWLV2_NAME = "google/owlv2-base-patch16-ensemble"


def _load_owlv2(device: str):
    if "owlv2" not in _MODELS:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        proc = Owlv2Processor.from_pretrained(_OWLV2_NAME)
        model = Owlv2ForObjectDetection.from_pretrained(_OWLV2_NAME).to(device).eval()
        _MODELS["owlv2"] = (proc, model)
    return _MODELS["owlv2"]


def owlv2_record(
    bgr: np.ndarray, *, device: str = "cuda", max_side: int = 840, threshold: float = 0.12
) -> dict[str, Any]:
    """OWLv2 open-vocabulary detection of belt-domain objects by text prompt -> boxes + labels."""
    import cv2
    import torch

    from .. import render

    frame = _cap_long_side(bgr, max_side)
    proc, model = _load_owlv2(device)
    phrases = list(_OPENVOCAB_PHRASES)
    pil = _to_pil(frame)
    inputs = proc(text=[phrases], images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([[pil.size[1], pil.size[0]]], device=device)
    res = proc.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target, threshold=threshold, text_labels=[phrases],
    )[0]
    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    labels = res.get("text_labels")
    if labels is None:
        idx = res["labels"].cpu().numpy()
        labels = [phrases[int(i)] for i in idx]
    ov = frame.copy()
    order = np.argsort(-scores)[:20]
    for rank, j in enumerate(order):
        x0, y0, x1, y1 = (int(v) for v in boxes[j])
        col = _PALETTE_BGR[rank % len(_PALETTE_BGR)]
        cv2.rectangle(ov, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        render._text(ov, f"{labels[j]} {scores[j]:.2f}", (x0 + 3, max(14, y0 - 5)),
                     scale=render._fs(ov), color=col)
    n = int(len(order))
    summ = (f"OWLv2 open-vocabulary detection ({len(phrases)} belt-domain text prompts): "
            f"{n} object(s) above {threshold:.2f}. Detects foreign objects / people by text, "
            "no belt-specific training.")
    render.draw_legend(ov, [((0, 200, 255), "open-vocab detection (text-prompted)")])
    render.draw_summary(ov, summ)
    _empty_cache()
    top = [{"label": labels[int(j)], "score": round(float(scores[int(j)]), 3)} for j in order[:8]]
    return _record(
        method_id="detection.owlv2", capability="detection", family="open_vocab_detection",
        name="OWLv2 open-vocabulary detection",
        reference="OWLv2, Minderer et al. 2023 (arXiv:2306.09683); google/owlv2-base-patch16-ensemble",
        metric_name="n_detections", metric_value=float(n),
        summary=summ, overlay_b64=render.to_png_b64(ov),
        extra={"threshold": threshold, "prompts": phrases, "top": top, "model": _OWLV2_NAME},
    )


# =========================================================================================
# GroundedSAM: GroundingDINO open-vocab boxes -> SAM masks
# =========================================================================================
_GDINO_NAME = "IDEA-Research/grounding-dino-tiny"
_SAM_NAME = "facebook/sam-vit-base"


def _load_gdino(device: str):
    if "gdino" not in _MODELS:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        proc = AutoProcessor.from_pretrained(_GDINO_NAME)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_NAME).to(device).eval()
        _MODELS["gdino"] = (proc, model)
    return _MODELS["gdino"]


def _load_sam(device: str):
    if "sam" not in _MODELS:
        from transformers import SamModel, SamProcessor

        proc = SamProcessor.from_pretrained(_SAM_NAME)
        model = SamModel.from_pretrained(_SAM_NAME).to(device).eval()
        _MODELS["sam"] = (proc, model)
    return _MODELS["sam"]


def _gdino_boxes(frame: np.ndarray, device: str, box_threshold: float, text_threshold: float):
    import torch

    proc, model = _load_gdino(device)
    pil = _to_pil(frame)
    text = ". ".join(_OPENVOCAB_PHRASES) + " ."
    inputs = proc(images=pil, text=text.lower(), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target = [(pil.size[1], pil.size[0])]
    try:
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=target,
        )[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=target,
        )[0]
    boxes = res["boxes"].cpu().numpy()
    labels = res.get("text_labels") or res.get("labels")
    scores = res["scores"].cpu().numpy()
    return boxes, list(labels), scores


def _sam_masks_for_boxes(frame: np.ndarray, boxes: np.ndarray, device: str) -> np.ndarray:
    import torch

    proc, model = _load_sam(device)
    pil = _to_pil(frame)
    box_list = [[float(v) for v in b] for b in boxes]
    inputs = proc(pil, input_boxes=[box_list], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # (n_boxes, n_multi, H, W) bool
    iou = outputs.iou_scores.cpu().numpy()[0]  # (n_boxes, n_multi)
    best = iou.argmax(axis=1)
    out = np.stack([masks[i, best[i]].numpy() for i in range(masks.shape[0])], axis=0)
    return out.astype(bool)


def grounded_sam_record(
    bgr: np.ndarray, *, device: str = "cuda", weights_dir: str | Path | None = None,
    max_side: int = 896, box_threshold: float = 0.2, text_threshold: float = 0.18,
    max_boxes: int = 10,
) -> dict[str, Any]:
    """GroundingDINO open-vocab boxes from belt-domain prompts -> SAM masks (colored + labeled)."""
    import cv2

    from .. import render

    frame = _cap_long_side(bgr, max_side)
    h, w = frame.shape[:2]
    boxes, labels, scores = _gdino_boxes(frame, device, box_threshold, text_threshold)
    if boxes.shape[0] == 0:
        _empty_cache()
        summ = ("GroundedSAM (GroundingDINO + SAM): no open-vocabulary object above threshold on "
                "this frame for the belt-domain prompts.")
        ov = render.message_overlay(frame, "GroundedSAM (open-vocab segmentation)", summ)
        return _record(
            method_id="segmentation.grounded_sam", capability="segmentation",
            family="open_vocab_segmentation", name="GroundedSAM open-vocab segmentation",
            reference="Grounding-DINO (Liu 2023, arXiv:2303.05499) + SAM (Kirillov 2023)",
            metric_name="n_masks", metric_value=0.0, summary=summ,
            overlay_b64=render.to_png_b64(ov),
            extra={"model": f"{_GDINO_NAME} + {_SAM_NAME}", "belt_coverage_frac": 0.0},
        )
    keep = np.argsort(-scores)[:max_boxes]
    boxes, labels = boxes[keep], [labels[int(i)] for i in keep]
    scores = scores[keep]
    masks = _sam_masks_for_boxes(frame, boxes, device)

    color = np.zeros_like(frame)
    belt_px = 0
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            masks[i] = m
        color[m] = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        if "belt" in labels[i]:
            belt_px = max(belt_px, int(m.sum()))
    ov = cv2.addWeighted(frame, 0.55, color, 0.45, 0)
    for i in range(masks.shape[0]):
        ys, xs = np.nonzero(masks[i])
        if xs.size == 0:
            continue
        col = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, col, 2, cv2.LINE_AA)
        render._text(ov, f"{labels[i]} {scores[i]:.2f}", (int(xs.min()) + 3,
                     max(14, int(ys.min()) - 5)), scale=render._fs(ov), color=col)
    n = int(masks.shape[0])
    belt_cov = belt_px / float(h * w)
    summ = (f"GroundedSAM (GroundingDINO open-vocab boxes -> SAM masks): {n} object mask(s) from "
            f"belt-domain text prompts; belt coverage {belt_cov*100:.0f}% of frame. Open-vocabulary, "
            "no belt-specific training.")
    render.draw_legend(ov, [((200, 200, 200), "SAM mask boundary (open-vocab labelled)")])
    render.draw_summary(ov, summ)
    _empty_cache()
    top = [{"label": labels[i], "score": round(float(scores[i]), 3)} for i in range(min(n, 8))]
    return _record(
        method_id="segmentation.grounded_sam", capability="segmentation",
        family="open_vocab_segmentation", name="GroundedSAM open-vocab segmentation",
        reference="Grounding-DINO (Liu 2023, arXiv:2303.05499) + SAM (Kirillov 2023); grounding-dino-tiny + sam-vit-base",
        metric_name="n_masks", metric_value=float(n), summary=summ,
        overlay_b64=render.to_png_b64(ov),
        extra={"belt_coverage_frac": round(belt_cov, 4), "detections": top,
               "model": f"{_GDINO_NAME} + {_SAM_NAME}"},
    )


# =========================================================================================
# SAM2 automatic ("segment everything") masks
# =========================================================================================
_SAM2_WEIGHT = "sam2_b.pt"


def _sam2_model(device: str, weights_dir: str | Path | None):
    if "sam2" not in _MODELS:
        from ultralytics import SAM

        wpath = None
        if weights_dir is not None:
            base = Path(weights_dir)
            for cand in (base / _SAM2_WEIGHT, base / "weights" / _SAM2_WEIGHT):
                if cand.is_file():
                    wpath = str(cand)
                    break
        model = SAM(wpath or _SAM2_WEIGHT)  # ultralytics fetches sam2_b.pt by name if absent
        _MODELS["sam2"] = model
    return _MODELS["sam2"]


def sam2_record(
    bgr: np.ndarray, *, device: str = "cuda", weights_dir: str | Path | None = None,
    max_side: int = 1024,
) -> dict[str, Any]:
    """SAM2 automatic mask generation -> mask-boundary overlay (higher quality than MobileSAM)."""
    import cv2
    from skimage.segmentation import find_boundaries

    from .. import render

    frame = _cap_long_side(bgr, max_side)
    h, w = frame.shape[:2]
    model = _sam2_model(device, weights_dir)
    dev = 0 if str(device).startswith("cuda") else "cpu"
    results = model(frame[:, :, ::-1], verbose=False, device=dev)
    label_map = np.zeros((h, w), dtype=np.int32)
    n = 0
    total = 0
    for res in results:
        if res.masks is None:
            continue
        data = res.masks.data.cpu().numpy()  # (n, H, W)
        for m in data:
            mb = m > 0.5
            if mb.shape != (h, w):
                mb = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            area = int(mb.sum())
            if area < 0.0008 * h * w:
                continue
            n += 1
            total += area
            label_map[mb] = n
    if n == 0:
        raise RuntimeError("SAM2 produced no masks")
    color = np.zeros_like(frame)
    rng = np.random.default_rng(34)
    for lid in range(1, n + 1):
        color[label_map == lid] = tuple(int(c) for c in rng.integers(70, 256, size=3))
    ov = cv2.addWeighted(frame, 0.55, color, 0.45, 0)
    ov[find_boundaries(label_map, mode="outer")] = (255, 255, 255)
    cov = total / float(h * w)
    summ = (f"SAM 2 automatic mask generation (Hiera image encoder): {n} class-agnostic masks, "
            f"{cov*100:.0f}% frame coverage. A stronger prompt-free segmenter than MobileSAM.")
    render.draw_legend(ov, [((255, 255, 255), "SAM 2 mask boundary"),
                            ((0, 120, 235), "segmented region")])
    render.draw_summary(ov, summ)
    _empty_cache()
    return _record(
        method_id="segmentation.sam2", capability="segmentation", family="foundation_segmentation",
        name="SAM 2 automatic masks",
        reference="SAM 2, Ravi et al. 2024 (arXiv:2408.00714); ultralytics sam2_b.pt",
        metric_name="n_masks", metric_value=float(n), summary=summ,
        overlay_b64=render.to_png_b64(ov),
        extra={"coverage_frac": round(cov, 4), "weight": _SAM2_WEIGHT},
    )


# =========================================================================================
# LIVE-ladder registry callables (precompute/GPU-only -> graceful weights_absent live)
# =========================================================================================
def _foundation_absent(
    method_id: str, capability: str, reference: str, approx_bytes: int
) -> dict[str, Any]:
    return weights_absent(
        method_id, capability, "foundation", reference,
        weight=f"{method_id} (HuggingFace / ultralytics foundation weights)",
        searched=["<precompute/GPU lane only>"], approx_bytes=int(approx_bytes),
        web_drivable=False,
        hint=("open-vocabulary / foundation-model tier: hosted only in the offline precompute "
              "(GPU, device='cuda') lane and REPLAYED as committed overlays for catalogue cases. "
              "Not served live on the CPU/VPS runtime. Provision via the beltvision [gpu] extra "
              "and run beltvision.methods.foundation.*_record on cuda."),
    )


def grounded_sam(image: Any, **_: Any) -> dict[str, Any]:
    """LIVE surface for GroundedSAM (precompute/GPU-only -> graceful weights_absent)."""
    return _foundation_absent(
        "segmentation.grounded_sam", "segmentation",
        "Grounding-DINO (arXiv:2303.05499) + SAM (Kirillov 2023)", 900 * _MB)


def dinov2(image: Any, **_: Any) -> dict[str, Any]:
    """LIVE surface for DINOv2 dense features (precompute/GPU-only -> graceful weights_absent)."""
    return _foundation_absent(
        "features.dinov2", "features", "DINOv2 (arXiv:2304.07193)", 350 * _MB)


def sam2(image: Any, **_: Any) -> dict[str, Any]:
    """LIVE surface for SAM 2 automatic masks (precompute/GPU-only -> graceful weights_absent)."""
    return _foundation_absent(
        "segmentation.sam2", "segmentation", "SAM 2 (arXiv:2408.00714)", 320 * _MB)


def owlv2(image: Any, **_: Any) -> dict[str, Any]:
    """LIVE surface for OWLv2 open-vocab detection (precompute/GPU-only -> graceful weights_absent)."""
    return _foundation_absent(
        "detection.owlv2", "detection", "OWLv2 (arXiv:2306.09683)", 640 * _MB)


def depth_anything_v2(image: Any, **_: Any) -> dict[str, Any]:
    """LIVE surface for Depth-Anything-V2 (precompute/GPU-only -> graceful weights_absent)."""
    return _foundation_absent(
        "depth.depth_anything_v2", "depth", "Depth-Anything-V2 (arXiv:2406.09414)", 100 * _MB)


def dinov2_knn(image: Any, **_: Any) -> dict[str, Any]:
    """LIVE surface for DINOv2 kNN anomaly (precompute/GPU-only -> graceful weights_absent)."""
    return _foundation_absent(
        "anomaly.dinov2_knn", "anomaly", "AnomalyDINO / DINOv2 (arXiv:2304.07193)", 350 * _MB)


__all__ = [
    "dinov2_record",
    "dinov2_knn_record",
    "depth_anything_record",
    "owlv2_record",
    "grounded_sam_record",
    "sam2_record",
    "grounded_sam",
    "dinov2",
    "sam2",
    "owlv2",
    "depth_anything_v2",
    "dinov2_knn",
]
