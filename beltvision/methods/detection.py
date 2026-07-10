"""Capability 2a: object detection via ONNX Runtime CPU (>= 1 learned, LIVE).

``detection.onnx_detector`` (M5) runs a permissive, general/COCO-pretrained real-time
detector through ONNX Runtime on the CPU. The preferred model is RT-DETR (Apache-2.0,
NMS-free); a permissive YOLO-family ONNX is also supported via the ``format`` param. The
ONNX file is provisioned by ``beltvision.models.download`` (opt-in httpx fetch or local
drop-in / export from the ``[dl]`` lane). Absent a weight, the method returns a graceful
``weights_absent`` result (never an exception), and reports the expected model size so the
gate still classifies the lane honestly.

Reference: RT-DETR arXiv:2304.08069 (Apache-2.0); COCO labels are general-pretrained transfer.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models import ensure_weight, search_dirs, weight_path
from ._common import as_bgr, cap, result, timed, weights_absent

_MAX_BOXES = 100
_REF = "RT-DETR arXiv:2304.08069 (Apache-2.0); COCO-pretrained general transfer"

# COCO-80 class names (the general-pretrained label space).
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


def _letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    import cv2

    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, px, py


def _label(cls_id: int) -> str:
    return COCO_LABELS[cls_id] if 0 <= cls_id < len(COCO_LABELS) else f"class_{cls_id}"


def onnx_detector(
    image: Any,
    *,
    weights: str | None = None,
    score_threshold: float = 0.35,
    fmt: str = "rtdetr",
    download: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Run ONNX Runtime CPU detection on a frame -> boxes + scores + labels."""
    from ..models.download import WEIGHTS

    spec = WEIGHTS["detector_onnx"]
    path = weights if weights else ensure_weight("detector_onnx", download=download)
    searched = [weights] if weights else search_dirs()
    if path is None or not _exists(path):
        return weights_absent(
            "detection.onnx_detector", "detection", "learned", _REF,
            weight="detector.onnx", searched=searched, approx_bytes=spec.approx_bytes,
            web_drivable=True,
            hint=(
                f"provide an Apache-2.0 detector ONNX at {weight_path('detector_onnx')} "
                "(BELTVISION_DETECTOR_ONNX_URL for opt-in httpx download, or export RT-DETR "
                "via the [dl] extra). Absent => weights_absent."
            ),
        )

    bgr = as_bgr(image)
    h, w = bgr.shape[:2]
    try:
        with timed() as t:
            boxes, scores, labels, in_size = _run_detector(bgr, str(path), score_threshold, fmt)
        size = _filesize(path)
    except Exception as exc:  # noqa: BLE001 - broken/mismatched model degrades gracefully
        return weights_absent(
            "detection.onnx_detector", "detection", "learned", _REF,
            weight="detector.onnx", searched=[str(path)], approx_bytes=spec.approx_bytes,
            web_drivable=True, hint=f"weight present but inference failed: {type(exc).__name__}: {exc}",
        )

    dets = [
        {
            "bbox_xyxy": [round(float(v), 2) for v in box],
            "score": round(float(sc), 4),
            "label": _label(int(cl)),
            "label_id": int(cl),
        }
        for box, sc, cl in zip(boxes, scores, labels, strict=False)
    ]
    dets.sort(key=lambda d: d["score"], reverse=True)
    payload = {
        "shape": [int(h), int(w)],
        "format": fmt,
        "input_size": int(in_size),
        "score_threshold": float(score_threshold),
        "n_detections": len(dets),
        "detections": cap(dets, _MAX_BOXES),
        "label_space": "COCO-80 (general-pretrained transfer)",
    }
    return result(
        "detection.onnx_detector", "detection", "learned", _REF,
        payload=payload, model_bytes=size, infer_ms=t.ms,
        web_drivable=size <= 40 * 1024 * 1024,
    )


def _run_detector(
    bgr: np.ndarray, path: str, thr: float, fmt: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    size = int(shape[-1]) if isinstance(shape[-1], int) and shape[-1] > 1 else 640
    canvas, scale, px, py = _letterbox(bgr, size)
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW RGB
    outputs = sess.run(None, {inp.name: blob})
    h, w = bgr.shape[:2]
    if fmt == "rtdetr":
        boxes, scores, labels = _decode_rtdetr(outputs, size, thr)
    else:
        boxes, scores, labels = _decode_yolo(outputs, thr)
    # Undo letterbox back to original pixel coordinates.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    return boxes, scores, labels, size


def _decode_rtdetr(outputs: list[np.ndarray], size: int, thr: float):
    """RT-DETR head: normalized cxcywh boxes + per-class logits/scores, NMS-free."""
    arrs = [np.asarray(o) for o in outputs]
    boxes_raw = next(a for a in arrs if a.ndim == 3 and a.shape[-1] == 4)[0]
    logits = next(a for a in arrs if a.ndim == 3 and a.shape[-1] != 4)[0]
    scores_all = logits if logits.max() <= 1.0 + 1e-6 else 1.0 / (1.0 + np.exp(-logits))
    cls = scores_all.argmax(axis=1)
    conf = scores_all.max(axis=1)
    keep = conf >= thr
    b = boxes_raw[keep]
    xyxy = np.empty_like(b)
    if b.shape[0]:
        cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        s = size if b.max() <= 1.5 else 1.0  # normalized -> pixels
        xyxy[:, 0] = (cx - bw / 2) * s
        xyxy[:, 1] = (cy - bh / 2) * s
        xyxy[:, 2] = (cx + bw / 2) * s
        xyxy[:, 3] = (cy + bh / 2) * s
    return xyxy, conf[keep], cls[keep]


def _decode_yolo(outputs: list[np.ndarray], thr: float):
    """YOLO head: [1, 4+nc, N] or [1, N, 4+nc] cxcywh + class scores, with NMS."""
    import cv2

    pred = np.asarray(outputs[0])
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:  # (4+nc, N) -> (N, 4+nc)
        pred = pred.T
    boxes_cxcywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    conf = cls_scores.max(axis=1)
    cls = cls_scores.argmax(axis=1)
    keep = conf >= thr
    b = boxes_cxcywh[keep]
    conf, cls = conf[keep], cls[keep]
    xyxy = np.empty_like(b)
    if b.shape[0]:
        xyxy[:, 0] = b[:, 0] - b[:, 2] / 2
        xyxy[:, 1] = b[:, 1] - b[:, 3] / 2
        xyxy[:, 2] = b[:, 0] + b[:, 2] / 2
        xyxy[:, 3] = b[:, 1] + b[:, 3] / 2
        idx = cv2.dnn.NMSBoxes(
            [[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in xyxy],
            conf.tolist(), thr, 0.45,
        )
        idx = np.asarray(idx).reshape(-1) if len(idx) else np.array([], dtype=int)
        return xyxy[idx], conf[idx], cls[idx]
    return xyxy, conf, cls


def _exists(path: Any) -> bool:
    from pathlib import Path

    return path is not None and Path(str(path)).is_file()


def _filesize(path: Any) -> int:
    from pathlib import Path

    p = Path(str(path))
    return p.stat().st_size if p.is_file() else 0
