"""Capability 3: segmentation.

- ``segmentation.slic`` (M19) - SLIC superpixels, classical, LIVE-SERVER, always runs.
- ``segmentation.mobile_sam`` (M17) - MobileSAM / FastSAM automatic mask generation, a
  learned foundation-model method behind the ``[dl]`` extra. Loads a permissive weight
  (MobileSAM Tiny-ViT, ~40 MB, Apache-2.0) provisioned by ``beltvision.models.download``
  and runs on CPU. If the weight or the DL runtime is absent it returns a graceful
  ``weights_absent`` result (never an exception).

References: Achanta et al. 2012 (SLIC), TPAMI 34(11); MobileSAM
https://github.com/ChaoningZhang/MobileSAM ; FastSAM https://docs.ultralytics.com/models/fast-sam/.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models import ensure_weight, search_dirs, weight_path
from ._common import as_bgr, cap, result, timed, weights_absent
from .preprocess import apply_clahe_lab

_MAX_MASKS = 48


def slic(
    image: Any, *, n_segments: int = 200, compactness: float = 10.0, sigma: float = 1.0, **_: Any
) -> dict[str, Any]:
    """SLIC superpixel over-segmentation (classical pre-seg / cheap region proposer)."""
    from skimage.color import rgb2lab
    from skimage.segmentation import slic as sk_slic

    bgr = apply_clahe_lab(as_bgr(image))
    rgb = bgr[:, :, ::-1]
    h, w = bgr.shape[:2]
    with timed() as t:
        lab = rgb2lab(rgb / 255.0)
        labels = sk_slic(
            lab,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=0,
            channel_axis=-1,
        )
        ids, counts = np.unique(labels, return_counts=True)
        n_actual = int(ids.size)
        sizes = counts.astype(np.float64)
    payload = {
        "shape": [int(h), int(w)],
        "n_segments_requested": int(n_segments),
        "n_segments_actual": n_actual,
        "compactness": float(compactness),
        "segment_size_px": {
            "mean": round(float(sizes.mean()), 2),
            "std": round(float(sizes.std()), 2),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
        },
        "coverage_px": int(sizes.sum()),
    }
    return result(
        "segmentation.slic", "segmentation", "classical",
        "Achanta et al. 2012 (SLIC superpixels), TPAMI 34(11)",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=False,
    )


def _summarize_masks(masks: list[dict[str, Any]], h: int, w: int) -> dict[str, Any]:
    entries = []
    total = 0
    for m in masks:
        area = int(m.get("area", 0))
        total += area
        entries.append(
            {
                "bbox_xywh": [int(v) for v in m.get("bbox", [0, 0, 0, 0])],
                "area_px": area,
                "stability": round(float(m.get("stability_score", 0.0)), 4),
            }
        )
    entries.sort(key=lambda e: e["area_px"], reverse=True)
    return {
        "n_masks": len(masks),
        "coverage_frac": round(total / max(h * w, 1), 4),
        "masks": cap(entries, _MAX_MASKS),
    }


def mobile_sam(
    image: Any, *, weights: str | None = None, download: bool = False, **_: Any
) -> dict[str, Any]:
    """MobileSAM/FastSAM automatic masks (learned, [dl]); graceful weights_absent."""
    ref = "MobileSAM (Tiny-ViT 5M), Apache-2.0; FastSAM (YOLOv8-seg)"
    from ..models.download import WEIGHTS

    spec = WEIGHTS["mobile_sam"]
    path = weights if weights else ensure_weight("mobile_sam", download=download)
    searched = [weights] if weights else search_dirs()
    if path is None or not _exists(path):
        return weights_absent(
            "segmentation.mobile_sam", "segmentation", "learned", ref,
            weight="mobile_sam.pt", searched=searched, approx_bytes=spec.approx_bytes,
            web_drivable=False,
            hint=(
                f"provision the Apache-2.0 MobileSAM weight into {weight_path('mobile_sam')} "
                "(pip install -e .[dl]); download.py can fetch it via httpx (opt-in)."
            ),
        )

    bgr = apply_clahe_lab(as_bgr(image))
    h, w = bgr.shape[:2]
    try:
        with timed() as t:
            masks = _run_sam(bgr, str(path))
        size = _filesize(path)
    except Exception as exc:  # noqa: BLE001 - a broken/incompatible weight degrades gracefully
        return weights_absent(
            "segmentation.mobile_sam", "segmentation", "learned", ref,
            weight="mobile_sam.pt", searched=[str(path)], approx_bytes=spec.approx_bytes,
            web_drivable=False, hint=f"weight present but inference failed: {type(exc).__name__}: {exc}",
        )
    payload = {"shape": [int(h), int(w)], "backend": "ultralytics-SAM", **_summarize_masks(masks, h, w)}
    return result(
        "segmentation.mobile_sam", "segmentation", "learned", ref,
        payload=payload, model_bytes=size, infer_ms=t.ms, web_drivable=False,
    )


def _run_sam(bgr: np.ndarray, weight_path_str: str) -> list[dict[str, Any]]:
    """Run automatic mask generation with the Ultralytics SAM wrapper (supports mobile_sam.pt)."""
    from ultralytics import SAM

    model = SAM(weight_path_str)
    results = model(bgr[:, :, ::-1], verbose=False)
    out: list[dict[str, Any]] = []
    for res in results:
        if res.masks is None:
            continue
        data = res.masks.data.cpu().numpy()  # (n, H, W)
        for m in data:
            ys, xs = np.nonzero(m > 0.5)
            if xs.size == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            out.append(
                {
                    "area": int((m > 0.5).sum()),
                    "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                    "stability_score": float(m.max()),
                }
            )
    return out


def _exists(path: Any) -> bool:
    from pathlib import Path

    return Path(str(path)).is_file()


def _filesize(path: Any) -> int:
    from pathlib import Path

    p = Path(str(path))
    return p.stat().st_size if p.is_file() else 0
