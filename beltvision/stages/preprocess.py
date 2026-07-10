"""Stage 1: preprocess.

CLAHE-first for the dusty scene (mandatory for the COLA 34 regime), then a
dust/haze severity score so downstream confidences can be honestly down-weighted.
Contract 1 validation runs here before any pixels are transformed: bad input is
rejected, not coerced.

Rework surface: the exact denoise chain (bilateral, FFT mesh-notch) and the haze
model. The frozen part is that CLAHE is the first operation and the same op runs in
training/precompute so train and serve match.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..context import StageContext
from ..core.trace import stage_timer
from ..io.contract import validate_image


def _apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _haze_severity(img_bgr: np.ndarray) -> float:
    """Dark-channel-style haze severity in [0, 1]; higher means hazier/dustier."""
    dark = img_bgr.min(axis=2).astype(np.float32)
    return float(np.clip(dark.mean() / 255.0, 0.0, 1.0))


def preprocess(ctx: StageContext) -> dict[str, Any]:
    with stage_timer(ctx.trace, "preprocess") as t:
        ingestion = validate_image(ctx.image_bgr, ctx.params)
        if not ingestion.accepted:
            raise ValueError(f"ingestion rejected for {ctx.case_id}: {ingestion.reasons}")

        clahe = _apply_clahe(ctx.image_bgr)
        haze = _haze_severity(ctx.image_bgr)
        ctx.state["clahe_bgr"] = clahe
        ctx.state["ingestion"] = ingestion.to_dict()
        ctx.state["haze_severity"] = round(haze, 4)

        summary = {
            "ingestion": ingestion.to_dict(),
            "clahe": {"clip_limit": 2.0, "tile": 8, "color_space": "LAB-L"},
            "haze_severity": round(haze, 4),
            "flagged": ingestion.flagged,
        }
        ctx.state["preprocess_summary"] = summary
        t.note(flagged=ingestion.flagged, haze=round(haze, 3))
    return summary
