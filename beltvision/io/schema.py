"""Ingestion schema types (Contract 1).

These declare, in code, exactly what the pipeline accepts and what a validation
verdict looks like. ``data/README.md`` documents the same contract in prose; the two
must agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Verdicts (frozen vocabulary) -------------------------------------------------
ACCEPT = "accept"
REJECT = "reject"
FLAG = "flag"
VERDICTS = (ACCEPT, REJECT, FLAG)

# --- Accepted media -------------------------------------------------------------
IMAGE_FORMATS = ("jpg", "jpeg", "png", "bmp", "webp")
VIDEO_FORMATS = ("mp4", "avi", "mov")

# --- Ranges and limits (units are explicit) -------------------------------------
MIN_SIDE_PX = 64          # reject anything smaller on the short side
MAX_SIDE_PX = 8192        # reject anything larger on the long side
MAX_IMAGE_BYTES = 20 * 1024 * 1024      # 20 MB per image
MAX_VIDEO_BYTES = 500 * 1024 * 1024     # 500 MB per clip
CHANNELS = 3              # BGR / RGB three-channel only

# Outlier policy threshold: L-channel standard deviation below this is flagged (not
# rejected) as low-contrast haze, so downstream confidences can be down-weighted.
MIN_CONTRAST_STD = 12.0


@dataclass
class IngestionParams:
    """The declared parameters that accompany an uploaded frame or clip."""

    media_type: str = "image"          # "image" | "video"
    declared_format: str | None = None  # file extension without the dot, lower-case
    px_per_mm: float | None = None      # optional calibration; None => relative units
    max_side: int = 1024                # working resolution cap applied after accept
    source: str = "user-upload"


@dataclass
class IngestionResult:
    """The outcome of validating one frame against Contract 1."""

    verdict: str
    reasons: list[str] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verdict in (ACCEPT, FLAG)

    @property
    def flagged(self) -> bool:
        return self.verdict == FLAG

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reasons": list(self.reasons), "measured": self.measured}
