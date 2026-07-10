"""The six frozen pipeline stages.

The stage NAMES and SIGNATURES are frozen (``STAGES`` in ``beltvision.pipeline``); the
stage BODIES are the per-product rework surface. Each stage is
``(StageContext) -> dict`` and records its timing into ``ctx.trace``.

The bodies here implement a real, lightweight, deterministic method ladder using the
slim runtime stack (numpy + opencv). The heavy SOTA engines (anomalib/EfficientAD,
RT-DETR/D-FINE, SegFormer, SAM2, DINOv2, boxmot) plug into these same stages in the
precompute lane; each stage docstring marks where they attach.
"""
from __future__ import annotations

from .evaluate import evaluate
from .export import export
from .feature_extraction import feature_extraction
from .infer import infer
from .preprocess import preprocess
from .train import train

__all__ = [
    "preprocess",
    "feature_extraction",
    "train",
    "infer",
    "evaluate",
    "export",
]
