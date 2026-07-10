"""beltvision: a reusable computer-vision inspection engine for industrial conveyor belts.

This package owns the science and the contracts, not any single product's serving
layer. It provides:

- Two data contracts: ingestion (Contract 1, the bring-your-own-data gate) and the
  artifact manifest (Contract 2, what a viewer replays).
- A measured live/precompute lane gate that assigns a capability to a lane from
  measured numbers (model bytes, inference milliseconds, trace bytes), never a label.
- A named six-stage pipeline (preprocess, feature_extraction, train, infer, evaluate,
  export) whose stage signatures are frozen and whose bodies are the rework surface.
- The LIVE-tier method ladder (``beltvision.methods``): a registry of CLAHE-first,
  tier-tagged, JSON-safe methods across preprocessing, geometry, granulometry,
  segmentation, anomaly, detection and tracking. Learned methods degrade gracefully to
  ``weights_absent`` when an optional weight is missing. ``beltvision.models`` locates
  and (opt-in) downloads those weights.
- A case registry, deterministic synthetic scenes, and framework-free ONNX artifact
  descriptors.

The engine is deliberately importable with only a slim classical stack (numpy,
opencv, scikit-image, scipy). Heavy deep-learning engines (torch, onnxruntime,
ultralytics, anomalib, transformers) are optional extras imported lazily inside the
precompute lane only, so the import boundary stays clean.
"""
from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
