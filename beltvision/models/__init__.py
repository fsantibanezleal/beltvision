"""beltvision.models: optional learned-weight provisioning (locate + download + cache).

This package is distinct from ``beltvision.model`` (singular), which only *describes*
an exported ONNX artifact for the gate/manifest. ``beltvision.models`` is where the
optional pretrained weights that the LIVE learned methods consume are located on disk
and, opt-in, downloaded.

It is the single root of the ``weights_absent`` contract: a learned method asks
:func:`ensure_weight`; if the file is not locally present (and no opt-in download was
requested, or a download failed) it gets ``None`` back and returns a graceful
``{"status": "weights_absent", ...}`` result rather than raising. Downloads use httpx
(never curl) and are strictly OPT-IN, so tests and the default runtime stay fully
offline and deterministic.
"""
from __future__ import annotations

from .download import (
    WEIGHTS,
    WeightSpec,
    download_weight,
    ensure_weight,
    is_present,
    search_dirs,
    weight_path,
    weights_dir,
)

__all__ = [
    "WEIGHTS",
    "WeightSpec",
    "download_weight",
    "ensure_weight",
    "is_present",
    "search_dirs",
    "weight_path",
    "weights_dir",
]
