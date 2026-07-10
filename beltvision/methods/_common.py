"""Shared plumbing for the LIVE method ladder.

Every method in :mod:`beltvision.methods` returns a JSON-safe ``dict`` with the same
envelope: a ``status`` (``"ok"`` or ``"weights_absent"``), the measured gate inputs
(``model_bytes``, ``infer_ms``, ``trace_bytes``, ``web_drivable``), and the lane the
gate assigns from those numbers (never a hand-typed label). The capability-specific
payload is merged into the same dict. This module builds that envelope so each method
body only computes its payload and declares its cost.

Two hard contracts live here:
- ``result(...)`` runs the payload through :func:`core.manifest.jsonable` and classifies
  the lane with :func:`core.gate.classify_lane`, so no numpy leaks and no unmeasured lane.
- ``weights_absent(...)`` is the graceful fallback: a learned method whose optional weight
  is missing returns this instead of raising, and it still reports the *expected* model
  bytes so the gate classifies the lane honestly while the weight is absent.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..core.gate import classify_lane
from ..core.manifest import jsonable

# Keys the envelope owns; everything else in a result dict is capability payload. Used to
# split a ladder result back into (envelope, metrics) when folding it into a manifest.
ENVELOPE_KEYS = frozenset(
    {
        "method",
        "capability",
        "tier",
        "status",
        "lane",
        "web_drivable",
        "model_bytes",
        "infer_ms",
        "trace_bytes",
        "gate",
        "reference",
        "notes",
    }
)


def payload_bytes(payload: Any) -> int:
    """Serialized (compact UTF-8 JSON) size of a payload: the gate's ``trace_bytes``."""
    return len(json.dumps(jsonable(payload), separators=(",", ":")).encode("utf-8"))


@contextmanager
def timed():
    """Time a block in milliseconds: ``with timed() as t: ...; t.ms``."""

    class _T:
        ms = 0.0

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.ms = (time.perf_counter() - start) * 1000.0


def as_bgr(image: Any) -> np.ndarray:
    """Coerce an input into an ``(H, W, 3)`` uint8 BGR array.

    Accepts a 3-channel BGR frame (the common case), a single-channel grayscale image
    (broadcast to 3 channels), or a float image (clipped to ``[0, 255]``). Genuinely
    malformed input raises ``ValueError`` (the app layer maps that to a 4xx); this is
    distinct from the weights-absent path, which never raises.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got shape {tuple(arr.shape)}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"image too small: {tuple(arr.shape)}")
    return np.ascontiguousarray(arr)


def result(
    method_id: str,
    capability: str,
    tier: str,
    reference: str,
    *,
    payload: dict[str, Any],
    model_bytes: int,
    infer_ms: float,
    web_drivable: bool,
    status: str = "ok",
    notes: str | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-safe method result with a measured lane verdict."""
    trace_bytes = payload_bytes({**payload, "method": method_id})
    verdict = classify_lane(
        model_bytes=int(model_bytes),
        infer_ms=float(infer_ms),
        trace_bytes=int(trace_bytes),
        web_drivable=bool(web_drivable),
    )
    envelope = {
        "method": method_id,
        "capability": capability,
        "tier": tier,
        "status": status,
        "lane": verdict.lane,
        "web_drivable": bool(web_drivable),
        "model_bytes": int(model_bytes),
        "infer_ms": round(float(infer_ms), 3),
        "trace_bytes": int(trace_bytes),
        "gate": verdict.to_dict(),
        "reference": reference,
        "notes": notes,
    }
    return jsonable({**envelope, **payload})


def weights_absent(
    method_id: str,
    capability: str,
    tier: str,
    reference: str,
    *,
    weight: str,
    searched: Iterable[Any],
    approx_bytes: int = 0,
    web_drivable: bool = True,
    hint: str = "",
    infer_ms: float = 0.0,
) -> dict[str, Any]:
    """The graceful fallback when an optional weight is missing (never raises)."""
    payload = {
        "weight": weight,
        "searched": [str(p) for p in searched],
        "hint": hint,
    }
    return result(
        method_id,
        capability,
        tier,
        reference,
        payload=payload,
        model_bytes=int(approx_bytes),
        infer_ms=infer_ms,
        web_drivable=web_drivable,
        status="weights_absent",
    )


def cap(values: Any, n: int) -> list[Any]:
    """Cap a sequence to at most ``n`` items so the JSON trace stays compact."""
    items = list(values)
    return items[:n]
