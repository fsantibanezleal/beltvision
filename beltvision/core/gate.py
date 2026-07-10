"""The measured live/precompute gate.

A capability is assigned to exactly one lane, and the assignment is decided from
MEASUREMENTS, never from a hand-typed label. ``classify_lane`` is the single
authority; its verdict and the measured numbers are written into the manifest, and
CI fails the build if a manifest claims a lane the gate would not grant.

The thresholds below are sized to the deployment budget: a CPU-only VPS with
roughly 3.7 GiB total RAM (about 2.0-2.5 GiB of headroom for models, systemd caps
MemoryHigh at 2G), and a browser onnxruntime-web (WASM) backend that runs roughly
15-17x slower than native GPU. They are product knobs, but they are explicit and
the gate is the only place they live.
"""
from __future__ import annotations

from dataclasses import dataclass

_MB = 1024 * 1024

# --- Lane names (frozen vocabulary; the web app and the manifest use these) ---
LANE_LIVE_WEB = "live-web"
LANE_LIVE_SERVER = "live-server"
LANE_PRECOMPUTE = "precompute"

LANES = (LANE_LIVE_WEB, LANE_LIVE_SERVER, LANE_PRECOMPUTE)

# --- LIVE-WEB gate (onnxruntime-web / WASM in the visitor's browser) ---
# Small models only: WASM is ~15-17x slower than native GPU and the artifact ships
# to every visitor, so both the model and the replay trace must be tiny.
LIVE_WEB_MAX_MODEL_BYTES = 40 * _MB
LIVE_WEB_MAX_INFER_MS = 800
LIVE_WEB_MAX_TRACE_BYTES = 2 * _MB

# --- LIVE-SERVER gate (FastAPI CPU on the VPS, within the ~2 GiB budget) ---
# Bigger than the browser can hold, still CPU-affordable and lazy-loaded/evicted.
LIVE_SERVER_MAX_MODEL_BYTES = 500 * _MB
LIVE_SERVER_MAX_INFER_MS = 1500
LIVE_SERVER_MAX_TRACE_BYTES = 8 * _MB


@dataclass(frozen=True)
class LaneVerdict:
    """The gate's decision plus the measurements it was decided from."""

    lane: str
    web_drivable: bool
    model_bytes: int
    infer_ms: float
    trace_bytes: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "web_drivable": self.web_drivable,
            "model_bytes": int(self.model_bytes),
            "infer_ms": round(float(self.infer_ms), 3),
            "trace_bytes": int(self.trace_bytes),
            "reasons": list(self.reasons),
        }


def classify_lane(
    model_bytes: int,
    infer_ms: float,
    trace_bytes: int,
    web_drivable: bool,
) -> LaneVerdict:
    """Return the lane a capability may run in, decided from measurements.

    LIVE-WEB requires ALL of: a browser-drivable path (ONNX/algorithm), model bytes
    within the web gate, inference time within the web gate, and a compact trace.
    If any fails, LIVE-SERVER is tried against its (larger) gate. If that also
    fails, the capability is PRECOMPUTE by definition and its results must be
    committed as an artifact rather than served live.
    """
    model_bytes = int(model_bytes)
    trace_bytes = int(trace_bytes)
    infer_ms = float(infer_ms)
    reasons: list[str] = []

    web_ok = (
        web_drivable
        and model_bytes <= LIVE_WEB_MAX_MODEL_BYTES
        and infer_ms <= LIVE_WEB_MAX_INFER_MS
        and trace_bytes <= LIVE_WEB_MAX_TRACE_BYTES
    )
    if web_ok:
        reasons.append("within live-web gate")
        return LaneVerdict(
            LANE_LIVE_WEB, web_drivable, model_bytes, infer_ms, trace_bytes, tuple(reasons)
        )

    if not web_drivable:
        reasons.append("not web-drivable")
    if model_bytes > LIVE_WEB_MAX_MODEL_BYTES:
        reasons.append(f"model_bytes {model_bytes} > live-web {LIVE_WEB_MAX_MODEL_BYTES}")
    if infer_ms > LIVE_WEB_MAX_INFER_MS:
        reasons.append(f"infer_ms {infer_ms:.1f} > live-web {LIVE_WEB_MAX_INFER_MS}")
    if trace_bytes > LIVE_WEB_MAX_TRACE_BYTES:
        reasons.append(f"trace_bytes {trace_bytes} > live-web {LIVE_WEB_MAX_TRACE_BYTES}")

    server_ok = (
        model_bytes <= LIVE_SERVER_MAX_MODEL_BYTES
        and infer_ms <= LIVE_SERVER_MAX_INFER_MS
        and trace_bytes <= LIVE_SERVER_MAX_TRACE_BYTES
    )
    if server_ok:
        reasons.append("within live-server gate")
        return LaneVerdict(
            LANE_LIVE_SERVER, web_drivable, model_bytes, infer_ms, trace_bytes, tuple(reasons)
        )

    if model_bytes > LIVE_SERVER_MAX_MODEL_BYTES:
        reasons.append(f"model_bytes {model_bytes} > live-server {LIVE_SERVER_MAX_MODEL_BYTES}")
    if infer_ms > LIVE_SERVER_MAX_INFER_MS:
        reasons.append(f"infer_ms {infer_ms:.1f} > live-server {LIVE_SERVER_MAX_INFER_MS}")
    if trace_bytes > LIVE_SERVER_MAX_TRACE_BYTES:
        reasons.append(f"trace_bytes {trace_bytes} > live-server {LIVE_SERVER_MAX_TRACE_BYTES}")
    reasons.append("falls back to precompute")
    return LaneVerdict(
        LANE_PRECOMPUTE, web_drivable, model_bytes, infer_ms, trace_bytes, tuple(reasons)
    )
