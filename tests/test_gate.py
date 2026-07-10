"""Measured gate tests: the lane is decided from numbers, never a label."""
from __future__ import annotations

from beltvision.core.gate import (
    LANE_LIVE_SERVER,
    LANE_LIVE_WEB,
    LANE_PRECOMPUTE,
    LIVE_SERVER_MAX_MODEL_BYTES,
    LIVE_WEB_MAX_INFER_MS,
    LIVE_WEB_MAX_MODEL_BYTES,
    classify_lane,
)

_MB = 1024 * 1024


def test_small_fast_web_drivable_is_live_web():
    v = classify_lane(model_bytes=5 * _MB, infer_ms=120, trace_bytes=50_000, web_drivable=True)
    assert v.lane == LANE_LIVE_WEB


def test_medium_model_falls_to_live_server():
    v = classify_lane(model_bytes=120 * _MB, infer_ms=400, trace_bytes=50_000, web_drivable=True)
    assert v.lane == LANE_LIVE_SERVER
    assert any("live-web" in r for r in v.reasons)


def test_not_web_drivable_but_affordable_is_live_server():
    v = classify_lane(model_bytes=5 * _MB, infer_ms=120, trace_bytes=50_000, web_drivable=False)
    assert v.lane == LANE_LIVE_SERVER
    assert "not web-drivable" in v.reasons


def test_huge_model_is_precompute():
    v = classify_lane(
        model_bytes=LIVE_SERVER_MAX_MODEL_BYTES + _MB,
        infer_ms=200,
        trace_bytes=10_000,
        web_drivable=True,
    )
    assert v.lane == LANE_PRECOMPUTE


def test_slow_inference_is_precompute():
    v = classify_lane(model_bytes=1 * _MB, infer_ms=5000, trace_bytes=10_000, web_drivable=True)
    assert v.lane == LANE_PRECOMPUTE


def test_web_boundaries_inclusive():
    # Exactly at the web gate boundary is still live-web (<=, not <).
    v = classify_lane(
        model_bytes=LIVE_WEB_MAX_MODEL_BYTES,
        infer_ms=LIVE_WEB_MAX_INFER_MS,
        trace_bytes=1000,
        web_drivable=True,
    )
    assert v.lane == LANE_LIVE_WEB


def test_verdict_roundtrips_to_dict():
    v = classify_lane(model_bytes=1_000, infer_ms=10.0, trace_bytes=1_000, web_drivable=True)
    d = v.to_dict()
    assert d["lane"] == LANE_LIVE_WEB
    assert set(d) == {"lane", "web_drivable", "model_bytes", "infer_ms", "trace_bytes", "reasons"}
