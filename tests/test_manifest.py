"""Contract 2 (artifact manifest) tests: build, validate, drift, index, round-trip."""
from __future__ import annotations

import pytest

from beltvision.core.gate import classify_lane
from beltvision.core.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_index,
    build_manifest,
    build_method_result,
    read_manifest,
    validate_manifest,
    write_manifest,
)

_MB = 1024 * 1024


def _method(method: str = "patch_anomaly", capability: str = "anomaly") -> dict:
    verdict = classify_lane(model_bytes=2 * _MB, infer_ms=50, trace_bytes=1000, web_drivable=True)
    return build_method_result(
        method=method,
        capability=capability,
        tier="learned",
        verdict=verdict,
        metrics={"max_score": 0.9},
    )


def _manifest(case_id: str = "synth_tear_gt", category: str = "synthetic-control") -> dict:
    return build_manifest(
        case_id=case_id,
        category=category,
        source="synthetic (labeled)",
        license="synthetic",
        seed=34,
        created_utc="2026-07-10T00:00:00+00:00",
        preprocess={"clahe": {"clip_limit": 2.0}},
        methods=[_method()],
        models=[{"name": "m", "bytes": 1024}],
        artifact={"path": "data/derived/artifacts/x.json", "bytes": 200, "format": "json"},
    )


def test_build_manifest_valid():
    m = _manifest()
    validate_manifest(m)  # must not raise
    assert m["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert m["methods"][0]["lane"] == "live-web"


def test_missing_key_fails():
    m = _manifest()
    del m["artifact"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_manifest(m)


def test_bad_lane_fails():
    m = _manifest()
    m["methods"][0]["lane"] = "gpu-cluster"
    with pytest.raises(ValueError, match="lane"):
        validate_manifest(m)


def test_empty_methods_fails():
    m = _manifest()
    m["methods"] = []
    with pytest.raises(ValueError, match="non-empty"):
        validate_manifest(m)


def test_unknown_tier_rejected():
    verdict = classify_lane(model_bytes=1, infer_ms=1, trace_bytes=1, web_drivable=True)
    with pytest.raises(ValueError, match="tier"):
        build_method_result(
            method="x", capability="anomaly", tier="magic", verdict=verdict, metrics={}
        )


def test_write_read_roundtrip(tmp_path):
    m = _manifest()
    path = tmp_path / "manifests" / "synth_tear_gt.json"
    write_manifest(m, path)
    assert path.exists()
    again = read_manifest(path)
    assert again["case_id"] == m["case_id"]


def test_build_index_groups_and_counts():
    a = _manifest("synth_tear_gt", "synthetic-control")
    b = _manifest("neu_detect", "surface-detect")
    index = build_index([a, b])
    assert index["n_cases"] == 2
    assert "synthetic-control" in index["by_category"]
    assert index["lane_counts"]["live-web"] >= 2
