"""Feature-toolbox smoke + straight-line geometry tests.

Two guarantees:

1. The classical feature/edge/keypoint/texture bench (``features.run_all``) runs the whole
   toolbox on a frame, returns AT LEAST 16 methods, and every method yields a NON-EMPTY
   drawn overlay (a real base64 PNG data URL) plus a FINITE scalar metric. Every feature
   method is also individually registered in the ladder REGISTRY, tier-tagged, and runs to
   an ``ok`` envelope.
2. The consolidated geometry analysis keeps the belt centreline STRAIGHT: on a straight
   synthetic belt the least-squares centreline curvature is < 0.05 (no reintroduced
   parabola/curve), and it never withholds - it always emits an estimate + overlay.
"""
from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from beltvision.cases.synthetic import synth_scene
from beltvision.methods import (
    REGISTRY,
    TIERS,
    families,
    features,
    method_index,
    methods_by_tier,
    run,
)
from beltvision.methods.beltline import compute_belt_geometry


@pytest.fixture
def belt_image() -> np.ndarray:
    """A deterministic synthetic belt frame at a diagonal orientation."""
    return synth_scene(orientation_deg=35, loaded=True).image


def _is_png_data_url(s: object) -> bool:
    if not isinstance(s, str) or not s.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(s.split(",", 1)[1])
    return len(raw) > 500 and raw[:8] == b"\x89PNG\r\n\x1a\n"


# --- 1) feature bench: >= 16 methods, each a real overlay + finite metric ---------------
def test_run_all_returns_at_least_16_methods(belt_image):
    out = features.run_all(belt_image)
    assert out["n_methods"] == len(out["methods"])
    assert out["n_methods"] >= 16, f"expected >=16 feature methods, got {out['n_methods']}"


def test_each_feature_method_has_overlay_and_finite_metric(belt_image):
    out = features.run_all(belt_image)
    seen_families: set[str] = set()
    for m in out["methods"]:
        assert set(m) >= {"id", "name", "family", "tier", "reference",
                          "metric_name", "metric_value", "overlay_b64"}
        assert _is_png_data_url(m["overlay_b64"]), f"{m['id']} overlay is not a real PNG"
        assert isinstance(m["metric_value"], (int, float)) and math.isfinite(m["metric_value"]), (
            f"{m['id']} metric {m['metric_name']} not finite: {m['metric_value']}"
        )
        assert m["tier"] == "classical"
        seen_families.add(m["family"])
    # the bench spans every family (edges, lines, superpixels, shape, corners, texture)
    assert {features.FAM_EDGE, features.FAM_LINES, features.FAM_SUPERPIXEL,
            features.FAM_SHAPE, features.FAM_CORNERS, features.FAM_TEXTURE} <= seen_families


def test_distinct_overlays(belt_image):
    # different operators must produce visually different overlays (no copy-paste stub)
    out = features.run_all(belt_image)
    digests = {m["overlay_b64"] for m in out["methods"]}
    assert len(digests) == len(out["methods"]), "some feature overlays are identical"


@pytest.mark.parametrize("mid", features.feature_ids())
def test_every_feature_method_is_registered_and_runs(mid, belt_image):
    assert mid in REGISTRY, f"{mid} missing from the ladder REGISTRY"
    spec = REGISTRY[mid]
    assert spec.capability == "features"
    assert spec.tier in TIERS
    res = run(mid, belt_image)
    assert res["status"] == "ok"
    assert res["method"] == mid
    assert _is_png_data_url(res["overlay_b64"])
    assert math.isfinite(res["metric_value"])


def test_run_all_unknown_method_raises(belt_image):
    with pytest.raises(KeyError):
        features.run_all(belt_image, methods=["features.not_a_thing"])


# --- tier tagging + grouping helpers ----------------------------------------------------
def test_every_registry_method_has_a_valid_tier():
    for mid, spec in REGISTRY.items():
        assert spec.tier in TIERS, f"{mid} has non-tier {spec.tier!r}"


def test_methods_by_tier_partitions_registry():
    by_tier = methods_by_tier()
    assert set(by_tier) <= set(TIERS)
    flat = [m for ids in by_tier.values() for m in ids]
    assert sorted(flat) == sorted(REGISTRY)
    # classical is the big bench; sota holds the learned/deep methods.
    assert len(by_tier["classical"]) >= 16
    assert {"segmentation.semantic_layers", "anomaly.conv_ae",
            "detection.onnx_detector"} <= set(by_tier["sota"])


def test_method_index_and_families_are_json_shaped():
    idx = method_index()
    assert len(idx) == len(REGISTRY)
    for e in idx:
        assert set(e) == {"id", "capability", "tier", "family", "reference", "summary"}
    fams = families()
    assert features.FAM_EDGE in fams and len(fams[features.FAM_EDGE]) >= 6


# --- 2) consolidated geometry stays STRAIGHT --------------------------------------------
def test_centreline_is_straight_on_a_straight_synthetic_belt():
    # The corrected belt geometry fits a STRAIGHT least-squares centreline; on a straight
    # belt the curvature residual must be small (< 0.05) - no reintroduced parabola/curve.
    sc = synth_scene(orientation_deg=35, loaded=True)
    geo = compute_belt_geometry(sc.belt_mask | sc.content_mask, gray=None)
    assert "centreline_xy" in geo
    assert geo["curvature"] < 0.05, f"centreline curvature {geo['curvature']} implies a curve"
    assert geo["curved"] is False
    # a straight centreline is exactly two endpoints (a line), not a wiggly polyline
    assert len(geo["centreline_xy"]) == 2


@pytest.mark.parametrize("deg", [10.0, 35.0, 90.0, 135.0])
def test_geometry_math_recovers_orientation_and_stays_straight(deg):
    sc = synth_scene(orientation_deg=deg, loaded=True)
    geo = compute_belt_geometry(sc.belt_mask | sc.content_mask, gray=None)
    err = abs((geo["axis_angle_deg"] - deg + 90.0) % 180.0 - 90.0)
    assert err <= 6.0, f"axis {geo['axis_angle_deg']} vs gt {deg}"
    assert geo["curvature"] < 0.05


def test_geometry_analysis_emits_estimate_and_overlay_never_withholds():
    sc = synth_scene(orientation_deg=35, loaded=True)
    res = features.geometry_analysis(sc.image, view_type="top_carrying")
    assert res["method"] == "geometry.analysis"
    assert res["status"] == "ok"
    assert _is_png_data_url(res["overlay_b64"])
    # the consolidated read always surfaces the estimate + the three cross-checks
    assert res["orientation_deg"] is not None
    for key in ("hough", "ransac_line", "radon", "obb"):
        assert key in res
    assert res["confidence"] in ("low", "medium", "high")
