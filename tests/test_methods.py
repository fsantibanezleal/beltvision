"""LIVE method-ladder tests.

For EACH registered method: run it on a synthetic belt frame and assert a real, non-empty,
JSON-serializable result carrying a measured lane verdict. For each learned method with an
optional weight: assert a missing weight degrades to ``status == "weights_absent"`` (never
an exception). Plus per-capability content checks and the download/graceful-fallback contract.
"""
from __future__ import annotations

import json

import pytest

from beltvision.core.manifest import validate_manifest
from beltvision.methods import (
    REGISTRY,
    learned_methods,
    list_methods,
    methods_by_capability,
    run,
    run_ladder,
    to_manifest_method,
)
from beltvision.methods.geometry import reset_kalman
from beltvision.methods.tracking import reset_tracker

# Learned methods that require an optional downloaded/exported weight (weights_absent-capable).
WEIGHT_GATED = {"segmentation.mobile_sam", "anomaly.conv_ae", "detection.onnx_detector"}
# Everything else must produce a real "ok" result with no weight at all.
ALWAYS_OK = set(list_methods()) - WEIGHT_GATED

_ENVELOPE = {
    "method", "capability", "tier", "status", "lane", "web_drivable",
    "model_bytes", "infer_ms", "trace_bytes", "gate", "reference",
}


@pytest.fixture(autouse=True)
def _reset_persistent_state():
    reset_kalman()
    reset_tracker()
    yield
    reset_kalman()
    reset_tracker()


def _assert_envelope(res: dict) -> None:
    assert isinstance(res, dict) and res, "result must be a non-empty dict"
    json.dumps(res)  # must be JSON-serializable (raises TypeError otherwise)
    assert _ENVELOPE <= set(res), f"missing envelope keys: {_ENVELOPE - set(res)}"
    assert res["status"] in ("ok", "weights_absent")
    assert res["lane"] in ("live-web", "live-server", "precompute")
    assert isinstance(res["infer_ms"], (int, float))
    assert isinstance(res["model_bytes"], int)


# --- every method: real, non-empty, JSON-serializable, gate-tagged --------------------
@pytest.mark.parametrize("method_id", list_methods())
def test_method_runs_and_is_json_serializable(method_id, synth_image):
    res = run(method_id, synth_image)
    _assert_envelope(res)
    assert res["method"] == method_id
    assert res["capability"] == REGISTRY[method_id].capability
    if method_id in ALWAYS_OK:
        assert res["status"] == "ok", f"{method_id} should run without any weight"


# --- learned methods: a missing optional weight is graceful, never an exception -------
@pytest.mark.parametrize("method_id", sorted(WEIGHT_GATED))
def test_missing_weight_yields_weights_absent(method_id, synth_image):
    res = run(method_id, synth_image, weights="C:/definitely/not/here.bin")
    _assert_envelope(res)
    assert res["status"] == "weights_absent"
    assert "hint" in res and "searched" in res
    # An absent weight still reports the expected model size so the gate stays honest.
    assert res["model_bytes"] > 0


def test_registry_has_at_least_twelve_live_methods_with_two_learned():
    from beltvision.methods import TIERS

    # Every method carries a maturity tier in {classical, sota, beyond_sota}.
    live = [m for m in list_methods() if REGISTRY[m].tier in TIERS]
    assert len(live) == len(list_methods()), "every method must carry a valid maturity tier"
    assert len(live) >= 12, f"expected >=12 live-tier methods, got {len(live)}"
    # learned/deep methods are the sota (+ beyond_sota) tier.
    assert len(learned_methods()) >= 2, "expected >=2 learned (sota) methods"


def test_all_six_capabilities_plus_preprocess_present():
    caps = set(methods_by_capability())
    assert {"preprocess", "geometry", "granulometry", "segmentation", "anomaly",
            "detection", "tracking"} <= caps


def test_run_unknown_method_raises_keyerror(synth_image):
    with pytest.raises(KeyError):
        run("nope.not_a_method", synth_image)


# --- per-capability content -----------------------------------------------------------
def test_preprocess_reports_haze(synth_image):
    r = run("preprocess.clahe_lab", synth_image)
    assert "haze_before" in r and "haze_after" in r
    assert 0.0 <= r["haze_after"]["severity"] <= 1.0


def test_hough_edges_split_left_right(synth_image):
    r = run("geometry.hough_edges", synth_image)
    assert "left_segments" in r and "right_segments" in r
    assert isinstance(r["n_candidates"], int)


def test_belt_geometry_is_mask_derived_no_polynomial(synth_image):
    # The only edge model is the belt-mask shape (medial axis); there is NO polynomial fit.
    r = run("geometry.belt_geometry", synth_image)
    assert "axis_angle_deg" in r or r["confidence"] == "low"
    assert "coeffs" not in r  # no parametric curve coefficients anywhere
    assert r["confidence"] in ("low", "medium", "high")


def test_ransac_and_misalignment_methods_removed():
    # The degree-2 polynomial edge fit and the axis-assuming misalignment are gone.
    assert "geometry.ransac_edges" not in list_methods()
    assert "geometry.misalignment" not in list_methods()


def test_radon_orientation_in_range(synth_image):
    r = run("geometry.radon_orientation", synth_image)
    assert 0.0 <= r["structure_orientation_deg"] <= 180.0
    assert r["orientation_strength"] > 0


def test_semantic_layers_four_classes_never_weights_absent(synth_image):
    r = run("segmentation.semantic_layers", synth_image)
    assert r["status"] == "ok"  # backbone degrades to the classical prior, never weights_absent
    assert set(r["coverage"]) == {"external", "belt", "content", "foreign"}


def test_kalman_persists_across_calls(synth_image):
    r1 = run("geometry.kalman_edge", synth_image, camera_id="camA")
    r2 = run("geometry.kalman_edge", synth_image, camera_id="camA")
    assert r2["n_updates"] >= r1["n_updates"]
    assert r2["n_updates"] == 2  # two updates accumulated on the same camera


def test_obb_returns_oriented_boxes(synth_image):
    r = run("geometry.obb", synth_image)
    if r["belt_obb"]:
        assert len(r["belt_obb"]["box_points"]) == 4
    assert isinstance(r["region_obbs"], list)


def test_watershed_psd_percentiles_and_curve(synth_image):
    r = run("granulometry.watershed_psd", synth_image)
    assert r["n_particles"] >= 0
    assert {"D10", "D50", "D80"} <= set(r)
    assert r["unit"] == "px" and "relative" in r["calibration"]
    assert isinstance(r["psd_curve"], list)


def test_watershed_psd_mm_when_calibrated(synth_image):
    r = run("granulometry.watershed_psd", synth_image, px_per_mm=5.0)
    assert r["unit"] == "mm" and r["calibration"] == "absolute-mm"


def test_slic_superpixels(synth_image):
    r = run("segmentation.slic", synth_image, n_segments=120)
    assert r["n_segments_actual"] > 1
    assert r["segment_size_px"]["mean"] > 0


def test_padim_lite_returns_heatmap_and_score(synth_image):
    r = run("anomaly.padim_lite", synth_image)
    assert r["status"] == "ok"
    assert r["feature_dim"] >= 1
    assert 0 <= r["peak_row"] < r["grid_rows"]
    assert r["residual_heatmap"] is None or isinstance(r["residual_heatmap"], list)


def test_padim_lite_with_normal_set(synth_image):
    r = run("anomaly.padim_lite", synth_image, normal_images=[synth_image, synth_image])
    assert r["status"] == "ok"
    assert r["fit_source"].startswith("normal-set")


def test_conv_ae_exposes_architecture_even_when_absent(synth_image):
    r = run("anomaly.conv_ae", synth_image)
    assert r["status"] == "weights_absent"
    assert r["architecture"]["family"] == "convolutional-autoencoder"


def test_detector_weights_absent_is_live_server(synth_image):
    r = run("detection.onnx_detector", synth_image)
    assert r["status"] == "weights_absent"
    # ~65 MB expected model pushes it past the web gate: honest live-server verdict.
    assert r["lane"] == "live-server"


def test_optical_flow_belt_speed(synth_image):
    r = run("tracking.optical_flow", synth_image)
    assert "belt_speed_px_per_frame" in r
    assert "demo" in r["source"]
    r2 = run("tracking.optical_flow", synth_image, prev_image=synth_image)
    assert r2["source"] == "consecutive-frames"


def test_bytetrack_demo_and_real_boxes(synth_image):
    demo = run("tracking.bytetrack_associate", synth_image)
    assert demo["n_tracks"] >= 1 and demo["n_frames_processed"] == 2
    boxes = [[10, 10, 40, 40, 0.9], [80, 20, 110, 50, 0.8]]
    real = run("tracking.bytetrack_associate", synth_image, detections=boxes)
    assert real["source"] == "detector-boxes"
    assert real["n_tracks"] == 2


# --- ladder + manifest folding --------------------------------------------------------
def test_run_ladder_covers_registry(synth_image):
    results = run_ladder(synth_image)
    assert set(results) == set(list_methods())
    for res in results.values():
        _assert_envelope(res)


def test_ladder_results_fold_into_valid_manifest(synth_image):
    from beltvision.core.manifest import build_manifest

    methods = [to_manifest_method(run(m, synth_image)) for m in list_methods()]
    manifest = build_manifest(
        case_id="synth_tear_gt", category="synthetic-control",
        source="synthetic (labeled)", license="synthetic", seed=34,
        created_utc="2026-07-10T00:00:00+00:00",
        preprocess={"clahe": {"clip_limit": 2.0}}, methods=methods,
        models=[{"name": "none", "bytes": 0}],
        artifact={"path": "x.json", "bytes": 1, "format": "json"},
    )
    validate_manifest(manifest)
    assert len(manifest["methods"]) == len(list_methods())
