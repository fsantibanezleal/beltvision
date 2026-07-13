"""Robust staged (cascade) belt analysis (beltvision.methods.robust).

Proves the cascade contract: orientation consensus is sane; belt_band returns two limits with
the centreline as their MIDLINE (never a medial-axis diagonal); the damage ensemble runs inside
the band and reports per-pipeline + an honest RGB note; edge_condition consumes the validated
limits and gates honestly when the band is low-confidence.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from beltvision.cases.synthetic import synth_scene
from beltvision.methods import robust


def _is_png_data_url(s: object) -> bool:
    if not isinstance(s, str) or not s.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(s.split(",", 1)[1])
    return len(raw) > 500 and raw[:8] == b"\x89PNG\r\n\x1a\n"


def _line_offset(line, normal):
    p0, p1 = robust._line_pts(line)
    return float(((p0 + p1) / 2.0) @ normal)


@pytest.fixture
def vertical_belt() -> np.ndarray:
    return synth_scene(orientation_deg=90.0, loaded=False).image


# --- orientation consensus --------------------------------------------------------------
def test_axis_circular_mean_wraps_at_180():
    assert min(robust._axis_circular_mean([1.0, 179.0]), 180 - robust._axis_circular_mean([1.0, 179.0])) <= 2.0


def test_orientation_consensus_on_vertical_belt(vertical_belt):
    import cv2

    gray = cv2.cvtColor(vertical_belt, cv2.COLOR_BGR2GRAY)
    ori = robust.orientation_consensus(gray)
    assert set(ori) == {"angle_deg", "agreement", "per_method"}
    assert robust._ang_diff(ori["angle_deg"], 90.0) <= 12.0
    assert 0.0 <= ori["agreement"] <= 1.0


# --- belt_band: the centreline is the MIDLINE of the two limits (the core contract) ------
def test_belt_band_found_on_synthetic_and_centreline_is_midline(vertical_belt):
    rec = robust.belt_band(vertical_belt)
    assert rec["status"] == "ok" and _is_png_data_url(rec["overlay_b64"])
    assert rec["found"] is True
    assert rec["width_px"] > 0
    # centreline offset along the normal == mean of the two limit offsets (midline property)
    _axis, normal = robust._normal_unit(rec["orientation_deg"])
    sa = _line_offset(rec["edge_a"], normal)
    sb = _line_offset(rec["edge_b"], normal)
    sc = _line_offset(rec["centreline"], normal)
    assert abs(sc - 0.5 * (sa + sb)) <= 2.0, "centreline must be the midline of the two limits"
    # width == |offset_a - offset_b|
    assert abs(rec["width_px"] - abs(sa - sb)) <= 3.0


def test_belt_band_reports_per_pipeline_and_confidence():
    rec = robust.belt_band(synth_scene(orientation_deg=90.0, loaded=True).image)
    pp = rec["per_pipeline"]
    assert {"orientation_consensus", "normal_projection", "constrained_hough"} <= set(pp)
    assert rec["confidence_label"] in {"low", "medium", "high"}
    assert 0.0 <= rec["confidence"] <= 1.0


def test_band_mask_from_edges_is_between_the_limits(vertical_belt):
    rec = robust.belt_band(vertical_belt)
    mask = robust.band_mask_from_edges(vertical_belt.shape[:2], rec["edge_a"], rec["edge_b"])
    assert mask.dtype == bool and mask.shape == vertical_belt.shape[:2]
    assert 0 < int(mask.sum()) < mask.size  # a proper sub-band, not empty and not the whole frame


# --- damage: RGB anomaly ensemble inside the band, honest note --------------------------
def test_damage_runs_inside_band_with_per_pipeline_and_honest_note(vertical_belt):
    band = robust.belt_band(vertical_belt)
    rec = robust.damage(vertical_belt, band=band)
    assert rec["status"] == "ok" and _is_png_data_url(rec["overlay_b64"])
    assert {"illumination_residual", "wavelet_residual", "fft_bandstop_residual",
            "morphological"} <= set(rec["per_pipeline"])
    assert 0.0 <= rec["severity"] <= 1.0
    assert "RGB-only" in rec["note"]  # states the anomaly limitation honestly


# --- edge_condition: consumes validated limits; gates honestly --------------------------
def test_edge_condition_gates_when_band_low_confidence():
    # a flat/near-empty frame -> no confident band -> edge_condition must say n/a, not invent.
    flat = np.full((240, 320, 3), 120, dtype=np.uint8)
    rec = robust.edge_condition(flat)
    assert rec["applicable"] is False and rec["status"] == "na"
    assert _is_png_data_url(rec["overlay_b64"])


def test_edge_condition_ok_on_confident_band(vertical_belt):
    band = robust.belt_band(vertical_belt)
    if not band["found"]:
        pytest.skip("synthetic band not confident in this environment")
    rec = robust.edge_condition(vertical_belt, band=band)
    assert rec["applicable"] is True
    assert "edge_a" in rec and "edge_b" in rec and "verdict" in rec
