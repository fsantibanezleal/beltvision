"""The constrained-detector fix, proven on a synthetic cross-hatch frame.

The defect: a raw-image Hough locks onto every direction. The fix: run the detector on a
preprocessed, ROI-masked, orientation-BANDED edge map. These tests prove that on a frame
carrying BOTH vertical and horizontal lines, the constrained Hough returns lines ONLY within
the vertical band, while a raw full-theta Hough returns out-of-band (horizontal) lines too.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from beltvision.methods import constrained


def _is_png_data_url(s: object) -> bool:
    if not isinstance(s, str) or not s.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(s.split(",", 1)[1])
    return len(raw) > 500 and raw[:8] == b"\x89PNG\r\n\x1a\n"


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


@pytest.fixture
def crosshatch() -> np.ndarray:
    """A dark frame with strong VERTICAL and HORIZONTAL white lines (a cross-hatch)."""
    import cv2

    img = np.full((200, 200, 3), 18, dtype=np.uint8)
    for x in (60, 100, 140):
        cv2.line(img, (x, 0), (x, 199), (235, 235, 235), 2, cv2.LINE_8)
    for y in (60, 100, 140):
        cv2.line(img, (0, y), (199, y), (235, 235, 235), 2, cv2.LINE_8)
    rng = np.random.default_rng(34)
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    return img


# --- stage 1: preprocessing produces a binary edge map, ROI-restricted ------------------
def test_preprocess_for_lines_is_binary_and_roi_restricted(crosshatch):
    be = constrained.preprocess_for_lines(crosshatch, edge="canny")
    assert be.dtype == bool and be.shape == (200, 200)
    assert 0 < be.sum() < be.size

    roi = np.zeros((200, 200), dtype=bool)
    roi[:, 50:150] = True
    be_roi = constrained.preprocess_for_lines(crosshatch, roi_mask=roi, edge="canny")
    assert be_roi.sum() <= be.sum()
    assert not be_roi[:, :50].any() and not be_roi[:, 150:].any()  # nothing outside the ROI


# --- stage 2: the gradient-orientation gate keeps vertical, drops horizontal -------------
def test_gradient_orientation_gate_keeps_vertical_drops_horizontal(crosshatch):
    import cv2

    gray = cv2.cvtColor(crosshatch, cv2.COLOR_BGR2GRAY)
    be = constrained.preprocess_for_lines(crosshatch, edge="canny")
    gated = constrained.gradient_orientation_gate(be, gray, theta_center_deg=90.0, theta_band_deg=20.0)
    assert gated.sum() < be.sum()                    # the gate removes edges
    # vertical lines live in columns ~60/100/140; horizontal lines in rows ~60/100/140.
    vertical_kept = gated[:, 58:63].sum() + gated[:, 98:103].sum() + gated[:, 138:143].sum()
    horizontal_kept = gated[58:63, :].sum() + gated[98:103, :].sum() + gated[138:143, :].sum()
    assert vertical_kept > horizontal_kept          # the vertical (in-axis) edges dominate


# --- stage 3: the constrained Hough returns IN-BAND lines only (THE proof) ---------------
def test_constrained_hough_returns_in_band_lines_only(crosshatch):
    be = constrained.preprocess_for_lines(crosshatch, edge="canny")
    center, band = 90.0, 20.0
    rec = constrained.hough_constrained(be, center, band, min_len_px=60.0, bgr=crosshatch)
    assert rec["status"] == "ok"
    assert _is_png_data_url(rec["overlay_b64"])
    assert rec["metric_value"] >= 1                 # found at least one belt line
    angles = rec["angles_deg"]
    assert angles, "constrained Hough found no lines"
    for a in angles:
        assert _ang_diff(a, center) <= band + 1e-6, f"line at {a}deg escaped the band"


def test_constrained_beats_raw_hough_on_out_of_band_lines(crosshatch):
    import cv2

    # constrained: every returned line is within the vertical band.
    be = constrained.preprocess_for_lines(crosshatch, edge="canny")
    rec = constrained.hough_constrained(be, 90.0, 20.0, min_len_px=60.0, bgr=crosshatch)
    assert all(_ang_diff(a, 90.0) <= 20.0 + 1e-6 for a in rec["angles_deg"])

    # raw: a full-theta probabilistic Hough on the same edges DOES return horizontal lines.
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(crosshatch, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                      50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=8)
    assert lines is not None
    raw_angles = [float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
                  for x1, y1, x2, y2 in lines.reshape(-1, 4)]
    out_of_band = [a for a in raw_angles if _ang_diff(a, 90.0) > 25.0]
    assert out_of_band, "raw Hough should have found out-of-band (horizontal) lines"


# --- constrained RANSAC rejects out-of-band models --------------------------------------
def test_constrained_ransac_returns_in_band_lines_only(crosshatch):
    be = constrained.preprocess_for_lines(crosshatch, edge="canny")
    rec = constrained.ransac_line_constrained(be, 90.0, 20.0, bgr=crosshatch, min_inliers=20)
    assert rec["status"] == "ok"
    assert _is_png_data_url(rec["overlay_b64"])
    for a in rec["angles_deg"]:
        assert _ang_diff(a, 90.0) <= 20.0 + 1e-6, f"RANSAC line at {a}deg escaped the band"


# --- the image-first registry wrappers run on a real synthetic belt ----------------------
def test_registry_wrappers_run_on_synthetic_belt():
    from beltvision.cases.synthetic import synth_scene

    img = synth_scene(orientation_deg=90.0, loaded=True).image
    h = constrained.hough_constrained_method(img, view_type="top")
    r = constrained.ransac_line_constrained_method(img, view_type="top")
    assert h["status"] == "ok" and r["status"] == "ok"
    assert _is_png_data_url(h["overlay_b64"]) and _is_png_data_url(r["overlay_b64"])
    # a near-vertical belt: any lines found sit near the 90 deg belt axis.
    for a in h["angles_deg"]:
        assert _ang_diff(a, h["theta_center_deg"]) <= h["theta_band_deg"] + 1e-6


# --- belt-edge pair extraction: pick the TWO edges, report width + centreline -------------
def test_extract_belt_edges_finds_the_pair_and_measures_width():
    # two horizontal (axis 0deg) edges, 300 px apart on the belt normal.
    segs = [
        {"p0": [50, 100], "p1": [950, 100], "angle_deg": 0.0, "support_px": 500},
        {"p0": [50, 400], "p1": [950, 400], "angle_deg": 0.0, "support_px": 500},
    ]
    r = constrained.extract_belt_edges(segs, theta_center_deg=0.0, frame_shape=(500, 1000))
    assert r["found"] is True
    assert abs(r["width_px"] - 300.0) < 1.0
    # edge_a is the smaller-normal-position edge (y~100), edge_b the larger (y~400).
    assert r["edge_a"]["p0"][1] < r["edge_b"]["p0"][1]
    assert r["centreline"] is not None  # a centreline was synthesised at the mid position


def test_extract_belt_edges_is_noise_robust_two_clusters_from_many_lines():
    rng = np.random.default_rng(0)
    segs = []
    for _ in range(6):  # a cluster near y=100
        segs.append({"p0": [10, 100 + int(rng.integers(-3, 4))],
                     "p1": [990, 100 + int(rng.integers(-3, 4))],
                     "angle_deg": 0.0, "support_px": 400})
    for _ in range(6):  # a cluster near y=400
        segs.append({"p0": [10, 400 + int(rng.integers(-3, 4))],
                     "p1": [990, 400 + int(rng.integers(-3, 4))],
                     "angle_deg": 0.0, "support_px": 400})
    r = constrained.extract_belt_edges(segs, theta_center_deg=0.0, frame_shape=(500, 1000))
    assert r["found"] is True
    assert 290.0 < r["width_px"] < 310.0  # the two physical edges, not noise-to-noise spread


def test_extract_belt_edges_needs_two_segments():
    assert constrained.extract_belt_edges([], theta_center_deg=0.0)["found"] is False
    one = [{"p0": [0, 0], "p1": [10, 0], "angle_deg": 0.0, "support_px": 5}]
    assert constrained.extract_belt_edges(one, theta_center_deg=0.0)["found"] is False
