"""Exact-number tests for the pure measurement primitives (beltvision.methods.measure).

Measurement must be deterministic and correct to the pixel, so every assertion here pins an
EXACT expected value on a known input (no tolerances beyond floating-point rounding).
"""
from __future__ import annotations

import numpy as np
import pytest

from beltvision.methods import measure as m


# --- angles -----------------------------------------------------------------------------
def test_angle_between_perpendicular_is_90():
    assert m.angle_between([0, 0, 1, 0], [0, 0, 0, 1]) == 90.0


def test_angle_between_45_degrees():
    assert m.angle_between([0, 0, 1, 1], [0, 0, 1, 0]) == 45.0


def test_angle_between_collinear_is_zero_and_opposite_is_180():
    assert m.angle_between([0, 0, 2, 0], [0, 0, 5, 0]) == 0.0
    assert m.angle_between([0, 0, 1, 0], [0, 0, -1, 0]) == 180.0


def test_angle_between_accepts_point_pairs():
    assert m.angle_between([[0, 0], [1, 0]], [[0, 0], [0, 1]]) == 90.0


def test_line_angle_wraps_to_0_180():
    assert m.line_angle([0, 0, 1, 0]) == 0.0
    assert m.line_angle([0, 0, 0, 1]) == 90.0
    assert m.line_angle([0, 0, -1, 0]) == 0.0  # undirected axis


def test_degenerate_line_raises():
    with pytest.raises(ValueError):
        m.angle_between([1, 1, 1, 1], [0, 0, 1, 0])


# --- length / area / perimeter ----------------------------------------------------------
def test_segment_length_3_4_5():
    assert m.segment_length([0, 0, 3, 4]) == 5.0


def test_polygon_area_square_and_triangle():
    assert m.polygon_area([[0, 0], [0, 2], [2, 2], [2, 0]]) == 4.0
    assert m.polygon_area([[0, 0], [4, 0], [0, 3]]) == 6.0


def test_polygon_area_is_orientation_independent():
    cw = [[0, 0], [0, 2], [2, 2], [2, 0]]
    ccw = list(reversed(cw))
    assert m.polygon_area(cw) == m.polygon_area(ccw) == 4.0


def test_polygon_perimeter_square():
    assert m.polygon_perimeter([[0, 0], [0, 2], [2, 2], [2, 0]]) == 8.0


def test_polygon_area_needs_three_points():
    with pytest.raises(ValueError):
        m.polygon_area([[0, 0], [1, 1]])


# --- object counting / density ----------------------------------------------------------
def test_count_objects_two_blobs_exact_areas():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1     # 3x3 = 9 px
    mask[10:15, 10:15] = 1  # 5x5 = 25 px
    res = m.count_objects(mask)
    assert res["count"] == 2
    assert sorted(res["areas"]) == [9.0, 25.0]
    assert res["mean_area"] == 17.0
    assert res["total_area"] == 34.0


def test_count_objects_area_range_filters():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    mask[10:15, 10:15] = 1
    res = m.count_objects(mask, area_range=(10.0, 100.0))
    assert res["count"] == 1
    assert res["areas"] == [25.0]
    assert res["mean_area"] == 25.0


def test_density_exact():
    assert m.density(4, 100.0) == 0.04
    assert m.density(3, 0.0) == 0.0


# --- calibration + conversions ----------------------------------------------------------
def test_calibrate_scale_and_conversions():
    ppm = m.calibrate_scale(known_len_px=100.0, known_len_mm=20.0)
    assert ppm == 5.0
    assert m.px_to_mm(10.0, ppm) == 2.0
    assert m.mm_to_px(2.0, ppm) == 10.0
    assert m.px2_to_mm2(100.0, ppm) == 4.0


def test_calibrate_scale_rejects_nonpositive():
    with pytest.raises(ValueError):
        m.calibrate_scale(100.0, 0.0)
    with pytest.raises(ValueError):
        m.px_to_mm(10.0, 0.0)
