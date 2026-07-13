"""ROI rasterisation exactness + belt-limit / orientation-band priors (beltvision.methods.roi)."""
from __future__ import annotations

from beltvision.methods import roi


def test_rasterize_polygon_fills_interior():
    ann = {"type": "polygon", "points": [[2, 2], [2, 7], [7, 7], [7, 2]], "label": "belt"}
    mask = roi.rasterize([ann], (10, 10))
    assert mask.dtype == bool and mask.shape == (10, 10)
    assert mask[4, 4]          # inside
    assert not mask[0, 0]      # outside
    assert not mask[9, 9]      # outside


def test_rasterize_rect_two_corners_is_filled_block():
    ann = {"type": "rect", "points": [[2, 2], [6, 6]], "label": "content"}
    mask = roi.rasterize([ann], (10, 10))
    assert mask[2, 2] and mask[6, 6] and mask[4, 4]
    assert not mask[1, 1] and not mask[7, 7]
    # a filled inclusive 5x5 block
    assert int(mask.sum()) == 25


def test_rasterize_line_has_thickness():
    ann = {"type": "line", "points": [[5, 1], [5, 8]], "label": "belt-limit", "width": 3}
    mask = roi.rasterize([ann], (10, 10))
    assert mask[4, 5] and mask[5, 5] and mask[6, 5]   # centre + both sides (width 3)
    assert not mask[5, 0]


def test_combine_by_label_selects_only_matching_labels():
    a = {"type": "rect", "points": [[0, 0], [3, 3]], "label": "belt"}
    b = {"type": "rect", "points": [[6, 6], [9, 9]], "label": "content"}
    belt = roi.combine_by_label([a, b], (10, 10), "belt")
    assert belt[1, 1] and not belt[7, 7]
    both = roi.combine_by_label([a, b], (10, 10), ["belt", "content"])
    assert both[1, 1] and both[7, 7]


def test_belt_limit_prior_from_region_gives_orientation_and_edges():
    # a tall (vertical) labelled band -> orientation near 90 deg + two long edges.
    ann = {"type": "rect", "points": [[40, 10], [60, 90]], "label": "expected-belt-limits"}
    prior = roi.belt_limit_prior([ann], (100, 100))
    assert prior["found"] is True
    assert prior["mask"].any()
    assert abs((prior["orientation_deg"] - 90.0 + 90.0) % 180.0 - 90.0) <= 6.0
    assert prior["edge_lines"] is not None and len(prior["edge_lines"]) == 2


def test_belt_limit_prior_from_two_guide_lines():
    lines = [
        {"type": "line", "points": [[40, 5], [40, 95]], "label": "belt-limit"},
        {"type": "line", "points": [[60, 5], [60, 95]], "label": "belt-limit"},
    ]
    prior = roi.belt_limit_prior(lines, (100, 100))
    assert prior["found"] is True and prior["source"] == "two-guide-lines"
    assert len(prior["edge_lines"]) == 2
    assert abs((prior["orientation_deg"] - 90.0 + 90.0) % 180.0 - 90.0) <= 6.0


def test_belt_limit_prior_absent_when_no_belt_label():
    prior = roi.belt_limit_prior([{"type": "rect", "points": [[0, 0], [3, 3]], "label": "content"}],
                                 (100, 100))
    assert prior["found"] is False
    assert not prior["mask"].any()


def test_orientation_band_uses_annotation_orientation():
    ann = {"type": "rect", "points": [[40, 10], [60, 90]], "label": "expected-belt-limits"}
    center, band = roi.orientation_band("top", [ann], (100, 100))
    assert abs((center - 90.0 + 90.0) % 180.0 - 90.0) <= 6.0
    assert band == 22.5


def test_orientation_band_view_defaults_when_no_annotation():
    c_top, b_top = roi.orientation_band("top", None)
    assert c_top == 90.0 and b_top >= 35.0        # top/end default near-vertical, wide band
    c_lat, b_lat = roi.orientation_band("lateral", None)
    assert c_lat == 0.0 and b_lat >= 35.0          # lateral default near-horizontal


def test_orientation_band_two_separated_vertical_strips_stay_vertical():
    # THE cola34 fix: one ROI drawn per belt edge. Two tall (vertical) strips placed far
    # apart left/right. Rasterising them jointly gives a wide box -> a WRONG horizontal axis;
    # measuring each strip on its own and averaging must recover the true ~90 deg belt axis.
    left = {"type": "rect", "points": [[100, 40], [130, 360]], "label": "belt"}
    right = {"type": "rect", "points": [[600, 40], [630, 360]], "label": "belt"}
    center, band = roi.orientation_band("end", [left, right], (400, 800))
    assert abs((center - 90.0 + 90.0) % 180.0 - 90.0) <= 8.0, f"got {center}, expected ~90"


def test_orientation_band_two_separated_horizontal_strips_stay_horizontal():
    # dual check: two wide (horizontal) strips stacked top/bottom -> ~0 deg, not 90.
    top = {"type": "rect", "points": [[40, 100], [360, 130]], "label": "belt"}
    bot = {"type": "rect", "points": [[40, 600], [360, 630]], "label": "belt"}
    center, _band = roi.orientation_band("lateral", [top, bot], (800, 400))
    assert min(center, 180.0 - center) <= 8.0, f"got {center}, expected ~0"


def test_axis_circular_mean_wraps_at_180():
    # 1 deg and 179 deg are both ~horizontal; their axis mean is 0/180, never 90.
    m = roi._axis_circular_mean([1.0, 179.0])
    assert min(m, 180.0 - m) <= 2.0
