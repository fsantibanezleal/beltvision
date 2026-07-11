"""Stage 1 view-recognition tests.

The classical view classifier must tell a LOADED top-carrying belt from an EMPTY
return strand. The real COLA 34 frame (proprietary, not committed here) is validated as
``end_return`` in the Colia-side smoke; here we use labelled synthetic frames so the test
is self-contained and license-clean.
"""
from __future__ import annotations

from beltvision import recognize_view
from beltvision.cases.synthetic import gt_scene
from beltvision.views import VIEW_ANALYSES, VIEW_TYPES, analyses_for_view


def test_recognize_loaded_top_carrying():
    sc = gt_scene("horizontal_loaded")
    rv = recognize_view(sc.image)
    assert rv["view_type"] == "top_carrying"
    assert 0.0 <= rv["confidence"] <= 1.0
    assert set(rv["scores"]) == set(VIEW_TYPES)


def test_recognize_empty_return_strand_not_loaded():
    sc = gt_scene("vertical_empty")
    rv = recognize_view(sc.image)
    # an empty strand must NOT be called a loaded top-carrying view
    assert rv["view_type"] != "top_carrying"


def test_view_analyses_map_is_consistent():
    for vt in VIEW_TYPES:
        analyses = analyses_for_view(vt)
        assert "semantic" in analyses  # the 4-class backbone is always offered
        assert analyses == VIEW_ANALYSES[vt]
    # end_return inspects the belt (not content); top_carrying inspects content (not cuts)
    assert "content" not in VIEW_ANALYSES["end_return"]
    assert "content" in VIEW_ANALYSES["top_carrying"]
    assert "damage" in VIEW_ANALYSES["end_return"]
    assert "damage" not in VIEW_ANALYSES["top_carrying"]
