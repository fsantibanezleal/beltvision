"""Pipeline Studio DAG engine tests (beltvision.pipeline_graph).

Proves: templates run end-to-end and return a per-node step result (overlay + metrics) for
EVERY node; the executor threads images topologically; one failing node is recorded in
``errors`` and never aborts the run; and a cycle is reported, not hung on.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from beltvision import pipeline_graph as pg
from beltvision.cases.synthetic import synth_scene


def _is_png_data_url(s: object) -> bool:
    if not isinstance(s, str) or not s.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(s.split(",", 1)[1])
    return len(raw) > 500 and raw[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def belt_image() -> np.ndarray:
    return synth_scene(orientation_deg=90.0, loaded=True, with_damage=True).image


@pytest.fixture
def rois(belt_image) -> list[dict]:
    h, w = belt_image.shape[:2]
    return [
        {"type": "rect", "points": [[int(0.30 * w), 0], [int(0.70 * w), h]],
         "label": "expected-belt-limits"},
        {"type": "rect", "points": [[int(0.34 * w), int(0.1 * h)],
                                    [int(0.66 * w), int(0.9 * h)]], "label": "content"},
    ]


# --- catalogue / templates --------------------------------------------------------------
def test_op_registry_and_catalog_shape():
    assert len(pg.OP_REGISTRY) >= 25
    for entry in pg.op_catalog():
        assert set(entry) == {"id", "category", "reference", "params_schema"}
    cats = {spec["category"] for spec in pg.OP_REGISTRY.values()}
    assert {"source_roi", "preprocess", "transform", "binarize_morphology",
            "detect", "measure"} <= cats


def test_templates_present_and_get_template_is_a_copy():
    assert set(pg.list_templates()) == {"belt_detection", "belt_condition", "material_on_belt"}
    spec = pg.get_template("belt_detection")
    spec["nodes"].append({"id": "junk", "op": "to_gray", "inputs": []})
    assert len(pg.get_template("belt_detection")["nodes"]) < len(spec["nodes"])  # deep copy
    with pytest.raises(KeyError):
        pg.get_template("nope")


# --- each template runs end-to-end, every node exposes a step result --------------------
@pytest.mark.parametrize("name", ["belt_detection", "belt_condition", "material_on_belt"])
def test_template_runs_end_to_end_with_per_node_results(name, belt_image, rois):
    spec = pg.get_template(name)
    out = pg.run_pipeline(spec, belt_image, rois=rois, priors={"view": "top", "px_per_mm": 5.0})
    assert out["errors"] == [], f"{name} produced errors: {out['errors']}"
    assert len(out["nodes"]) == len(spec["nodes"])
    for node in out["nodes"]:
        assert node["status"] == "ok", f"{name}.{node['id']} status {node['status']}"
        assert _is_png_data_url(node["overlay_b64"]), f"{name}.{node['id']} has no overlay"
        assert isinstance(node["metrics"], dict)


def test_belt_detection_measures_lines(belt_image, rois):
    out = pg.run_pipeline(pg.get_template("belt_detection"), belt_image, rois=rois,
                          priors={"view": "top"})
    measure = next(n for n in out["nodes"] if n["op"] == "measure_lines")
    assert measure["metrics"]["n_lines"] >= 1


def test_material_template_reports_granulometry(belt_image, rois):
    out = pg.run_pipeline(pg.get_template("material_on_belt"), belt_image, rois=rois,
                          priors={"view": "top", "px_per_mm": 5.0})
    gran = next(n for n in out["nodes"] if n["op"] == "granulometry")
    assert "n_particles" in gran["metrics"] and "D50" in gran["metrics"]


# --- robustness: a failing node is recorded, never aborts -------------------------------
def test_one_failing_node_is_recorded_and_downstream_still_runs(belt_image):
    spec = {"nodes": [
        {"id": "a", "op": "clahe", "params": {}, "inputs": []},
        {"id": "bad", "op": "not_a_real_op", "params": {}, "inputs": ["a"]},
        {"id": "c", "op": "to_gray", "params": {}, "inputs": ["a"]},
    ]}
    out = pg.run_pipeline(spec, belt_image)
    ids = {n["id"]: n for n in out["nodes"]}
    assert ids["a"]["status"] == "ok"
    assert ids["bad"]["status"] == "error"
    assert ids["c"]["status"] == "ok"                     # sibling still ran
    assert any(e.get("id") == "bad" for e in out["errors"])


def test_cycle_is_reported_not_hung(belt_image):
    spec = {"nodes": [
        {"id": "x", "op": "to_gray", "params": {}, "inputs": ["y"]},
        {"id": "y", "op": "to_gray", "params": {}, "inputs": ["x"]},
    ]}
    out = pg.run_pipeline(spec, belt_image)
    assert out["errors"], "a cyclic graph must be reported as an error"
    assert any("cycle" in str(e.get("error", "")).lower() for e in out["errors"])


def test_empty_spec_is_a_noop():
    img = synth_scene(orientation_deg=90.0).image
    out = pg.run_pipeline({"nodes": []}, img)
    assert out["nodes"] == [] and out["errors"] == []
