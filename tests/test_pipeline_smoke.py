"""Pipeline smoke test: import beltvision and run one case through all six stages."""
from __future__ import annotations

import beltvision
from beltvision.core.manifest import validate_manifest
from beltvision.pipeline import STAGE_FUNCS, STAGES, run_case


def test_package_metadata():
    assert beltvision.__version__
    assert STAGES == ("preprocess", "feature_extraction", "train", "infer", "evaluate", "export")


def test_stage_funcs_match_stage_names():
    assert tuple(STAGE_FUNCS) == STAGES
    for name in STAGES:
        assert callable(STAGE_FUNCS[name])


def test_run_synthetic_case_end_to_end(out_dir):
    manifest = run_case("synth_tear_gt", quick=True, out_root=out_dir)
    validate_manifest(manifest)

    assert manifest["case_id"] == "synth_tear_gt"
    assert manifest["category"] == "synthetic-control"
    assert manifest["seed"] == 34

    methods = {m["method"] for m in manifest["methods"]}
    assert {"edge_geometry", "patch_anomaly"} <= methods

    # Every method carries a measured lane verdict and JSON-native metrics.
    for m in manifest["methods"]:
        assert m["lane"] in ("live-web", "live-server", "precompute")
        assert isinstance(m["infer_ms"], (int, float))
        assert isinstance(m["metrics"], dict)

    # The manifest and the compact artifact were written under the isolated out dir.
    assert (out_dir / "manifests" / "synth_tear_gt.json").exists()
    assert (out_dir / "artifacts" / "synth_tear_gt.json").exists()


def test_run_is_deterministic(out_dir):
    m1 = run_case("synth_psd_gt", quick=True, out_root=out_dir / "a")
    m2 = run_case("synth_psd_gt", quick=True, out_root=out_dir / "b")
    # Determinism check on a stable metric (timings are excluded, they vary).
    s1 = m1["methods"][1]["metrics"]["max_score"]
    s2 = m2["methods"][1]["metrics"]["max_score"]
    assert s1 == s2


def test_all_six_stages_traced(out_dir):
    from beltvision.context import StageContext
    from beltvision.registry import get_case, load_image
    from beltvision.stages import (
        evaluate,
        export,
        feature_extraction,
        infer,
        preprocess,
        train,
    )

    ctx = StageContext(
        spec=get_case("synth_tear_gt"),
        image_bgr=load_image("synth_tear_gt"),
        out_root=out_dir,
    )
    for fn in (preprocess, feature_extraction, train, infer, evaluate, export):
        fn(ctx)
    assert ctx.trace.stage_names() == STAGES
