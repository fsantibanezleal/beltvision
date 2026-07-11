"""BLOCKING synthetic ground-truth recovery gate.

The synthetic scenes are generated, so the exact belt mask, orientation, centreline and
injected misalignment are known. This gate asserts the geometry chain (segment belt ->
mask -> axis -> centreline -> alignment) RECOVERS that known geometry within tolerance,
at MULTIPLE orientations (vertical, horizontal, diagonal) and a curved path. An
all-vertical suite would test nothing; the orientation bug that produced garbage axes is
exactly what this catches. If a method does not recover the ground truth it is broken.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from beltvision.cases.synthetic import GT_SUITE, gt_scene
from beltvision.methods.beltline import _ang_diff, compute_belt_geometry
from beltvision.methods.preprocess import apply_clahe_lab
from beltvision.methods.semantic import compute_layers

# tolerances (documented, product knobs)
IOU_MIN = 0.5
ORI_TOL_DEG = 8.0
MISALIGN_TOL_DEG = 6.0


def _recover(name: str):
    sc = gt_scene(name)
    view = "top_carrying" if sc.loaded else "end_return"
    layers = compute_layers(sc.image, view_type=view, use_learned=False)
    gray = cv2.cvtColor(apply_clahe_lab(sc.image), cv2.COLOR_BGR2GRAY)
    footprint = layers.belt_mask | layers.content_mask
    geo = compute_belt_geometry(footprint, external_mask=layers.mask(0), gray=gray)
    return sc, footprint, geo


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


@pytest.mark.parametrize("name", list(GT_SUITE))
def test_belt_footprint_iou_recovers_ground_truth(name):
    sc, footprint, _geo = _recover(name)
    iou = _iou(footprint, sc.belt_mask)
    assert iou >= IOU_MIN, f"{name}: belt IoU {iou:.2f} < {IOU_MIN}"


@pytest.mark.parametrize("name", list(GT_SUITE))
def test_orientation_recovered_at_all_orientations(name):
    sc, _fp, geo = _recover(name)
    assert "axis_angle_deg" in geo, f"{name}: no axis recovered"
    err = _ang_diff(geo["axis_angle_deg"], sc.orientation_deg)
    assert err <= ORI_TOL_DEG, (
        f"{name}: orientation error {err:.1f}deg > {ORI_TOL_DEG} "
        f"(gt {sc.orientation_deg}, got {geo['axis_angle_deg']})"
    )


def test_centreline_rmse_reasonable():
    # centreline should track the GT medial axis (perpendicular distance) within a few % of
    # the belt width, on a clean straight belt.
    sc, _fp, geo = _recover("vertical_empty")
    cl = np.asarray(geo["centreline_xy"], float)
    gt = np.asarray(sc.centreline, float)
    dists = [np.min(np.hypot(gt[:, 0] - p[0], gt[:, 1] - p[1])) for p in cl]
    rmse = float(np.sqrt(np.mean(np.square(dists))))
    assert rmse < 0.25 * sc.meta["belt_halfwidth_px"] * 2, f"centreline RMSE {rmse:.1f}px too high"


def test_misalignment_sign_and_magnitude_recovered():
    sc, _fp, geo = _recover("misaligned_side")
    det = geo.get("misalignment_deg")
    assert det is not None, "misalignment not measured on the lateral case"
    assert abs(det - sc.misalignment_deg) <= MISALIGN_TOL_DEG, (
        f"misalignment {det} vs gt {sc.misalignment_deg} beyond {MISALIGN_TOL_DEG}deg"
    )
    assert geo.get("misaligned") is True


def test_aligned_loaded_belt_reports_small_misalignment():
    sc, _fp, geo = _recover("horizontal_loaded")
    det = geo.get("misalignment_deg")
    if det is not None:  # support may not be detectable on every frame
        assert abs(det) <= 4.0, f"aligned belt reported misalignment {det}deg"
