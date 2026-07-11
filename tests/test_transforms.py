"""Classical transform records: real overlays + finite metrics (beltvision.methods.transforms)."""
from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from beltvision.cases.synthetic import synth_scene
from beltvision.methods import transforms as tf


def _is_png_data_url(s: object) -> bool:
    if not isinstance(s, str) or not s.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(s.split(",", 1)[1])
    return len(raw) > 500 and raw[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def belt_image() -> np.ndarray:
    return synth_scene(orientation_deg=90.0, loaded=True).image


_RECORD_FNS = [
    tf.fft_spectrum, tf.fft_orientation, tf.fft_filter, tf.phot,
    tf.dwt_decompose, tf.dwt_reconstruct, tf.wavelet_denoise,
]


@pytest.mark.parametrize("fn", _RECORD_FNS, ids=lambda f: f.__name__)
def test_transform_returns_real_overlay_and_finite_metric(fn, belt_image):
    rec = fn(belt_image)
    assert rec["status"] == "ok"
    assert rec["tier"] == "classical"
    assert _is_png_data_url(rec["overlay_b64"]), f"{fn.__name__} overlay is not a real PNG"
    assert math.isfinite(rec["metric_value"]), f"{fn.__name__} metric not finite"
    assert rec["family"] in (tf.FAM_FREQ, tf.FAM_WAVELET)


def test_transform_overlays_are_distinct(belt_image):
    overlays = {fn(belt_image)["overlay_b64"] for fn in _RECORD_FNS}
    assert len(overlays) == len(_RECORD_FNS), "some transform overlays are identical"


def test_fft_orientation_range_and_period(belt_image):
    rec = tf.fft_orientation(belt_image)
    assert 0.0 <= rec["metric_value"] < 180.0
    assert rec["period_px"] > 0.0


def test_dwt_reconstruct_removing_approx_keeps_only_residual(belt_image):
    # keeping the detail bands only must leave a bounded residual-energy fraction.
    rec = tf.dwt_reconstruct(belt_image, keep=["detail"])
    assert 0.0 <= rec["metric_value"] <= 1.0
    assert rec["keep"] == ["detail"]


def test_fft_filter_kinds_all_run(belt_image):
    for kind in ("directional", "band", "low", "high", "notch"):
        rec = tf.fft_filter(belt_image, kind=kind)
        assert rec["status"] == "ok"
        assert 0.0 <= rec["metric_value"] <= 1.0
        assert _is_png_data_url(rec["overlay_b64"])


def test_array_helpers_expose_threadable_images(belt_image):
    import cv2

    gray = cv2.cvtColor(belt_image, cv2.COLOR_BGR2GRAY)
    recon, retained = tf.fft_reconstruct_array(gray, kind="low")
    assert recon.shape == gray.shape and 0.0 <= retained <= 1.0
    res, frac, lvl = tf.dwt_reconstruct_array(gray, keep=["detail"])
    assert res.shape == gray.shape and lvl >= 1
    den, removed = tf.wavelet_denoise_array(gray)
    assert den.shape == gray.shape and removed >= 0.0
