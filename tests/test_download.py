"""Model-weight provisioning tests: locate + graceful absence + opt-in download contract.

Downloads are opt-in and never triggered here; the suite stays fully offline. What is
tested is the contract the learned methods rely on: an absent weight yields ``None`` (not
an exception), the weights directory is env-overridable, and the spec metadata is honest.
"""
from __future__ import annotations

import pytest

from beltvision.models import (
    WEIGHTS,
    download_weight,
    ensure_weight,
    is_present,
    weight_path,
    weights_dir,
)


@pytest.fixture
def isolated_weights_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BELTVISION_WEIGHTS_DIR", str(tmp_path / "weights"))
    return tmp_path / "weights"


def test_registry_has_permissive_specs():
    assert {"mobile_sam", "detector_onnx", "conv_ae_onnx"} <= set(WEIGHTS)
    for spec in WEIGHTS.values():
        assert spec.approx_bytes > 0
        assert spec.license and spec.reference
        assert spec.filename


def test_weights_dir_is_env_overridable(isolated_weights_dir):
    d = weights_dir()
    assert d == isolated_weights_dir
    assert d.is_dir()


def test_absent_weight_returns_none_without_download(isolated_weights_dir):
    assert ensure_weight("mobile_sam") is None
    assert not is_present("mobile_sam")


def test_present_weight_is_located(isolated_weights_dir):
    target = weight_path("mobile_sam")
    target.write_bytes(b"not-a-real-model-but-present")
    assert is_present("mobile_sam")
    assert ensure_weight("mobile_sam") == target


def test_unknown_weight_raises_keyerror(isolated_weights_dir):
    with pytest.raises(KeyError):
        ensure_weight("no_such_weight")


def test_download_without_url_raises_loudly(isolated_weights_dir):
    # conv_ae_onnx has no committed URL (honesty): the loud path raises a clear error.
    with pytest.raises(RuntimeError, match="no download URL"):
        download_weight("conv_ae_onnx")


def test_ensure_with_failed_download_stays_graceful(isolated_weights_dir):
    # Opt-in download of a URL-less weight must degrade to None, never raise.
    assert ensure_weight("conv_ae_onnx", download=True) is None


def test_url_env_override(isolated_weights_dir, monkeypatch):
    monkeypatch.setenv("BELTVISION_DETECTOR_ONNX_URL", "https://example.invalid/model.onnx")
    assert WEIGHTS["detector_onnx"].resolved_url() == "https://example.invalid/model.onnx"
