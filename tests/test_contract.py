"""Contract 1 (ingestion) tests: accept / reject / flag with the outlier policy."""
from __future__ import annotations

import numpy as np

from beltvision.io.contract import validate_bytes, validate_image
from beltvision.io.schema import ACCEPT, FLAG, REJECT, IngestionParams


def _solid(h: int, w: int, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_accepts_in_range_high_contrast(synth_image):
    result = validate_image(synth_image)
    assert result.verdict in (ACCEPT, FLAG)
    assert result.accepted
    assert "contrast_std" in result.measured


def test_rejects_too_small():
    result = validate_image(_solid(32, 32))
    assert result.verdict == REJECT
    assert any("short side" in r for r in result.reasons)


def test_rejects_wrong_channels():
    gray = np.full((128, 128), 100, dtype=np.uint8)
    result = validate_image(gray)
    assert result.verdict == REJECT


def test_rejects_wrong_dtype():
    img = np.zeros((128, 128, 3), dtype=np.float32)
    result = validate_image(img)
    assert result.verdict == REJECT


def test_rejects_oversize_long_side():
    # A 1 x 9000 strip: short side ok in channels but long side out of range.
    img = np.full((100, 9000, 3), 128, dtype=np.uint8)
    result = validate_image(img)
    assert result.verdict == REJECT
    assert any("long side" in r for r in result.reasons)


def test_flags_low_contrast_haze():
    # A near-flat frame has L-std below the contrast floor: flagged, not rejected.
    flat = _solid(128, 128, 128)
    result = validate_image(flat)
    assert result.verdict == FLAG
    assert result.accepted and result.flagged


def test_unsupported_format_rejected():
    params = IngestionParams(declared_format="gif")
    result = validate_image(_solid(128, 128), params)
    assert result.verdict == REJECT
    assert any("unsupported image format" in r for r in result.reasons)


def test_validate_bytes_rejects_corrupt():
    result = validate_bytes(b"not-an-image", IngestionParams(declared_format="png"))
    assert result.verdict == REJECT


def test_validate_bytes_size_cap():
    huge = b"\x00" * (21 * 1024 * 1024)
    result = validate_bytes(huge, IngestionParams(media_type="image"))
    assert result.verdict == REJECT
    assert any("cap" in r for r in result.reasons)
