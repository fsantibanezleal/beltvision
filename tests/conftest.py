"""Shared pytest fixtures.

``beltvision`` is installed (``pip install -e .[dev]``), so no path shimming is
needed; the suite imports the package the same way a consumer would.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synth_image() -> np.ndarray:
    """A deterministic synthetic belt frame (BGR uint8)."""
    from beltvision.cases.synthetic import synth_belt_scene

    return synth_belt_scene(seed=34, with_tear=True)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """An isolated output root for a pipeline run."""
    return tmp_path / "derived"
