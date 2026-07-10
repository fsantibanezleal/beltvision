"""Case image loaders.

Real cases load a committed frame from ``data/reference`` or ``data/examples``. When
that frame is not present (for example the proprietary COLA 34 reference lives in the
git-ignored vault), a case may fall back to a deterministic synthetic scene so the
pipeline and its tests always run offline without private data.
"""
from __future__ import annotations

from .synthetic import synth_belt_scene

__all__ = ["synth_belt_scene"]
