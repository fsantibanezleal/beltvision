"""The object threaded through the six pipeline stages.

Stages are pure-ish functions of ``(StageContext) -> dict``: they read the frame and
prior results from ``ctx.state``, do their work, stash results back into
``ctx.state``, and record timing into ``ctx.trace``. Keeping all shared state on one
context (rather than passing tuples around) is what lets the stage *signatures* stay
frozen while the stage *bodies* are the per-product rework surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core.rng import DEFAULT_SEED, seeded_rng
from .core.trace import Trace
from .io.schema import IngestionParams
from .registry import CaseSpec


@dataclass
class StageContext:
    """Mutable per-run context shared by every stage."""

    spec: CaseSpec
    image_bgr: np.ndarray
    seed: int = DEFAULT_SEED
    quick: bool = False
    params: IngestionParams = field(default_factory=IngestionParams)
    data_root: Path = field(default_factory=lambda: Path("data"))
    out_root: Path = field(default_factory=lambda: Path("data/derived"))
    trace: Trace | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = Trace(case_id=self.spec.case_id)

    @property
    def case_id(self) -> str:
        return self.spec.case_id

    @property
    def category(self) -> str:
        return self.spec.category

    def rng(self, salt: str = "") -> np.random.Generator:
        from .core.rng import derive_seed

        return seeded_rng(derive_seed(self.seed, salt) if salt else self.seed)
