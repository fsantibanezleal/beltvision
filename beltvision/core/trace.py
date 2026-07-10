"""Stage trace: timing and event record for one pipeline run.

The trace is what the replay artifact carries into the web app, and its serialized
size is one of the inputs to the measured gate (a run whose trace is too heavy is
not a live-web candidate). Keeping the trace compact is a hard requirement, not a
nicety, so ``trace_bytes`` is measured and recorded in the manifest.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageRecord:
    """One entry in the trace: a stage name, its wall-clock cost, and notes."""

    name: str
    elapsed_ms: float
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "elapsed_ms": round(self.elapsed_ms, 3), "notes": self.notes}


@dataclass
class Trace:
    """Accumulates per-stage records for a single case run."""

    case_id: str
    stages: list[StageRecord] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def record(self, name: str, elapsed_ms: float, **notes: Any) -> StageRecord:
        rec = StageRecord(name=name, elapsed_ms=elapsed_ms, notes=dict(notes))
        self.stages.append(rec)
        return rec

    def event(self, message: str) -> None:
        self.events.append(message)

    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stages": [s.to_dict() for s in self.stages],
            "events": list(self.events),
        }

    def trace_bytes(self) -> int:
        """Serialized (UTF-8 JSON) size of the trace in bytes.

        This is the ``trace_bytes`` input the gate reads; it must stay small enough
        that the replay artifact is cheap to ship.
        """
        return len(json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8"))


class stage_timer:  # noqa: N801 - context-manager reads as a verb at call sites
    """Context manager that times a stage and records it into a trace.

    Example
    -------
    >>> with stage_timer(trace, "preprocess") as t:
    ...     ...  # do work
    ...     t.note(rows=len(rows))
    """

    def __init__(self, trace: Trace, name: str) -> None:
        self._trace = trace
        self._name = name
        self._notes: dict[str, Any] = {}
        self._start = 0.0

    def note(self, **notes: Any) -> None:
        self._notes.update(notes)

    def __enter__(self) -> stage_timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._trace.record(self._name, elapsed_ms, **self._notes)
