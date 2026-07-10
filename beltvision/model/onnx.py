"""ONNX artifact descriptors (framework-free).

Training and ONNX export are precompute-lane operations that require the heavy venv
(torch/anomalib/ultralytics/transformers). Those imports live inside ``stages/`` and
run offline. This module only *describes* an exported artifact so the runtime, the
gate, and the manifest can reason about it with numpy-or-less.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OnnxArtifact:
    """A compact, committed ONNX model artifact and its measured properties."""

    name: str
    path: str
    bytes: int
    opset: int
    input_shape: tuple[int, ...]
    mean_infer_ms_cpu: float
    quantized_int8: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "bytes": int(self.bytes),
            "opset": int(self.opset),
            "input_shape": list(self.input_shape),
            "mean_infer_ms_cpu": round(float(self.mean_infer_ms_cpu), 3),
            "quantized_int8": bool(self.quantized_int8),
        }


def describe_artifact(
    path: str | Path,
    *,
    name: str,
    opset: int,
    input_shape: tuple[int, ...],
    mean_infer_ms_cpu: float,
    quantized_int8: bool = False,
) -> OnnxArtifact:
    """Build an ``OnnxArtifact``, reading the on-disk byte size when the file exists."""
    p = Path(path)
    size = p.stat().st_size if p.exists() else 0
    return OnnxArtifact(
        name=name,
        path=str(path),
        bytes=size,
        opset=opset,
        input_shape=tuple(int(x) for x in input_shape),
        mean_infer_ms_cpu=float(mean_infer_ms_cpu),
        quantized_int8=quantized_int8,
    )
