"""beltvision.precompute: the offline PRECOMPUTE lane for belt-specific learned anomaly.

Trains the belt-specific learned anomaly models on REAL normal frames, exports the conv-AE
to ONNX so the weight-gated live method becomes real, fits the PaDiM / PatchCore-lite banks,
and produces an honest held-out learned-vs-classical benchmark. Everything is deterministic
given the seed and reproducible via::

    python -m beltvision.precompute --data <dir> --models-out <dir> --bench-out <dir>

This lane needs the heavy extra (torch, onnx, onnxruntime, scikit-learn, torchvision); it is
never imported by the slim runtime. See ``beltvision.methods.anomaly`` for the live serving
side that consumes the exported ONNX.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backbone import ResNetPatchFeatures
from .benchmark import _downscale, compute_benchmark
from .dataset import build_split
from .train import (
    fit_padim,
    fit_patchcore,
    save_padim,
    save_patchcore,
    train_conv_ae,
)

__all__ = ["run_precompute", "build_split", "compute_benchmark"]

_CONV_AE_NAME = "conv_ae.onnx"
_PADIM_NAME = "padim_ironore.npz"
_PATCHCORE_NAME = "patchcore_ironore.npz"
_BENCH_NAME = "benchmark.json"
_DESCRIPTOR_NAME = "learned_artifacts.json"


def _load_dataset_meta(data_dir: Path) -> dict[str, Any]:
    manifest = data_dir / "manifest.json"
    if not manifest.is_file():
        return {"name": "ironore_foreign_object", "note": "no manifest.json found next to frames"}
    m = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "name": m.get("dataset", "ironore_foreign_object"),
        "title": m.get("title"),
        "doi": m.get("doi"),
        "record": m.get("record"),
        "license": m.get("license"),
        "attribution": m.get("attribution"),
        "counts_by_category": m.get("counts_by_category"),
        "n_frames": m.get("n_frames"),
    }


def run_precompute(
    *,
    data_dir: str | Path,
    models_out: str | Path,
    bench_out: str | Path,
    seed: int = 34,
    train_frac: float = 0.7,
    input_size: int = 256,
    epochs: int = 120,
    grid: int = 8,
    sel_dims: int = 100,
    coreset_size: int = 1024,
    work_long_side: int = 512,
) -> dict[str, Any]:
    """Run the full precompute lane: train + fit + export + benchmark. Returns a summary."""
    data_dir = Path(data_dir)
    models_out = Path(models_out)
    bench_out = Path(bench_out)
    models_out.mkdir(parents=True, exist_ok=True)
    bench_out.mkdir(parents=True, exist_ok=True)

    split = build_split(data_dir, seed=seed, train_frac=train_frac)
    split.assert_leakage_free()

    # 1) conv-AE (L1) -> ONNX (opset 17)
    onnx_path = models_out / _CONV_AE_NAME
    conv_meta = train_conv_ae(
        split.train_normal,
        onnx_out=onnx_path,
        input_size=input_size,
        epochs=epochs,
        seed=seed,
    )

    # 2) frozen-backbone features of the normal training frames (same pipeline as eval)
    backbone = ResNetPatchFeatures(input_size=input_size, grid=grid)
    train_frames = [_downscale(f.bgr, work_long_side) for f in split.train_normal]
    train_feats = backbone.extract(train_frames)

    # 3) PaDiM + PatchCore-lite banks (compact, committed under models/)
    padim = fit_padim(train_feats, sel_dims=sel_dims, seed=seed)
    padim_bytes = save_padim(padim, models_out / _PADIM_NAME, grid=grid, input_size=input_size)
    patchcore = fit_patchcore(train_feats, coreset_size=coreset_size, seed=seed)
    patchcore_bytes = save_patchcore(
        patchcore, models_out / _PATCHCORE_NAME, grid=grid, input_size=input_size
    )

    # 4) held-out learned-vs-classical benchmark
    dataset_meta = _load_dataset_meta(data_dir)
    benchmark = compute_benchmark(
        split,
        onnx_path=str(onnx_path),
        onnx_bytes=int(conv_meta["bytes"]),
        padim=padim,
        padim_bytes=padim_bytes,
        patchcore=patchcore,
        patchcore_bytes=patchcore_bytes,
        backbone=backbone,
        seed=seed,
        work_long_side=work_long_side,
        input_size=input_size,
        dataset_meta=dataset_meta,
        conv_ae_meta={
            "epochs": conv_meta["epochs"],
            "n_train": conv_meta["n_train"],
            "final_l1_loss": conv_meta["final_l1_loss"],
            "onnx_torch_max_abs_diff": conv_meta["onnx_torch_max_abs_diff"],
            "opset": conv_meta["opset"],
            "input_shape": conv_meta["input_shape"],
        },
    )
    bench_path = bench_out / _BENCH_NAME
    bench_path.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")

    # 5) a compact descriptor of the committed learned artifacts (provenance for models/)
    descriptor = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": int(seed),
        "trained_on": dataset_meta,
        "split": split.counts(),
        "artifacts": [
            {
                "name": _CONV_AE_NAME,
                "method": "anomaly.conv_ae",
                "family": "learned",
                "bytes": int(conv_meta["bytes"]),
                "opset": conv_meta["opset"],
                "input_shape": conv_meta["input_shape"],
                "onnx_torch_max_abs_diff": conv_meta["onnx_torch_max_abs_diff"],
                "reference": conv_meta["reference"],
            },
            {
                "name": _PADIM_NAME,
                "method": "anomaly.padim",
                "family": "learned",
                "bytes": int(padim_bytes),
                "backbone": "torchvision resnet18 IMAGENET1K_V1 (frozen) layer2+layer3",
                "grid": grid,
                "sel_dims": padim["sel_dims"],
                "reference": padim["reference"],
            },
            {
                "name": _PATCHCORE_NAME,
                "method": "anomaly.patchcore",
                "family": "learned",
                "bytes": int(patchcore_bytes),
                "backbone": "torchvision resnet18 IMAGENET1K_V1 (frozen) layer2+layer3",
                "coreset_size": patchcore["coreset_size"],
                "reference": patchcore["reference"],
            },
        ],
    }
    (models_out / _DESCRIPTOR_NAME).write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    return {
        "split": split.counts(),
        "conv_ae": conv_meta,
        "padim_bytes": padim_bytes,
        "patchcore_bytes": patchcore_bytes,
        "benchmark_path": str(bench_path),
        "models_out": str(models_out),
        "methods": [
            {"id": m["id"], "auroc": m["image_auroc"], "ap": m["average_precision"]}
            for m in benchmark["methods"]
        ],
    }
