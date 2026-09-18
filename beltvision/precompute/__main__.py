"""CLI entrypoint for the precompute lane: ``python -m beltvision.precompute ...``.

Trains the conv-AE (-> ONNX opset 17), fits the PaDiM / PatchCore-lite banks on the real
normal frames, and writes the held-out benchmark. Deterministic given ``--seed``.

Example::

    python -m beltvision.precompute \
        --data  ../Acc_cv_colia/data/reference/ironore \
        --models-out ../Acc_cv_colia/models \
        --bench-out  ../Acc_cv_colia/data/derived/benchmark
"""
from __future__ import annotations

import argparse
import json
import sys

from . import run_precompute


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m beltvision.precompute",
        description="Train belt-specific learned anomaly models + export ONNX + benchmark.",
    )
    p.add_argument("--data", required=True, help="dir with normal/ foreign_object_*/ anomaly/")
    p.add_argument("--models-out", required=True, help="dir for committed model artifacts")
    p.add_argument("--bench-out", required=True, help="dir for benchmark.json")
    p.add_argument("--seed", type=int, default=34)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--sel-dims", type=int, default=100)
    p.add_argument("--coreset-size", type=int, default=1024)
    p.add_argument("--work-long-side", type=int, default=512)
    p.add_argument("--beyond-sota", action="store_true",
                   help="also evaluate the foundation methods (DINOv2-kNN, OWLv2); needs a GPU + [gpu] extra")
    p.add_argument("--device", default="cpu", help="device for the beyond-SOTA backbones (e.g. cuda)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_precompute(
        data_dir=args.data,
        models_out=args.models_out,
        bench_out=args.bench_out,
        seed=args.seed,
        train_frac=args.train_frac,
        input_size=args.input_size,
        epochs=args.epochs,
        grid=args.grid,
        sel_dims=args.sel_dims,
        coreset_size=args.coreset_size,
        work_long_side=args.work_long_side,
        beyond_sota=args.beyond_sota,
        device=args.device,
    )
    print(json.dumps(summary["split"], indent=2))
    print("conv_ae:", summary["conv_ae"]["bytes"], "bytes,",
          "final L1", summary["conv_ae"]["final_l1_loss"],
          "onnx-parity", summary["conv_ae"]["onnx_torch_max_abs_diff"])
    print("padim bank:", summary["padim_bytes"], "bytes; patchcore bank:",
          summary["patchcore_bytes"], "bytes")
    for m in summary["methods"]:
        print(f"  {m['id']:20s} AUROC={m['auroc']:.4f}  AP={m['ap']:.4f}")
    print("benchmark ->", summary["benchmark_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
