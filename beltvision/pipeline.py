"""The pipeline orchestrator.

One entry point runs the six frozen stages in order for one case or all cases. A run
is idempotent, deterministic in ``(params, seed)``, and offline. The stage NAMES are
frozen as ``STAGES``; the stage bodies live in ``beltvision.stages``.

CLI
---
    python -m beltvision.pipeline <case>       # run one case
    python -m beltvision.pipeline all          # run every case (skips cases
                                               #   whose data is not present)
    python -m beltvision.pipeline all --seed 7 --quick --out data/derived
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .context import StageContext
from .core.manifest import build_index
from .core.rng import DEFAULT_SEED
from .registry import get_case, list_cases, load_image
from .stages import evaluate, export, feature_extraction, infer, preprocess, train

# Frozen stage names, in execution order.
STAGES: tuple[str, ...] = (
    "preprocess",
    "feature_extraction",
    "train",
    "infer",
    "evaluate",
    "export",
)

# Name -> stage function. Keys must equal STAGES.
STAGE_FUNCS: dict[str, Callable[[StageContext], dict[str, Any]]] = {
    "preprocess": preprocess,
    "feature_extraction": feature_extraction,
    "train": train,
    "infer": infer,
    "evaluate": evaluate,
    "export": export,
}

assert tuple(STAGE_FUNCS) == STAGES, "STAGE_FUNCS keys must match STAGES order"


def _maybe_downscale(img, quick: bool):
    if not quick:
        return img
    import cv2

    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= 192:
        return img
    scale = 192 / long_side
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def run_case(
    case_id: str,
    *,
    seed: int = DEFAULT_SEED,
    quick: bool = False,
    data_root: str | Path = "data",
    out_root: str | Path = "data/derived",
) -> dict[str, Any]:
    """Run all six stages for one case and return its validated manifest."""
    spec = get_case(case_id)
    image = _maybe_downscale(load_image(case_id, data_root, seed), quick)
    ctx = StageContext(
        spec=spec,
        image_bgr=image,
        seed=seed,
        quick=quick,
        data_root=Path(data_root),
        out_root=Path(out_root),
    )
    for name in STAGES:
        STAGE_FUNCS[name](ctx)
    return ctx.state["manifest"]


def run_all(
    *,
    seed: int = DEFAULT_SEED,
    quick: bool = False,
    data_root: str | Path = "data",
    out_root: str | Path = "data/derived",
) -> list[dict[str, Any]]:
    """Run every registered case; skip cases whose (vault-only) data is absent."""
    manifests: list[dict[str, Any]] = []
    for case_id in list_cases():
        try:
            manifests.append(
                run_case(
                    case_id, seed=seed, quick=quick, data_root=data_root, out_root=out_root
                )
            )
            print(f"[ok] {case_id}")
        except FileNotFoundError as exc:
            print(f"[skip] {case_id}: {exc}")
    if manifests:
        index = build_index(manifests)
        index_path = Path(out_root) / "manifests" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(f"[index] {index_path} ({index['n_cases']} cases)")
    return manifests


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m beltvision.pipeline",
        description="Run the beltvision offline pipeline for one case or all cases.",
    )
    p.add_argument("case", nargs="?", default="all", help="a case id, or 'all' (default)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic seed")
    p.add_argument("--quick", action="store_true", help="downscale for a fast smoke run")
    p.add_argument("--data", default="data", help="data root (default: data)")
    p.add_argument("--out", default="data/derived", help="output root (default: data/derived)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.case == "all":
        run_all(seed=args.seed, quick=args.quick, data_root=args.data, out_root=args.out)
    else:
        manifest = run_case(
            args.case, seed=args.seed, quick=args.quick, data_root=args.data, out_root=args.out
        )
        print(f"[ok] {args.case}: {len(manifest['methods'])} methods, "
              f"lanes={[m['lane'] for m in manifest['methods']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
