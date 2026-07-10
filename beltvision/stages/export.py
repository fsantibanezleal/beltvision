"""Stage 6: export.

Writes the two committed outputs of a run: the compact replay artifact (the anomaly
heatmap grid plus the trace, small enough to ship) and the Contract 2 manifest that
describes the run. ``build_manifest`` validates before writing, so an export can
never emit a manifest the gate/schema would reject.

Rework surface: exporting trained heavy models to ONNX (+INT8) into ``models/`` and
recording their measured descriptors happens here in the precompute lane. The frozen
part is that every run ends by writing a validated manifest and a compact artifact.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..context import StageContext
from ..core.manifest import build_manifest, jsonable
from ..core.trace import stage_timer


def _write_artifact(ctx: StageContext) -> dict[str, Any]:
    grid = ctx.state.get("anomaly_grid")
    artifact = {
        "case_id": ctx.case_id,
        "category": ctx.category,
        "preprocess": ctx.state.get("preprocess_summary", {}),
        "anomaly_grid": jsonable(grid.round(4)) if grid is not None else None,
        "trace": ctx.trace.to_dict(),
    }
    path = ctx.out_root / "artifacts" / f"{ctx.case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(artifact), separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")
    return {"path": str(path), "bytes": len(payload.encode("utf-8")), "format": "json"}


def export(ctx: StageContext) -> dict[str, Any]:
    with stage_timer(ctx.trace, "export") as t:
        artifact = _write_artifact(ctx)

        models = [
            {
                "name": "patch_gaussian_normal_model",
                "kind": "statistical",
                "bytes": int(ctx.state.get("normal_model_bytes", 0)),
                "format": "in-manifest",
            }
        ]

        manifest = build_manifest(
            case_id=ctx.case_id,
            category=ctx.category,
            source=ctx.spec.source,
            license=ctx.spec.license,
            seed=ctx.seed,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            preprocess=ctx.state.get("preprocess_summary", {}),
            methods=ctx.state["methods"],
            models=models,
            artifact=artifact,
        )

        manifest_path = ctx.out_root / "manifests" / f"{ctx.case_id}.json"
        from ..core.manifest import write_manifest

        write_manifest(manifest, manifest_path)

        ctx.state["manifest"] = manifest
        ctx.state["manifest_path"] = str(manifest_path)
        t.note(manifest=str(manifest_path), artifact_bytes=artifact["bytes"])
    return {"manifest_path": str(manifest_path), "artifact": artifact}
