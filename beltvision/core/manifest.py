"""Contract 2: the artifact manifest (pipeline -> consumer).

The manifest is the compact, audited description of one case run that a viewer
replays. Every numeric claim a consumer shows must trace back to a field here, and
any schema mirror a consumer keeps (for example a TypeScript type in a web front end)
must match this schema exactly or the consumer's build fails (the drift guard).

Design rules:
- JSON-only primitive types (str, int, float, bool, list, dict). No numpy leaks: run
  values through ``jsonable`` before writing.
- Every method result carries the measured gate inputs and the lane verdict, so a
  consumer can render the lane story of a manifest 1:1.
- The schema is versioned; a breaking change bumps ``MANIFEST_SCHEMA_VERSION`` and
  any consumer's mirror together.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gate import LANES

MANIFEST_SCHEMA_VERSION = "1.0"

# Required keys for a case manifest and for each method result. The validator and
# the TS mirror both key off these, so keep the two in lockstep.
CASE_REQUIRED_KEYS = (
    "schema_version",
    "case_id",
    "category",
    "source",
    "license",
    "seed",
    "created_utc",
    "preprocess",
    "methods",
    "models",
    "artifact",
)

METHOD_REQUIRED_KEYS = (
    "method",
    "capability",
    "tier",
    "lane",
    "model_bytes",
    "infer_ms",
    "trace_bytes",
    "web_drivable",
    "metrics",
)

# Compute tiers a method may declare (documentation axis, distinct from the lane
# which is measured by the gate).
TIERS = ("classical", "learned", "foundation")


def jsonable(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays and tuples to JSON-native types.

    Gives the pipeline and any consuming runtime one shared definition of a JSON-safe
    payload, closing the classic numpy-into-JSON leak at the contract boundary.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    # numpy is optional at import time; handle it structurally without importing it.
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return jsonable(obj.tolist())
    if hasattr(obj, "item") and obj.__class__.__module__ == "numpy":
        return jsonable(obj.item())
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else None  # NaN/Inf -> None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", "replace")
    return obj


def build_method_result(
    *,
    method: str,
    capability: str,
    tier: str,
    verdict: Any,
    metrics: dict[str, Any],
    artifact_ref: str | None = None,
    reference: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Assemble one method result from a gate ``LaneVerdict`` and its metrics."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return jsonable(
        {
            "method": method,
            "capability": capability,
            "tier": tier,
            "lane": verdict.lane,
            "model_bytes": verdict.model_bytes,
            "infer_ms": verdict.infer_ms,
            "trace_bytes": verdict.trace_bytes,
            "web_drivable": verdict.web_drivable,
            "metrics": metrics,
            "artifact_ref": artifact_ref,
            "reference": reference,
            "notes": notes,
        }
    )


def build_manifest(
    *,
    case_id: str,
    category: str,
    source: str,
    license: str,  # noqa: A002 - the field is named "license" in the schema
    seed: int,
    created_utc: str,
    preprocess: dict[str, Any],
    methods: list[dict[str, Any]],
    models: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a full case manifest and validate it before returning."""
    manifest = jsonable(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "case_id": case_id,
            "category": category,
            "source": source,
            "license": license,
            "seed": int(seed),
            "created_utc": created_utc,
            "preprocess": preprocess,
            "methods": methods,
            "models": models,
            "artifact": artifact,
        }
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``manifest`` violates Contract 2.

    Called by ``build_manifest`` and by ``tests/test_manifest.py``; CI runs the same
    check so a drifted or malformed manifest fails the build.
    """
    missing = [k for k in CASE_REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing keys: {missing}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {manifest['schema_version']!r} != {MANIFEST_SCHEMA_VERSION!r}"
        )
    if not isinstance(manifest["methods"], list) or not manifest["methods"]:
        raise ValueError("manifest.methods must be a non-empty list")
    for i, m in enumerate(manifest["methods"]):
        mmiss = [k for k in METHOD_REQUIRED_KEYS if k not in m]
        if mmiss:
            raise ValueError(f"methods[{i}] missing keys: {mmiss}")
        if m["lane"] not in LANES:
            raise ValueError(f"methods[{i}].lane {m['lane']!r} not in {LANES}")
        if not isinstance(m["metrics"], dict):
            raise ValueError(f"methods[{i}].metrics must be a dict")


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Validate and write a manifest as pretty JSON. Returns the path written."""
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a manifest from disk."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def build_index(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact index over many case manifests.

    Groups case ids by category and rolls up how many methods land in each lane, so
    the web app and CI can reason over the whole corpus without loading every
    artifact.
    """
    by_category: dict[str, list[str]] = {}
    lane_counts: dict[str, int] = {lane: 0 for lane in LANES}
    cases: list[dict[str, Any]] = []
    for m in manifests:
        validate_manifest(m)
        by_category.setdefault(m["category"], []).append(m["case_id"])
        for method in m["methods"]:
            lane_counts[method["lane"]] = lane_counts.get(method["lane"], 0) + 1
        cases.append(
            {
                "case_id": m["case_id"],
                "category": m["category"],
                "n_methods": len(m["methods"]),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "n_cases": len(manifests),
        "by_category": by_category,
        "lane_counts": lane_counts,
        "cases": cases,
    }
