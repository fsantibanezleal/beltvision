"""Case registry: the cases the pipeline can run, grouped by CATEGORY.

Every case carries a category (so ``docs/cases/`` can render a coverage matrix), a
source + license (so committed artifacts stay license-clean), the capabilities it
exercises, and whether it is synthetic. Non-commercial (vault-only) sources are
marked so CI and reviewers can keep them out of the public artifact set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cases.synthetic import synth_belt_scene

# Reference anchor: Case 1 is the selected COLA 34 frame (crushed ore, mesh, dust).
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass(frozen=True)
class CaseSpec:
    """Static description of one case."""

    case_id: str
    category: str
    source: str
    license: str
    capabilities: tuple[str, ...]
    synthetic: bool = False
    commercial_safe: bool = True  # False => vault-only, never in the public artifact
    tear: bool = False            # synthetic scenes: paint a scripted tear
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# The catalogue mirrors the build plan's case table. Real sources whose data lives in
# the git-ignored vault load their committed frame from data/; when that frame is
# absent the loader raises (real cases are never faked), except the two synthetic
# controls which are generated deterministically.
CASES: dict[str, CaseSpec] = {
    "cola34_reference": CaseSpec(
        "cola34_reference", "reference",
        "COLA 34 proprietary CCTV frame", "proprietary-internal",
        ("preprocess", "edges", "geometry", "segmentation", "granulometry", "anomaly"),
        commercial_safe=False,
        notes="Anchor frame: crushed ore, protective mesh, extreme dust. Vault-only.",
    ),
    "beltcrack_seq": CaseSpec(
        "beltcrack_seq", "belt-crack",
        "BeltCrack14ks/9kd", "unverified",
        ("segmentation", "anomaly", "tracking"),
        commercial_safe=False, notes="License unverified; vault-only until confirmed.",
    ),
    "ironore_foreign": CaseSpec(
        "ironore_foreign", "foreign-object",
        "Mendeley 10.17632/s25x2bnshz.1", "CC-BY-4.0",
        ("detection", "tracking"),
        notes="Confirm CC BY on the Mendeley record before shipping.",
    ),
    "ironore_flow": CaseSpec(
        "ironore_flow", "granulometry",
        "Mendeley iron-ore video", "CC-BY-4.0",
        ("granulometry", "tracking"),
    ),
    "mvtec_leather": CaseSpec(
        "mvtec_leather", "anomaly-proxy",
        "MVTec AD leather", "non-commercial",
        ("anomaly",), commercial_safe=False,
    ),
    "mvtec_tile": CaseSpec(
        "mvtec_tile", "anomaly-proxy",
        "MVTec AD tile", "non-commercial",
        ("anomaly",), commercial_safe=False,
    ),
    "mvtec_ad2_hard": CaseSpec(
        "mvtec_ad2_hard", "anomaly-hard",
        "MVTec AD 2", "non-commercial",
        ("anomaly",), commercial_safe=False,
    ),
    "severstal_seg": CaseSpec(
        "severstal_seg", "surface-seg",
        "Severstal steel defects", "competition-rules",
        ("segmentation",), commercial_safe=False,
    ),
    "neu_detect": CaseSpec(
        "neu_detect", "surface-detect",
        "NEU-DET surface defects", "research-use",
        ("detection",),
    ),
    "kolektor_imbalance": CaseSpec(
        "kolektor_imbalance", "anomaly-imbalance",
        "KolektorSDD2", "non-commercial",
        ("anomaly",), commercial_safe=False,
    ),
    "mot17_track": CaseSpec(
        "mot17_track", "tracking",
        "MOT17", "non-commercial",
        ("tracking",), commercial_safe=False,
    ),
    "dancetrack_similar": CaseSpec(
        "dancetrack_similar", "tracking-hard",
        "DanceTrack", "non-commercial",
        ("tracking",), commercial_safe=False,
    ),
    "synth_psd_gt": CaseSpec(
        "synth_psd_gt", "synthetic-control",
        "Colia synthetic (labeled)", "synthetic",
        ("preprocess", "edges", "granulometry", "anomaly"),
        synthetic=True, notes="Labeled synthetic control with known particle sizes.",
    ),
    "synth_tear_gt": CaseSpec(
        "synth_tear_gt", "synthetic-control",
        "Colia synthetic (labeled)", "synthetic",
        ("preprocess", "edges", "anomaly", "segmentation"),
        synthetic=True, tear=True, notes="Labeled synthetic control with a scripted tear.",
    ),
}

# Frozen mechanism: cases grouped by category.
CATEGORIES: dict[str, list[str]] = {}
for _cid, _spec in CASES.items():
    CATEGORIES.setdefault(_spec.category, []).append(_cid)


def get_case(case_id: str) -> CaseSpec:
    if case_id not in CASES:
        raise KeyError(f"unknown case {case_id!r}; known: {sorted(CASES)}")
    return CASES[case_id]


def list_cases() -> list[str]:
    return list(CASES.keys())


def cases_by_category() -> dict[str, list[str]]:
    return {cat: list(ids) for cat, ids in CATEGORIES.items()}


def public_cases() -> list[str]:
    """Cases whose license permits shipping a committed artifact publicly."""
    return [cid for cid, spec in CASES.items() if spec.commercial_safe]


def _find_committed_frame(case_id: str, data_root: Path) -> Path | None:
    for sub in ("reference", "examples"):
        for ext in IMAGE_EXTS:
            candidate = data_root / sub / f"{case_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_image(case_id: str, data_root: str | Path = "data", seed: int = 34) -> np.ndarray:
    """Load a BGR frame for a case.

    Synthetic cases are generated deterministically. Real cases load a committed
    frame from ``data/reference`` or ``data/examples``; if the frame is absent (its
    data lives in the vault) a ``FileNotFoundError`` is raised. Real cases are never
    faked with synthetic pixels.
    """
    spec = get_case(case_id)
    if spec.synthetic:
        return synth_belt_scene(seed=seed, with_tear=spec.tear)

    frame = _find_committed_frame(case_id, Path(data_root))
    if frame is None:
        raise FileNotFoundError(
            f"no committed frame for case {case_id!r} under {data_root}/reference|examples; "
            "its data lives in the git-ignored vault (see data/README.md)"
        )
    import cv2

    img = cv2.imread(str(frame), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read frame for case {case_id!r}: {frame}")
    return img
