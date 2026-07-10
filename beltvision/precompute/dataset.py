"""Leakage-safe dataset assembly for the belt-anomaly precompute lane.

The iron-ore reference sample is laid out as one directory per category::

    <data>/normal/*.jpg               # defect-free ore flow (the "normal" class)
    <data>/foreign_object_wood/*.jpg  # wood debris on the stream (anomalous)
    <data>/foreign_object_tool/*.jpg  # tools / objects on the belt (anomalous)
    <data>/anomaly/*.jpg              # author-flagged anomaly frames (anomalous)

Anomaly-detection discipline (MVTec / research file 14 section 2): the learned models
see ONLY normal frames in training; every anomalous frame is test-only, and a subset of
the normal frames is held out for the negative side of the test set. No frame appears in
both train and test (checked here, and by ``tests/test_precompute.py``). The split is a
pure function of the seed so a run is reproducible.

This is a SMALL-SAMPLE proxy: a few dozen frames per class, frame-level (not
sequence-level) split. Frames are sampled spread-out across the source recording (the
downloader's ``pick_spaced``), which limits adjacent-frame near-duplicate leakage, but the
honest label on every number produced downstream is "small-sample proxy" (see benchmark).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Category -> is-anomalous. "normal" is the only defect-free class; the rest are positives.
CATEGORIES: dict[str, bool] = {
    "normal": False,
    "foreign_object_wood": True,
    "foreign_object_tool": True,
    "anomaly": True,
}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Frame:
    """One loaded frame: its path, category, anomaly label, and BGR pixels."""

    path: Path
    category: str
    is_anomalous: bool
    bgr: np.ndarray


@dataclass
class Split:
    """A leakage-safe split: normal-only train; held-out normal + all anomalous test."""

    train_normal: list[Frame] = field(default_factory=list)
    heldout_normal: list[Frame] = field(default_factory=list)
    heldout_anomalous: list[Frame] = field(default_factory=list)

    @property
    def heldout(self) -> list[Frame]:
        """The full held-out evaluation set (normal negatives + anomalous positives)."""
        return self.heldout_normal + self.heldout_anomalous

    @property
    def labels(self) -> np.ndarray:
        """Binary labels for :pyattr:`heldout` (0 = normal, 1 = anomalous)."""
        return np.array(
            [0] * len(self.heldout_normal) + [1] * len(self.heldout_anomalous), dtype=np.int32
        )

    def counts(self) -> dict[str, int]:
        return {
            "train_normal": len(self.train_normal),
            "heldout_normal": len(self.heldout_normal),
            "heldout_anomalous": len(self.heldout_anomalous),
            "n_pos": len(self.heldout_anomalous),
            "n_neg": len(self.heldout_normal),
        }

    def assert_leakage_free(self) -> None:
        """Raise if any frame path appears in more than one bucket."""
        train = {f.path.resolve() for f in self.train_normal}
        held = {f.path.resolve() for f in self.heldout}
        overlap = train & held
        if overlap:
            raise AssertionError(f"split leakage: {len(overlap)} frame(s) in train AND test")


def _list_category(data_dir: Path, category: str) -> list[Path]:
    cat_dir = data_dir / category
    if not cat_dir.is_dir():
        return []
    files = [p for p in cat_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file()]
    return sorted(files, key=lambda p: p.name)


def _read_bgr(path: Path) -> np.ndarray | None:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def build_split(data_dir: str | Path, *, seed: int = 34, train_frac: float = 0.65) -> Split:
    """Assemble the leakage-safe split from the category directories.

    Normal frames are shuffled deterministically and cut at ``train_frac`` into
    train / held-out; every anomalous frame is held-out. Missing/undecodable files are
    skipped (never fabricated).
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(int(seed))

    normals: list[Frame] = []
    anomalous: list[Frame] = []
    for category, is_anom in CATEGORIES.items():
        for path in _list_category(data_dir, category):
            bgr = _read_bgr(path)
            if bgr is None:
                continue
            frame = Frame(path=path, category=category, is_anomalous=is_anom, bgr=bgr)
            (anomalous if is_anom else normals).append(frame)

    if not normals:
        raise FileNotFoundError(
            f"no NORMAL frames under {data_dir}/normal; run scripts/download_datasets.py first"
        )

    order = rng.permutation(len(normals))
    n_train = max(1, int(round(len(normals) * float(train_frac))))
    n_train = min(n_train, len(normals) - 1) if len(normals) > 1 else 1
    train_idx = set(order[:n_train].tolist())

    split = Split(
        train_normal=[normals[i] for i in sorted(train_idx)],
        heldout_normal=[normals[i] for i in range(len(normals)) if i not in train_idx],
        heldout_anomalous=anomalous,
    )
    split.assert_leakage_free()
    return split
