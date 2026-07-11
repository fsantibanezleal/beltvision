"""Locate and (opt-in) download the optional pretrained weights of the learned methods.

Design rules
------------
- **Downloads are opt-in.** A method never triggers a network fetch implicitly; it only
  *locates* a weight already on disk. A human (or an explicit ``download=True`` /
  ``download_weight``) fetches the file once, then the VPS/browser runs inference on the
  cached copy. This keeps the default runtime and the test suite fully offline.
- **httpx, never curl.** curl is a non-authorized app in this environment; the download
  path streams the file with httpx and writes it atomically (temp file then rename).
- **Graceful, never a 500.** :func:`ensure_weight` returns a ``Path`` or ``None``; it
  never raises for an absent weight or a network error. The loud, raising variant is
  :func:`download_weight`, for a human running a provisioning step who wants the error.
- **Honesty.** A ``WeightSpec`` carries its license and reference. ``approx_bytes`` is the
  published/expected size so the gate can classify the lane honestly even while the
  weight is absent (a 65 MB detector is ``live-server`` whether or not it is downloaded
  yet). A URL that we could not verify at build time is marked in ``notes`` and may be
  overridden per weight via the ``BELTVISION_<NAME>_URL`` environment variable.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

_MB = 1024 * 1024


@dataclass(frozen=True)
class WeightSpec:
    """A single optional weight the ladder may use, and where it comes from."""

    name: str
    filename: str
    approx_bytes: int
    license: str  # noqa: A003 - "license" is the field's real name
    reference: str
    url: str | None = None
    sha256: str | None = None
    notes: str = ""

    def resolved_url(self) -> str | None:
        """The download URL, allowing a per-weight environment override.

        ``BELTVISION_MOBILE_SAM_URL`` overrides the ``mobile_sam`` weight, and so on, so
        a user can point at a verified mirror without editing code.
        """
        env_key = f"BELTVISION_{self.name.upper()}_URL"
        return os.environ.get(env_key) or self.url


# The optional weights the LIVE learned methods can consume. Only permissive licenses are
# registered for the shipped default; AGPL routes (Ultralytics FastSAM/YOLO) are reachable
# by dropping a file in the weights dir but are not auto-downloaded here.
WEIGHTS: dict[str, WeightSpec] = {
    "mobile_sam": WeightSpec(
        name="mobile_sam",
        filename="mobile_sam.pt",
        approx_bytes=40728226,
        license="Apache-2.0",
        reference="MobileSAM (Tiny-ViT 5M) https://github.com/ChaoningZhang/MobileSAM",
        url="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        sha256="6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
        notes=(
            "Apache-2.0 Tiny-ViT SAM encoder, ~40 MB, CPU-affordable automatic masks. "
            "URL + sha256 verified via httpx; opt-in fetch (download_weight/ensure_weight "
            "download=True) enables segmentation.mobile_sam to run for real."
        ),
    ),
    "detector_onnx": WeightSpec(
        name="detector_onnx",
        filename="detector.onnx",
        approx_bytes=65 * _MB,
        license="Apache-2.0 (RT-DETR arch); general/COCO-pretrained transfer",
        reference="RT-DETR arXiv:2304.08069 (Apache-2.0); export via [dl] ultralytics RTDETR",
        url=None,
        notes=(
            "No verified public ONNX URL is committed (honesty: an unverified URL is not "
            "printed as fact). Provide one via BELTVISION_DETECTOR_ONNX_URL, export locally "
            "with the [dl] extra (RTDETR(...).export(format='onnx')), or drop detector.onnx "
            "into the weights dir. Absent => graceful weights_absent."
        ),
    ),
    "conv_ae_onnx": WeightSpec(
        name="conv_ae_onnx",
        filename="conv_ae.onnx",
        approx_bytes=8 * _MB,
        license="trained-in-precompute (this repo)",
        reference="Conv-AE anomaly baseline, Bergmann et al. 2019 (MVTec AD, CVPR)",
        url=None,
        notes="Trained offline in the precompute lane then exported (ONNX INT8). Absent => weights_absent.",
    ),
    "belt_segmenter_onnx": WeightSpec(
        name="belt_segmenter_onnx",
        filename="belt_segmenter.onnx",
        approx_bytes=15 * _MB,
        license="trained-in-precompute (this repo); synthetic exact-GT + MobileSAM weak labels",
        reference=(
            "SegFormer-B0 (Xie et al. 2021, NeurIPS 'SegFormer'); nvidia/mit-b0 encoder "
            "(NVIDIA license, encoder-only transfer); MobileSAM Apache-2.0 weak belt labels. "
            "Real discharge/end-view training frames: Velenje coal mine (CC BY 3.0), Kieswerk "
            "Kronau gravel (CC0), aggregate sand discharge (CC BY 2.0), cubes-on-conveyor."
        ),
        url=None,
        notes=(
            "Trained belt/scene segmenter: 4 classes {external,belt,content,foreign}, 256x256 "
            "input, opset 17, CPU-affordable via onnxruntime. It is the PRIMARY scene segmenter "
            "in beltvision.methods.semantic when present; absent => graceful classical-prior "
            "fallback. Retrained on REAL CC-licensed discharge/end-view frames (the target COLA "
            "34 domain) weighted heavily over synthetic exact-GT, so it isolates the belt on "
            "real discharge frames where the colour prior grabbed rocks/mesh. The compact weight "
            "is committed in the Colia data repo under models/; point BELTVISION_WEIGHTS_DIR at "
            "that dir (or drop belt_segmenter.onnx into the weights dir) to activate it. No "
            "public URL (this-repo artifact)."
        ),
    ),
}


def weights_dir() -> Path:
    """The primary weights directory (env-overridable), created on demand."""
    override = os.environ.get("BELTVISION_WEIGHTS_DIR")
    root = Path(override) if override else (Path.home() / ".cache" / "beltvision" / "weights")
    root.mkdir(parents=True, exist_ok=True)
    return root


def search_dirs() -> list[Path]:
    """Directories searched for an existing weight, most-preferred first."""
    dirs = [weights_dir(), Path(__file__).resolve().parent / "weights"]
    seen: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    return seen


def _spec(name: str) -> WeightSpec:
    if name not in WEIGHTS:
        raise KeyError(f"unknown weight {name!r}; known: {sorted(WEIGHTS)}")
    return WEIGHTS[name]


def weight_path(name: str) -> Path:
    """The canonical (primary-dir) path for a weight, whether or not it exists."""
    return weights_dir() / _spec(name).filename


def _find_existing(name: str) -> Path | None:
    filename = _spec(name).filename
    for d in search_dirs():
        candidate = d / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def is_present(name: str) -> bool:
    """True if the weight is already on disk in any search directory."""
    return _find_existing(name) is not None


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_weight(name: str, *, timeout: float = 120.0, chunk: int = 1 << 16) -> Path:
    """Download a weight with httpx (loud: raises on any failure). Returns the path.

    For a human provisioning step. Methods use :func:`ensure_weight`, which wraps this
    and degrades to ``None`` on failure. Verifies ``sha256`` when the spec declares one.
    """
    spec = _spec(name)
    url = spec.resolved_url()
    if not url:
        raise RuntimeError(
            f"weight {name!r} has no download URL; set BELTVISION_{name.upper()}_URL, "
            f"drop {spec.filename} into {weights_dir()}, or export it in the [dl] lane. "
            f"{spec.notes}"
        )
    try:
        import httpx
    except ModuleNotFoundError as exc:  # httpx lives in the [dl] extra
        raise ModuleNotFoundError(
            "httpx is required to download weights; install the [dl] extra "
            "(pip install -e .[dl]) or drop the weight file in the weights dir"
        ) from exc

    dest = weights_dir() / spec.filename
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for block in resp.iter_bytes(chunk):
                fh.write(block)
    if spec.sha256:
        digest = _sha256(tmp)
        if digest.lower() != spec.sha256.lower():
            tmp.unlink(missing_ok=True)
            raise ValueError(f"sha256 mismatch for {name!r}: got {digest}, want {spec.sha256}")
    tmp.replace(dest)
    return dest


def ensure_weight(name: str, *, download: bool = False, timeout: float = 120.0) -> Path | None:
    """Return the local path to a weight, or ``None`` if it cannot be provided.

    Never raises for an absent weight or a failed/aborted download: this is the entry
    point the learned methods call so a missing weight becomes a graceful
    ``weights_absent`` result, never an exception. Pass ``download=True`` to opt in to a
    one-shot httpx fetch (still graceful: a network error returns ``None``).
    """
    existing = _find_existing(name)
    if existing is not None:
        return existing
    if not download:
        return None
    try:
        return download_weight(name, timeout=timeout)
    except Exception:  # noqa: BLE001 - opt-in download stays graceful for the method path
        return None
