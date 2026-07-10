"""Capability 2b: unsupervised anomaly detection (>= 1 learned, LIVE).

- ``anomaly.padim_lite`` (M9) - a PaDiM-style per-patch multivariate Gaussian with a
  Mahalanobis residual heatmap. Uses classical patch features (mean/std/gradient/Laplacian)
  so it runs LIVE on CPU with no downloaded weight, fitting the "normal" distribution from a
  supplied normal-image set or, absent one, from the frame's own patch statistics. This is a
  genuine learned/statistical method (it fits a covariance model), just at classical cost.
- ``anomaly.conv_ae`` (L1) - the mandated convolutional-autoencoder anomaly baseline. This
  module ships the architecture descriptor and a real ONNX/torch inference path; the model
  is TRAINED offline in the precompute lane, so absent a weight the method returns a graceful
  ``weights_absent`` (never an exception).

References: PaDiM arXiv:2011.08785; Bergmann et al. 2019 (MVTec AD, autoencoder + SSIM
baseline), CVPR 2019.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models import ensure_weight, search_dirs, weight_path
from ._common import as_bgr, result, timed, weights_absent
from .preprocess import apply_clahe_lab

_PATCH = 16
_MAX_GRID_CELLS = 1024


def _patch_features(gray: np.ndarray, patch: int) -> tuple[np.ndarray, int, int]:
    """Return an (rows*cols, d) feature matrix plus the grid shape."""
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    h, w = gray.shape
    rows = len(range(0, h - patch + 1, patch))
    cols = len(range(0, w - patch + 1, patch))
    feats = []
    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            g = gray[y : y + patch, x : x + patch].astype(np.float32)
            gm = grad[y : y + patch, x : x + patch]
            lp = lap[y : y + patch, x : x + patch]
            feats.append([g.mean(), g.std(), gm.mean(), gm.std(), float(lp.var())])
    mat = np.asarray(feats, dtype=np.float64) if feats else np.zeros((1, 5), np.float64)
    return mat, rows, cols


def _fit_gaussian(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and (shrinkage-regularized) inverse covariance for Mahalanobis."""
    mean = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    cov = np.atleast_2d(cov)
    reg = 1e-3 * np.trace(cov) / max(cov.shape[0], 1)
    cov_inv = np.linalg.pinv(cov + reg * np.eye(cov.shape[0]))
    return mean, cov_inv


def padim_lite(
    image: Any, *, patch: int = _PATCH, normal_images: list[Any] | None = None, **_: Any
) -> dict[str, Any]:
    """Per-patch Gaussian + Mahalanobis anomaly heatmap (learned/statistical, live CPU)."""
    import cv2

    gray = cv2.cvtColor(apply_clahe_lab(as_bgr(image)), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    with timed() as t:
        if normal_images:
            train = np.concatenate(
                [
                    _patch_features(cv2.cvtColor(apply_clahe_lab(as_bgr(im)), cv2.COLOR_BGR2GRAY), patch)[0]
                    for im in normal_images
                ],
                axis=0,
            )
            fit_source = f"normal-set({len(normal_images)})"
        else:
            train, _, _ = _patch_features(gray, patch)
            fit_source = "self-reference"
        mean, cov_inv = _fit_gaussian(train)

        query, rows, cols = _patch_features(gray, patch)
        centered = query - mean
        maha = np.sqrt(np.einsum("ij,jk,ik->i", centered, cov_inv, centered).clip(min=0.0))
        cells = rows * cols
        grid = maha[:cells].reshape(rows, cols) if cells else maha.reshape(1, -1)
        gmax = float(grid.max()) if grid.size else 0.0
        norm = (grid / gmax) if gmax > 0 else grid
        peak = int(np.argmax(norm)) if norm.size else 0
        peak_rc = (peak // norm.shape[1], peak % norm.shape[1])
        image_score = float(np.percentile(maha, 99)) if maha.size else 0.0

    heatmap = norm.round(4).tolist() if norm.size <= _MAX_GRID_CELLS else None
    model_bytes = int(mean.nbytes + cov_inv.nbytes)
    payload = {
        "shape": [int(h), int(w)],
        "fit_source": fit_source,
        "feature_dim": int(train.shape[1]),
        "grid_rows": int(norm.shape[0]),
        "grid_cols": int(norm.shape[1]),
        "n_patches": int(query.shape[0]),
        "image_score": round(image_score, 4),
        "max_patch_score": round(gmax, 4),
        "peak_row": int(peak_rc[0]),
        "peak_col": int(peak_rc[1]),
        "residual_heatmap": heatmap,
    }
    return result(
        "anomaly.padim_lite", "anomaly", "learned",
        "PaDiM arXiv:2011.08785 (per-patch Gaussian, Mahalanobis)",
        payload=payload, model_bytes=model_bytes, infer_ms=t.ms, web_drivable=False,
    )


def conv_ae_architecture(latent_dim: int = 128, input_size: int = 256) -> dict[str, Any]:
    """JSON-safe descriptor of the conv-AE the precompute lane trains and exports."""
    return {
        "family": "convolutional-autoencoder",
        "input": [1, input_size, input_size],
        "encoder": [
            {"conv": [1, 32], "stride": 2},
            {"conv": [32, 64], "stride": 2},
            {"conv": [64, 128], "stride": 2},
            {"conv": [128, latent_dim], "stride": 2},
        ],
        "decoder": [
            {"deconv": [latent_dim, 128], "stride": 2},
            {"deconv": [128, 64], "stride": 2},
            {"deconv": [64, 32], "stride": 2},
            {"deconv": [32, 1], "stride": 2},
        ],
        "loss": "L1 reconstruction (+ optional SSIM)",
        "anomaly_score": "per-pixel |input - reconstruction| residual heatmap",
        "latent_dim": int(latent_dim),
    }


def build_conv_ae(latent_dim: int = 128):
    """Build the conv-AE as a torch ``nn.Module`` (precompute lane; lazy ``[dl]`` import)."""
    import torch
    from torch import nn

    class ConvAE(nn.Module):
        def __init__(self, latent: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(True),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
                nn.Conv2d(128, latent, 3, 2, 1), nn.ReLU(True),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent, 128, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    return ConvAE(int(latent_dim))


def _preprocess_ae(gray: np.ndarray, size: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return resized[None, None, :, :]  # NCHW, 1 channel


def conv_ae(
    image: Any, *, weights: str | None = None, input_size: int = 256, download: bool = False, **_: Any
) -> dict[str, Any]:
    """Conv-AE reconstruction-error anomaly (learned, L1). Graceful weights_absent."""
    import cv2

    ref = "Bergmann et al. 2019 (MVTec AD autoencoder baseline), CVPR 2019"
    from ..models.download import WEIGHTS

    spec = WEIGHTS["conv_ae_onnx"]
    arch = conv_ae_architecture(input_size=input_size)

    onnx_path = weights if weights else ensure_weight("conv_ae_onnx", download=download)
    pt_path = None if weights else _find_local("conv_ae.pt")
    searched = [weights] if weights else search_dirs()
    if not (onnx_path and _exists(onnx_path)) and not (pt_path and _exists(pt_path)):
        res = weights_absent(
            "anomaly.conv_ae", "anomaly", "learned", ref,
            weight="conv_ae.onnx | conv_ae.pt", searched=searched, approx_bytes=spec.approx_bytes,
            web_drivable=True,
            hint=(
                f"train the conv-AE offline (precompute lane), export to {weight_path('conv_ae_onnx')} "
                "(ONNX INT8, live-web) or a .pt (live-server); this method then runs real inference."
            ),
        )
        res["architecture"] = arch
        return res

    gray = cv2.cvtColor(apply_clahe_lab(as_bgr(image)), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x = _preprocess_ae(gray, input_size)
    try:
        with timed() as t:
            if onnx_path and _exists(onnx_path):
                recon, backend, size = _run_ae_onnx(x, str(onnx_path))
            else:
                recon, backend, size = _run_ae_torch(x, str(pt_path), input_size)
            residual = np.abs(x - recon)[0, 0]
            image_score = float(residual.mean())
            peak = np.unravel_index(int(np.argmax(residual)), residual.shape)
    except Exception as exc:  # noqa: BLE001
        res = weights_absent(
            "anomaly.conv_ae", "anomaly", "learned", ref,
            weight="conv_ae", searched=searched, approx_bytes=spec.approx_bytes, web_drivable=True,
            hint=f"weight present but inference failed: {type(exc).__name__}: {exc}",
        )
        res["architecture"] = arch
        return res

    payload = {
        "shape": [int(h), int(w)],
        "backend": backend,
        "input_size": int(input_size),
        "image_score": round(image_score, 6),
        "max_residual": round(float(residual.max()), 6),
        "peak_row_frac": round(float(peak[0]) / input_size, 4),
        "peak_col_frac": round(float(peak[1]) / input_size, 4),
        "architecture": arch,
    }
    return result(
        "anomaly.conv_ae", "anomaly", "learned", ref,
        payload=payload, model_bytes=size, infer_ms=t.ms, web_drivable=True,
    )


def _run_ae_onnx(x: np.ndarray, path: str) -> tuple[np.ndarray, str, int]:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x.astype(np.float32)})[0]
    return np.asarray(out), "onnxruntime-cpu", _filesize(path)


def _run_ae_torch(x: np.ndarray, path: str, input_size: int) -> tuple[np.ndarray, str, int]:
    import torch

    state = torch.load(path, map_location="cpu")
    model = build_conv_ae()
    model.load_state_dict(state if isinstance(state, dict) and "encoder.0.weight" in state else state["model"])
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x.astype(np.float32))).numpy()
    return out, "torch-cpu", _filesize(path)


def _find_local(filename: str) -> Any:

    for d in search_dirs():
        p = d / filename
        if p.is_file():
            return p
    return None


def _exists(path: Any) -> bool:
    from pathlib import Path

    return path is not None and Path(str(path)).is_file()


def _filesize(path: Any) -> int:
    from pathlib import Path

    p = Path(str(path))
    return p.stat().st_size if p.is_file() else 0
