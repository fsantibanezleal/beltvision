"""Per-method image-level anomaly scorers used by the held-out benchmark.

Each scorer returns a scalar where a HIGHER value means MORE anomalous, so
``sklearn.metrics.roc_auc_score`` reads them consistently (positive class = anomalous).

The learned scorers deliberately reuse the exact live method bodies where possible:
``conv_ae`` and ``padim_lite`` are scored through ``beltvision.methods.anomaly`` so the
benchmark number equals what the live ``/api/analyze`` endpoint returns. PaDiM and
PatchCore-lite score from the frozen-backbone features fitted in ``train.py``. The
classical baseline is a training-free blur-residual energy (the classical analogue of the
conv-AE reconstruction residual).
"""
from __future__ import annotations

import numpy as np

from ..methods.anomaly import conv_ae as _conv_ae_method
from ..methods.anomaly import padim_lite as _padim_lite_method
from ..methods.preprocess import apply_clahe_lab

# --- learned: conv-AE + padim_lite via the real live method bodies -------------------------

def conv_ae_score(bgr: np.ndarray, *, onnx_path: str, input_size: int = 256) -> float:
    """Mean absolute reconstruction residual from the trained conv-AE ONNX (live method)."""
    res = _conv_ae_method(bgr, weights=onnx_path, input_size=input_size)
    if res.get("status") != "ok":
        raise RuntimeError(f"conv_ae did not run: {res.get('status')} / {res.get('hint')}")
    return float(res["image_score"])


def padim_lite_score(bgr: np.ndarray, *, normal_images: list[np.ndarray]) -> float:
    """Classical-feature PaDiM (live method) Mahalanobis image score, fit on train normals."""
    res = _padim_lite_method(bgr, normal_images=normal_images)
    return float(res["image_score"])


# --- learned: PaDiM + PatchCore-lite from frozen-backbone features -------------------------

def padim_score_from_features(
    feat_pd: np.ndarray, *, means: np.ndarray, inv_covs: np.ndarray, sel_idx: np.ndarray
) -> float:
    """Max Mahalanobis distance over positions for one frame's features (P, D)."""
    f = feat_pd[:, sel_idx].astype(np.float64)  # (P, d_sel)
    centered = f - means.astype(np.float64)  # (P, d_sel)
    # distance^2 per position: sum((c @ inv) * c, axis=1)
    tmp = np.einsum("pij,pj->pi", inv_covs.astype(np.float64), centered)
    d2 = np.einsum("pi,pi->p", centered, tmp).clip(min=0.0)
    return float(np.sqrt(d2).max())


def patchcore_score_from_features(feat_pd: np.ndarray, *, coreset: np.ndarray) -> float:
    """Max over positions of the nearest-neighbour L2 distance to the coreset (P, D)."""
    q = feat_pd.astype(np.float32)  # (P, D)
    c = coreset.astype(np.float32)  # (M, D)
    q2 = (q * q).sum(axis=1)[:, None]  # (P, 1)
    c2 = (c * c).sum(axis=1)[None, :]  # (1, M)
    d2 = q2 + c2 - 2.0 * (q @ c.T)  # (P, M)
    nn = np.sqrt(np.clip(d2.min(axis=1), 0.0, None))  # (P,)
    return float(nn.max())


# --- classical baseline: training-free blur-residual energy --------------------------------

def classical_residual_score(bgr: np.ndarray) -> float:
    """Mean absolute high-pass (gray - Gaussian blur) residual: the classical AE analogue."""
    import cv2

    gray = cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=8.0)
    residual = np.abs(gray - blur)
    return float(residual.mean())


# --- robustness perturbation: synthetic dust / haze ----------------------------------------

def add_dust(bgr: np.ndarray, *, severity: float = 0.45, seed: int = 0) -> np.ndarray:
    """Apply a deterministic atmospheric dust/haze perturbation to a BGR frame.

    Combines an atmospheric-scattering haze blend (I*t + A*(1-t)), a mild blur, and additive
    Gaussian sensor noise. Used to measure each method's robustness drop (research file 14
    section 4): the same frame is re-scored under dust and the AUROC delta reported.
    """
    import cv2

    rng = np.random.default_rng(int(seed))
    img = bgr.astype(np.float32)
    t = float(np.clip(1.0 - severity, 0.05, 1.0))  # transmission
    a = 205.0  # airlight (dusty grey-white)
    hazy = img * t + a * (1.0 - t)
    hazy = cv2.GaussianBlur(hazy, (0, 0), sigmaX=1.0 + 2.0 * severity)
    noise = rng.normal(0.0, 6.0 * severity, size=img.shape).astype(np.float32)
    out = np.clip(hazy + noise, 0, 255).astype(np.uint8)
    return out
