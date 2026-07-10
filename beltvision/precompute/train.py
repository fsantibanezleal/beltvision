"""Train / fit the belt-specific learned anomaly models (precompute lane).

Three artifacts are produced from NORMAL frames only:

- **conv-AE (L1)** - the mandated convolutional-autoencoder baseline (Bergmann et al. 2019).
  Trained to reconstruct CLAHE grayscale belt frames with an L1 loss, then exported to ONNX
  (opset 17). Anomaly score at serve time is the mean absolute reconstruction residual, run
  live by ``beltvision.methods.anomaly.conv_ae`` via onnxruntime.
- **PaDiM** (arXiv:2011.08785) - a per-position multivariate Gaussian over frozen ResNet-18
  layer2+layer3 features; anomaly score is the max Mahalanobis distance over positions.
- **PatchCore-lite** (arXiv:2106.08265) - a subsampled coreset memory bank of normal patch
  features; anomaly score is the max over query patches of the nearest-neighbour distance.
  This is a "lite" random-coreset CPU approximation (labeled as such), not the full greedy
  coreset + FAISS build.

Everything is deterministic given the seed. Torch / onnx are imported lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..methods.anomaly import build_conv_ae, conv_ae_architecture
from ..methods.preprocess import apply_clahe_lab
from .dataset import Frame


def _seed_everything(seed: int) -> None:
    import torch

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _ae_tensor(frames: list[Frame], input_size: int) -> np.ndarray:
    """Stack CLAHE grayscale frames into an (n, 1, S, S) float32 [0, 1] array."""
    import cv2

    xs = []
    for fr in frames:
        gray = cv2.cvtColor(apply_clahe_lab(fr.bgr), cv2.COLOR_BGR2GRAY)
        r = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
        xs.append(r.astype(np.float32) / 255.0)
    arr = np.stack(xs, axis=0)[:, None, :, :]
    return np.ascontiguousarray(arr)


def train_conv_ae(
    train_normal: list[Frame],
    *,
    onnx_out: Path,
    input_size: int = 256,
    latent_dim: int = 128,
    epochs: int = 120,
    lr: float = 1e-3,
    seed: int = 34,
) -> dict[str, Any]:
    """Train the conv-AE on normal frames and export it to ONNX (opset 17). Returns metadata."""
    import torch
    from torch import nn

    _seed_everything(seed)
    model = build_conv_ae(latent_dim)
    model.train()

    x = torch.from_numpy(_ae_tensor(train_normal, input_size))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    n = x.shape[0]
    batch = min(8, n)
    g = torch.Generator().manual_seed(int(seed))
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = x[idx]
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(idx)
        losses.append(epoch_loss / n)

    model.eval()
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 1, input_size, input_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        opset_version=17,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes=None,
    )

    parity = _onnx_parity(model, onnx_out, input_size)
    return {
        "artifact": str(onnx_out),
        "bytes": int(onnx_out.stat().st_size),
        "opset": 17,
        "input_shape": [1, 1, input_size, input_size],
        "latent_dim": int(latent_dim),
        "epochs": int(epochs),
        "n_train": int(n),
        "final_l1_loss": round(float(losses[-1]), 6) if losses else None,
        "onnx_torch_max_abs_diff": parity,
        "architecture": conv_ae_architecture(latent_dim=latent_dim, input_size=input_size),
        "reference": "Bergmann et al. 2019 (MVTec AD autoencoder baseline), CVPR 2019",
    }


def _onnx_parity(model, onnx_path: Path, input_size: int) -> float | None:
    """Max abs diff between torch and onnxruntime on a fixed probe (ONNX export sanity)."""
    try:
        import onnxruntime as ort
        import torch
    except Exception:
        return None
    probe = np.random.default_rng(0).random((1, 1, input_size, input_size)).astype(np.float32)
    with torch.no_grad():
        t_out = model(torch.from_numpy(probe)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_out = sess.run(None, {sess.get_inputs()[0].name: probe})[0]
    return round(float(np.max(np.abs(t_out - o_out))), 8)


# --- PaDiM ---------------------------------------------------------------------------------

def fit_padim(
    features: np.ndarray, *, sel_dims: int = 100, seed: int = 34
) -> dict[str, Any]:
    """Fit a per-position Gaussian over ``features`` (n, P, D). Returns a compact bank dict.

    ``features`` are the frozen-backbone patch features of the NORMAL training frames.
    A random subset of ``sel_dims`` channels is selected (seeded, PaDiM's dimensionality
    reduction); per position the mean and a regularized inverse covariance are stored.
    """
    n, p, d = features.shape
    rng = np.random.default_rng(int(seed))
    sel = np.sort(rng.choice(d, size=min(sel_dims, d), replace=False)).astype(np.int32)
    feats = features[:, :, sel]  # (n, P, d_sel)
    d_sel = feats.shape[2]

    means = np.zeros((p, d_sel), dtype=np.float32)
    inv_covs = np.zeros((p, d_sel, d_sel), dtype=np.float32)
    eye = np.eye(d_sel, dtype=np.float64)
    for pos in range(p):
        block = feats[:, pos, :].astype(np.float64)  # (n, d_sel)
        mu = block.mean(axis=0)
        cov = np.cov(block, rowvar=False)
        cov = np.atleast_2d(cov)
        reg = 0.01 * (np.trace(cov) / max(d_sel, 1)) + 1e-6
        inv = np.linalg.inv(cov + reg * eye)
        means[pos] = mu.astype(np.float32)
        inv_covs[pos] = inv.astype(np.float32)

    return {
        "kind": "padim",
        "means": means,
        "inv_covs": inv_covs,
        "sel_idx": sel,
        "n_train": int(n),
        "n_positions": int(p),
        "feature_dim": int(d),
        "sel_dims": int(d_sel),
        "reference": "PaDiM arXiv:2011.08785 (per-position Gaussian, Mahalanobis)",
    }


def save_padim(bank: dict[str, Any], path: Path, *, grid: int, input_size: int) -> int:
    """Write the PaDiM bank as a compressed npz. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        means=bank["means"],
        inv_covs=bank["inv_covs"],
        sel_idx=bank["sel_idx"],
        grid=np.int32(grid),
        input_size=np.int32(input_size),
        n_train=np.int32(bank["n_train"]),
    )
    return int(path.stat().st_size)


# --- PatchCore-lite ------------------------------------------------------------------------

def fit_patchcore(
    features: np.ndarray, *, coreset_size: int = 1024, seed: int = 34
) -> dict[str, Any]:
    """Build a random-coreset memory bank from normal patch features (n, P, D)."""
    n, p, d = features.shape
    flat = features.reshape(n * p, d).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    m = min(int(coreset_size), flat.shape[0])
    idx = rng.choice(flat.shape[0], size=m, replace=False)
    coreset = flat[np.sort(idx)]
    return {
        "kind": "patchcore_lite",
        "coreset": coreset,
        "n_train": int(n),
        "n_positions": int(p),
        "feature_dim": int(d),
        "coreset_size": int(m),
        "reference": "PatchCore arXiv:2106.08265 (coreset memory + kNN); lite random coreset",
    }


def save_patchcore(bank: dict[str, Any], path: Path, *, grid: int, input_size: int) -> int:
    """Write the PatchCore-lite coreset as a compressed npz. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coreset=bank["coreset"],
        grid=np.int32(grid),
        input_size=np.int32(input_size),
        n_train=np.int32(bank["n_train"]),
    )
    return int(path.stat().st_size)
