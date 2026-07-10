"""Stage 3: train.

Fits a tiny, deterministic "normal model": the per-feature mean and variance of the
patch features from stage 2. Anomaly scoring in stage 4 is the Mahalanobis-style
distance of a patch from this normal distribution. This is the same shape as the
learned anomaly baselines (a conv autoencoder's reconstruction error, PaDiM's
per-patch Gaussian), just at classical cost so it runs live and offline in a test.

Rework surface: the real learned anomaly methods (conv autoencoder L1, EfficientAD
L2, PaDiM, PatchCore, AnomalyDINO) are trained here in the precompute lane and
exported to ONNX in stage 6. The frozen part is that training is deterministic given
``(features, seed)`` and produces a compact, serializable model.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..context import StageContext
from ..core.trace import stage_timer

EPS = 1e-6


def train(ctx: StageContext) -> dict[str, Any]:
    with stage_timer(ctx.trace, "train") as t:
        feats: np.ndarray = ctx.state["patch_features"]
        # A run is a pure function of (features, seed); the rng is drawn so the seed
        # is genuinely threaded even though this estimator is closed-form.
        _ = ctx.rng("train")
        mean = feats.mean(axis=0)
        var = feats.var(axis=0) + EPS

        model = {"mean": mean.astype(np.float32), "var": var.astype(np.float32)}
        ctx.state["normal_model"] = model
        # Model "bytes" = the serialized parameter array size (compact by design).
        model_bytes = int(mean.nbytes + var.nbytes)
        ctx.state["normal_model_bytes"] = model_bytes

        summary = {
            "estimator": "patch-gaussian",
            "feature_dim": int(mean.shape[0]),
            "model_bytes": model_bytes,
            "n_train_patches": int(feats.shape[0]),
        }
        ctx.state["train_summary"] = summary
        t.note(model_bytes=model_bytes, feature_dim=int(mean.shape[0]))
    return summary
