"""Held-out, learned-vs-classical anomaly benchmark (research file 14).

Computes, on the leakage-safe held-out split, image-level AUROC + average precision for
each method, plus the two axes ADR-0015 requires alongside the headline number:

- **robustness**: re-score every held-out frame under a synthetic dust/haze perturbation
  and report the AUROC drop.
- **cost**: committed model bytes and measured CPU inference milliseconds per frame.

Everything is labeled a SMALL-SAMPLE PROXY with the exact N per class; nothing is
fabricated. If a learned method does not beat the classical baseline, the real number is
reported (the honest outcome).
"""
from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .dataset import Split
from .scorers import (
    add_dust,
    classical_residual_score,
    conv_ae_score,
    padim_lite_score,
    padim_score_from_features,
    patchcore_score_from_features,
)


def _downscale(bgr: np.ndarray, long_side: int) -> np.ndarray:
    import cv2

    h, w = bgr.shape[:2]
    if max(h, w) <= long_side:
        return bgr
    s = long_side / float(max(h, w))
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _auroc_ap(labels: np.ndarray, scores: list[float]) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.asarray(scores, dtype=np.float64)
    return float(roc_auc_score(labels, s)), float(average_precision_score(labels, s))


def _ms_stats(times: list[float]) -> dict[str, float]:
    a = np.asarray(times, dtype=np.float64) * 1000.0
    return {
        "cpu_ms_mean": round(float(a.mean()), 3) if a.size else 0.0,
        "cpu_ms_p95": round(float(np.percentile(a, 95)), 3) if a.size else 0.0,
    }


def compute_benchmark(
    split: Split,
    *,
    onnx_path: str,
    onnx_bytes: int,
    padim: dict[str, Any],
    padim_bytes: int,
    patchcore: dict[str, Any],
    patchcore_bytes: int,
    backbone: Any,
    seed: int = 34,
    work_long_side: int = 512,
    input_size: int = 256,
    dataset_meta: dict[str, Any] | None = None,
    conv_ae_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every method on the held-out split and assemble the benchmark JSON payload."""
    labels = split.labels
    heldout = split.heldout
    train_norms = [_downscale(f.bgr, work_long_side) for f in split.train_normal]
    clean = [_downscale(f.bgr, work_long_side) for f in heldout]
    dust = [add_dust(c, seed=seed + i) for i, c in enumerate(clean)]

    ids = ["classical_residual", "padim_lite", "conv_ae", "padim", "patchcore"]
    clean_scores: dict[str, list[float]] = {m: [] for m in ids}
    dust_scores: dict[str, list[float]] = {m: [] for m in ids}
    times: dict[str, list[float]] = {m: [] for m in ids}

    padim_kw = dict(means=padim["means"], inv_covs=padim["inv_covs"], sel_idx=padim["sel_idx"])
    coreset = patchcore["coreset"]

    def _score_all(frames: list[np.ndarray], store: dict[str, list[float]], timed: bool) -> None:
        for fr in frames:
            t0 = time.perf_counter()
            store["classical_residual"].append(classical_residual_score(fr))
            t1 = time.perf_counter()
            store["padim_lite"].append(padim_lite_score(fr, normal_images=train_norms))
            t2 = time.perf_counter()
            store["conv_ae"].append(conv_ae_score(fr, onnx_path=onnx_path, input_size=input_size))
            t3 = time.perf_counter()
            feat = backbone.extract([fr])[0]  # (P, D)
            t4 = time.perf_counter()
            store["padim"].append(padim_score_from_features(feat, **padim_kw))
            t5 = time.perf_counter()
            store["patchcore"].append(patchcore_score_from_features(feat, coreset=coreset))
            t6 = time.perf_counter()
            if timed:
                backbone_ms = t4 - t3
                times["classical_residual"].append(t1 - t0)
                times["padim_lite"].append(t2 - t1)
                times["conv_ae"].append(t3 - t2)
                times["padim"].append(backbone_ms + (t5 - t4))
                times["patchcore"].append(backbone_ms + (t6 - t5))

    _score_all(clean, clean_scores, timed=True)
    _score_all(dust, dust_scores, timed=False)

    specs = {
        "classical_residual": {
            "label": "Classical residual (Gaussian high-pass energy)",
            "family": "classical",
            "reference": "Blur-residual energy: training-free classical analogue of the conv-AE",
            "artifact": None,
            "model_bytes": 0,
            "lane": "live-server",
            "score_semantics": "mean |gray - Gaussian blur| (higher = more anomalous)",
        },
        "padim_lite": {
            "label": "PaDiM-lite (classical patch Gaussian)",
            "family": "learned-statistical",
            "reference": "PaDiM arXiv:2011.08785 (classical-feature variant; live CPU)",
            "artifact": None,
            "model_bytes": 0,
            "lane": "live-server",
            "score_semantics": "max Mahalanobis over 16px patches (fit on normal train set, in RAM)",
        },
        "conv_ae": {
            "label": "Conv-AE (L1 reconstruction)",
            "family": "learned",
            "reference": "Bergmann et al. 2019 (MVTec AD autoencoder baseline), CVPR 2019",
            "artifact": "models/conv_ae.onnx",
            "model_bytes": int(onnx_bytes),
            "lane": "live-web",
            "score_semantics": "mean absolute reconstruction residual (higher = more anomalous)",
        },
        "padim": {
            "label": "PaDiM (ResNet-18 backbone Gaussian)",
            "family": "learned",
            "reference": "PaDiM arXiv:2011.08785 (per-position Gaussian, Mahalanobis)",
            "artifact": "models/padim_ironore.npz",
            "model_bytes": int(padim_bytes),
            "lane": "live-server",
            "score_semantics": "max Mahalanobis over positions (frozen ResNet-18 layer2+3)",
        },
        "patchcore": {
            "label": "PatchCore-lite (coreset kNN)",
            "family": "learned",
            "reference": "PatchCore arXiv:2106.08265 (coreset memory + kNN; lite random coreset)",
            "artifact": "models/patchcore_ironore.npz",
            "model_bytes": int(patchcore_bytes),
            "lane": "live-server",
            "score_semantics": "max nearest-neighbour distance to coreset over positions",
        },
    }

    methods_out: list[dict[str, Any]] = []
    for mid in ids:
        auroc, ap = _auroc_ap(labels, clean_scores[mid])
        d_auroc, _ = _auroc_ap(labels, dust_scores[mid])
        spec = specs[mid]
        methods_out.append(
            {
                "id": mid,
                "label": spec["label"],
                "family": spec["family"],
                "reference": spec["reference"],
                "artifact": spec["artifact"],
                "declared_lane": spec["lane"],
                "score_semantics": spec["score_semantics"],
                "image_auroc": round(auroc, 4),
                "average_precision": round(ap, 4),
                "robustness": {
                    "perturbation": "synthetic dust/haze (atmospheric blend + blur + noise)",
                    "dust_image_auroc": round(d_auroc, 4),
                    "auroc_drop": round(auroc - d_auroc, 4),
                },
                "cost": {
                    "model_bytes": spec["model_bytes"],
                    **_ms_stats(times[mid]),
                    "work_long_side_px": int(work_long_side),
                },
            }
        )

    # Best (by held-out AUROC) among learned vs the classical baseline, reported honestly.
    classical = next(m for m in methods_out if m["family"] == "classical")
    learned = [m for m in methods_out if m["family"] != "classical"]
    best_learned = max(learned, key=lambda m: m["image_auroc"])
    ranking_note = (
        f"Best learned method on this held-out proxy: {best_learned['id']} "
        f"(AUROC {best_learned['image_auroc']:.3f}) vs classical baseline "
        f"{classical['id']} (AUROC {classical['image_auroc']:.3f}). "
        + (
            "Learned beats classical here."
            if best_learned["image_auroc"] >= classical["image_auroc"]
            else "Classical baseline beats the learned methods here (reported honestly)."
        )
    )

    counts = split.counts()
    base_rate = round(counts["n_pos"] / max(counts["n_pos"] + counts["n_neg"], 1), 4)
    payload: dict[str, Any] = {
        "schema": "colia-benchmark/1.0",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": int(seed),
        "small_sample_proxy": True,
        "task": "image-level unsupervised anomaly detection (normal vs foreign-object/anomaly)",
        "dataset": dataset_meta or {},
        "split": {
            "policy": (
                "normal-only training; all foreign-object/anomaly frames test-only "
                "(MVTec discipline); a subset of normal frames held out as negatives"
            ),
            "granularity": "frame-level (small-sample proxy; not sequence-level)",
            "leakage_free": True,
            **counts,
            "positive_base_rate": base_rate,
            "note": (
                "Frames were sampled spread across the source recording, which limits "
                "adjacent-frame near-duplicate leakage, but this remains a small-sample proxy."
            ),
        },
        "metric_defs": {
            "image_auroc": "Area under ROC, image-level (MVTec AD protocol; Bergmann et al. CVPR 2019)",
            "average_precision": "Area under precision-recall (sklearn average_precision_score)",
            "auroc_drop": "image_auroc(clean) - image_auroc(dust-perturbed); robustness axis",
            "cpu_ms": "Measured wall-clock per-frame scoring on this CPU (includes backbone for PaDiM/PatchCore)",
        },
        "conv_ae_training": conv_ae_meta or {},
        "methods": methods_out,
        "ranking_note": ranking_note,
        "external_context": {
            "note": "External published numbers on DIFFERENT data; framing only, NOT Colia's results.",
            "patchcore_mvtec_ad": "99.1% image / 98.1% pixel AUROC / 93.5% PRO (arXiv:2106.08265)",
            "padim_mvtec_ad": "~0.962 image AUROC WideResNet50 (arXiv:2011.08785)",
            "anomalydino_mvtec_ad": "one-shot 93.1% -> 96.6% image AUROC (arXiv:2405.14529)",
        },
        "caveats": [
            "SMALL-SAMPLE PROXY: a few dozen frames per class; AUROC has wide confidence intervals.",
            f"Average precision is inflated by the {base_rate:.0%} positive base rate of this held-out "
            "set (AP random baseline = prevalence); image AUROC is the prevalence-independent headline.",
            "Frame-level split on a single in-lab recording; not a multi-belt generalization test.",
            "PaDiM/PatchCore fitted on a frozen ImageNet ResNet-18 (not belt-pretrained); banks are compact.",
            "PaDiM assumes spatial alignment; belt content flows, so localization is approximate.",
            "Numbers are reproducible from (seed, committed artifacts) via python -m beltvision.precompute.",
        ],
    }
    return payload
