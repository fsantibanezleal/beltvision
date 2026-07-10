"""Precompute-lane tests: leakage-safe split, conv-AE ONNX export, benchmark compute.

The split/leakage checks are pure classical (numpy + opencv) and always run. The training
export and benchmark smoke need the heavy extra (torch / onnx / onnxruntime / scikit-learn);
they ``importorskip`` and additionally verify the torch<->numpy bridge is usable, so a slim
runtime venv (numpy 2.x, no working torch bridge) skips them cleanly rather than erroring.

The benchmark smoke injects a tiny stub backbone, so it does NOT need the ResNet-18 download:
it exercises the real conv-AE ONNX scorer, the classical + padim_lite live methods, the PaDiM
/ PatchCore fitters, and the AUROC/AP/robustness/cost assembly on a leakage-safe tiny split.
"""
from __future__ import annotations

import numpy as np
import pytest

from beltvision.precompute.dataset import build_split


def _write_scene(path, seed: int, kind: str) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    img = rng.integers(40, 90, size=(48, 64, 3), dtype=np.uint8)
    if kind == "anomalous":
        # paint a bright foreign blob so anomalous frames are separable from normal texture
        cv2.rectangle(img, (20, 12), (40, 32), (240, 240, 240), -1)
    cv2.imwrite(str(path), img)


def _make_dataset(root, n_normal: int = 6, n_anom: int = 4):
    (root / "normal").mkdir(parents=True)
    (root / "foreign_object_wood").mkdir(parents=True)
    (root / "anomaly").mkdir(parents=True)
    for i in range(n_normal):
        _write_scene(root / "normal" / f"n{i}.jpg", 100 + i, "normal")
    half = n_anom // 2
    for i in range(half):
        _write_scene(root / "foreign_object_wood" / f"w{i}.jpg", 300 + i, "anomalous")
    for i in range(n_anom - half):
        _write_scene(root / "anomaly" / f"a{i}.jpg", 500 + i, "anomalous")
    return root


def _torch_usable() -> bool:
    try:
        import torch

        torch.from_numpy(np.zeros((1,), dtype=np.float32))
        return True
    except Exception:
        return False


# --- always-on: leakage-safe split (classical) --------------------------------------------

def test_split_is_leakage_free_and_labeled(tmp_path):
    _make_dataset(tmp_path, n_normal=6, n_anom=4)
    split = build_split(tmp_path, seed=34, train_frac=0.5)
    split.assert_leakage_free()
    c = split.counts()
    assert c["train_normal"] >= 1
    assert c["heldout_normal"] >= 1
    assert c["heldout_anomalous"] == 4
    # labels: normal negatives first (0), anomalous positives (1); both classes present
    labels = split.labels
    assert set(labels.tolist()) == {0, 1}
    assert labels.sum() == 4


def test_split_no_frame_in_train_and_test(tmp_path):
    _make_dataset(tmp_path, n_normal=8, n_anom=4)
    split = build_split(tmp_path, seed=7, train_frac=0.7)
    train = {f.path.name for f in split.train_normal}
    held = {f.path.name for f in split.heldout}
    assert train.isdisjoint(held)


def test_split_requires_normal_frames(tmp_path):
    (tmp_path / "anomaly").mkdir(parents=True)
    _write_scene(tmp_path / "anomaly" / "a.jpg", 1, "anomalous")
    with pytest.raises(FileNotFoundError):
        build_split(tmp_path, seed=1)


# --- heavy: conv-AE train -> ONNX export ---------------------------------------------------

@pytest.mark.skipif(not _torch_usable(), reason="torch/numpy bridge unavailable (slim venv)")
def test_conv_ae_trains_and_exports_onnx(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from beltvision.methods.anomaly import conv_ae as conv_ae_method
    from beltvision.precompute.train import train_conv_ae

    root = _make_dataset(tmp_path / "data", n_normal=4, n_anom=2)
    split = build_split(root, seed=34, train_frac=0.75)
    onnx_out = tmp_path / "models" / "conv_ae.onnx"
    meta = train_conv_ae(
        split.train_normal, onnx_out=onnx_out, input_size=32, epochs=2, seed=34
    )
    assert onnx_out.is_file()
    assert 0 < meta["bytes"] < 20 * 1024 * 1024
    assert meta["opset"] == 17
    assert meta["onnx_torch_max_abs_diff"] is not None
    assert meta["onnx_torch_max_abs_diff"] < 1e-2

    # the live method now returns REAL output (status ok) with this weight present
    res = conv_ae_method(split.heldout[0].bgr, weights=str(onnx_out), input_size=32)
    assert res["status"] == "ok"
    assert "image_score" in res and res["backend"].startswith("onnxruntime")


# --- heavy: benchmark compute on a tiny stub-backbone split --------------------------------

class _StubBackbone:
    """Deterministic tiny per-position features (no torchvision / no download)."""

    grid = 2
    feature_dim = 8

    def extract(self, frames):
        import cv2

        out = []
        for f in frames:
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (2, 2)).astype(np.float32)
            g = g.reshape(4) / 255.0
            feat = np.stack(
                [np.array([v, v * v, np.sin(v), np.cos(v), 0.5 * v, v + 1, v - 1, v * 2]) for v in g]
            )
            out.append(feat)
        return np.stack(out).astype(np.float32)  # (n, 4, 8)


@pytest.mark.skipif(not _torch_usable(), reason="torch/numpy bridge unavailable (slim venv)")
def test_benchmark_compute_smoke(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("sklearn")
    from beltvision.precompute.benchmark import compute_benchmark
    from beltvision.precompute.train import fit_padim, fit_patchcore, train_conv_ae

    root = _make_dataset(tmp_path / "data", n_normal=6, n_anom=4)
    split = build_split(root, seed=34, train_frac=0.5)
    onnx_out = tmp_path / "models" / "conv_ae.onnx"
    meta = train_conv_ae(split.train_normal, onnx_out=onnx_out, input_size=32, epochs=2, seed=34)

    backbone = _StubBackbone()
    train_feats = backbone.extract([f.bgr for f in split.train_normal])
    padim = fit_padim(train_feats, sel_dims=6, seed=34)
    patchcore = fit_patchcore(train_feats, coreset_size=8, seed=34)

    payload = compute_benchmark(
        split,
        onnx_path=str(onnx_out),
        onnx_bytes=int(meta["bytes"]),
        padim=padim,
        padim_bytes=1234,
        patchcore=patchcore,
        patchcore_bytes=567,
        backbone=backbone,
        seed=34,
        work_long_side=64,
        input_size=32,
        dataset_meta={"name": "unit-test"},
    )
    assert payload["small_sample_proxy"] is True
    assert payload["split"]["n_pos"] == 4
    assert payload["split"]["n_neg"] >= 1
    ids = {m["id"] for m in payload["methods"]}
    assert {"classical_residual", "padim_lite", "conv_ae", "padim", "patchcore"} == ids
    for m in payload["methods"]:
        assert 0.0 <= m["image_auroc"] <= 1.0
        assert 0.0 <= m["average_precision"] <= 1.0
        assert "auroc_drop" in m["robustness"]
        assert "cpu_ms_mean" in m["cost"]
