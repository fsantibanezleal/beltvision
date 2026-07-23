#!/usr/bin/env python3
"""Run beltvision's classical geometry+semantic chain on the deterministic synthetic
ground-truth suite and write the measured accuracy to ../data/bv.json.

The synthetic scenes (beltvision.cases.synthetic) are generated, so the exact belt mask,
orientation, centreline, injected misalignment and 4-class label map are KNOWN. This runner
measures how well the training-free classical chain (CLAHE -> semantic layers -> PCA axis ->
centreline -> alignment) RECOVERS that known geometry, at vertical / horizontal / diagonal /
curved / misaligned belts. Numbers are exact and reproducible from (seed, code); no data is
downloaded and nothing is fabricated. use_learned=False throughout (classical core only).

Run:  python run_bench.py      (with beltvision + core deps installed)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from beltvision.cases.synthetic import (
    BELT, CONTENT, EXTERNAL, FOREIGN, CLASS_NAMES, GT_SUITE, gt_scene,
)
from beltvision.methods.beltline import _ang_diff, compute_belt_geometry
from beltvision.methods.preprocess import apply_clahe_lab
from beltvision.methods.semantic import compute_layers

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def run_scene(name: str) -> dict:
    sc = gt_scene(name)
    view = "top_carrying" if sc.loaded else "end_return"
    t0 = time.perf_counter()
    layers = compute_layers(sc.image, view_type=view, use_learned=False)
    gray = cv2.cvtColor(apply_clahe_lab(sc.image), cv2.COLOR_BGR2GRAY)
    footprint = layers.belt_mask | layers.content_mask
    geo = compute_belt_geometry(footprint, external_mask=layers.mask(EXTERNAL), gray=gray)
    ms = (time.perf_counter() - t0) * 1000.0

    belt_iou = iou(footprint, sc.belt_mask)
    ori_err = None
    if "axis_angle_deg" in geo:
        ori_err = round(float(_ang_diff(geo["axis_angle_deg"], sc.orientation_deg)), 2)

    class_iou = {}
    for c in (EXTERNAL, BELT, CONTENT, FOREIGN):
        gt_c = sc.label_map == c
        if int(gt_c.sum()) > 0:
            class_iou[CLASS_NAMES[c]] = round(iou(layers.label_map == c, gt_c), 3)

    mis_det = geo.get("misalignment_deg")
    mis_err = None if mis_det is None else round(abs(float(mis_det) - sc.misalignment_deg), 2)

    cl_rmse = None
    if abs(sc.curvature) < 1e-9 and "centreline_xy" in geo:
        cl = np.asarray(geo["centreline_xy"], float)
        gt = np.asarray(sc.centreline, float)
        d = [float(np.min(np.hypot(gt[:, 0] - p[0], gt[:, 1] - p[1]))) for p in cl]
        if d:
            cl_rmse = round(float(np.sqrt(np.mean(np.square(d)))), 2)

    return {
        "name": name,
        "orientation_gt_deg": round(float(sc.orientation_deg), 1),
        "loaded": bool(sc.loaded),
        "curved": bool(abs(sc.curvature) > 1e-9),
        "axis_angle_deg": geo.get("axis_angle_deg"),
        "orientation_label": geo.get("orientation_label"),
        "confidence": geo.get("confidence"),
        "belt_iou": round(belt_iou, 3),
        "ori_err_deg": ori_err,
        "class_iou": class_iou,
        "misalign_gt_deg": round(float(sc.misalignment_deg), 2),
        "misalign_det_deg": mis_det,
        "misalign_err_deg": mis_err,
        "centreline_rmse_px": cl_rmse,
        "belt_width_px": round(2.0 * float(sc.meta["belt_halfwidth_px"]), 1),
        "cpu_ms": round(ms, 1),
        "size": sc.meta["size"],
    }


def main() -> None:
    rows = [run_scene(name) for name in GT_SUITE]

    ori = [r["ori_err_deg"] for r in rows if r["ori_err_deg"] is not None]
    belt = [r["belt_iou"] for r in rows]
    mis = [r for r in rows if r["misalign_err_deg"] is not None]
    ms = [r["cpu_ms"] for r in rows]

    summary = {
        "n_scenes": len(rows),
        "orientation_span_deg": sorted({r["orientation_gt_deg"] for r in rows}),
        "belt_iou_mean": round(float(np.mean(belt)), 3),
        "belt_iou_min": round(float(np.min(belt)), 3),
        "ori_err_mean_deg": round(float(np.mean(ori)), 2) if ori else None,
        "ori_err_max_deg": round(float(np.max(ori)), 2) if ori else None,
        "misalign_err_max_deg": round(float(np.max([r["misalign_err_deg"] for r in mis])), 2) if mis else None,
        "cpu_ms_mean": round(float(np.mean(ms)), 1),
        "cpu_ms_max": round(float(np.max(ms)), 1),
    }

    payload = {
        "schema": "beltvision-gt-benchmark/1.0",
        "task": "classical geometry+semantic recovery on the synthetic GT suite (use_learned=False)",
        "note": ("Synthetic scenes with EXACT known geometry; no data downloaded, nothing fabricated. "
                 "Measures how the training-free classical chain recovers belt mask, axis orientation, "
                 "centreline and injected misalignment across orientations and a curved path."),
        "package_version": _pkg_version(),
        "scenes": rows,
        "summary": summary,
    }
    out = DATA / "bv.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps(summary, indent=2))


def _pkg_version() -> str:
    try:
        import beltvision
        return getattr(beltvision, "__version__", "0.11.3")
    except Exception:
        return "0.11.3"


if __name__ == "__main__":
    main()
