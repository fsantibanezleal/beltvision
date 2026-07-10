"""Stage 5: evaluate.

Attaches honest metrics to each method result. For a synthetic control with a
scripted tear, the tear location is known, so the anomaly method gets a real
localization score (does the heatmap peak land near the tear?). For cases without
ground truth, only self-consistency metrics are reported and are labeled as such.

Rework surface: the full evaluation battery (mAP, IoU/PRO, image+pixel AUROC,
D50/D80 error, MOTA/IDF1/HOTA) on held-out, leakage-safe splits attaches here. The
frozen part is that metrics are computed against ground truth where it exists and
never fabricated where it does not.
"""
from __future__ import annotations

from typing import Any

from ..context import StageContext
from ..core.trace import stage_timer


def _tear_localization(ctx: StageContext) -> dict[str, Any]:
    """Score how close the anomaly peak is to the scripted synthetic tear centre.

    The tear runs through the image centre, so the ground-truth peak column is the
    grid centre column. Returns a normalized closeness in [0, 1].
    """
    grid = ctx.state.get("anomaly_grid")
    if grid is None:
        return {}
    rows, cols = grid.shape
    peak_row, peak_col = ctx.state["anomaly_peak_rc"]
    gt_col = (cols - 1) / 2.0
    col_err = abs(peak_col - gt_col) / max(cols - 1, 1)
    return {
        "gt": "synthetic-tear-centre",
        "peak_col_error_norm": round(float(col_err), 4),
        "localization_score": round(float(1.0 - col_err), 4),
    }


def evaluate(ctx: StageContext) -> dict[str, Any]:
    with stage_timer(ctx.trace, "evaluate") as t:
        methods: list[dict[str, Any]] = ctx.state["methods"]
        has_tear_gt = ctx.spec.synthetic and ctx.spec.tear

        for m in methods:
            if m["method"] == "patch_anomaly" and has_tear_gt:
                m["metrics"]["evaluation"] = _tear_localization(ctx)
            else:
                m["metrics"]["evaluation"] = {
                    "gt": "none",
                    "note": "self-consistency only; no ground truth for this case",
                }

        summary = {
            "ground_truth": "synthetic-tear" if has_tear_gt else "none",
            "n_evaluated": len(methods),
        }
        ctx.state["evaluate_summary"] = summary
        t.note(ground_truth=summary["ground_truth"])
    return summary
