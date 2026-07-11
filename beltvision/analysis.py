"""The 3-stage view-aware analysis orchestrator.

``analyze_scene`` runs the whole tool on a single frame:

1. STAGE 1 - recognise the view (or accept a caller override).
2. STAGE 2 - map the view to its applicable analyses.
3. STAGE 3 - segment the 4 semantic layers ONCE, then run each applicable analysis
   FROM those layers (belt geometry from the belt mask, content from the mineral mask,
   foreign from the foreign class, ...), producing a legible per-analysis overlay + a
   plain-language summary + metrics.

The result is grouped BY ANALYSIS so a front end can show one horizontal tab per
analysis. Overlays are returned as base64 PNG data URLs (rendered by :mod:`render`); the
numeric metrics are JSON-safe. Nothing here ever returns ``weights_absent``: the semantic
backbone degrades to its classical prior and says so; an analysis that does not apply to
the view/mask reports a worded status instead of a fabricated result.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import render
from .cases.synthetic import EXTERNAL
from .methods import analyses as ana
from .methods._common import as_bgr
from .methods.beltline import compute_belt_geometry
from .methods.preprocess import apply_clahe_lab, haze_severity
from .methods.semantic import compute_layers
from .views import ANALYSIS_META, VIEW_LABELS, analyses_for_view, recognize_view


def _mk(analysis_id: str, status: str, summary: str, metrics: dict,
        overlay_bgr: np.ndarray | None, *, applicable: bool = True,
        lane: str = "live-server", mode: str = "live") -> dict[str, Any]:
    meta = ANALYSIS_META.get(analysis_id, {})
    return {
        "id": analysis_id,
        "title": meta.get("title", analysis_id),
        "about": meta.get("about", ""),
        "layer": meta.get("layer", "all"),
        "status": status,
        "applicable": applicable,
        "summary": summary,
        "metrics": metrics,
        "lane": lane,
        "mode": mode,
        "overlay": render.to_png_b64(overlay_bgr) if overlay_bgr is not None else None,
    }


def analyze_scene(
    image: Any,
    *,
    view_type: str | None = None,
    analyses: list[str] | None = None,
    px_per_mm: float | None = None,
    use_learned: bool = False,
    want_overlays: bool = True,
    detections: list | None = None,
) -> dict[str, Any]:
    """Run the full 3-stage view-aware analysis on one frame."""
    t0 = time.time()
    bgr = apply_clahe_lab(as_bgr(image))

    # STAGE 1: recognise (or accept override)
    recognized = recognize_view(bgr)
    used_view = view_type or recognized["view_type"]
    overridden = bool(view_type and view_type != recognized["view_type"])

    # STAGE 2: applicable analyses for the view
    applicable = analyses_for_view(used_view)
    wanted = [a for a in (analyses or applicable) if a in applicable]

    # STAGE 3: segment once, then derive
    layers = compute_layers(bgr, view_type=used_view, use_learned=use_learned)
    belt_mask = layers.belt_mask
    # The belt FOOTPRINT is the physical belt region: exposed belt plus whatever content
    # rides on top of it. Geometry (axis/centreline/edges) is derived from the footprint so
    # a loaded belt's axis is still recovered; damage/edges/surface use the exposed belt.
    belt_footprint = belt_mask | layers.content_mask
    lane = "precompute" if use_learned else "live-server"
    mode = "precompute" if use_learned else "live"

    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    geo = compute_belt_geometry(belt_footprint, external_mask=layers.mask(EXTERNAL), gray=gray)

    out: dict[str, Any] = {}
    for aid in wanted:
        if aid == "semantic":
            ov = render.semantic_overlay(bgr, layers.label_map, layers.coverage, layers.engine) \
                if want_overlays else None
            pct = {k: round(v * 100, 1) for k, v in layers.coverage.items()}
            out[aid] = _mk(aid, "ok",
                           f"Semantic layers ({layers.engine}): belt {pct['belt']}%, "
                           f"content {pct['content']}%, foreign {pct['foreign']}%, "
                           f"external {pct['external']}%.",
                           {"engine": layers.engine, "coverage": layers.coverage,
                            "n_regions": layers.n_regions}, ov, lane=lane, mode=mode)
        elif aid == "belt_geometry":
            ov = render.geometry_overlay(bgr, geo) if want_overlays else None
            if geo.get("confidence") == "low":
                if "axis_angle_deg" in geo:
                    s = (f"Belt found: axis {geo['axis_angle_deg']:.0f}deg "
                         f"({geo['orientation']}); region broad/ambiguous, edges + width "
                         f"low confidence.")
                else:
                    s = f"Low confidence: {geo.get('reason', 'belt ambiguous')}."
                out[aid] = _mk(aid, "low_confidence", s, geo, ov, applicable=True,
                               lane=lane, mode=mode)
            else:
                s = (f"Belt axis {geo['axis_angle_deg']:.0f}deg ({geo['orientation']}), "
                     f"width ~{geo['mean_width_px']:.0f}px, "
                     f"{'curved' if geo.get('curved') else 'straight'}.")
                if geo.get("misalignment_deg") is not None:
                    s += (f" Misalignment vs support {geo['misalignment_deg']:+.1f}deg "
                          f"({'misaligned' if geo.get('misaligned') else 'aligned'}).")
                out[aid] = _mk(aid, "ok", s, geo, ov, lane=lane, mode=mode)
        elif aid == "damage":
            dmg = ana.belt_damage(bgr, belt_mask)
            ov = render.damage_overlay(bgr, dmg, belt_mask) if want_overlays else None
            if not dmg.get("applicable"):
                out[aid] = _mk(aid, "na", f"Not applicable: {dmg.get('reason')}.", dmg, ov,
                               applicable=False, lane=lane, mode=mode)
            else:
                out[aid] = _mk(aid, "ok",
                               f"{dmg['n_damage_regions']} damage region(s), severity "
                               f"{dmg['severity']:.2f} ({dmg['severity_label']}).",
                               dmg, ov, lane=lane, mode=mode)
        elif aid == "edges":
            if geo.get("confidence") == "low":
                ec = {"applicable": False, "status": "na",
                      "reason": "belt edges low confidence (region broad/ambiguous)"}
            else:
                ec = ana.edge_condition(belt_mask, geo.get("edge_a_xy"), geo.get("edge_b_xy"))
            ov = render.edges_overlay(bgr, geo, ec) if want_overlays else None
            if not ec.get("applicable"):
                out[aid] = _mk(aid, "na", f"Not applicable: {ec.get('reason')}.", ec, ov,
                               applicable=False, lane=lane, mode=mode)
            else:
                out[aid] = _mk(aid, "ok", ec["verdict"] +
                               f" (roughness {ec['worst_roughness_px']:.1f}px).", ec, ov,
                               lane=lane, mode=mode)
        elif aid == "surface":
            surf = ana.surface_state(bgr, belt_mask)
            ov = render.surface_overlay(bgr, surf) if want_overlays else None
            uni = surf.get("surface_uniformity")
            out[aid] = _mk(aid, "ok",
                           f"Surface uniformity {('n/a' if uni is None else f'{uni:.2f}')}; "
                           f"haze {surf['haze']['severity']:.2f}.", surf, ov, lane=lane, mode=mode)
        elif aid == "dust":
            haze = haze_severity(bgr)
            ov = render.dust_overlay(bgr, haze) if want_overlays else None
            out[aid] = _mk(aid, "ok", f"Dust/haze severity {haze['severity']:.2f}.",
                           {"haze": haze}, ov, lane="live-web", mode=mode)
        elif aid == "content":
            content = ana.content_quantity(bgr, belt_mask, layers.content_mask, px_per_mm=px_per_mm)
            ov = render.content_overlay(bgr, layers.content_mask, content) if want_overlays else None
            out[aid] = _mk(aid, "ok",
                           f"Content coverage {content['coverage_pct']:.0f}% "
                           f"({content['load_label']} load).", content, ov, lane=lane, mode=mode)
        elif aid == "foreign":
            fo = ana.foreign_objects(bgr, layers.foreign_mask, belt_mask, detections=detections)
            ov = render.foreign_overlay(bgr, fo) if want_overlays else None
            out[aid] = _mk(aid, "ok", fo["verdict"], fo, ov, lane=lane, mode=mode)

    return {
        "engine": "beltvision",
        "shape": [int(bgr.shape[0]), int(bgr.shape[1])],
        "recognized_view": recognized,
        "view_type": used_view,
        "view_label": VIEW_LABELS.get(used_view, used_view),
        "view_overridden": overridden,
        "applicable_analyses": applicable,
        "analyses": out,
        "layers_engine": layers.engine,
        "mode": mode,
        "latency_s": round(time.time() - t0, 3),
    }
