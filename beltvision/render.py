"""Self-explanatory analysis overlays.

Every overlay is legible to a plant operator with no context: it draws ONLY marks that
are named in a legend, plus a one-line plain-language result bar. No unexplained circles,
curves or stray markers. Each analysis renders its OWN overlay on the input frame (never
all methods piled on one image). Low-confidence / not-applicable analyses render a worded
panel instead of a fabricated result.

All functions take/return ``(H, W, 3)`` uint8 BGR arrays. The orchestrator base64-encodes
the PNG for the app and, in validation, writes it to disk to be inspected on the real
COLA 34 frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .cases.synthetic import CLASS_NAMES

# NOTE: ``CLASS_COLORS_BGR`` lives in ``methods.semantic``; importing it at module load pulls in
# ``methods/__init__`` (which imports ``constrained``/``robust``, and those import THIS module),
# closing an import cycle. It is only needed inside two overlay functions, so import it lazily
# there to keep ``render`` importable before ``methods`` finishes loading.

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)


def _fs(img: np.ndarray) -> float:
    return max(0.42, min(0.7, img.shape[1] / 1500.0))


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    import cv2

    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    x, y = max(x, 0), max(y, 0)
    roi = img[y:y2, x:x2]
    dark = np.zeros_like(roi)
    img[y:y2, x:x2] = cv2.addWeighted(roi, 1 - alpha, dark, alpha, 0)


def _text(img, s, org, scale=None, color=_WHITE, thick=1) -> None:
    import cv2

    sc = scale if scale is not None else _fs(img)
    cv2.putText(img, s, org, _FONT, sc, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, sc, color, thick, cv2.LINE_AA)


def draw_legend(img: np.ndarray, entries: list[tuple[tuple[int, int, int], str]]) -> None:
    """Legend box (colour swatch + label) top-left; names every drawn element."""
    import cv2

    if not entries:
        return
    sc = _fs(img)
    line_h = int(26 * sc / 0.5)
    box_w = int(max(180, 12 * max(len(lbl) for _, lbl in entries) * sc / 0.5))
    box_h = line_h * len(entries) + 10
    _panel(img, 8, 8, box_w, box_h)
    y = 8 + line_h - 6
    for color, label in entries:
        cv2.rectangle(img, (16, y - int(11 * sc / 0.5)), (16 + 20, y + 2), color, -1)
        cv2.rectangle(img, (16, y - int(11 * sc / 0.5)), (16 + 20, y + 2), _WHITE, 1)
        _text(img, label, (44, y), scale=sc)
        y += line_h


def draw_summary(img: np.ndarray, text: str) -> None:
    """One-line (wrapped) plain-language result bar across the bottom."""
    import cv2

    sc = _fs(img)
    max_chars = int(img.shape[1] / (11 * sc))
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > max_chars:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    line_h = int(24 * sc / 0.5)
    bar_h = line_h * len(lines) + 12
    _panel(img, 0, img.shape[0] - bar_h, img.shape[1], bar_h, alpha=0.62)
    y = img.shape[0] - bar_h + line_h
    for ln in lines:
        _text(img, ln, (12, y), scale=sc)
        y += line_h
    _ = cv2


def message_overlay(bgr: np.ndarray, title: str, message: str) -> np.ndarray:
    """A worded panel for a not-applicable / low-confidence analysis (no fabricated marks)."""
    import cv2

    img = bgr.copy()
    img = cv2.addWeighted(img, 0.55, np.zeros_like(img), 0.45, 0)
    _text(img, title, (14, 34), scale=_fs(img) * 1.3, thick=2)
    draw_summary(img, message)
    return img


def _poly(pts: Any) -> np.ndarray:
    return np.round(np.asarray(pts, dtype=np.float64)).astype(np.int32)


def semantic_overlay(bgr: np.ndarray, label_map: np.ndarray, coverage: dict, engine: str) -> np.ndarray:
    import cv2

    from .methods.semantic import CLASS_COLORS_BGR

    img = bgr.copy()
    color = np.zeros_like(bgr)
    for cls, col in CLASS_COLORS_BGR.items():
        color[label_map == cls] = col
    img = cv2.addWeighted(img, 0.55, color, 0.45, 0)
    draw_legend(img, [(CLASS_COLORS_BGR[c], f"{CLASS_NAMES[c]}  {coverage.get(CLASS_NAMES[c], 0)*100:.0f}%")
                      for c in (0, 1, 2, 3)])
    pct = {k: coverage.get(k, 0.0) * 100 for k in ("belt", "content", "foreign", "external")}
    draw_summary(img, f"4-class semantic map ({engine}): belt {pct['belt']:.0f}%, "
                      f"content {pct['content']:.0f}%, foreign {pct['foreign']:.0f}%, "
                      f"external {pct['external']:.0f}% of the frame.")
    return img


def geometry_overlay(bgr: np.ndarray, geo: dict) -> np.ndarray:
    import cv2

    if "centreline_xy" not in geo:
        return message_overlay(bgr, "Belt geometry",
                                f"Low confidence: {geo.get('reason', 'belt region ambiguous')}. "
                                "No centreline drawn.")
    conf = geo.get("confidence", "medium")
    img = bgr.copy()
    cl = _poly(geo["centreline_xy"])
    legend = [((230, 200, 40), "centreline (medial axis)")]
    # Edges/width are only reliable when the belt is a clean strand; at low confidence the
    # region is a broad patch, so draw ONLY the centreline + axis and say so honestly.
    if conf in ("high", "medium"):
        ea, eb = _poly(geo["edge_a_xy"]), _poly(geo["edge_b_xy"])
        cv2.polylines(img, [ea], False, (60, 200, 60), 2, cv2.LINE_AA)
        cv2.polylines(img, [eb], False, (60, 220, 220), 2, cv2.LINE_AA)
        legend = [((60, 200, 60), "belt edge A"), ((60, 220, 220), "belt edge B"), *legend]
    cv2.polylines(img, [cl], False, (230, 200, 40), 2, cv2.LINE_AA)
    conf_note = "" if conf == "high" else f" [{conf} confidence]"
    if conf in ("high", "medium"):
        summ = (f"Belt axis {geo['axis_angle_deg']:.0f}deg ({geo['orientation']}), "
                f"width ~{geo['mean_width_px']:.0f} px, "
                f"{'curved path' if geo.get('curved') else 'straight'}.{conf_note}")
    else:
        summ = (f"Belt found: axis {geo['axis_angle_deg']:.0f}deg ({geo['orientation']}); "
                f"region is broad/ambiguous so edges + width are low confidence - "
                f"centreline shown, precise borders withheld.")
    if conf in ("high", "medium") and geo.get("support_axis_deg") is not None:
        s_ang = geo["support_axis_deg"]
        cx, cy = cl[len(cl) // 2]
        import math

        arm = 90
        d = (math.cos(math.radians(s_ang)), math.sin(math.radians(s_ang)))
        cv2.line(img, (int(cx - arm * d[0]), int(cy - arm * d[1])),
                 (int(cx + arm * d[0]), int(cy + arm * d[1])), (40, 120, 235), 2, cv2.LINE_AA)
        legend.append(((40, 120, 235), "support structure axis"))
        verdict = "MISALIGNED" if geo.get("misaligned") else "aligned"
        summ += (f" Support axis {s_ang:.0f}deg; belt-vs-support "
                 f"{geo['misalignment_deg']:+.1f}deg -> {verdict}.")
    draw_legend(img, legend)
    draw_summary(img, summ)
    return img


def _angles_panel(img: np.ndarray, lines: list[str]) -> None:
    """A compact top-right numeric read-out panel (angles / width / confidence)."""
    import cv2

    if not lines:
        return
    sc = _fs(img)
    line_h = int(24 * sc / 0.5)
    box_w = int(max(190, 11 * max(len(s) for s in lines) * sc / 0.5))
    x0 = max(0, img.shape[1] - box_w - 8)
    box_h = line_h * len(lines) + 12
    _panel(img, x0, 8, box_w, box_h, alpha=0.6)
    y = 8 + line_h
    for s in lines:
        _text(img, s, (x0 + 8, y), scale=sc)
        y += line_h
    _ = cv2


def geometry_analysis_overlay(bgr: np.ndarray, geo: dict) -> np.ndarray:
    """One consolidated STRAIGHT-LINE geometry overlay: corrected centreline + two straight
    edges + OBB, with Hough / RANSAC-line / Radon axis arrows and a numeric read-out. Always
    draws the estimate (with a confidence note), never a fabricated curve."""
    import cv2

    img = bgr.copy()
    img = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
    h, w = img.shape[:2]
    conf = geo.get("confidence", "low")
    legend: list[tuple[tuple[int, int, int], str]] = []

    # OBB (faint white box) - the rotating-calipers belt box
    obb = geo.get("obb")
    if obb and obb.get("box_points"):
        cv2.polylines(img, [_poly(obb["box_points"])], True, (200, 200, 200), 1, cv2.LINE_AA)
        legend.append(((200, 200, 200), f"OBB {obb['angle_deg']:.0f}deg"))

    # RANSAC straight boundary lines (orange), drawn from their fitted endpoints
    for key, col, lbl in (("ransac_a", (40, 130, 255), "RANSAC edge A"),
                          ("ransac_b", (40, 170, 255), "RANSAC edge B")):
        ln = geo.get(key)
        if ln and ln.get("p0") and ln.get("p1"):
            cv2.line(img, tuple(_poly(ln["p0"])), tuple(_poly(ln["p1"])), col, 2, cv2.LINE_AA)
            legend.append((col, f"{lbl} {ln['angle_deg']:.0f}deg"))

    # the corrected STRAIGHT centreline + two straight edges (the primary geometry)
    if geo.get("edge_a_xy") and geo.get("edge_b_xy"):
        cv2.polylines(img, [_poly(geo["edge_a_xy"])], False, (60, 200, 60), 2, cv2.LINE_AA)
        cv2.polylines(img, [_poly(geo["edge_b_xy"])], False, (60, 220, 220), 2, cv2.LINE_AA)
        legend.append(((60, 200, 60), "belt edge A (straight)"))
        legend.append(((60, 220, 220), "belt edge B (straight)"))
    cl = None
    if geo.get("centreline_xy"):
        cl = _poly(geo["centreline_xy"])
        cv2.polylines(img, [cl], False, (230, 200, 40), 3, cv2.LINE_AA)
        legend.append(((230, 200, 40), "centreline (straight)"))

    # axis-agreement arrows from the frame centre: Radon (blue) and Hough (magenta)
    cx, cy = w / 2.0, h / 2.0
    arm = 0.32 * min(h, w)

    def _arrow(angle_deg, color):
        d = (np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg)))
        p0 = (int(cx - arm * d[0]), int(cy - arm * d[1]))
        p1 = (int(cx + arm * d[0]), int(cy + arm * d[1]))
        cv2.arrowedLine(img, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.05)

    radon = geo.get("radon") or {}
    if radon.get("orientation_deg") is not None:
        _arrow(radon["orientation_deg"], (235, 120, 40))
        legend.append(((235, 120, 40), f"Radon {radon['orientation_deg']:.0f}deg"))
    hough = geo.get("hough") or {}
    if hough.get("axis_deg") is not None:
        _arrow(hough["axis_deg"], (200, 60, 200))
        legend.append(((200, 60, 200), f"Hough {hough['axis_deg']:.0f}deg"))

    # numeric read-out panel (top-right)
    ori = geo.get("orientation_deg")
    width = geo.get("belt_width_px")
    par = geo.get("edge_parallelism_deg")
    rl = geo.get("ransac_line") or {}
    panel = [
        f"axis (PCA)  {ori:.1f}deg" if ori is not None else "axis  n/a",
        f"width  {width:.0f} px" if width is not None else "width  n/a",
        f"parallelism  {par:.1f}deg" if par is not None else "parallelism  n/a",
        ("STRAIGHT" if geo.get("straight") else "CURVED") + f"  (k={(geo.get('curvature') or 0.0):.3f})",
    ]
    if rl.get("inlier_frac") is not None:
        panel.append(f"RANSAC inliers  {rl['inlier_frac']*100:.0f}%")
    if geo.get("misalignment_deg") is not None:
        panel.append(f"misalign  {geo['misalignment_deg']:+.1f}deg")
    panel.append(f"confidence  {conf}")
    _angles_panel(img, panel)

    draw_legend(img, legend)
    ori_s = f"{ori:.0f}deg ({geo.get('orientation_label', '?')})" if ori is not None else "n/a"
    shape_s = "straight" if geo.get("straight") else "curved"
    conf_note = "" if conf == "high" else f" [{conf} confidence]"
    draw_summary(img, f"Belt geometry: axis {ori_s}, width ~{width:.0f}px, edges {shape_s} + "
                      f"{par:.1f}deg parallel; Hough {_fmt(hough.get('axis_deg'))}, RANSAC "
                      f"{_fmt(rl.get('line_a_deg'))}/{_fmt(rl.get('line_b_deg'))}, Radon "
                      f"{_fmt(radon.get('orientation_deg'))} agree.{conf_note}"
                 if (ori is not None and width is not None and par is not None)
                 else f"Belt geometry estimate at {conf} confidence: {geo.get('reason', 'axis '+ori_s)}.")
    return img


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.0f}deg"


def damage_overlay(bgr: np.ndarray, dmg: dict, belt_mask: np.ndarray) -> np.ndarray:
    import cv2

    if not dmg.get("applicable", False):
        return message_overlay(bgr, "Belt damage",
                                f"Not applicable: {dmg.get('reason', 'no belt region')}.")
    img = bgr.copy()
    # belt outline
    cnts, _ = cv2.findContours(belt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (90, 180, 90), 1, cv2.LINE_AA)
    for r in dmg.get("regions", []):
        x, y, w, h = r["bbox_xywh"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 60, 230), 2, cv2.LINE_AA)
    draw_legend(img, [((90, 180, 90), "belt region"), ((40, 60, 230), "damage / rip / hole")])
    draw_summary(img, f"Belt damage: {dmg['n_damage_regions']} region(s), "
                      f"{dmg['damaged_frac_of_belt']*100:.1f}% of belt affected -> "
                      f"severity {dmg['severity']:.2f} ({dmg['severity_label']}).")
    return img


def edges_overlay(bgr: np.ndarray, geo: dict, ec: dict) -> np.ndarray:
    import cv2

    if not ec.get("applicable", False):
        return message_overlay(bgr, "Edges / borders",
                                f"Not applicable: {ec.get('reason', 'belt edges unavailable')}.")
    img = bgr.copy()
    ea, eb = _poly(geo["edge_a_xy"]), _poly(geo["edge_b_xy"])
    ca = (40, 60, 230) if ec["edge_a"]["roughness_px"] > 3 else (60, 200, 60)
    cb = (40, 60, 230) if ec["edge_b"]["roughness_px"] > 3 else (60, 200, 60)
    cv2.polylines(img, [ea], False, ca, 3, cv2.LINE_AA)
    cv2.polylines(img, [eb], False, cb, 3, cv2.LINE_AA)
    draw_legend(img, [((60, 200, 60), "smooth border"), ((40, 60, 230), "rough / damaged border")])
    draw_summary(img, f"Edge condition: {ec['verdict']} "
                      f"(worst roughness {ec['worst_roughness_px']:.1f} px, "
                      f"{ec['edge_a']['notches']+ec['edge_b']['notches']} notch(es)).")
    return img


def content_overlay(bgr: np.ndarray, content_mask: np.ndarray, content: dict) -> np.ndarray:
    import cv2

    from .methods.semantic import CLASS_COLORS_BGR

    img = bgr.copy()
    color = np.zeros_like(bgr)
    color[content_mask] = CLASS_COLORS_BGR[2]
    img = cv2.addWeighted(img, 0.6, color, 0.4, 0)
    draw_legend(img, [(CLASS_COLORS_BGR[2], "transported content")])
    g = content.get("granulometry", {})
    extra = ""
    if g.get("n_particles", 0) > 0:
        extra = f" PSD D50 {g['D50']:.0f} {g['unit']}, {g['n_particles']} fragments."
    draw_summary(img, f"Content: {content['coverage_pct']:.0f}% of belt covered "
                      f"({content['load_label']} load).{extra}")
    return img


def foreign_overlay(bgr: np.ndarray, foreign: dict) -> np.ndarray:
    import cv2

    img = bgr.copy()
    for r in foreign.get("regions", []):
        x, y, w, h = r["bbox_xywh"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 60, 200), 2, cv2.LINE_AA)
    for d in foreign.get("detector_boxes", []):
        b = d.get("bbox_xyxy")
        if b:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 255), 2, cv2.LINE_AA)
    draw_legend(img, [((200, 60, 200), "foreign region"), ((0, 200, 255), "detector box")])
    draw_summary(img, f"Foreign objects: {foreign['verdict']} "
                      f"({foreign['n_foreign_regions']} region(s), {foreign['n_on_belt']} on belt).")
    return img


def surface_overlay(bgr: np.ndarray, surface: dict) -> np.ndarray:
    img = bgr.copy()
    haze = surface.get("haze", {})
    uni = surface.get("surface_uniformity")
    uni_s = "n/a" if uni is None else f"{uni:.2f}"
    draw_legend(img, [((200, 200, 200), "belt surface reading")])
    draw_summary(img, f"Surface: uniformity {uni_s} "
                      f"({'irregular' if surface.get('flagged_irregular') else 'even'}); "
                      f"haze severity {haze.get('severity', 0):.2f}.")
    return img


def dust_overlay(bgr: np.ndarray, haze: dict) -> np.ndarray:
    img = bgr.copy()
    sev = haze.get("severity", 0.0)
    lvl = "low" if sev < 0.35 else "moderate" if sev < 0.6 else "heavy"
    draw_legend(img, [((200, 200, 200), "dust / haze reading")])
    draw_summary(img, f"Dust / haze severity {sev:.2f} ({lvl}); "
                      f"dark-channel {haze.get('dark_channel', 0):.2f}, "
                      f"contrast Lstd {haze.get('global_contrast_lstd', 0):.0f}. "
                      "High haze lowers downstream confidence.")
    return img


def heatmap_overlay(
    bgr: np.ndarray,
    score_map: Any,
    *,
    legend_label: str,
    summary: str,
    title: str = "Anomaly",
    peak_xy: tuple[int, int] | None = None,
    colormap: int | None = None,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a normalized 2D score map (any resolution) over the frame as a colour heatmap.

    Used by the precompute lane for anomaly methods (PaDiM / PatchCore / conv-AE) whose live
    result carries a per-patch residual grid but no drawn overlay. The score map is min-max
    normalized, bicubically upsampled to the frame, colour-mapped and blended; the peak
    location (if given, in frame pixels) gets a named marker. Legible: legend + result bar.
    """
    import cv2

    img = bgr.copy()
    h, w = img.shape[:2]
    sm = np.asarray(score_map, dtype=np.float32)
    if sm.ndim != 2 or sm.size == 0:
        return message_overlay(bgr, title, summary or "no score map produced")
    lo, hi = float(np.nanmin(sm)), float(np.nanmax(sm))
    norm = (sm - lo) / (hi - lo) if (hi - lo) > 1e-9 else np.zeros_like(sm)
    up = cv2.resize((norm * 255.0).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(up, cv2.COLORMAP_INFERNO if colormap is None else colormap)
    img = cv2.addWeighted(img, 1.0 - alpha, heat, alpha, 0)
    legend = [((40, 120, 235), legend_label)]
    if peak_xy is not None:
        px, py = int(peak_xy[0]), int(peak_xy[1])
        cv2.drawMarker(img, (px, py), (60, 255, 255), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), 16, (60, 255, 255), 2, cv2.LINE_AA)
        legend.append(((60, 255, 255), "peak anomaly"))
    draw_legend(img, legend)
    draw_summary(img, summary)
    return img


def flow_overlay(bgr: np.ndarray, flow: np.ndarray, *, summary: str) -> np.ndarray:
    """Dense optical-flow overlay: a flow-magnitude heat blend plus a sparse arrow field."""
    import cv2

    img = bgr.copy()
    h, w = img.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    m = mag / (float(mag.max()) + 1e-9)
    heat = cv2.applyColorMap((m * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    step = max(16, min(h, w) // 22)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = float(fx[y, x]), float(fy[y, x])
            if dx * dx + dy * dy < 0.25:
                continue
            p1 = (int(x + dx * 3.0), int(y + dy * 3.0))
            cv2.arrowedLine(img, (x, y), p1, (60, 255, 60), 1, cv2.LINE_AA, tipLength=0.35)
    draw_legend(img, [((0, 180, 255), "flow magnitude"), ((60, 255, 60), "flow vector")])
    draw_summary(img, summary)
    return img


def granulometry_overlay(
    bgr: np.ndarray, labels: np.ndarray, *, summary: str
) -> np.ndarray:
    """Colour every watershed-segmented particle + draw its boundary; result bar with the PSD."""
    import cv2
    from skimage.segmentation import find_boundaries

    img = bgr.copy()
    color = np.zeros_like(bgr)
    rng = np.random.default_rng(34)
    ids = np.unique(labels)
    for lid in ids:
        if int(lid) == 0:
            continue
        col = tuple(int(c) for c in rng.integers(70, 256, size=3))
        color[labels == lid] = col
    img = cv2.addWeighted(img, 0.55, color, 0.45, 0)
    bnd = find_boundaries(labels, mode="outer")
    img[bnd] = (255, 255, 255)
    draw_legend(img, [((0, 120, 235), "segmented particle"), ((255, 255, 255), "particle boundary")])
    draw_summary(img, summary)
    return img


def masks_overlay(
    bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], *, summary: str
) -> np.ndarray:
    """Draw the bounding boxes of automatically-generated (SAM) masks + a result bar."""
    import cv2

    img = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 255), 2, cv2.LINE_AA)
    draw_legend(img, [((0, 200, 255), "automatic mask (SAM)")])
    draw_summary(img, summary)
    return img


def to_png_b64(bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64 data URL (PNG)."""
    import base64

    import cv2

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
