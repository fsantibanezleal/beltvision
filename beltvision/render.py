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
from .methods.semantic import CLASS_COLORS_BGR

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


def to_png_b64(bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64 data URL (PNG)."""
    import base64

    import cv2

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
