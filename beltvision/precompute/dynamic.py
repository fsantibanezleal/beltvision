"""The temporal / dynamic sequence-video engine (offline PRECOMPUTE lane).

Given an ordered frame sequence (or frames sampled from a video), this runs the per-frame
ladder (CLAHE -> semantic segmentation -> belt geometry -> content coverage) PLUS the
genuinely TEMPORAL analyses that only exist across frames:

- object / particle TRACKING: blob boxes detected per frame (contrast blobs inside the belt
  footprint) associated into persistent tracks with stable ids by a ByteTrack-style
  associator (``methods.tracking.ByteTrackAssociator``).
- belt SPEED + direction: dense Farneback optical flow between consecutive frames -> a
  per-frame belt-speed (relative px/frame) and material-flow direction.
- belt-edge / alignment DRIFT: the belt footprint centroid (and axis angle) tracked over
  time -> a lateral wander trend vs a baseline computed from the first frames.
- content COVERAGE over time: the material coverage fraction of the belt footprint per frame.
- foreign-object / object EVENTS: track births/deaths and foreign-region appear/clear and
  belt stop/start transitions, emitted as a timestamped event stream.

It renders one legible annotated overlay per frame (belt edges/centreline, tracks with ids,
a flow arrow and a live metrics HUD) and, in the precompute lane, encodes those overlays to
a compact H.264 mp4 via ``imageio`` + ``imageio-ffmpeg`` (bundled ffmpeg, no system install).
The numeric result is a set of metric TIMELINES committed as JSON. Everything is deterministic
given the seed: the classical ops are deterministic and no randomness is used in association.

The runtime (the app's slim CPU venv) never imports this module - it only REPLAYS the
committed mp4 + timelines. ``imageio`` is imported lazily inside :func:`encode_video` so the
per-frame analysis (:func:`analyze_sequence`) runs with only the classical stack present.

References: Farneback 2003 (SCIA, dense optical flow); ByteTrack arXiv:2110.06864 (MIT);
Kalman 1960 (wander trend); CLAHE clip2.0/8x8 LAB-L (dust remedy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import __version__
from ..cases.synthetic import CLASS_NAMES, CONTENT, EXTERNAL
from ..methods.beltline import compute_belt_geometry
from ..methods.preprocess import apply_clahe_lab, haze_severity
from ..methods.semantic import CLASS_COLORS_BGR, compute_layers
from ..methods.tracking import ByteTrackAssociator

_TARGET_LONG_SIDE = 640
_MAX_BLOBS = 40
_MAX_EVENTS = 200
_WHITE = (255, 255, 255)
_FLOW_COLOR = (255, 220, 60)      # cyan-ish, BGR
_CENTRELINE_COLOR = (230, 200, 40)
_EDGE_A_COLOR = (60, 200, 60)
_EDGE_B_COLOR = (60, 220, 220)
# a stable, distinct palette so a given track id keeps its colour across frames
_TRACK_PALETTE = (
    (66, 135, 245), (48, 200, 120), (200, 90, 220), (60, 200, 235),
    (235, 120, 60), (120, 90, 235), (40, 170, 250), (200, 200, 60),
    (90, 220, 160), (245, 90, 150),
)


@dataclass
class SequenceResult:
    """The full temporal result of one sequence: timelines + annotated frames + events."""

    timelines: dict[str, list[Any]]
    annotated_frames: list[np.ndarray]
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    frame_size: tuple[int, int]           # (h, w) of the analysed/rendered frames
    n_frames: int
    layers_engine: str
    view_type: str | None


# --- frame prep -------------------------------------------------------------------------
def _resize_long_side(bgr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    ls = max(h, w)
    if ls <= long_side:
        return bgr
    scale = long_side / float(ls)
    return cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


# --- per-frame blob detection (the boxes the tracker associates) ------------------------
def _detect_blobs(clahe_bgr: np.ndarray, footprint: np.ndarray, max_blobs: int) -> np.ndarray:
    """Contrast blobs (ore fragments / bright spots) inside the belt footprint -> (N,5) boxes.

    Returns ``[x0, y0, x1, y1, score]`` rows. The score is an area rank in ``[0.25, 0.95]`` so
    the ByteTrack high/low two-stage association is genuinely exercised. The detector is
    classical (black-hat/top-hat contrast + connected components); tracking is what is temporal.
    """
    h, w = clahe_bgr.shape[:2]
    gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    resp = cv2.max(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k),
                   cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)).astype(np.float32)
    region = footprint if footprint.sum() > 0.02 * h * w else np.ones((h, w), bool)
    resp[~region] = 0.0
    vals = resp[region]
    thr = max(float(np.percentile(vals, 88)) if vals.size else 0.0, 10.0)
    mask = ((resp >= thr) & region).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(9, int(0.00015 * h * w))
    max_area = int(0.05 * h * w)
    boxes: list[tuple[float, float, float, float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((float(x), float(y), float(x + bw), float(y + bh), float(area)))
    if not boxes:
        return np.zeros((0, 5), dtype=np.float64)
    boxes.sort(key=lambda b: b[4], reverse=True)
    boxes = boxes[:max_blobs]
    m = len(boxes)
    out = np.zeros((m, 5), dtype=np.float64)
    for rank, (x0, y0, x1, y1, _area) in enumerate(boxes):
        score = 0.95 - 0.70 * (rank / max(m - 1, 1))  # area rank -> [0.25, 0.95]
        out[rank] = (x0, y0, x1, y1, score)
    return out


# --- per-frame optical flow (belt speed + direction) ------------------------------------
def _optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> dict[str, float | bool]:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    mean_dx, mean_dy = float(fx.mean()), float(fy.mean())
    speed = float(np.median(mag))
    direction = float(np.degrees(np.arctan2(mean_dy, mean_dx)))
    return {
        "speed": round(speed, 4),
        "direction": round(direction, 2),
        "mean_dx": round(mean_dx, 4),
        "mean_dy": round(mean_dy, 4),
        "moving": bool(speed > 0.25),
    }


def _centroid_x(mask: np.ndarray) -> float | None:
    if mask.sum() < 50:
        return None
    xs = np.nonzero(mask.any(axis=0))[0]
    ys, cols = np.nonzero(mask)
    return float(cols.mean()) if cols.size else (float(xs.mean()) if xs.size else None)


# --- the temporal analysis over a whole sequence ----------------------------------------
def analyze_sequence(
    frames: list[np.ndarray],
    *,
    view_type: str | None = None,
    seed: int = 34,
    use_learned: bool = False,
    long_side: int = _TARGET_LONG_SIDE,
    max_blobs: int = _MAX_BLOBS,
) -> SequenceResult:
    """Run the per-frame ladder + temporal analyses over an ordered frame sequence."""
    if len(frames) < 2:
        raise ValueError("a sequence needs at least 2 frames for temporal analysis")
    # seed is recorded for provenance; the classical pipeline + association are deterministic.
    _rng = np.random.default_rng(seed)

    prepared = [_resize_long_side(np.ascontiguousarray(f), long_side) for f in frames]
    h, w = prepared[0].shape[:2]
    prepared = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                if f.shape[:2] != (h, w) else f for f in prepared]

    tracker = ByteTrackAssociator(iou_thresh=0.2, max_age=8)
    tl: dict[str, list[Any]] = {
        "t": [], "belt_speed_px_per_frame": [], "flow_direction_deg": [],
        "coverage_pct": [], "drift_px": [], "center_offset_frac": [],
        "axis_angle_deg": [], "track_count": [], "n_foreign": [], "haze": [], "moving": [],
    }
    annotated: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    centroids: list[float | None] = []
    prev_gray: np.ndarray | None = None
    prev_ids: set[int] = set()
    prev_foreign = 0
    prev_moving: bool | None = None
    last_angle = 90.0
    layers_engine = "classical-prior"

    for i, bgr_raw in enumerate(prepared):
        clahe = apply_clahe_lab(bgr_raw)
        gray = cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)
        layers = compute_layers(clahe, view_type=view_type, use_learned=use_learned)
        layers_engine = layers.engine
        footprint = layers.belt_mask | layers.content_mask
        geo = compute_belt_geometry(footprint, external_mask=layers.mask(EXTERNAL), gray=gray)

        # coverage of the belt footprint by the transported material
        fp_area = int(footprint.sum())
        coverage = (int(layers.content_mask.sum()) / fp_area) if fp_area > 0 else 0.0

        # optical flow (belt speed + direction) vs the previous frame
        if prev_gray is None:
            flow = {"speed": 0.0, "direction": 0.0, "mean_dx": 0.0, "mean_dy": 0.0, "moving": False}
        else:
            flow = _optical_flow(prev_gray, gray)

        # tracking: blob boxes -> persistent tracks. A track counts only once CONFIRMED
        # (>=2 hits and updated this frame), so the count + event stream reflect genuinely
        # associated fragments rather than one-frame detector blips.
        blobs = _detect_blobs(clahe, footprint, max_blobs)
        tracks = tracker.update(blobs)
        active = [tr for tr in tracks if tr.time_since_update == 0 and tr.hits >= 2]
        active_ids = {tr.track_id for tr in active}

        # drift: belt footprint centroid x (baseline resolved after the loop)
        cx = _centroid_x(footprint)
        centroids.append(cx)
        cx_val = cx if cx is not None else (w / 2.0)
        center_offset_frac = (cx_val - w / 2.0) / w

        # axis angle over time (carry the last valid reading through low-confidence frames)
        if geo.get("confidence") != "low" and "axis_angle_deg" in geo:
            last_angle = float(geo["axis_angle_deg"])
        angle = last_angle

        haze = haze_severity(bgr_raw)["severity"]
        n_foreign = int(_count_regions(layers.foreign_mask, h, w))

        # --- events (births/deaths, foreign appear/clear, belt stop/start) ---
        for new_id in sorted(active_ids - prev_ids):
            _push_event(events, i, "object_appear", {"track_id": int(new_id)})
        for gone_id in sorted(prev_ids - active_ids):
            _push_event(events, i, "object_disappear", {"track_id": int(gone_id)})
        if n_foreign > prev_foreign:
            _push_event(events, i, "foreign_appear", {"n_foreign": n_foreign})
        elif n_foreign < prev_foreign and n_foreign == 0:
            _push_event(events, i, "foreign_clear", {"n_foreign": n_foreign})
        if prev_moving is not None and bool(flow["moving"]) != prev_moving:
            _push_event(events, i, "belt_start" if flow["moving"] else "belt_stop",
                        {"belt_speed_px_per_frame": flow["speed"]})

        # --- timelines row ---
        tl["t"].append(i)
        tl["belt_speed_px_per_frame"].append(round(float(flow["speed"]), 4))
        tl["flow_direction_deg"].append(round(float(flow["direction"]), 2))
        tl["coverage_pct"].append(round(100.0 * coverage, 2))
        tl["center_offset_frac"].append(round(float(center_offset_frac), 4))
        tl["axis_angle_deg"].append(round(float(angle), 2))
        tl["track_count"].append(int(len(active)))
        tl["n_foreign"].append(n_foreign)
        tl["haze"].append(round(float(haze), 4))
        tl["moving"].append(bool(flow["moving"]))

        annotated.append(_draw_overlay(
            clahe, layers, geo, active, flow,
            hud={"i": i, "n": len(prepared), "speed": float(flow["speed"]),
                 "direction": float(flow["direction"]), "coverage": 100.0 * coverage,
                 "tracks": len(active), "offset": center_offset_frac, "moving": bool(flow["moving"]),
                 "haze": float(haze)}))

        prev_gray = gray
        prev_ids = active_ids
        prev_foreign = n_foreign
        prev_moving = bool(flow["moving"])

    # resolve drift against a baseline centroid (median of the first up to 3 valid centroids)
    valid = [c for c in centroids if c is not None]
    baseline = float(np.median(valid[:3])) if valid else (w / 2.0)
    drift = []
    last = baseline
    for c in centroids:
        last = c if c is not None else last
        drift.append(round(float(last - baseline), 2))
    tl["drift_px"] = drift

    summary = _summarize(tl, events, baseline, (h, w))
    return SequenceResult(
        timelines=tl, annotated_frames=annotated, events=events, summary=summary,
        frame_size=(h, w), n_frames=len(prepared), layers_engine=layers_engine,
        view_type=view_type,
    )


def _count_regions(mask: np.ndarray, h: int, w: int) -> int:
    if mask.sum() == 0:
        return 0
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = max(30, int(0.0006 * h * w))
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area))


def _push_event(events: list[dict[str, Any]], t: int, ev_type: str, detail: dict[str, Any]) -> None:
    if len(events) < _MAX_EVENTS:
        events.append({"t": int(t), "type": ev_type, **detail})


def _summarize(tl: dict[str, list[Any]], events: list[dict[str, Any]],
               baseline: float, hw: tuple[int, int]) -> dict[str, Any]:
    speed = np.asarray(tl["belt_speed_px_per_frame"], dtype=np.float64)
    drift = np.asarray(tl["drift_px"], dtype=np.float64)
    cover = np.asarray(tl["coverage_pct"], dtype=np.float64)
    tracks = np.asarray(tl["track_count"], dtype=np.float64)
    moving = np.asarray(tl["moving"], dtype=bool)
    n = len(tl["t"])
    # a simple least-squares drift trend (px per frame) over the wander series
    trend = float(np.polyfit(np.arange(n), drift, 1)[0]) if n >= 2 else 0.0
    ev_counts: dict[str, int] = {}
    for e in events:
        ev_counts[e["type"]] = ev_counts.get(e["type"], 0) + 1
    return {
        "n_frames": int(n),
        "belt_speed_mean_px_per_frame": round(float(speed.mean()), 3),
        "belt_speed_max_px_per_frame": round(float(speed.max()), 3),
        "moving_fraction": round(float(moving.mean()), 3),
        "coverage_mean_pct": round(float(cover.mean()), 2),
        "coverage_max_pct": round(float(cover.max()), 2),
        "track_count_mean": round(float(tracks.mean()), 2),
        "track_count_max": int(tracks.max()) if n else 0,
        "drift_px_max_abs": round(float(np.abs(drift).max()), 2) if n else 0.0,
        "drift_trend_px_per_frame": round(trend, 4),
        "baseline_centroid_x_px": round(baseline, 2),
        "frame_height_px": int(hw[0]),
        "frame_width_px": int(hw[1]),
        "n_events": len(events),
        "event_counts": ev_counts,
        "calibration": "relative px/frame (no fps or px_per_mm; m/s not fabricated)",
    }


# --- rendering --------------------------------------------------------------------------
def _fs(img: np.ndarray) -> float:
    return max(0.4, min(0.62, img.shape[1] / 1400.0))


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float,
          color: tuple[int, int, int] = _WHITE, thick: int = 1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    x, y = max(x, 0), max(y, 0)
    roi = img[y:y2, x:x2]
    img[y:y2, x:x2] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def _poly(pts: Any) -> np.ndarray:
    return np.round(np.asarray(pts, dtype=np.float64)).astype(np.int32)


def _draw_hud(img: np.ndarray, hud: dict[str, Any]) -> None:
    sc = _fs(img)
    lines = [
        f"frame {hud['i'] + 1}/{hud['n']}",
        f"belt speed {hud['speed']:.2f} px/f  ({'moving' if hud['moving'] else 'stopped'})",
        f"flow dir {hud['direction']:+.0f} deg",
        f"coverage {hud['coverage']:.0f}%",
        f"tracks {hud['tracks']}",
        f"drift {hud['offset'] * 100:+.1f}% of width",
        f"haze {hud['haze']:.2f}",
    ]
    line_h = int(20 * sc / 0.5)
    box_w = int(210 * sc / 0.5)
    box_h = line_h * len(lines) + 10
    x = img.shape[1] - box_w - 8
    _panel(img, x, 8, box_w, box_h, alpha=0.58)
    y = 8 + line_h - 5
    for ln in lines:
        _text(img, ln, (x + 8, y), sc)
        y += line_h


def _draw_legend(img: np.ndarray, entries: list[tuple[tuple[int, int, int], str]]) -> None:
    if not entries:
        return
    sc = _fs(img)
    line_h = int(22 * sc / 0.5)
    box_w = int(max(150, 11 * max(len(lbl) for _, lbl in entries)) * sc / 0.5)
    box_h = line_h * len(entries) + 8
    _panel(img, 8, 8, box_w, box_h, alpha=0.55)
    y = 8 + line_h - 6
    for color, label in entries:
        cv2.rectangle(img, (16, y - int(10 * sc / 0.5)), (34, y + 1), color, -1)
        cv2.rectangle(img, (16, y - int(10 * sc / 0.5)), (34, y + 1), _WHITE, 1)
        _text(img, label, (42, y), sc)
        y += line_h


def _draw_overlay(clahe_bgr: np.ndarray, layers: Any, geo: dict[str, Any],
                  tracks: list[Any], flow: dict[str, float | bool],
                  hud: dict[str, Any]) -> np.ndarray:
    img = clahe_bgr.copy()
    h, w = img.shape[:2]

    # 1) content tint (so the load reads at a glance)
    if layers.content_mask.sum() > 0:
        color = np.zeros_like(img)
        color[layers.content_mask] = CLASS_COLORS_BGR[CONTENT]
        img = cv2.addWeighted(img, 0.82, color, 0.18, 0)

    legend: list[tuple[tuple[int, int, int], str]] = []
    # 2) belt edges + centreline (or footprint contour at low confidence)
    conf = geo.get("confidence")
    if conf in ("high", "medium") and "edge_a_xy" in geo:
        cv2.polylines(img, [_poly(geo["edge_a_xy"])], False, _EDGE_A_COLOR, 2, cv2.LINE_AA)
        cv2.polylines(img, [_poly(geo["edge_b_xy"])], False, _EDGE_B_COLOR, 2, cv2.LINE_AA)
        cv2.polylines(img, [_poly(geo["centreline_xy"])], False, _CENTRELINE_COLOR, 2, cv2.LINE_AA)
        legend += [(_EDGE_A_COLOR, "belt edge"), (_CENTRELINE_COLOR, "centreline")]
    else:
        footprint = (layers.belt_mask | layers.content_mask).astype(np.uint8)
        cnts, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (90, 180, 90), 1, cv2.LINE_AA)
        legend.append(((90, 180, 90), "belt footprint"))

    # 3) tracks (boxes + ids), coloured by id
    for tr in tracks:
        x0, y0, x1, y1 = (int(round(v)) for v in tr.bbox)
        col = _TRACK_PALETTE[tr.track_id % len(_TRACK_PALETTE)]
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        _text(img, f"#{tr.track_id}", (x0, max(y0 - 3, 9)), _fs(img) * 0.9, col, 1)
    if tracks:
        legend.append(((66, 135, 245), "tracked fragment (id)"))

    # 4) flow arrow from the frame centre (length scaled by belt speed)
    if bool(flow["moving"]):
        cx, cy = w // 2, h // 2
        speed = float(flow["speed"])
        length = float(np.clip(speed * 6.0, 16.0, 0.22 * max(h, w)))
        ang = np.radians(float(flow["direction"]))
        ex, ey = int(cx + length * np.cos(ang)), int(cy + length * np.sin(ang))
        cv2.arrowedLine(img, (cx, cy), (ex, ey), _FLOW_COLOR, 3, cv2.LINE_AA, tipLength=0.3)
        legend.append((_FLOW_COLOR, "belt flow"))

    _draw_legend(img, legend)
    _draw_hud(img, hud)
    return img


# --- encoding (imageio + bundled ffmpeg; lazy import) -----------------------------------
def encode_video(frames_bgr: list[np.ndarray], out_path: str | Path, *,
                 fps: int = 12, hold: int = 3, quality: int = 6) -> int:
    """Encode BGR overlay frames to a compact H.264 mp4. Returns the file size in bytes.

    Each source frame is written ``hold`` times at ``fps`` so a short sampled sequence plays
    back over a few smooth seconds. ``imageio``/``imageio-ffmpeg`` bundle ffmpeg (no system
    install, no network). Frames are padded to even dimensions (yuv420p requirement).
    """
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=int(fps), codec="libx264", quality=int(quality),
        macro_block_size=16, pixelformat="yuv420p", ffmpeg_log_level="error",
    )
    try:
        for fr in frames_bgr:
            rgb = cv2.cvtColor(_pad_macro(fr), cv2.COLOR_BGR2RGB)
            for _ in range(max(1, int(hold))):
                writer.append_data(rgb)
    finally:
        writer.close()
    return out_path.stat().st_size


def _pad_macro(bgr: np.ndarray, macro: int = 16) -> np.ndarray:
    """Pad to a multiple of ``macro`` (libx264/yuv420p macro-block) so ffmpeg never resizes."""
    h, w = bgr.shape[:2]
    ph = (-h) % macro
    pw = (-w) % macro
    if ph or pw:
        bgr = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return bgr


# --- full precompute for one case (analyse + render + encode + write artifacts) ---------
def precompute_sequence(
    frames: list[np.ndarray],
    out_dir: str | Path,
    *,
    case_id: str,
    kind: str = "sequence",
    view_type: str | None = None,
    seed: int = 34,
    use_learned: bool = False,
    device: str = "cpu",
    source: str = "committed-frames",
    fps: int = 12,
    hold: int = 3,
    long_side: int = _TARGET_LONG_SIDE,
    video_name: str = "annotated.mp4",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyse a sequence, render + encode the annotated video, write timelines + manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyze_sequence(frames, view_type=view_type, seed=seed,
                           use_learned=use_learned, long_side=long_side)

    video_path = out_dir / video_name
    video_bytes = encode_video(res.annotated_frames, video_path, fps=fps, hold=hold)

    timelines = {
        **res.timelines,
        "events": res.events,
        "units": {
            "belt_speed_px_per_frame": "relative px/frame (uncalibrated)",
            "flow_direction_deg": "degrees from +x (image), atan2(dy,dx)",
            "coverage_pct": "percent of belt footprint covered by material",
            "drift_px": "belt-footprint centroid lateral offset vs the baseline (px)",
            "center_offset_frac": "centroid offset from frame centre / width",
            "axis_angle_deg": "belt axis angle from x-axis (deg)",
            "track_count": "active tracks in the frame",
        },
        "notes": ("Relative px/frame - no fps or px_per_mm calibration, so speed is NOT in m/s. "
                  "Blob boxes are classical; ByteTrack association is the temporal step."),
    }
    (out_dir / "timelines.json").write_text(json.dumps(timelines, indent=2) + "\n", encoding="utf-8")

    duration_s = round(res.n_frames * max(1, hold) / float(fps), 2)
    manifest = {
        "case_id": case_id,
        "kind": kind,
        "engine": "beltvision",
        "engine_version": __version__,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": int(seed),
        "device": device,
        "source": source,
        "view_type": res.view_type,
        "layers_engine": res.layers_engine,
        "used_learned": bool(use_learned),
        "n_frames": res.n_frames,
        "frame_size": [int(res.frame_size[0]), int(res.frame_size[1])],
        "fps": int(fps),
        "hold": int(hold),
        "duration_s": duration_s,
        "annotated_video": video_name,
        "annotated_video_bytes": int(video_bytes),
        "annotated_video_codec": "h264/mp4 (libx264, yuv420p)",
        "timelines_file": "timelines.json",
        "temporal_methods": [
            "optical_flow(Farneback 2003) belt speed + direction",
            "bytetrack(arXiv:2110.06864) blob-box association",
            "belt-footprint centroid drift (wander trend)",
            "content coverage over time",
            "foreign / object appear-disappear events",
        ],
        "metrics_summary": res.summary,
        "classes": list(CLASS_NAMES.values()),
    }
    if extra_meta:
        manifest.update(extra_meta)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
