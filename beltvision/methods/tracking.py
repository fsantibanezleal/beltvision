"""Capability 5: tracking (classical, LIVE-SERVER).

- ``tracking.optical_flow`` (M23) - Farneback dense optical flow -> belt-speed and
  material-flow direction; also a cheap "is the belt moving/stopped" check. Needs two
  frames; pass ``prev_image`` for real consecutive frames. Absent one it demonstrates the
  computation on a known self-shift of the frame and labels the source honestly.
- ``tracking.bytetrack_associate`` (M24) - a ByteTrack-style associator that consumes
  detector boxes (the detector is the learned cost). Two-stage IoU association of high- and
  low-confidence boxes, with track birth/death and stable ids; state persists per tracker id
  across calls. With no boxes supplied it runs a small deterministic two-frame demo so the
  method always returns a real, non-empty result.

References: Farneback 2003 (SCIA); ByteTrack arXiv:2110.06864 (MIT).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._common import as_bgr, cap, result, timed
from .preprocess import apply_clahe_lab

_MAX_TRACKS = 64


# --- M23: Farneback optical flow -------------------------------------------------------
def optical_flow(
    image: Any, *, prev_image: Any | None = None, shift_px: int = 4, **_: Any
) -> dict[str, Any]:
    """Dense Farneback flow -> belt speed (px/frame), direction, and a moving/stopped flag."""
    import cv2

    curr = cv2.cvtColor(apply_clahe_lab(as_bgr(image)), cv2.COLOR_BGR2GRAY)
    h, w = curr.shape
    if prev_image is not None:
        prev = cv2.cvtColor(apply_clahe_lab(as_bgr(prev_image)), cv2.COLOR_BGR2GRAY)
        source = "consecutive-frames"
    else:
        # No previous frame: demonstrate the computation on a known vertical self-shift.
        prev = np.roll(curr, -int(shift_px), axis=0)
        source = f"self-shift-demo(dy={shift_px})"
    with timed() as t:
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        mean_dx, mean_dy = float(fx.mean()), float(fy.mean())
        speed = float(np.median(mag))
        direction = float(np.degrees(np.arctan2(mean_dy, mean_dx)))
        moving = bool(speed > 0.25)
    payload = {
        "shape": [int(h), int(w)],
        "source": source,
        "mean_flow_dx": round(mean_dx, 4),
        "mean_flow_dy": round(mean_dy, 4),
        "belt_speed_px_per_frame": round(speed, 4),
        "flow_direction_deg": round(direction, 3),
        "max_flow_magnitude": round(float(mag.max()), 4),
        "moving": moving,
        "calibration": "relative-px/frame (no fps or px_per_mm; m/s not fabricated)",
    }
    return result(
        "tracking.optical_flow", "tracking", "classical",
        "Farneback 2003 (SCIA) dense optical flow",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=False,
    )


# --- M24: ByteTrack-style associator ---------------------------------------------------
def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xyxy boxes."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


@dataclass
class _Track:
    track_id: int
    bbox: list[float]
    score: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0


@dataclass
class ByteTrackAssociator:
    """Two-stage IoU association of detector boxes into persistent tracks (ByteTrack-style)."""

    high_thresh: float = 0.5
    low_thresh: float = 0.1
    iou_thresh: float = 0.3
    max_age: int = 30
    tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 1
    n_frames: int = 0

    def _match(self, tracks: list[_Track], dets: np.ndarray) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        matches: list[tuple[int, int]] = []
        if not tracks or dets.shape[0] == 0:
            return matches, set(range(len(tracks))), set(range(dets.shape[0]))
        track_boxes = np.array([t.bbox for t in tracks], dtype=np.float64)
        iou = _iou(track_boxes, dets)
        used_t: set[int] = set()
        used_d: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for ti, di in order:
            ti, di = int(ti), int(di)
            if ti in used_t or di in used_d or iou[ti, di] < self.iou_thresh:
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)
        unmatched_t = set(range(len(tracks))) - used_t
        unmatched_d = set(range(dets.shape[0])) - used_d
        return matches, unmatched_t, unmatched_d

    def update(self, detections: np.ndarray) -> list[_Track]:
        """One frame: detections as an (N, 5) array of [x0, y0, x1, y1, score]."""
        self.n_frames += 1
        det = np.asarray(detections, dtype=np.float64).reshape(-1, 5) if len(detections) else np.zeros((0, 5))
        high = det[det[:, 4] >= self.high_thresh]
        low = det[(det[:, 4] < self.high_thresh) & (det[:, 4] >= self.low_thresh)]

        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        matches, unmatched_t, unmatched_d = self._match(self.tracks, high[:, :4])
        for ti, di in matches:
            tr = self.tracks[ti]
            tr.bbox = high[di, :4].tolist()
            tr.score = float(high[di, 4])
            tr.hits += 1
            tr.time_since_update = 0

        # Stage 2: recover lost tracks with low-confidence boxes.
        remaining = [self.tracks[i] for i in unmatched_t]
        m2, um_t2, _ = self._match(remaining, low[:, :4])
        for ri, di in m2:
            tr = remaining[ri]
            tr.bbox = low[di, :4].tolist()
            tr.score = float(low[di, 4])
            tr.hits += 1
            tr.time_since_update = 0

        for di in unmatched_d:  # birth from unmatched high-confidence detections
            self.tracks.append(
                _Track(track_id=self._next_id, bbox=high[di, :4].tolist(), score=float(high[di, 4]))
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks


# Persistent per-tracker state so successive calls continue the same ids.
_TRACKERS: dict[str, ByteTrackAssociator] = {}


def reset_tracker(tracker_id: str | None = None) -> None:
    """Clear persistent associator state (all trackers, or one)."""
    if tracker_id is None:
        _TRACKERS.clear()
    else:
        _TRACKERS.pop(tracker_id, None)


def _demo_frames() -> list[np.ndarray]:
    """Two deterministic frames of boxes translating rightward (a self-check for the associator)."""
    f0 = np.array([[10, 10, 40, 40, 0.9], [80, 20, 110, 50, 0.85], [50, 70, 78, 96, 0.3]], float)
    f1 = np.array([[16, 12, 46, 42, 0.92], [86, 22, 116, 52, 0.8], [56, 72, 84, 98, 0.35]], float)
    return [f0, f1]


def bytetrack_associate(
    image: Any,
    *,
    detections: Any | None = None,
    tracker_id: str = "default",
    persist: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Associate detector boxes into tracks (ByteTrack-style). The detector is the cost."""
    bgr = as_bgr(image)
    h, w = bgr.shape[:2]
    with timed() as t:
        if detections is not None:
            assoc = _TRACKERS.setdefault(tracker_id, ByteTrackAssociator()) if persist else ByteTrackAssociator()
            tracks = assoc.update(np.asarray(detections, dtype=np.float64))
            n_frames = assoc.n_frames
            source = "detector-boxes"
        else:
            assoc = ByteTrackAssociator()
            for frame in _demo_frames():
                tracks = assoc.update(frame)
            n_frames = assoc.n_frames
            source = "synthetic-boxes-demo (supply detections=... to track real detector output)"
    track_out = [
        {
            "track_id": int(tr.track_id),
            "bbox_xyxy": [round(float(v), 2) for v in tr.bbox],
            "score": round(float(tr.score), 4),
            "age": int(tr.age),
            "hits": int(tr.hits),
        }
        for tr in tracks
    ]
    payload = {
        "shape": [int(h), int(w)],
        "source": source,
        "n_frames_processed": int(n_frames),
        "n_tracks": len(track_out),
        "tracks": cap(track_out, _MAX_TRACKS),
        "note": "association is classical; the detector is the learned cost (see detection.onnx_detector)",
    }
    return result(
        "tracking.bytetrack_associate", "tracking", "classical",
        "ByteTrack arXiv:2110.06864 (MIT) association; detector-driven",
        payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True,
    )
