"""Temporal / dynamic sequence-video engine tests.

The per-frame ladder + temporal analysis (:func:`analyze_sequence`) is pure classical
(numpy + opencv), so it always runs. Video ENCODING needs ``imageio``/``imageio-ffmpeg``
(the precompute lane), so those tests ``importorskip`` and a slim venv skips them cleanly.

The synthetic sequence is a bright belt strand on a dark frame with a small bright blob that
translates a few pixels per frame: this exercises optical-flow belt speed, blob detection and
ByteTrack association, and the centroid drift trend deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

from beltvision.precompute.dynamic import analyze_sequence, precompute_sequence


def _synthetic_sequence(n: int = 3, size: tuple[int, int] = (96, 128), shift: int = 4):
    """``n`` frames from a wide textured canvas viewed through a window that slides right.

    A smooth moving texture (locks optical flow) carries several discrete bright fragments
    (ore-like blobs the detector/tracker follow); sliding the window by ``shift`` px per frame
    translates the whole scene, so belt speed ~= ``shift`` and the fragments track."""
    h, w = size
    rng = np.random.default_rng(0)
    canvas = np.zeros((h, w + n * shift + 8), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0 : canvas.shape[1]].astype(np.float32)
    canvas += 100 + 30 * np.sin(xx / 9.0) + 20 * np.cos(yy / 7.0)  # smooth moving texture
    canvas += rng.normal(0, 6, canvas.shape)                       # mild grain locks the flow
    for fx, fy in [(18, 22), (40, 60), (66, 34), (92, 68), (120, 26), (150, 54)]:
        canvas[fy - 4 : fy + 4, fx - 4 : fx + 4] = 245             # discrete bright fragments
    canvas = np.clip(canvas, 0, 255)
    frames = []
    for i in range(n):
        win = canvas[:, i * shift : i * shift + w]
        frames.append(np.repeat(win[:, :, None], 3, axis=2).astype(np.uint8))
    return frames


def test_analyze_sequence_shapes_and_timelines():
    frames = _synthetic_sequence(n=3)
    res = analyze_sequence(frames, view_type="end_return", seed=34)
    assert res.n_frames == 3
    assert len(res.annotated_frames) == 3
    for a in res.annotated_frames:
        assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
    # every timeline series has one value per frame
    for key in ("t", "belt_speed_px_per_frame", "flow_direction_deg", "coverage_pct",
                "drift_px", "center_offset_frac", "axis_angle_deg", "track_count",
                "n_foreign", "haze", "moving"):
        assert key in res.timelines, key
        assert len(res.timelines[key]) == 3, key
    assert res.timelines["t"] == [0, 1, 2]
    # first-frame speed is 0 (no previous frame); later frames measure real motion
    assert res.timelines["belt_speed_px_per_frame"][0] == 0.0
    assert res.timelines["belt_speed_px_per_frame"][1] >= 0.0
    # summary is JSON-safe scalars
    assert res.summary["n_frames"] == 3
    assert "belt_speed_mean_px_per_frame" in res.summary
    assert res.summary["calibration"].startswith("relative")


def test_analyze_sequence_is_deterministic():
    frames = _synthetic_sequence(n=4)
    a = analyze_sequence(frames, view_type="end_return", seed=34)
    b = analyze_sequence(frames, view_type="end_return", seed=34)
    assert a.timelines == b.timelines
    assert a.events == b.events
    assert a.summary == b.summary


def test_analyze_sequence_requires_two_frames():
    with pytest.raises(ValueError, match="at least 2 frames"):
        analyze_sequence(_synthetic_sequence(n=1))


def test_tracking_and_flow_are_real():
    # clearly-moving fragments are detected + associated into tracks, and the dense flow
    # recovers the belt speed (~= the window shift of 4 px/frame)
    frames = _synthetic_sequence(n=5, shift=4)
    res = analyze_sequence(frames, view_type="top_carrying", seed=34)
    assert max(res.timelines["track_count"]) >= 1
    moving_speeds = res.timelines["belt_speed_px_per_frame"][1:]
    assert max(moving_speeds) > 1.0                 # real motion recovered
    assert 2.0 < res.summary["belt_speed_max_px_per_frame"] < 8.0
    assert any(e["type"] == "object_appear" for e in res.events)


def test_precompute_sequence_writes_artifacts(tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    frames = _synthetic_sequence(n=4)
    out = tmp_path / "case"
    manifest = precompute_sequence(
        frames, out, case_id="unit_seq", view_type="end_return", seed=34, fps=8, hold=2
    )
    assert (out / "annotated.mp4").is_file()
    assert (out / "timelines.json").is_file()
    assert (out / "manifest.json").is_file()
    assert manifest["annotated_video_bytes"] > 0
    assert manifest["n_frames"] == 4
    assert manifest["engine"] == "beltvision"
    assert "optical_flow" in manifest["temporal_methods"][0]
    # the encoded mp4 is decodable and compact
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(out / "annotated.mp4"))
    n_out = sum(1 for _ in reader.iter_data())
    assert n_out == 4 * 2  # n_frames * hold
