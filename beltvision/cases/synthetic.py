"""Deterministic synthetic belt scenes, WITH ground truth.

These are honest, labeled, offline stand-ins that let the pipeline run end-to-end
with no external or private data. They are NOT presented as real inference: any
manifest built from a synthetic case is tagged ``synthetic`` in its source/license
fields so the web app can label it as such.

Two generators live here:

- :func:`synth_belt_scene` - the original near-vertical dusty belt frame, kept for
  backward compatibility with older tests/consumers.
- :func:`synth_scene` - a richer generator that emits a :class:`SyntheticScene`
  carrying the EXACT ground truth alongside the frame: the 4-class label map, the
  belt mask, the two belt edges, the medial-axis centreline, the belt orientation
  angle, the support-structure axis (for misalignment), injected damage regions and
  injected foreign-object boxes. Belts are generated at ARBITRARY orientation and an
  optional curvature so the geometry chain is validated against known geometry at
  vertical, horizontal, diagonal and curved paths (an all-vertical suite tests
  nothing and lets orientation bugs survive).

The 4 semantic classes are the engine-wide vocabulary:
``EXTERNAL=0`` (structure/frame/mesh/background), ``BELT=1`` (rubber surface),
``CONTENT=2`` (ore/material on the belt), ``FOREIGN=3`` (tramp objects).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.rng import seeded_rng

# Frozen 4-class vocabulary shared by the generator, the semantic method and the app.
EXTERNAL, BELT, CONTENT, FOREIGN = 0, 1, 2, 3
CLASS_NAMES = {EXTERNAL: "external", BELT: "belt", CONTENT: "content", FOREIGN: "foreign"}


def synth_belt_scene(
    seed: int = 34,
    size: tuple[int, int] = (256, 384),
    dust: float = 0.35,
    with_tear: bool = False,
) -> np.ndarray:
    """Return a deterministic BGR uint8 belt scene (legacy near-vertical generator)."""
    rng = seeded_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    base = 60.0 + 30.0 * (yy / max(h - 1, 1))
    scene = np.stack([base, base * 0.98, base * 0.95], axis=-1)
    for edge_x in (int(0.18 * w), int(0.82 * w)):
        rail = np.exp(-((xx - edge_x) ** 2) / (2 * (0.012 * w) ** 2))
        scene += rail[..., None] * np.array([70.0, 70.0, 75.0])
    n_rocks = 40
    for _ in range(n_rocks):
        cx = rng.uniform(0.22 * w, 0.78 * w)
        cy = rng.uniform(0.05 * h, 0.95 * h)
        rx = rng.uniform(4.0, 16.0)
        ry = rx * rng.uniform(0.6, 1.2)
        val = rng.uniform(90.0, 180.0)
        blob = np.exp(-(((xx - cx) ** 2) / (2 * rx**2) + ((yy - cy) ** 2) / (2 * ry**2)))
        scene += blob[..., None] * np.array([val, val * 0.95, val * 0.9])
    mesh = 6.0 * (np.sin(xx / 9.0) * np.sin(yy / 9.0))
    scene += mesh[..., None]
    if with_tear:
        line = np.abs((xx - 0.5 * w) - 0.4 * (yy - 0.5 * h))
        streak = np.exp(-(line**2) / (2 * (0.01 * w) ** 2))
        scene += streak[..., None] * np.array([120.0, 120.0, 140.0])
    noise = rng.normal(0.0, 6.0, size=(h, w, 1)).astype(np.float32)
    scene = (1.0 - dust) * scene + dust * 150.0 + noise
    return np.clip(scene, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class SyntheticScene:
    """A synthetic frame plus its exact ground truth (for numeric validation)."""

    image: np.ndarray                 # (H, W, 3) uint8 BGR
    label_map: np.ndarray             # (H, W) int in {0,1,2,3}
    belt_mask: np.ndarray             # (H, W) bool
    content_mask: np.ndarray          # (H, W) bool
    foreign_mask: np.ndarray          # (H, W) bool
    orientation_deg: float            # belt axis angle from the x-axis, [0,180)
    curvature: float                  # 0 => straight; !=0 => curved path
    centreline: np.ndarray            # (N, 2) float (x, y) GT medial axis
    edge_a: np.ndarray                # (N, 2) float (x, y) one long border
    edge_b: np.ndarray                # (N, 2) float (x, y) other long border
    support_axis_deg: float           # supporting-structure axis angle (misalignment ref)
    misalignment_deg: float           # GT belt_axis - support_axis
    damage_boxes: list[tuple[int, int, int, int]]  # xywh injected damage
    foreign_boxes: list[tuple[int, int, int, int]]  # xywh injected foreign objects
    loaded: bool
    meta: dict[str, Any] = field(default_factory=dict)


def _unit(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (direction, normal) unit vectors for an axis angle in degrees (x-axis ref)."""
    th = np.radians(theta_deg)
    d = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
    n = np.array([-np.sin(th), np.cos(th)], dtype=np.float64)
    return d, n


def _centreline_points(
    cx: float, cy: float, direction: np.ndarray, normal: np.ndarray,
    half_len: float, curvature: float, n: int = 60,
) -> np.ndarray:
    """Sample a centreline: straight along ``direction``, bent by ``curvature`` along normal."""
    t = np.linspace(-half_len, half_len, n)
    bend = curvature * (t**2 - half_len**2 / 3.0)  # zero-mean quadratic bow
    pts = (
        np.array([cx, cy])[None, :]
        + t[:, None] * direction[None, :]
        + bend[:, None] * normal[None, :]
    )
    return pts


def _rocky_texture(rng: np.random.Generator, h: int, w: int, base: np.ndarray) -> np.ndarray:
    """A cellular ore-like texture (BGR float), reddish-grey with fragment structure."""
    import cv2

    tex = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 2.0)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-6)
    frag = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    frag = cv2.GaussianBlur(frag, (0, 0), 5.0)
    combined = 0.6 * tex + 0.4 * (frag > 0.15).astype(np.float32)
    out = base[None, None, :] + combined[..., None] * np.array([28.0, 40.0, 60.0])
    return out


def synth_scene(
    seed: int = 34,
    *,
    orientation_deg: float = 90.0,
    curvature: float = 0.0,
    loaded: bool = False,
    with_damage: bool = False,
    with_foreign: bool = False,
    support_offset_deg: float = 0.0,
    dust: float = 0.25,
    size: tuple[int, int] = (300, 400),
    belt_halfwidth_frac: float = 0.26,
) -> SyntheticScene:
    """Generate a labeled synthetic belt scene at an arbitrary orientation, with GT.

    Parameters
    ----------
    orientation_deg:
        Belt axis angle from the x-axis. ``90`` is a near-vertical belt, ``0`` a
        horizontal belt, ``30``/``45`` diagonals.
    curvature:
        Bow of the centreline along its normal (``0`` => straight, ~``1e-3`` => a
        visibly curved path).
    loaded:
        If True, fill the belt interior with ore texture (a top-carrying load); else
        the belt is an empty return strand (mineral coverage ~0).
    with_damage / with_foreign:
        Inject a longitudinal rip + hole / a bright foreign object on the belt.
    support_offset_deg:
        Angle of the supporting structure relative to the belt. The GT misalignment
        is ``-support_offset_deg`` (belt parallel to its support => aligned).
    """
    import cv2

    rng = seeded_rng(seed)
    h, w = size
    cx, cy = w * 0.5, h * 0.5
    direction, normal = _unit(orientation_deg)
    half_len = 0.62 * max(h, w)
    half_w = belt_halfwidth_frac * min(h, w)

    # --- external background: structured rocky clutter ---
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg_base = np.array([95.0, 100.0, 105.0])
    scene = _rocky_texture(rng, h, w, bg_base).astype(np.float32)
    scene += (12.0 * np.sin(xx / 7.0) * np.sin(yy / 7.0))[..., None]  # faint mesh

    # --- belt band via a thick polyline of the centreline (arbitrary shape) ---
    centre = _centreline_points(cx, cy, direction, normal, half_len, curvature)
    belt_mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(centre).astype(np.int32)
    cv2.polylines(belt_mask, [poly], False, 255, thickness=int(2 * half_w), lineType=cv2.LINE_8)
    belt_mask = belt_mask > 0

    # Belt surface: darker, smoother rubber (BGR) with a clear longitudinal weave along the
    # belt direction (an oriented/coherent texture - the opposite of a random material pile).
    belt_rgb = np.array([70.0, 68.0, 66.0]) + rng.normal(0.0, 2.0, size=(h, w, 1)).astype(np.float32)
    weave = 13.0 * np.sin((xx * direction[0] + yy * direction[1]) / 3.5)
    belt_rgb = belt_rgb + weave[..., None]
    scene = np.where(belt_mask[..., None], belt_rgb, scene)

    # Empty return strand: along-axis dust/water streaks (the return-strand signature). These
    # are a strongly ORIENTED texture (unlike a random material pile), so the view classifier
    # can tell an empty streaked belt from a loaded one.
    if not loaded:
        streaks = np.zeros((h, w), dtype=np.float32)
        for _ in range(22):
            off = rng.uniform(-half_w * 0.85, half_w * 0.85)
            base = np.array([cx, cy]) + off * normal
            p0 = (base - half_len * direction).astype(int)
            p1 = (base + half_len * direction).astype(int)
            cv2.line(streaks, tuple(p0), tuple(p1), float(rng.uniform(40, 80)), 1, cv2.LINE_AA)
        streaks[~belt_mask] = 0.0
        scene = scene + streaks[..., None]

    # --- mineral load on the belt (top-carrying) ---
    content_mask = np.zeros((h, w), dtype=bool)
    if loaded:
        core = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(core, [poly], False, 255, thickness=int(2 * half_w * 0.78), lineType=cv2.LINE_8)
        content_mask = (core > 0)
        ore = _rocky_texture(rng, h, w, np.array([55.0, 78.0, 120.0]))
        scene = np.where(content_mask[..., None], ore, scene)

    # --- injected damage (longitudinal rip + a hole) on the belt ---
    damage_boxes: list[tuple[int, int, int, int]] = []
    if with_damage:
        # rip: a thin dark broken line offset from the centre, along the belt.
        rip_centre = _centreline_points(
            cx + normal[0] * half_w * 0.4, cy + normal[1] * half_w * 0.4,
            direction, normal, half_len * 0.55, curvature,
        )
        rip = np.round(rip_centre).astype(np.int32)
        dmg = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(dmg, [rip], False, 255, thickness=3, lineType=cv2.LINE_8)
        dmg = dmg > 0
        scene[dmg] = np.array([25.0, 24.0, 22.0])
        ys, xs = np.nonzero(dmg)
        damage_boxes.append((int(xs.min()), int(ys.min()),
                             int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)))
        # hole: a dark ellipse on the belt.
        hole_c = (centre[len(centre) // 3]).astype(int)
        cv2.ellipse(scene, (int(hole_c[0]), int(hole_c[1])), (10, 7),
                    float(orientation_deg), 0, 360, (18, 18, 16), -1)
        damage_boxes.append((int(hole_c[0] - 11), int(hole_c[1] - 8), 22, 16))

    # --- injected foreign object on the belt ---
    foreign_mask = np.zeros((h, w), dtype=bool)
    foreign_boxes: list[tuple[int, int, int, int]] = []
    if with_foreign:
        fc = (centre[2 * len(centre) // 3]).astype(int)
        fx, fy = int(fc[0]), int(fc[1])
        fw2, fh2 = 22, 12
        cv2.rectangle(scene, (fx - fw2, fy - fh2), (fx + fw2, fy + fh2), (60, 170, 205), -1)
        foreign_mask[fy - fh2:fy + fh2, fx - fw2:fx + fw2] = True
        foreign_mask &= belt_mask
        foreign_boxes.append((fx - fw2, fy - fh2, 2 * fw2, 2 * fh2))

    # --- supporting structure (idler/frame line), the misalignment reference ---
    support_axis_deg = float((orientation_deg + support_offset_deg) % 180.0)
    s_dir, s_norm = _unit(support_axis_deg)
    for side in (-1.0, 1.0):
        base_pt = np.array([cx, cy]) + side * (half_w + 16.0) * normal
        s_line = _centreline_points(base_pt[0], base_pt[1], s_dir, s_norm, half_len * 0.9, 0.0)
        sp = np.round(s_line).astype(np.int32)
        # a distinct dark structural beam (frame/stringer) - the misalignment reference
        cv2.polylines(scene, [sp], False, (38, 40, 46), thickness=4, lineType=cv2.LINE_8)

    # dust haze
    noise = rng.normal(0.0, 4.0, size=(h, w, 1)).astype(np.float32)
    scene = (1.0 - dust) * scene + dust * 150.0 + noise
    image = np.clip(scene, 0, 255).astype(np.uint8)

    # label map with priority FOREIGN > CONTENT > BELT > EXTERNAL
    label_map = np.zeros((h, w), dtype=np.int32)
    label_map[belt_mask] = BELT
    label_map[content_mask] = CONTENT
    label_map[foreign_mask] = FOREIGN

    # GT edges: offset the centreline by +/- half_w along the local normal.
    edge_a = centre + half_w * normal[None, :]
    edge_b = centre - half_w * normal[None, :]

    return SyntheticScene(
        image=image,
        label_map=label_map,
        belt_mask=belt_mask,
        content_mask=content_mask,
        foreign_mask=foreign_mask,
        orientation_deg=float(orientation_deg % 180.0),
        curvature=float(curvature),
        centreline=centre.astype(np.float64),
        edge_a=edge_a.astype(np.float64),
        edge_b=edge_b.astype(np.float64),
        support_axis_deg=support_axis_deg,
        misalignment_deg=float(-support_offset_deg),
        damage_boxes=damage_boxes,
        foreign_boxes=foreign_boxes,
        loaded=bool(loaded),
        meta={
            "seed": int(seed),
            "size": [int(h), int(w)],
            "belt_halfwidth_px": float(half_w),
        },
    )


# A small, fixed validation suite spanning orientations + a curved path, so the
# geometry chain is exercised at vertical, horizontal, diagonal and curved belts.
GT_SUITE: dict[str, dict[str, Any]] = {
    "vertical_empty": dict(orientation_deg=90.0, loaded=False, with_damage=True),
    "horizontal_loaded": dict(orientation_deg=2.0, loaded=True, with_foreign=True),
    "diag30_loaded": dict(orientation_deg=30.0, loaded=True, with_foreign=True),
    "diag45_empty": dict(orientation_deg=45.0, loaded=False, with_damage=True),
    "curved_loaded": dict(orientation_deg=75.0, curvature=1.1e-3, loaded=True),
    "misaligned_side": dict(orientation_deg=88.0, loaded=False, support_offset_deg=9.0),
}


def gt_scene(name: str, seed: int = 34) -> SyntheticScene:
    """Build one named scene from :data:`GT_SUITE`."""
    if name not in GT_SUITE:
        raise KeyError(f"unknown GT scene {name!r}; known: {sorted(GT_SUITE)}")
    return synth_scene(seed=seed, **GT_SUITE[name])
