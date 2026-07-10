"""Deterministic synthetic belt scenes.

These are honest, labeled, offline stand-ins that let the pipeline run end-to-end
with no external or private data. They are NOT presented as real inference: any
manifest built from a synthetic case is tagged ``synthetic`` in its source/license
fields so the web app can label it as such.

The scene mimics the visual signatures the pipeline cares about: a belt with two
rubber edges, crushed-ore fragments, a protective mesh, and dust haze. An optional
scripted "tear" gives the anomaly stages a known target.
"""
from __future__ import annotations

import numpy as np

from ..core.rng import seeded_rng


def synth_belt_scene(
    seed: int = 34,
    size: tuple[int, int] = (256, 384),
    dust: float = 0.35,
    with_tear: bool = False,
) -> np.ndarray:
    """Return a deterministic BGR uint8 belt scene.

    Parameters
    ----------
    seed:
        Seeds every random draw so the frame is reproducible.
    size:
        (height, width) in pixels.
    dust:
        0..1 haze strength; higher washes out contrast (the COLA 34 regime).
    with_tear:
        If True, paint a diagonal light streak simulating a longitudinal tear.
    """
    rng = seeded_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Base belt: a dark rubber surface with a gentle top-to-bottom lighting gradient.
    base = 60.0 + 30.0 * (yy / max(h - 1, 1))
    scene = np.stack([base, base * 0.98, base * 0.95], axis=-1)  # BGR-ish, cool grey

    # Belt edges: two brighter near-vertical rubber rails.
    for edge_x in (int(0.18 * w), int(0.82 * w)):
        rail = np.exp(-((xx - edge_x) ** 2) / (2 * (0.012 * w) ** 2))
        scene += (rail[..., None] * np.array([70.0, 70.0, 75.0]))

    # Crushed-ore fragments: bright ellipses of varying size scattered on the belt.
    n_rocks = 40
    for _ in range(n_rocks):
        cx = rng.uniform(0.22 * w, 0.78 * w)
        cy = rng.uniform(0.05 * h, 0.95 * h)
        rx = rng.uniform(4.0, 16.0)
        ry = rx * rng.uniform(0.6, 1.2)
        val = rng.uniform(90.0, 180.0)
        blob = np.exp(-(((xx - cx) ** 2) / (2 * rx**2) + ((yy - cy) ** 2) / (2 * ry**2)))
        scene += blob[..., None] * np.array([val, val * 0.95, val * 0.9])

    # Protective mesh: faint periodic grid overlay.
    mesh = 6.0 * (np.sin(xx / 9.0) * np.sin(yy / 9.0))
    scene += mesh[..., None]

    # Optional scripted tear: a diagonal bright streak across the belt.
    if with_tear:
        line = np.abs((xx - 0.5 * w) - 0.4 * (yy - 0.5 * h))
        streak = np.exp(-(line**2) / (2 * (0.01 * w) ** 2))
        scene += streak[..., None] * np.array([120.0, 120.0, 140.0])

    # Dust haze: lift the black point and add low-frequency noise, cutting contrast.
    noise = rng.normal(0.0, 6.0, size=(h, w, 1)).astype(np.float32)
    scene = (1.0 - dust) * scene + dust * 150.0 + noise

    return np.clip(scene, 0, 255).astype(np.uint8)
