"""User annotation -> region-of-interest masks and belt priors (guidance layer).

The guided pipeline lets a user draw regions on a frame and label them, then constrains
every detector to those regions. This module turns the drawn annotations into the boolean
masks and orientation priors the constrained detectors consume.

An ``annotation`` is a plain JSON dict::

    {"type": "freehand" | "polygon" | "rect" | "line",
     "points": [[x, y], ...], "label": str, "width": int (optional, line thickness)}

- ``freehand`` / ``polygon`` -> a filled polygon (``cv2.fillPoly``).
- ``rect`` -> a filled rectangle from two opposite corners, or a filled quad from 4 points.
- ``line`` -> a thick line stroke of ``width`` pixels (default 8).

Priors:
- :func:`belt_limit_prior` reads a labelled belt band (``expected-belt-limits`` /
  ``belt-section`` / ...) and returns a band mask, a guide orientation (deg from the
  x-axis, ``[0, 180)``) and the two guide edge lines.
- :func:`orientation_band` returns the ``(theta_center_deg, theta_band_deg)`` orientation
  gate a constrained Hough / RANSAC uses: for a ``top`` / ``end`` view the belt axis near
  the annotated / detected orientation, for a ``lateral`` view the near-horizontal long
  edges, falling back to a PCA (of the annotation points) or a Radon estimate (of the
  image) when no belt-limit annotation is drawn.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Labels that mean "this region delimits the belt" (the belt-limit prior source).
BELT_LIMIT_LABELS = frozenset(
    {"expected-belt-limits", "belt-limits", "belt-limit", "belt-section", "belt"}
)
_DEFAULT_LINE_WIDTH = 8


def _hw(shape: Any) -> tuple[int, int]:
    """Extract ``(H, W)`` from a shape tuple or an array-like ``shape`` attribute."""
    s = tuple(shape.shape) if hasattr(shape, "shape") else tuple(shape)
    if len(s) < 2:
        raise ValueError(f"shape must carry at least (H, W), got {s}")
    return int(s[0]), int(s[1])


def _points(ann: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(ann.get("points", []), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1:
        raise ValueError(f"annotation points must be an (N, 2) array, got shape {pts.shape}")
    return pts


def _rasterize_one(ann: dict[str, Any], h: int, w: int) -> np.ndarray:
    """Rasterize a single annotation to a boolean ``(H, W)`` mask."""
    import cv2

    ann_type = str(ann.get("type", "polygon")).lower()
    pts = _points(ann)
    mask = np.zeros((h, w), dtype=np.uint8)
    ipts = np.round(pts).astype(np.int32)

    if ann_type == "line":
        width = int(ann.get("width", _DEFAULT_LINE_WIDTH))
        if ipts.shape[0] == 1:
            cv2.circle(mask, tuple(ipts[0]), max(1, width // 2), 255, -1)
        else:
            cv2.polylines(mask, [ipts], isClosed=False, color=255,
                          thickness=max(1, width), lineType=cv2.LINE_8)
    elif ann_type == "rect":
        if ipts.shape[0] == 2:
            (x0, y0), (x1, y1) = ipts
            cv2.rectangle(mask, (int(min(x0, x1)), int(min(y0, y1))),
                          (int(max(x0, x1)), int(max(y0, y1))), 255, -1)
        else:
            cv2.fillPoly(mask, [ipts], 255)
    else:  # freehand / polygon (and any unknown type) -> filled polygon
        if ipts.shape[0] >= 3:
            cv2.fillPoly(mask, [ipts], 255)
        else:
            cv2.polylines(mask, [ipts], isClosed=False, color=255,
                          thickness=_DEFAULT_LINE_WIDTH, lineType=cv2.LINE_8)
    return mask > 0


def rasterize(annotations: list[dict[str, Any]], shape: Any) -> np.ndarray:
    """Rasterize every annotation to a single combined boolean mask at frame resolution."""
    h, w = _hw(shape)
    mask = np.zeros((h, w), dtype=bool)
    for ann in annotations or []:
        mask |= _rasterize_one(ann, h, w)
    return mask


def combine_by_label(
    annotations: list[dict[str, Any]], shape: Any, labels: Any
) -> np.ndarray:
    """Boolean mask of the annotations whose ``label`` is in ``labels`` (str or iterable)."""
    wanted = {labels} if isinstance(labels, str) else set(labels)
    selected = [a for a in (annotations or []) if str(a.get("label", "")) in wanted]
    return rasterize(selected, shape)


def _pts_of(annotations: list[dict[str, Any]]) -> np.ndarray:
    """Concatenate the points of a set of annotations into one ``(N, 2)`` array."""
    chunks = [np.asarray(a.get("points", []), dtype=np.float64) for a in annotations]
    chunks = [c for c in chunks if c.ndim == 2 and c.shape[0] >= 1 and c.shape[1] == 2]
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _long_edge_angle(points: np.ndarray) -> float | None:
    """Orientation (deg, ``[0, 180)``) of the long side of the point set's min-area rect.

    Two points -> the segment direction. >=3 points -> the longest edge of the rotating-
    calipers oriented box (the belt axis), which is convention-free and robust to shape.
    """
    import cv2

    if points.shape[0] < 2:
        return None
    if points.shape[0] == 2:
        d = points[1] - points[0]
        return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    box = cv2.boxPoints(cv2.minAreaRect(np.round(points).astype(np.int32)))
    best_len = -1.0
    best_ang = 0.0
    for i in range(4):
        d = box[(i + 1) % 4] - box[i]
        length = float(np.hypot(d[0], d[1]))
        if length > best_len:
            best_len = length
            best_ang = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    return best_ang


def _mask_orientation(annotations: list[dict[str, Any]], shape: Any) -> float | None:
    """Belt-axis orientation from the rasterised region's pixels (robust to rect/quad corners)."""
    mask = rasterize(annotations, shape)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return _long_edge_angle(np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1))


def _long_sides(points: np.ndarray) -> list[list[list[float]]] | None:
    """The two long sides of the point set's oriented box, as ``[[x0,y0],[x1,y1]]`` lines."""
    import cv2

    if points.shape[0] < 3:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(np.round(points).astype(np.int32)))
    sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    lengths = [float(np.hypot((b - a)[0], (b - a)[1])) for a, b in sides]
    i = int(np.argmax(lengths[:2]))  # 0 or 1: pick the longer of the first two sides
    out = []
    for j in (i, i + 2):
        a, b = sides[j]
        out.append([[round(float(a[0]), 1), round(float(a[1]), 1)],
                    [round(float(b[0]), 1), round(float(b[1]), 1)]])
    return out


def belt_limit_prior(annotations: list[dict[str, Any]], shape: Any) -> dict[str, Any]:
    """Derive the belt-band prior from belt-limit-labelled annotations.

    Returns ``{found, mask, orientation_deg, edge_lines, source}``. ``edge_lines`` are the
    two guide edges (the drawn lines themselves when two lines were labelled, else the two
    long sides of the labelled region). ``orientation_deg`` is the belt axis in ``[0, 180)``.
    """
    import cv2

    h, w = _hw(shape)
    belt_anns = [a for a in (annotations or []) if str(a.get("label", "")) in BELT_LIMIT_LABELS]
    empty = {"found": False, "mask": np.zeros((h, w), dtype=bool),
             "orientation_deg": None, "edge_lines": None, "source": None}
    if not belt_anns:
        return empty

    lines = [a for a in belt_anns if str(a.get("type", "")).lower() == "line"]
    if len(lines) >= 2:
        # two guide edges: band = convex hull between them, orientation = box long side.
        pts = _pts_of(lines)
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        edges = [[[round(float(x), 1), round(float(y), 1)] for x, y in _points(ln)[:2]]
                 for ln in lines[:2]]
        return {"found": True, "mask": mask > 0,
                "orientation_deg": (round(_long_edge_angle(pts), 2)
                                    if _long_edge_angle(pts) is not None else None),
                "edge_lines": edges, "source": "two-guide-lines"}

    # region prior: filled labelled region, orientation + edges from its oriented box
    # (computed on the rasterised pixels so a 2-corner rect gives the belt axis, not a diagonal).
    mask = rasterize(belt_anns, shape)
    ys, xs = np.nonzero(mask)
    mask_pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    ang = _long_edge_angle(mask_pts)
    return {"found": True, "mask": mask,
            "orientation_deg": (round(ang, 2) if ang is not None else None),
            "edge_lines": _long_sides(mask_pts), "source": "region"}


def _view_default_angle(view_type: str | None) -> float:
    """The default belt axis for a view kind when no annotation / image estimate exists."""
    v = str(view_type or "").lower()
    if v.startswith("lateral") or v.startswith("side"):
        return 0.0     # a lateral view sees the belt long edges near-horizontal
    return 90.0        # top / end view: default near-vertical belt travel


def orientation_band(
    view_type: str | None,
    annotations: list[dict[str, Any]] | None,
    shape: Any = None,
    image: np.ndarray | None = None,
    default_band_deg: float = 22.5,
) -> tuple[float, float]:
    """Return the ``(theta_center_deg, theta_band_deg)`` orientation gate for detectors.

    ``theta_center_deg`` is the expected belt AXIS direction from the x-axis (``[0, 180)``);
    a constrained Hough / RANSAC keeps only lines whose direction is within
    ``+/- theta_band_deg`` of it. Priority: a belt-limit annotation orientation, else a PCA
    of all annotation points, else a Radon estimate of ``image``, else the view default
    (with a wider band, since it is only a guess).
    """
    anns = annotations or []
    center: float | None = None

    if anns:
        belt_anns = [a for a in anns if str(a.get("label", "")) in BELT_LIMIT_LABELS]
        src = belt_anns if belt_anns else anns
        if shape is not None:
            center = _mask_orientation(src, shape)  # rasterise -> oriented box (rect-safe)
        if center is None:
            center = _long_edge_angle(_pts_of(src))

    if center is None and image is not None:
        from .features import _radon_orientation

        try:
            import cv2

            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY) \
                if np.asarray(image).ndim == 3 else np.asarray(image)
            center = float(_radon_orientation(gray)[0])
        except Exception:  # noqa: BLE001 - a Radon failure just falls through to the default
            center = None

    if center is None:
        return _view_default_angle(view_type), float(max(default_band_deg, 35.0))
    return round(float(center) % 180.0, 2), float(default_band_deg)


__all__ = [
    "BELT_LIMIT_LABELS",
    "rasterize",
    "combine_by_label",
    "belt_limit_prior",
    "orientation_band",
]
