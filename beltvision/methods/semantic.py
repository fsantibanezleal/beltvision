"""The backbone: 4-class semantic segmentation of a conveyor-belt scene.

Every downstream analysis derives from these layers, so this is computed first:

- ``EXTERNAL`` (0) - structure, frame, chute, mesh screen, background, environment.
- ``BELT``     (1) - the belt surface itself (empty rubber or the carrier under a load).
- ``CONTENT``  (2) - the ore/material carried on the belt.
- ``FOREIGN``  (3) - tramp metal, wood, tools, anything that is not belt or ore.

Two engines produce the labelled map, and the method reports which one ran:

1. ``classical-prior`` (always available, both venvs): a per-pixel colour + texture +
   spatial prior over the 4 classes. It is dust-robust (CLAHE-first), needs no weight,
   and is the live-thin tier that fits the VPS.
2. ``open-vocab`` (heavy/offline lane): MobileSAM automatic masks give crisp,
   class-agnostic regions; each region is labelled by (a) an open-vocabulary CLIP
   zero-shot score against the four class prompts when ``transformers`` is present, or
   (b) the classical prior's majority vote inside the region otherwise. This yields a
   crisp, object-aware 4-class map and is what the precompute lane commits per case.

The two fuse: when SAM regions are available they crispen the classical prior's
boundaries; uncovered pixels keep the prior's argmax. The result is never
``weights_absent`` - absent the learned stack it degrades to the classical prior and
says so in ``engine``.

References: Kirillov et al. 2023 (SAM); MobileSAM (Tiny-ViT, Apache-2.0); Radford et al.
2021 (CLIP); Grounding-DINO arXiv:2303.05499 (open-vocabulary detection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..cases.synthetic import BELT, CLASS_NAMES, CONTENT, EXTERNAL, FOREIGN
from ._common import as_bgr, result, timed
from .preprocess import apply_clahe_lab

_WORK = 384  # working long-side for the per-pixel prior (upsampled back to full res)

# CLIP text prompts for the open-vocabulary per-region labeller (class -> prompts). The
# tool is a GENERAL conveyor-belt inspector, so CONTENT = the transported material of ANY
# domain. Domain-specific content prompt sets are selectable via ``content_prompts=``;
# the default set spans mining, aggregates, food, packages, recycling and luggage.
CONTENT_PROMPT_SETS: dict[str, list[str]] = {
    "general": ["a pile of transported material on a conveyor belt", "bulk material",
                "boxes and packages", "food product", "aggregate or gravel",
                "recycling or waste", "luggage"],
    "mining": ["a pile of iron ore", "crushed rock and mineral", "ore material on a belt"],
    "aggregate": ["gravel", "crushed stone", "sand and aggregate"],
    "food": ["food product", "grain", "fruit or vegetables"],
    "packages": ["cardboard boxes", "parcels and packages", "luggage"],
    "recycling": ["recycling material", "mixed waste", "plastic and paper"],
}

CLIP_PROMPTS: dict[int, list[str]] = {
    EXTERNAL: ["steel structure", "metal mesh screen", "concrete wall", "background",
               "machine frame", "chute"],
    BELT: ["a black rubber conveyor belt surface", "empty conveyor belt", "bare belt"],
    CONTENT: CONTENT_PROMPT_SETS["general"],
    FOREIGN: ["a piece of wood", "a metal tool", "a foreign object", "a rag or debris"],
}

CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    EXTERNAL: (90, 90, 90),      # grey
    BELT: (60, 180, 75),         # green
    CONTENT: (40, 120, 235),     # orange
    FOREIGN: (200, 60, 200),     # magenta
}


@dataclass
class Layers:
    """A computed 4-class segmentation at full frame resolution."""

    label_map: np.ndarray                    # (H, W) int in {0,1,2,3}
    engine: str                              # "classical-prior" | "open-vocab(...)"
    coverage: dict[str, float]               # class name -> fraction of frame
    n_regions: int = 0
    scores: dict[str, Any] = field(default_factory=dict)

    def mask(self, cls: int) -> np.ndarray:
        return self.label_map == cls

    @property
    def belt_mask(self) -> np.ndarray:
        return self.label_map == BELT

    @property
    def content_mask(self) -> np.ndarray:
        return self.label_map == CONTENT

    @property
    def foreign_mask(self) -> np.ndarray:
        return self.label_map == FOREIGN


def _work_resize(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    import cv2

    h, w = bgr.shape[:2]
    scale = _WORK / float(max(h, w))
    if scale >= 1.0:
        return bgr, 1.0
    small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def _features(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel colour + texture features on a CLAHE-first frame."""
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum, a_ch, b_ch = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # local texture energy: std of gradient over a window
    k = 11
    mean = cv2.blur(grad, (k, k))
    sq = cv2.blur(grad * grad, (k, k))
    tex = np.sqrt(np.clip(sq - mean * mean, 0, None))
    # structure-tensor coherence: ~1 where texture is ORIENTED (belt streaks / straight
    # edges), ~0 where it is isotropic (random rock, periodic mesh). This separates the
    # belt strand from the cluttered rock/mesh far better than raw texture energy.
    jxx = cv2.blur(gx * gx, (k, k))
    jyy = cv2.blur(gy * gy, (k, k))
    jxy = cv2.blur(gx * gy, (k, k))
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-3)
    sat = np.sqrt(a_ch * a_ch + b_ch * b_ch)  # colour saturation in LAB
    warmth = a_ch + 0.4 * b_ch                # reddish/orange material is positive
    return {"L": lum, "A": a_ch, "B": b_ch, "grad": grad, "tex": tex, "coh": coh,
            "sat": sat, "warmth": warmth}


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _classical_scores(bgr: np.ndarray, view_type: str | None) -> np.ndarray:
    """Return an (H, W, 4) soft score stack over {external,belt,mineral,foreign}."""
    f = _features(bgr)
    h, w = bgr.shape[:2]
    tex_n = _norm(f["tex"])
    warm_n = _norm(f["warmth"])
    sat_n = _norm(f["sat"])
    lum_n = _norm(f["L"])
    coh_n = _norm(f["coh"])
    isotropic = tex_n * (1.0 - coh_n)   # random/periodic clutter (rock, mesh)

    # spatial prior: distance from the frame border (0 at border, 1 at centre)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
    centrality = np.clip(1.0 - r, 0.0, 1.0)
    border = 1.0 - centrality

    # CONTENT: warm, saturated, ISOTROPICALLY textured (a pile of material), central.
    s_min = 0.6 * warm_n + 0.25 * sat_n + 0.3 * isotropic
    # BELT: neutral colour (low saturation), a COHERENT/oriented surface (streaks or smooth
    # rubber, not random clutter), central. Coherence is what picks the strand out of rock.
    s_belt = 0.45 * (1.0 - sat_n) + 0.35 * coh_n + 0.3 * centrality - 0.4 * warm_n - 0.5 * isotropic
    # EXTERNAL: isotropic clutter (rock/mesh) OR peripheral OR extreme brightness.
    s_ext = 0.6 * isotropic + 0.35 * border + 0.2 * np.abs(lum_n - 0.5) * 2.0
    # FOREIGN: colour/brightness outlier - bright & saturated & not warm (bluish/greenish).
    cool = _norm(-f["warmth"])
    s_for = 0.5 * cool * sat_n + 0.4 * np.clip(lum_n - 0.7, 0, None) * 2.0
    s_for *= 0.5  # conservative: only strong outliers become foreign

    # view-informed priors
    if view_type == "top_carrying":
        s_min += 0.4 * centrality
        s_belt -= 0.1
    elif view_type == "end_return":
        s_belt += 0.35 * centrality + 0.25 * coh_n
        s_min -= 0.6    # empty return strand: strongly suppress content
        s_for -= 0.2
    elif view_type == "side_profile":
        s_min += 0.15 * centrality

    stack = np.stack([s_ext, s_belt, s_min, s_for], axis=-1).astype(np.float32)
    # softmax over classes for a probability-like stack
    stack -= stack.max(axis=-1, keepdims=True)
    ex = np.exp(stack)
    return ex / (ex.sum(axis=-1, keepdims=True) + 1e-9)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask."""
    import cv2

    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def _clean_labels(scores: np.ndarray, view_type: str | None) -> np.ndarray:
    """Argmax the score stack, then tidy the belt into one coherent region."""
    import cv2

    labels = scores.argmax(axis=-1).astype(np.int32)
    # morphological tidy per class
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cls in (BELT, CONTENT):
        m = (labels == cls).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        labels[(labels == cls) & (m == 0)] = EXTERNAL
        labels[(m > 0)] = cls
    # keep the belt as its single largest region (a belt is one connected strand)
    belt = labels == BELT
    if belt.sum() > 0:
        big = _largest_component(belt)
        labels[belt & ~big] = EXTERNAL
    # mineral only where it sits within/over the belt region context (dilated belt)
    return labels


def _sam_regions(bgr: np.ndarray) -> list[np.ndarray] | None:
    """MobileSAM automatic masks (list of bool masks) or None if the stack is absent."""
    from ..models import ensure_weight

    path = ensure_weight("mobile_sam")
    if path is None:
        return None
    try:
        from ultralytics import SAM
    except Exception:
        return None
    try:
        model = SAM(str(path))
        res = model(bgr[:, :, ::-1], verbose=False)
        out: list[np.ndarray] = []
        for r in res:
            if r.masks is None:
                continue
            data = r.masks.data.cpu().numpy()
            for m in data:
                out.append(m > 0.5)
        return out or None
    except Exception:
        return None


def _clip_label_regions(
    bgr: np.ndarray, regions: list[np.ndarray], prior: np.ndarray
) -> tuple[np.ndarray, str]:
    """Label each SAM region by CLIP zero-shot (if available) else by the prior's vote."""
    labels = prior.argmax(axis=-1).astype(np.int32)
    engine = "open-vocab(MobileSAM+prior-vote)"
    clip = _try_clip()
    if clip is not None:
        engine = "open-vocab(MobileSAM+CLIP)"
    h, w = labels.shape
    for m in sorted(regions, key=lambda z: int(z.sum())):  # small first, big overwrite
        if m.shape != (h, w):
            import cv2

            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        area = int(m.sum())
        if area < 0.002 * h * w:
            continue
        cls = _classify_region(bgr, m, prior, clip)
        labels[m] = cls
    return labels, engine


_CLIP_CACHE: dict[str, Any] = {}


def _try_clip():
    if "clip" in _CLIP_CACHE:
        return _CLIP_CACHE["clip"]
    obj = None
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor

        name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(name)
        proc = CLIPProcessor.from_pretrained(name)
        model.eval()
        prompts, index = [], []
        for cls, texts in CLIP_PROMPTS.items():
            for t in texts:
                prompts.append(t)
                index.append(cls)
        obj = {"model": model, "proc": proc, "prompts": prompts, "index": np.array(index)}
    except Exception:
        obj = None
    _CLIP_CACHE["clip"] = obj
    return obj


def _classify_region(bgr, mask, prior, clip) -> int:
    """Return the class id for a SAM region."""
    if clip is None:
        # prior majority vote within the region
        votes = prior[mask].mean(axis=0)
        return int(np.argmax(votes))
    try:
        import torch
        from PIL import Image

        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = bgr[y0:y1 + 1, x0:x1 + 1][:, :, ::-1]
        pil = Image.fromarray(crop)
        inp = clip["proc"](text=clip["prompts"], images=pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = clip["model"](**inp)
        probs = out.logits_per_image.softmax(dim=1).numpy()[0]
        # aggregate prompt probs per class
        best_cls, best_p = EXTERNAL, -1.0
        for cls in (EXTERNAL, BELT, CONTENT, FOREIGN):
            p = float(probs[clip["index"] == cls].sum())
            if p > best_p:
                best_cls, best_p = cls, p
        return int(best_cls)
    except Exception:
        votes = prior[mask].mean(axis=0)
        return int(np.argmax(votes))


def compute_layers(
    image: Any, *, view_type: str | None = None, use_learned: bool = True
) -> Layers:
    """Compute the 4-class semantic map. Numpy-level entry used by the orchestrator."""
    import cv2

    bgr_full = apply_clahe_lab(as_bgr(image))
    h, w = bgr_full.shape[:2]
    small, _scale = _work_resize(bgr_full)
    prior = _classical_scores(small, view_type)

    engine = "classical-prior"
    n_regions = 0
    regions = _sam_regions(small) if use_learned else None
    if regions:
        labels_small, engine = _clip_label_regions(small, regions, prior)
        n_regions = len(regions)
    else:
        labels_small = _clean_labels(prior, view_type)

    # tidy + upsample to full resolution
    labels_small = _postprocess(labels_small, view_type)
    label_map = cv2.resize(labels_small.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(np.int32)

    total = float(h * w)
    coverage = {CLASS_NAMES[c]: round(float((label_map == c).sum()) / total, 4)
                for c in (EXTERNAL, BELT, CONTENT, FOREIGN)}
    return Layers(label_map=label_map, engine=engine, coverage=coverage,
                  n_regions=n_regions, scores={"view_type": view_type})


def _postprocess(labels: np.ndarray, view_type: str | None) -> np.ndarray:
    """Enforce coherence: one belt strand; mineral sits on/over the belt footprint."""
    import cv2

    labels = labels.copy()
    belt = (labels == BELT) | (labels == CONTENT)  # belt footprint incl. its load
    belt = cv2.morphologyEx(belt.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if belt.sum() > 0:
        big = _largest_component(belt > 0)
        # mineral outside the belt footprint is spurious -> external
        labels[(labels == CONTENT) & ~big] = EXTERNAL
        labels[(labels == BELT) & ~big] = EXTERNAL
    return labels


def semantic_layers(
    image: Any, *, view_type: str | None = None, use_learned: bool = False, **_: Any
) -> dict[str, Any]:
    """Method wrapper: run the 4-class segmentation and return a JSON-safe summary.

    The live method defaults to the classical prior (``use_learned=False``); the heavy
    open-vocab path (MobileSAM automatic masks, ~minutes on CPU) is opt-in and used by the
    offline precompute lane which passes ``use_learned=True``.
    """
    ref = "SAM (Kirillov 2023) / MobileSAM Apache-2.0 + CLIP (Radford 2021); classical colour/texture prior"
    with timed() as t:
        layers = compute_layers(image, view_type=view_type, use_learned=use_learned)
    from ..models import is_present

    model_bytes = 40 * 1024 * 1024 if (use_learned and is_present("mobile_sam")) else 0
    payload = {
        "shape": [int(layers.label_map.shape[0]), int(layers.label_map.shape[1])],
        "classes": list(CLASS_NAMES.values()),
        "engine": layers.engine,
        "n_regions": layers.n_regions,
        "coverage": layers.coverage,
        "view_type": view_type,
    }
    return result(
        "segmentation.semantic_layers", "segmentation", "learned", ref,
        payload=payload, model_bytes=model_bytes, infer_ms=t.ms,
        web_drivable=(model_bytes == 0),
    )
