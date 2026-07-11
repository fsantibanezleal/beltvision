"""Train the compact 4-class belt semantic segmenter (precompute / GPU lane).

This is the reproducible recipe that produced the committed ``belt_segmenter.onnx`` weight
(SegFormer-B0, opset 17) which :mod:`beltvision.methods.semantic` loads as its PRIMARY
scene segmenter. Torch / transformers / onnx are imported lazily so importing this module
(and the whole package) stays torch-free for the CPU runtime.

Data (no permissive belt-mask dataset exists, so labels are built):

- **synthetic** - :func:`beltvision.cases.synthetic.synth_scene` emits the EXACT 4-class GT
  (external/belt/content/foreign) at arbitrary orientation, curvature, load and dust. This
  forces orientation invariance and is the exact-GT gate (belt IoU per orientation).
- **REAL discharge / end-view weak labels** - MobileSAM masks on real, CC-licensed
  discharge-view frames (:func:`build_pseudo_labels`), the target domain for COLA 34:
    * Kieswerk Kronau gravel conveyor (CC0) + aggregate sand discharge (CC BY 2.0) ->
      CONTENT. The non-belt region is UNAMBIGUOUS background (lake, sky, ground, plant),
      so it is kept EXTERNAL - the strong negative that stops the model calling bright
      rocks / mesh "belt" (the prior model's COLA 34 failure). ``ignore_outside=False``.
    * Velenje coal mine unloading (CC BY 3.0) -> BELT (the pale belt strand the coal
      discharges from). The scene is very dark, so the surroundings are an ambiguous dark
      mix -> ``ignore_outside=True`` (supervise only the positive belt region; never teach
      "dark = external", which would contradict the dark-rubber belt).
    * A few cubes-on-conveyor frames (empty dark belt over a bright metal table) -> BELT.
  The real discharge domain is WEIGHTED HEAVILY (``real_repeat``) - it is the target. The
  hazy COLA 34 end-view is held out for grounded validation, never trained on.

The video frames are extracted from the git-ignored ``data/raw`` vault with
:func:`extract_video_frames` (they are training input, not committed); only a small,
downscaled still sample per source is committed under ``data/reference/end_view``.

Preprocessing is byte-for-byte what :mod:`beltvision.methods.semantic` feeds the ONNX at
serve time: ``apply_clahe_lab`` -> resize -> BGR2RGB -> /255 -> ImageNet-normalize.

References: Xie et al. 2021 (SegFormer, NeurIPS); Kirillov et al. 2023 (SAM) / MobileSAM
(Apache-2.0); Ronneberger et al. 2015 (U-Net, the from-scratch fallback).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..cases.synthetic import BELT, CONTENT, EXTERNAL, synth_scene
from ..methods.preprocess import apply_clahe_lab

# Kept in lockstep with beltvision.methods.semantic (_SEG_INPUT/_SEG_MEAN/_SEG_STD).
SEG_INPUT = 256
NUM_CLASSES = 4
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ORIENTS = (2, 20, 35, 50, 65, 90, 110, 125, 140, 160)


# --- frame extraction from the vaulted CC discharge videos --------------------------------

def extract_video_frames(
    clips: list[tuple[str, int, str]], video_dir: Path, out_dir: Path, max_long: int = 1280
) -> list[dict[str, Any]]:
    """Sample frames from vaulted CC videos to ``out_dir`` (training input, not committed).

    ``clips`` is a list of ``(filename, stride_in_source_frames, tag)``. Static-camera .ogv
    clips can be seek-sampled; here we read sequentially and keep every ``stride``-th frame.
    Records the source clip + source-frame index + time per frame for provenance.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for clip, stride, tag in clips:
        cap = cv2.VideoCapture(str(video_dir / clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        d = out_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        kept = idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_long:
                    s = max_long / float(max(h, w))
                    frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                                       interpolation=cv2.INTER_AREA)
                name = f"{tag}_{kept:03d}.jpg"
                cv2.imwrite(str(d / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.append({"clip": clip, "tag": tag, "file": f"{tag}/{name}",
                                 "src_frame": idx, "src_time_s": round(idx / fps, 3)})
                kept += 1
            idx += 1
        cap.release()
    (out_dir / "extract_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --- weak real labels via MobileSAM -------------------------------------------------------

@dataclass(frozen=True)
class PseudoSource:
    """A real-frame glob and the class its SAM region maps to, with a point prompt.

    ``ignore_outside`` marks a dark/ambiguous scene (e.g. coal discharge): only the positive
    region is supervised, the surroundings are left IGNORE. When False the non-belt region is
    unambiguous background and is kept EXTERNAL (the strong negative for the belt class).
    """

    glob: str
    cls: int                  # BELT (empty / discharge belt) or CONTENT (loaded belt)
    points: list              # list of [x, y] frac point prompts (all positive)
    ignore_outside: bool = False
    stride: int = 1
    limit: int = 24
    bright_min: float = 0.0   # skip frames darker than this mean grey (near-black)
    area_lo: float = 0.05     # reject degenerate SAM regions (too small)
    area_hi: float = 0.6      # ... or too large


# The committed-stills sources (real, license-clean end/discharge view). The discharge-video
# frame directories (from extract_video_frames) are appended by the caller with matching
# PseudoSource entries pointing at the extracted-frames root.
DISCHARGE_SOURCES: list[PseudoSource] = [
    PseudoSource("end_view/kieswerk_gravel_cc0/kieswerk_*.jpg", CONTENT,
                 [[0.30, 0.62], [0.45, 0.52], [0.60, 0.46], [0.72, 0.42]], area_hi=0.6),
    PseudoSource("end_view/aggregate_discharge_ccby/sand_discharge_onto_conveyor.jpg", CONTENT,
                 [[0.30, 0.74], [0.45, 0.68], [0.15, 0.66]], area_hi=0.5),
    PseudoSource("end_view/velenje_coalmine_ccby/unloading_*.jpg", BELT,
                 [[0.45, 0.72], [0.30, 0.70]], ignore_outside=True, area_hi=0.5),
    PseudoSource("packages/cubes_conveyor/*.jpg", BELT,
                 [[0.15, 0.55], [0.35, 0.50], [0.55, 0.47], [0.75, 0.42]], limit=6, area_hi=0.7),
]


def build_pseudo_labels(
    ref_dir: Path, sam_weight: Path, out_dir: Path, sources: list[PseudoSource]
) -> list[dict[str, Any]]:
    """Run MobileSAM (ultralytics) on the real frames and write per-frame pseudo-labels.

    Writes ``<stem>.npy`` (uint8 label map) + ``<stem>.qa.jpg`` (overlay for human QA) +
    ``manifest.json`` (with per-record ``ignore_outside``). A human keeps only the good ones
    (belt / content actually captured); the kept subset feeds :func:`train_belt_segmenter`.
    """
    import cv2
    from ultralytics import SAM

    out_dir.mkdir(parents=True, exist_ok=True)
    model = SAM(str(sam_weight))
    manifest: list[dict[str, Any]] = []
    for s in sources:
        files = sorted(ref_dir.glob(s.glob))[::s.stride][: s.limit]
        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            if cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean() < s.bright_min:
                continue
            h, w = bgr.shape[:2]
            pts = [[p[0] * w, p[1] * h] for p in s.points]
            res = model(bgr, points=pts, labels=[1] * len(pts), verbose=False)
            r = res[0]
            if r.masks is None or len(r.masks.data) == 0:
                continue
            m = np.any(r.masks.data.cpu().numpy() > 0.5, axis=0)
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            if n > 1:
                m = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            area = float(m.mean())
            if area < s.area_lo or area > s.area_hi:
                continue
            label_map = np.full((h, w), EXTERNAL, np.uint8)
            label_map[m] = s.cls
            stem = f"{f.parent.name}__{f.stem}"
            np.save(out_dir / f"{stem}.npy", label_map)
            col = np.zeros_like(bgr)
            col[m] = (60, 180, 75) if s.cls == BELT else (40, 120, 235)
            cv2.imwrite(str(out_dir / f"{stem}.qa.jpg"), cv2.addWeighted(bgr, 0.6, col, 0.4, 0))
            manifest.append({"stem": stem, "src": str(f.relative_to(ref_dir)),
                             "cls": int(s.cls), "ignore_outside": bool(s.ignore_outside),
                             "source": f.parent.name, "area_frac": round(area, 4)})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_pseudo_labels(
    ref_dir: Path, pl_dir: Path, keep: set[str] | None = None
) -> list[tuple[np.ndarray, np.ndarray, int, str]]:
    """Load kept pseudo-labels as (clahe_bgr, label, fill, source) tuples.

    ``fill`` is the border value for a rotation aug: EXTERNAL for scored-background sources,
    IGNORE for the ambiguous (``ignore_outside``) dark sources, whose surroundings become
    IGNORE so the loss never supervises them.
    """
    import cv2

    man = json.loads((pl_dir / "manifest.json").read_text())
    out = []
    for rec in man:
        if keep is not None and rec["stem"] not in keep:
            continue
        lab = np.load(pl_dir / f"{rec['stem']}.npy").astype(np.uint8)
        bgr = cv2.imread(str(ref_dir / rec["src"]))
        if bgr is None:
            continue
        fill = EXTERNAL
        if rec.get("ignore_outside"):
            lab[lab == EXTERNAL] = IGNORE
            fill = IGNORE
        out.append((apply_clahe_lab(bgr), lab, fill, rec.get("source", rec["stem"])))
    return out


# --- preprocessing + augmentation ---------------------------------------------------------

def to_input(bgr_clahe: np.ndarray, size: int = SEG_INPUT) -> np.ndarray:
    """CLAHE BGR frame -> (3, S, S) float32, exactly the serve-time transform."""
    import cv2

    img = cv2.resize(bgr_clahe, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def _augment(bgr, lab, rng, fill=EXTERNAL):
    import cv2

    h, w = bgr.shape[:2]
    ang = rng.uniform(-180, 180)  # full-circle rotation -> orientation invariance
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.85, 1.15))
    bgr = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lab = cv2.warpAffine(lab, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=int(fill))
    if rng.random() < 0.5:
        bgr, lab = bgr[:, ::-1].copy(), lab[:, ::-1].copy()
    if rng.random() < 0.5:
        bgr, lab = bgr[::-1].copy(), lab[::-1].copy()
    if rng.random() < 0.6:  # haze / dust: blend toward grey
        a = rng.uniform(0.1, 0.45)
        haze = (1 - a) * bgr.astype(np.float32) + a * rng.uniform(120, 180)
        bgr = haze.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5, 7]))
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
    if rng.random() < 0.6:
        jit = rng.uniform(0.7, 1.3) * bgr.astype(np.float32) + rng.uniform(-25, 25)
        bgr = jit.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        noisy = bgr.astype(np.float32) + rng.normal(0, rng.uniform(3, 12), bgr.shape)
        bgr = noisy.clip(0, 255).astype(np.uint8)
    return bgr, lab


def _make_synth(idx, seed_base, rng):
    ori = float(rng.choice(ORIENTS) + rng.uniform(-8, 8))
    curv = float(rng.choice([0.0, 0.0, 8e-4, 1.2e-3]) * rng.choice([-1, 1]))
    sc = synth_scene(
        seed=seed_base + idx, orientation_deg=ori, curvature=curv,
        loaded=rng.random() < 0.5, with_damage=rng.random() < 0.4, with_foreign=rng.random() < 0.4,
        support_offset_deg=rng.uniform(-10, 10) if rng.random() < 0.3 else 0.0,
        dust=rng.uniform(0.12, 0.5), belt_halfwidth_frac=rng.uniform(0.16, 0.30),
    )
    return apply_clahe_lab(sc.image), sc.label_map.astype(np.uint8), EXTERNAL


# --- model --------------------------------------------------------------------------------

def build_model():
    """SegFormer-B0 (nvidia/mit-b0 pretrained encoder); U-Net fallback if unavailable."""
    try:
        from transformers import SegformerForSemanticSegmentation

        m = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
        return m, "segformer-b0"
    except Exception:
        return _unet(NUM_CLASSES), "unet-small"


def _unet(nc):
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = blk(3, 32)
            self.e2 = blk(32, 64)
            self.e3 = blk(64, 128)
            self.e4 = blk(128, 256)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.d3 = blk(256, 128)
            self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.d2 = blk(128, 64)
            self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.d1 = blk(64, 32)
            self.out = nn.Conv2d(32, nc, 1)

        def forward(self, x):
            import torch

            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            d = self.d3(torch.cat([self.u3(e4), e3], 1))
            d = self.d2(torch.cat([self.u2(d), e2], 1))
            d = self.d1(torch.cat([self.u1(d), e1], 1))
            return self.out(d)

    return UNet()


def _forward_logits(model, kind, x):
    from torch.nn import functional as F  # noqa: N812

    if kind.startswith("segformer"):
        lo = model(pixel_values=x).logits
        return F.interpolate(lo, size=(SEG_INPUT, SEG_INPUT), mode="bilinear", align_corners=False)
    return model(x)


@dataclass
class SegTrainResult:
    """Measured outcome of a training run (goes straight into the model card)."""

    kind: str
    real_heldout_fullframe_iou_hard: float   # true full-frame IoU (kieswerk/sand/cubes)
    real_heldout_recall_proxy_soft: float    # velenje ignore-outside recall proxy
    real_iou_per_source: dict
    synth_belt_iou_per_orientation: dict
    synth_min_orientation_iou: float
    onnx_bytes: int
    cpu_ms: float
    parity: float
    n_synth: int
    real_repeat: int
    n_real: int
    epochs_run: int
    history: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _scene_aware_split(real):
    """Hold out the later ~20% (>=1) of EACH source (temporally later for video clips)."""
    by_src: dict[str, list] = {}
    for item in real:
        by_src.setdefault(item[3], []).append(item)
    tr, val = [], []
    for items in by_src.values():
        n_val = max(1, len(items) // 5)
        tr += items[:-n_val]
        val += items[-n_val:]
    return tr, val


def _iou_valid(pred, gt, classes):
    valid = gt != IGNORE
    pf = np.isin(pred, classes) & valid
    gf = np.isin(gt, classes) & valid
    u = int((pf | gf).sum())
    return float((pf & gf).sum() / u) if u else float("nan")


def train_belt_segmenter(
    real: list[tuple[np.ndarray, np.ndarray, int, str]],
    *,
    onnx_out: Path,
    ignore_outside_sources: set[str] | None = None,
    device: str = "cuda",
    n_synth: int = 700,
    real_repeat: int = 14,
    epochs: int = 70,
    patience: int = 14,
    batch: int = 16,
    seed: int = 34,
) -> SegTrainResult:
    """Train on synthetic exact-GT + REAL discharge pseudo-labels (weighted heavily).

    ``real`` is the kept set from :func:`load_pseudo_labels` (clahe_bgr, label, fill, source).
    Held-out val is scene-aware; early-stop maximises 0.55*hard + 0.15*soft + 0.30*synth,
    where hard = the true full-frame IoU sources and soft = the ignore-outside recall proxy.
    Everything is seeded; the ONNX is opset 17 and parity-checked against onnxruntime (CPU).
    """
    import random
    import time

    import cv2
    import torch
    from torch import nn
    from torch.nn import functional as F  # noqa: N812

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    ign = set(ignore_outside_sources or [])
    tr_real, val_real = _scene_aware_split(real)

    def resize_label(lab):
        return cv2.resize(lab, (SEG_INPUT, SEG_INPUT), interpolation=cv2.INTER_NEAREST).astype(np.int64)

    class TrainDS(torch.utils.data.Dataset):
        def __init__(self):
            self.epoch = 0
            self.n_real = len(tr_real) * (real_repeat if tr_real else 0)

        def __len__(self):
            return n_synth + self.n_real

        def __getitem__(self, i):
            r = np.random.default_rng(seed * 1000003 + i + 7919 * self.epoch)
            if i < n_synth:
                bgr, lab, fill = _make_synth(i, 100000, r)
            else:
                bgr, lab, fill, _src = tr_real[(i - n_synth) % len(tr_real)]
                bgr, lab = bgr.copy(), lab.copy()
            bgr, lab = _augment(bgr, lab, r, fill)
            return torch.from_numpy(to_input(bgr)), torch.from_numpy(resize_label(lab))

    tl = torch.utils.data.DataLoader(TrainDS(), batch_size=batch, shuffle=True, drop_last=True)
    train_ds = tl.dataset

    model, kind = build_model()
    model = model.to(device)
    class_weights = torch.tensor([1.0, 1.5, 1.2, 2.5], device=device)  # foreign is rare
    ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE)
    lr = 6e-4 if kind.startswith("unet") else 8e-5
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def dice(logits, y):
        p = F.softmax(logits, 1)
        oh = F.one_hot(y.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float()
        valid = (y != IGNORE).unsqueeze(1).float()
        p = p * valid
        oh = oh * valid
        inter = (p * oh).sum((0, 2, 3))
        union = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        return (1 - (2 * inter + 1) / (union + 1)).mean()

    @torch.no_grad()
    def predict(bgr_clahe):
        x = torch.from_numpy(to_input(bgr_clahe))[None].to(device)
        return _forward_logits(model, kind, x).argmax(1)[0].cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def eval_real():
        model.eval()
        per: dict[str, list] = {}
        for bgr, lab, _fill, src in val_real:
            gt = cv2.resize(lab, (SEG_INPUT, SEG_INPUT), interpolation=cv2.INTER_NEAREST)
            per.setdefault(src, []).append(_iou_valid(predict(bgr), gt, [BELT, CONTENT]))
        summary = {s: float(np.nanmean(v)) for s, v in per.items()}
        hard = [summary[s] for s in summary if s not in ign]
        soft = [summary[s] for s in summary if s in ign]
        return summary, (float(np.mean(hard)) if hard else 0.0), (float(np.mean(soft)) if soft else 0.0)

    @torch.no_grad()
    def eval_synth():
        model.eval()
        rows = {}
        for ori in ORIENTS:
            ious = []
            for k, loaded in enumerate([False, True]):
                sc = synth_scene(seed=900000 + ori * 7 + k, orientation_deg=float(ori),
                                 loaded=loaded, with_damage=(k == 0), with_foreign=(k == 1),
                                 dust=0.3, belt_halfwidth_frac=0.24)
                pred = predict(apply_clahe_lab(sc.image))
                gtl = cv2.resize(sc.label_map.astype(np.uint8), (SEG_INPUT, SEG_INPUT),
                                 interpolation=cv2.INTER_NEAREST)
                cls = [BELT] if not loaded else [BELT, CONTENT]
                ious.append(_iou_valid(pred, gtl, cls))
            rows[ori] = round(float(np.mean(ious)), 4)
        return rows

    best = -1.0
    best_state = None
    bad = 0
    hist: list[dict[str, Any]] = []
    for ep in range(epochs):
        train_ds.epoch = ep
        model.train()
        for x, y in tl:
            y = y.to(device)
            logits = _forward_logits(model, kind, x.to(device))
            loss = ce(logits, y) + 0.5 * dice(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        real_sum, hard, soft = eval_real()
        synth = eval_synth()
        synth_mean = float(np.mean(list(synth.values())))
        score = 0.55 * hard + 0.15 * soft + 0.30 * synth_mean
        hist.append({"ep": ep, "hard": round(hard, 4), "soft": round(soft, 4),
                     "synth_mean": round(synth_mean, 4)})
        if score > best:
            best = score
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    real_sum, hard, soft = eval_real()
    synth = eval_synth()
    synth_min = float(np.min(list(synth.values())))

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, x):
            return _forward_logits(self.m, kind, x)

    wrap = Wrap().to(device).eval()
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, SEG_INPUT, SEG_INPUT), device=device)
    torch.onnx.export(wrap, dummy, str(onnx_out), opset_version=17,
                      input_names=["input"], output_names=["logits"], dynamic_axes=None)

    import onnxruntime as ort

    probe = np.random.default_rng(0).standard_normal((1, 3, SEG_INPUT, SEG_INPUT)).astype(np.float32)
    with torch.no_grad():
        t_out = wrap(torch.from_numpy(probe).to(device)).cpu().numpy()
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    o_out = sess.run(None, {"input": probe})[0]
    parity = float(np.max(np.abs(t_out - o_out)))
    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, {"input": probe})
        lat.append((time.perf_counter() - t0) * 1000)

    return SegTrainResult(
        kind=kind,
        real_heldout_fullframe_iou_hard=hard,
        real_heldout_recall_proxy_soft=soft,
        real_iou_per_source={k: round(v, 4) for k, v in real_sum.items()},
        synth_belt_iou_per_orientation=synth,
        synth_min_orientation_iou=synth_min,
        onnx_bytes=int(onnx_out.stat().st_size), cpu_ms=float(np.median(lat)), parity=parity,
        n_synth=n_synth, real_repeat=real_repeat, n_real=len(tr_real),
        epochs_run=len(hist), history=hist)
