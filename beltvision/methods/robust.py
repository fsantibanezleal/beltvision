"""Robust, staged (cascade) belt analysis — the multi-pipeline replacement.

Each analysis is CONDITIONED on the previous stage and computed from SEVERAL complementary
pipelines that are fused with an agreement-based confidence — never one fragile method that
gives up. The cascade (see wip/colia/research/21-cascade-belt-analysis.md):

    global alignment (orientation consensus) -> belt limits (two parallel edges) ->
    centreline = MIDLINE of the two limits -> width -> damage/edges INSIDE the band.

The crux fix: the centreline is the midline between the two detected belt limits, NOT the
medial axis of a broad segmentation blob (which produced the diagonal-nonsense line). And the
two limits are found by projecting the oriented gradient onto the belt-NORMAL axis and taking
the two most-prominent opposite-polarity peaks (the parallel-edges + step-polarity prior),
which is robust to in-band texture (water/dust average out along the belt direction) where raw
Hough locks onto noise.

References: Radon orientation robust to noise/illumination (PMC2706151); structure tensor
(arXiv:2411.10497); belt sides = two lines, centreline = their midline + belt-edge Hough/
least-squares (MDPI Appl.Sci. 10/7/2402); height-direction projection belt-edge cue
(chinaminingmagazine 355c3c6b). See the research note for the full bibliography.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..render import draw_legend, draw_summary, to_png_b64
from ._common import as_bgr, result, timed

_TIER = "classical"
_CAP = "geometry"
FAM_ROBUST = "robust_cascade"


# --- angle helpers (axis angles are period-180) -----------------------------------------
def _ang_diff(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _axis_circular_mean(angles: list[float], weights: list[float] | None = None) -> float:
    """Circular mean of axis angles in [0,180) via double-angle averaging."""
    a = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w * np.sin(a)))
    c = float(np.sum(w * np.cos(a)))
    return float((np.degrees(np.arctan2(s, c)) / 2.0) % 180.0)


# --- stage 2: global alignment (orientation consensus) ----------------------------------
def _structure_tensor_orientation(gray: np.ndarray) -> tuple[float, float]:
    """Dominant STRUCTURE orientation (deg, [0,180)) + coherence via the structure tensor."""
    import cv2

    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ksize = (0, 0)
    jxx = cv2.GaussianBlur(gx * gx, ksize, 3.0)
    jyy = cv2.GaussianBlur(gy * gy, ksize, 3.0)
    jxy = cv2.GaussianBlur(gx * gy, ksize, 3.0)
    sxx, syy, sxy = float(jxx.sum()), float(jyy.sum()), float(jxy.sum())
    # dominant GRADIENT orientation; structures (lines) run perpendicular -> +90.
    grad_angle = 0.5 * np.degrees(np.arctan2(2.0 * sxy, sxx - syy))
    line_angle = float((grad_angle + 90.0) % 180.0)
    denom = sxx + syy
    coherence = float(np.hypot(sxx - syy, 2.0 * sxy) / denom) if denom > 1e-9 else 0.0
    return line_angle, coherence


def orientation_consensus(gray: np.ndarray) -> dict[str, Any]:
    """Fuse Radon + FFT + structure-tensor dominant orientation; agreement = confidence."""
    from .features import _radon_orientation
    from .transforms import fft_orientation_array

    radon_ang, radon_str = _radon_orientation(gray)
    fft_ang, _period, fft_str = fft_orientation_array(gray)
    st_ang, st_coh = _structure_tensor_orientation(gray)

    angles = [radon_ang, fft_ang, st_ang]
    # weight by each method's own strength/coherence (normalised), floor so none is zero.
    w = np.array([radon_str, fft_str, st_coh * 10.0], dtype=np.float64)
    w = np.clip(w / (w.max() + 1e-9), 0.15, 1.0).tolist()
    consensus = _axis_circular_mean(angles, w)
    # agreement: 1 - mean pairwise angular deviation / 45deg (0 when >=45deg spread).
    devs = [_ang_diff(a, consensus) for a in angles]
    agreement = float(np.clip(1.0 - float(np.mean(devs)) / 45.0, 0.0, 1.0))
    return {
        "angle_deg": round(consensus, 2),
        "agreement": round(agreement, 3),
        "per_method": {
            "radon": {"angle_deg": round(radon_ang, 2), "strength": round(radon_str, 3)},
            "fft": {"angle_deg": round(fft_ang, 2), "strength": round(fft_str, 3)},
            "structure_tensor": {"angle_deg": round(st_ang, 2), "coherence": round(st_coh, 3)},
        },
    }


# --- stage 4: belt limits via normal-axis gradient projection ---------------------------
def _normal_unit(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis_unit, normal_unit) for a belt AXIS angle in degrees."""
    a = np.radians(angle_deg)
    axis = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    normal = np.array([np.cos(a + np.pi / 2.0), np.sin(a + np.pi / 2.0)], dtype=np.float64)
    return axis, normal


def _offset_to_line(s: float, normal: np.ndarray, axis: np.ndarray,
                    shape: tuple[int, int]) -> list[list[float]]:
    """A line {p : p.normal = s} clipped to the frame, returned as two endpoints [x,y]."""
    h, w = shape[:2]
    p0 = s * normal  # closest point on the line to the origin
    diag = float(np.hypot(h, w))
    a = p0 - diag * axis
    b = p0 + diag * axis
    return [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]


def _prominent_peaks(profile: np.ndarray, margin: int, max_peaks: int = 8) -> list[int]:
    """Indices of local maxima of ``profile`` (>0), excluding a border margin, by value desc."""
    p = profile
    n = p.size
    peaks = []
    for i in range(max(1, margin), min(n - 1, n - margin)):
        if p[i] > 0 and p[i] >= p[i - 1] and p[i] >= p[i + 1]:
            peaks.append(i)
    peaks.sort(key=lambda i: float(p[i]), reverse=True)
    return peaks[:max_peaks]


def _sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    g = gray.astype(np.float32)
    return (cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _project_at(
    gx: np.ndarray, gy: np.ndarray, gray: np.ndarray, angle_deg: float,
    roi_mask: np.ndarray | None = None,
    min_width_frac: float = 0.06, max_width_frac: float = 0.9,
) -> dict[str, Any]:
    """Two belt limits from the normal-projected signed gradient, for a GIVEN orientation.

    A belt band has TWO parallel step edges of OPPOSITE polarity, so projecting the signed
    gradient onto the belt normal gives a 1-D profile with a strong +peak at one limit and a
    strong -peak at the other. Among the enclosed candidate pairs, prefer the one whose interior
    is the most HOMOGENEOUS (low gradient energy) — the belt surface is smoother than the rocky
    background or the support beams, which rejects beam/background pairs that over-widen the band.
    Returns the limit lines + width + a ``score`` used to pick the orientation in the sweep.
    """
    h, w = gray.shape[:2]
    axis, normal = _normal_unit(angle_deg)
    g_n = gx * normal[0] + gy * normal[1]           # signed gradient along the normal
    g_mag = np.abs(gx) + np.abs(gy)                  # gradient energy (for interior homogeneity)

    ys, xs = np.mgrid[0:h, 0:w]
    s_full = xs * normal[0] + ys * normal[1]
    if roi_mask is not None:
        m = np.asarray(roi_mask) > 0
        gn, sm, gm = g_n[m], s_full[m], g_mag[m]
    else:
        gn, sm, gm = g_n.ravel(), s_full.ravel(), g_mag.ravel()

    s_min, s_max = float(sm.min()), float(sm.max())
    nbins = max(64, int(s_max - s_min))
    bins = np.linspace(s_min, s_max, nbins + 1)
    idx = np.clip(np.digitize(sm, bins) - 1, 0, nbins - 1)
    signed = np.zeros(nbins, dtype=np.float64)
    energy = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.float64)
    np.add.at(signed, idx, gn)
    np.add.at(energy, idx, gm)
    np.add.at(counts, idx, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)  # sigma~2 bins: stable peaks (a sharper
    k /= k.sum()                                       # kernel picks streak/weave noise on
    prof = np.convolve(signed, k, mode="same")         # empty belts)
    energy_density = energy / np.maximum(counts, 1.0)       # mean gradient energy per bin

    span = s_max - s_min
    min_sep, max_sep = max(4.0, min_width_frac * span), max_width_frac * span
    margin = max(2, int(0.03 * nbins))
    pos_peaks = _prominent_peaks(prof, margin)
    neg_peaks = _prominent_peaks(-prof, margin)
    scale = float(np.percentile(np.abs(prof), 95)) + 1e-6
    interior_scale = float(np.percentile(energy_density, 90)) + 1e-6
    frame_centre_off = (w * 0.5) * normal[0] + (h * 0.5) * normal[1]  # for a centrality prior

    best = None  # (score, s_pos, s_neg, edge_strength)
    for pi in pos_peaks:
        for ni in neg_peaks:
            sep = abs(centers[pi] - centers[ni])
            if not (min_sep <= sep <= max_sep):
                continue
            edge_strength = min(float(prof[pi]), float(-prof[ni]))
            if edge_strength <= 0:
                continue
            lo, hi = sorted([pi, ni])
            interior = float(energy_density[lo + 1:hi].mean()) if hi - lo > 2 else interior_scale
            # smooth interior (belt) -> low energy -> high homogeneity bonus in [0.4, 1.4]
            homogeneity = float(np.clip(1.4 - interior / interior_scale, 0.4, 1.4))
            # interior-edge penalty: a clean belt band has NO strong signed-gradient peak between
            # its two limits. A wider pair that ENCLOSES the real belt edges (e.g. two support
            # beams bracketing the belt) has strong interior peaks -> reject it. This removes the
            # ~+17% width bias from beams and prefers the true belt-edge pair.
            inner = np.abs(prof[lo + 1:hi]) if hi - lo > 2 else np.array([0.0])
            inner_max = float(inner.max()) if inner.size else 0.0
            clean = float(np.clip(1.0 - inner_max / (edge_strength + 1e-6), 0.2, 1.0))
            # centrality prior: the belt usually spans a central region, not a thin band hugging
            # a frame edge (e.g. the HORIZON on a real oblique frame). Mild bonus for a band whose
            # centre sits near the frame centre (never a hard reject — an off-centre belt still
            # scores, just lower). Keeps the central synthetic belts and rejects the horizon.
            band_centre = 0.5 * (centers[pi] + centers[ni])
            centrality = float(np.clip(1.25 - abs(band_centre - frame_centre_off) / (0.4 * span),
                                       0.55, 1.25))
            score = edge_strength * homogeneity * clean * centrality
            if best is None or score > best[0]:
                best = (score, float(centers[pi]), float(centers[ni]), edge_strength)

    if best is not None:
        score, s_pos, s_neg, edge_strength = best
        width_ok = True
    else:
        pos_i, neg_i = int(np.argmax(prof)), int(np.argmin(prof))
        s_pos, s_neg = float(centers[pos_i]), float(centers[neg_i])
        edge_strength = min(abs(prof[pos_i]), abs(prof[neg_i]))
        score = 0.0
        width_ok = min_sep <= abs(s_pos - s_neg) <= max_sep
    prom = edge_strength / scale
    sep = abs(s_pos - s_neg)
    confidence = float(np.clip((prom - 1.0) / 3.0, 0.0, 1.0)) * (1.0 if width_ok else 0.25)

    s_a, s_b = sorted([s_pos, s_neg])
    return {
        "edge_a": _offset_to_line(s_a, normal, axis, (h, w)),
        "edge_b": _offset_to_line(s_b, normal, axis, (h, w)),
        "centreline": _offset_to_line((s_a + s_b) / 2.0, normal, axis, (h, w)),
        "width_px": round(sep, 1), "confidence": round(confidence, 3),
        "width_ok": bool(width_ok), "offsets": [round(s_a, 1), round(s_b, 1)],
        "score": float(score),
    }


def _sweep_orientation(
    gray: np.ndarray, roi_mask: np.ndarray | None, prior_deg: float | None,
) -> tuple[float, dict[str, Any]]:
    """Find the belt orientation by MAXIMISING the two-parallel-edge band score over a θ sweep.

    Global texture estimators (Radon/FFT/structure-tensor) misfire when the belt is not the
    dominant structure (e.g. an empty return strand amid rocky background), flipping ~90deg.
    The belt's DEFINING feature is two parallel opposite-polarity edges bounding a homogeneous
    band; scoring each candidate orientation by that signature recovers the true axis robustly.
    ``prior_deg`` (the consensus) only breaks near-ties. Coarse 6deg sweep, then ±5deg refine.
    """
    gx, gy = _sobel(gray)

    def _scored(theta: float) -> dict[str, Any]:
        p = _project_at(gx, gy, gray, theta, roi_mask=roi_mask)
        s = p["score"]
        if prior_deg is not None:  # tiny prior bonus (<=6%) to break ties toward the consensus
            s *= 1.0 + 0.06 * (1.0 - min(_ang_diff(theta, prior_deg) / 90.0, 1.0))
        p["_rank"] = s
        return p

    coarse = [(t, _scored(float(t))) for t in range(0, 180, 6)]
    best_t, best_p = max(coarse, key=lambda kv: kv[1]["_rank"])
    for t in range(int(best_t) - 5, int(best_t) + 6):
        p = _scored(float(t % 180))
        if p["_rank"] > best_p["_rank"]:
            best_t, best_p = float(t % 180), p
    best_p.pop("_rank", None)
    return float(best_t) % 180.0, best_p


def _project_limits(
    gray: np.ndarray, angle_deg: float, roi_mask: np.ndarray | None = None,
    min_width_frac: float = 0.06, max_width_frac: float = 0.9,
) -> dict[str, Any]:
    """Backward-compatible wrapper: compute gradients then project at a fixed orientation."""
    gx, gy = _sobel(gray)
    return _project_at(gx, gy, gray, angle_deg, roi_mask, min_width_frac, max_width_frac)


def _hough_limits(bgr: np.ndarray, angle_deg: float, band_deg: float,
                  roi_mask: np.ndarray | None) -> dict[str, Any] | None:
    """Cross-check: constrained Hough in the consensus band -> belt-edge pair (or None)."""
    from . import constrained

    be = constrained.preprocess_for_lines(bgr, roi_mask=roi_mask, denoise="gaussian", edge="canny")
    gray = _clahe_gray(bgr)
    be = constrained.gradient_orientation_gate(be, gray, angle_deg, band_deg)
    rec = constrained.hough_constrained(be, angle_deg, band_deg, roi_mask=roi_mask, bgr=None)
    segs = rec.get("segments") or []
    if len(segs) < 2:
        return None
    pair = constrained.extract_belt_edges(segs, angle_deg, frame_shape=bgr.shape[:2])
    if not pair.get("found"):
        return None
    return {"edge_a": pair["edge_a"], "edge_b": pair["edge_b"],
            "width_px": pair["width_px"], "centreline": pair.get("centreline")}


def _clahe_gray(bgr: np.ndarray) -> np.ndarray:
    import cv2

    from .preprocess import apply_clahe_lab

    return cv2.cvtColor(apply_clahe_lab(bgr), cv2.COLOR_BGR2GRAY)


def _line_pts(line: Any) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of a line given as [[x,y],[x,y]] OR {"p0":[x,y],"p1":[x,y]}."""
    if isinstance(line, dict):
        return np.asarray(line["p0"], dtype=np.float64), np.asarray(line["p1"], dtype=np.float64)
    return np.asarray(line[0], dtype=np.float64), np.asarray(line[1], dtype=np.float64)


def _line_mid_offset(line: Any, normal: np.ndarray) -> float:
    p0, p1 = _line_pts(line)
    return float(((p0 + p1) / 2.0) @ normal)


def belt_band(
    image: Any, *, semantic_obb: dict | None = None, roi_mask: np.ndarray | None = None,
    band_deg: float = 20.0, **_: Any,
) -> dict[str, Any]:
    """Robust belt geometry: orientation consensus -> two limits (projection + Hough fused)
    -> centreline = midline of the limits -> width. Returns the uniform method record."""
    import cv2

    bgr = as_bgr(image)
    gray = _clahe_gray(bgr)
    h, w = gray.shape[:2]
    with timed() as t:
        # orientation comes from the BAND STRUCTURE (a θ sweep maximising the two-parallel-edge
        # signature), NOT global texture — the consensus is only a tie-break prior. This fixes
        # the ~90deg flip on empty belts where the belt is not the dominant scene texture.
        ori = orientation_consensus(gray)
        angle, proj = _sweep_orientation(gray, roi_mask, prior_deg=ori["angle_deg"])
        _axis, normal = _normal_unit(angle)

        hough = _hough_limits(bgr, angle, band_deg, roi_mask)

        # fuse: projection is the primary limit estimate; Hough is the cross-check. Agreement =
        # do the two independent width/position estimates match?
        per_pipeline: dict[str, Any] = {
            "orientation_consensus": ori,
            "normal_projection": {k: proj[k] for k in ("width_px", "confidence", "offsets", "width_ok")},
            "constrained_hough": ({"width_px": hough["width_px"]} if hough else {"found": False}),
        }
        agree = 0.0
        if hough is not None:
            wa, wb = proj["width_px"], hough["width_px"]
            width_match = 1.0 - min(abs(wa - wb) / max(wa, wb, 1.0), 1.0)
            # also compare centre offsets
            ca = _line_mid_offset(proj["centreline"], normal)
            cb = _line_mid_offset(hough["centreline"], normal) if hough.get("centreline") else ca
            centre_match = 1.0 - min(abs(ca - cb) / max(0.15 * (w + h), 1.0), 1.0)
            agree = float(np.clip(0.5 * width_match + 0.5 * centre_match, 0.0, 1.0))

        # confidence: the projection<->Hough cross-check is the most honest signal (two
        # INDEPENDENT limit estimates agreeing), so it dominates; orientation agreement is
        # de-weighted (Radon has a 90deg ambiguity that misfires on clean horizontal belts).
        has_xcheck = hough is not None
        confidence = float(np.clip(
            0.30 * proj["confidence"] + 0.25 * ori["agreement"] + 0.45 * agree, 0.0, 1.0))
        # a confident "found" REQUIRES the independent cross-check to corroborate the band;
        # when it does not (ambiguous frame like a water curtain), drop to candidates + low
        # confidence and defer to the guided Studio rather than overclaiming.
        corroborated = has_xcheck and agree >= 0.5
        if not corroborated:
            confidence = min(confidence, 0.34)
        found = bool(proj["width_ok"] and corroborated and confidence >= 0.4)
        edge_a, edge_b = proj["edge_a"], proj["edge_b"]
        centre = proj["centreline"]
        width_px = proj["width_px"]

        conf_label = ("high" if confidence >= 0.66 else "medium" if confidence >= 0.4 else "low")

    # --- overlay: two limits + centreline (midline), Hough cross-check faded ---
    img = bgr.copy()
    if hough is not None:
        for e in (hough["edge_a"], hough["edge_b"]):
            p0 = (int(round(e["p0"][0])), int(round(e["p0"][1])))
            p1 = (int(round(e["p1"][0])), int(round(e["p1"][1])))
            cv2.line(img, p0, p1, (120, 120, 120), 1, cv2.LINE_AA)  # gray cross-check

    def _dl(line, color, thick):
        p0 = (int(round(line[0][0])), int(round(line[0][1])))
        p1 = (int(round(line[1][0])), int(round(line[1][1])))
        cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)

    # Always draw the best candidate limits + centreline — solid when found, thinner when it is
    # only a low-confidence candidate (never draw nothing; never invent a single medial line).
    thick = 3 if found else 2
    _dl(edge_a, (60, 230, 80), thick)     # limit A — green
    _dl(edge_b, (80, 180, 255), thick)    # limit B — orange
    _dl(centre, (230, 210, 60), 2)        # centreline (midline of A,B) — cyan/yellow
    tag = "belt limit" if found else "candidate limit"
    legend = [((60, 230, 80), f"{tag} A"), ((80, 180, 255), f"{tag} B"),
              ((230, 210, 60), "centreline = midline of A,B")]
    if hough is not None:
        legend.append(((120, 120, 120), "Hough cross-check"))
    draw_legend(img, legend)
    if found:
        draw_summary(img, f"Belt band FOUND (confidence {conf_label} {confidence:.2f}): axis "
                          f"{angle:.0f}deg, width {width_px:.0f}px. Centreline is the MIDLINE of "
                          f"the two limits (projection + Hough cross-check {agree*100:.0f}% agree).")
    else:
        draw_summary(img, f"Belt band candidates (confidence {conf_label} {confidence:.2f}): axis "
                          f"{angle:.0f}deg (agreement {ori['agreement']:.2f}), candidate width "
                          f"{width_px:.0f}px. Ambiguous frame — best-guess limits shown; refine "
                          "with guided ROIs in the Studio.")

    payload = {
        "name": "Robust belt geometry (cascade)", "family": FAM_ROBUST,
        "found": found, "confidence": round(confidence, 3), "confidence_label": conf_label,
        "orientation_deg": angle, "orientation_agreement": ori["agreement"],
        "edge_a": edge_a, "edge_b": edge_b, "centreline": centre, "width_px": width_px,
        "cross_check_agreement": round(agree, 3),
        "per_pipeline": per_pipeline,
        "metric_name": "confidence", "metric_value": round(confidence, 3),
        "summary": f"belt band {conf_label} conf, axis {angle:.0f}deg, width {width_px:.0f}px",
    }
    res = result("geometry.belt_band_robust", _CAP, _TIER,
                 "Radon/FFT/structure-tensor consensus + normal-projection two-limits + "
                 "constrained-Hough cross-check (MDPI 10/7/2402; PMC2706151)",
                 payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True)
    res["overlay_b64"] = to_png_b64(img)
    res["id"] = res["method"]
    return res


# --- band mask from the two limits ------------------------------------------------------
def band_mask_from_edges(shape: tuple[int, int], edge_a: Any, edge_b: Any) -> np.ndarray:
    """Boolean mask of the belt band: pixels whose normal-offset lies between the two limits."""
    h, w = shape[:2]
    a0, a1 = _line_pts(edge_a)
    b0, b1 = _line_pts(edge_b)
    axis_v = (a1 - a0)
    n = np.hypot(axis_v[0], axis_v[1])
    if n < 1e-6:
        return np.ones((h, w), dtype=bool)
    axis_v = axis_v / n
    normal = np.array([-axis_v[1], axis_v[0]], dtype=np.float64)
    s_a = float(((a0 + a1) / 2.0) @ normal)
    s_b = float(((b0 + b1) / 2.0) @ normal)
    lo, hi = sorted([s_a, s_b])
    ys, xs = np.mgrid[0:h, 0:w]
    s = xs * normal[0] + ys * normal[1]
    return (s >= lo) & (s <= hi)


# --- stage 5: damage — RGB anomaly ENSEMBLE inside the validated band -------------------
def _norm01_in(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalise ``x`` to [0,1] using robust percentiles computed INSIDE ``mask``."""
    v = x[mask]
    if v.size < 16:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~mask] = 0.0
    return out


def damage(image: Any, *, band: dict | None = None,
           content_mask: np.ndarray | None = None, **_: Any) -> dict[str, Any]:
    """Damage as an RGB-ANOMALY ensemble INSIDE the validated belt band (honest: no laser/depth).

    Runs four complementary anomaly pipelines — illumination residual, wavelet texture-removal
    residual, FFT band-stop residual, morphological black/top-hat — each restricted to the belt
    band, normalises them inside the band, and FUSES to one heatmap + flagged regions. Reports
    each pipeline's contribution and states the RGB-only anomaly limitation honestly.

    Cascade: when a ``content_mask`` (from the semantic layer) is given, the transported material
    is EXCLUDED from the band so damage is read on the exposed belt only — the ore/coal texture is
    not mistaken for belt defects (the belt-vs-content point). Each pipeline's mean response inside
    the belt is reported in ``per_pipeline`` for inspection.
    """
    import cv2

    from .transforms import dwt_reconstruct_array, fft_reconstruct_array

    bgr = as_bgr(image)
    gray = _clahe_gray(bgr).astype(np.float32)
    h, w = gray.shape[:2]
    with timed() as t:
        if band is None:
            band = belt_band(bgr)
        mask = (band_mask_from_edges((h, w), band["edge_a"], band["edge_b"])
                if band.get("edge_a") else np.ones((h, w), bool))
        # belt/content split: exclude the transported material so damage reads the exposed belt.
        content_excluded_px = 0
        if content_mask is not None:
            cm = np.asarray(content_mask) > 0
            if cm.shape == mask.shape:
                content_excluded_px = int((mask & cm).sum())
                mask = mask & ~cm
        band_area = int(mask.sum())
        confident_band = bool(band.get("found"))

        # 1) illumination-normalised residual (flat-field): deviation from a large-scale blur
        bg = cv2.GaussianBlur(gray, (0, 0), max(h, w) / 16.0)
        illum = _norm01_in(np.abs(gray - bg), mask)
        # 2) wavelet texture-removal residual (detail subbands = local high-freq anomalies)
        try:
            detail, _r, _l = dwt_reconstruct_array(gray, keep=["detail"], level=2)
            dwt = _norm01_in(np.abs(cv2.resize(detail, (w, h))), mask)
        except Exception:  # noqa: BLE001
            dwt = np.zeros_like(gray)
        # 3) FFT band-stop: remove periodic belt/mesh texture, residual = non-periodic anomalies
        try:
            recon, _e = fft_reconstruct_array(gray, kind="high", r_low=0.06, r_high=0.5)
            fft = _norm01_in(np.abs(recon), mask)
        except Exception:  # noqa: BLE001
            fft = np.zeros_like(gray)
        # 4) morphological black/top-hat (dark rips + bright scratches)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        g8 = gray.astype(np.uint8)
        morph = _norm01_in(cv2.max(cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, k),
                                   cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, k)).astype(np.float32), mask)

        maps = {"illumination_residual": illum, "wavelet_residual": dwt,
                "fft_bandstop_residual": fft, "morphological": morph}
        fused = np.mean(np.stack(list(maps.values()), 0), axis=0)
        fused[~mask] = 0.0

        # flag anomalies: high percentile INSIDE the band, then clean + connected components
        thr = float(np.percentile(fused[mask], 98.5)) if band_area else 1.0
        flag = (fused >= max(thr, 0.35)) & mask
        flag = cv2.morphologyEx(flag.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        min_area = max(30, int(0.0015 * max(band_area, 1)))
        n_lab, lab, stats, _c = cv2.connectedComponentsWithStats(flag.astype(np.uint8), 8)
        regions = []
        for i in range(1, n_lab):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_area:
                continue
            x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            regions.append({"bbox_xywh": [x, y, bw, bh], "area_px": a,
                            "elongation": round(max(bw, bh) / max(min(bw, bh), 1), 2)})
        regions.sort(key=lambda d: d["area_px"], reverse=True)
        flagged_area = sum(r["area_px"] for r in regions)
        damaged_frac = flagged_area / max(band_area, 1)
        severity = float(np.clip(damaged_frac * 50.0, 0.0, 1.0)) if regions else 0.0
        sev_label = ("none" if severity < 0.1 else "minor" if severity < 0.4
                     else "moderate" if severity < 0.7 else "severe")
        per_pipeline = {k: {"mean_in_band": round(float(v[mask].mean()), 4) if band_area else 0.0}
                        for k, v in maps.items()}
        # honest cascade caveat: a heavily-textured band is likely LOADED (content present), so
        # belt-surface damage can only be read on the visible belt between the material.
        textured_frac = float((fused[mask] > 0.5).mean()) if band_area else 0.0
        likely_loaded = textured_frac > 0.35

    # overlay: fused heatmap inside the band + region boxes
    heat = (cv2.applyColorMap((fused * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    img = bgr.copy()
    img[mask] = cv2.addWeighted(img, 0.45, heat, 0.55, 0)[mask]
    for r in regions[:24]:
        x, y, bw, bh = r["bbox_xywh"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 240), 2, cv2.LINE_AA)
    draw_legend(img, [((40, 120, 240), "fused anomaly heatmap"), ((60, 60, 240), "flagged region")])
    band_note = "" if confident_band else " (belt band low-confidence — heatmap shown in best-guess band)"
    split_note = (f" Content ({content_excluded_px} px) excluded — reading exposed belt only."
                  if content_excluded_px > 0 else "")
    draw_summary(img, f"Damage = RGB anomaly ENSEMBLE on the exposed belt (illum+wavelet+FFT+"
                      f"morph): {len(regions)} region(s), severity {sev_label} ({severity:.2f})"
                      f"{band_note}.{split_note} RGB-only anomaly — no laser/depth ground truth.")

    payload = {
        "name": "Robust belt damage (anomaly ensemble)", "family": FAM_ROBUST,
        "applicable": bool(band_area > 0), "band_confident": confident_band,
        "band_area_px": band_area, "damaged_area_px": int(flagged_area),
        "damaged_frac_of_band": round(damaged_frac, 4), "n_damage_regions": len(regions),
        "regions": regions[:24], "severity": round(severity, 3), "severity_label": sev_label,
        "per_pipeline": per_pipeline,
        "content_excluded_px": content_excluded_px, "belt_content_split": content_excluded_px > 0,
        "textured_frac": round(textured_frac, 3), "likely_loaded": likely_loaded,
        "note": ("RGB-only anomaly detection (no laser-stripe / depth / labelled defects); "
                 "confidence is bounded — corroborate high-severity flags visually."
                 + (" Band is heavily textured → belt likely LOADED; belt-surface damage is "
                    "limited to the visible belt between the material (needs the belt/content "
                    "semantic split for a clean reading)." if likely_loaded else "")),
        "metric_name": "severity", "metric_value": round(severity, 3),
        "summary": f"{len(regions)} anomaly region(s), severity {sev_label}",
    }
    res = result("analysis.damage_robust", "anomaly", _TIER,
                 "RGB anomaly ensemble (illumination/wavelet/FFT-bandstop/morphology) in the "
                 "belt band; line-laser/depth/trained-detector are the GT-grade alternatives",
                 payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True)
    res["overlay_b64"] = to_png_b64(img)
    res["id"] = res["method"]
    return res


# --- stage 6: edge condition FROM the validated limits ----------------------------------
def edge_condition(image: Any, *, band: dict | None = None, **_: Any) -> dict[str, Any]:
    """Border condition sampled ALONG each validated belt limit (roughness / notch / continuity).

    Consumes the belt band's two limits (never a raw mask boundary). Samples the gradient
    magnitude profile along each limit line; a healthy belt edge is a strong, continuous,
    straight response — fraying/notches/missing chunks show as drops and spikes in that profile.
    """
    import cv2

    bgr = as_bgr(image)
    gray = _clahe_gray(bgr).astype(np.float32)
    h, w = gray.shape[:2]
    gmag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    with timed() as t:
        if band is None:
            band = belt_band(bgr)
        if not band.get("edge_a") or not band.get("found"):
            payload = {"name": "Robust edge condition", "family": FAM_ROBUST,
                       "applicable": False, "status": "na",
                       "reason": "belt limits low-confidence — draw guided ROIs in the Studio",
                       "metric_name": "worst_roughness", "metric_value": 0.0, "summary": "n/a"}
            res = result("analysis.edge_condition_robust", "geometry", _TIER,
                         "edge condition sampled along the validated belt limits",
                         payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True)
            img = bgr.copy()
            draw_summary(img, "Edge condition needs confident belt limits — belt band was "
                              "low-confidence on this frame. Refine with guided ROIs in the Studio.")
            res["overlay_b64"] = to_png_b64(img)
            res["id"] = res["method"]
            return res

        def _profile(edge: Any) -> dict[str, Any]:
            p0, p1 = _line_pts(edge)
            npts = int(max(32, np.hypot(*(p1 - p0))))
            ts = np.linspace(0.0, 1.0, npts)
            pts = (p0[None, :] * (1 - ts)[:, None] + p1[None, :] * ts[:, None])
            xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
            ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
            g = gmag[ys, xs]
            inb = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            g = g[inb]
            if g.size < 8:
                return {"strength": 0.0, "continuity": 0.0, "notches": 0, "roughness": 0.0}
            gn = g / (np.percentile(g, 90) + 1e-6)
            continuity = float(np.mean(gn > 0.3))                    # fraction of strong edge
            notches = int(np.sum(gn < 0.15))                         # local drops (missing chunk)
            roughness = round(float(np.std(g) / (np.mean(g) + 1e-6)), 3)
            return {"strength": round(float(np.mean(g)), 2), "continuity": round(continuity, 3),
                    "notches": notches, "roughness": roughness}

        ea, eb = _profile(band["edge_a"]), _profile(band["edge_b"])
        worst_rough = max(ea["roughness"], eb["roughness"])
        worst_cont = min(ea["continuity"], eb["continuity"])
        # Conservative RGB heuristic: a real belt edge profile varies a lot, so flag fraying
        # ONLY on strong joint evidence (very rough AND discontinuous). Report it as a
        # low-confidence RGB indicator, not a hard defect verdict.
        frayed = bool(worst_rough > 1.6 and worst_cont < 0.4)
        verdict = ("possible border fraying / missing chunk (RGB heuristic — verify)"
                   if frayed else "borders continuous")

    img = bgr.copy()
    for edge, col in ((band["edge_a"], (60, 230, 80)), (band["edge_b"], (80, 180, 255))):
        p0, p1 = _line_pts(edge)
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 2, cv2.LINE_AA)
    draw_legend(img, [((60, 230, 80), "limit A profile"), ((80, 180, 255), "limit B profile")])
    draw_summary(img, f"Edge condition along the two validated limits: {verdict}. worst roughness "
                      f"{worst_rough:.2f}, min continuity {worst_cont:.2f}, "
                      f"notches A/B {ea['notches']}/{eb['notches']}.")
    payload = {
        "name": "Robust edge condition", "family": FAM_ROBUST, "applicable": True, "status": "ok",
        "edge_a": ea, "edge_b": eb, "worst_roughness": worst_rough, "min_continuity": worst_cont,
        "frayed_or_chunks": frayed, "verdict": verdict,
        "metric_name": "worst_roughness", "metric_value": worst_rough,
        "summary": verdict,
    }
    res = result("analysis.edge_condition_robust", "geometry", _TIER,
                 "edge profile (strength/continuity/roughness/notches) sampled along the "
                 "validated belt limits",
                 payload=payload, model_bytes=0, infer_ms=t.ms, web_drivable=True)
    res["overlay_b64"] = to_png_b64(img)
    res["id"] = res["method"]
    return res


__all__ = [
    "FAM_ROBUST",
    "orientation_consensus",
    "belt_band",
    "band_mask_from_edges",
    "damage",
    "edge_condition",
]
