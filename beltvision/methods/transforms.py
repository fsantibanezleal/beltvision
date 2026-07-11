"""Classical transforms as inspectable records: Fourier (FFT) and wavelet (DWT).

These are the frequency- and multiresolution-domain tools the surface-inspection literature
uses to characterise a regular texture and isolate the local anomalies that break it. Each
public ``*`` method returns the same uniform overlay-carrying record the rest of the ladder
does; the ``*_array`` helpers expose the raw transformed image so the Pipeline Studio can
thread a transform's output into the next node without recomputing (single source of the math).

Fourier (``numpy.fft``, no new dependency)
- :func:`fft_spectrum`  log-magnitude power spectrum (orientation/period fingerprint).
- :func:`fft_orientation`  dominant spectral peak -> texture orientation + period.
- :func:`fft_filter`  directional / band / low / high / notch mask + inverse FFT -> a
  reconstructed image that suppresses the regular structure or isolates a frequency range.
- :func:`phot`  Phase-Only Transform: reconstruct from unit-magnitude spectrum -> an
  unsupervised surface-defect anomaly map.

Wavelet (``PyWavelets``)
- :func:`dwt_decompose`  multilevel decomposition as a subband montage.
- :func:`dwt_reconstruct`  keep selected subbands -> remove the repetitive texture / enhance
  local anomalies, with a residual-energy metric.
- :func:`wavelet_denoise`  translation-invariant BayesShrink wavelet shrinkage.

References: fabric-defect detection via FFT + Gabor
(https://medcraveonline.com/JTEFT/fabric-defect-detection-using-fourier-transform-and-gabor-filters.html);
Phase-Only Transform, Aiger & Talbot (https://perso.esiee.fr/~aigerd/phot.pdf); wavelet
reconstruction for surface inspection (Pattern Recognition, S0031320300000716); steel-surface
wavelet descriptor (MDPI Materials 2024, 17/23/5873); translation-invariant wavelet shrinkage
for textile surfaces.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..render import draw_legend, draw_summary, heatmap_overlay, to_png_b64
from ._common import as_bgr, result, timed
from .preprocess import apply_clahe_lab

_CAP = "transform"
_TIER = "classical"
FAM_FREQ = "frequency_transform"
FAM_WAVELET = "wavelet"

_FFT_REF = ("Fourier power spectrum for texture orientation/period + directional/band/notch "
            "filtering (fabric-defect FFT+Gabor, medcraveonline JTEFT)")
_PHOT_REF = "Phase-Only Transform anomaly detector; Aiger & Talbot (perso.esiee.fr/~aigerd/phot.pdf)"
_DWT_REF = ("Multiresolution wavelet reconstruction / subband selection for surface-defect "
            "enhancement (Pattern Recognition S0031320300000716; MDPI Materials 2024 17/23/5873)")
_DENOISE_REF = ("Translation-invariant BayesShrink wavelet shrinkage (skimage denoise_wavelet + "
                "cycle_spin); textile/steel surface denoising")


# --- shared plumbing --------------------------------------------------------------------
def _prep(image: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(clahe_bgr, gray_float32)`` (CLAHE-first, like the rest of the ladder)."""
    import cv2

    bgr = apply_clahe_lab(as_bgr(image))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return bgr, gray


def _to_gray_f32(image: Any) -> np.ndarray:
    """Coerce an arbitrary array to a float32 grayscale plane (no CLAHE)."""
    import cv2

    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(as_bgr(arr), cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _darken(bgr: np.ndarray, factor: float = 0.5) -> np.ndarray:
    import cv2

    return cv2.addWeighted(bgr, 1.0 - factor, np.zeros_like(bgr), factor, 0)


def _heat_image(field: np.ndarray, shape_hw: tuple[int, int], colormap: int | None = None) -> np.ndarray:
    """Colour-map a field and resize it to the frame (its own overlay image)."""
    import cv2

    cm = cv2.COLORMAP_INFERNO if colormap is None else colormap
    heat = cv2.applyColorMap((_norm01(field) * 255).astype(np.uint8), cm)
    h, w = shape_hw
    if heat.shape[:2] != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    return heat


def _record(
    method_id: str, name: str, reference: str, *, family: str, metric_name: str,
    metric_value: float, overlay: np.ndarray, infer_ms: float, summary: str,
    web_drivable: bool = True, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name, "family": family, "summary": summary,
        "metric_name": metric_name, "metric_value": round(float(metric_value), 5),
    }
    if extra:
        payload.update(extra)
    res = result(method_id, _CAP, _TIER, reference, payload=payload,
                 model_bytes=0, infer_ms=infer_ms, web_drivable=web_drivable)
    res["overlay_b64"] = to_png_b64(overlay)
    res["id"] = res["method"]
    return res


# --- Fourier: pure-array helpers --------------------------------------------------------
def _spectrum(gray: np.ndarray) -> np.ndarray:
    """Shifted 2-D FFT of a mean-removed gray image."""
    g = gray - float(gray.mean())
    return np.fft.fftshift(np.fft.fft2(g))


def spectrum_magnitude_array(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-magnitude spectrum + its peak-to-mean concentration (DC excluded)."""
    g = _to_gray_f32(gray)
    h, w = g.shape
    mag = np.log1p(np.abs(_spectrum(g)))
    cy, cx = h // 2, w // 2
    peak = mag.copy()
    peak[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0
    ratio = float(peak.max() / (peak.mean() + 1e-9))
    return mag, ratio


def fft_orientation_array(gray: np.ndarray) -> tuple[float, float, float]:
    """Dominant spectral peak -> ``(axis_deg, period_px, strength)``."""
    g = _to_gray_f32(gray)
    h, w = g.shape
    mag = np.abs(_spectrum(g))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(xx - cx, yy - cy)
    mag = mag.copy()
    mag[radius < 3.0] = 0.0
    py, px = divmod(int(np.argmax(mag)), w)
    du, dv = float(px - cx), float(py - cy)
    r = float(np.hypot(du, dv))
    axis = float((np.degrees(np.arctan2(dv, du)) + 90.0) % 180.0)  # stripes _|_ freq vector
    period = float(max(h, w) / r) if r > 1e-6 else 0.0
    strength = float(mag.max() / (mag.mean() + 1e-9))
    return axis, period, strength


def fft_reconstruct_array(
    gray: np.ndarray, *, kind: str = "directional", orientation_deg: float = 0.0,
    width_deg: float = 20.0, r_low: float = 0.06, r_high: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Apply a frequency-domain mask and inverse-transform. Returns ``(recon, energy_retained)``."""
    g = _to_gray_f32(gray)
    h, w = g.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(xx - cx, yy - cy)
    rmax = float(np.hypot(cy, cx)) + 1e-6
    f = _spectrum(g)
    k = str(kind).lower()
    if k == "low":
        mask = radius <= r_high * rmax
    elif k == "high":
        mask = radius >= r_low * rmax
    elif k == "band":
        mask = (radius >= r_low * rmax) & (radius <= r_high * rmax)
    elif k == "notch":
        mag = np.abs(f).copy()
        mag[radius < max(3.0, r_low * rmax)] = 0.0
        mask = np.ones((h, w), dtype=bool)
        flat = mag.ravel()
        if flat.size:
            for i in np.argpartition(flat, -6)[-6:]:
                ry, rx = divmod(int(i), w)
                mask[max(0, ry - 2):ry + 3, max(0, rx - 2):rx + 3] = False
                sy, sx = 2 * cy - ry, 2 * cx - rx
                if 0 <= sy < h and 0 <= sx < w:
                    mask[max(0, sy - 2):sy + 3, max(0, sx - 2):sx + 3] = False
    else:  # directional wedge around orientation_deg (frequency-vector angle)
        ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 180.0
        diff = np.abs(ang - (float(orientation_deg) % 180.0)) % 180.0
        diff = np.minimum(diff, 180.0 - diff)
        mask = (diff <= float(width_deg)) & (radius >= r_low * rmax)
    total = float(np.sum(np.abs(f) ** 2)) + 1e-9
    retained = float(np.sum((np.abs(f) ** 2)[mask]) / total)
    recon = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
    return recon, retained


def phot_array(gray: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Phase-Only Transform anomaly map. Returns ``(amap, peak_ratio, px, py)``."""
    import cv2

    g = _to_gray_f32(gray)
    f = np.fft.fft2(g - float(g.mean()))
    amap = np.abs(np.fft.ifft2(f / (np.abs(f) + 1e-9)))
    amap = cv2.GaussianBlur(amap.astype(np.float32), (0, 0), 1.5)
    peak = float(amap.max() / (amap.mean() + 1e-9))
    py, px = np.unravel_index(int(np.argmax(amap)), amap.shape)
    return amap, peak, int(px), int(py)


# --- Wavelet: pure-array helpers --------------------------------------------------------
def dwt_montage_array(
    gray: np.ndarray, *, wavelet: str = "db2", level: int = 2
) -> tuple[np.ndarray, float, int]:
    """Subband montage + detail-energy fraction + the effective level used."""
    import pywt

    g = _to_gray_f32(gray)
    lvl = int(max(1, min(level, pywt.dwtn_max_level(g.shape, wavelet))))
    coeffs = pywt.wavedec2(g, wavelet, level=lvl)
    arr, _slices = pywt.coeffs_to_array(coeffs)
    approx = coeffs[0]
    detail_energy = sum(float(np.sum(b.astype(np.float64) ** 2))
                        for det in coeffs[1:] for b in det)
    total = detail_energy + float(np.sum(approx.astype(np.float64) ** 2)) + 1e-9
    return np.abs(arr), float(detail_energy / total), lvl


def dwt_reconstruct_array(
    gray: np.ndarray, *, wavelet: str = "db2", level: int = 2, keep: list[str] | None = None
) -> tuple[np.ndarray, float, int]:
    """Reconstruct from selected subbands. Returns ``(recon, residual_energy_frac, level)``."""
    import pywt

    keep_set = set(keep) if keep is not None else {"detail"}
    g = _to_gray_f32(gray)
    h, w = g.shape
    lvl = int(max(1, min(level, pywt.dwtn_max_level((h, w), wavelet))))
    coeffs = pywt.wavedec2(g, wavelet, level=lvl)
    approx = coeffs[0]
    new_coeffs: list[Any] = [approx if "approx" in keep_set else np.zeros_like(approx)]
    for i, det in enumerate(coeffs[1:], start=1):
        det_level = lvl - i + 1  # coeffs[1] is the coarsest detail level
        if "detail" in keep_set or f"detail{det_level}" in keep_set:
            new_coeffs.append(det)
        else:
            new_coeffs.append(tuple(np.zeros_like(b) for b in det))
    recon = pywt.waverec2(new_coeffs, wavelet)[:h, :w]
    orig_energy = float(np.sum((g - g.mean()).astype(np.float64) ** 2)) + 1e-9
    residual_frac = float(np.sum(recon.astype(np.float64) ** 2) / orig_energy)
    return recon, residual_frac, lvl


def wavelet_denoise_array(gray: np.ndarray, *, wavelet: str = "db2") -> tuple[np.ndarray, float]:
    """Translation-invariant BayesShrink denoise. Returns ``(denoised_uint8, noise_removed_gray)``."""
    from skimage.restoration import cycle_spin, denoise_wavelet

    g01 = (_to_gray_f32(gray) / 255.0).astype(np.float64)

    def _dn(x: np.ndarray) -> np.ndarray:
        return denoise_wavelet(x, wavelet=wavelet, mode="soft", method="BayesShrink",
                               rescale_sigma=True)
    den = cycle_spin(g01, func=_dn, max_shifts=3, func_kw={}, workers=1)
    removed = float(np.std(g01 - den) * 255.0)
    return np.clip(den * 255.0, 0, 255).astype(np.uint8), removed


# --- Fourier: records -------------------------------------------------------------------
def fft_spectrum(image: Any, **_: Any) -> dict[str, Any]:
    """Log-magnitude Fourier power spectrum; metric = peak-to-mean spectral concentration."""
    bgr, gray = _prep(image)
    h, w = gray.shape
    with timed() as t:
        mag, ratio = spectrum_magnitude_array(gray)
    overlay = _heat_image(mag, (h, w))
    draw_legend(overlay, [((0, 165, 255), "log |FFT| (frequency domain)")])
    draw_summary(overlay, f"FFT power spectrum (log magnitude): peak-to-mean concentration "
                          f"{ratio:.1f}x. Bright off-centre peaks mark the dominant periodic "
                          "texture; their position encodes its orientation and period.")
    return _record("transform.fft_spectrum", "FFT power spectrum", _FFT_REF, family=FAM_FREQ,
                   metric_name="spectral_concentration", metric_value=ratio, overlay=overlay,
                   infer_ms=t.ms, summary=f"spectral peak/mean {ratio:.1f}x", web_drivable=False)


def fft_orientation(image: Any, **_: Any) -> dict[str, Any]:
    """Dominant spectral peak -> texture orientation (deg) + spatial period (px)."""
    import cv2

    bgr, gray = _prep(image)
    h, w = gray.shape
    with timed() as t:
        axis, period, strength = fft_orientation_array(gray)
    img = _darken(bgr, 0.5)
    cy, cx = h / 2.0, w / 2.0
    d = np.array([np.cos(np.radians(axis)), np.sin(np.radians(axis))])
    arm = 0.4 * min(h, w)
    p0 = (int(cx - arm * d[0]), int(cy - arm * d[1]))
    p1 = (int(cx + arm * d[0]), int(cy + arm * d[1]))
    cv2.arrowedLine(img, p0, p1, (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.05)
    cv2.arrowedLine(img, p1, p0, (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.05)
    draw_legend(img, [((0, 200, 255), f"FFT texture orientation {axis:.0f}deg")])
    draw_summary(img, f"FFT dominant orientation {axis:.1f}deg, period ~{period:.1f}px "
                      f"(peak {strength:.1f}x mean). The spectral peak's position gives the "
                      "repeating texture's direction and wavelength.")
    return _record("transform.fft_orientation", "FFT texture orientation", _FFT_REF,
                   family=FAM_FREQ, metric_name="orientation_deg", metric_value=axis,
                   overlay=img, infer_ms=t.ms, web_drivable=False,
                   summary=f"orientation {axis:.0f}deg, period {period:.0f}px",
                   extra={"period_px": round(period, 2), "peak_strength": round(strength, 2)})


def fft_filter(
    image: Any, *, kind: str = "directional", orientation_deg: float = 0.0,
    width_deg: float = 20.0, r_low: float = 0.06, r_high: float = 0.5, **_: Any,
) -> dict[str, Any]:
    """Frequency-domain mask (directional / band / low / high / notch) + inverse FFT."""
    bgr, gray = _prep(image)
    h, w = gray.shape
    with timed() as t:
        recon, retained = fft_reconstruct_array(gray, kind=kind, orientation_deg=orientation_deg,
                                                width_deg=width_deg, r_low=r_low, r_high=r_high)
    k = str(kind).lower()
    overlay = _heat_image(np.abs(recon), (h, w))
    draw_legend(overlay, [((0, 165, 255), f"FFT {k}-filtered reconstruction")])
    draw_summary(overlay, f"FFT {k} filter: kept {retained*100:.1f}% of the spectral energy, "
                          "inverse-transformed back to the image. Removing the regular "
                          "spectral peaks leaves the local, non-periodic anomalies.")
    return _record("transform.fft_filter", f"FFT {k} filter", _FFT_REF, family=FAM_FREQ,
                   metric_name="energy_retained", metric_value=retained, overlay=overlay,
                   infer_ms=t.ms, web_drivable=False, summary=f"{k} filter, {retained*100:.0f}% energy",
                   extra={"kind": k, "orientation_deg": float(orientation_deg),
                          "r_low": float(r_low), "r_high": float(r_high)})


def phot(image: Any, **_: Any) -> dict[str, Any]:
    """Phase-Only Transform: reconstruct from unit-magnitude spectrum -> anomaly map."""
    bgr, gray = _prep(image)
    with timed() as t:
        amap, peak, px, py = phot_array(gray)
    overlay = heatmap_overlay(bgr, amap, legend_label="PHOT anomaly response",
                              summary=f"Phase-Only Transform anomaly map: peak {peak:.1f}x mean "
                                      "at the marked location. Discarding the magnitude keeps "
                                      "the phase, which concentrates on non-periodic surface "
                                      "defects.", title="PHOT anomaly", peak_xy=(px, py))
    return _record("transform.phot", "Phase-Only Transform", _PHOT_REF, family=FAM_FREQ,
                   metric_name="anomaly_peak_ratio", metric_value=peak, overlay=overlay,
                   infer_ms=t.ms, web_drivable=False, summary=f"PHOT anomaly peak {peak:.1f}x",
                   extra={"peak_xy": [px, py]})


# --- Wavelet: records -------------------------------------------------------------------
def dwt_decompose(image: Any, *, wavelet: str = "db2", level: int = 2, **_: Any) -> dict[str, Any]:
    """Multilevel DWT as a subband montage overlay; metric = detail-energy fraction."""
    bgr, gray = _prep(image)
    h, w = gray.shape
    with timed() as t:
        arr, detail_frac, lvl = dwt_montage_array(gray, wavelet=wavelet, level=level)
    overlay = _heat_image(arr, (h, w))
    draw_legend(overlay, [((0, 165, 255), f"DWT subbands ({wavelet}, {lvl} level)")])
    draw_summary(overlay, f"Wavelet decomposition ({wavelet}, {lvl} levels): the approximation "
                          f"(top-left) plus the horizontal/vertical/diagonal detail subbands. "
                          f"{detail_frac*100:.0f}% of the energy is in the detail bands.")
    return _record("transform.dwt_decompose", "Wavelet decomposition", _DWT_REF,
                   family=FAM_WAVELET, metric_name="detail_energy_frac", metric_value=detail_frac,
                   overlay=overlay, infer_ms=t.ms, web_drivable=False,
                   summary=f"{lvl}-level {wavelet}, detail energy {detail_frac*100:.0f}%",
                   extra={"wavelet": wavelet, "levels": lvl})


def dwt_reconstruct(
    image: Any, *, wavelet: str = "db2", level: int = 2, keep: list[str] | None = None, **_: Any,
) -> dict[str, Any]:
    """Reconstruct from selected subbands (default: details only) -> anomaly-enhanced residual."""
    bgr, gray = _prep(image)
    with timed() as t:
        recon, residual_frac, lvl = dwt_reconstruct_array(gray, wavelet=wavelet, level=level, keep=keep)
    keep_set = sorted(set(keep) if keep is not None else {"detail"})
    overlay = heatmap_overlay(bgr, np.abs(recon), legend_label="wavelet residual (anomaly)",
                              summary=f"Wavelet reconstruction keeping {keep_set}: the regular "
                                      f"texture is removed and {residual_frac*100:.0f}% residual "
                                      "energy remains, highlighting local surface anomalies "
                                      "(tears/cracks/foreign texture).", title="Wavelet anomaly")
    return _record("transform.dwt_reconstruct", "Wavelet subband reconstruction", _DWT_REF,
                   family=FAM_WAVELET, metric_name="residual_energy_frac",
                   metric_value=residual_frac, overlay=overlay, infer_ms=t.ms, web_drivable=False,
                   summary=f"kept {keep_set}, residual {residual_frac*100:.0f}%",
                   extra={"wavelet": wavelet, "levels": lvl, "keep": keep_set})


def wavelet_denoise(image: Any, *, wavelet: str = "db2", **_: Any) -> dict[str, Any]:
    """Translation-invariant BayesShrink wavelet shrinkage; metric = noise energy removed."""
    import cv2

    bgr, gray = _prep(image)
    with timed() as t:
        den_u8, removed = wavelet_denoise_array(gray, wavelet=wavelet)
    overlay = cv2.cvtColor(den_u8, cv2.COLOR_GRAY2BGR)
    draw_legend(overlay, [((0, 165, 255), f"wavelet-denoised ({wavelet}, TI BayesShrink)")])
    draw_summary(overlay, f"Translation-invariant wavelet denoising ({wavelet}, BayesShrink): "
                          f"removed noise of std {removed:.1f} gray levels while preserving "
                          "edges. A cleaner input for the downstream detectors.")
    return _record("transform.wavelet_denoise", "Wavelet denoise (TI BayesShrink)", _DENOISE_REF,
                   family=FAM_WAVELET, metric_name="noise_removed_gray", metric_value=removed,
                   overlay=overlay, infer_ms=t.ms, web_drivable=False,
                   summary=f"removed noise std {removed:.1f}", extra={"wavelet": wavelet})


__all__ = [
    "FAM_FREQ", "FAM_WAVELET",
    "fft_spectrum", "fft_orientation", "fft_filter", "phot",
    "dwt_decompose", "dwt_reconstruct", "wavelet_denoise",
    "spectrum_magnitude_array", "fft_orientation_array", "fft_reconstruct_array", "phot_array",
    "dwt_montage_array", "dwt_reconstruct_array", "wavelet_denoise_array",
]
