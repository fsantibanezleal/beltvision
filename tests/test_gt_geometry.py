"""BLOCKING ground-truth gate: the robust belt pipeline must recover KNOWN geometry.

Synthetic scenes carry exact GT (orientation, belt width, centreline). A method that cannot
recover the GT within tolerance is broken and must not ship. This gate exercises vertical /
horizontal / diagonal / curved / misaligned belts, loaded and empty — an all-vertical suite
would let orientation bugs (the ~90deg flip on empty belts, now fixed by the orientation sweep)
survive. Tolerances: orientation 10deg, width 30%, centre 15% of the frame min-dim. Straight
belts land at ~1deg; the 10deg budget covers only the two hardest cases (a CURVED belt, where a
straight-line fit measures the average tangent, and a support beam NEARLY PARALLEL to the belt).
"""
from __future__ import annotations

import numpy as np
import pytest

from beltvision.cases.synthetic import synth_scene
from beltvision.methods import robust

TOL_ANGLE_DEG = 10.0
TOL_WIDTH_FRAC = 0.30
TOL_CENTRE_FRAC = 0.15

# GT_SUITE-style cases + a dense orientation sweep (loaded and empty) so no orientation is special.
CASES: dict[str, dict] = {
    "vertical_empty": dict(orientation_deg=90.0, loaded=False, with_damage=True),
    "horizontal_loaded": dict(orientation_deg=2.0, loaded=True, with_foreign=True),
    "diag30_loaded": dict(orientation_deg=30.0, loaded=True),
    "diag45_empty": dict(orientation_deg=45.0, loaded=False, with_damage=True),
    "curved_loaded": dict(orientation_deg=75.0, curvature=1.1e-3, loaded=True),
    "misaligned_side": dict(orientation_deg=88.0, loaded=False, support_offset_deg=9.0),
}
for _o in (0.0, 40.0, 60.0, 120.0, 150.0):
    CASES[f"sweep_{int(_o):03d}_empty"] = dict(orientation_deg=_o, loaded=False)
    CASES[f"sweep_{int(_o):03d}_loaded"] = dict(orientation_deg=_o, loaded=True)


def _ang_err(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _perp_dist(point: np.ndarray, line) -> float:
    a = np.asarray(line["p0"] if isinstance(line, dict) else line[0], float)
    b = np.asarray(line["p1"] if isinstance(line, dict) else line[1], float)
    ab = b - a
    n = float(np.hypot(*ab))
    return float(np.hypot(*(point - a))) if n < 1e-6 else float(abs(np.cross(ab, point - a)) / n)


@pytest.mark.parametrize("name", list(CASES))
def test_robust_belt_band_recovers_gt(name: str):
    sc = synth_scene(**CASES[name])
    h, w = sc.image.shape[:2]
    gt_ang = sc.orientation_deg
    gt_w = 2.0 * sc.meta["belt_halfwidth_px"]
    gt_centre = np.array([w * 0.5, h * 0.5])  # the synthetic belt is centred

    rec = robust.belt_band(sc.image)

    ang_e = _ang_err(rec["orientation_deg"], gt_ang)
    width_e = abs(rec["width_px"] - gt_w) / gt_w
    centre_e = _perp_dist(gt_centre, rec["centreline"]) / min(h, w)

    assert ang_e <= TOL_ANGLE_DEG, f"{name}: orientation off by {ang_e:.1f}deg (got {rec['orientation_deg']:.1f}, GT {gt_ang:.1f})"
    assert width_e <= TOL_WIDTH_FRAC, f"{name}: width off by {width_e*100:.0f}% (got {rec['width_px']:.0f}, GT {gt_w:.0f})"
    assert centre_e <= TOL_CENTRE_FRAC, f"{name}: centreline off by {centre_e*100:.0f}% of min-dim"


def test_straight_belts_recover_orientation_tightly():
    """The non-adversarial straight belts must be recovered to a TIGHT 4deg (they land ~1deg)."""
    worst = 0.0
    for o in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0):
        for loaded in (False, True):
            sc = synth_scene(orientation_deg=o, loaded=loaded)
            rec = robust.belt_band(sc.image)
            worst = max(worst, _ang_err(rec["orientation_deg"], sc.orientation_deg))
    assert worst <= 4.0, f"a straight belt's orientation should be within 4deg; worst was {worst:.1f}deg"


def test_damage_severity_responds_to_injected_damage():
    """A belt WITH an injected rip/hole must read >= severity than the same belt without."""
    clean = synth_scene(orientation_deg=90.0, loaded=False, with_damage=False)
    torn = synth_scene(orientation_deg=90.0, loaded=False, with_damage=True)
    s_clean = robust.damage(clean.image)["severity"]
    s_torn = robust.damage(torn.image)["severity"]
    assert s_torn >= s_clean, f"injected damage should not read lower severity ({s_torn} < {s_clean})"
