#!/usr/bin/env python3
"""Regenerate the figures for the beltvision software note from the COMMITTED benchmark
(data/bv.json, produced by run_bench.py). Two figures:

  fig-geometry.pdf  - the classical geometry chain recovers belt orientation and footprint
                      across orientations (vertical/horizontal/diagonal/curved): (a) axis-angle
                      error per scene with the 8deg product tolerance; (b) belt-footprint IoU.
  fig-semantic.pdf  - per-class semantic recovery: strong on background / belt / mineral content,
                      but blind to small foreign objects in the CLASSICAL core (IoU 0), which is
                      exactly why the engine keeps an optional learned lane for that class.

Run:  python make_figs.py     (after run_bench.py has written ../data/bv.json)
Deps: matplotlib, numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

INK = "#1a1a2e"
GRID = "#d8d8e0"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})

SHORT = {
    "vertical_empty": "vertical\n90 empty",
    "horizontal_loaded": "horizontal\n2 loaded",
    "diag30_loaded": "diag\n30 loaded",
    "diag45_empty": "diag\n45 empty",
    "curved_loaded": "curved\n75 loaded",
    "misaligned_side": "near-vert\n88 misalign",
}


def _load():
    return json.loads((DATA / "bv.json").read_text(encoding="utf-8"))


def fig_geometry():
    d = _load()
    sc = d["scenes"]
    names = [s["name"] for s in sc]
    labels = [SHORT.get(n, n) for n in names]
    ori = [s["ori_err_deg"] if s["ori_err_deg"] is not None else 0.0 for s in sc]
    iou = [s["belt_iou"] for s in sc]
    curved = [s["curved"] for s in sc]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 3.1))
    x = np.arange(len(sc))

    # (a) orientation error, 8deg tolerance line
    cols = ["#e07a3f" if c else "#1b6ca8" for c in curved]
    a1.bar(x, ori, color=cols, edgecolor=INK, linewidth=0.6, width=0.66, zorder=3)
    a1.axhline(8.0, color="#b23a48", linewidth=1.1, linestyle="--", label="product tolerance (8 deg)")
    for xi, v in zip(x, ori):
        a1.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    a1.set_ylabel("axis-angle error (deg)")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=6.9)
    a1.set_ylim(0, 9.2)
    a1.set_title(f"(a) orientation recovered across paths\n(mean {d['summary']['ori_err_mean_deg']:.1f} deg; "
                 f"orange = curved)", fontsize=8.4)
    a1.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    a1.set_axisbelow(True)
    a1.legend(fontsize=6.9, frameon=True, facecolor="white", edgecolor=GRID, loc="upper left")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    # (b) belt-footprint IoU
    a2.bar(x, iou, color="#3fa34d", edgecolor=INK, linewidth=0.6, width=0.66, zorder=3)
    a2.axhline(0.5, color="#b23a48", linewidth=1.0, linestyle="--", label="gate floor (0.50)")
    for xi, v in zip(x, iou):
        a2.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2)
    a2.set_ylabel("belt-footprint IoU (belt $\\cup$ content vs GT)")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=6.9)
    a2.set_ylim(0, 1.0)
    a2.set_title(f"(b) belt region recovered\n(mean {d['summary']['belt_iou_mean']:.2f}, "
                 f"min {d['summary']['belt_iou_min']:.2f})", fontsize=8.4)
    a2.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    a2.set_axisbelow(True)
    a2.legend(fontsize=6.9, frameon=True, facecolor="white", edgecolor=GRID, loc="lower right")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "fig-geometry.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_semantic():
    d = _load()
    sc = d["scenes"]

    def mean_iou(cls, scenes):
        vals = [s["class_iou"][cls] for s in scenes if cls in s["class_iou"]]
        return (float(np.mean(vals)) if vals else 0.0, len(vals), vals)

    empty = [s for s in sc if not s["loaded"]]   # belt rubber exposed
    loaded = [s for s in sc if s["loaded"]]       # rubber covered by content
    # each surface class is scored where it is actually the exposed/observable surface:
    # external over all scenes; belt over EMPTY belts (loaded belts correctly label the
    # covered rubber as content, so belt-class there is not a segmentation failure);
    # content over loaded belts; foreign wherever injected.
    cats = [("external", "background /\nstructure", sc),
            ("belt", "belt rubber\n(empty belts)", empty),
            ("content", "mineral content\n(loaded belts)", loaded),
            ("foreign", "foreign\nobject", sc)]
    means, ns, spreads = [], [], []
    for key, _, scenes in cats:
        m, n, vals = mean_iou(key, scenes)
        means.append(m); ns.append(n); spreads.append(vals)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(cats))
    cols = ["#7d99b0", "#8a7d55", "#c07a2b", "#b23a48"]
    ax.bar(x, means, color=cols, edgecolor=INK, linewidth=0.6, width=0.6, zorder=3)
    # per-scene spread as light dots
    for xi, vals in zip(x, spreads):
        if len(vals) > 1:
            ax.scatter([xi] * len(vals), vals, s=14, color=INK, alpha=0.45, zorder=4)
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
        ax.text(xi, -0.075, f"n={n}", ha="center", va="top", fontsize=6.8, color="#555")
    ax.annotate("classical core does not\nisolate foreign objects;\nthat is the learned lane",
                xy=(3, 0.02), xytext=(2.15, 0.42), fontsize=7.0, color="#b23a48",
                ha="center", arrowprops=dict(arrowstyle="->", color="#b23a48", linewidth=1.0))
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cats], fontsize=8.0)
    ax.set_ylabel("per-class IoU vs synthetic ground truth")
    ax.set_ylim(0, 1.0)
    ax.set_title("Semantic recovery by class (classical core, use_learned=False)\n"
                 "strong on belt / content / background, blind to small foreign objects",
                 fontsize=8.4)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "fig-semantic.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_geometry()
    fig_semantic()
    print("wrote fig-geometry.pdf, fig-semantic.pdf")


if __name__ == "__main__":
    main()
