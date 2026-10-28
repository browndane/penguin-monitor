#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build P5's figures. The paper has eight tables and no figures at all.

WHY
---
The paper goes to an ecology journal, whose readers are ecologists. A paper
made entirely of tables reads as a computer-vision paper sent to the wrong
venue, and that is a desk-rejection risk independent of the science. These
three figures carry the paper's actual argument.

  fig1_cost_quality   annotation cost against mask quality. The thesis in
                      one image: near-gold masks at 5% of expert time.
  fig2_count_mae      count error by method on both gold corpora, the
                      metric an ecologist acts on.
  fig3_count_density  where counting breaks down as a colony gets denser.

Colour carries the highlight only; identity sits on the axis labels, so the
figures survive greyscale printing and colour-vision deficiency without
relying on hue separation.

OUTPUT
------
Vector PDFs written next to the paper, sized for a single journal column,
plus a 400 dpi PNG of each for slides and the web. Every save is measured for
clipping before it is accepted, because matplotlib under-measures a wide axis
label and drops the overflow off the right edge without warning.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(os.environ.get("P5_PAPER_DIR", "."))
RUNS = Path(os.path.expanduser("~/bench_any/runs"))

INK = "#333333"        # primary mark
MUTED = "#9a9a9a"      # secondary mark
HIGHLIGHT = "#0072B2"  # the proposed pipeline, one hue only
GRID = "#e4e4e4"


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 200,
    })


MIN_MARGIN_PX = 8
CLIPPED = []


def _margins(png):
    """Distance in pixels from each edge of the render to the nearest ink."""
    from PIL import Image
    ink = np.asarray(Image.open(png).convert("L")) < 250
    h, w = ink.shape
    rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    return {"top": int(rows[0]), "bottom": int(h - 1 - rows[-1]),
            "left": int(cols[0]), "right": int(w - 1 - cols[-1])}


def _save(fig, stem):
    """Write the PDF the paper includes and a PNG beside it, then measure.

    A margin of zero means ink runs to the edge of the canvas, which for these
    figures has always meant an axis label wider than the figure. Matplotlib's
    tight bounding box does not grow to hold it, so the label is cut and the
    only evidence is the render.
    """
    pdf = PAPER / f"{stem}.pdf"
    png = PAPER / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=400)
    m = _margins(png)
    tight = min(m.values())
    if tight < MIN_MARGIN_PX:
        CLIPPED.append(stem)
    verdict = "ok" if tight >= MIN_MARGIN_PX else f"CLIPPED, {tight}px"
    print(f"{stem:20s} {verdict:16s} margins {m}")
    return pdf


# ------------------------------------------------------------------ fig 1

def fig_cost_quality():
    """Expert hours against mask IoU, replotting Table 8 unchanged.

    The IoU values follow the recompute of 2026-07-30, so they must stay equal
    to the table's Mask IoU column; `mask_iou_route2.py` patches the list below
    when the SAM 2 row lands.
    """
    methods = ["Manual polygon\ndelineation",
               "SAM 2 box-prompt\n+ manual box",
               "SAM 3 raw\n(no review)",
               "SAM 3 + sieve\n+ 5 s review"]
    hours = [206.7, 53.3, 0.05, 10.0]   # 0 h plotted at 0.05 for the log axis
    iou = [1.00, 0.85, 0.83, 0.83]
    proposed = [False, False, False, True]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for h, q, m, p in zip(hours, iou, methods, proposed):
        ax.scatter(h, q, s=44 if p else 26,
                   color=HIGHLIGHT if p else INK,
                   zorder=3, edgecolor="white", linewidth=0.7)
    # Direct labels, placed to avoid the marks.
    offsets = [(-8, -20), (8, -18), (10, 6), (-6, 12)]
    for (h, q, m, p, off) in zip(hours, iou, methods, proposed, offsets):
        ax.annotate(m, (h, q), textcoords="offset points", xytext=off,
                    ha="center", fontsize=6.4,
                    color=HIGHLIGHT if p else INK,
                    fontweight="bold" if p else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 500)
    ax.set_ylim(0.78, 1.045)
    ax.set_xlabel("Expert hours per 1000 images, log scale")
    ax.set_ylabel("Mask IoU against gold")
    ax.set_xticks([0.05, 1, 10, 100])
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    out = _save(fig, "fig1_cost_quality")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ fig 2

GOLD = {
    # corpus: [(method, MAE), ...] at a fixed 0.5 score threshold
    "Boulders Beach": [("YOLOv11x-seg", 1.01), ("RF-DETR-seg-xlarge", 0.93),
                       ("RT-DETRv4", 0.84), ("RT-DETRv3", 0.84),
                       ("EoMT-Large", 0.83), ("RF-DETR-seg-large", 0.80),
                       ("SAM 3 + sieve", 0.71)],
    "SANParks Algoa": [("YOLOv11x-seg", 9.94), ("RT-DETRv3", 9.08),
                       ("RF-DETR-seg-large", 9.04), ("RT-DETRv4", 8.24),
                       ("RF-DETR-seg-xlarge", 6.12), ("EoMT-Large", 5.73),
                       ("SAM 3 + sieve", 5.65)],
}


def fig_count_mae():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, (corpus, rows) in zip(axes, GOLD.items()):
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [HIGHLIGHT if "SAM 3" in n else MUTED for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, height=0.62, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.2f}", va="center",
                    fontsize=6.6, color=INK)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel("Mean absolute count error")
        ax.set_title(corpus, loc="left", fontweight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    out = _save(fig, "fig2_count_mae")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ fig 3

def fig_count_density():
    """Signed count error against colony density, from the per-image CSVs."""
    src = RUNS / "obj2_counting"
    series = {
        "EoMT-Large": "perimage_african-penguin_eomt-large_b2_e125_i640_be32_fair_grid.csv",
        "DINOv3-M2F-ViT-H": "perimage_african-penguin_dinov3-m2f-vith16plus_b2_e125_i640_be32_fair_grid.csv",
        "DINOv3-M2F-S+": "perimage_african-penguin_dinov3-m2f-smallplus_b4_e125_i640_be32_fair_grid.csv",
    }
    bins = [(1, 1), (2, 5), (6, 20), (21, 400)]
    labels = ["1", "2-5", "6-20", "21+"]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    styles = [("-", INK), ("--", MUTED), (":", MUTED)]
    for (name, fn), (ls, col) in zip(series.items(), styles):
        path = src / fn
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path)))
        ys = []
        for lo, hi in bins:
            sub = [r for r in rows if lo <= int(r["gt_count"]) <= hi]
            ys.append(np.mean([int(r["pred_count"]) - int(r["gt_count"])
                               for r in sub]) if sub else np.nan)
        ax.plot(range(len(bins)), ys, ls, color=col, linewidth=1.3,
                marker="o", markersize=3, zorder=3)
        ax.annotate(name, (len(bins) - 1, ys[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=6.4, color=col, va="center")
    ax.axhline(0, color="#bbbbbb", linewidth=0.7, zorder=2)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.2, len(bins) + 0.9)
    ax.set_xlabel("Ground-truth birds in the image")
    ax.set_ylabel("Signed count error")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    out = _save(fig, "fig3_count_density")
    plt.close(fig)
    return out


def main():
    _style()
    for f in (fig_cost_quality, fig_count_mae, fig_count_density):
        f()
    if CLIPPED:
        print("clipped: " + ", ".join(CLIPPED))
        return 1
    print("all figures clear of the canvas edge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
