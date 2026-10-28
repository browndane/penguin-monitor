#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Validation curves and stopping points for the supervised segmentation runs.

Reads per-epoch validation traces from whichever sidecar each framework keeps:
  YOLO        mAP75_history.csv      (bench sidecar; drops the final re-val row)
  DINOv3-M2F  graphs/history.csv     (bench graphs sidecar)
  RF-DETR      events.out.tfevents.*  (TensorBoard scalars; epoch from the
                                       ``epoch`` tag, segm/mAP from val/*)
  EoMT        graphs/history.csv     (seg_mAP50 only -- mAP@50, not 50:95)

Plots mask mAP@0.5 (top) and box mAP@0.5 (bottom) -- the one mask metric every
native-mask family logs, so EoMT (which logs no box and no 50:95) plots on the
same axis as the rest. RF-DETR contributes to both panels (it logs
val/segm_mAP_50 and val/mAP_50). EoMT logs only seg_mAP50 (mAP@50), so it
enters the mask panel as an mAP@50 trace, clearly distinct. RT-DETR has no
native mask head (masks come from SAM 3 box-prompt) and is excluded from this
native-mask figure.

Writes ``fig4_training_dynamics.png`` into the P5 directory.

    ~/cvML/bin/python penguins/plot_training_dynamics.py
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = Path.home() / "bench_any" / "runs" / "segment"
OUT = (Path.home() / "paper"
       / "figures")

# YOLO run directory -> label. The AnimalP smoke run is excluded: five epochs,
# never reached the patience rule, and it is not a paper result.
SERIES = [
    ("african-penguin-inaturalist-sam-veryclean_yolo11x-seg_b4_e125_i640_be32",
     "YOLO11x-seg, iNaturalist"),
    ("african-penguin-inaturalist-sam-veryclean_yolo26x-seg_b4_e125_i640_be32",
     "YOLO26x-seg, iNaturalist"),
    ("african-penguin_yolo11x-seg_b0.98_e150_i640_be64_fair_grid",
     "YOLO11x-seg, African penguin"),
    ("african-penguin_yolo26x-seg_b0.98_e150_i640_be64_fair_grid",
     "YOLO26x-seg, African penguin"),
    ("yolo11xseg_comb_be8_v2",
     "YOLO11x-seg, combined dataset"),
]

# Native-augmentation YOLO runs (mosaic + mixup on, the framework default). The
# fair-aug YOLO runs above strip mosaic/mixup and the mask head plateaus early;
# the native runs keep the strong augmentation and the mask head climbs. Both
# ran on the iNaturalist combined set to the same 150-epoch budget, patience 15.
# ``[native]`` marks them in the legend so a reader sees the aug-policy contrast.
NATIVE_YOLO_SERIES = [
    ("african-penguin_yolo11x-seg_b8_e150_i640_be64_native_fig1",
     "YOLO11x-seg [native], African penguin"),
    ("african-penguin_yolo26x-seg_b8_e150_i640_be64_native_fig1",
     "YOLO26x-seg [native], African penguin"),
]

# Mask2Former runs: graphs/history.csv, mask head only, on Penguin-Watch.
M2F_SERIES = [
    ("pw_smallsite_co_dinov3-m2f-smallplus_b4_e125_i640_be8_fair_grid"
     "_pw_core_m2f_smallplus_stride5_2026-08-11",
     "DINOv3-M2F-S+, Penguin-Watch"),
    ("pw_smallsite_co_dinov3-m2f-vith16plus_b2_e125_i640_be8_fair_grid"
     "_pw_core_m2f_vith16plus_stride5_2026-08-11_rog",
     "DINOv3-M2F-ViT-H/16+, Penguin-Watch"),
]

# RF-DETR-seg runs: per-epoch trace is in TensorBoard. Both panels (segm + box).
RFDETR_SERIES = [
    ("african-penguin_rf-detr-seg-large_b2_e150_i640_be64_fair_grid",
     "RF-DETR-seg-L, African penguin"),
    ("african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid",
     "RF-DETR-seg-XL, African penguin"),
]

# EoMT: graphs/history.csv, but only seg_mAP50 (mAP@50) is logged, no 50:95 and
# no box metric. It enters the mask panel as an mAP@50 trace, distinct.
EOMT_SERIES = [
    ("african-penguin_eomt-large_b2_e150_i640_be64_fair_grid",
     "EoMT-L, African penguin"),
]


def load_yolo(run_dir):
    rows = list(csv.DictReader(open(run_dir / "mAP75_history.csv")))[:-1]
    args = (run_dir / "args.yaml").read_text()
    patience = int(re.search(r"^patience: (\d+)", args, re.M).group(1))
    return rows, patience


def load_tb(run_dir):
    """RF-DETR: read val mask + box mAP (both @0.5 and @0.5:0.95) from TB.

    The ``epoch`` tag is logged densely per training step (not per validation),
    so it cannot align to the validation scalars. Use the validation index
    (1..N) as the x-axis. RF-DETR validates once per epoch; N validations = N
    epochs trained (large=59, xlarge=65; both early-stopped under patience 15
    before the 150-epoch budget). Returns dict of traces keyed by tag suffix.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run_dir), size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag, key in (("val/segm_mAP_50", "seg50"),
                     ("val/segm_mAP_50_95", "seg5095"),
                     ("val/mAP_50", "box50"),
                     ("val/mAP_50_95", "box5095")):
        out[key] = [s.value for s in ea.Scalars(tag)]
    n = min(len(v) for v in out.values())
    out["n"] = n
    out["x"] = list(range(1, n + 1))
    return out


def load_history_seg50(run_dir):
    """EoMT: graphs/history.csv, seg_mAP50 (mAP@50) only. Returns
    (epochs, seg50) skipping rows where seg_mAP50 is empty (no eval that epoch)."""
    rows = list(csv.DictReader(open(run_dir / "graphs" / "history.csv")))
    ep, y = [], []
    for r in rows:
        if not r.get("seg_mAP50"):
            continue
        ep.append(int(float(r["epoch"])))
        y.append(float(r["seg_mAP50"]))
    return ep, y


def main():
    # Stacked, drawn at the P5 text width (390 pt = 5.4 in) so fonts print at
    # true size. The old 11-inch pair printed at half scale and its figure
    # legend overlapped the x-labels.
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.0), sharex=True, layout="constrained")
    # Distinct hues, one per family so the legend reads without blending.
    colours = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    m2f_c = ["#e6ab02", "#a6761d"]
    rf_c = ["#1f78b4", "#b2df8a"]
    eomt_c = ["#ff7f00"]
    # Native-aug YOLO: same hues as the fair YOLO11x/26x entries (1b9e77/d95f02)
    # so the pair reads as fair vs native, dotted to mark the policy.
    native_c = ["#1b9e77", "#d95f02"]

    # All traces are mask mAP@0.5 (top) and box mAP@0.5 (bottom). mAP@0.5 is
    # the one mask metric every native-mask family logs, so EoMT (which logs
    # no box and no 50:95) plots on the same axis as the rest with no caveat.
    # YOLO: raw trace only (no rolling mean -- the body argues the mask-head
    # oscillation has no upward trend; smoothing would misrepresent that).
    for (name, label), c in zip(SERIES, colours):
        rows, patience = load_yolo(RUNS / name)
        ep = [int(float(r["epoch"])) for r in rows]
        for ax, key in zip(axes, ("seg_mAP50", "box_mAP50")):
            y = [float(r[key]) for r in rows]
            ax.plot(ep, y, color=c, lw=1.4, label=label if ax is axes[0] else None)
            # Per-panel best: the peak of THIS curve, not a global fitness-best
            # drawn at one epoch across both panels (which sat below the box peak).
            bi = max(range(len(rows)), key=lambda i: y[i])
            ax.plot([ep[bi]], [y[bi]], marker="o", ms=6,
                    color=c, mec="white", mew=1.0, zorder=5)
            ax.axvline(ep[-1], color=c, ls=":", lw=1.0, alpha=0.55)

    # Native-aug YOLO: dotted, same hues as their fair counterparts. The mask
    # head climbs where the fair-aug mask head plateaus -- the figure's evidence
    # that YOLO needs the framework's native augmentation for its mask head.
    for (name, label), c in zip(NATIVE_YOLO_SERIES, native_c):
        rows, patience = load_yolo(RUNS / name)
        ep = [int(float(r["epoch"])) for r in rows]
        for ax, key in zip(axes, ("seg_mAP50", "box_mAP50")):
            y = [float(r[key]) for r in rows]
            ax.plot(ep, y, color=c, lw=1.6, ls=":", label=label if ax is axes[0] else None)
            bi = max(range(len(rows)), key=lambda i: y[i])
            ax.plot([ep[bi]], [y[bi]], marker="o", ms=6,
                    color=c, mec="white", mew=1.0, zorder=5)
            ax.axvline(ep[-1], color=c, ls=":", lw=1.0, alpha=0.55)

    # DINOv3-M2F: mask-only, dashed.
    for (name, label), c in zip(M2F_SERIES, m2f_c):
        rows = list(csv.DictReader(open(RUNS / name / "graphs" / "history.csv")))
        ep = [int(float(r["epoch"])) for r in rows]
        y = [float(r["seg_mAP50"]) for r in rows]
        axes[0].plot(ep, y, color=c, lw=1.8, ls="--", label=label)

    # RF-DETR: both panels, solid. Validates once per epoch; x = validation
    # index, which equals epochs trained (early-stops before the 150 budget).
    for (name, label), c in zip(RFDETR_SERIES, rf_c):
        tb = load_tb(RUNS / name)
        axes[0].plot(tb["x"], tb["seg50"][:tb["n"]], color=c, lw=1.8, label=label)
        axes[1].plot(tb["x"], tb["box50"][:tb["n"]], color=c, lw=1.8)

    # EoMT: mask panel only -- it logs no box metric. Now at mAP@0.5 like all.
    for (name, label), c in zip(EOMT_SERIES, eomt_c):
        ep, y = load_history_seg50(RUNS / name)
        axes[0].plot(ep, y, color=c, lw=1.8, ls="-.", label=label)

    for ax, title in zip(axes, ("Mask", "Box")):
        ax.set_title(f"{title} mAP@0.5 on validation")
        ax.set_ylabel("mAP@0.5")
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].set_xlabel("Epoch (YOLO, EoMT) / validation index (RF-DETR, M2F)")
    # One shared legend below both panels -- it never sits over the fluctuating
    # YOLO traces the in-panel lower-right legend overlapped. Proxy entries carry
    # the peak-marker and final-epoch meaning the old bottom caption spelled out,
    # so the legend is the sole explainer (no rasterised bottom text).
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="o", ms=6, color="0.55",
               mec="white", mew=1.0, ls="none"),
        Line2D([0], [0], color="0.55", ls=":", lw=1.0, alpha=0.7),
    ]
    labels += ["peak of curve", "final epoch"]
    # "outside" reserves the legend its own band under the axes, so it cannot
    # cover the x-label the way the old bbox_to_anchor placement did.
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=7.5)
    p = OUT / "fig4_training_dynamics.png"
    fig.savefig(p, dpi=250, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
