#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Per-method Mask PR overlay on the iNat test split (856 images).

Reconstructs the mask precision-recall curve for each method from retained
predictions via pycocotools, and overlays all on one graph. Methods with
native mask heads only; the two RT-DETR box detectors are excluded because
their masks are SAM2-prompted (the paper's comparability caveat covers this).
"""
import json
import os
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT = Path("~/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_gt_coco.json").expanduser()
OUT = Path("~/paper/figures/fig_pr_overlay.png").expanduser()

# (pred_path, display_label). Native mask heads on the iNat 856 test split.
METHODS = [
    ("~/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid/_bench_predictions_fixed.json", "RF-DETR-XL"),
    ("~/bench_any/runs/segment/african-penguin_rf-detr-seg-large_b2_e150_i640_be64_fair_grid/_bench_predictions_fixed.json", "RF-DETR-L"),
    ("~/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_predictions_resolved.json", "EoMT-L"),
    ("~/bench_any/runs/segment/african-penguin_dinov3-m2f-vith16plus_b2_e150_i640_be64_fair_grid/_bench_predictions_resolved.json", "DINOv3-M2F-H"),
    ("~/bench_any/runs/segment/african-penguin_yolo11x-seg_b8_e150_i640_be64_fork/_bench_predictions_resolved.json", "YOLOv11x"),
    ("~/bench_any/runs/obj2_counting/sam3_inat_test/sam3_inat_test_coco.json", "SAM3 (zero-shot)"),
]


def build_coco_preds(gt, pred_path, is_sam3=False):
    d = json.load(open(pred_path))
    if is_sam3:
        # SAM3 native uses sequential ids; remap by file_name to GT ids.
        fname_to_id = {im["file_name"]: im["id"] for im in gt["images"]}
        sam_fname_to_id = {im["file_name"]: im["id"] for im in d["images"]}
        preds = d["annotations"]
        out = []
        for p in preds:
            sam_id = p["image_id"]
            fn = next(im["file_name"] for im in d["images"] if im["id"] == sam_id)
            out.append({
                "image_id": fname_to_id[fn], "category_id": 1,
                "segmentation": p["segmentation"], "score": float(p.get("score", 1.0)),
            })
        return out
    # bench predictions are a flat list with image_id matching GT.
    return [{
        "image_id": p["image_id"], "category_id": 1,
        "segmentation": p.get("segmentation") or p.get("rle"),
        "score": float(p.get("score", 1.0)),
    } for p in d]


def pr_curve(preds):
    coco_gt = COCO(str(GT))
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.iouThrs = np.array([0.5])
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    pr = ev.eval["precision"][0, :, 0, 0, -1]  # iou 0.5, all areas, maxDet 100
    recall = ev.params.recThrs
    return recall, pr


def main():
    gt = json.load(open(GT))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for path, label in METHODS:
        path = Path(path).expanduser()
        if not path.exists():
            print(f"{label:14s} MISSING: {path}")
            continue
        try:
            is_sam3 = "sam3_inat_test" in str(path)
            preds = build_coco_preds(gt, path, is_sam3=is_sam3)
            r, p = pr_curve(preds)
            valid = p > -1
            ax.plot(r[valid], p[valid], label=label, linewidth=1.5)
            ap = float(np.mean(p[p > -1])) * 100
            print(f"{label:14s} AP@0.5 = {ap:.1f}")
        except Exception as e:
            print(f"{label:14s} FAILED: {type(e).__name__}: {str(e)[:80]}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    # The paper includes this at width=0.7\linewidth, which the preprint class
    # renders at 3.78 in, so 200 dpi landed at 344 effective against the publisher's
    # 500 floor for combination artwork. 300 gives 516.
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
