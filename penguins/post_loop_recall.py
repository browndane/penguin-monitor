#!/usr/bin/env python3
"""Post-loop recall study for P1 (CrossParadigm*.tex line 279).

The annotation pipeline runs detector proposals -> two-expert verification ->
SAM2 refinement. "YOLO-missed instances are recovered by this verification
step before SAM2 refinement" (post-loop recall). We measure it directly:

  YOLOv11x-seg (student) is run on the 856-image test split. A GT instance is
  "missed" if no YOLO pred box reaches IoU >= 0.5 with it. The pipeline's
  proposer is RF-DETR-Seg-XL; among the YOLO-missed GT instances, we count how
  many RF-DETR-Seg-XL recovers at IoU >= 0.5 (i.e. would surface to the human
  verification step). That fraction is the post-loop recall.

Run with the server venv (pycocotools not needed; pure box-IoU):
  ~/.virtualenvs/cvML/bin/python penguins/post_loop_recall.py
"""
import json
from collections import defaultdict

YOLO_RD = "/home/vorse/bench_any/runs/segment/african-penguin_yolo11x-seg_b0.98_e150_i640_be64_fair_grid/test"
RF_RD   = "/home/vorse/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid"
IOU_HIT = 0.50


def box_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    gt = json.load(open(f"{YOLO_RD}/_bench_gt_coco.json"))
    yolo = json.load(open(f"{YOLO_RD}/_bench_predictions_catfixed.json"))
    rf = json.load(open(f"{RF_RD}/_bench_predictions_fixed.json"))

    def group(preds):
        g = defaultdict(list)
        for p in preds:
            g[p["image_id"]].append(p["bbox"])
        return g

    yolo_by = group(yolo)
    rf_by = group(rf)

    total = len(gt["annotations"])
    yolo_missed = 0
    rf_recovered = 0
    for ann in gt["annotations"]:
        img = ann["image_id"]
        gbox = ann["bbox"]
        if max((box_iou(gbox, pb) for pb in yolo_by[img]), default=0.0) < IOU_HIT:
            yolo_missed += 1
            if max((box_iou(gbox, pb) for pb in rf_by[img]), default=0.0) >= IOU_HIT:
                rf_recovered += 1

    recovery = rf_recovered / yolo_missed if yolo_missed else 0.0
    out = {
        "what": "post-loop recall: fraction of YOLOv11x-missed GT instances "
                "recovered by RF-DETR-Seg-XL (pipeline proposer) at IoU>=0.5",
        "n_gt_instances": total,
        "yolo11x_missed": yolo_missed,
        "yolo11x_recall": round(1 - yolo_missed / total, 3),
        "rf_detr_xl_recovered_of_missed": rf_recovered,
        "post_loop_recovery_rate": round(recovery, 3),
        "pipeline_recall": round((total - yolo_missed + rf_recovered) / total, 3),
        "iou_threshold": IOU_HIT,
        "sources": {"yolo_preds": f"{YOLO_RD}/_bench_predictions_catfixed.json",
                    "rf_preds": f"{RF_RD}/_bench_predictions_fixed.json",
                    "gt": f"{YOLO_RD}/_bench_gt_coco.json"},
    }
    print(json.dumps(out, indent=2))
    import os
    op = "penguins/research/post_loop_recall.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"\n[saved] {op}")


if __name__ == "__main__":
    main()
