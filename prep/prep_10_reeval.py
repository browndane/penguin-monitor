"""Per-tier mask mAP@0.5 re-eval on p5_gold200 for the 3 evaluable rows.

Per paper (L170-179, L263): mask mAP@0.5, single IoU 0.5.
Aggregate = arithmetic MEAN of the three tier mAPs (unweighted) - advisor-verified
arithmetic: (95.7+88.3+75.2)/3 = 86.4 exactly.

Per-tier methodology:
  - Each GT detection tagged `tier` in {easy, medium, hard} by prep_09.
  - For tier T: filter GT to T (via filter_labels on ground_truth), keep ALL predictions
    (a pred matching a GT in T is a TP for T; a pred matching a GT NOT in T or no GT
    is a FP for T). This is the standard per-subset COCO eval.
  - Run ds.filter_labels(ground_truth, F("tier")==T).evaluate_detections(pred_field, use_masks=True, iou=0.5, compute_mAP=True)

Rows to eval:
  - SAM3 zero-shot + sieve:        predictions via flagship tool (sam3_prompt "a penguin" + sieve)
  - YOLOv11x-seg (supervised):     ckpt penguin-seg-combined_yolo11x-seg/weights/best.pt
  - RT-DETR + SAM3 box-prompt:     ckpt penguin-seg-combined_rtdetr-x/weights/best.pt + SAM3 box-prompt on rtdetr boxes

Prediction fields must already be on p5_gold200 (each row added as a separate field).
This script assumes pred fields exist: sam3_pred, yolo11x_pred, rtdetr_sam3_pred.
If a field is missing, it skips that row and prints a SKIP notice.

Outputs per row: easy/medium/hard/aggregate mAP@0.5 + paper target for comparison.
Manifest -> ~/bench_any/runs/p5_r1_gold/reeval_results.json
"""
import json, os, sys
from pathlib import Path

import numpy as np
import fiftyone as fo

DS = "p5_gold200"
OUT = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold/reeval_results.json"))

ROWS = [
    {"key": "sam3",      "pred_field": "sam3_pred",      "target": {"easy":95.7,"medium":88.3,"hard":75.2,"agg":86.4}, "label":"SAM3 + sieve (zero-shot)"},
    {"key": "yolo11x",   "pred_field": "yolo11x_pred",   "target": {"easy":95.1,"medium":86.4,"hard":70.9,"agg":84.1}, "label":"YOLOv11x-seg (supervised)"},
    {"key": "rtdetr_s3", "pred_field": "rtdetr_sam3_pred","target": {"easy":96.4,"medium":88.0,"hard":73.8,"agg":86.1}, "label":"RT-DETR + SAM3 box-prompt"},
]
TIERS = ["easy", "medium", "hard"]


def per_tier_mAP(ds, pred_field, tier):
    """Filter GT to tier, eval preds vs filtered GT, return mAP@0.5 (percent)."""
    view = ds.filter_labels("ground_truth", fo.ViewField("tier") == tier)
    # also need preds on all samples; filter_labels only touches ground_truth field
    res = view.evaluate_detections(pred_field, gt_field="ground_truth", use_masks=True, iou=0.5, compute_mAP=True)
    return res.mAP() * 100.0  # mAP is 0-1


def main():
    ds = fo.load_dataset(DS)
    fields = set(ds.get_field_schema().keys())
    results = []
    for row in ROWS:
        pf = row["pred_field"]
        if pf not in fields:
            print(f"SKIP {row['label']}: pred field '{pf}' not on dataset (predictions not generated yet)")
            results.append({**row, "status": "skip_no_pred", "easy":None,"medium":None,"hard":None,"agg":None})
            continue
        print(f"\n=== {row['label']} (pred={pf}) ===", flush=True)
        tier_maps = {}
        for t in TIERS:
            try:
                m = per_tier_mAP(ds, pf, t)
            except Exception as e:
                print(f"  {t}: ERR {type(e).__name__}: {str(e)[:120]}")
                m = None
            tier_maps[t] = m
            print(f"  {t}: {m:.1f}%" if m is not None else f"  {t}: None", flush=True)
        valid = [tier_maps[t] for t in TIERS if tier_maps[t] is not None]
        agg = float(np.mean(valid)) if len(valid)==3 else None
        tgt = row["target"]
        print(f"  AGG: {agg:.1f}% (target {tgt['agg']})" if agg is not None else "  AGG: None")
        if agg is not None:
            deltas = {t: round(tier_maps[t]-tgt[t],1) for t in TIERS}
            print(f"  deltas vs target: {deltas} | agg delta {round(agg-tgt['agg'],1)}")
        results.append({**row, "status":"evaluated", **tier_maps, "agg":agg,
                        "deltas": {t: round(tier_maps[t]-tgt[t],1) for t in TIERS} if agg else None,
                        "agg_delta": round(agg-tgt['agg'],1) if agg else None})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "metric":"mask mAP@0.5, aggregate=mean of 3 tiers"}, open(OUT,"w"), indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
