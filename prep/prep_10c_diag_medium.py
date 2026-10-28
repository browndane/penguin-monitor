#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Decisive diagnostic for the R1 medium-tier collapse.

Hypotheses under test (medium tier, all 3 rows):
  H_iscrowd (current, baseline): all GT present, non-tier iscrowd=1. Numbers: sam3 84.7, yolo11x 64.9, rtdetr 65.6.
  H_filter_labels: ONLY medium GT present, non-tier GT dropped entirely (paper's FiftyOne method).
  H_filter_labels+conf0.25: same, but preds filtered to confidence >= 0.25 (sieve threshold).
  H_filter_labels+conf0.5: same, preds conf >= 0.5.

If H_filter_labels jumps supervised medium back toward ~86 -> iscrowd TP-stealing confirmed (advisor right).
If supervised medium stays low under H_filter_labels but jumps under conf threshold -> pred-count/conf is the bug.
Targets: sam3 88.3, yolo11x 86.4, rtdetr 88.0.
"""
import os, sys, json, tempfile, copy
import numpy as np
os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
import fiftyone as fo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mu

DS = "p5_gold200"
TIERS_MAN = "/home/user/bench_any/runs/p5_r1_gold/gold200_tiers.json"
ROWS = [("sam3", "sam3_pred"), ("yolo11x", "yolo11x_pred"), ("rtdetr", "rtdetr_sam3_pred")]

def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle

def build_gt_filtered(ds, tier, tiers_manifest):
    """filter_labels: ONLY tier GT present, non-tier dropped entirely (no iscrowd)."""
    images, annotations = [], []
    img_id = ann_id = 0
    for s in ds.iter_samples():
        img_id += 1
        meta = s.metadata
        images.append({"id": img_id, "width": meta.width, "height": meta.height, "file_name": os.path.basename(s.filepath)})
        gt = s.ground_truth
        if gt is None or not gt.detections:
            continue
        per_det = tiers_manifest.get(s.filepath, [])
        tier_by_idx = {e["idx"]: e["tier"] for e in per_det}
        for di, d in enumerate(gt.detections):
            if tier_by_idx.get(di) != tier:
                continue  # DROP non-tier entirely
            mask = d.mask
            if mask is None or mask.sum() == 0:
                continue
            rle = encode_mask(mask)
            ys, xs = np.where(mask > 0)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bw, bh = max(x1-x0+1,1), max(y1-y0+1,1)
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": rle, "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(mask.sum()), "iscrowd": 0})
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id":1,"name":"penguin"}]}

def build_pred_coco(ds, pred_field, conf_floor=0.0):
    preds = []
    img_id = 0
    for s in ds.iter_samples():
        img_id += 1
        p = s[pred_field] if pred_field in s else None
        if p is None or not p.detections:
            continue
        for d in p.detections:
            c = float(d.confidence) if d.confidence is not None else 1.0
            if c < conf_floor:
                continue
            mask = d.mask
            if mask is None or mask.sum() == 0:
                continue
            rle = encode_mask(mask)
            preds.append({"image_id": img_id, "category_id": 1, "segmentation": rle, "score": c})
    return preds

def eval_one(gt_dict, pred_list):
    if not pred_list:
        return None
    pred_list = copy.deepcopy(pred_list)
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt_dict, gtf); gtf.close()
    try:
        coco_gt = COCO(gtf.name)
        coco_dt = coco_gt.loadRes(pred_list)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [1, 10, 100]
        ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate(); ev.accumulate()
        pr = ev.eval["precision"][0, :, 0, 0, -1]
        valid = pr[pr > -1]
        return float(np.mean(valid)) * 100.0 if len(valid) else 0.0
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:120]}"); return None
    finally:
        os.unlink(gtf.name)

def main():
    ds = fo.load_dataset(DS)
    tm = json.load(open(TIERS_MAN))
    tiers_manifest = tm["manifest"]
    print("=== building MEDIUM tier GT (filter_labels: only medium GT) ===", flush=True)
    gt_med = build_gt_filtered(ds, "medium", tiers_manifest)
    n_active = len(gt_med["annotations"])
    print(f"medium GT (filter_labels): {len(gt_med['images'])} imgs, {n_active} anns (all active, no iscrowd)", flush=True)
    print(f"pred counts per row:", flush=True)
    for key, field in ROWS:
        allp = build_pred_coco(ds, field, 0.0)
        p25 = [p for p in allp if p["score"] >= 0.25]
        p50 = [p for p in allp if p["score"] >= 0.5]
        print(f"  {key}: total={len(allp)} conf>=0.25={len(p25)} conf>=0.5={len(p50)}", flush=True)
    print("\n=== evaluating medium tier ===", flush=True)
    print(f"{'row':<10} {'iscrowd(base)':>14} {'filter_all':>11} {'filter_c0.25':>13} {'filter_c0.5':>12} {'target':>8}", flush=True)
    base = {"sam3": 84.7, "yolo11x": 64.9, "rtdetr": 65.6}
    tgt = {"sam3": 88.3, "yolo11x": 86.4, "rtdetr": 88.0}
    for key, field in ROWS:
        allp = build_pred_coco(ds, field, 0.0)
        r_all = eval_one(gt_med, allp)
        r_25 = eval_one(gt_med, [p for p in allp if p["score"] >= 0.25])
        r_50 = eval_one(gt_med, [p for p in allp if p["score"] >= 0.5])
        def fmt(x): return f"{x:.1f}" if x is not None else "None"
        print(f"{key:<10} {base[key]:>14.1f} {fmt(r_all):>11} {fmt(r_25):>13} {fmt(r_50):>12} {tgt[key]:>8.1f}", flush=True)

if __name__ == "__main__":
    main()
