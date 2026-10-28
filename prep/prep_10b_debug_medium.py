#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Reproduce the medium-tier COCOeval crash with a FULL traceback.
Runs ONLY the medium tier for the sam3_pred row, no try/except."""
import os, sys, json, traceback
import numpy as np
os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
import fiftyone as fo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mu

DS = "p5_gold200"
TIERS_MAN = "/home/user/bench_any/runs/p5_r1_gold/gold200_tiers.json"

def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle

def build_gt_coco(ds, tier, tiers_manifest):
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
            t = tier_by_idx.get(di)
            if t is None:
                continue
            mask = d.mask
            if mask is None or mask.sum() == 0:
                continue
            rle = encode_mask(mask)
            ys, xs = np.where(mask > 0)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bw, bh = max(x1-x0+1,1), max(y1-y0+1,1)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": rle, "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(mask.sum()), "iscrowd": 0 if t == tier else 1, "_tier": t,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id":1,"name":"penguin"}]}

def build_pred_coco(ds, pred_field):
    preds = []
    img_id = 0
    for s in ds.iter_samples():
        img_id += 1
        p = s[pred_field] if pred_field in s else None
        if p is None or not p.detections:
            continue
        for d in p.detections:
            mask = d.mask
            if mask is None or mask.sum() == 0:
                continue
            rle = encode_mask(mask)
            preds.append({"image_id": img_id, "category_id": 1, "segmentation": rle,
                          "score": float(d.confidence) if d.confidence is not None else 1.0})
    return preds

def main():
    ds = fo.load_dataset(DS)
    tm = json.load(open(TIERS_MAN))
    tiers_manifest = tm["manifest"]
    print(f"building GT for medium tier...", flush=True)
    gt = build_gt_coco(ds, "medium", tiers_manifest)
    n_active = sum(1 for a in gt["annotations"] if a["iscrowd"] == 0)
    n_ignore = sum(1 for a in gt["annotations"] if a["iscrowd"] == 1)
    print(f"GT: {len(gt['images'])} imgs, {len(gt['annotations'])} anns ({n_active} active, {n_ignore} ignore)", flush=True)
    print(f"building preds for sam3_pred...", flush=True)
    preds = build_pred_coco(ds, "sam3_pred")
    print(f"preds: {len(preds)}", flush=True)
    import tempfile
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt, gtf); gtf.close()
    print(f"GT written to {gtf.name} ({os.path.getsize(gtf.name)//1024//1024} MB)", flush=True)
    print(f"--- NOW RUNNING COCOeval (no try/except) ---", flush=True)
    coco_gt = COCO(gtf.name)
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.iouThrs = np.array([0.5])
    ev.params.maxDets = [1, 10, 100]
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    print("evaluate()...", flush=True)
    ev.evaluate()
    print("accumulate()...", flush=True)
    ev.accumulate()
    pr = ev.eval["precision"][0, :, 0, 0, -1]
    valid = pr[pr > -1]
    ap = float(np.mean(valid)) * 100.0 if len(valid) else 0.0
    print(f"medium AP = {ap:.1f}%", flush=True)
    os.unlink(gtf.name)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
