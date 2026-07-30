#!/usr/bin/env python3
"""Per-corpus breakdown of the YOLOv11x medium-tier collapse.
Which corpus drives the -22pp gap? v2-fixed corpora (SANParks/Boulders) or PW/iNat?"""
import os, sys, json, tempfile, copy
import numpy as np
os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
import fiftyone as fo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mu

DS = "p5_gold200"
TIERS_MAN = "/home/vorse/bench_any/runs/p5_r1_gold/gold200_tiers.json"

def corpus_of(fp):
    b = os.path.basename(fp)
    if b.startswith("PW_"): return "PW"
    if b.startswith("1_") and "rf." in b: return "SANParks"  # roboflow-normalized sanparks
    if "rf." in b: return "Boulders"  # roboflow boulders
    return "iNat"

def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle

def eval_subset(ds, tiers_manifest, pred_field, corpus_filter=None, tier="medium"):
    images, annotations, preds = [], [], []
    img_id = ann_id = 0
    for s in ds.iter_samples():
        corp = corpus_of(s.filepath)
        if corpus_filter and corp != corpus_filter:
            continue
        img_id += 1
        meta = s.metadata
        images.append({"id": img_id, "width": meta.width, "height": meta.height, "file_name": os.path.basename(s.filepath)})
        gt = s.ground_truth
        per_det = tiers_manifest.get(s.filepath, [])
        tier_by_idx = {e["idx"]: e["tier"] for e in per_det}
        if gt and gt.detections:
            for di, d in enumerate(gt.detections):
                if tier_by_idx.get(di) != tier:
                    continue
                mask = d.mask
                if mask is None or mask.sum() == 0:
                    continue
                rle = encode_mask(mask)
                ys, xs = np.where(mask > 0)
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bw, bh = max(x1-x0+1,1), max(y1-y0+1,1)
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                    "segmentation": rle, "bbox": [float(x0),float(y0),float(bw),float(bh)],
                    "area": float(mask.sum()), "iscrowd": 0})
                ann_id += 1
        p = s[pred_field] if pred_field in s else None
        if p and p.detections:
            for d in p.detections:
                mask = d.mask
                if mask is None or mask.sum() == 0:
                    continue
                rle = encode_mask(mask)
                preds.append({"image_id": img_id, "category_id": 1, "segmentation": rle,
                              "score": float(d.confidence) if d.confidence is not None else 1.0})
    if not preds or not annotations:
        return None, len(images), len(annotations), len(preds)
    preds = copy.deepcopy(preds)
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"images": images, "annotations": annotations, "categories":[{"id":1,"name":"x"}]}, gtf); gtf.close()
    try:
        coco_gt = COCO(gtf.name); coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.iouThrs = np.array([0.5]); ev.params.maxDets = [1,10,100]
        ev.params.areaRng = [[0,1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate(); ev.accumulate()
        pr = ev.eval["precision"][0,:,0,0,-1]
        v = pr[pr>-1]
        return (float(np.mean(v))*100.0 if len(v) else 0.0), len(images), len(annotations), len(preds)
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:100]}"); return None, len(images), len(annotations), len(preds)
    finally:
        os.unlink(gtf.name)

def main():
    ds = fo.load_dataset(DS)
    tm = json.load(open(TIERS_MAN)); tiers_manifest = tm["manifest"]
    print("=== YOLOv11x medium tier, per-corpus (filter_labels, iscrowd=0) ===", flush=True)
    print(f"{'corpus':<10} {'mAP50':>7} {'imgs':>5} {'GT':>5} {'preds':>6}", flush=True)
    for corp in ["iNat", "PW", "SANParks", "Boulders"]:
        ap, ni, ng, npd = eval_subset(ds, tiers_manifest, "yolo11x_pred", corpus_filter=corp, tier="medium")
        print(f"{corp:<10} {(f'{ap:.1f}' if ap is not None else 'None'):>7} {ni:>5} {ng:>5} {npd:>6}", flush=True)
    print("\n=== SAM3 medium tier, per-corpus ===", flush=True)
    print(f"{'corpus':<10} {'mAP50':>7} {'imgs':>5} {'GT':>5} {'preds':>6}", flush=True)
    for corp in ["iNat", "PW", "SANParks", "Boulders"]:
        ap, ni, ng, npd = eval_subset(ds, tiers_manifest, "sam3_pred", corpus_filter=corp, tier="medium")
        print(f"{corp:<10} {(f'{ap:.1f}' if ap is not None else 'None'):>7} {ni:>5} {ng:>5} {npd:>6}", flush=True)

if __name__ == "__main__":
    main()
