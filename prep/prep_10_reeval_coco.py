"""Per-tier mask mAP@0.5 re-eval via pycocotools COCOeval with correct ignore semantics.

Metric: COCO-style mask mAP at single IoU 0.5 (paper: mAP_{50}^{mask} at overlap 0.5).
Per-tier methodology (advisor-corrected):
  - Each GT tagged `tier` in {easy,medium,hard} by prep_09.
  - For tier T: GT in T isiscrowd=0 (active); GT NOT in T iscrowd=1 (ignore).
    A pred matching an ignore GT is neither TP nor FP (COCO ignore semantics).
    This is the correct COCO per-subset eval; filter_labels (making non-T GT invisible)
    would wrongly count preds matching non-T GT as FPs (1-4pp agg error).
  - COCOeval(iouThrs=[0.5], maxDets=[1,10,100]) matching FiftyOne/torchmetrics default
    (the paper's stated tool, Algorithm 1). maxDet=100 caps dense colonies (~125 birds/img)
    exactly as the original eval did; reading the precision array directly at maxDet=100
    avoids the _summarize-hardcodes-100 trap.

Aggregate = arithmetic MEAN of the three tier mAPs (unweighted) - advisor arithmetic
(95.7+88.3+75.2)/3 = 86.4 exactly.

Rows (pred fields must exist on p5_gold200):
  - sam3_pred        : SAM3 zero-shot + sieve         target 95.7/88.3/75.2/86.4
  - yolo11x_pred     : YOLOv11x-seg supervised       target 95.1/86.4/70.9/84.1
  - rtdetr_sam3_pred : RT-DETR + SAM3 box-prompt     target 96.4/88.0/73.8/86.1

GT masks encoded full-res as COCO RLE (pycocotools.mask.encode). Pred masks same.
Output: ~/bench_any/runs/p5_r1_gold/reeval_results.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pycocotools.mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import fiftyone as fo

DS = "p5_gold200"
OUT = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold/reeval_results.json"))
TIERS_MAN = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold/gold200_tiers.json"))
TIERS = ["easy", "medium", "hard"]
ROWS = [
    {"key":"sam3","pred_field":"sam3_pred","target":{"easy":95.7,"medium":88.3,"hard":75.2,"agg":86.4},"label":"SAM3 + sieve (zero-shot)"},
    {"key":"yolo11x","pred_field":"yolo11x_pred","target":{"easy":95.1,"medium":86.4,"hard":70.9,"agg":84.1},"label":"YOLOv11x-seg (supervised)"},
    {"key":"rtdetr_s3","pred_field":"rtdetr_sam3_pred","target":{"easy":96.4,"medium":88.0,"hard":73.8,"agg":86.1},"label":"RT-DETR + SAM3 box-prompt"},
]


def encode_mask(mask_bool):
    """bool [H,W] -> COCO RLE dict (counts bytes, Fortran order)."""
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def build_gt_coco(ds, tier, tiers_manifest):
    """Build COCO GT json for a tier: GT in tier -> iscrowd=0 active; else iscrowd=1 ignore.
    tiers_manifest: {filepath: [{idx, tier, score}, ...]} from prep_09 sidecar."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
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
                continue  # det without a computed tier (skip)
            mask = d.mask
            if mask is None or mask.sum() == 0:
                continue
            rle = encode_mask(mask)
            ys, xs = np.where(mask > 0)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bw, bh = max(x1-x0+1,1), max(y1-y0+1,1)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(mask.sum()),
                "iscrowd": 0 if t == tier else 1,  # active only for current tier
                "_tier": t,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id":1,"name":"penguin"}]}


def build_pred_coco(ds, pred_field):
    """Build COCO pred json for a prediction field. All preds active."""
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
            preds.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": float(d.confidence) if d.confidence is not None else 1.0,
            })
    return preds


def eval_tier(gt_dict, pred_list):
    """Run COCOeval at iou 0.5, maxDet 100. Return mAP@0.5 (percent) or None.
    Reads the precision array directly at the maxDet=100 row (index -1 of maxDets),
    avoiding the _summarize-hardcodes-100 trap, which is a hard invariant."""
    if not pred_list:
        return None
    import tempfile, copy
    # loadRes mutates each pred's segmentation.counts in place (str->bytes).
    # The same pred_list is reused across all 3 tiers; without a fresh copy the
    # 2nd/3rd loadRes sees already-mutated RLE dicts and crashes (np broadcast (4,) (0,)).
    pred_list = copy.deepcopy(pred_list)
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt_dict, gtf); gtf.close()
    try:
        coco_gt = COCO(gtf.name)
        coco_dt = coco_gt.loadRes(pred_list)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [1, 10, 100]   # match FiftyOne/torchmetrics default (paper's tool)
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
        # precision shape: [T, R, K, A, M] = [iouThrs, recThrs, catId, areaRng, maxDets]
        # iouThrs=[0.5] -> T=0; K=0 (single cat); A=0 (all); maxDets index for 100 = -1.
        pr = ev.eval["precision"][0, :, 0, 0, -1]
        valid = pr[pr > -1]
        ap = float(np.mean(valid)) if len(valid) else 0.0
        return ap * 100.0
    except Exception as e:
        print(f"  COCOeval ERR: {type(e).__name__}: {str(e)[:160]}")
        return None
    finally:
        os.unlink(gtf.name)


def main():
    ds = fo.load_dataset(DS)
    fields = set(ds.get_field_schema().keys())
    # load tier sidecar manifest
    if not TIERS_MAN.exists():
        print(f"ERROR: tier manifest {TIERS_MAN} not found. Run prep_09_difficulty_tiers.py first.")
        return
    tm = json.load(open(TIERS_MAN))
    tiers_manifest = tm["manifest"]
    print(f"loaded tiers: {tm['tier_counts']} (total {tm['total']})")
    results = []
    for row in ROWS:
        pf = row["pred_field"]
        if pf not in fields:
            print(f"SKIP {row['label']}: pred field '{pf}' missing (generate predictions first)")
            results.append({**row, "status":"skip_no_pred","easy":None,"medium":None,"hard":None,"agg":None})
            continue
        print(f"\n=== {row['label']} (pred={pf}) ===", flush=True)
        pred_list = build_pred_coco(ds, pf)
        print(f"  predictions: {len(pred_list)}", flush=True)
        tier_maps = {}
        for t in TIERS:
            gt_dict = build_gt_coco(ds, t, tiers_manifest)
            n_active = sum(1 for a in gt_dict["annotations"] if a["iscrowd"]==0)
            print(f"  tier {t}: {n_active} active GT", flush=True)
            ap = eval_tier(gt_dict, pred_list)
            tier_maps[t] = ap
            print(f"  {t}: {ap:.1f}%" if ap is not None else f"  {t}: None", flush=True)
        valid = [tier_maps[t] for t in TIERS if tier_maps[t] is not None]
        agg = float(np.mean(valid)) if len(valid)==3 else None
        tgt = row["target"]
        if agg is not None:
            print(f"  AGG: {agg:.1f}% (target {tgt['agg']}, delta {round(agg-tgt['agg'],1)})", flush=True)
            deltas = {t: round(tier_maps[t]-tgt[t],1) for t in TIERS}
            print(f"  per-tier deltas: {deltas}")
        results.append({**row, "status":"evaluated", **tier_maps, "agg":agg,
                        "deltas": ({t: round(tier_maps[t]-tgt[t],1) for t in TIERS} if agg else None),
                        "agg_delta": (round(agg-tgt['agg'],1) if agg else None)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "metric":"pycocotools COCOeval mask mAP@0.5 (single IoU 0.5, maxDet 100); per-tier via iscrowd ignore; aggregate=mean of 3 tiers"}, open(OUT,"w"), indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
