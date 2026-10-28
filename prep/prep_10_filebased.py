# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Phase 1 prep_10 (file-based): per-tier mask mAP@0.5 re-eval via pycocotools COCOeval.

Metric: COCO-style mask mAP at single IoU 0.5 (paper: mAP_{50}^{mask} at overlap 0.5).
Per-tier methodology (advisor-corrected, UNCHANGED from prep_10):
  - Each GT tagged `tier` in {easy,medium,hard} by prep_09.
  - For tier T: GT in T is iscrowd=0 (active); GT NOT in T is iscrowd=1 (ignore).
    A pred matching an ignore GT is neither TP nor FP (COCO ignore semantics).
  - COCOeval(iouThrs=[0.5], maxDets=[1,10,100]) matching FiftyOne/torchmetrics default.
    Reading precision[0,:,:,0,-1] dodges the _summarize-hardcodes-100 trap.
  - copy.deepcopy(pred_list) before every loadRes (loadRes mutates preds; 2nd call crashes).

Aggregate = arithmetic MEAN of the three tier mAPs (unweighted) - advisor arithmetic
(95.7+88.3+75.2)/3 = 86.4 exactly.

File-based: reads tiers+gold RLEs from prep_09 sidecar ({corpus}_tiers.json) and preds
from prep_11 ({corpus}_sam3_pred.json). All keyed by image FILENAME. Corpora SEPARATE.

Row: sam3_pred (SAM3 zero-shot + sieve) target 95.7/88.3/75.2/86.4.

Output: ~/bench_any/runs/p5_r1_v2/<corpus>/<corpus>_reeval_sam3.json
"""
import json, os, sys, argparse, tempfile, copy
from pathlib import Path
import numpy as np
import pycocotools.mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

TIERS = ["easy", "medium", "hard"]
TARGET = {"easy": 95.7, "medium": 88.3, "hard": 75.2, "agg": 86.4}


def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def build_gt_coco(tiers_manifest, tier):
    """Build COCO GT json for a tier: GT in tier -> iscrowd=0 active; else iscrowd=1 ignore.
    tiers_manifest: {filename: [{idx, tier, rle, bbox[x0,y0,x1,y1], area, width, height}, ...]}.
    Reads pre-computed RLEs from prep_09 (no re-rasterization). Keyed by FILENAME."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    for fname, dets in tiers_manifest.items():
        img_id += 1
        if not dets:
            images.append({"id": img_id, "width": 0, "height": 0, "file_name": fname})
            continue
        W = dets[0]["width"]; H = dets[0]["height"]
        images.append({"id": img_id, "width": W, "height": H, "file_name": fname})
        for e in dets:
            t = e["tier"]
            rle = e["rle"]
            x0, y0, x1, y1 = e["bbox"]
            bw = x1 - x0; bh = y1 - y0
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(e["area"]),
                "iscrowd": 0 if t == tier else 1,
                "_tier": t,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "penguin"}]}


def build_pred_coco(preds_per_img, tiers_manifest):
    """Build COCO pred json from {filename: [{rle, score, box}, ...]}.
    Map filenames to img_ids consistent with build_gt_coco ordering."""
    fname_to_id = {}
    img_id = 0
    for fname in tiers_manifest.keys():
        img_id += 1
        fname_to_id[fname] = img_id
    preds = []
    for fname, plist in preds_per_img.items():
        iid = fname_to_id.get(fname)
        if iid is None:
            continue
        for p in plist:
            preds.append({
                "image_id": iid,
                "category_id": 1,
                "segmentation": p["rle"],
                "score": float(p["score"]),
            })
    return preds


def eval_tier(gt_dict, pred_list):
    """Run COCOeval at iou 0.5, maxDet 100. Return mAP@0.5 (percent) or None.
    Reads the precision array directly at the maxDet=100 row (index -1 of maxDets),
    avoiding the _summarize-hardcodes-100 trap, which is a hard invariant."""
    if not pred_list:
        return None
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tiers", required=True, help="prep_09 tiers JSON")
    ap.add_argument("--preds", required=True, help="prep_11 preds JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tm = json.load(open(args.tiers))
    tiers_manifest = tm["manifest"]
    print(f"loaded tiers: {tm['tier_counts']} (total {tm['total']})", flush=True)

    preds_data = json.load(open(args.preds))
    preds_per_img = preds_data["preds"]
    chosen = preds_data.get("chosen_config", {})
    print(f"preds: {preds_data['n_preds']} total, config={chosen}", flush=True)

    pred_list = build_pred_coco(preds_per_img, tiers_manifest)
    print(f"  COCO preds built: {len(pred_list)}", flush=True)

    tier_maps = {}
    tier_n = {}
    for t in TIERS:
        gt_dict = build_gt_coco(tiers_manifest, t)
        n_active = sum(1 for a in gt_dict["annotations"] if a["iscrowd"] == 0)
        tier_n[t] = n_active
        print(f"  tier {t}: {n_active} active GT", flush=True)
        ap_t = eval_tier(gt_dict, pred_list)
        tier_maps[t] = ap_t
        print(f"  {t}: {ap_t:.1f}%" if ap_t is not None else f"  {t}: None", flush=True)

    valid = [tier_maps[t] for t in TIERS if tier_maps[t] is not None]
    agg = float(np.mean(valid)) if len(valid) == 3 else None
    # Instance-weighted aggregate beside the unweighted one; under terciles the
    # two coincide to a rounding error.
    scored = [t for t in TIERS if tier_maps[t] is not None]
    w = sum(tier_n[t] for t in scored)
    agg_iw = (sum(tier_maps[t] * tier_n[t] for t in scored) / w) if w else None
    if agg is not None:
        print(f"  AGG: {agg:.1f}% (target {TARGET['agg']}, delta {round(agg - TARGET['agg'], 1)})", flush=True)
        deltas = {t: round(tier_maps[t] - TARGET[t], 1) for t in TIERS}
        print(f"  per-tier deltas: {deltas}", flush=True)

    # recall (overall, from greedy match) - quick separate calc
    def quick_recall():
        from pycocotools import mask as _mu
        n_gt = 0; n_tp = 0
        for fname, dets in tiers_manifest.items():
            glds = [e["rle"] for e in dets]
            n_gt += len(glds)
            preds = preds_per_img.get(fname, [])
            if not preds:
                continue
            preds_sorted = sorted(preds, key=lambda p: -p["score"])
            used = [False] * len(glds)
            for p in preds_sorted:
                pm = _mu.decode(p["rle"])
                best = 0.0; best_g = -1
                for gi, grle in enumerate(glds):
                    if used[gi]:
                        continue
                    gm = _mu.decode(grle)
                    iou = float(np.logical_and(pm, gm).sum()) / max(float(np.logical_or(pm, gm).sum()), 1)
                    if iou > best:
                        best = iou; best_g = gi
                    del gm
                if best_g >= 0 and best >= 0.5:
                    used[best_g] = True; n_tp += 1
                del pm
        return (n_tp / n_gt) if n_gt > 0 else 0.0
    rec = quick_recall()
    print(f"  recall (overall, IoU 0.5 greedy): {rec:.4f}", flush=True)

    result = {
        "corpus": args.corpus,
        "chosen_config": chosen,
        "easy": tier_maps.get("easy"),
        "medium": tier_maps.get("medium"),
        "hard": tier_maps.get("hard"),
        "agg": agg,
        "agg_iw": agg_iw,
        "n_gt_per_tier": tier_n,
        "tiers_file": args.tiers,
        "recall": round(rec, 4),
        "target": TARGET,
        "deltas": ({t: round(tier_maps[t] - TARGET[t], 1) for t in TIERS} if agg else None),
        "agg_delta": (round(agg - TARGET["agg"], 1) if agg else None),
        "metric": "pycocotools COCOeval mask mAP@0.5 (single IoU 0.5, maxDet 100); per-tier via iscrowd ignore; aggregate=mean of 3 tiers",
    }
    json.dump(result, open(out, "w"), indent=2)
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
