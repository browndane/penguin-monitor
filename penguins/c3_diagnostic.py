#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Phase C3 diagnostic: WHY do the papers' difficulty-vs-failure r (0.379-0.526)
differ from C3's recomputed r (~0.05-0.21)?

Tests 3 hypotheses on the eomt-large run (native mask head, surviving preds, cat_id=1):
  A. AP-gap correlation: difficulty vs CONTINUOUS per-instance AP-gap (1 - per-instance AP@0.5),
     not binary failure. P1 may report this.
  B. IoU threshold: matching at IoU 0.75 vs 0.5.
  C. Binary failure r (the C3 baseline) for direct comparison.

All on the 856-image / 5695-instance iNaturalist-veryclean test split, Proposed weights.
CPU only.
"""
import json, os
import numpy as np
from scipy import stats
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT = "/home/user/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_gt_coco.json"
PRED = "/home/user/bench_any/runs/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_predictions_resolved.json"

# Difficulty constants (ap_difficulty.py L52-58)
A_REF, N_MAX, ALPHA, BETA, GAMMA = 0.04, 8.0, 0.40, 0.35, 0.25
TAU_1, TAU_2 = 0.30, 0.58


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def diff_factors(gt_boxes, W, H):
    n = len(gt_boxes)
    S = np.zeros(n); O = np.zeros(n); D = np.zeros(n)
    areas = [max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) for b in gt_boxes]
    rel = [a / (W*H) for a in areas]
    cx = [(b[0]+b[2])/2 for b in gt_boxes]
    cy = [(b[1]+b[3])/2 for b in gt_boxes]
    diag = [((b[2]-b[0])**2 + (b[3]-b[1])**2)**0.5 for b in gt_boxes]
    for i in range(n):
        S[i] = 1 - min(rel[i]/A_REF, 1.0)
        O[i] = max((iou(gt_boxes[i], gt_boxes[j]) for j in range(n) if j != i), default=0.0)
        r = 2.0 * diag[i]
        D[i] = min((1 + sum(1 for j in range(n) if j != i and ((cx[i]-cx[j])**2+(cy[i]-cy[j])**2)**0.5 <= r)) / N_MAX, 1.0)
    return S, O, D


def load_gt(path):
    c = json.load(open(path))
    id2im = {im["id"]: im for im in c["images"]}
    by_img = {}
    wh = {}
    for a in c["annotations"]:
        im = id2im[a["image_id"]]
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append((float(x), float(y), float(x+w), float(y+h)))
        wh[a["image_id"]] = (int(im["width"]), int(im["height"]))
    return by_img, wh


def per_instance_ap_gap(gt_path, pred_path, iou_thr=0.5):
    """For each GT instance: 1 - (its per-instance AP). Uses pycocotools per-GT match.
    Per-instance AP = 1.0 if matched (AP contribution), else 0; gap = 0 if matched, 1 if missed.
    BUT we also compute a CONTINUOUS version: per-instance 'quality' = best-matching det score,
    gap = 1 - best_score. This is the continuous analogue.
    """
    coco_gt = COCO(gt_path)
    with open(pred_path) as f:
        anns = json.load(f)
    gt_cats = set(coco_gt.getCatIds())
    if gt_cats and all(a.get("category_id") not in gt_cats for a in anns):
        anns = [{**a, "category_id": next(iter(gt_cats))} for a in anns]
    coco_dt = coco_gt.loadRes(anns)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.iouThrs = [iou_thr]
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    # binary: matched or not, per GT, via gtMatches
    # gtMatches[T][G] = matched det id (0 = unmatched)
    by_img = {}
    for e in ev.evalImgs:
        if e is None: continue
        gtm = np.asarray(e["gtMatches"][0])  # T=0 (iou_thr), G GTs
        dtscores = np.asarray(e["dtScores"])
        dtm = np.asarray(e["dtMatches"][0])
        # for each GT, its matched det's score (0 if unmatched)
        g = []
        for gidx in range(gtm.shape[0] if gtm.ndim > 0 else (1 if gtm.size else 0)):
            mid = gtm[gidx] if gtm.ndim > 0 else 0
            sc = 0.0
            if mid > 0:
                # find the det index with this id
                loc = np.where(dtm == mid)[0]
                if len(loc) > 0:
                    sc = float(dtscores[loc[0]])
            g.append(sc)
        by_img[e["image_id"]] = g
    return by_img


def main():
    gt_by_img, wh = load_gt(GT)
    print(f"GT: {len(gt_by_img)} images, {sum(len(v) for v in gt_by_img.values())} instances")

    # --- Hypothesis C: binary failure r (C3 baseline) at IoU 0.5 ---
    print("\n=== Hypothesis C: binary failure r (Proposed weights) ===")
    for iou_match in [0.5, 0.75]:
        diffs, fails = [], []
        with open(PRED) as f:
            preds = json.load(f)
        gt_cats = {1}
        if all(a.get("category_id") not in gt_cats for a in preds):
            preds = [{**a, "category_id": 1} for a in preds]
        # group preds by image
        pred_by_img = {}
        for a in preds:
            x, y, w, h = a["bbox"]
            pred_by_img.setdefault(a["image_id"], []).append(((x, y, x+w, y+h), a["score"]))
        n_det = 0
        for img_id, gts in gt_by_img.items():
            W, H = wh[img_id]
            S, O, D = diff_factors(gts, W, H)
            ps = pred_by_img.get(img_id, [])
            ps.sort(key=lambda p: -p[1])
            used = set()
            for i, g in enumerate(gts):
                best_iou, best_k = iou_match - 1e-9, -1
                for k, (pb, sc) in enumerate(ps):
                    if k in used: continue
                    ii = iou(g, pb)
                    if ii > best_iou:
                        best_iou, best_k = ii, k
                det = best_k >= 0 and best_iou >= iou_match
                if det:
                    used.add(best_k); n_det += 1
                diffs.append(ALPHA*S[i] + BETA*O[i] + GAMMA*D[i])
                fails.append(0.0 if det else 1.0)
        r, _ = stats.pearsonr(diffs, fails)
        print(f"  IoU match {iou_match}: binary-failure Pearson r = {r:.4f} (n={len(diffs)}, detect {n_det}/{len(diffs)}={n_det/len(diffs):.3f})")

    # --- Hypothesis A: continuous AP-gap = 1 - matched-det-score ---
    print("\n=== Hypothesis A: continuous (1 - matched det score) gap r ===")
    for iou_thr in [0.5, 0.75]:
        sc_by_img = per_instance_ap_gap(GT, PRED, iou_thr)
        diffs, gaps = [], []
        n_matched = 0
        for img_id, gts in gt_by_img.items():
            W, H = wh[img_id]
            S, O, D = diff_factors(gts, W, H)
            scores = sc_by_img.get(img_id, [])
            for i in range(len(gts)):
                sc = scores[i] if i < len(scores) else 0.0
                if sc > 0: n_matched += 1
                diffs.append(ALPHA*S[i] + BETA*O[i] + GAMMA*D[i])
                gaps.append(1.0 - sc)
        r, _ = stats.pearsonr(diffs, gaps)
        print(f"  IoU {iou_thr}: (1-matched-score) gap Pearson r = {r:.4f} (n={len(diffs)}, matched {n_matched})")

    # --- Hypothesis A2: per-instance AP @ 0.5 (continuous, 101-pt per image) vs difficulty ---
    # Using the per-image AP from bootstrap_ci style, mapped to each instance in that image
    print("\n=== Hypothesis A2: per-IMAGE AP-gap (1 - per-image AP@0.5) r ===")
    coco_gt = COCO(GT)
    with open(PRED) as f:
        anns = json.load(f)
    if all(a.get("category_id") != 1 for a in anns):
        anns = [{**a, "category_id": 1} for a in anns]
    coco_dt = coco_gt.loadRes(anns)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    img_ap = {}
    for e in ev.evalImgs:
        if e is None: continue
        npig = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
        if npig == 0:
            img_ap[e["image_id"]] = 0.0; continue
        dtm = np.asarray(e["dtMatches"][0]); dtig = np.asarray(e["dtIgnore"][0])
        tps = np.logical_and(dtm, np.logical_not(dtig)).astype(float)
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig)).astype(float)
        if tps.size == 0:
            img_ap[e["image_id"]] = 0.0; continue
        tp = np.cumsum(tps); fp = np.cumsum(fps); rec = tp / npig
        prec = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), 0.0)
        ap = sum(float(prec[rec >= r].max()) if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)) / 101
        img_ap[e["image_id"]] = ap
    diffs, gaps = [], []
    for img_id, gts in gt_by_img.items():
        W, H = wh[img_id]
        S, O, D = diff_factors(gts, W, H)
        ap = img_ap.get(img_id, 0.0)
        for i in range(len(gts)):
            diffs.append(ALPHA*S[i] + BETA*O[i] + GAMMA*D[i])
            gaps.append(1.0 - ap)
    r, _ = stats.pearsonr(diffs, gaps)
    print(f"  per-image AP@0.5-gap Pearson r = {r:.4f} (n={len(diffs)})  [paper claims 0.526 for eomt]")

    print("\n=== SUMMARY ===")
    print(f"eomt-large paper claim: difficulty-vs-failure r = 0.526 (decoder/NMS-free) or 0.379 (NMS-based)")
    print(f"C3 binary failure r (Proposed): ~0.213")
    print(f"Decision: if AP-gap r ~ 0.5, papers used continuous AP-gap, not binary failure.")


if __name__ == "__main__":
    main()
