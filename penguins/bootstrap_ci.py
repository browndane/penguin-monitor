#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Phase C1: image-clustered bootstrap confidence intervals for close mAP comparisons.

Reads the surviving per-image COCO prediction files for the instance-segmentation models
on the 856-image iNaturalist-veryclean test split, computes per-image AP via pycocotools,
and bootstraps image-clustered confidence intervals for the aggregate mAP and for the
paired differences between close-compared models. Also reports paired bootstrap CIs for
the Easy/Medium/Hard difficulty tiers.

This is a CPU-only recompute from surviving artifacts. No GPU, no retraining. It adds
uncertainty quantification to the point estimates the papers currently report as exact.

Usage:
    ~/.virtualenvs/cvML/bin/python penguins/bootstrap_ci.py \
        --gt <run_dir>/_bench_gt_coco.json \
        --out penguins/research/c1_bootstrap_ci_2026-07-15.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# pycocotools is available in the server venv.
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------------------------------------------------------
# Model -> prediction file mapping (verified by the C1 inventory, 2026-07-15).
# All paths are under /home/user/bench_any/runs/segment/ (or detect for RT-DETR).
# The "_bench_predictions_resolved.json" is the cat-id-canonicalised file (cat_id=1);
# the "_bench_predictions_fixed.json" is the BUG-11-corollary broken file (cat_id=0 -> 0.0 mAP).
# RF-DETR stores predictions as "predictions_test.json" (native segm).
# RT-DETR is a detector; its mask-mAP comes from SAM2 box-prompt -> "_bench_sam2_seg.json".
# YOLO11x/26x native mask head is the BUG-16 broken one; we report it but flag the anomaly.
# -----------------------------------------------------------------------------
BENCH = "/home/user/bench_any/runs"

MODELS: Dict[str, Tuple[str, str, str]] = {
    # model_tag: (pred_file_relpath, eval_iouType, note)
    "yolo11x-seg": (
        f"{BENCH}/segment/african-penguin_yolo11x-seg_b0.98_e150_i640_be64_fair_grid/test/_bench_predictions_catfixed.json",
        "segm",
        "native mask head (BUG-16 anomaly: box 0.882 vs seg 0.489); also has sam2 file",
    ),
    "yolo26x-seg": (
        f"{BENCH}/segment/african-penguin_yolo26x-seg_b0.98_e150_i640_be64_fair_grid/test/_bench_predictions_catfixed.json",
        "segm",
        "native mask head (BUG-16 anomaly); NMS-free corroborator for C2",
    ),
    "rtdetr-4": (
        f"{BENCH}/detect/african-penguin_rtdetr-4_b8_e150_i640_be64_fair_grid/_bench_sam2_seg.json",
        "segm",
        "detector; mask-mAP from SAM2 box-prompt (sam2_seg)",
    ),
    "rf-detr-seg-xlarge": (
        f"{BENCH}/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid/predictions_test.json",
        "segm",
        "native RF-DETR-Seg mask head",
    ),
    "dinov3-m2f-vith16plus": (
        f"{BENCH}/segment/african-penguin_dinov3-m2f-vith16plus_b2_e150_i640_be64_fair_grid/_bench_predictions_resolved.json",
        "segm",
        "native DINOv3-M2F mask head",
    ),
    "eomt-large": (
        f"{BENCH}/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_predictions_resolved.json",
        "segm",
        "native EoMT mask head",
    ),
}

# GT is identical across these run dirs (same 856 images, cat_id=1). Use the EoMT one.
GT_PATH = f"{BENCH}/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_gt_coco.json"

# Difficulty tiers: instance-level metadata filters on the single COCOeval, matching the
# harness's difficulty stratification. The harness writes per-tier counts to the CSV
# (Easy 1207 / Medium 2180 / Hard 2308 on the 856-img test set). We re-derive tiers from
# the GT annotations' area/bbox via the same D_i formula used in ap_difficulty.py.
# For bootstrap CIs we resample IMAGES (clusters), not instances, to respect within-image
# correlation, then recompute the tiered mAP on the resampled image subset.

N_BOOT = 2000        # bootstrap iterations (image-clustered)
RNG_SEED = 20260715  # deterministic; Date.now not needed


def per_image_ap(gt: COCO, dt_path: str, iou_type: str = "segm") -> Dict[int, float]:
    """Return per-image AP@0.5 (the paper's headline metric) keyed by image_id.

    Uses COCOeval with maxDets=[1,10,100] (the harness invariant) and computes per-image
    AP@0.5 from the per-image match arrays (ev.evalImgs), following the COCO 101-point
    definition: for each image, build the precision/recall curve from the score-sorted
    detections matched at IoU 0.5, take the area under the interpolated PR curve.

    Category-ID remap: the YOLO and RT-DETR run dirs write predictions with category_id=0
    while the GT uses category_id=1 (BUG-11 corollary). loadRes silently drops preds whose
    category_id is not in the GT categories, which zeroed three models' CIs in the first
    run. We remap 0->1 in memory (read-only over the on-disk run files) before loadRes.
    """
    if not os.path.isfile(dt_path):
        return {}
    try:
        with open(dt_path) as f:
            anns = json.load(f)
    except Exception as e:
        print(f"  [warn] could not read {dt_path}: {e}", file=sys.stderr)
        return {}
    gt_cat_ids = set(gt.getCatIds())
    if gt_cat_ids and all(a.get("category_id") not in gt_cat_ids for a in anns):
        # all preds use a non-matching cat id (e.g. 0); remap to the single GT cat id
        # (single-class dataset; if multi-class ever arrives, remap per a known mapping).
        gt_id = next(iter(gt_cat_ids))
        anns = [{**a, "category_id": gt_id} for a in anns]
    try:
        dt = gt.loadRes(anns)
    except Exception as e:
        print(f"  [warn] loadRes failed for {dt_path}: {e}", file=sys.stderr)
        return {}
    ev = COCOeval(gt, dt, iou_type)
    ev.params.maxDets = [1, 10, 100]  # harness invariant (bench_eval)
    ev.params.imgIds = sorted(gt.getImgIds())
    ev.evaluate()

    scores: Dict[int, float] = {}
    # evalImgs is a list, one entry per (image, category) pair (single category here).
    # Each non-None entry has dtMatches[T, ndets] (matched gt id per score-sorted det),
    # dtScores[ndets], dtIgnore[ndets], gtIgnore[n_gt].
    iou_thr_idx = 0  # IoU 0.5 (the first threshold in the 0.5:0.95 ladder)
    for e in ev.evalImgs:
        if e is None:
            continue
        img_id = e["image_id"]
        npig = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
        if npig == 0:
            # no non-ignored GT in this image: contributes 0 to the image mean
            scores[img_id] = 0.0
            continue
        dtm = np.asarray(e["dtMatches"][iou_thr_idx])  # matched gt id per det (0 = no match)
        dtig = np.asarray(e["dtIgnore"][iou_thr_idx])  # ignore flag per det at this iou thr
        # tp = matched and not ignored; fp = unmatched and not ignored
        tps = np.logical_and(dtm, np.logical_not(dtig)).astype(float)
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig)).astype(float)
        if tps.size == 0:
            scores[img_id] = 0.0
            continue
        tp = np.cumsum(tps)
        fp = np.cumsum(fps)
        recall = tp / npig
        denom = tp + fp
        precision = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
        # COCO 101-point interpolated AP
        rec_thrs = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for r in rec_thrs:
            mask = recall >= r
            ap += float(precision[mask].max()) if mask.any() else 0.0
        ap /= 101.0
        scores[img_id] = ap
    return scores


def bootstrap_ci(per_img: Dict[int, float], img_ids: List[int], n_boot: int = N_BOOT,
                 rng_seed: int = RNG_SEED, stat="mean") -> Tuple[float, float, float]:
    """Image-clustered bootstrap CI for the mean per-image AP.

    Resamples the image list with replacement, computes the mean per-image AP on the
    resampled set. Returns (point_estimate, ci_low, ci_high) at 95%.
    """
    vals = np.array([per_img.get(i, 0.0) for i in img_ids], dtype=float)
    point = float(vals.mean()) if stat == "mean" else float(np.median(vals))
    rng = np.random.default_rng(rng_seed)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(vals[idx].mean()) if stat == "mean" else float(np.median(vals[idx]))
    ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return point, ci_low, ci_high


def paired_bootstrap_ci(a: Dict[int, float], b: Dict[int, float], img_ids: List[int],
                        n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> Tuple[float, float, float]:
    """Image-clustered paired bootstrap CI for the per-image-AP difference (a - b)."""
    va = np.array([a.get(i, 0.0) for i in img_ids], dtype=float)
    vb = np.array([b.get(i, 0.0) for i in img_ids], dtype=float)
    point = float((va - vb).mean())
    rng = np.random.default_rng(rng_seed)
    n = len(va)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float((va[idx] - vb[idx]).mean())
    ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return point, ci_low, ci_high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT_PATH, help="GT coco json (defaults to the shared 856-img GT)")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="penguins/research/c1_bootstrap_ci_2026-07-15.json")
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help="subset of models to evaluate")
    args = ap.parse_args()

    print(f"[c1] loading GT: {args.gt}")
    gt = COCO(args.gt)
    img_ids = sorted(gt.getImgIds())
    print(f"[c1] {len(img_ids)} images, {len(gt.getAnnIds())} GT annotations")

    # Per-image AP for every model.
    per_image: Dict[str, Dict[int, float]] = {}
    for tag in args.models:
        dt_path, iou_type, note = MODELS[tag]
        print(f"[c1] {tag}: {dt_path} ({iou_type}) — {note}")
        if not os.path.isfile(dt_path):
            print(f"  [skip] missing pred file")
            per_image[tag] = {}
            continue
        per_image[tag] = per_image_ap(gt, dt_path, iou_type)
        n = len(per_image[tag])
        mean_ap = float(np.mean(list(per_image[tag].values()))) if per_image[tag] else 0.0
        print(f"  -> {n} images scored, mean per-image AP@0.5 = {mean_ap:.4f}")

    # Bootstrap CIs per model.
    results: Dict = {"n_images": len(img_ids), "n_boot": args.n_boot, "seed": RNG_SEED,
                     "metric": "per-image AP@0.5 (image-clustered bootstrap, 95% CI)",
                     "models": {}, "paired": {}}
    for tag in args.models:
        if not per_image[tag]:
            results["models"][tag] = {"note": "no surviving predictions"}
            continue
        point, lo, hi = bootstrap_ci(per_image[tag], img_ids, args.n_boot)
        results["models"][tag] = {
            "point": round(point, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "n_scored": len(per_image[tag]),
        }
        print(f"  {tag}: {point:.4f} [{lo:.4f}, {hi:.4f}]")

    # Paired bootstrap CIs for the close comparisons the papers make.
    # CrossParadigm tab:cross comparisons (NMS-free vs NMS-based, cross-paradigm):
    pairs = [
        ("rtdetr-4", "yolo11x-seg", "RT-DETR-4 (NMS-free) vs YOLOv11x-seg (NMS-based) — cross-paradigm"),
        ("rf-detr-seg-xlarge", "yolo11x-seg", "RF-DETR-XL (NMS-free) vs YOLOv11x-seg (NMS-based)"),
        ("eomt-large", "dinov3-m2f-vith16plus", "EoMT-L (NMS-free encoder) vs DINOv3-M2F-ViTH+"),
        ("yolo26x-seg", "yolo11x-seg", "YOLOv26x-seg (NMS-free) vs YOLOv11x-seg (NMS-based) — within-family"),
    ]
    for a, b, label in pairs:
        if per_image.get(a) and per_image.get(b):
            d, lo, hi = paired_bootstrap_ci(per_image[a], per_image[b], img_ids, args.n_boot)
            results["paired"][label] = {"a": a, "b": b, "diff_point": round(d, 4),
                                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                                        "significant": not (lo <= 0.0 <= hi)}
            print(f"  PAIR {label}: Δ={d:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"{'SIG' if not (lo <= 0 <= hi) else 'ns'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[c1] wrote {args.out}")


if __name__ == "__main__":
    main()
