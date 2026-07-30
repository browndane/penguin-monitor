#!/usr/bin/env python3
"""Mask-mAP of the prompt union against the gold COCO GT, no GPU.

The two prompts' SAM3 mask predictions are already on disk. Merge them per
image, dedup with the sieve + class-agnostic NMS (one mask per bird), build a
COCO pred json whose image-id scheme matches the existing gt_coco.json, and run
pycocotools segm eval. Compares 'a penguin', 'penguin' and union at the
threshold where the count probe found union best (0.40).

Self-writes results to prompt_union_maskmAP.txt.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from count_decompose import _rle_from_ann
from sanity_sieve import mask_passes, class_agnostic_nms

RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
THR = 0.40
OUT = RUNS / "prompt_union_maskmAP.txt"


def box_from_rle(rle):
    bb = mu.toBbox(rle)
    return (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))


def coco_segm_eval(gt_path, pred_json):
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(pred_json) if pred_json else COCO()
    if not pred_json:
        return 0.0, 0.0
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    mAP50 = ev.stats[1]   # mAP @ IoU=0.50
    mAP5095 = ev.stats[0] # mAP @ IoU=0.50:0.95
    return float(mAP50), float(mAP5095)


def build_pred(corpus, prompt_stem, thr):
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    fn_to_iid = {im["file_name"]: im["id"] for im in gt["images"]}
    src = json.loads((RUNS / "sam3_prompts_gold" / f"{corpus}_{prompt_stem}.json").read_text())
    src_fn = {i["id"]: i["file_name"] for i in src["images"]}
    per = defaultdict(list)
    for a in src["annotations"]:
        if float(a.get("score", 1.0)) < thr:
            continue
        fn = src_fn.get(a["image_id"])
        iid = fn_to_iid.get(fn)
        if iid is None:
            continue
        h = next(im["height"] for im in gt["images"] if im["id"] == iid)
        w = next(im["width"] for im in gt["images"] if im["id"] == iid)
        rle = _rle_from_ann(a, h, w)
        bb = mu.toBbox(rle)
        box = (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))
        per[iid].append((box, float(a.get("score", 1.0)), rle, h, w))
    # sieve + NMS
    preds = []
    for iid, items in per.items():
        kept = [(b, s, r, h, w) for (b, s, r, h, w) in items
                if mask_passes(None, (w, h), bbox=b)]
        if not kept:
            continue
        boxes = [k[0] for k in kept]
        scores = [k[1] for k in kept]
        rles = [k[2] for k in kept]
        idx = class_agnostic_nms(boxes, scores, 0.55)
        for i in idx:
            r = rles[i]
            rr = dict(r)
            if isinstance(rr["counts"], bytes):
                rr["counts"] = rr["counts"].decode("utf-8")
            preds.append({"image_id": iid, "category_id": 1,
                          "bbox": list(mu.toBbox(r)), "score": 1.0,
                          "segmentation": rr})
    return preds


def build_union(corpus, thr):
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    fn_to_iid = {im["file_name"]: im["id"] for im in gt["images"]}
    per = defaultdict(list)
    for stem in ("penguin", "a_penguin"):
        src = json.loads((RUNS / "sam3_prompts_gold" / f"{corpus}_{stem}.json").read_text())
        src_fn = {i["id"]: i["file_name"] for i in src["images"]}
        for a in src["annotations"]:
            if float(a.get("score", 1.0)) < thr:
                continue
            fn = src_fn.get(a["image_id"])
            iid = fn_to_iid.get(fn)
            if iid is None:
                continue
            h = next(im["height"] for im in gt["images"] if im["id"] == iid)
            w = next(im["width"] for im in gt["images"] if im["id"] == iid)
            rle = _rle_from_ann(a, h, w)
            bb = mu.toBbox(rle)
            box = (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))
            per[iid].append((box, float(a.get("score", 1.0)), rle, h, w))
    preds = []
    for iid, items in per.items():
        kept = [(b, s, r, h, w) for (b, s, r, h, w) in items
                if mask_passes(None, (w, h), bbox=b)]
        if not kept:
            continue
        boxes = [k[0] for k in kept]
        scores = [k[1] for k in kept]
        rles = [k[2] for k in kept]
        idx = class_agnostic_nms(boxes, scores, 0.55)
        for i in idx:
            r = rles[i]
            rr = dict(r)
            if isinstance(rr["counts"], bytes):
                rr["counts"] = rr["counts"].decode("utf-8")
            preds.append({"image_id": iid, "category_id": 1,
                          "bbox": list(mu.toBbox(r)), "score": 1.0,
                          "segmentation": rr})
    return preds


def main():
    lines = [f"# Prompt-union mask mAP, thr={THR}, 2026-07-29\n\n",
             "SAM3 raw masks from each prompt, sieve + class-agnostic NMS deduped,\n",
             "pycocotools segm eval against the gold COCO GT. No GPU, no SAM2 refine.\n\n",
             "| corpus | cfg | mask mAP@50 | mask mAP@50-95 |\n|---|---|---|---|\n"]
    for corpus, label in CORPORA.items():
        gt_path = RUNS / f"sam2refine_{corpus}" / "gt_coco.json"
        tmp = RUNS / f"_tmp_pred.json"
        for name, fn in (("penguin", lambda: build_pred(corpus, "penguin", THR)),
                         ("a penguin", lambda: build_pred(corpus, "a_penguin", THR)),
                         ("union", lambda: build_union(corpus, THR))):
            preds = fn()
            tmp.write_text(json.dumps(preds))
            mAP50, mAP5095 = coco_segm_eval(gt_path, preds)
            lines.append(f"| {label} | {name} | {mAP50:.3f} | {mAP5095:.3f} |\n")
            print(f"{label} {name}: mAP50={mAP50:.3f} mAP50-95={mAP5095:.3f} ({len(preds)} preds)")
        lines.append("| | | | |\n")
    OUT.write_text("".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
