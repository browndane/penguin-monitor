#!/usr/bin/env python3
"""Test whether tab:percorpus_v1's published cells are an un-stratified aggregate.

WHY
---
percorpus_v1_regen.py re-scored the six cells as P5's stated metric, the mean of
three per-tier mAP@0.5 values, and every cell came in 13.8 to 30.5 points below
the published number. The tier mix on the gold-200 subset is heavily skewed:
Penguin-Watch carries 40 easy, 1120 medium and 224 hard instances, the
web-scraped corpus 51, 331 and 12. An unweighted mean over three tiers of that
shape is dominated by the sparsest tier, and the hard tier is where the gap sits.

The paper says at line 163 that every figure comes from the same 200-image gold
subset, so the sample is not the explanation. The remaining candidate is the
aggregation: a single COCOeval over all ground truth, with no tier split, is
instance-weighted and would sit near the medium tier rather than a third of the
way to the hard one. This script computes that and compares.

If the un-stratified numbers reproduce the published cells, tab:percorpus_v1 is
sound and its caption is what needs correcting. If they do not, the cells have no
reproducible protocol.

OUTPUT
------
~/bench_any/runs/percorpus_v1_regen/unstratified.json and a printed result block.
Short enough to run in the foreground; nothing polls it.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from percorpus_v1_regen import (  # noqa: E402
    DS, METHODS, PAPER, TIERS_MAN, collect,
)

import fiftyone as fo  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

OUT = Path(os.path.expanduser("~/bench_any/runs/percorpus_v1_regen"))


def eval_all(images, gt_rows, preds):
    """One COCOeval over every ground-truth instance, no tier split."""
    if not preds or not gt_rows:
        return None
    anns = [{"id": i, "image_id": g["image_id"], "category_id": 1,
             "segmentation": g["segmentation"], "bbox": g["bbox"],
             "area": g["area"], "iscrowd": 0}
            for i, g in enumerate(gt_rows)]
    gt_dict = {"images": images, "annotations": anns,
               "categories": [{"id": 1, "name": "penguin"}]}
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt_dict, gtf)
    gtf.close()
    try:
        coco_gt = COCO(gtf.name)
        coco_dt = coco_gt.loadRes(copy.deepcopy(preds))
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [1, 10, 100]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
        pr = ev.eval["precision"][0, :, 0, 0, -1]
        valid = pr[pr > -1]
        return float(np.mean(valid)) * 100.0 if len(valid) else 0.0
    finally:
        os.unlink(gtf.name)


def main():
    ds = fo.load_dataset(DS)
    tiers_manifest = json.load(open(TIERS_MAN))["manifest"]
    results = {}

    print("=== RESULT BLOCK: tab:percorpus_v1 un-stratified ===", flush=True)
    print(f"{'corpus':7s} {'method':28s} {'unstrat':>8s} {'paper':>7s} {'delta':>7s}", flush=True)
    for corpus in ("PW", "iNat"):
        for field, label in METHODS:
            images, gt_rows, preds = collect(ds, tiers_manifest, corpus, field)
            agg = eval_all(images, gt_rows, preds)
            paper = PAPER.get((corpus, field))
            delta = None if agg is None else round(agg - paper, 1)
            results.setdefault(corpus, {})[field] = {
                "label": label, "unstratified": agg, "paper": paper, "delta": delta,
                "n_images": len(images), "n_gt": len(gt_rows), "n_preds": len(preds)}
            print(f"{corpus:7s} {label:28s} {agg:8.1f} {paper:7.1f} {delta:+7.1f}", flush=True)
    (OUT / "unstratified.json").write_text(json.dumps(results, indent=2))
    print("=== END RESULT BLOCK ===", flush=True)


if __name__ == "__main__":
    main()
