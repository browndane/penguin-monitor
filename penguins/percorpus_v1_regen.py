#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Regenerate tab:percorpus_v1 from artefacts that still exist on disk.

WHY
---
The six numbers in P5's tab:percorpus_v1, 88.0/89.4/89.1 for Penguin-Watch and
84.4/86.2/86.7 for the web-scraped corpus, have had no producing artefact since
the provenance audit of 2026-07-11. The gold-200 FiftyOne dataset still holds
v1 gold masks for the Penguin-Watch and iNaturalist halves along with all three
prediction fields, so the table can be rebuilt rather than cut.

The two corpora keep their v1 gold here, so unlike the SANParks and Boulders
re-eval this is not a train/eval-standard mismatch. If the regenerated numbers
land near the published ones the table is confirmed. If they do not, the table
comes out.

PROTOCOL
--------
pycocotools COCOeval segm, single IoU 0.5, maxDets [1,10,100], per-tier through
iscrowd-ignore rather than dropping out-of-tier GT, aggregate = mean of the
three tiers. That is P5's stated metric, and it is prep_13's eval_tier verbatim.

OUTPUT
------
~/bench_any/runs/percorpus_v1_regen/results.json, a FINDINGS section, and the
sentinel _percorpus_v1_regen_done. Nothing polls it.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")

sys.path.insert(0, os.path.expanduser("~/bench_any/datasets/prep"))

import fiftyone as fo  # noqa: E402
import pycocotools.mask as mu  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

DS = "p5_gold200"
TIERS_MAN = os.path.expanduser("~/bench_any/runs/p5_r1_gold/gold200_tiers.json")
OUT = Path(os.path.expanduser("~/bench_any/runs/percorpus_v1_regen"))
FINDINGS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting/FINDINGS.md"))
SENTINEL = OUT / "_percorpus_v1_regen_done"
TIERS = ["easy", "medium", "hard"]

METHODS = [("yolo11x_pred", "YOLOv11x-seg"),
           ("rtdetr_sam3_pred", "RT-DETR + SAM 3 box-prompt"),
           ("sam3_pred", "SAM 3 + sieve (zero-shot)")]

# The cells the paper carried before 2026-07-30, superseded by this script's own output.
# Kept so a re-run still reports how far the regenerated figures sit from the old ones.
PAPER = {("PW", "yolo11x_pred"): 88.0, ("PW", "rtdetr_sam3_pred"): 89.4,
         ("PW", "sam3_pred"): 89.1,
         ("iNat", "yolo11x_pred"): 84.4, ("iNat", "rtdetr_sam3_pred"): 86.2,
         ("iNat", "sam3_pred"): 86.7}


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "percorpus_v1_regen.log", "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def corpus_of(fp):
    b = os.path.basename(fp)
    if b.startswith("PW_"):
        return "PW"
    if b.startswith("1_") and "rf." in b:
        return "SANParks"
    if "rf." in b:
        return "Boulders"
    return "iNat"


def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def collect(ds, tiers_manifest, corpus, pred_field):
    """One pass over the corpus: GT with its tier label, and the predictions."""
    images, gt_rows, preds = [], [], []
    img_id = 0
    for s in ds.iter_samples():
        if corpus_of(s.filepath) != corpus:
            continue
        img_id += 1
        meta = s.metadata
        images.append({"id": img_id, "width": meta.width, "height": meta.height,
                       "file_name": os.path.basename(s.filepath)})
        tier_by_idx = {e["idx"]: e["tier"] for e in tiers_manifest.get(s.filepath, [])}
        gt = s.ground_truth
        if gt and gt.detections:
            for di, d in enumerate(gt.detections):
                mask = d.mask
                if mask is None or mask.sum() == 0:
                    continue
                ys, xs = np.where(mask > 0)
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                gt_rows.append({"image_id": img_id, "segmentation": encode_mask(mask),
                                "bbox": [float(x0), float(y0),
                                         float(max(x1 - x0 + 1, 1)), float(max(y1 - y0 + 1, 1))],
                                "area": float(mask.sum()),
                                "tier": tier_by_idx.get(di)})
        p = s[pred_field] if pred_field in s else None
        if p and p.detections:
            for d in p.detections:
                mask = d.mask
                if mask is None or mask.sum() == 0:
                    continue
                preds.append({"image_id": img_id, "category_id": 1,
                              "segmentation": encode_mask(mask),
                              "score": float(d.confidence) if d.confidence is not None else 1.0})
    return images, gt_rows, preds


def eval_tier(images, gt_rows, preds, tier):
    """iscrowd-ignore per-tier COCOeval, prep_13's eval_tier verbatim."""
    if not preds or not gt_rows:
        return None
    anns = []
    for i, g in enumerate(gt_rows):
        anns.append({"id": i, "image_id": g["image_id"], "category_id": 1,
                     "segmentation": g["segmentation"], "bbox": g["bbox"],
                     "area": g["area"], "iscrowd": 0 if g["tier"] == tier else 1})
    gt_dict = {"images": images, "annotations": anns,
               "categories": [{"id": 1, "name": "penguin"}]}
    pred_list = copy.deepcopy(preds)
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt_dict, gtf)
    gtf.close()
    try:
        coco_gt = COCO(gtf.name)
        coco_dt = coco_gt.loadRes(pred_list)
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
    except Exception as e:
        log(f"    COCOeval ERR: {type(e).__name__}: {str(e)[:160]}")
        return None
    finally:
        os.unlink(gtf.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== tab:percorpus_v1 regeneration start ===")
    ds = fo.load_dataset(DS)
    tiers_manifest = json.load(open(TIERS_MAN))["manifest"]
    results = {}

    for corpus in ("PW", "iNat"):
        for field, label in METHODS:
            images, gt_rows, preds = collect(ds, tiers_manifest, corpus, field)
            tier_ap = {t: eval_tier(images, gt_rows, preds, t) for t in TIERS}
            valid = [tier_ap[t] for t in TIERS if tier_ap[t] is not None]
            agg = float(np.mean(valid)) if len(valid) == 3 else None
            paper = PAPER.get((corpus, field))
            results.setdefault(corpus, {})[field] = {
                "label": label, "tiers": tier_ap, "agg": agg, "paper": paper,
                "delta": None if agg is None else round(agg - paper, 1),
                "n_images": len(images), "n_gt": len(gt_rows), "n_preds": len(preds),
            }
            log(f"  {corpus:5s} {label:28s} agg={agg if agg is None else round(agg, 1)} "
                f"paper={paper} imgs={len(images)} gt={len(gt_rows)} preds={len(preds)}")
            (OUT / "results.json").write_text(json.dumps(results, indent=2))

    out = [f"\n## 24. tab:percorpus_v1 regenerated from the gold-200 v1 masks "
           f"({time.strftime('%Y-%m-%d')})\n\n",
           "Penguin-Watch and iNaturalist keep their v1 gold in the `p5_gold200` FiftyOne\n"
           "dataset, so these two corpora can be re-scored without the v1-trained against\n"
           "v2-cleaned mismatch that confounded the SANParks and Boulders re-eval. Metric\n"
           "is P5's stated one: COCOeval segm at IoU 0.5, per-tier through iscrowd-ignore,\n"
           "aggregate = mean of three tiers.\n\n",
           "| corpus | method | regenerated agg | in paper | delta |\n|---|---|---|---|---|\n"]
    for corpus, per in results.items():
        for field, r in per.items():
            a = "-" if r["agg"] is None else f"{r['agg']:.1f}"
            d = "-" if r["delta"] is None else f"{r['delta']:+.1f}"
            out.append(f"| {corpus} | {r['label']} | {a} | {r['paper']} | {d} |\n")
    out.append("\nThe gold-200 subset is a seed-42 reconstruction of 50 images per corpus, not\n"
               "the original v1 evaluation set, so exact agreement is not expected. Close\n"
               "agreement confirms the published cells. A large gap means the table should\n"
               "carry these numbers or come out.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))

    SENTINEL.write_text("done\n")
    log("=== tab:percorpus_v1 regeneration complete ===")


if __name__ == "__main__":
    main()
