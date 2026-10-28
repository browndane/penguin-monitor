#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Count-error decomposition: WHY the predicted count is wrong, not just by how much.

WHY
---
Objective 2 of the roadmap asks for count error split into *missed* and
*merged* per difficulty tier, not a single undifferentiated error number.
Those two failure modes point at opposite fixes. A detector that misses
small birds needs resolution or a better backbone; a detector that fuses
two adjacent birds into one mask needs a different suppression or mask
head. Reporting only MAE hides which one you have.

Four events are counted per image, and they are exhaustive over the
mismatch between prediction and ground truth:

  matched   one prediction sits on one GT instance at IoU >= --iou
  missed    a GT instance no prediction claims          -> UNDER-count
  merged    one prediction covers >= 2 GT instances     -> UNDER-count
  split     >= 2 predictions sit inside one GT instance -> OVER-count
  spurious  a prediction on no GT instance at all       -> OVER-count

Merge and split are detected by containment, not by IoU. Two penguins
fused into one mask give a prediction whose IoU with each GT is around
0.5 or below, so an IoU-only test scores it as two misses and never sees
the merge. Containment asks a different question: how much of this GT
instance lies inside that prediction.

INPUT
-----
A run directory containing `_bench_gt_coco.json` and
`_bench_predictions_resolved.json` (both written by `bench_eval.py`), or
those two paths given explicitly. Optionally a difficulty CSV from
`ap_difficulty.py` to stratify the output by tier.

OUTPUT
------
- per-image CSV: gt_count, pred_count, error, and the five event counts
- aggregate to stdout and CSV: MAE, RMSE, over/under rate, accuracy@exact,
  accuracy@+-1, and each event as a rate per GT instance
- with --sweep_score, the same aggregate at a range of score thresholds,
  since counting needs a threshold and mAP does not

This is NOT a smoke test: it reads JSON, writes CSV/stdout. No model, no
orchestrator, no GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


# ---------------------------------------------------------------- loading

def _rle_from_ann(ann, h, w):
    """Return a single RLE for one COCO annotation, polygon or RLE alike."""
    seg = ann.get("segmentation")
    if seg is None:
        # Box-only prediction. Fall back to the box as a rectangle so
        # detection-only models stay comparable with the mask models.
        x, y, bw, bh = ann["bbox"]
        poly = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
        rles = mask_utils.frPyObjects(poly, h, w)
        return mask_utils.merge(rles)
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        return mask_utils.merge(rles)
    if isinstance(seg, dict):
        counts = seg.get("counts")
        if isinstance(counts, list):
            return mask_utils.frPyObjects(seg, h, w)
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return rle
    raise ValueError(f"unrecognised segmentation type: {type(seg)}")


def _load_gt(path: Path):
    """Return {image_id: {'hw': (h, w), 'rles': [...], 'file': str}}."""
    coco = json.loads(Path(path).read_text())
    meta = {im["id"]: (im["height"], im["width"], im.get("file_name", ""))
            for im in coco["images"]}
    out = {iid: {"hw": (hw[0], hw[1]), "rles": [], "boxes": [], "file": hw[2]}
           for iid, hw in meta.items()}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        iid = ann["image_id"]
        if iid not in out:
            continue
        h, w, _ = meta[iid]
        out[iid]["rles"].append(_rle_from_ann(ann, h, w))
        out[iid]["boxes"].append(ann["bbox"])
    return out


def _load_preds(path: Path, gt, min_score: float):
    """Return {image_id: [rle, ...]} for predictions scoring >= min_score."""
    raw = json.loads(Path(path).read_text())
    by_img = defaultdict(list)
    for p in raw:
        if float(p.get("score", 1.0)) < min_score:
            continue
        iid = p["image_id"]
        g = gt.get(iid)
        if g is None:
            continue
        h, w = g["hw"]
        by_img[iid].append(_rle_from_ann(p, h, w))
    return by_img


# ------------------------------------------------------------- difficulty

def _tiers_for_image(boxes, h, w, tau1=None, tau2=None):
    """Easy/Medium/Hard per GT instance, using ap_difficulty's own scorer.

    Importing the scorer rather than restating the equations means the tier
    boundaries here cannot drift from the ones P1 and P2 report. tau1/tau2
    override Eq. 5's boundaries, for asking how much the tier story depends
    on where they were drawn.
    """
    from ap_difficulty import (BBox, Instance, compute_difficulty_factors,
                               assign_difficulty_tiers, TAU_1, TAU_2)
    insts = []
    for (x, y, bw, bh) in boxes:
        insts.append(Instance(
            bbox=BBox(x_center=(x + bw / 2) / w, y_center=(y + bh / 2) / h,
                      width=bw / w, height=bh / h),
            image_id="", instance_id=0))
    compute_difficulty_factors(insts, w, h)
    assign_difficulty_tiers(insts,
                            tau1=TAU_1 if tau1 is None else tau1,
                            tau2=TAU_2 if tau2 is None else tau2)
    return [i.difficulty_tier for i in insts]


# ----------------------------------------------------------- decomposition

def _containment(pred_rles, gt_rles):
    """Return (frac_of_gt_in_pred, frac_of_pred_in_gt), both (n_pred, n_gt)."""
    n_p, n_g = len(pred_rles), len(gt_rles)
    if n_p == 0 or n_g == 0:
        return np.zeros((n_p, n_g)), np.zeros((n_p, n_g))
    # maskUtils.iou(A, B, iscrowd=1) returns intersection / area(A), the
    # *first* argument, because a crowd region contributes nothing to the
    # union. Getting this direction backwards silently turns merges into
    # splits, so the two calls are named for what they actually return.
    frac_pred_in_gt = np.asarray(
        mask_utils.iou(pred_rles, gt_rles, [1] * n_g), dtype=np.float64
    ).reshape(n_p, n_g)
    frac_gt_in_pred = np.asarray(
        mask_utils.iou(gt_rles, pred_rles, [1] * n_p), dtype=np.float64
    ).reshape(n_g, n_p).T
    return frac_gt_in_pred, frac_pred_in_gt


def _decompose_image(pred_rles, gt_rles, iou_thr: float, contain_thr: float):
    """Classify one image's predictions against its ground truth.

    Returns a dict of event counts. Matching is one-to-one and optimal
    (Hungarian on the IoU matrix), so a prediction cannot be matched twice
    and the leftover sets are unambiguous.
    """
    n_p, n_g = len(pred_rles), len(gt_rles)
    ev = {"matched": 0, "missed": 0, "merged": 0, "split": 0, "spurious": 0,
          "merged_gt_absorbed": 0, "split_extra_preds": 0, "merged_unmatched": 0}
    # What happened to each GT instance, so an outcome can be attributed to
    # a difficulty tier rather than only counted. The four fates partition GT
    # exactly (see the assert below), so being split cannot be one of them: a
    # GT instance can be matched *and* split. It rides alongside as a flag.
    fate = ["missed"] * n_g
    was_split = [False] * n_g
    if n_g == 0:
        ev["spurious"] = n_p
        ev["_fate"], ev["_split"] = fate, was_split
        return ev
    if n_p == 0:
        ev["missed"] = n_g
        ev["_fate"], ev["_split"] = fate, was_split
        return ev

    iou = np.asarray(
        mask_utils.iou(pred_rles, gt_rles, [0] * n_g), dtype=np.float64
    ).reshape(n_p, n_g)

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] >= iou_thr
    m_pred, m_gt = set(rows[keep].tolist()), set(cols[keep].tolist())
    ev["matched"] = int(keep.sum())
    for j in m_gt:
        fate[j] = "matched"

    unmatched_pred = [i for i in range(n_p) if i not in m_pred]
    unmatched_gt = [j for j in range(n_g) if j not in m_gt]

    frac_gt_in_pred, frac_pred_in_gt = _containment(pred_rles, gt_rles)

    # A merge is one prediction substantially covering two or more GT
    # instances. Only the GT instances left *unmatched* were actually lost
    # to the merge; a GT that found its own prediction elsewhere was not
    # swallowed, however much some other loose mask overlaps it. Counting
    # those too would let matched + missed + absorbed exceed the GT total.
    # A merging prediction sits on real birds, so it is not a false
    # positive. It represents one of the instances it covers and absorbs
    # the rest. When the assignment already matched it to one of them, the
    # match is that representation and everything else it swallowed is
    # absorbed.
    absorbed_gt = set()
    merge_represented = set()
    for i in range(n_p):
        covered = [j for j in range(n_g) if frac_gt_in_pred[i, j] >= contain_thr]
        lost = [j for j in covered
                if j not in m_gt and j not in absorbed_gt and j not in merge_represented]
        if len(covered) < 2 or not lost:
            continue
        ev["merged"] += 1
        if i in m_pred:
            ev["merged_gt_absorbed"] += len(lost)
            absorbed_gt.update(lost)
            for j in lost:
                fate[j] = "absorbed"
        else:
            ev["merged_unmatched"] += 1
            merge_represented.add(lost[0])
            fate[lost[0]] = "merge_represented"
            ev["merged_gt_absorbed"] += len(lost) - 1
            absorbed_gt.update(lost[1:])
            for j in lost[1:]:
                fate[j] = "absorbed"
            m_pred.add(i)

    # A split is one GT instance holding two or more predictions. The
    # surplus predictions drive the over-count, and by the same argument
    # only the unmatched ones are surplus.
    split_preds = set()
    for j in range(n_g):
        inside = [i for i in range(n_p) if frac_pred_in_gt[i, j] >= contain_thr]
        extra = [i for i in inside if i not in m_pred and i not in split_preds]
        if len(inside) >= 2 and extra:
            ev["split"] += 1
            ev["split_extra_preds"] += len(extra)
            split_preds.update(extra)
            was_split[j] = True

    # Whatever is left over once merge and split are accounted for.
    ev["missed"] = len([j for j in unmatched_gt
                        if j not in absorbed_gt and j not in merge_represented])
    ev["spurious"] = len([i for i in range(n_p)
                          if i not in m_pred and i not in split_preds])

    # Both sides must balance exactly, or the decomposition is lying.
    assert (ev["matched"] + ev["merged_unmatched"] + ev["merged_gt_absorbed"]
            + ev["missed"]) == n_g, f"GT accounting broken: {ev} vs n_gt={n_g}"
    ev["_fate"], ev["_split"] = fate, was_split
    assert (ev["matched"] + ev["merged_unmatched"] + ev["split_extra_preds"]
            + ev["spurious"]) == n_p, f"pred accounting broken: {ev} vs n_pred={n_p}"
    return ev


# --------------------------------------------------------------- reporting

def _aggregate(per_image):
    n = len(per_image)
    if n == 0:
        return None
    errs = [r["pred_count"] - r["gt_count"] for r in per_image]
    abs_errs = [abs(e) for e in errs]
    gt_total = sum(r["gt_count"] for r in per_image)
    agg = {
        "n_images": n,
        "gt_instances": gt_total,
        "pred_instances": sum(r["pred_count"] for r in per_image),
        "MAE_count": round(sum(abs_errs) / n, 4),
        "RMSE_count": round(math.sqrt(sum(e * e for e in errs) / n), 4),
        "mean_signed_error": round(sum(errs) / n, 4),
        "overcount_rate": round(sum(1 for e in errs if e > 0) / n, 4),
        "undercount_rate": round(sum(1 for e in errs if e < 0) / n, 4),
        "accuracy_exact": round(sum(1 for e in errs if e == 0) / n, 4),
        "accuracy_within1": round(sum(1 for e in abs_errs if e <= 1) / n, 4),
    }
    for k in ("matched", "missed", "merged", "merged_unmatched", "split",
              "spurious", "merged_gt_absorbed", "split_extra_preds"):
        tot = sum(r[k] for r in per_image)
        agg[f"{k}_total"] = tot
        agg[f"{k}_per_gt"] = round(tot / gt_total, 5) if gt_total else 0.0
    return agg


from ap_difficulty import TAU_1, TAU_2

_FATES = ("matched", "missed", "absorbed", "merge_represented")


def _tier_columns(tiers):
    """Flatten the tier table into CSV columns, one per tier and outcome.

    Rates are per GT instance *within the tier*, which is the whole point:
    a raw Hard-tier miss count says nothing when Hard is also the rarest tier.
    """
    out = {}
    for tier in ("Easy", "Medium", "Hard"):
        n = tiers.get(tier, {}).get("total", 0)
        low = tier.lower()
        out[f"{low}_gt"] = n
        for fate in _FATES + ("split",):
            c = tiers.get(tier, {}).get(fate, 0)
            out[f"{low}_{fate}"] = c
            out[f"{low}_{fate}_rate"] = round(c / n, 4) if n else 0.0
    out["spurious_untiered"] = tiers.get("untiered", {}).get("spurious", 0)
    return out


def _run(gt_path, pred_path, min_score, iou_thr, contain_thr, by_tier=False,
         tau1=None, tau2=None):
    gt = _load_gt(gt_path)
    preds = _load_preds(pred_path, gt, min_score)
    rows, tiers = [], defaultdict(lambda: defaultdict(int))
    for iid, g in gt.items():
        p = preds.get(iid, [])
        ev = _decompose_image(p, g["rles"], iou_thr, contain_thr)
        fate = ev.pop("_fate", [])
        was_split = ev.pop("_split", [])
        if by_tier:
            # A spurious prediction covers no GT instance, so it has no tier.
            # Counted for every image, including the ones with no GT at all.
            tiers["untiered"]["spurious"] += ev["spurious"]
            if g["rles"]:
                h, w = g["hw"]
                per_img = defaultdict(int)
                for t, f, s in zip(_tiers_for_image(g["boxes"], h, w, tau1, tau2),
                                   fate, was_split):
                    tiers[t][f] += 1
                    tiers[t]["total"] += 1
                    per_img[f"{t}_{f}"] += 1
                    per_img[f"{t}_total"] += 1
                    if s:
                        tiers[t]["split"] += 1
                        per_img[f"{t}_split"] += 1
                # Per-image tier counts, so a bootstrap can resample images
                # without re-decomposing every draw.
                ev.update(per_img)
        rows.append({"image_id": iid, "file": g["file"],
                     "gt_count": len(g["rles"]), "pred_count": len(p), **ev})
    return (rows, tiers) if by_tier else rows


def _resolve_run_dir(run_dir):
    """Find the GT and prediction JSON inside a run dir, whichever wrapper wrote it.

    Only DINOv3-M2F and EoMT use the flat `_bench_predictions_resolved.json` this
    script originally assumed. RF-DETR writes `predictions_test.json` at the root,
    Ultralytics writes into a `<split>/` subdir under string image ids that
    bench_eval then rewrites. Preference order matters: an ID-fixed file matches the
    GT's integer image ids, a raw one overlaps it by zero images and scores as a
    total miss rather than an error.
    """
    pred_names = ("_bench_predictions_catfixed.json",     # YOLO: id-fixed then cat-fixed
                  "_bench_predictions_resolved.json",     # DINOv3-M2F: category 0→1
                  "_bench_predictions_fixed.json",        # id-fixed only
                  "predictions_test.json",                # RF-DETR, already int ids
                  "predictions.json")                     # last resort, may be raw
    gt = next((p for p in sorted(run_dir.rglob("_bench_gt_coco.json"))
               if not any(part.startswith("_superseded_") for part in p.parts)), None)
    if gt is None:
        return run_dir / "_bench_gt_coco.json", run_dir / pred_names[1]
    # Prefer a prediction file beside the GT — for YOLO both live in <split>/.
    for name in pred_names:
        for cand in (gt.parent / name, run_dir / name):
            if cand.exists():
                return gt, cand
    return gt, run_dir / pred_names[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", type=Path,
                    help="run dir holding _bench_gt_coco.json + _bench_predictions_resolved.json")
    ap.add_argument("--gt", type=Path, help="explicit COCO GT json")
    ap.add_argument("--preds", type=Path, help="explicit COCO predictions json")
    ap.add_argument("--min_score", type=float, default=0.5,
                    help="score threshold for a prediction to count (default 0.5)")
    ap.add_argument("--sweep_score", action="store_true",
                    help="report at 0.05..0.95 and name the MAE-optimal threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="match IoU (default 0.5)")
    ap.add_argument("--contain", type=float, default=0.5,
                    help="containment fraction for merge/split (default 0.5)")
    ap.add_argument("--out", type=Path, help="write per-image CSV here")
    ap.add_argument("--out_agg", type=Path, help="append aggregate row to this CSV")
    ap.add_argument("--tag", default="", help="label for the aggregate row")
    ap.add_argument("--by_tier", action="store_true",
                    help="stratify GT outcomes by ap_difficulty Easy/Medium/Hard tier")
    ap.add_argument("--tau1", type=float, default=None,
                    help="override Eq. 5 Easy/Medium boundary (default 0.30)")
    ap.add_argument("--tau2", type=float, default=None,
                    help="override Eq. 5 Medium/Hard boundary (default 0.58)")
    args = ap.parse_args()

    if args.run_dir:
        gt_path, pred_path = _resolve_run_dir(args.run_dir)
    else:
        gt_path, pred_path = args.gt, args.preds
    if not gt_path or not pred_path or not Path(gt_path).exists() or not Path(pred_path).exists():
        sys.exit(f"missing GT or predictions: {gt_path} / {pred_path}")

    tag = args.tag or (args.run_dir.name if args.run_dir else "run")

    if args.sweep_score:
        best = None
        print(f"{'score':>6} {'MAE':>8} {'signed':>8} {'missed':>8} {'merged':>8} "
              f"{'split':>8} {'spurious':>9}")
        for s in [round(x * 0.05, 2) for x in range(1, 20)]:
            rows = _run(gt_path, pred_path, s, args.iou, args.contain)
            a = _aggregate(rows)
            print(f"{s:>6.2f} {a['MAE_count']:>8.3f} {a['mean_signed_error']:>8.3f} "
                  f"{a['missed_per_gt']:>8.4f} {a['merged_per_gt']:>8.4f} "
                  f"{a['split_per_gt']:>8.4f} {a['spurious_per_gt']:>9.4f}")
            if best is None or a["MAE_count"] < best[1]["MAE_count"]:
                best = (s, a, rows)
        score, agg, rows = best
        print(f"\nMAE-optimal score threshold: {score}")
        # The sweep runs untiered to stay cheap; tier once, at the winner.
        tiers = _run(gt_path, pred_path, score, args.iou, args.contain,
                     by_tier=True, tau1=args.tau1,
                     tau2=args.tau2)[1] if args.by_tier else None
    else:
        score = args.min_score
        out = _run(gt_path, pred_path, score, args.iou, args.contain,
                   by_tier=args.by_tier, tau1=args.tau1, tau2=args.tau2)
        rows, tiers = out if args.by_tier else (out, None)
        agg = _aggregate(rows)

    if tiers is not None:
        agg.update(_tier_columns(tiers))
        agg["tau1"] = args.tau1 if args.tau1 is not None else TAU_1
        agg["tau2"] = args.tau2 if args.tau2 is not None else TAU_2

    print(f"\n{tag}  (score>={score}, IoU>={args.iou}, containment>={args.contain})")
    for k, v in agg.items():
        print(f"  {k:<24} {v}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Tier columns only appear on images that have GT in that tier, so the
        # header is the union over rows, not the first row's keys.
        fields = list(rows[0].keys())
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval=0)
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-image CSV -> {args.out}")

    if args.out_agg:
        args.out_agg.parent.mkdir(parents=True, exist_ok=True)
        row = {"tag": tag, "score_thr": score, "iou_thr": args.iou,
               "contain_thr": args.contain, **agg}
        exists = args.out_agg.exists()
        with open(args.out_agg, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(f"aggregate row -> {args.out_agg}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------- gold-set adapter

def load_gold(labels_dir, images_dir, pred_json, min_score):
    """Load the P5 gold corpora, whose formats differ from the bench runs.

    Ground truth is YOLO polygons normalised to the image. Predictions are
    `{filename: [{rle, score, box}, ...]}` with RLE already at full gold
    resolution. Returns [(pred_rles, gt_rles, gt_boxes, hw), ...] per image.
    """
    from PIL import Image
    preds = json.loads(Path(pred_json).read_text())
    # The SAM3 files wrap the per-image dict under "preds" alongside the
    # sieve configuration; the supervised files are already flat.
    if isinstance(preds, dict) and "preds" in preds:
        preds = preds["preds"]
    out = []
    for lbl in sorted(Path(labels_dir).glob("*.txt")):
        stem = lbl.stem
        img = next((p for p in Path(images_dir).glob(stem + ".*")), None)
        if img is None:
            continue
        w, h = Image.open(img).size
        gt_rles, gt_boxes = [], []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            xy = [float(v) for v in parts[1:]]
            poly = [xy[i] * (w if i % 2 == 0 else h) for i in range(len(xy))]
            rle = mask_utils.merge(mask_utils.frPyObjects([poly], h, w))
            gt_rles.append(rle)
            xs, ys = poly[0::2], poly[1::2]
            gt_boxes.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
        pred_rles = []
        for d in preds.get(img.name, []):
            if float(d.get("score", 1.0)) < min_score:
                continue
            r = dict(d["rle"])
            if isinstance(r["counts"], str):
                r["counts"] = r["counts"].encode("utf-8")
            pred_rles.append(r)
        out.append((pred_rles, gt_rles, gt_boxes, (h, w)))
    return out
