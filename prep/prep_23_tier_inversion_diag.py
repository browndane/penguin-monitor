#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Diagnose the per-corpus tercile tier inversion (SANParks, iNaturalist).

Three checks, all CPU, all over cached *_pred.json and the tercile sidecars:

  A  matched-image subset. Rescore the three tiers using only the images that
     contain at least one instance of every tier, so the iscrowd=1 out-of-tier
     leniency is identical across tiers. If the inversion vanishes here it is
     an artefact of image support, not of the difficulty score.
  B  detection budget. Report the per-image prediction count against the
     maxDets=100 ceiling, since truncation interacts with which GT are ignored.
  C  instance-level rank correlation between the stored difficulty score and
     whether the instance was matched at IoU 0.5, over every instance rather
     than three tier aggregates. This is the only check with real power on a
     51-image corpus.

Writes a json per corpus and prints a bounded block for FINDINGS.md.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pycocotools.mask as mu
from scipy.stats import spearmanr, pointbiserialr

sys.path.insert(0, str(Path(__file__).parent))
from prep_10_filebased import build_gt_coco, build_pred_coco, eval_tier

TIERS = ["easy", "medium", "hard"]


def matched_flags(dets, plist, iou_thr=0.5):
    """Greedy one-to-one match of preds (score-sorted) to GT masks at iou_thr.
    Returns a bool per GT instance in dets order."""
    if not dets:
        return []
    gt_rles = [d["rle"] for d in dets]
    if not plist:
        return [False] * len(dets)
    order = np.argsort([-p["score"] for p in plist])
    pr_rles = [plist[i]["rle"] for i in order]
    ious = mu.iou(pr_rles, gt_rles, [0] * len(gt_rles))
    ious = np.asarray(ious).reshape(len(pr_rles), len(gt_rles))
    taken = [False] * len(dets)
    for pi in range(len(pr_rles)):
        row = ious[pi].copy()
        for gi in range(len(dets)):
            if taken[gi]:
                row[gi] = -1
        gi = int(np.argmax(row))
        if row[gi] >= iou_thr:
            taken[gi] = True
    return taken


def run_corpus(corpus, tiers_path, preds, out_dir):
    tm = json.load(open(tiers_path))
    manifest = tm["manifest"]
    result = {"corpus": corpus, "tiers": str(tiers_path), "models": {}}

    full_ids = [f for f, dets in manifest.items()
                if {d["tier"] for d in dets} == set(TIERS)]
    result["n_images"] = len(manifest)
    result["n_images_all_three_tiers"] = len(full_ids)
    sub_manifest = {f: manifest[f] for f in full_ids}
    result["n_instances_subset"] = sum(len(v) for v in sub_manifest.values())

    for tag, pred_path in preds.items():
        pdata = json.load(open(pred_path))
        preds_per_img = pdata["preds"] if "preds" in pdata else pdata
        m = {}

        # B: detection budget
        counts = [len(v) for v in preds_per_img.values()]
        m["max_dets_per_image"] = int(max(counts)) if counts else 0
        m["n_images_at_100"] = int(sum(1 for c in counts if c >= 100))

        # A: full corpus and matched-image subset
        for label, mani in (("full", manifest), ("subset", sub_manifest)):
            pl = build_pred_coco(preds_per_img, mani)
            scores = {}
            for t in TIERS:
                gt = build_gt_coco(mani, t)
                n_active = sum(1 for a in gt["annotations"] if a["iscrowd"] == 0)
                ap = eval_tier(gt, pl)
                scores[t] = None if ap is None else round(ap, 2)
                scores[f"n_{t}"] = n_active
            scores["monotone"] = all(
                scores[a] is not None and scores[b] is not None and scores[a] >= scores[b]
                for a, b in (("easy", "medium"), ("medium", "hard")))
            m[label] = scores

        # C: instance-level correlation
        d_scores, matched = [], []
        for fname, dets in manifest.items():
            flags = matched_flags(dets, preds_per_img.get(fname, []))
            for det, hit in zip(dets, flags):
                d_scores.append(det["score"])
                matched.append(1.0 if hit else 0.0)
        d_scores = np.asarray(d_scores)
        matched = np.asarray(matched)
        rho, p = spearmanr(d_scores, matched)
        rpb, ppb = pointbiserialr(matched, d_scores)
        m["instance_level"] = {
            "n": int(len(d_scores)),
            "recall": round(float(matched.mean()) * 100, 2),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p),
            "point_biserial_r": round(float(rpb), 4),
            "point_biserial_p": float(ppb),
            "mean_score_matched": round(float(d_scores[matched == 1].mean()), 4) if matched.any() else None,
            "mean_score_missed": round(float(d_scores[matched == 0].mean()), 4) if (matched == 0).any() else None,
        }
        result["models"][tag] = m
        print(f"  {tag}: full {m['full']['easy']}/{m['full']['medium']}/{m['full']['hard']} "
              f"subset {m['subset']['easy']}/{m['subset']['medium']}/{m['subset']['hard']} "
              f"rho {m['instance_level']['spearman_rho']:+.3f}", flush=True)

    out = Path(out_dir) / f"{corpus}_tier_inversion_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    print(f"  wrote {out}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--pred", action="append", required=True,
                    help="tag=path, repeatable")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    preds = dict(p.split("=", 1) for p in a.pred)
    print(f"== {a.corpus} ==", flush=True)
    run_corpus(a.corpus, a.tiers, preds, a.out_dir)


if __name__ == "__main__":
    main()
