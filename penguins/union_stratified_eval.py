#!/usr/bin/env python3
"""Per-tier (Easy/Medium/Hard) mask mAP for the UNION prompt config, CPU-only.

CORRECTED version. The first attempt was wrong on two counts:
  1. recomputed tiers via compute_difficulty_from_coco_gt (235/235/299) instead of
     the paper's tiers.json assignment (204/535/30). Fixed: load paper tiers.json.
  2. used bench_eval.stratified_coco_eval, which SUBSETS GT (removes non-target
     tiers) so a prediction matching a Hard GT counts as a false positive in the
     Easy eval, depressing every tier. Fixed: mark non-target tiers iscrowd=1
     (absorbed, not FP), per the pycocotools-iscrowd-ignore-per-tier memory.

Result is directly comparable to the paper's existing SAM3+sieve rows
(boulders_reeval_sam3.json / sanparks_reeval_sam3.json), which used the same
iscrowd-ignore method and the same tiers.json split.

Self-reports to §18 of ~/bench_any/runs/obj2_counting/FINDINGS.md and touches
_union_stratified_done. §17 (the broken version) is left in place for the record.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from prompt_union_maskmap import build_union, CORPORA, RUNS, THR

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
OUT_TXT = RUNS / "union_stratified_v2.txt"
DONE = RUNS / "_union_stratified_v2_done"
FINDINGS = RUNS / "FINDINGS.md"

TIER_FILE = {"boulders-penguin": R1 / "boulders" / "boulders_tiers.json",
             "sanparks-algoa": R1 / "sanparks" / "sanparks_tiers.json"}


def tier_map(corpus, gt_dict):
    """file_name -> [tier per ann by per-image idx], joined to gt ann ids."""
    tj = json.loads(TIER_FILE[corpus].read_text())
    man = tj["manifest"]
    fn_to_iid = {im["file_name"]: im["id"] for im in gt_dict["images"]}
    # gt anns grouped per image, sorted by int(id)
    from collections import defaultdict
    by_img = defaultdict(list)
    for a in gt_dict["annotations"]:
        by_img[a["image_id"]].append(a)
    for iid in by_img:
        by_img[iid].sort(key=lambda a: int(a["id"]))
    ann_id_to_tier = {}
    for fn, entry in man.items():
        iid = fn_to_iid[fn]
        gts = by_img[iid]
        for k, e in enumerate(entry):
            if k < len(gts):
                t = e["tier"]
                t = t.capitalize() if t.lower() in ("easy", "medium", "hard") else t
                ann_id_to_tier[gts[k]["id"]] = t
    return ann_id_to_tier


def per_tier_eval(gt_path, preds, ann_id_to_tier):
    """iscrowd-ignore per-tier COCO segm eval. Non-target tier GT -> iscrowd=1."""
    gt = json.loads(Path(gt_path).read_text())
    tier_counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    for a in gt["annotations"]:
        t = ann_id_to_tier.get(a["id"], "Medium")
        tier_counts[t] += 1
    # materialise bbox+area for loadRes
    for p in preds:
        seg = p["segmentation"]
        rle = {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}
        p["bbox"] = [float(x) for x in mu.toBbox(rle)]
        p["area"] = float(mu.area(rle))
    tmp = RUNS / "_tmp_pred_strat_v2.json"
    tmp.write_text(json.dumps(preds))

    out = {}
    for tier_name in ("Easy", "Medium", "Hard"):
        gt_t = json.loads(Path(gt_path).read_text())
        for a in gt_t["annotations"]:
            t = ann_id_to_tier.get(a["id"], "Medium")
            a["iscrowd"] = 1 if t != tier_name else 0
        gt_obj = COCO()
        gt_obj.dataset = gt_t
        gt_obj.createIndex()
        cocoDt = gt_obj.loadRes(str(tmp))
        ev = COCOeval(gt_obj, cocoDt, "segm")
        ev.params.maxDets = [1, 10, 100]
        ev.params.iouThrs = __import__("numpy").array([0.5])
        ev.evaluate(); ev.accumulate()
        pr = ev.eval["precision"]
        # precision[0,:,0,0,-1] : T=iouThrs[0]=0.5, R, K=0, A=0, M=maxDets index -1=100
        m50 = float(pr[0, :, 0, 0, -1].mean()) * 100
        out[tier_name] = {"mAP50": m50, "n": tier_counts[tier_name]}
    return out


def main():
    lines = ["# Union prompt stratified per-tier mask mAP (iscrowd-ignore, paper tiers) 2026-07-29\n\n",
             "SAM3 union of 'penguin' + 'a penguin', sieve + class-agnostic NMS, thr 0.40.\n",
             "Per-tier via iscrowd=1 ignore (NOT GT-subset), paper tiers.json split. Directly comparable to P5 tab:percorpus.\n",
             "§17 was the broken version (recomputed tiers + GT-subset) and is left for the record.\n\n",
             "| corpus | Easy | Medium | Hard | tier-mean agg mAP@50 | tier counts (E/M/H) |\n",
             "|---|---|---|---|---|---|\n"]
    for corpus, label in CORPORA.items():
        gt_path = RUNS / f"sam2refine_{corpus}" / "gt_coco.json"
        preds = build_union(corpus, THR)
        gt_dict = json.loads(Path(gt_path).read_text())
        ann_id_to_tier = tier_map(corpus, gt_dict)
        res = per_tier_eval(gt_path, preds, ann_id_to_tier)
        e, m, h = res["Easy"]["mAP50"], res["Medium"]["mAP50"], res["Hard"]["mAP50"]
        agg = (e + m + h) / 3
        tc = f"{res['Easy']['n']}/{res['Medium']['n']}/{res['Hard']['n']}"
        lines.append(f"| {label} | {e:.1f} | {m:.1f} | {h:.1f} | {agg:.1f} | {tc} |\n")
        print(f"[union-strat-v2] {label}: E={e:.1f} M={m:.1f} H={h:.1f} agg50={agg:.1f} counts={tc}")
    table = "".join(lines)
    OUT_TXT.write_text(table)
    with open(FINDINGS, "a") as fh:
        fh.write(f"\n## 18. Union stratified per-tier (iscrowd-ignore, paper tiers) (2026-07-29)\n\n{table}\n")
    DONE.touch()
    print(f"Wrote {OUT_TXT}, touched {DONE}")


if __name__ == "__main__":
    main()
