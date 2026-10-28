#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""SAM3 union prompt + deployed border rule, per-tier mask mAP (CPU, no GPU).

D-L16970 agreement: report the union prompt through the FULL deployed sieve
(border rule included) as the SAM3 zero-shot headline. The first version of
this script used sanity_sieve's A_MIN=324 / NMS=0.55 (CANON) while the
single-prompt headline (81.6 / 58.3) uses the ASRUN config (min_area=64,
nms_iou=0.8, max_dim_frac=1.0, score_floor=0.25) — an apples-to-oranges
comparison that produced 51.2 / 45.3, a sieve artifact, not a prompt result.

This version imports the authors' _sieve + _score + _fn_hw from relsize_stratified_
recompute.py and feeds the union preds through them at the SAME asrun config
+ fiftyone border the single-prompt headline uses. No second sieve
implementation: the union is just a different input to the same scorer.

Path: union of 'penguin' + 'a penguin' raw preds (score >= 0.40, the union
prompt threshold) -> same _sieve(asrun, fiftyone) -> same _score (prep_10
build_gt_coco + eval_tier, iscrowd-ignore per tier, mAP@0.5).

Self-reports to union_border_stratified.txt + touches _union_border_stratified_done.
    ~/cvML/bin/python penguins/union_border_stratified_eval.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import pycocotools.mask as mu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "archived" / "2026-08-15_penguins_cleanup"))
sys.path.insert(0, str("/home/user/bench_any/datasets/prep"))

from relsize_stratified_recompute import _sieve, _score, _fn_hw  # noqa: E402

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
SUB = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks"}
THR = 0.40
OUT_TXT = RUNS / "union_border_stratified.txt"
DONE = RUNS / "_union_border_stratified_done"

# The exact config the single-prompt+border headline (81.6/58.3) uses.
ASRUN = {"min_area": 64, "nms_iou": 0.8, "max_dim_frac": 1.0, "score_floor": 0.25}


def build_union_raw(corpus, thr):
    """Union of 'penguin' + 'a penguin' in the _sam3_raw format: {fn: [{rle,score,box}]}.

    Reads the per-prompt gold COCO files, merges by image file_name, filters at
    the union prompt threshold. No sieve here — _sieve applies it."""
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    fn_set = {im["file_name"] for im in gt["images"]}
    by_fn = defaultdict(list)
    for stem in ("penguin", "a_penguin"):
        src = json.loads((RUNS / "sam3_prompts_gold" / f"{corpus}_{stem}.json").read_text())
        iid_fn = {i["id"]: i["file_name"] for i in src["images"]}
        for a in src["annotations"]:
            fn = iid_fn.get(a["image_id"])
            if fn is None or fn not in fn_set:
                continue
            if float(a.get("score", 1.0)) < thr:
                continue
            seg = a["segmentation"]
            rle = {"size": seg["size"],
                   "counts": seg["counts"].encode() if isinstance(seg["counts"], str)
                   else seg["counts"]}
            bb = mu.toBbox(rle)
            box = [float(bb[0]), float(bb[1]), float(bb[0] + bb[2]), float(bb[1] + bb[3])]
            by_fn[fn].append({"rle": {"size": list(seg["size"]),
                                      "counts": seg["counts"]},
                              "score": float(a.get("score", 1.0)), "box": box})
    return dict(by_fn)


def main():
    lines = ["# SAM3 union prompt + deployed border rule, per-tier mask mAP@0.50, "
             "thr=0.40, asrun sieve, 2026-08-25\n\n",
             "Union of 'penguin' + 'a penguin' through the SAME _sieve(asrun, fiftyone) "
             "as the single-prompt headline (81.6/58.3). asrun: min_area=64, nms_iou=0.8, "
             "max_dim_frac=1.0, score_floor=0.25; fiftyone border k=0.6 x median-all-box-area, "
             "border 0.5% of image dim. Per-tier via prep_10 eval_tier, iscrowd-ignore, mAP@0.5.\n\n",
             "| corpus | Easy | Medium | Hard | tier-mean agg mAP@50 (excl-Easy on SANParks) | n_preds |\n",
             "|---|---|---|---|---|---|\n"]
    summary = {}
    for corpus, label in CORPORA.items():
        sub = SUB[corpus]
        man = json.loads((R1 / sub / f"{sub}_tiers.json").read_text())["manifest"]
        fn_hw = _fn_hw(corpus)
        raw = build_union_raw(corpus, THR)
        n_pre = sum(len(v) for v in raw.values())
        sieved = _sieve(raw, fn_hw, ASRUN, "fiftyone", 0.0)
        n_post = sum(len(v) for v in sieved.values())
        scored = _score(sieved, man)
        e = scored["easy"]
        m = scored["medium"]
        h = scored["hard"]
        if label == "SANParks":
            agg = (m + h) / 2
            agg_note = f"{agg:.1f} (excl-Easy)"
        else:
            agg = (e + m + h) / 3
            agg_note = f"{agg:.1f}"
        lines.append(f"| {label} | {e:.1f} | {m:.1f} | {h:.1f} | {agg_note} | {n_pre}->{n_post} |\n")
        print(f"[union+border asrun] {label}: E={e:.1f} M={m:.1f} H={h:.1f} agg={agg_note} "
              f"preds={n_pre}->{n_post}")
        summary[label] = {"Easy": e, "Medium": m, "Hard": h, "agg": agg,
                          "n_pre": n_pre, "n_post": n_post}
    OUT_TXT.write_text("".join(lines))
    (RUNS / "union_border_stratified.json").write_text(json.dumps(summary, indent=2))
    DONE.touch()
    print(f"\nWrote {OUT_TXT}, {RUNS / 'union_border_stratified.json'}, touched {DONE}")


if __name__ == "__main__":
    main()
