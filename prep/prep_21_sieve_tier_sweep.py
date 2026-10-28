#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Score all 54 sieve candidates per difficulty tier, 2026-08-01.

The prep_11 sweep ranked its candidates on overall AP@0.5 with a recall floor,
so the hard tier was never something the sweep could see. This asks the question
the sweep could not: did a candidate exist that held the aggregate and did
better on hard.

Nothing re-runs SAM~3. The raw masks are cached in `{corpus}_sam3_raw.json` and
`sieve_apply` works on run-length encodings without decoding them, so the whole
sweep is processor work. Scoring is prep_10's, unchanged and imported rather
than copied: out-of-tier ground truth marked `iscrowd=1`, single intersection
threshold 0.5, `maxDets` 1/10/100. The three ground-truth structures are built
once and reused across all 54 candidates.

    python prep_21_sieve_tier_sweep.py --corpus pw_v2_200 \
        --tiers ~/bench_any/runs/p5_r1_gold/pw_v2_200_tiers_tercile.json \
        --raw   ~/bench_any/runs/p5_r1_gold/pw200_zeroshot/pw_v2_200_sam3_raw.json \
        --out   ~/bench_any/runs/p5_r1_gold/pw200_zeroshot_tercile/pw_v2_200_sieve_tier_sweep.json

Pointing `--tiers` at the original fixed-threshold sidecar is the back-compat
gate: the chosen config must reproduce its published easy, medium and hard.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_10_filebased import build_gt_coco, build_pred_coco, eval_tier, TIERS
from prep_11_filebased import sieve_apply, MIN_AREAS, NMS_IOUS, MAX_DIM_FRACS, SCORES


def preds_for_config(raw, tiers_manifest, cfg):
    """Apply one sieve config to the cached raw SAM~3 output."""
    out = {}
    for fname, raw_preds in raw.items():
        dets = tiers_manifest.get(fname, [])
        if not dets:
            continue
        W = dets[0]["width"]; H = dets[0]["height"]
        scores = np.array([p["score"] for p in raw_preds])
        boxes = np.array([p["box"] for p in raw_preds]) if len(raw_preds) else np.zeros((0, 4))
        rles = [p["rle"] for p in raw_preds]
        keep = sieve_apply(rles, scores, boxes, cfg["min_area"], cfg["nms_iou"],
                           cfg["max_dim_frac"], cfg["score_floor"], W, H)
        out[fname] = [{"rle": rles[k], "score": float(scores[k])} for k in keep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tiers_data = json.load(open(os.path.expanduser(args.tiers)))
    tiers_manifest = tiers_data["manifest"]
    raw = json.load(open(os.path.expanduser(args.raw)))
    print(f"tiers {args.tiers}\n  mode={tiers_data.get('tier_mode', 'fixed 0.30/0.58 (prep_09)')} "
          f"counts={tiers_data.get('tier_counts')}", flush=True)

    gt_by_tier = {}
    n_gt_per_tier = {}
    for t in TIERS:
        gt_by_tier[t] = build_gt_coco(tiers_manifest, t)
        n_gt_per_tier[t] = sum(1 for a in gt_by_tier[t]["annotations"] if a["iscrowd"] == 0)
    print(f"active gold per tier: {n_gt_per_tier}", flush=True)

    rows = []
    for min_area in MIN_AREAS:
        for nms_iou in NMS_IOUS:
            for max_dim_frac in MAX_DIM_FRACS:
                for score_floor in SCORES:
                    cfg = {"min_area": min_area, "nms_iou": nms_iou,
                           "max_dim_frac": max_dim_frac, "score_floor": score_floor}
                    per_img = preds_for_config(raw, tiers_manifest, cfg)
                    pred_list = build_pred_coco(per_img, tiers_manifest)
                    scores = {}
                    for t in TIERS:
                        scores[t] = eval_tier(gt_by_tier[t], pred_list) if n_gt_per_tier[t] else None
                    got = [scores[t] for t in TIERS if scores[t] is not None]
                    agg = float(np.mean(got)) if len(got) == len(TIERS) else None
                    w = sum(n_gt_per_tier[t] for t in TIERS if scores[t] is not None)
                    agg_iw = (sum(scores[t] * n_gt_per_tier[t]
                                  for t in TIERS if scores[t] is not None) / w) if w else None
                    row = {**cfg, "n_preds": sum(len(v) for v in per_img.values()),
                           "easy": scores["easy"], "medium": scores["medium"],
                           "hard": scores["hard"], "agg": agg, "agg_iw": agg_iw}
                    rows.append(row)
                    print(f"  {cfg} -> easy {scores['easy']} medium {scores['medium']} "
                          f"hard {scores['hard']} agg {agg}", flush=True)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"corpus": args.corpus, "tiers_file": os.path.expanduser(args.tiers),
               "tier_mode": tiers_data.get("tier_mode", "fixed 0.30/0.58 (prep_09)"),
               "n_gt_per_tier": n_gt_per_tier,
               "metric": "pycocotools COCOeval mask mAP@0.5, out-of-tier gold iscrowd=1, "
                         "maxDets 1/10/100; identical to prep_10_filebased",
               "sweep": rows}, open(out, "w"), indent=2)
    print(f"\n{len(rows)} candidates -> {out}", flush=True)


if __name__ == "__main__":
    main()
