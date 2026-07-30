#!/usr/bin/env python3
"""Count-accuracy analysis: predicted per-frame count vs ground-truth count.

WHY
---
The 4th paper's first realistic, ground-truthable result on the penguin demo
is COUNT ACCURACY — does the detector+tracker report the right number of
penguins per frame? This needs only per-frame GT COUNT (a human can count "2
visible penguins" without naming them), NOT identity GT (which is circular
and slow). So this is the experiment that runs as soon as `gt_scaffold.py`
output is hand-filled in the `gt_count` column.

It also reports the tracker's over/under-count profile by difficulty — the
exact signal the 4th paper's FAD gate would route on (frames where the count
is wrong are hard frames).

INPUT
-----
`gt_annotations.csv` produced by `gt_scaffold.py`, with `gt_count` filled by
a human for each frame (one value per frame; instance_idx rows within a frame
share the same gt_count). If `gt_count` is blank for a frame, that frame is
skipped (so you can fill count first, run this, fill identities later).

OUTPUT
------
- per-frame: predicted_count (non-blank seed rows), gt_count, abs error
- aggregate: MAE, RMSE, overcount-rate, undercount-rate, accuracy@exact,
  accuracy@±1, per-difficulty breakdown (using gt_visible as a proxy when
  available: frames with any gt_visible<=0 are 'hard').

This is NOT a smoke test: it reads one CSV, writes one CSV/stdout. No model,
no orchestrator, no ~/bench_any/runs/, no Fish/YOLO/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def _load(csv_path: Path):
    """Return {frame_idx: {'gt_count': int|None, 'pred_count': int,
    'visible_flags': list, 'seed_ids': list}}."""
    frames = defaultdict(lambda: {"gt_count": None, "pred_count": 0,
                                   "visible_flags": [], "seed_ids": []})
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                fi = int(r["frame_idx"])
            except (ValueError, KeyError):
                continue
            fr = frames[fi]
            # predicted count = number of non-blank seed rows in this frame
            sid = r.get("seed_track_id", "")
            if sid != "" and sid != "-1":
                fr["pred_count"] += 1
                fr["seed_ids"].append(int(sid))
            # gt count: one value per frame, take the max to be robust to
            # the annotator filling it once per frame (blank rows carry "")
            gc = r.get("gt_count", "")
            if gc != "":
                try:
                    val = int(gc)
                    if fr["gt_count"] is None or val > fr["gt_count"]:
                        fr["gt_count"] = val
                except ValueError:
                    pass
            gv = r.get("gt_visible", "")
            if gv != "":
                try:
                    fr["visible_flags"].append(int(gv))
                except ValueError:
                    pass
    return frames


def _summarize(frames):
    have_gt = [(fi, fr) for fi, fr in frames.items() if fr["gt_count"] is not None]
    if not have_gt:
        return None
    errs = []
    over = under = exact = within1 = 0
    hard_errs = []
    easy_errs = []
    for fi, fr in have_gt:
        e = fr["pred_count"] - fr["gt_count"]
        errs.append(abs(e))
        if e > 0: over += 1
        elif e < 0: under += 1
        else: exact += 1
        if abs(e) <= 1: within1 += 1
        # hard = any occluded/uncertain flag OR a count error (the paper's
        # definition: a frame is hard if the detector got the count wrong).
        is_hard = (any(v <= 0 for v in fr["visible_flags"])
                   or e != 0)
        (hard_errs if is_hard else easy_errs).append(abs(e))
    n = len(have_gt)
    mae = sum(errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    return {
        "n_frames_with_gt": n,
        "n_frames_total": len(frames),
        "MAE_count": round(mae, 3),
        "RMSE_count": round(rmse, 3),
        "accuracy_exact": round(exact / n, 3),
        "accuracy_within1": round(within1 / n, 3),
        "overcount_rate": round(over / n, 3),
        "undercount_rate": round(under / n, 3),
        "hard_frame_share": round(len(hard_errs) / n, 3),
        "MAE_easy": round(sum(easy_errs) / max(1, len(easy_errs)), 3) if easy_errs else 0.0,
        "MAE_hard": round(sum(hard_errs) / max(1, len(hard_errs)), 3) if hard_errs else 0.0,
        "n_easy": len(easy_errs),
        "n_hard": len(hard_errs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="gt_annotations.csv (from gt_scaffold.py, gt_count filled)")
    ap.add_argument("--out", default=None,
                    help="optional output JSON for the aggregate metrics")
    ap.add_argument("--per_frame_out", default=None,
                    help="optional per-frame CSV (frame_idx,pred,gt,abs_err,is_hard)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"[count_acc] not found: {csv_path}")
    frames = _load(csv_path)
    summary = _summarize(frames)
    if summary is None:
        print("[count_acc] no frames with gt_count filled — fill the gt_count "
              "column in the CSV first (one integer per frame), then re-run.")
        sys.exit(0)

    print(f"[count_acc] {csv_path}")
    print(f"  frames with GT: {summary['n_frames_with_gt']}/{summary['n_frames_total']}")
    print(f"  MAE  (mean abs count error): {summary['MAE_count']}")
    print(f"  RMSE:                         {summary['RMSE_count']}")
    print(f"  exact-accuracy:              {summary['accuracy_exact']}")
    print(f"  within-±1 accuracy:          {summary['accuracy_within1']}")
    print(f"  overcount rate:              {summary['overcount_rate']}")
    print(f"  undercount rate:             {summary['undercount_rate']}")
    print(f"  hard-frame share:            {summary['hard_frame_share']}  "
          f"(easy={summary['n_easy']}, hard={summary['n_hard']})")
    print(f"  MAE on easy frames:          {summary['MAE_easy']}")
    print(f"  MAE on hard frames:          {summary['MAE_hard']}  "
          f"<-- the cost-asymmetry the 4th paper's FAD gate would target")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[count_acc] wrote aggregate -> {args.out}")
    if args.per_frame_out:
        with open(args.per_frame_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "pred_count", "gt_count", "abs_err",
                        "is_hard"])
            for fi in sorted(frames):
                fr = frames[fi]
                if fr["gt_count"] is None:
                    continue
                e = fr["pred_count"] - fr["gt_count"]
                is_hard = (any(v <= 0 for v in fr["visible_flags"]) or e != 0)
                w.writerow([fi, fr["pred_count"], fr["gt_count"], abs(e),
                            int(is_hard)])
        print(f"[count_acc] wrote per-frame -> {args.per_frame_out}")


if __name__ == "__main__":
    main()
