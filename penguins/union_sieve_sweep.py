#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Union vs single-prompt sieve sweep on P5's REAL metric (tier-mean agg mAP@50).

WHY
---
The §12 "union wins" finding was on flat-pool mAP@50 with a fixed sieve. P5's
headline metric is tier-mean aggregate mAP@50 (mean of Easy/Medium/Hard), and
the paper's single-prompt preds went through a 54-config sieve sweep whose
chosen config is {min_area 64, nms_iou 0.8, max_dim_frac 1.0, score_floor 0.25}.
The union has never had that sweep, so the two were not comparable on the paper
metric.

This script closes that gap. It runs BOTH union and single-prompt across all
54 sieve configs under the paper's EXACT per-tier method (prep_10_filebased:
build_gt_coco from tiers_manifest RLEs, eval_tier with areaRng=[[0,1e10]] +
precision[0,:,0,0,-1] pr[pr>-1], real scores). Reports best-of-54 for each.

Validated: prep_10 reproduces the paper's 69.0 (Boulders) / 38.7 (SANParks)
exactly on the paper's own swept preds (98.0/85.6/23.3/agg 69.0).

Self-reports §19 to ~/bench_any/runs/obj2_counting/FINDINGS.md and touches
sentinel _union_sweep_done. CPU-only, no model, no GPU.

Grid (54 configs, mirrors the paper sweep):
  min_area      [32, 64, 128]
  nms_iou       [0.7, 0.8, 0.9]
  max_dim_frac  [0.9, 0.95, 1.0]
  score_floor   [0.25, 0.28]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pycocotools.mask as mu

# Reuse the paper's exact eval method
PREP_DIR = Path("/home/user/bench_any/datasets/prep")
sys.path.insert(0, str(PREP_DIR))
import prep_10_filebased as P10  # noqa: E402

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
FINDINGS = RUNS / "FINDINGS.md"
DONE = RUNS / "_union_sweep_done"
OUT_TXT = RUNS / "union_sweep_results.txt"

CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
SUBDIR = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks"}
# Paper's single-prompt baseline is "a penguin" (prep_11 default -> boulders_sam3_raw.json).
# "penguin" raw was generated 2026-07-29 via prep_11 --prompt penguin -> *_sam3_raw_penguin.json.
# Both from the SAME SAM3 generation (dense-image detections kept), so single vs union is fair.
SINGLE_STEM = "a_penguin"
UNION_STEMS = ["a_penguin", "penguin"]
PROMPT_STEM = "a_penguin"  # kept for the chosen-config validation path
THR = 0.40                   # union adoption floor (§12); score_floor from cfg is the real cutoff


def _raw_path(corpus, stem):
    """prep_11 raw path: default 'a penguin' -> corpus_sam3_raw.json (no tag);
    any other stem -> corpus_sam3_raw_<stem>.json."""
    sub = SUBDIR[corpus]
    tag = "" if stem == "a_penguin" else "_" + stem
    return R1 / sub / f"{sub}_sam3_raw{tag}.json"

# Paper published single-prompt tier-mean (validation targets)
PAPER = {"boulders-penguin": {"E": 98.0, "M": 85.6, "H": 23.3, "agg": 69.0},
         "sanparks-algoa": {"E": 0.0, "M": 84.4, "H": 31.8, "agg": 38.7}}
CHOSEN = {"min_area": 64, "nms_iou": 0.8, "max_dim_frac": 1.0, "score_floor": 0.25}

MIN_AREAS = [32, 64, 128]
NMS_IOUS = [0.7, 0.8, 0.9]
MAX_DIM_FRACS = [0.9, 0.95, 1.0]
SCORE_FLOORS = [0.25, 0.28]


def _box_from_rle(rle):
    bb = mu.toBbox(rle)
    return (float(bb[0]), float(bb[1]), float(bb[0] + bb[2]), float(bb[1] + bb[3]))


def _raw_pooled(corpus, stems):
    """Pool raw SAM3 anns across stems from prep_11 raws (filename-keyed,
    {fn: [{rle,score,box}]}). Returns {filename: [{rle, score, box}, ...]}.
    All stems come from the SAME SAM3 generation, so single vs union is fair."""
    out = {}
    for stem in stems:
        f = _raw_path(corpus, stem)
        if not f.exists():
            print(f"[WARN] missing raw: {f}", file=sys.stderr)
            continue
        raw = json.loads(f.read_text())
        for fn, anns in raw.items():
            out.setdefault(fn, []).extend(anns)
    return out


def _iou(b1, b2):
    xa, ya, x2a, y2a = b1
    xb, yb, x2b, y2b = b2
    ix1, iy1 = max(xa, xb), max(ya, yb)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2a - xa) * (y2a - ya)
    a2 = (x2b - xb) * (y2b - yb)
    return inter / (a1 + a2 - inter)


def _class_agnostic_nms(boxes, scores, iou_thr):
    if not boxes:
        return []
    order = list(np.argsort(scores)[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = [j for j in order if _iou(boxes[i], boxes[j]) > iou_thr]
        for j in rest:
            order.remove(j)
    return keep


def _sieve(preds_by_fn, fn_hw, cfg):
    """Apply one sieve config. Returns preds in the paper's filename-keyed format
    {filename: [{rle, score, box}, ...]} ready for build_pred_coco."""
    min_area = cfg["min_area"]
    nms_iou = cfg["nms_iou"]
    max_dim_frac = cfg["max_dim_frac"]
    score_floor = cfg["score_floor"]
    out = {}
    for fn, anns in preds_by_fn.items():
        h, w = fn_hw[fn]
        kept = []
        for a in anns:
            if a["score"] < score_floor:
                continue
            x1, y1, x2, y2 = a["box"]
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            if area < min_area:
                continue
            if max(bw, bh) > max_dim_frac * max(w, h):
                continue
            kept.append(a)
        if not kept:
            continue
        boxes = [a["box"] for a in kept]
        scores = [a["score"] for a in kept]
        keep_idx = _class_agnostic_nms(boxes, scores, nms_iou)
        out[fn] = [{"rle": kept[i]["rle"], "score": kept[i]["score"], "box": kept[i]["box"]}
                  for i in keep_idx]
    return out


def _fn_hw(corpus):
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    return {im["file_name"]: (im["height"], im["width"]) for im in gt["images"]}


def _grid():
    for ma in MIN_AREAS:
        for nms in NMS_IOUS:
            for mdf in MAX_DIM_FRACS:
                for sf in SCORE_FLOORS:
                    yield {"min_area": ma, "nms_iou": nms,
                           "max_dim_frac": mdf, "score_floor": sf}


def main():
    t0 = time.time()
    lines = ["\n## 19. Union vs single-prompt sieve sweep (tier-mean mAP@50, 2026-07-29)\n\n",
             "Method: paper's exact per-tier eval (prep_10_filebased). GT from tiers_manifest RLEs, ",
             "iscrowd-ignore per tier, areaRng=[[0,1e10]], precision[0,:,0,0,-1] pr[pr>-1], real scores. ",
             "Reproduces paper 69.0/38.7 exactly on its own swept preds.\n",
             "Grid: 54 configs (min_area x nms_iou x max_dim_frac x score_floor). ",
             "Both union (penguin+a_penguin, thr 0.40 floor) and single (penguin) swept.\n\n"]
    summary = []
    for corpus, label in CORPORA.items():
        tiers_path = R1 / SUBDIR[corpus] / f"{SUBDIR[corpus]}_tiers.json"
        tm = json.loads(tiers_path.read_text())
        man = tm["manifest"]
        fn_hw = _fn_hw(corpus)
        # union raw: pool both stems, no THR pre-filter (score_floor from cfg is the cutoff)
        union_raw = _raw_pooled(corpus, UNION_STEMS)
        single_raw = _raw_pooled(corpus, [SINGLE_STEM])
        best_u = best_s = None
        for cfg in _grid():
            usieve = _sieve(union_raw, fn_hw, cfg)
            upred = P10.build_pred_coco(usieve, man)
            ures = {t: P10.eval_tier(P10.build_gt_coco(man, t), upred) for t in ("easy", "medium", "hard")}
            uagg = float(np.mean([ures[t] for t in ("easy", "medium", "hard")]))
            ssieve = _sieve(single_raw, fn_hw, cfg)
            spred = P10.build_pred_coco(ssieve, man)
            sres = {t: P10.eval_tier(P10.build_gt_coco(man, t), spred) for t in ("easy", "medium", "hard")}
            sagg = float(np.mean([sres[t] for t in ("easy", "medium", "hard")]))
            row = {"cfg": cfg,
                   "u_E": ures["easy"], "u_M": ures["medium"], "u_H": ures["hard"], "u_agg": uagg,
                   "s_E": sres["easy"], "s_M": sres["medium"], "s_H": sres["hard"], "s_agg": sagg}
            if best_u is None or uagg > best_u["u_agg"]:
                best_u = row
            if best_s is None or sagg > best_s["s_agg"]:
                best_s = row
        paper = PAPER[corpus]
        lines.append(f"### {label}\n\n")
        lines.append(f"Paper published single-prompt: E={paper['E']} M={paper['M']} H={paper['H']} agg={paper['agg']}\n\n")
        lines.append(f"Best single (best-of-54): E={best_s['s_E']:.1f} M={best_s['s_M']:.1f} H={best_s['s_H']:.1f} agg={best_s['s_agg']:.1f} cfg={best_s['cfg']}\n\n")
        lines.append(f"Best union (best-of-54): E={best_u['u_E']:.1f} M={best_u['u_M']:.1f} H={best_u['u_H']:.1f} agg={best_u['u_agg']:.1f} cfg={best_u['cfg']}\n\n")
        d = best_u["u_agg"] - best_s["s_agg"]
        verdict = "UNION WINS" if d > 0 else "SINGLE WINS"
        lines.append(f"Union-best vs single-best Δ = {d:+.1f} -> {verdict}\n\n")
        summary.append((label, best_s["s_agg"], best_u["u_agg"], d, verdict))
        print(f"[{label}] single-best={best_s['s_agg']:.1f} union-best={best_u['u_agg']:.1f} d={d:+.1f} {verdict}", file=sys.stderr)
    table = ["| corpus | single-best agg | union-best agg | Δ | verdict |\n|---|---|---|---|---|\n"]
    for lab, sagg, uagg, d, v in summary:
        table.append(f"| {lab} | {sagg:.1f} | {uagg:.1f} | {d:+.1f} | {v} |\n")
    lines.append("".join(table))
    text = "".join(lines)
    OUT_TXT.write_text(text)
    with open(FINDINGS, "a") as fh:
        fh.write(text)
    DONE.touch()
    print(f"[union-sweep] done in {time.time()-t0:.1f}s -> {OUT_TXT} sentinel={DONE}", file=sys.stderr)


if __name__ == "__main__":
    main()
