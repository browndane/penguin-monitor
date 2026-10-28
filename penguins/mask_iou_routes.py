#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Recompute the Mask IoU column of P5's tab:annotation_effort.

WHY

That column, 1.00 / 0.87 / 0.84 / 0.91, is the one set of numbers in the paper
that traces to no artefact. It is machine-computed, unlike the stopwatch columns
beside it, so it should have one. The provenance audit of 2026-07-30 searched and
found none: no JSON under ~/bench_any/runs carries a mask_iou or mean_iou key,
the point-noise study reports 0.8067 on Boulders which is none of the four, and
tab:zeroshot_mask gives 0.889 for SAM 2 from SAM 3 boxes rather than 0.87.

The protocol behind the printed values is also unrecorded, and "mean mask IoU
against gold" has several defensible readings that give materially different
numbers. So this script computes all five against the v2 gold and reports them
side by side, rather than picking one and presenting it as the answer. If one
reading reproduces 0.84 for the SAM 3 raw route, that identifies the protocol
and the rest of the column follows from it. If none does, the column has no
recoverable basis and that is the finding.

Covers the two routes computable from artefacts already on disk. Route 1, manual
polygon delineation, is 1.00 by definition, since it is the gold. Route 2, SAM 2
from hand-drawn boxes, needs the card and is not run here.

All IoU work is done with pycocotools RLE ops, so no full-resolution mask is ever
decoded. CPU only.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pycocotools.mask as mu

RUNS = Path.home() / "bench_any" / "runs"
R1 = RUNS / "p5_r1_v2"
OUT_DIR = RUNS / "p5_mask_iou_routes"

CORPORA = {"boulders-penguin": ("boulders", "Boulders"),
           "sanparks-algoa": ("sanparks", "SANParks")}
# the sieve configuration the paper adopts, matching CHOSEN in fusion_strategies_sweep.py
CHOSEN = {"min_area": 64, "nms_iou": 0.8, "max_dim_frac": 1.0, "score_floor": 0.25}
PAPER = {"sam3_raw": 0.84, "sam3_sieve": 0.91}


def load_gold(corpus):
    g = json.loads((RUNS / "obj2_counting" / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    by_id = {im["id"]: im["file_name"] for im in g["images"]}
    hw = {im["file_name"]: (im["height"], im["width"]) for im in g["images"]}
    gold = {fn: [] for fn in by_id.values()}
    for a in g["annotations"]:
        gold[by_id[a["image_id"]]].append(a["segmentation"])
    return gold, hw


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sieve(anns, hw, cfg):
    """Score floor, minimum area and maximum dimension fraction, then box NMS."""
    h, w = hw
    kept = []
    for a in anns:
        if a["score"] < cfg["score_floor"]:
            continue
        x1, y1, x2, y2 = a["box"]
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < cfg["min_area"]:
            continue
        if max(bw / w, bh / h) > cfg["max_dim_frac"]:
            continue
        kept.append(a)
    order = list(np.argsort([a["score"] for a in kept])[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(kept[i]["box"], kept[j]["box"]) <= cfg["nms_iou"]]
    return [kept[i] for i in keep]


def variants(pred_rles, gold_rles):
    """Five readings of 'mean mask IoU against gold' for one image."""
    if not gold_rles:
        return None
    if not pred_rles:
        return {"matched_pairs": None, "per_gold": 0.0, "per_pred": None,
                "per_image_matched": 0.0, "semantic_union": 0.0,
                "n_pred": 0, "n_gold": len(gold_rles), "matched": 0}
    # IoU matrix over RLEs, no mask is decoded
    m = np.asarray(mu.iou(pred_rles, gold_rles, [0] * len(gold_rles)))
    m = m.reshape(len(pred_rles), len(gold_rles))
    # greedy one-to-one matching, highest IoU first
    pairs = []
    used_p, used_g = set(), set()
    flat = sorted(((m[i, j], i, j) for i in range(m.shape[0]) for j in range(m.shape[1])
                   if m[i, j] > 0), reverse=True)
    for v, i, j in flat:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); pairs.append(v)
    union_iou = float(np.asarray(mu.iou([mu.merge(pred_rles)], [mu.merge(gold_rles)], [0])).ravel()[0])
    return {"matched_pairs": float(np.mean(pairs)) if pairs else None,
            "per_gold": float(np.sum(pairs) / len(gold_rles)),
            "per_pred": float(np.sum(pairs) / len(pred_rles)),
            "per_image_matched": float(np.mean(pairs)) if pairs else 0.0,
            "semantic_union": union_iou,
            "n_pred": len(pred_rles), "n_gold": len(gold_rles), "matched": len(pairs)}


def route(corpus, sub, use_sieve):
    gold, hw = load_gold(corpus)
    raw = json.loads((R1 / sub / f"{sub}_sam3_raw.json").read_text())
    per_image, pair_pool, n_gold_tot, n_pred_tot = [], [], 0, 0
    for fn, g_rles in gold.items():
        anns = raw.get(fn, [])
        if use_sieve:
            anns = sieve(anns, hw[fn], CHOSEN)
        v = variants([a["rle"] for a in anns], g_rles)
        if v is None:
            continue
        per_image.append(v)
        n_gold_tot += v["n_gold"]; n_pred_tot += v["n_pred"]
    # pooled over instances, not over images, for the three instance-level readings
    tot_matched_sum = sum(v["per_gold"] * v["n_gold"] for v in per_image)
    tot_matched_n = sum(v["matched"] for v in per_image)
    return {"matched_pairs": round(tot_matched_sum / tot_matched_n, 4) if tot_matched_n else None,
            "per_gold": round(tot_matched_sum / n_gold_tot, 4) if n_gold_tot else None,
            "per_pred": round(tot_matched_sum / n_pred_tot, 4) if n_pred_tot else None,
            "per_image_matched": round(float(np.mean([v["per_image_matched"] for v in per_image])), 4),
            "semantic_union": round(float(np.mean([v["semantic_union"] for v in per_image])), 4),
            "n_images": len(per_image), "n_gold": n_gold_tot, "n_pred": n_pred_tot}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for corpus, (sub, label) in CORPORA.items():
        for name, use_sieve in (("sam3_raw", False), ("sam3_sieve", True)):
            res.setdefault(name, {})[label] = route(corpus, sub, use_sieve)

    keys = ["matched_pairs", "per_gold", "per_pred", "per_image_matched", "semantic_union"]
    L = ["\n## 26. tab:annotation_effort Mask IoU column, recomputed (2026-07-30)\n\n",
         "Five readings of \"mean mask IoU against gold\", against the v2 gold. The printed "
         "column gives 0.84 for SAM 3 raw and 0.91 for the reviewed pipeline. Route 1 is 1.00 "
         "by definition and route 2 needs the card, so neither appears here.\n\n",
         "| route | corpus | matched pairs | per gold inst | per pred | per image | semantic union | imgs | gold | pred |\n",
         "|---|---|---|---|---|---|---|---|---|---|\n"]
    for name in ("sam3_raw", "sam3_sieve"):
        for label, r in res[name].items():
            L.append(f"| {name} | {label} | " + " | ".join(
                f"{r[k]:.4f}" if r[k] is not None else "-" for k in keys) +
                f" | {r['n_images']} | {r['n_gold']} | {r['n_pred']} |\n")

    L.append("\n### Which reading, if any, reproduces the printed cell\n\n")
    for name in ("sam3_raw", "sam3_sieve"):
        target = PAPER[name]
        best, bd = None, 1e9
        for k in keys:
            vals = [res[name][lab][k] for lab in res[name] if res[name][lab][k] is not None]
            if not vals:
                continue
            pooled = float(np.mean(vals))
            if abs(pooled - target) < bd:
                best, bd = (k, pooled), abs(pooled - target)
        L.append(f"- {name}, printed {target}: closest reading is `{best[0]}` at "
                 f"{best[1]:.4f}, off by {bd:.4f}. "
                 f"{'Within rounding, so this identifies the protocol.' if bd < 0.005 else 'Not within rounding.'}\n")
    L.append("\nA reading that lands within 0.005 of the printed value on both routes identifies "
             "the protocol. Anything else means the column cannot be reconstructed and should be "
             "recomputed under a stated protocol rather than defended.\n")

    (OUT_DIR / "mask_iou_routes.json").write_text(json.dumps(res, indent=2))
    (OUT_DIR / "mask_iou_routes.txt").write_text("".join(L))
    print("".join(L), file=sys.stderr)
    print(f"[mask-iou] {time.time()-t0:.1f}s -> {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
