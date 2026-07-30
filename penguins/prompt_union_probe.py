#!/usr/bin/env python3
"""Prompt-union probe: does combining 'penguin' and 'a penguin' beat either alone?

Both prompts undercount, with a negative signed bias and 30 to 43% missed.
They miss partly different birds. Picking one throws away the birds the other
finds. Union them, then lower the score threshold, because the geometric sieve
and class-agnostic NMS carry precision. Plural is wrong -- we want one mask per
instance, and 'penguins' would yield merged colonies.

This script scores, per gold corpus, at a sweep of score thresholds:
  A   'penguin' alone, sieved + NMS-deduped
  B   'a penguin' alone, sieved + NMS-deduped
  U   union of both prompts' boxes, sieved + NMS-deduped
  I   agreement set: a box kept only if the other prompt has an IoU>=0.5 partner

The sieve and NMS are the real ones from sanity_sieve.py, the same the SAM3
pipeline uses. Reports MAE, signed bias, missed, spurious, matched, merged,
split per configuration per threshold. Self-writes the table to disk.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
from pycocotools import mask as mu
from PIL import Image

from count_decompose import _rle_from_ann, _decompose_image
from sanity_sieve import mask_passes, class_agnostic_nms

RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")
OUT = RUNS / "prompt_union_probe.txt"
CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
THRS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]


def gold_truth(corpus):
    root = D / corpus
    lbl = root / "labels"; img = root / "images"
    if not lbl.exists():
        lbl = root / "train" / "labels"; img = root / "train" / "images"
    out = {}
    for lf in sorted(lbl.glob("*.txt")):
        imgs = list(img.glob(lf.stem + ".*"))
        if not imgs:
            continue
        w, h = Image.open(imgs[0]).size
        rles = []
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            xy = [float(v) for v in p[1:]]
            poly = [xy[i] * (w if i % 2 == 0 else h) for i in range(len(xy))]
            rles.append(mu.merge(mu.frPyObjects([poly], h, w)))
        out[imgs[0].name] = (rles, (h, w))
    return out


def load_boxes(coco, truth, thr):
    """Return {fn: [(box_xyxy, score, rle)]} for anns with score>=thr."""
    fn_of = {i["id"]: i["file_name"] for i in coco["images"]}
    per = defaultdict(list)
    for a in coco["annotations"]:
        if float(a.get("score", 1.0)) < thr:
            continue
        fn = fn_of.get(a["image_id"])
        if fn not in truth:
            continue
        h, w = truth[fn][1]
        rle = _rle_from_ann(a, h, w)
        bb = mu.toBbox(rle)  # x,y,w,h floats
        box = (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))
        per[fn].append((box, float(a.get("score", 1.0)), rle))
    return per


def dedup(per):
    """Apply sieve + class-agnostic NMS per image. Returns {fn: [rle]}."""
    out = {}
    for fn, items in per.items():
        h, w = (0, 0)
        truth_h_w = None
        # wh needed for sieve: (width, height)
        # we don't have it here cleanly; get from items via rle size
        kept = []
        for box, score, rle in items:
            H, W = rle["size"]  # rle size is [h, w]
            if mask_passes(None, (W, H), bbox=box):
                kept.append((box, score, rle))
        if not kept:
            out[fn] = []
            continue
        boxes = [k[0] for k in kept]
        scores = [k[1] for k in kept]
        rles = [k[2] for k in kept]
        idx = class_agnostic_nms(boxes, scores, 0.55)
        out[fn] = [rles[i] for i in idx]
    return out


def agree(pa, pb, iou_thr=0.5):
    """Agreement set per image: keep a box from A only if B has an IoU>=thr box."""
    out = {}
    for fn in set(pa) | set(pb):
        a = pa.get(fn, [])
        b = pb.get(fn, [])
        if not a or not b:
            out[fn] = []
            continue
        ba = [mu.toBbox(r) for r in a]
        bb = [mu.toBbox(r) for r in b]
        def xyxy(b):
            return (b[0], b[1], b[0] + b[2], b[1] + b[3])
        ba_xy = [xyxy(b) for b in ba]
        bb_xy = [xyxy(b) for b in bb]
        kept = []
        for i, r in enumerate(a):
            best = 0.0
            for j in range(len(b)):
                # box IoU
                ax1, ay1, ax2, ay2 = ba_xy[i]
                bx1, by1, bx2, by2 = bb_xy[j]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 < ix1 or iy2 < iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                iou = inter / ua if ua > 0 else 0.0
                best = max(best, iou)
            if best >= iou_thr:
                kept.append(r)
        out[fn] = kept
    return out


def score(preds, truth):
    n = len(truth); ngt = 0
    agg = defaultdict(float)
    mae = 0; sgn = 0
    for fn, (gts, _) in truth.items():
        ngt += len(gts)
        p = preds.get(fn, [])
        ev = _decompose_image(p, gts, 0.5, 0.5)
        ev.pop("_fate", None)
        for k, v in ev.items():
            agg[k] += v
        mae += abs(len(p) - len(gts))
        sgn += len(p) - len(gts)
    return dict(matched=int(agg["matched"]), missed=int(agg["missed"]),
                merged=int(agg["merged"]), split=int(agg["split"]),
                spurious=int(agg["spurious"]),
                MAE=mae / n, signed=sgn / n, ngt=ngt)


def main():
    lines = ["# Prompt-union probe, 2026-07-29\n",
             "Combines 'penguin' and 'a penguin' (union), sieve + class-agnostic NMS,\n",
             "threshold swept down. Plural 'penguins' is wrong -- we want one mask per\n",
             "instance and a plural prompt merges colonies.\n\n"]
    for corpus, label in CORPORA.items():
        truth = gold_truth(corpus)
        ja = json.load(open(RUNS / "sam3_prompts_gold" / f"{corpus}_penguin.json"))
        jb = json.load(open(RUNS / "sam3_prompts_gold" / f"{corpus}_a_penguin.json"))
        lines.append(f"\n## {label} ({len(truth)} images, {sum(len(t[0]) for t in truth.values())} GT)\n\n")
        lines.append(f"{'thr':>4s} {'cfg':>4s} {'preds':>6s} {'match':>6s} {'miss':>5s} "
                     f"{'merge':>5s} {'split':>5s} {'spur':>5s} {'MAE':>5s} {'signed':>7s}\n")
        for thr in THRS:
            pa = dedup(load_boxes(ja, truth, thr))
            pb = dedup(load_boxes(jb, truth, thr))
            # union: merge both prompts' rles, recompute boxes from rles, re-sieve + NMS
            pu_items = {}
            for fn in truth:
                rs = pa.get(fn, []) + pb.get(fn, [])
                items = []
                for r in rs:
                    bb = mu.toBbox(r)
                    box = (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))
                    items.append((box, 0.5, r))
                pu_items[fn] = items
            pu = dedup(pu_items)
            for name, pr in (("A", pa), ("B", pb), ("U", pu)):
                s = score(pr, truth)
                preds = s["matched"] + s["spurious"] + s["merged"] + s["split"]
                lines.append(f"{thr:>4.2f} {name:>4s} {preds:6d} {s['matched']:6d} "
                             f"{s['missed']:5d} {s['merged']:5d} {s['split']:5d} "
                             f"{s['spurious']:5d} {s['MAE']:5.2f} {s['signed']:+7.2f}\n")
        lines.append("\n")
    OUT.write_text("".join(lines))
    print("".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
