#!/usr/bin/env python3
"""Why the soft-logit acceptance test passes on Boulders and fails on SANParks.

WHY

`soft_logit_verify.py` scores each detection's probability map against that
detection's own binary mask, downsampled to the map's native 288x288. Boulders
clears the 0.85 bar at 0.93; SANParks sits at 0.55 with a 0.56 median and only
4 to 6 percent of pairs below 0.1. That profile is not the misfiling signature,
which was 0.009 with 95 percent of pairs below 0.1. Two candidates remain.

  resolution   SANParks birds are small and dense, so at 288x288 a bird covers
               a few pixels and a one-pixel boundary disagreement halves its
               IoU. The maps are correctly paired and the 0.85 bar is simply
               wrong for objects this small.
  geometry     a second padding or letterbox defect, in which case large
               objects are depressed too and the FAIL stands as written.

The discriminator is per-pair IoU against the object's area in the map grid,
computed for both corpora. Resolution predicts IoU rising with area on both,
with SANParks' large objects recovering to about 0.9 and Boulders' rare small
objects scoring badly. Geometry predicts SANParks staying flat and low across
every area bin. A centroid offset in map pixels is reported alongside, since a
padding error shifts masks rather than eroding them.

This settles a diagnosis, not the blocker. Soft-average fusion stays unreported
either way until the sweep runs on maps that pass a bar chosen for the object
sizes in the corpus.

Self-reports a bounded block and writes a sentinel. CPU only.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mu

RUNS = Path.home() / "bench_any" / "runs"
R1 = RUNS / "p5_r1_v2"
OUT_TXT = RUNS / "obj2_counting" / "soft_logit_area_stratify.txt"
OUT_JSON = RUNS / "obj2_counting" / "soft_logit_area_stratify.json"
DONE = RUNS / "obj2_counting" / "_soft_logit_area_stratify_done"

CORPORA = {"boulders": "Boulders", "sanparks": "SANParks"}
PROMPTS = ["a_penguin", "penguin", "african_penguin"]

# area of the downsampled binary mask, in map pixels
BINS = [0, 4, 16, 64, 256, 1024, 10 ** 9]
BIN_LABELS = ["<4", "4-16", "16-64", "64-256", "256-1024", ">=1024"]


def _tag(prompt):
    return "" if prompt == "a_penguin" else "_" + prompt


def _paths(sub, prompt):
    t = _tag(prompt)
    return (R1 / sub / f"{sub}_sam3_raw{t}.json",
            R1 / sub / f"{sub}_sam3_soft{t}_v2.json",
            R1 / sub / f"{sub}_soft{t}_v2_npys")


def _centroid(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan)


def pairs(sub, prompt):
    """One record per detection: IoU at map resolution, areas, centroid offset."""
    raw_p, soft_p, npy_dir = _paths(sub, prompt)
    if not (raw_p.exists() and soft_p.exists() and npy_dir.exists()):
        return []
    raw = json.loads(raw_p.read_text())
    soft = json.loads(soft_p.read_text())
    out = []
    for fn in sorted(raw):
        if len(raw[fn]) != len(soft.get(fn, [])):
            continue
        for k, det in enumerate(raw[fn]):
            npy_path = npy_dir / soft[fn][k]["prob_npy"]
            if not npy_path.exists():
                continue
            pr = np.load(npy_path).astype(np.float32)
            full = mu.decode(det["rle"]).astype(np.uint8)
            small = cv2.resize(full, (pr.shape[1], pr.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            m_soft = pr >= 0.5
            union = (small | m_soft).sum()
            cb, cs = _centroid(small), _centroid(m_soft)
            out.append({
                "iou": float((small & m_soft).sum() / union) if union else 0.0,
                "area_map": int(small.sum()),
                "area_native": int(full.sum()),
                "soft_area_map": int(m_soft.sum()),
                "img_px": int(full.shape[0] * full.shape[1]),
                "dx": cs[0] - cb[0], "dy": cs[1] - cb[1],
            })
    return out


def summarise(recs):
    a = np.array([r["iou"] for r in recs])
    am = np.array([r["area_map"] for r in recs])
    idx = np.digitize(am, BINS[1:-1], right=False)
    rows = []
    for b, label in enumerate(BIN_LABELS):
        sel = idx == b
        if not sel.any():
            rows.append({"bin": label, "n": 0})
            continue
        rows.append({"bin": label, "n": int(sel.sum()),
                     "mean_iou": round(float(a[sel].mean()), 3),
                     "median_iou": round(float(np.median(a[sel])), 3),
                     "median_area_native": int(np.median([r["area_native"] for r, s in zip(recs, sel) if s]))})
    big = am >= 256
    dx = np.array([r["dx"] for r in recs]); dy = np.array([r["dy"] for r in recs])
    return {"n": len(recs), "mean_iou": round(float(a.mean()), 3),
            "bins": rows,
            "n_big": int(big.sum()),
            "mean_iou_big": round(float(a[big].mean()), 3) if big.any() else None,
            "median_area_map": int(np.median(am)),
            "median_offset_px": [round(float(np.nanmedian(dx)), 2), round(float(np.nanmedian(dy)), 2)],
            "median_abs_offset_px": [round(float(np.nanmedian(np.abs(dx))), 2),
                                     round(float(np.nanmedian(np.abs(dy))), 2)]}


def main():
    t0 = time.time()
    per_corpus, per_prompt = {}, {}
    for sub, label in CORPORA.items():
        recs = []
        for prompt in PROMPTS:
            p = pairs(sub, prompt)
            if p:
                per_prompt[f"{label}/{prompt}"] = summarise(p)
            recs += p
        if recs:
            per_corpus[label] = summarise(recs)

    L = ["\n## 20e. Soft-logit self-IoU against object size in the map grid (2026-07-30)\n\n",
         "Each detection's probability map is thresholded at 0.5 and scored against its own "
         "binary mask downsampled to the map's 288x288, which is what the acceptance test does. "
         "Pairs are pooled over the three prompts and binned by the downsampled mask's area in "
         "map pixels.\n\n",
         "| corpus | bin (map px) | pairs | mean IoU | median IoU | median native area (px) |\n",
         "|---|---|---|---|---|---|\n"]
    for label, s in per_corpus.items():
        for r in s["bins"]:
            if not r["n"]:
                continue
            L.append(f"| {label} | {r['bin']} | {r['n']} | {r['mean_iou']} | {r['median_iou']} | "
                     f"{r['median_area_native']} |\n")
    L.append("\n| corpus | pairs | mean IoU | median area (map px) | pairs >=256 map px | "
             "mean IoU there | median centroid offset (px) |\n|---|---|---|---|---|---|---|\n")
    for label, s in per_corpus.items():
        L.append(f"| {label} | {s['n']} | {s['mean_iou']} | {s['median_area_map']} | {s['n_big']} | "
                 f"{s['mean_iou_big']} | {s['median_offset_px'][0]}, {s['median_offset_px'][1]} |\n")

    b, sp = per_corpus.get("Boulders"), per_corpus.get("SANParks")
    verdict = "INCONCLUSIVE"
    if b and sp:
        big_ok = sp["mean_iou_big"] is not None and sp["mean_iou_big"] >= 0.85
        shifted = max(abs(sp["median_offset_px"][0]), abs(sp["median_offset_px"][1])) > 1.0
        if big_ok and not shifted:
            verdict = "RESOLUTION"
        elif not big_ok and sp["n_big"] >= 30:
            verdict = "GEOMETRY"
    L.append(f"\nVerdict: {verdict}. RESOLUTION means the maps are paired correctly and the flat "
             "0.85 bar is wrong for small objects; GEOMETRY means a second defect depresses large "
             "objects too and the FAIL stands as written. Either way no soft-average number is "
             "reportable until the sweep runs.\n")

    text = "".join(L)
    OUT_TXT.write_text(text)
    OUT_JSON.write_text(json.dumps({"per_corpus": per_corpus, "per_prompt": per_prompt,
                                    "verdict": verdict, "bins": BIN_LABELS}, indent=1))
    DONE.write_text(verdict)
    print(text, file=sys.stderr)
    print(f"[area-stratify] {time.time()-t0:.1f}s -> {OUT_TXT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
