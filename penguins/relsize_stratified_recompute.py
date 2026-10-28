#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Per-tier mask mAP@50 under each border rule, on P5's headline metric.

WHY
---
Every mask mAP in P5 was computed with no border rule at all (union_sieve_sweep's
54-config grid sweeps min_area, nms_iou, max_dim_frac and score_floor, and its
_sieve carries no edge test). The precision/count sweep in p5_sieve_composition
shows the relative border rule beats the absolute one on both corpora, but that
sweep reports precision, recall and count error, not mask mAP. So the headline
69.0 and 38.7 have never been recomputed under the rule that wins.

This closes that gap. Same eval path as union_sieve_sweep (prep_10_filebased:
build_gt_coco from the tiers manifest, iscrowd-ignore per tier, eval_tier), same
chosen config, three border rules:

  none      what the paper reports today
  stub      absolute, reject a border-touching mask under area_k * a_min
  relsize   relative, reject a border-touching mask under k * median interior area
  fiftyone  the FiftyOne caller's rule: border margin 0.5% of the image dimension,
            reject under FIFTYONE_K * median box area over every detection

Border-rule implementations are imported from sanity_sieve, never retyped.

OUTPUT
------
~/bench_any/runs/p5_relsize_stratified/relsize_stratified.json and a summary
block appended to summary.md. Touches _done. CPU only, no GPU.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

PREP_DIR = Path("/home/user/bench_any/datasets/prep")
sys.path.insert(0, str(PREP_DIR))
import prep_10_filebased as P10  # noqa: E402

from sanity_sieve import (  # noqa: E402
    reject_edge_stub, reject_edge_relsize, median_interior_area,
    A_MIN, RHO_MAX, NMS_IOU, EDGE_BORDER, EDGE_STUB_AREA_K,
)
from sieve_composition import _edge_rejects, FIFTYONE_K  # noqa: E402

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
OUT = Path.home() / "bench_any" / "runs" / "p5_relsize_stratified"

CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
SUBDIR = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks"}

# The config every published mask mAP was computed under (union_sieve_sweep.CHOSEN),
# and the sieve's own constants, which the precision and count table runs under
# (sieve_composition.CANON: sanity_sieve's A_MIN, RHO_MAX and NMS_IOU, no score floor).
BASES = {
    "asrun": {"min_area": 64, "nms_iou": 0.8, "max_dim_frac": 1.0, "score_floor": 0.25},
    "canon": {"min_area": A_MIN, "nms_iou": NMS_IOU, "max_dim_frac": RHO_MAX,
              "score_floor": 0.0},
}

# Published single-prompt tier-mean, the reproduction target for the "none" arm.
PAPER = {"boulders-penguin": {"E": 98.0, "M": 85.6, "H": 23.3, "agg": 69.0},
         "sanparks-algoa": {"E": 0.0, "M": 84.4, "H": 31.8, "agg": 38.7}}

RELSIZE_KS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TIERS = ("easy", "medium", "hard")


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "relsize_stratified.log", "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def _iou(b1, b2):
    xa, ya, x2a, y2a = b1
    xb, yb, x2b, y2b = b2
    ix1, iy1 = max(xa, xb), max(ya, yb)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2a - xa) * (y2a - ya)
    a2 = (x2b - xb) * (y2b - yb)
    return inter / (a1 + a2 - inter)


def _nms(boxes, scores, iou_thr):
    if not boxes:
        return []
    order = list(np.argsort(scores)[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in [j for j in order if _iou(boxes[i], boxes[j]) > iou_thr]:
            order.remove(j)
    return keep


def _raw(corpus):
    sub = SUBDIR[corpus]
    return json.loads((R1 / sub / f"{sub}_sam3_raw.json").read_text())


def _fn_hw(corpus):
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    return {im["file_name"]: (im["height"], im["width"]) for im in gt["images"]}


def _sieve(preds_by_fn, fn_hw, cfg, edge_rule, relsize_k):
    """CHOSEN config plus one border rule. Border test runs after the geometric
    tests and before non-maximum suppression, matching sanity_sieve's ordering."""
    out = {}
    for fn, anns in preds_by_fn.items():
        h, w = fn_hw[fn]
        img_wh = (w, h)
        kept = []
        for a in anns:
            if a["score"] < cfg["score_floor"]:
                continue
            x1, y1, x2, y2 = a["box"]
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            if area < cfg["min_area"]:
                continue
            if max(bw, bh) > cfg["max_dim_frac"] * max(w, h):
                continue
            kept.append((a, (x1, y1, x2, y2), area))

        if kept and edge_rule != "none":
            bboxes = [b for _, b, _ in kept]
            areas = [ar for _, _, ar in kept]
            med = median_interior_area(bboxes, areas, img_wh, EDGE_BORDER)
            med_all = float(np.median([max(1, b[2] - b[0] + 1) * max(1, b[3] - b[1] + 1)
                                       for b in bboxes]))
            survivors = []
            for a, bb, ar in kept:
                if edge_rule == "stub":
                    rej = reject_edge_stub(bb, img_wh, ar, A_MIN, EDGE_BORDER,
                                           EDGE_STUB_AREA_K)
                elif edge_rule == "fiftyone":
                    rej = _edge_rejects("fiftyone", bb, img_wh, ar, med, {},
                                        median_all_box=med_all)
                else:
                    rej = reject_edge_relsize(bb, img_wh, ar, med, EDGE_BORDER,
                                              relsize_k)
                if not rej:
                    survivors.append((a, bb, ar))
            kept = survivors

        if not kept:
            continue
        boxes = [b for _, b, _ in kept]
        scores = [a["score"] for a, _, _ in kept]
        keep_idx = _nms(boxes, scores, cfg["nms_iou"])
        out[fn] = [{"rle": kept[i][0]["rle"], "score": kept[i][0]["score"],
                    "box": kept[i][0]["box"]} for i in keep_idx]
    return out


def _score(sieved, man):
    pred = P10.build_pred_coco(sieved, man)
    per_tier = {t: P10.eval_tier(P10.build_gt_coco(man, t), pred) for t in TIERS}
    per_tier["agg"] = float(np.mean([per_tier[t] for t in TIERS]))
    per_tier["n_kept"] = int(sum(len(v) for v in sieved.values()))
    return per_tier


def _run_base(corpus, label, base_name, cfg, man, fn_hw, raw):
    arms = {}
    log(f"\n=== {label} / {base_name} {cfg} ===")
    arms["none"] = _score(_sieve(raw, fn_hw, cfg, "none", 0.0), man)
    p = PAPER[corpus]
    d = arms["none"]["agg"] - p["agg"]
    log(f"  none    E={arms['none']['easy']:5.1f} M={arms['none']['medium']:5.1f} "
        f"H={arms['none']['hard']:5.1f} agg={arms['none']['agg']:5.1f} "
        f"(paper {p['agg']:.1f}, delta {d:+.2f})")

    arms["stub"] = _score(_sieve(raw, fn_hw, cfg, "stub", 0.0), man)
    log(f"  stub    E={arms['stub']['easy']:5.1f} M={arms['stub']['medium']:5.1f} "
        f"H={arms['stub']['hard']:5.1f} agg={arms['stub']['agg']:5.1f}")

    arms["fiftyone"] = _score(_sieve(raw, fn_hw, cfg, "fiftyone", FIFTYONE_K), man)
    a = arms["fiftyone"]
    log(f"  f1one   E={a['easy']:5.1f} M={a['medium']:5.1f} "
        f"H={a['hard']:5.1f} agg={a['agg']:5.1f} kept={a['n_kept']}")

    for k in RELSIZE_KS:
        key = f"relsize_{k:.2f}"
        arms[key] = _score(_sieve(raw, fn_hw, cfg, "relsize", k), man)
        a = arms[key]
        log(f"  rel {k:.2f} E={a['easy']:5.1f} M={a['medium']:5.1f} "
            f"H={a['hard']:5.1f} agg={a['agg']:5.1f} kept={a['n_kept']}")

    best = max((kk for kk in arms if kk.startswith("relsize")),
               key=lambda kk: arms[kk]["agg"])
    log(f"  best relative: {best} agg={arms[best]['agg']:.1f} "
        f"vs none {arms['none']['agg']:.1f} "
        f"({arms[best]['agg'] - arms['none']['agg']:+.1f})")
    return {"cfg": cfg, "arms": arms, "best_relsize": best}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"bases": BASES, "corpora": {}}

    for corpus, label in CORPORA.items():
        man = json.loads((R1 / SUBDIR[corpus] /
                          f"{SUBDIR[corpus]}_tiers.json").read_text())["manifest"]
        fn_hw = _fn_hw(corpus)
        raw = _raw(corpus)
        bases = {b: _run_base(corpus, label, b, c, man, fn_hw, raw)
                 for b, c in BASES.items()}
        results["corpora"][corpus] = {"label": label, "paper": PAPER[corpus],
                                      "bases": bases}

    (OUT / "relsize_stratified.json").write_text(json.dumps(results, indent=1))

    lines = [f"\n## Border rule vs per-tier mask mAP@50 ({time.strftime('%Y-%m-%d')})\n\n",
             "Eval: prep_10_filebased, tiers manifest GT, iscrowd-ignore per tier. ",
             f"Base configurations {BASES}. Border rules from sanity_sieve and ",
             "sieve_composition.\n\n",
             "| Corpus | Base | Border rule | Easy | Medium | Hard | Aggregate |\n",
             "|---|---|---|---|---|---|---|\n"]
    for corpus, c in results["corpora"].items():
        for base, b in c["bases"].items():
            for arm, a in b["arms"].items():
                lines.append(f"| {c['label']} | {base} | {arm} | {a['easy']:.1f} | "
                             f"{a['medium']:.1f} | {a['hard']:.1f} | {a['agg']:.1f} |\n")
    (OUT / "summary.md").write_text("".join(lines))

    (OUT / "_done").touch()
    log(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
