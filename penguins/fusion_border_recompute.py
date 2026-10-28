#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Prompt-fusion strategies re-scored with the pipeline's own border rule.

WHY
---
tab:fusion reports a single-prompt baseline of 69.0 on Boulders and a best fused
figure of 70.2. Both were computed with no border test. The border rule the
pipeline actually runs with moves the same single-prompt predictions to 81.6, so
a 1.2 point fusion gain sits inside a 12.6 point border effect. Reporting the
fusion table on the old baseline while the per-corpus table moves to the border
rule would put two different figures for one configuration in one paper.

This re-runs the fusion sweep unchanged except for the border test. Strategies,
the 54 cell grid, tau, and the per-strategy selection rule are imported from
fusion_strategies_sweep and never retyped; the border rule is imported from
sanity_sieve and sieve_composition, as in relsize_stratified_recompute. The only
edit is that _pass_sieve gains a border arm, applied to every strategy and to the
baseline alike, so each row moves for one reason.

Both arms are reported. "none" must reproduce the published table, which is the
check that the wrapper changed nothing else.

OUTPUT
------
~/bench_any/runs/p5_fusion_border/fusion_border.json plus a summary block.
CPU only, no GPU.
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
sys.path.insert(0, "/home/user/bench_any/datasets/prep")

import fusion_strategies_sweep as F  # noqa: E402
from sanity_sieve import (  # noqa: E402
    reject_edge_stub, median_interior_area, A_MIN, EDGE_BORDER, EDGE_STUB_AREA_K,
)
from sieve_composition import _edge_rejects, FIFTYONE_K  # noqa: E402

OUT = Path.home() / "bench_any" / "runs" / "p5_fusion_border"
TIERS = ("easy", "medium", "hard")
ARMS = ("none", "stub", "fiftyone")

# Published tab:fusion, the reproduction target for the "none" arm.
PAPER = {"boulders-penguin": {"single": 69.0, "mask_union": 70.2},
         "sanparks-algoa": {"single": 38.7, "mask_intersection": 40.0}}

_ORIG_PASS_SIEVE = F._pass_sieve
_ARM = "none"


def _pass_sieve_border(items, fn_hw, fn, cfg):
    """The sweep's own geometric tests, then the border test, matching
    sanity_sieve's ordering: border runs after the geometric tests and before
    non-maximum suppression."""
    kept = _ORIG_PASS_SIEVE(items, fn_hw, fn, cfg)
    if not kept or _ARM == "none":
        return kept
    h, w = fn_hw[fn]
    img_wh = (w, h)
    boxes, areas = [], []
    for a in kept:
        x1, y1, x2, y2 = a["box"]
        boxes.append((x1, y1, x2, y2))
        areas.append((x2 - x1) * (y2 - y1))
    med = median_interior_area(boxes, areas, img_wh, EDGE_BORDER)
    med_all = float(np.median([max(1, b[2] - b[0] + 1) * max(1, b[3] - b[1] + 1)
                               for b in boxes]))
    out = []
    for a, bb, ar in zip(kept, boxes, areas):
        if _ARM == "stub":
            rej = reject_edge_stub(bb, img_wh, ar, A_MIN, EDGE_BORDER, EDGE_STUB_AREA_K)
        else:
            rej = _edge_rejects("fiftyone", bb, img_wh, ar, med, {}, median_all_box=med_all)
        if not rej:
            out.append(a)
    return out


F._pass_sieve = _pass_sieve_border


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "fusion_border.log", "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def _tier_scores(pred, man):
    return [F._eval50(F.P10.build_gt_coco(man, t), pred) for t in TIERS]


def _baseline(raw_pp, fn_hw, man):
    cfg = F.CHOSEN
    by_fn = {fn: F._pass_sieve(raw_pp.get(F.SINGLE, {}).get(fn, []), fn_hw, fn, cfg)
             for fn in man}
    by_fn = {fn: [by_fn[fn][i] for i in F._nms(by_fn[fn], cfg["nms_iou"])] for fn in man}
    return _tier_scores(F._to_coco(by_fn, man), man)


def _strategy(name, raw_pp, fn_hw, man, soft_data):
    """Best cell of the sweep's grid for one strategy, its own selection rule."""
    best = None
    for cfg in F._grid():
        if name == "union":
            fused = {fn: F.fuse_union(raw_pp, fn, cfg, fn_hw) for fn in man}
        elif name == "consensus":
            fused = {fn: F.fuse_consensus(raw_pp, fn, cfg, fn_hw) for fn in man}
        elif name == "soft_average":
            if soft_data is None:
                return None
            fused = {fn: F.fuse_soft(raw_pp, fn, cfg, fn_hw, soft_data) for fn in man}
        else:
            mode = name.replace("mask_", "")
            fused = {fn: F.fuse_mask(raw_pp, fn, cfg, fn_hw, mode) for fn in man}
        r50 = _tier_scores(F._to_coco(fused, man), man)
        agg = float(np.mean(r50))
        if best is None or agg > best["agg50"]:
            best = {"cfg": cfg, "r50": r50, "agg50": agg}
    return best


def main():
    global _ARM
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    strategies = ["union", "consensus", "mask_union", "mask_intersection",
                  "mask_majority", "soft_average"]
    results = {"arms": list(ARMS), "corpora": {}}

    for corpus, label in F.CORPORA.items():
        man = json.loads((F.R1 / F.SUBDIR[corpus] /
                          f"{F.SUBDIR[corpus]}_tiers.json").read_text())["manifest"]
        fn_hw = F._fn_hw(corpus)
        raw_pp = F._raw_per_prompt(corpus)
        soft_data = F._load_soft(corpus)
        per_arm = {}
        for arm in ARMS:
            _ARM = arm
            log(f"\n=== {label} / border={arm} ===")
            base = _baseline(raw_pp, fn_hw, man)
            entry = {"single": {"r50": base, "agg50": float(np.mean(base))}}
            log(f"  single      agg50={entry['single']['agg50']:5.1f}")
            for s in strategies:
                b = _strategy(s, raw_pp, fn_hw, man, soft_data)
                if b is None:
                    continue
                entry[s] = b
                log(f"  {s:17s} agg50={b['agg50']:5.1f} "
                    f"cfg=ma{b['cfg']['min_area']}n{b['cfg']['nms_iou']}"
                    f"d{b['cfg']['max_dim_frac']}s{b['cfg']['score_floor']}")
            per_arm[arm] = entry
        results["corpora"][corpus] = {"label": label, "paper": PAPER[corpus],
                                      "arms": per_arm}
        (OUT / "fusion_border.json").write_text(json.dumps(results, indent=1))

    lines = [f"\n## Prompt fusion under the border rule ({time.strftime('%Y-%m-%d')})\n\n",
             "Sweep, strategies and grid from fusion_strategies_sweep; border rule from ",
             "sanity_sieve and sieve_composition. Per-strategy best cell, tier-mean mAP@50.\n\n",
             "| Corpus | Border | Strategy | Easy | Medium | Hard | Aggregate |\n",
             "|---|---|---|---|---|---|---|\n"]
    for corpus, c in results["corpora"].items():
        for arm, entry in c["arms"].items():
            for s, b in entry.items():
                lines.append(f"| {c['label']} | {arm} | {s} | {b['r50'][0]:.1f} | "
                             f"{b['r50'][1]:.1f} | {b['r50'][2]:.1f} | {b['agg50']:.1f} |\n")
    (OUT / "summary.md").write_text("".join(lines))
    (OUT / "_done").touch()
    log(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
