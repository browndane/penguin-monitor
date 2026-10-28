#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Score the four African-penguin-trained checkpoints that had never been run
out of distribution, and report the in-distribution minus out-of-distribution gap.

WHY
---
tab:percorpus carries six supervised comparators. Four more models were trained on
the same african-penguin fair-grid corpus, under the same gradient-step-matched
protocol, and only ever scored on the held-out in-distribution split: yolo26x-seg,
eomt-smallplus, dinov3-m2f-smallplus and dinov3-m2f-vith16plus. Their checkpoints
are on disk and the two v2 gold corpora are on disk, so the missing cells are an
inference pass, not a training run.

All four carry native masks. Nothing here prompts SAM 2 or SAM 3 from their boxes:
the in-distribution number they are compared against is a native-mask seg mAP50, so
swapping the mask source out of distribution would compare two different things and
the gap would not mean what it says.

The in-distribution side comes from PAPER in prep_13_supervised_reeval.py, which was
checked cell by cell against the e150/be64 fair-grid rows of benchmark_results.csv.
Note the two sides are not the same metric: in-distribution is whole-split seg mAP50
and out-of-distribution is the mean of three difficulty tiers, so the gap reads as
coverage loss on gold rather than as one metric measured twice.

OUTPUT
------
Per cell, prep_13's own <ckpt>_pred.json and <ckpt>_reeval.json under the out dir.
Then ood_coverage_gap_results.json, a FINDINGS section, and the sentinel
_ood_coverage_gap_done. Nothing polls it; read the log when it is finished.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PREP = Path(os.path.expanduser("~/bench_any/datasets/prep/prep_13_supervised_reeval.py"))
PY = os.path.expanduser("~/cvML/bin/python")
OUT = Path(os.path.expanduser("~/bench_any/runs/ood_coverage_gap"))
FINDINGS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting/FINDINGS.md"))
SENTINEL = OUT / "_ood_coverage_gap_done"
LOG = OUT / "ood_coverage_gap.log"

CKPTS = ["yolo26x-seg", "eomt-smallplus", "dinov3-m2f-smallplus", "dinov3-m2f-vith16plus"]
CORPORA = ["boulders", "sanparks"]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def run_cell(ckpt, corpus):
    reeval = OUT / corpus / f"{ckpt}_reeval.json"
    if reeval.is_file():
        log(f"  [{ckpt}/{corpus}] already scored, reusing {reeval.name}")
        return json.loads(reeval.read_text())
    t0 = time.time()
    cmd = [PY, str(PREP), "--ckpt", ckpt, "--corpus", corpus, "--out-dir", str(OUT)]
    log(f"  [{ckpt}/{corpus}] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
    if proc.returncode != 0 or not reeval.is_file():
        log(f"  [{ckpt}/{corpus}] FAILED rc={proc.returncode} after {time.time()-t0:.0f}s\n{tail}")
        return None
    log(f"  [{ckpt}/{corpus}] done in {time.time()-t0:.0f}s")
    return json.loads(reeval.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== OOD coverage gap start {time.strftime('%Y-%m-%d %H:%M')} ===")
    results = {}
    for ckpt in CKPTS:
        for corpus in CORPORA:
            r = run_cell(ckpt, corpus)
            results.setdefault(ckpt, {})[corpus] = r
            (OUT / "ood_coverage_gap_results.json").write_text(json.dumps(results, indent=2))

    lines = [f"\n## 24. Out-of-distribution coverage for the four unscored supervised "
             f"checkpoints ({time.strftime('%Y-%m-%d')})\n\n",
             "yolo26x-seg, eomt-smallplus, dinov3-m2f-smallplus and dinov3-m2f-vith16plus were\n"
             "trained on the african-penguin fair-grid corpus alongside the six comparators in\n"
             "`tab:percorpus` and had only ever been scored in distribution. All four carry native\n"
             "masks, so nothing here is refined by SAM 2 or SAM 3. In-distribution figures are the\n"
             "e150/be64 fair-grid seg mAP50 rows of `benchmark_results.csv`; out-of-distribution\n"
             "figures are prep_13's per-tier COCOeval mask mAP@0.5 averaged over the three tiers.\n"
             "The two sides are different metrics, so read the gap as coverage lost on gold rather\n"
             "than as one metric measured twice. The gold sets are the v2 tier manifests behind every\n"
             "other out-of-distribution number in the paper, `p5_r1_v2/boulders/boulders_tiers.json`\n"
             "over 123 images and `p5_r1_v2/sanparks/sanparks_tiers.json` over 51.\n\n",
             "| model | in-dist mAP50 | Boulders agg | gap | SANParks agg | gap |\n"
             "|---|---|---|---|---|---|\n"]
    for ckpt in CKPTS:
        per = results.get(ckpt, {})
        ind = None
        for r in per.values():
            if r and r.get("paper_in_dist_mAP50") is not None:
                ind = r["paper_in_dist_mAP50"] * 100.0
        cells = [f"{ind:.1f}" if ind is not None else "-"]
        for corpus in CORPORA:
            r = per.get(corpus)
            agg = r.get("agg") if r else None
            cells.append(f"{agg:.1f}" if agg is not None else "-")
            cells.append(f"{agg - ind:+.1f}" if (agg is not None and ind is not None) else "-")
        lines.append("| " + ckpt + " | " + " | ".join(cells) + " |\n")
    lines.append("\nA dash is a cell that did not produce a score; the log says why.\n")
    lines.append("SANParks holds a single easy-tier gold instance against 801 medium and 52 hard, "
                 "so the easy column there is one instance carrying a third of the aggregate and "
                 "should not be read as a tier score. Per-tier gold counts sit in each "
                 "`<ckpt>_reeval.json` under `n_gt_per_tier`.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(lines))
    log("".join(lines))

    SENTINEL.write_text("done\n")
    log(f"=== OOD coverage gap complete {time.strftime('%Y-%m-%d %H:%M')} ===")


if __name__ == "__main__":
    main()
