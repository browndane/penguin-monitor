#!/usr/bin/env python3
"""Soft-average fusion on the CORRECTED soft-logit run (prep_11b v2).

Supersedes soft_average_only.py, which ran against the defective soft files and
reported MISSING. Two bugs made those files unusable: boxes were stored as XYWH
when SAM3 returns XYXY, and SAM3 was prompted with the file tag "a_penguin"
instead of the text "a penguin", which changed the detection set. Both are fixed,
so the v2 soft files align with the raw sets one-for-one.

The patch below redirects fusion_strategies_sweep's soft paths to the _v2 files.
It sits at module level because the workers use the spawn start method and
re-import this module in the child, so a patch applied inside main() would not
reach them.

Self-reports to FINDINGS.md §20d and touches its own sentinel. CPU only.
"""
from __future__ import annotations
import json, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np

SWEEP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(Path.home() / "bench_any" / "datasets" / "prep"))
import fusion_strategies_sweep as FS
import prep_10_filebased as P10


def _soft_path_v2(corpus, prompt):
    sub = FS.SUBDIR[corpus]
    tag = "" if prompt == "a_penguin" else "_" + prompt
    return (FS.R1 / sub / f"{sub}_sam3_soft{tag}_v2.json",
            FS.R1 / sub / f"{sub}_soft{tag}_v2_npys")


FS._soft_path = _soft_path_v2

RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
FINDINGS = RUNS / "FINDINGS.md"
DONE = RUNS / "_soft_v2_eval_done"
OUT = RUNS / "soft_v2_eval_results.txt"


def _run_corpus(corpus, label, out_q):
    t0 = time.time()
    man = json.loads((FS.R1 / FS.SUBDIR[corpus] / f"{FS.SUBDIR[corpus]}_tiers.json").read_text())["manifest"]
    fn_hw = FS._fn_hw(corpus)
    raw_pp = FS._raw_per_prompt(corpus)
    soft_data = FS._load_soft(corpus)
    if soft_data is None:
        out_q.put((corpus, label, None, None, "NO_SOFT_DATA"))
        return
    # alignment check: soft and raw must agree per image or the fusion is meaningless
    misaligned = 0
    for p in FS.PROMPTS:
        rp, sp = raw_pp.get(p, {}), soft_data["data"].get(p, {})
        misaligned += sum(1 for fn in sp if len(rp.get(fn, [])) != len(sp[fn]))
    best = None
    for cfg in FS._grid():
        fused = {fn: (FS.fuse_soft(raw_pp, fn, cfg, fn_hw, soft_data) or []) for fn in man}
        pred = FS._to_coco(fused, man)
        r50 = [FS._eval50(P10.build_gt_coco(man, t), pred) for t in ("easy", "medium", "hard")]
        agg50 = float(np.mean(r50))
        if best is None or agg50 > best["agg50"]:
            r95 = [FS._eval5095(P10.build_gt_coco(man, t), pred) for t in ("easy", "medium", "hard")]
            best = {"cfg": cfg, "r50": r50, "agg50": agg50, "r95": r95, "agg95": float(np.mean(r95))}
    out_q.put((corpus, label, best, misaligned, f"{time.time()-t0:.1f}s"))
    print(f"[soft-v2/{label}] agg50={best['agg50']:.1f} agg5095={best['agg95']:.2f} "
          f"misaligned={misaligned} ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)


def main():
    t0 = time.time()
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    for corpus, label in FS.CORPORA.items():
        p = ctx.Process(target=_run_corpus, args=(corpus, label, out_q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    results = {}
    import queue as _q
    deadline = time.time() + 10
    while len(results) < len(FS.CORPORA) and time.time() < deadline:
        try:
            corpus, label, best, misal, info = out_q.get(timeout=2)
            results[corpus] = (label, best, misal, info)
        except _q.Empty:
            continue
    lines = ["\n## 20d. Soft-average on the corrected soft-logit run (2026-07-29)\n\n",
             "Supersedes 20b, which reported MISSING against the defective files. "
             "Same metric path as 20: prep_10 tier-mean, 54-config grid, real scores. "
             "The misaligned column counts images where the soft and raw detection "
             "counts still disagree, and should be 0.\n\n",
             "| corpus | E50 | M50 | H50 | agg50 | E5095 | M5095 | H5095 | agg5095 | best cfg | misaligned | time |\n",
             "|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for corpus, label in FS.CORPORA.items():
        r = results.get(corpus)
        if r is None or r[1] is None:
            lines.append(f"| {label} | - | - | - | - | - | - | - | - | - | - | {r[3] if r else 'MISSING'} |\n")
            continue
        label, b, misal, info = r
        lines.append(f"| {label} | {b['r50'][0]:.1f} | {b['r50'][1]:.1f} | {b['r50'][2]:.1f} | "
                     f"{b['agg50']:.1f} | {b['r95'][0]:.1f} | {b['r95'][1]:.1f} | {b['r95'][2]:.1f} | "
                     f"{b['agg95']:.2f} | ma{b['cfg']['min_area']}n{b['cfg']['nms_iou']}"
                     f"d{b['cfg']['max_dim_frac']}s{b['cfg']['score_floor']} | {misal} | {info} |\n")
    text = "".join(lines)
    OUT.write_text(text)
    with open(FINDINGS, "a") as fh:
        fh.write(text)
    DONE.touch()
    print(f"[soft-v2] done {time.time()-t0:.1f}s sentinel={DONE}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
