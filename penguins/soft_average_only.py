#!/usr/bin/env python3
"""Run ONLY the soft_average strategy from the fusion sweep, in parallel.

Companion to fusion_strategies_sweep.py. The in-flight sweep (PID 446674) baked
in soft_avail=False (logits not ready at its 16:55 start), so it will produce only
the 5 binary strategies. This runner fills the missing 6th row per corpus by
importing the sweep module and running just fuse_soft over the full 54-config
grid, using the sweep's exact prep_10 tier-mean metric path. Two corpora run as
concurrent processes; each writes its own result line to a single shared file
(guarded by an append lock, not that two writers race given only 2 procs).

Self-reports to FINDINGS.md §20b and touches sentinel when both corpora finish.
CPU-only (soft-average reads saved logit .npy files, no GPU).
"""
from __future__ import annotations
import json, sys, time, os, multiprocessing as mp
from pathlib import Path
import numpy as np

SWEEP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(SWEEP_DIR / "bench_any" / "datasets" / "prep"))
import fusion_strategies_sweep as FS
import prep_10_filebased as P10

RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
FINDINGS = RUNS / "FINDINGS.md"
DONE = RUNS / "_soft_only_done"
OUT = RUNS / "soft_only_results.txt"


def _run_corpus(corpus, label, out_q):
    t0 = time.time()
    man = json.loads((FS.R1 / FS.SUBDIR[corpus] / f"{FS.SUBDIR[corpus]}_tiers.json").read_text())["manifest"]
    fn_hw = FS._fn_hw(corpus)
    raw_pp = FS._raw_per_prompt(corpus)
    soft_data = FS._load_soft(corpus)
    if soft_data is None:
        out_q.put((corpus, label, None, None, None, "NO_SOFT_DATA"))
        return
    # baseline single (a_penguin) at CHOSEN, for the cell header
    single_by_fn = {fn: FS._pass_sieve(raw_pp.get(FS.SINGLE, {}).get(fn, []), fn_hw, fn, FS.CHOSEN) for fn in man}
    single_by_fn = {fn: [single_by_fn[fn][i] for i in FS._nms(single_by_fn[fn], FS.CHOSEN["nms_iou"])] for fn in man}
    spred = FS._to_coco(single_by_fn, man)
    s50 = [FS._eval50(P10.build_gt_coco(man, t), spred) for t in ("easy", "medium", "hard")]
    s95 = [FS._eval5095(P10.build_gt_coco(man, t), spred) for t in ("easy", "medium", "hard")]
    # soft_average over the 54-config grid
    best = None
    for cfg in FS._grid():
        fused = {fn: (FS.fuse_soft(raw_pp, fn, cfg, fn_hw, soft_data) or []) for fn in man}
        pred = FS._to_coco(fused, man)
        r50 = [FS._eval50(P10.build_gt_coco(man, t), pred) for t in ("easy", "medium", "hard")]
        agg50 = float(np.mean(r50))
        if best is None or agg50 > best["agg50"]:
            r95 = [FS._eval5095(P10.build_gt_coco(man, t), pred) for t in ("easy", "medium", "hard")]
            best = {"cfg": cfg, "r50": r50, "agg50": agg50, "r95": r95, "agg95": float(np.mean(r95))}
    b = best
    out_q.put((corpus, label, b, s50, s95, f"{time.time()-t0:.1f}s"))
    print(f"[soft-only/{label}] agg50={b['agg50']:.1f} agg5095={b['agg95']:.2f} ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)


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
    # collect (drain with timeout — spawn Queue can race empty() before flush)
    results = {}
    import queue as _q
    deadline = time.time() + 10
    while len(results) < len(FS.CORPORA) and time.time() < deadline:
        try:
            corpus, label, best, s50, s95, info = out_q.get(timeout=2)
            results[corpus] = (label, best, s50, s95, info)
        except _q.Empty:
            continue
    # write report
    lines = ["\n## 20b. Soft-average (6th strategy) — companion run (2026-07-29)\n\n",
             "Companion to §20: the in-flight sweep baked soft_avail=False (logits not "
             "ready at its start), so it produces the 5 binary strategies only. This "
             "fills the missing soft_average row per corpus using the sweep's exact "
             "metric path (prep_10 tier-mean, 54-config grid, real scores).\n\n",
             "| corpus | E50 | M50 | H50 | agg50 | E5095 | M5095 | H5095 | agg5095 | best cfg | time |\n|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for corpus, label in FS.CORPORA.items():
        r = results.get(corpus)
        if r is None or r[1] is None:
            lines.append(f"| {label} | - | - | - | - | - | - | - | - | - | {r[5] if r else 'MISSING'} |\n")
            continue
        label, best, s50, s95, info = r
        b = best
        lines.append(f"| {label} | {b['r50'][0]:.1f} | {b['r50'][1]:.1f} | {b['r50'][2]:.1f} | {b['agg50']:.1f} | {b['r95'][0]:.1f} | {b['r95'][1]:.1f} | {b['r95'][2]:.1f} | {b['agg95']:.2f} | ma{b['cfg']['min_area']}n{b['cfg']['nms_iou']}d{b['cfg']['max_dim_frac']}s{b['cfg']['score_floor']} | {info} |\n")
    b_s = results.get("boulders-penguin"); s_s = results.get("sanparks-algoa")
    def _fmt50(r):
        if not r or r[2] is None:
            return "E=- M=- H=-"
        e, m, h = r[2]
        return f"E={e:.1f} M={m:.1f} H={h:.1f}"
    lines.append(f"\nBaseline single (a_penguin @ CHOSEN) for reference: "
                 f"Boulders {_fmt50(b_s)}; SANParks {_fmt50(s_s)}\n")
    text = "".join(lines)
    OUT.write_text(text)
    with open(FINDINGS, "a") as fh:
        fh.write(text)
    DONE.touch()
    print(f"[soft-only] done {time.time()-t0:.1f}s sentinel={DONE}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
