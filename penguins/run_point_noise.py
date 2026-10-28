#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Run point_noise_study on boulders gold, then write FINDINGS section 16 + sentinel.

Self-reporting: runs the harness, reads its CSV, appends a summary table to
FINDINGS.md, touches _point_noise_done. Nothing polls this.
"""
from __future__ import annotations
import os, sys, csv, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = os.path.expanduser("~/cvML/bin/python")
RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
OUT_CSV = RUNS / "point_noise" / "results.csv"
FINDINGS = RUNS / "FINDINGS.md"
SENTINEL = RUNS / "_point_noise_done"
LOG = RUNS / "point_noise.log"


def main():
    LOG.write_text("")
    def log(m):
        print(m, flush=True)
        with open(LOG, "a") as f:
            f.write(m.rstrip() + "\n")
    log("=== point noise start ===")
    cmd = [PY, str(HERE / "point_noise_study.py"),
           "--corpus", "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train",
           "--out", str(OUT_CSV),
          ]
    log(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(r.stdout[-2000:])
    if r.returncode != 0:
        log(f"FAILED rc={r.returncode}")
        log(r.stderr[-2000:])
        SENTINEL.write_text("failed\n")
        return
    # read CSV, build summary
    rows = list(csv.DictReader(open(OUT_CSV)))
    log(f"rows: {len(rows)}")
    out = ["\n## 16. Point-noise harness against generic gold GT (2026-07-29)\n\n",
           "Synthetic annotators (jittered gold-instance centres) prompt SAM 2 on the\n",
           "boulders gold corpus (first 30 images). The merge-and-agreement step recovers\n",
           "masks from crowd clicks; this reports how much sloppiness it absorbs and where\n",
           "it breaks. min_annotators = 4 by default.\n\n",
           "| noise | extra_click_rate | min_ann | n_imgs | n_gt | recall | precision | mean IoU | sieve reject |\n",
           "|---|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        out.append(f"| {r['noise_level']} | {r['extra_click_rate']} | {r['min_annotators']} | "
                   f"{r['n_images']} | {r['n_gt_instances']} | {r.get('recall','n/a')} | "
                   f"{r.get('precision','n/a')} | {r.get('mean_iou','n/a')} | "
                   f"{r.get('sieve_reject_rate','n/a')} |\n")
    out.append("\nDecision-5 generic/synthetic GT stand-in; the real Penguin-Watch path and\n")
    out.append("schema arrive later from the user. Full sweep (all 123 images, all noise\n")
    out.append("levels) pending; this is the readiness check.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))
    SENTINEL.write_text("done\n")
    log("=== point noise complete ===")


if __name__ == "__main__":
    main()
