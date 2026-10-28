#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Acceptance test for the soft-logit probability maps that feed soft-average fusion.

WHY

`prep_11b_soft_logit.py` saves one sigmoid probability map per detection. It used
to select them with `probs[:n_keep]`, a prefix of SAM 3's query axis, while
`post_process_instance_segmentation` filters with `scores > threshold` and keeps
a scattered subset of that axis. Every map was therefore filed under the wrong
detection, which collapsed soft-average fusion to 1.3 tier-mean mAP@50 on
Boulders and 1.1 on SANParks, including 0.1 and 0.0 on the easy tier where every
other strategy scores about 98.

The check that catches it: a detection's own probability map, thresholded at 0.5,
should reproduce that detection's own binary mask. Before the fix this scored
mean IoU 0.009 with 95 percent of pairs below 0.1, two non-empty masks that do
not touch. After the fix it should sit near 0.9, short of 1.0 only because the
map is stored at native 288x288 and the binary mask at full resolution.

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
OUT_TXT = RUNS / "obj2_counting" / "soft_logit_verify.txt"
DONE = RUNS / "obj2_counting" / "_soft_logit_verify_done"

CORPORA = {"boulders": "Boulders", "sanparks": "SANParks"}
PROMPTS = ["a_penguin", "penguin", "african_penguin"]

MAX_IMGS = 40          # images sampled per corpus/prompt
MAX_DETS_PER_IMG = 8
PASS_IOU = 0.85        # mean self-IoU below this means the maps are still misfiled


def _tag(prompt):
    return "" if prompt == "a_penguin" else "_" + prompt


def _paths(sub, prompt):
    t = _tag(prompt)
    return (R1 / sub / f"{sub}_sam3_raw{t}.json",
            R1 / sub / f"{sub}_sam3_soft{t}_v2.json",
            R1 / sub / f"{sub}_soft{t}_v2_npys")


def check(sub, prompt):
    raw_p, soft_p, npy_dir = _paths(sub, prompt)
    if not (raw_p.exists() and soft_p.exists() and npy_dir.exists()):
        return {"prompt": prompt, "status": "MISSING", "detail": f"{raw_p.name} / {soft_p.name}"}
    raw = json.loads(raw_p.read_text())
    soft = json.loads(soft_p.read_text())

    # 1. the two sets must describe the same detections, in the same order
    count_mismatch = [fn for fn in raw if len(raw[fn]) != len(soft.get(fn, []))]
    ident = differ = 0
    for fn in raw:
        if len(raw[fn]) == len(soft.get(fn, [])):
            for a, s in zip(raw[fn], soft[fn]):
                if abs(a["score"] - s["score"]) < 1e-9 and abs(a["box"][0] - s["box"][0]) < 1e-6:
                    ident += 1
                else:
                    differ += 1

    # 2. each map must reconstruct its own detection's mask
    ious = []
    for fn in sorted(raw)[:MAX_IMGS]:
        if len(raw[fn]) != len(soft.get(fn, [])):
            continue
        for k in range(min(MAX_DETS_PER_IMG, len(raw[fn]))):
            npy_path = npy_dir / soft[fn][k]["prob_npy"]
            if not npy_path.exists():
                continue
            pr = np.load(npy_path).astype(np.float32)
            m_bin = mu.decode(raw[fn][k]["rle"]).astype(np.uint8)
            m_bin = cv2.resize(m_bin, (pr.shape[1], pr.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            m_soft = pr >= 0.5
            union = (m_bin | m_soft).sum()
            ious.append(float((m_bin & m_soft).sum() / union) if union else 0.0)

    a = np.array(ious) if ious else np.zeros(1)
    ok = (not count_mismatch) and differ == 0 and float(a.mean()) >= PASS_IOU
    return {"prompt": prompt, "status": "PASS" if ok else "FAIL",
            "n_imgs": len(raw), "count_mismatch": len(count_mismatch),
            "identical_dets": ident, "differing_dets": differ,
            "n_pairs": len(ious), "mean_iou": round(float(a.mean()), 4),
            "median_iou": round(float(np.median(a)), 4),
            "frac_below_0.1": round(float((a < 0.1).mean()), 3)}


def main():
    t0 = time.time()
    rows, all_ok = [], True
    for sub, label in CORPORA.items():
        for prompt in PROMPTS:
            r = check(sub, prompt)
            r["corpus"] = label
            rows.append(r)
            all_ok &= r["status"] == "PASS"

    lines = ["\n## 20d. Soft-logit map alignment, acceptance test (2026-07-30)\n\n",
             "Each detection's own probability map is thresholded at 0.5 and scored "
             "against that detection's own binary mask. The prefix-slice defect in "
             "`prep_11b_soft_logit.py` gave mean IoU 0.009 here; a correct query "
             f"select should give about 0.9. Pass bar {PASS_IOU}.\n\n",
             "| corpus | prompt | imgs | count mismatch | dets differing | pairs | mean IoU | median | frac<0.1 | verdict |\n",
             "|---|---|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        if r["status"] == "MISSING":
            lines.append(f"| {r['corpus']} | {r['prompt']} | - | - | - | - | - | - | - | MISSING |\n")
            continue
        lines.append(f"| {r['corpus']} | {r['prompt']} | {r['n_imgs']} | {r['count_mismatch']} | "
                     f"{r['differing_dets']} | {r['n_pairs']} | {r['mean_iou']} | {r['median_iou']} | "
                     f"{r['frac_below_0.1']} | {r['status']} |\n")
    lines.append(f"\nOverall: {'PASS' if all_ok else 'FAIL'}. "
                 "A FAIL means soft-average fusion must not be re-run or reported.\n")

    text = "".join(lines)
    OUT_TXT.write_text(text)
    DONE.write_text("PASS" if all_ok else "FAIL")
    print(text, file=sys.stderr)
    print(f"[soft-verify] {time.time()-t0:.1f}s -> {OUT_TXT}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
