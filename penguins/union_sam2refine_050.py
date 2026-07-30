#!/usr/bin/env python3
"""Union+SAM2 control pass at thr 0.50 (matches §10 single-prompt threshold).

Self-reporting: runs SAM2 on the union boxes at 0.50, evals against gold,
appends section 13b to FINDINGS.md, touches _union_sam2_050_done. Reuses
union_sam2refine.py's logic with THR overridden.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# import the module so we can reuse union_boxes + main machinery
import union_sam2refine as u
u.THR = 0.50  # override the 0.40 used in §13
u.FINDINGS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting/FINDINGS.md"))
u.SENTINEL = Path(os.path.expanduser("~/bench_any/runs/obj2_counting/_union_sam2_050_done"))

def main():
    u.log("=== union SAM2-refine control @ thr 0.50 start ===")
    from bench_eval import sam2_seg_from_boxes
    results = {}
    for corpus, (label, images_dir) in u.CORPORA.items():
        outdir = u.RUNS / f"union_sam2refine_050_{corpus}"
        outdir.mkdir(parents=True, exist_ok=True)
        boxes, gt_path = u.union_boxes(corpus, images_dir)
        box_path = outdir / "union_boxes.json"
        box_path.write_text(json.dumps(boxes))
        u.log(f"  {label}: {len(boxes)} union boxes @0.50 over {images_dir}")
        m = sam2_seg_from_boxes(images_dir, str(box_path), u.CK, u.CFG,
                                str(gt_path), str(outdir / "sam2_masks.json"))
        results[label] = m
        u.log(f"  {label}: mAP50={m['sam2_mAP50']:.3f} mAP50-95={m['sam2_mAP50_95']:.3f}")
    out = ["\n## 13b. Union+SAM2 control at thr 0.50 (2026-07-29)\n\n",
           "Union of 'penguin' + 'a penguin' at thr 0.50 (matching §10's single-prompt threshold), ",
           "sieve + NMS, SAM 2 prompted for masks, segm COCOeval against gold. Controls the §13 ",
           "finding that SAM2 hurt union masks at thr 0.40.\n\n",
           "| corpus | mask mAP@50 | mask mAP@50-95 |\n|---|---|---|\n"]
    for label, m in results.items():
        out.append(f"| {label} (union+SAM2 @0.50) | {m['sam2_mAP50']:.3f} | {m['sam2_mAP50_95']:.3f} |\n")
    out.append("\nCompare: §13 union+SAM2 @0.40 (B 0.530/0.475, S 0.656/0.367); ")
    out.append("§10 single+SAM2 @0.50 (B 0.550/0.518, S 0.814/0.477); ")
    out.append("§12 union raw SAM3 @0.40 (B 0.543/0.507, S 0.690/0.433).\n")
    with open(u.FINDINGS, "a") as f:
        f.write("".join(out))
    u.SENTINEL.write_text("done\n")
    u.log("=== union SAM2-refine control @ 0.50 complete ===")

if __name__ == "__main__":
    main()
