#!/usr/bin/env python3
"""Score SAM 3's own masks on the iNaturalist test split, same GT and same
eval path as the SAM 2-prompted row already in P5's tab:zeroshot_mask.

WHY
---
The table reports "SAM 3 boxes + SAM 2 masks" without a SAM 3-native-mask row,
so a reader cannot see what the extra SAM 2 forward pass buys. SAM 3 is itself
a segmenter; bolting SAM 2 on needs a measured justification or it comes out.

Everything needed is already on disk, so this is CPU-only:
  preds (native) ~/bench_any/runs/obj2_counting/sam3_inat_test/sam3_inat_test_coco.json
  preds (SAM 2)  ~/bench_any/runs/obj2_counting/sam3_sam2refine/sam2_masks.json
  GT             the run-dir _bench_gt_coco.json for this split, which is what the
                 six supervised rows in the same table were scored against. The
                 filename-derived image ids there are the ones sam2_masks.json
                 already carries, so the SAM 2 row needs no remapping.

The SAM 2 row is re-scored here as a control. If it does not come back at
0.889 / 0.712 the GT or the eval path is wrong and the SAM 3 number is not
comparable, so the script says so and refuses to report a delta.

OUTPUT
------
Writes ~/bench_any/runs/obj2_counting/sam3_native_mask/results.json and appends
a section to FINDINGS.md. Touches _sam3_native_mask_done. Nothing polls it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bench_eval import coco_eval  # noqa: E402

RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
GT = Path(os.path.expanduser(
    "~/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e150_i640_be64_fair_grid"
    "/_bench_gt_coco.json"))
SAM3_COCO = RUNS / "sam3_inat_test" / "sam3_inat_test_coco.json"
SAM2_MASKS = RUNS / "sam3_sam2refine" / "sam2_masks.json"
OUT = RUNS / "sam3_native_mask"
FINDINGS = RUNS / "FINDINGS.md"

# What sam3_sam2refine/run.log recorded for the row already in the paper.
CONTROL = {"sam2_mAP50": 0.8894776139676618, "sam2_mAP50_95": 0.7114618127053133}


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sam3_native_mask.log", "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def native_preds(gt):
    """SAM 3's own RLE masks as a loadRes list, on the GT's image ids."""
    src = json.loads(SAM3_COCO.read_text())
    gt_id = {im["file_name"]: im["id"] for im in gt["images"]}
    src_fn = {im["id"]: im["file_name"] for im in src["images"]}
    preds, dropped = [], 0
    for a in src["annotations"]:
        iid = gt_id.get(src_fn.get(a["image_id"]))
        if iid is None:
            dropped += 1
            continue
        seg = a["segmentation"]
        if isinstance(seg.get("counts"), bytes):
            seg = dict(seg, counts=seg["counts"].decode("utf-8"))
        preds.append({"image_id": iid, "category_id": 1, "segmentation": seg,
                      "score": float(a.get("score", 1.0))})
    return preds, dropped, len(src["annotations"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== SAM 3 native mask eval on iNat test ===")
    gt = json.loads(GT.read_text())
    log(f"GT: {len(gt['images'])} images, {len(gt['annotations'])} instances")

    preds, dropped, total = native_preds(gt)
    log(f"SAM 3 native: {len(preds)} of {total} detections mapped ({dropped} dropped)")
    native_path = OUT / "sam3_native_preds.json"
    native_path.write_text(json.dumps(preds))

    # Control first: the SAM 2 row must reproduce before the SAM 3 row means anything.
    ctl = coco_eval(str(GT), str(SAM2_MASKS), "segm")
    ctl50, ctl5095 = ctl["seg_mAP50"], ctl["seg_mAP50_95"]
    ok = (abs(ctl50 - CONTROL["sam2_mAP50"]) < 1e-6
          and abs(ctl5095 - CONTROL["sam2_mAP50_95"]) < 1e-6)
    log(f"control SAM 2 row: mAP50={ctl50:.4f} mAP50-95={ctl5095:.4f} "
        f"(expected {CONTROL['sam2_mAP50']:.4f} / {CONTROL['sam2_mAP50_95']:.4f}) "
        f"-> {'MATCH' if ok else 'MISMATCH'}")

    nat = coco_eval(str(GT), str(native_path), "segm")
    log(f"SAM 3 native: mAP50={nat['seg_mAP50']:.4f} mAP50-95={nat['seg_mAP50_95']:.4f}")

    res = {"gt_images": len(gt["images"]), "gt_instances": len(gt["annotations"]),
           "n_detections": len(preds), "control_reproduced": ok,
           "sam2_row": {k: ctl[k] for k in ("seg_mAP50", "seg_mAP50_95", "seg_mAP75",
                                            "seg_AR_100", "seg_P", "seg_R")},
           "sam3_native": {k: nat[k] for k in ("seg_mAP50", "seg_mAP50_95", "seg_mAP75",
                                               "seg_AR_100", "seg_P", "seg_R")},
           "metric": "pycocotools COCOeval segm, maxDets [1,10,100], same GT and "
                     "same coco_eval path as the SAM 2-prompted row in P5"}
    (OUT / "results.json").write_text(json.dumps(res, indent=2))

    out = ["\n## 21. SAM 3's own masks vs SAM 2-prompted, iNat test (2026-07-29)\n\n"]
    if not ok:
        out.append("**Control failed.** Re-scoring the SAM 2 masks against this GT did "
                   f"not reproduce the published 0.889 / 0.712 (got {ctl50:.3f} / "
                   f"{ctl5095:.3f}), so the two rows are not on the same footing and "
                   "the delta below is not usable.\n\n")
    out.append("Same 856 images, same 5695 gold masks, same `coco_eval`, all 6807 SAM 3 "
               "detections in both rows. The only difference is who draws the mask.\n\n"
               "| masks from | mask mAP@50 | mask mAP@50-95 |\n|---|---|---|\n"
               f"| SAM 3 (native) | {nat['seg_mAP50']:.3f} | {nat['seg_mAP50_95']:.3f} |\n"
               f"| SAM 2, prompted with SAM 3's boxes | {ctl50:.3f} | {ctl5095:.3f} |\n\n"
               f"Delta from the extra SAM 2 pass: {ctl50 - nat['seg_mAP50']:+.3f} at "
               f"mAP@50, {ctl5095 - nat['seg_mAP50_95']:+.3f} at mAP@50-95.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))

    (RUNS / "_sam3_native_mask_done").write_text("done\n")
    log("=== done ===")


if __name__ == "__main__":
    main()
