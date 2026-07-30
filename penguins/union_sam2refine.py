#!/usr/bin/env python3
"""SAM 2 refinement on the prompt-union boxes, gold corpora. Self-reporting.

Builds the union of "penguin" + "a penguin" boxes at thr 0.40, sieve + NMS
dedups to one box per bird, prompts SAM2 for masks, evals against the gold COCO
GT. Appends the result to FINDINGS.md (section 13) and writes the numbers to a
sentinel file. Nothing polls this.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from pycocotools import mask as mu
from PIL import Image
from count_decompose import _rle_from_ann
from sanity_sieve import mask_passes, class_agnostic_nms

RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")
CORPORA = {"boulders-penguin": ("Boulders", str(D / "boulders-penguin" / "train" / "images")),
           "sanparks-algoa": ("SANParks", str(D / "sanparks-algoa" / "train" / "images"))}
THR = 0.40
CK = "/home/vorse/sam2/checkpoints/checkpoints/sam2.1_hiera_large.pt"
CFG = "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
FINDINGS = RUNS / "FINDINGS.md"
SENTINEL = RUNS / "_union_sam2refine_done"


def log(m):
    print(m, flush=True)
    with open(RUNS / "union_sam2refine.log", "a") as f:
        f.write(m.rstrip() + "\n")


def union_boxes(corpus, images_dir):
    """Union boxes at thr, sieve+NMS, remap onto gt_coco image ids. Returns box list + gt_path."""
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    fn_to_iid = {im["file_name"]: im["id"] for im in gt["images"]}
    per = defaultdict(list)
    for stem in ("penguin", "a_penguin"):
        src = json.loads((RUNS / "sam3_prompts_gold" / f"{corpus}_{stem}.json").read_text())
        src_fn = {i["id"]: i["file_name"] for i in src["images"]}
        for a in src["annotations"]:
            if float(a.get("score", 1.0)) < THR:
                continue
            fn = src_fn.get(a["image_id"])
            iid = fn_to_iid.get(fn)
            if iid is None:
                continue
            im = next(im for im in gt["images"] if im["id"] == iid)
            h, w = im["height"], im["width"]
            rle = _rle_from_ann(a, h, w)
            bb = mu.toBbox(rle)
            box = (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))
            per[iid].append((box, float(a.get("score", 1.0)), w, h))
    preds = []
    for iid, items in per.items():
        kept = [(b, s, w, h) for (b, s, w, h) in items if mask_passes(None, (w, h), bbox=b)]
        if not kept:
            continue
        boxes = [k[0] for k in kept]
        scores = [k[1] for k in kept]
        idx = class_agnostic_nms(boxes, scores, 0.55)
        for i in idx:
            b = boxes[i]
            preds.append({"image_id": iid, "category_id": 1,
                          "bbox": [float(b[0]), float(b[1]),
                                   float(b[2] - b[0]), float(b[3] - b[1])],
                          "score": 1.0})
    return preds, RUNS / f"sam2refine_{corpus}" / "gt_coco.json"


def main():
    log("=== union SAM2-refine start ===")
    from bench_eval import sam2_seg_from_boxes
    results = {}
    for corpus, (label, images_dir) in CORPORA.items():
        outdir = RUNS / f"union_sam2refine_{corpus}"
        outdir.mkdir(parents=True, exist_ok=True)
        boxes, gt_path = union_boxes(corpus, images_dir)
        box_path = outdir / "union_boxes.json"
        box_path.write_text(json.dumps(boxes))
        log(f"  {label}: {len(boxes)} union boxes over {images_dir}")
        m = sam2_seg_from_boxes(images_dir, str(box_path), CK, CFG,
                                str(gt_path), str(outdir / "sam2_masks.json"))
        results[label] = m
        log(f"  {label}: mAP50={m['sam2_mAP50']:.3f} mAP50-95={m['sam2_mAP50_95']:.3f}")
    out = ["\n## 13. SAM 2 refine on the union boxes\n\n",
           "Union of 'penguin' + 'a penguin' at thr 0.40, sieve + NMS, SAM 2 prompted for masks,",
           " segm COCOeval against gold.\n\n",
           "| corpus | mask mAP@50 | mask mAP@50-95 |\n|---|---|---|\n"]
    for label, m in results.items():
        out.append(f"| {label} (union boxes + SAM 2 masks) | {m['sam2_mAP50']:.3f} | {m['sam2_mAP50_95']:.3f} |\n")
    out.append("\nCompare against section 10 (single-prompt SAM 2 refine) and the supervised gold in P5.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))
    SENTINEL.write_text("done\n")
    log("=== union SAM2-refine complete ===")


if __name__ == "__main__":
    main()
