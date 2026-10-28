#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Give every supervised baseline SAM 2 masks on the two v2-gold OOD corpora.

WHY
---
tab:percorpus scores the supervised comparators on masks that come either from
their own head or, for the RT-DETR pair, from SAM 3 prompted with their boxes.
FINDINGS section 22 showed SAM 2 is the better refiner from a detector's boxes
on the in-distribution split, so the out-of-distribution comparators are running
with the weaker option while SAM 3 runs with its own masks. This script gives
each of them SAM 2 and re-scores, so the comparison is like for like.

No retraining and no re-detection. The boxes already sit in the per-model
<ckpt>_pred.json written by prep_13_supervised_reeval.py, so this is a SAM 2
forward pass over saved boxes.

CONTROL
-------
For every model the unchanged predictions are re-scored first. If the aggregate
does not reproduce the existing <ckpt>_reeval.json to 1e-6 the eval path or the
tier manifest has drifted, and that model is reported as a control failure
rather than a result.

OUTPUT
------
<corpus>/<ckpt>_sam2_pred.json and <ckpt>_sam2_reeval.json under the out dir,
a consolidated ood_sam2_refine_results.json, a FINDINGS section, and the
sentinel _ood_sam2_refine_done. Nothing polls it.
"""
from __future__ import annotations

import multiprocessing as mp

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, os.path.expanduser("~/bench_any/datasets/prep"))

import prep_13_supervised_reeval as P13  # noqa: E402

RUNS = Path(os.path.expanduser("~/bench_any/runs/p5_r1_v2"))
FINDINGS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting/FINDINGS.md"))
OUT = Path(os.path.expanduser("~/bench_any/runs/ood_sam2_refine"))
SENTINEL = OUT / "_ood_sam2_refine_done"
LOG = OUT / "ood_sam2_refine.log"

CKPTS = ["rf-detr-seg-xlarge", "rf-detr-seg-large", "rtdetr-4", "rtdetr-3",
         "eomt-large", "yolo11x-seg", "yolo11x-seg-native"]
SAM2_CK = "/home/user/sam2/checkpoints/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "/home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def eval_preds(tag, corpus, preds, manifest, out_dir):
    """prep_13's own tier eval, unchanged."""
    return P13.run_eval_for_ckpt(tag, corpus, preds, manifest, out_dir)


def sam2_masks_for(preds, manifest, images_dir, work):
    """Prompt SAM 2 with every box in `preds`; return the same dict with new rles.

    Boxes carry no image id of their own, so a throwaway COCO shell built from
    the tier manifest supplies the filename-to-id mapping both ways.
    """
    from bench_eval import sam2_seg_from_boxes

    shell = P13.build_gt_coco(manifest, "easy")
    fn_to_id = {im["file_name"]: im["id"] for im in shell["images"]}
    id_to_fn = {v: k for k, v in fn_to_id.items()}

    boxes, order = [], []
    for fname, plist in preds.items():
        iid = fn_to_id.get(fname)
        if iid is None:
            continue
        for p in plist:
            x0, y0, x1, y1 = p["box"]
            boxes.append({"image_id": iid, "category_id": 1,
                          "bbox": [float(x0), float(y0),
                                   float(x1 - x0), float(y1 - y0)],
                          "score": float(p["score"])})
            order.append((fname, float(p["score"])))
    work.mkdir(parents=True, exist_ok=True)
    box_path = work / "boxes.json"
    gt_path = work / "gt_shell.json"
    out_path = work / "sam2_masks.json"
    box_path.write_text(json.dumps(boxes))
    gt_path.write_text(json.dumps(shell))

    t0 = time.time()
    sam2_seg_from_boxes(images_dir, str(box_path), SAM2_CK, SAM2_CFG,
                        str(gt_path), str(out_path))
    log(f"    SAM 2 over {len(boxes)} boxes in {time.time() - t0:.0f}s")

    if not out_path.is_file():
        return None
    refined = {}
    for m in json.loads(out_path.read_text()):
        fname = id_to_fn.get(m["image_id"])
        if fname is None:
            continue
        seg = m["segmentation"]
        if isinstance(seg.get("counts"), bytes):
            seg = dict(seg, counts=seg["counts"].decode("utf-8"))
        refined.setdefault(fname, []).append(
            {"rle": seg, "score": float(m.get("score", 1.0)), "box": None})
    return refined


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== OOD SAM 2 refine start ===")
    results = {}

    for corpus, cfg in P13.CORPORA.items():
        manifest = json.load(open(cfg["tiers"]))["manifest"]
        images_dir = cfg["images"]
        corpus_dir = RUNS / corpus
        out_dir = OUT / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"\n--- {corpus}: {len(manifest)} images ---")

        for tag in CKPTS:
            pred_path = corpus_dir / f"{tag}_pred.json"
            base_path = corpus_dir / f"{tag}_reeval.json"
            if not (pred_path.is_file() and base_path.is_file()):
                log(f"  [{tag}] SKIP, missing {pred_path.name} or {base_path.name}")
                continue
            preds = json.loads(pred_path.read_text())
            published = json.loads(base_path.read_text())

            ctl = eval_preds(f"{tag}_control", corpus, preds, manifest, out_dir)
            ok = (ctl["agg"] is not None and published["agg"] is not None
                  and abs(ctl["agg"] - published["agg"]) < 1e-6)
            log(f"  [{tag}] control agg {ctl['agg']} vs published {published['agg']} "
                f"-> {'MATCH' if ok else 'MISMATCH'}")

            refined = sam2_masks_for(preds, manifest, images_dir, out_dir / tag)
            if refined is None:
                log(f"  [{tag}] SAM 2 produced nothing; skipping")
                continue
            (out_dir / f"{tag}_sam2_pred.json").write_text(json.dumps(refined))
            new = eval_preds(f"{tag}_sam2", corpus, refined, manifest, out_dir)

            results.setdefault(corpus, {})[tag] = {
                "control_reproduced": ok,
                "published_agg": published["agg"],
                "control_agg": ctl["agg"],
                "sam2_agg": new["agg"],
                "published_tiers": {t: published.get(t) for t in ("easy", "medium", "hard")},
                "sam2_tiers": {t: new.get(t) for t in ("easy", "medium", "hard")},
                "published_recall": published.get("recall"),
                "sam2_recall": new.get("recall"),
            }
            (OUT / "ood_sam2_refine_results.json").write_text(json.dumps(results, indent=2))

    out = ["\n## 23. SAM 2 masks for the supervised baselines on the v2 gold corpora "
           f"({time.strftime('%Y-%m-%d')})\n\n",
           "Same boxes as `tab:percorpus`, taken straight from the per-model\n"
           "`<ckpt>_pred.json` under `~/bench_any/runs/p5_r1_v2/`. Only the mask source\n"
           "changes. Eval is prep_13's own per-tier COCOeval, aggregate = mean of the\n"
           "three tiers. Each model's unchanged predictions were re-scored first as a\n"
           "control against its published reeval.\n\n"]
    for corpus, per in results.items():
        out.append(f"### {corpus}\n\n| model | published agg | SAM 2 agg | delta | control |\n"
                   "|---|---|---|---|---|\n")
        for tag, r in per.items():
            pa, sa = r["published_agg"], r["sam2_agg"]
            d = f"{sa - pa:+.1f}" if (pa is not None and sa is not None) else "-"
            out.append(f"| {tag} | {pa if pa is None else f'{pa:.1f}'} | "
                       f"{sa if sa is None else f'{sa:.1f}'} | {d} | "
                       f"{'ok' if r['control_reproduced'] else 'FAILED'} |\n")
        out.append("\n")
    out.append("A control marked FAILED means the unchanged predictions no longer "
               "reproduce the published aggregate, so that row's delta is not usable.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))

    SENTINEL.write_text("done\n")
    log("=== OOD SAM 2 refine complete ===")


if __name__ == "__main__":
    main()
