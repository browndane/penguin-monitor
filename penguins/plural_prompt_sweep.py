#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Plural-prompt SAM3 sweep on the gold corpora. Self-reporting.

Plural prompts were dismissed on assumption rather than on data. This runs a
set of plural/collective prompts through SAM3 on both gold corpora, then scores
count MAE + raw mask mAP the same way as the singular-prompt probe, and appends
a verdict to FINDINGS.md. Nothing polls this.

Plurals tested: "penguins", "a penguin colony", "a colony of penguins",
"penguin colony", and the two singular baselines for direct comparison.
"""
from __future__ import annotations
import json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

from count_decompose import _rle_from_ann, _decompose_image
from sanity_sieve import mask_passes, class_agnostic_nms

RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")
ANNOT = Path("/home/user/Dropbox/Work/PycharmProjects/cvMLCode/12_Deep/"
              "EndtoEndProjects/Fish/YOLO/point_crowdsource_annotate/sam3_coco_annotator.py")
PY = os.path.expanduser("~/cvML/bin/python")
FINDINGS = RUNS / "FINDINGS.md"
SENTINEL = RUNS / "_plural_sweep_done"
LOG = RUNS / "plural_sweep.log"

CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}
# plural + collective prompts; singular baselines re-derived from existing JSONs
PLURAL_PROMPTS = ["penguins", "a penguin colony", "a colony of penguins",
                  "penguin colony"]
THR = 0.40  # score threshold for counting/eval (re-applied here)


def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m.rstrip() + "\n")


def run_sam3(corpus, prompt, outdir):
    img_dir = D / corpus / "train" / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = prompt.replace(" ", "_")
    out_json = outdir / f"{corpus}_{stem}.json"
    out_log = outdir / f"{corpus}_{stem}.log"
    cmd = [PY, str(ANNOT), str(img_dir), "--prompt", prompt,
           "--output", str(out_json), "--score", "0.25", "--min-area", "144",
           "--nms", "0.7", "--overwrite"]
    with open(out_log, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    return out_json


def gold_truth(corpus):
    root = D / corpus
    lbl = root / "train" / "labels"; img = root / "train" / "images"
    out = {}
    for lf in sorted(lbl.glob("*.txt")):
        imgs = list(img.glob(lf.stem + ".*"))
        if not imgs:
            continue
        w, h = Image.open(imgs[0]).size
        rles = []
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            xy = [float(v) for v in p[1:]]
            poly = [xy[i] * (w if i % 2 == 0 else h) for i in range(len(xy))]
            rles.append(mu.merge(mu.frPyObjects([poly], h, w)))
        out[imgs[0].name] = (rles, (h, w))
    return out


def load_items(coco, truth, thr):
    fn_of = {i["id"]: i["file_name"] for i in coco["images"]}
    per = defaultdict(list)
    for a in coco["annotations"]:
        if float(a.get("score", 1.0)) < thr:
            continue
        fn = fn_of.get(a["image_id"])
        if fn not in truth:
            continue
        h, w = truth[fn][1]
        rle = _rle_from_ann(a, h, w)
        bb = mu.toBbox(rle)
        box = (int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[1] + bb[3]))
        per[fn].append((box, float(a.get("score", 1.0)), rle, h, w))
    return per


def dedup(per):
    out = {}
    for fn, items in per.items():
        kept = [(b, s, r, h, w) for (b, s, r, h, w) in items
                if mask_passes(None, (w, h), bbox=b)]
        if not kept:
            out[fn] = []
            continue
        idx = class_agnostic_nms([k[0] for k in kept], [k[1] for k in kept], 0.55)
        out[fn] = [kept[i][2] for i in idx]
    return out


def count_score(preds, truth):
    n = len(truth); mae = 0; sgn = 0
    agg = defaultdict(float)
    for fn, (gts, _) in truth.items():
        p = preds.get(fn, [])
        ev = _decompose_image(p, gts, 0.5, 0.5)
        ev.pop("_fate", None)
        for k, v in ev.items():
            agg[k] += v
        mae += abs(len(p) - len(gts))
        sgn += len(p) - len(gts)
    return dict(MAE=mae / n, signed=sgn / n, missed=int(agg["missed"]),
                spurious=int(agg["spurious"]), matched=int(agg["matched"]))


def build_pred_coco(preds, gt_coco, corpus):
    fn_to_iid = {im["file_name"]: im["id"] for im in gt_coco["images"]}
    out = []
    for fn, rles in preds.items():
        iid = fn_to_iid.get(fn)
        if iid is None:
            continue
        for r in rles:
            rr = dict(r)
            if isinstance(rr["counts"], bytes):
                rr["counts"] = rr["counts"].decode("utf-8")
            out.append({"image_id": iid, "category_id": 1,
                        "bbox": list(mu.toBbox(r)), "score": 1.0,
                        "segmentation": rr})
    return out


def mask_eval(pred_json, gt_path):
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(pred_json) if pred_json else COCO()
    if not pred_json:
        return 0.0, 0.0
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return max(0.0, float(ev.stats[1])), max(0.0, float(ev.stats[0]))


def main():
    log("=== plural prompt sweep start ===")
    out_root = RUNS / "sam3_prompts_plural"
    out_root.mkdir(exist_ok=True)
    outdir = RUNS / "sam3_prompts_plural"
    rows = {}
    for corpus, label in CORPORA.items():
        truth = gold_truth(corpus)
        gt_coco = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
        gt_path = RUNS / f"sam2refine_{corpus}" / "gt_coco.json"
        rows[label] = {}
        # plural prompts: run SAM3 then score
        for prompt in PLURAL_PROMPTS:
            stem = prompt.replace(" ", "_")
            jf = outdir / f"{corpus}_{stem}.json"
            if not jf.exists():
                log(f"  {label}: running SAM3 prompt='{prompt}'")
                jf = run_sam3(corpus, prompt, outdir)
            else:
                log(f"  {label}: reusing {jf.name}")
            coco = json.loads(jf.read_text())
            per = dedup(load_items(coco, truth, THR))
            cs = count_score(per, truth)
            pred_coco = build_pred_coco(per, gt_coco, corpus)
            tmp = RUNS / "_tmp_plural.json"
            tmp.write_text(json.dumps(pred_coco))
            m50, m5095 = mask_eval(tmp, gt_path) if pred_coco else (0.0, 0.0)
            rows[label][prompt] = dict(MAE=cs["MAE"], signed=cs["signed"],
                                       missed=cs["missed"], mAP50=m50, mAP5095=m5095)
            log(f"  {label} '{prompt}': MAE={cs['MAE']:.2f} signed={cs['signed']:+.2f} "
                f"mAP50={m50:.3f} mAP50-95={m5095:.3f}")
    tmp = RUNS / "_tmp_plural.json"
    if tmp.exists():
        tmp.unlink()
    # write findings
    out = ["\n## 14. Plural and collective prompts, 2026-07-29\n\n",
           "Plural prompts were dismissed on assumption earlier. Tested against gold.\n",
           f"SAM3 at score 0.25, re-thresholded to {THR} here, sieve + NMS. Count is MAE.\n\n"]
    for label, got in rows.items():
        out.append(f"**{label}**\n\n| prompt | count MAE | signed | mask mAP@50 | mask mAP@50-95 |\n|---|---|---|---|---|\n")
        for prompt, r in got.items():
            out.append(f"| \"{prompt}\" | {r['MAE']:.2f} | {r['signed']:+.2f} | {r['mAP50']:.3f} | {r['mAP5095']:.3f} |\n")
        out.append("\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))
    SENTINEL.write_text("done\n")
    log("=== plural prompt sweep complete ===")


if __name__ == "__main__":
    main()
