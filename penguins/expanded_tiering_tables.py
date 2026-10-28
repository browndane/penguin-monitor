#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Generate the four-tiering expanded headline tables for P5.

For each gold corpus (boulders, sanparks) score every method under all four
tierings side by side: current terciles, variant A, variant B, variant C.
Emit one LaTeX table per corpus plus the limitation sentence.

Reads tier sidecars from p5_r1_v2/<corpus>/ and p5_r1_gold/, cached preds from
p5_r1_v2/<corpus>/. CPU only, reuses prep_10's iscrowd-ignore per-tier eval.
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, os.path.expanduser("~/bench_any/datasets/prep"))
from prep_10_filebased import build_gt_coco, build_pred_coco, eval_tier

TIERS = ["easy", "medium", "hard"]
V2 = Path(os.path.expanduser("~/bench_any/runs/p5_r1_v2"))
GOLD = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold"))
OUT = Path(os.path.expanduser("~/bench_any/runs/p5_r1_v2/expanded_tables"))
TEX = Path(os.path.expanduser(
    "~/paper"))

# Method display order matches the published tab:stratified_boulders.
METHODS = [
    ("rf-detr-seg-xlarge", "RF-DETR-seg-xlarge (supervised)"),
    ("rf-detr-seg-large", "RF-DETR-seg-large (supervised)"),
    ("eomt-large", "EoMT-large (supervised)"),
    ("sam3_raw", "SAM~3 raw (zero-shot)"),
    ("sam3_sieve", "SAM~3 + sieve (zero-shot)"),
    ("rtdetr-4", "RT-DETR-v4\\,$^{\\dagger}$ (supervised box, SAM~3 box-prompt)"),
    ("rtdetr-3", "RT-DETR-v3\\,$^{\\dagger}$ (supervised box, SAM~3 box-prompt)"),
    ("yolo11x-seg", "YOLOv11x-seg (supervised)"),
]

# Five tierings: the published prep_09 score-based tiers, the per-corpus
# rank terciles (prep_22, the ones that invert on SANParks), and three
# repair variants cut from the terciles.
TIERINGS = [
    ("published", "published (prep\\_09)"),
    ("tercile", "terciles"),
    ("variantA", "A"),
    ("variantB", "B"),
    ("variantC", "C"),
]

CORPORA = {
    "boulders": {
        "sidecar_dir": V2 / "boulders",
        "pred_dir": V2 / "boulders",
        "caption_n": "123 images, tiers Easy 204 / Medium 535 / Hard 30",
        "label": "tab:stratified_boulders",
        "title": "Boulders",
    },
    "sanparks": {
        "sidecar_dir": V2 / "sanparks",
        "pred_dir": V2 / "sanparks",
        "caption_n": "51 images, tiers Easy 1 / Medium 801 / Hard 52",
        "label": "tab:stratified_sanparks",
        "title": "SANParks",
    },
}


def load_tiers(corpus, tier_tag):
    sidecar_dir = CORPORA[corpus]["sidecar_dir"]
    if tier_tag == "published":
        path = sidecar_dir / f"{corpus}_tiers.json"
    elif tier_tag == "tercile":
        path = sidecar_dir / f"{corpus}_tiers_tercile.json"
    else:
        path = sidecar_dir / f"{corpus}_tiers_{tier_tag}.json"
    return json.load(open(path))["manifest"]


def load_preds(corpus, method):
    pd = CORPORA[corpus]["pred_dir"]
    # Method-specific file naming on disk.
    if method == "sam3_raw":
        path = pd / f"{corpus}_sam3_raw.json"
    elif method == "sam3_sieve":
        path = pd / f"{corpus}_sam3_pred.json"
    else:
        path = pd / f"{method}_pred.json"
    d = json.load(open(path))
    return d["preds"] if "preds" in d else d


def score_cell(mani, preds_per_img):
    pl = build_pred_coco(preds_per_img, mani)
    res, ns = {}, {}
    for t in TIERS:
        gt = build_gt_coco(mani, t)
        ns[t] = sum(1 for a in gt["annotations"] if a["iscrowd"] == 0)
        with contextlib.redirect_stdout(io.StringIO()):
            res[t] = eval_tier(gt, pl)
    vals = [res[t] for t in TIERS]
    res["agg"] = float(np.mean(vals))
    res["agg_iw"] = float(sum(res[t] * ns[t] for t in TIERS) / sum(ns.values()))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_scores = {}
    for corpus in CORPORA:
        all_scores[corpus] = {}
        for tier_tag, _ in TIERINGS:
            mani = load_tiers(corpus, tier_tag)
            all_scores[corpus][tier_tag] = {}
            for method, _ in METHODS:
                try:
                    preds = load_preds(corpus, method)
                except FileNotFoundError:
                    print(f"skip {corpus}/{method}/{tier_tag}: no preds", flush=True)
                    all_scores[corpus][tier_tag][method] = None
                    continue
                r = score_cell(mani, preds)
                all_scores[corpus][tier_tag][method] = r
                print(f"{corpus:9s} {method:22s} {tier_tag:8s} "
                      f"{r['easy']:.1f}/{r['medium']:.1f}/{r['hard']:.1f} "
                      f"agg {r['agg']:.1f}", flush=True)
    json.dump(all_scores, open(OUT / "expanded_scores.json", "w"), indent=2)

    # Emit one LaTeX table per corpus.
    for corpus, meta in CORPORA.items():
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Mask mAP@0.5 (\\%) on the {meta['title']} v2 gold-standard subset "
            f"({meta['caption_n']}), by difficulty tier under five tiering rules. The five "
            f"columns per method are the published prep\\_09 score-based tiers, the per-corpus "
            f"rank terciles (prep\\_22, which invert on SANParks), and three repair variants "
            f"(A rank-normalises the size term and the pooled remainder, B the size term "
            f"only, C keeps the rank-normalised size term and discards the rest). "
            f"The zero-shot model uses no training masks. RT-DETR variants are supervised box "
            f"detectors whose masks come from SAM~3 box-prompt, not independent mask heads.}}"
            f"\\label{{{meta['label']}}}",
            "\\footnotesize",
            "\\begin{tabular}{l" + "cccc" * 5 + "}",
            "\\toprule",
            "& \\multicolumn{5}{c}{\\textbf{Easy}} & "
            "\\multicolumn{5}{c}{\\textbf{Medium}} & "
            "\\multicolumn{5}{c}{\\textbf{Hard}} & "
            "\\multicolumn{5}{c}{\\textbf{Aggregate}} \\\\",
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-16}\\cmidrule(lr){17-21}",
            "\\textbf{Method} & pub & ter & A & B & C & pub & ter & A & B & C "
            "& pub & ter & A & B & C & pub & ter & A & B & C \\\\",
            "\\midrule",
        ]
        for method, display in METHODS:
            row = [display]
            for tier_label in ["easy", "medium", "hard", "agg"]:
                for tier_tag, _ in TIERINGS:
                    cell = all_scores[corpus][tier_tag].get(method)
                    v = cell[tier_label] if cell else float("nan")
                    row.append(f"{v:.1f}" if v == v else "--")
            lines.append(" & ".join(row) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        tex_path = OUT / f"{corpus}_expanded.tex"
        tex_path.write_text("\n".join(lines))
        print(f"wrote {tex_path}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
