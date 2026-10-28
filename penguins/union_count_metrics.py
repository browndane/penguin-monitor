#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""SAM3 union prompt count metrics (precision/recall/count_mae) through the
SAME asrun+fiftyone sieve the single-prompt headline uses. CPU, no GPU.

Reuses sieve_composition.run_arm / decode_image / iou_matrix / load_corpus
plumbing — no second sieve implementation. The union is just a different
input to the same scorer. Compares to the single-prompt count metrics so
the "report top method" call is made on BOTH axes (mAP and count), not mAP
alone.

    ~/cvML/bin/python penguins/union_count_metrics.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sieve_composition import (  # noqa: E402
    ASRUN, CANON, load_corpus, decode_image, iou_matrix, run_arm, score_corpus,
    CONTACT, _touches_border, EDGE_BORDER,
)
import sieve_composition as SC  # noqa: E402

R1 = Path.home() / "bench_any" / "runs" / "p5_r1_v2"
RUNS = Path.home() / "bench_any" / "runs" / "obj2_counting"
CORPORA = {"boulders-penguin": "boulders", "sanparks-algoa": "sanparks"}
THR = 0.40
OUT = RUNS / "union_count_metrics.json"
DONE = RUNS / "_union_count_metrics_done"


def build_union_raw(corpus, thr):
    """Union of 'penguin' + 'a penguin' raw preds in _sam3_raw format: {fn:[{rle,score,box}]}."""
    gt = json.loads((RUNS / f"sam2refine_{corpus}" / "gt_coco.json").read_text())
    fn_set = {im["file_name"] for im in gt["images"]}
    by_fn = {}
    for stem in ("penguin", "a_penguin"):
        src = json.loads((RUNS / "sam3_prompts_gold" / f"{corpus}_{stem}.json").read_text())
        iid_fn = {i["id"]: i["file_name"] for i in src["images"]}
        for a in src["annotations"]:
            fn = iid_fn.get(a["image_id"])
            if fn is None or fn not in fn_set:
                continue
            if float(a.get("score", 1.0)) < thr:
                continue
            seg = a["segmentation"]
            rle = {"size": list(seg["size"]), "counts": seg["counts"]}
            by_fn.setdefault(fn, []).append({"rle": rle, "score": float(a.get("score", 1.0)),
                                             "box": [float(x) for x in a["bbox"]]})
    return by_fn


def main():
    with_masks = os.environ.get("SIEVE_COMP_BBOX_ONLY", "0") != "1"
    out = {"mask_areas": with_masks, "thr": THR, "corpora": {}}
    for corpus, sub in CORPORA.items():
        # single-prompt via the authors' load_corpus (the published raw)
        sp_raw, hw, gt_boxes, gt_rles = load_corpus(corpus)
        SC.HW = hw
        CONTACT.clear()
        union_raw = build_union_raw(corpus, THR)

        def prep(raw):
            prepped = {}
            for fn, anns in raw.items():
                sboxes, mboxes, areas, scores, rles = decode_image(anns, hw[fn], with_masks)
                g_r, g_b = gt_rles.get(fn, []), gt_boxes.get(fn, [])
                h_, w_ = hw[fn]
                for i, bb in enumerate(sboxes):
                    if _touches_border(bb, (w_, h_), EDGE_BORDER):
                        CONTACT[(fn, i)] = SC.contact_ratio(rles[i], bb, (w_, h_))
                M = SC.iou_matrix(rles, mboxes, g_r, g_b, "mask")
                prepped[fn] = (areas, sboxes, scores, len(g_r), M)
            return prepped

        sp_data = prep(sp_raw)
        un_data = prep(union_raw)
        n_sp = sum(len(v[1]) for v in sp_data.values())
        n_un = sum(len(v[1]) for v in un_data.values())
        res = {"n_single": n_sp, "n_union": n_un}
        for cfg_name, cfg in (("canon", CANON), ("asrun", ASRUN)):
            sp_block = run_arm(sp_data, "fiftyone", cfg)
            un_block = run_arm(un_data, "fiftyone", cfg)
            res[cfg_name] = {"single_prompt": sp_block, "union": un_block}
            line = (f"[{corpus}/{cfg_name}] single-prompt P={sp_block['precision']:.4f} "
                    f"R={sp_block['recall']:.4f} MAE={sp_block['count_mae']:.4f} "
                    f"(n={n_sp}) | union P={un_block['precision']:.4f} "
                    f"R={un_block['recall']:.4f} MAE={un_block['count_mae']:.4f} (n={n_un})")
            print(line)
        out["corpora"][corpus] = res
    OUT.write_text(json.dumps(out, indent=2, default=float))
    DONE.touch()
    print(f"\nWrote {OUT}, touched {DONE}")


if __name__ == "__main__":
    main()
