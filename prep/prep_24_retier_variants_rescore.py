#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Retier every corpus under the three candidate repair variants and rescore
every cached cell, so the per-corpus table exists for whichever variant is
chosen.

Variants, all cut as per-corpus rank terciles like prep_22:

  A  rank-normalise the size term and the overlap-and-crowding remainder
     separately within the corpus, reweighted 0.40 / 0.60. Keeps all three
     published terms contributing.
  B  rank-normalise the size term alone and keep the overlap-and-crowding
     remainder raw, reweighted 0.40 / 0.60. Tests whether rank-normalising
     size alone (without also ranking the remainder) differs from C.
  C  rank the size term alone. Simplest, and empirically what variant B
     collapses to.

Writes `*_tiers_variantA.json` / `*_tiers_variantB.json` / `*_tiers_variantC.json`
beside the existing sidecars, never overwriting one, and a reeval json per cell
carrying `agg` and `agg_iw` in the prep_10 shape. Appends a bounded table to
FINDINGS.md.
"""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from prep_10_filebased import build_gt_coco, build_pred_coco, eval_tier

TIERS = ["easy", "medium", "hard"]
GOLD = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold"))
V2 = Path(os.path.expanduser("~/bench_any/runs/p5_r1_v2"))
NV = GOLD / "native_vs_sam2_tercile"
OUT = V2 / "tier_diag" / "variant_rescore"
FINDINGS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting/FINDINGS.md"))

CORPORA = {
    "sanparks": (V2 / "sanparks/sanparks_tiers_tercile.json", V2 / "sanparks"),
    "boulders": (V2 / "boulders/boulders_tiers_tercile.json", V2 / "boulders"),
    "inat_v1_50": (GOLD / "inat_v1_50_tiers_tercile.json", GOLD),
    "pw_v2_200": (GOLD / "pw_v2_200_tiers_tercile.json", GOLD),
}

CELLS = {
    "sanparks": {m: NV / "sanparks" / f"{m}_pred.json"
                 for m in ("eomt-large", "rf-detr-seg-large", "yolo11x-seg")},
    "boulders": {m: NV / "boulders" / f"{m}_pred.json"
                 for m in ("eomt-large", "rf-detr-seg-large", "yolo11x-seg")},
    "inat_v1_50": {m: NV / "inat_v1_50" / f"{m}_pred.json"
                   for m in ("eomt-large", "rf-detr-seg-large", "yolo11x-seg")},
    "pw_v2_200": dict(
        {m: NV / "pw_v2_200" / f"{m}_pred.json"
         for m in ("eomt-large", "rf-detr-seg-large", "yolo11x-seg")},
        **{m: GOLD / "pw200_scores/pw_v2_200" / f"{m}_pred.json"
           for m in ("rtdetr-4", "rtdetr-3")},
        **{"sam3_sieve": GOLD / "pw200_zeroshot/pw_v2_200_sam3_pred.json"},
    ),
}


def size_term(det):
    return 1.0 - min(det["area"] / (0.04 * det["width"] * det["height"]), 1.0)


def variant_scores(flat):
    S = np.array([size_term(d) for d in flat])
    total = np.array([d["score"] for d in flat])
    rest = total - 0.40 * S
    n = len(S)
    u = lambda x: (rankdata(x) - 0.5) / n
    return {
        "variantA": 0.40 * u(S) + 0.60 * u(rest),
        "variantB": 0.40 * u(S) + 0.60 * rest,
        "variantC": u(S),
    }


def assign(manifest, new_score):
    flat_order = [(f, i) for f, dets in manifest.items() for i in range(len(dets))]
    n = len(flat_order)
    tier = np.empty(n, dtype=object)
    for rank, idx in enumerate(np.argsort(new_score)):
        tier[idx] = "easy" if rank < n // 3 else ("medium" if rank < 2 * n // 3 else "hard")
    out = {}
    k = 0
    for f, dets in manifest.items():
        out[f] = [dict(d, tier=tier[k + j]) for j, d in enumerate(dets)]
        k += len(dets)
    return out


def score_cell(mani, pred_path):
    pdata = json.load(open(pred_path))
    preds = pdata["preds"] if "preds" in pdata else pdata
    pl = build_pred_coco(preds, mani)
    res, ns = {}, {}
    for t in TIERS:
        gt = build_gt_coco(mani, t)
        ns[t] = sum(1 for a in gt["annotations"] if a["iscrowd"] == 0)
        with contextlib.redirect_stdout(io.StringIO()):
            res[t] = eval_tier(gt, pl)
    vals = [res[t] for t in TIERS]
    res["agg"] = float(np.mean(vals))
    res["agg_iw"] = float(sum(res[t] * ns[t] for t in TIERS) / sum(ns.values()))
    res.update({f"n_{t}": ns[t] for t in TIERS})
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for corpus, (tiers_path, side_dir) in CORPORA.items():
        src = json.load(open(tiers_path))
        manifest = src["manifest"]
        flat = [d for dets in manifest.values() for d in dets]
        variants = variant_scores(flat)
        for vname, ns in variants.items():
            mani = assign(manifest, ns)
            sidecar = dict(src)
            sidecar["manifest"] = mani
            sidecar["tier_rule"] = f"{vname}, per-corpus rank terciles (prep_24, 2026-08-01)"
            counts = {t: sum(1 for dets in mani.values() for d in dets if d["tier"] == t)
                      for t in TIERS}
            sidecar["tier_counts"] = counts
            sp = Path(side_dir) / f"{corpus}_tiers_{vname}.json"
            json.dump(sidecar, open(sp, "w"))
            for model, pred_path in CELLS[corpus].items():
                if not Path(pred_path).exists():
                    print(f"skip {corpus}/{model}, no preds", flush=True)
                    continue
                r = score_cell(mani, pred_path)
                json.dump(r, open(OUT / f"{corpus}_{model}_{vname}_reeval.json", "w"), indent=2)
                table.setdefault(corpus, {}).setdefault(model, {})[vname] = r
                print(f"{corpus:11s} {model:18s} {vname} "
                      f"{r['easy']:.1f}/{r['medium']:.1f}/{r['hard']:.1f} "
                      f"agg {r['agg']:.1f} agg_iw {r['agg_iw']:.1f}", flush=True)
        print(f"-- {corpus} sidecars written to {side_dir}", flush=True)

    json.dump(table, open(OUT / "variant_rescore_summary.json", "w"), indent=2)

    lines = ["", "## Per-corpus table under all three repair variants, 2026-08-01", "",
             "Every cached cell rescored under variant A (all three published terms",
             "rank-normalised), variant B (size term rank-normalised, remainder kept",
             "raw), and variant C (size term alone). Sidecars written beside the",
             "originals as `*_tiers_variantA.json`, `*_tiers_variantB.json` and",
             "`*_tiers_variantC.json`; nothing overwritten. Cells are easy / medium",
             "/ hard, then the unweighted and instance-weighted aggregates. Monotone",
             "marks whether easy >= medium >= hard.",
             "", "| corpus | model | variant | easy / medium / hard | agg | agg_iw | monotone |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for corpus, models in table.items():
        for model, vs in models.items():
            for vname, r in vs.items():
                mono = r["easy"] >= r["medium"] >= r["hard"]
                lines.append(
                    f"| {corpus} | {model} | {vname[-1]} | "
                    f"{r['easy']:.1f} / {r['medium']:.1f} / {r['hard']:.1f} | "
                    f"{r['agg']:.1f} | {r['agg_iw']:.1f} | {'yes' if mono else 'NO'} |")
    lines += ["", "No variant is applied. `prep_09` and the tercile sidecars are unchanged,",
              "and no number has moved into `tab:percorpus_v1`.", ""]
    with open(FINDINGS, "a") as fh:
        fh.write("\n".join(lines))
    print("appended table to FINDINGS.md", flush=True)


if __name__ == "__main__":
    main()
