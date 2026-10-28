#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Reassign difficulty tiers by per-corpus rank tercile, 2026-08-01.

The instance difficulty score D_i is unchanged. Only the cut moves: the fixed
0.30 and 0.58 thresholds of prep_09 are replaced by the thirty-third and
sixty-seventh percentile of the corpus's own score distribution, cut by rank in
the manner of prep_17.site_terciles so a tie cannot push a tier oversize.

Every sidecar already carries a per-detection `score`, so this is a rewrite of
the `tier` field alone and no mask, box or image is touched. Input files are
never modified; output goes to a new path, which is what makes the old numbers
reproducible by checksum rather than by re-running anything.

    python prep_22_retier_terciles.py --in  <sidecar>.json --out <sidecar>_tercile.json
    python prep_22_retier_terciles.py --in gold200_tiers.json --out ... --split-by-corpus

`--split-by-corpus` is for the four-corpus v1 file `gold200_tiers.json`, whose
keys are absolute filepaths; every other sidecar holds one corpus already and
terciling the whole file is the per-corpus cut. Corpus identity reuses
prep_10d_per_corpus.corpus_of so tiering and reporting cannot disagree.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

TIERS = ["easy", "medium", "hard"]


def corpus_of(fp):
    """Verbatim from prep_10d_per_corpus.py:15, the existing corpus mapping."""
    b = os.path.basename(fp)
    if b.startswith("PW_"):
        return "PW"
    if b.startswith("1_") and "rf." in b:
        return "SANParks"
    if "rf." in b:
        return "Boulders"
    return "iNat"


def rank_terciles(items):
    """items: [(score, fname, det_index)]. Returns {(fname, det_index): tier}.

    Sorted by score with the filename and detection index breaking ties, so the
    assignment is deterministic across runs. Cuts follow prep_17.site_terciles:
    round(n/3) and round(2n/3) positions in the ranking, not interpolated
    percentiles of the score axis.
    """
    ranked = sorted(items, key=lambda e: (e[0], e[1], e[2]))
    n = len(ranked)
    c0, c1 = round(n / 3), round(2 * n / 3)
    out = {}
    for pos, (_score, fname, idx) in enumerate(ranked):
        t = "easy" if pos < c0 else ("medium" if pos < c1 else "hard")
        out[(fname, idx)] = t
    return out, ranked, (c0, c1)


def tie_report(ranked, cuts):
    """How many instances share the exact score sitting on each cut boundary.

    A large tie block straddling a cut means the split inside that block is
    arbitrary, which is worth stating rather than discovering later.
    """
    rep = {}
    scores = [e[0] for e in ranked]
    for label, c in zip(("easy|medium", "medium|hard"), cuts):
        if c <= 0 or c >= len(scores):
            rep[label] = {"boundary_score": None, "n_tied": 0}
            continue
        s = scores[c]
        rep[label] = {"boundary_score": s, "n_tied": sum(1 for v in scores if v == s)}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--split-by-corpus", action="store_true",
                    help="tercile within each corpus of a four-corpus file")
    args = ap.parse_args()

    data = json.load(open(os.path.expanduser(args.inp)))
    manifest = data["manifest"]

    groups = defaultdict(list)
    for fname, dets in manifest.items():
        g = corpus_of(fname) if args.split_by_corpus else data.get("corpus", "all")
        for e in dets:
            groups[g].append((e["score"], fname, e["idx"]))

    assignment = {}
    per_corpus = {}
    for g, items in groups.items():
        amap, ranked, cuts = rank_terciles(items)
        assignment.update(amap)
        per_corpus[g] = {
            "n_instances": len(items),
            "cut_positions": list(cuts),
            "cut_scores": [ranked[cuts[0]][0] if cuts[0] < len(ranked) else None,
                           ranked[cuts[1]][0] if cuts[1] < len(ranked) else None],
            "new_counts": dict(Counter(amap.values())),
            "ties_on_cut": tie_report(ranked, cuts),
        }

    old_counts = Counter()
    new_counts = Counter()
    for fname, dets in manifest.items():
        for e in dets:
            old_counts[e["tier"]] += 1
            e["tier"] = assignment[(fname, e["idx"])]
            new_counts[e["tier"]] += 1

    data["tier_counts"] = {t: new_counts[t] for t in TIERS}
    data["tier_mode"] = "per-corpus rank tercile (prep_22, 2026-08-01)"
    data["tier_mode_previous"] = "fixed thresholds 0.30 / 0.58 (prep_09)"
    data["tier_counts_previous"] = {t: old_counts[t] for t in TIERS}
    data["tercile_detail"] = per_corpus

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, "w"))
    print(f"{os.path.basename(args.inp)}: old {dict(old_counts)} -> new {dict(new_counts)}")
    for g, d in sorted(per_corpus.items()):
        print(f"  {g}: n={d['n_instances']} cuts@{d['cut_positions']} "
              f"scores {[None if s is None else round(s, 4) for s in d['cut_scores']]} "
              f"counts {d['new_counts']} ties {{"
              + ", ".join(f"{k}: {v['n_tied']}" for k, v in d['ties_on_cut'].items()) + "}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
