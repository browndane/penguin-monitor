#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Bootstrap confidence intervals for the per-tier outcome rates.

Resamples *images*, not instances. Instances within an image share a colony,
a camera and a crowding level, so treating them as independent would give
intervals several times too narrow. Reads the per-image CSV that
``count_decompose.py --by_tier --out`` writes, so no decomposition is redone.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

TIERS = ("Easy", "Medium", "Hard")
FATES = ("matched", "missed", "absorbed", "merge_represented", "split")


def _load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"no rows in {path}")

    def col(name):
        return np.array([float(r.get(name) or 0) for r in rows], dtype=np.float64)

    return {f"{t}_{k}": col(f"{t}_{k}")
            for t in TIERS for k in FATES + ("total",)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per_image", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="append rows to this CSV")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    cols = _load(args.per_image)
    n = len(next(iter(cols.values())))
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.draws, n))

    out_rows = []
    for t in TIERS:
        tot = cols[f"{t}_total"]
        if tot.sum() == 0:
            continue
        tot_b = tot[idx].sum(axis=1)
        for f in FATES:
            num_b = cols[f"{t}_{f}"][idx].sum(axis=1)
            # A draw with no GT in this tier has no rate to contribute.
            ok = tot_b > 0
            rate = num_b[ok] / tot_b[ok]
            lo, hi = np.percentile(rate, [2.5, 97.5])
            point = cols[f"{t}_{f}"].sum() / tot.sum()
            out_rows.append({"tag": args.tag, "tier": t, "outcome": f,
                             "point": round(point, 4), "lo95": round(lo, 4),
                             "hi95": round(hi, 4),
                             "halfwidth": round((hi - lo) / 2, 4),
                             "draws": args.draws, "n_images": n})
            print(f"  {t:<7} {f:<18} {point:.4f}  [{lo:.4f}, {hi:.4f}]")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        exists = args.out.exists()
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(out_rows)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
