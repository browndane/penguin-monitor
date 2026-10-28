#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Attribute every image in ``penguin-seg-combined`` to a source by content hash.

Filenames are not reliable here: 61 images in ``penguin-finder-marine-aerial-sam``
were renamed relative to their source, so attribution goes by md5 of the image
bytes. Candidate sources are every sibling of the combined directory under
``SEGMENT/``, plus the nested ``multiclass/`` and ``original_splits/`` trees,
which hold variants that are not siblings.

Emits JSON on stdout: per-source deduplicated counts, split membership,
cross-split leakage on content hash, and any image that matched nothing.

    ~/cvML/bin/python penguins/combined_provenance.py > /tmp/prov.json
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/"
            "RoboflowDatasets/SEGMENT")
COMBINED = ROOT / "penguin-seg-combined"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def images(d):
    for p in d.rglob("*"):
        if p.suffix.lower() in EXTS and p.is_file():
            yield p


def source_dirs():
    """Every candidate source, ordered so top-level siblings win attribution."""
    top = [d for d in sorted(ROOT.iterdir())
           if d.is_dir() and d.name not in ("penguin-seg-combined", "multiclass",
                                            "original_splits")]
    nested = []
    for parent in ("multiclass", "original_splits"):
        pd = ROOT / parent
        if pd.is_dir():
            nested += [d for d in sorted(pd.iterdir()) if d.is_dir()]
    return top + nested


def main():
    # Combined set: hash -> (split, [filenames])
    comb = {}
    dup_within = defaultdict(list)
    for split in ("train", "valid", "test"):
        sd = COMBINED / split
        if not sd.is_dir():
            continue
        for p in images(sd):
            h = md5(p)
            if h in comb:
                dup_within[h].append(f"{split}/{p.name}")
            else:
                comb[h] = (split, p.name)

    # Sources, first match wins so nested variants only claim what top-level
    # siblings did not.
    attributed = {}
    per_source = defaultdict(lambda: {"train": 0, "valid": 0, "test": 0})
    source_totals = {}
    for sd in source_dirs():
        key = str(sd.relative_to(ROOT))
        hs = {md5(p) for p in images(sd)}
        source_totals[key] = len(hs)
        for h in hs & comb.keys():
            if h not in attributed:
                attributed[h] = key
                per_source[key][comb[h][0]] += 1

    unattributed = [comb[h][1] for h in comb if h not in attributed]

    # Cross-split leakage: a hash appearing in more than one split is caught by
    # dup_within above, since comb is keyed on hash. Report the split pairs.
    leak = []
    for h, names in dup_within.items():
        splits = {comb[h][0]} | {n.split("/")[0] for n in names}
        if len(splits) > 1:
            leak.append({"md5": h, "splits": sorted(splits), "names": names})

    out = {
        "combined_unique_images": len(comb),
        "combined_by_split": {
            s: sum(1 for v in comb.values() if v[0] == s)
            for s in ("train", "valid", "test")},
        "duplicate_hashes_within_combined": len(dup_within),
        "cross_split_leakage": leak,
        "unattributed": unattributed,
        "per_source": {k: dict(v, total=sum(v.values()),
                               source_unique=source_totals[k])
                       for k, v in sorted(per_source.items(),
                                          key=lambda kv: -sum(kv[1].values()))},
        "sources_contributing_nothing": sorted(
            k for k in source_totals if k not in per_source),
    }
    out["attributed_total"] = sum(v["total"] for v in out["per_source"].values())
    json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
