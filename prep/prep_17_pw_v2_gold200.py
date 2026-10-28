#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Draw the 200-image v2 Penguin-Watch gold from the five user-verified sites.

Forty images per site, stratified by hardness tercile within that site's own
labelled pool. The image score is the mean per-instance D_i already recorded in
pw_v2_tiers.json; terciles are cut by rank, not by the absolute 0.30/0.58
thresholds, because roughly ninety percent of instances are medium and absolute
cuts leave the easy and hard strata too thin to read.

Replaces prep_16_pw_v2_gold.py, whose greedy per-tier instance floor drew fifty.

    python prep_17_pw_v2_gold200.py --seed 42
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

GOLD = Path.home() / "bench_any/runs/p5_r1_gold"
TIERS = GOLD / "pw_v2_tiers.json"
V1_MANIFEST = GOLD / "gold_manifest.json"
OUT_MANIFEST = GOLD / "gold_manifest_pw_v2_200.json"
OUT_DS = Path.home() / "bench_any/datasets/p5_r1_gold_pw_v2_200"
SITES = ["DAMOa", "HALFb", "HALFc", "LOCKb", "PETEe"]
VERIFIED_ROOT = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch/"
BANNED = ("pw_yolo_staging", "penguin-watch-unverified")
# tercile index -> how many frames to draw; the extra frame goes to hard
DRAW = {"t1_easy": 13, "t2_medium": 13, "t3_hard": 14}
TERCILES = ["t1_easy", "t2_medium", "t3_hard"]


def v1_paired_stems(images):
    """Stems the v1 gold drew that fall inside the five verified sites."""
    v1 = json.loads(V1_MANIFEST.read_text())
    out = []
    for p in v1["test_pw"]:
        stem = re.sub(r"^PW_[A-Za-z]+_", "", os.path.splitext(os.path.basename(p))[0])
        if stem in images:
            out.append(stem)
    return out


def site_terciles(pool, images):
    """Rank a site's labelled stems by mean_D and cut into three equal counts."""
    ranked = sorted(pool, key=lambda s: (mean_d(images[s]), s))
    n = len(ranked)
    cuts = [round(n / 3), round(2 * n / 3)]
    groups = {
        "t1_easy": ranked[: cuts[0]],
        "t2_medium": ranked[cuts[0]: cuts[1]],
        "t3_hard": ranked[cuts[1]:],
    }
    bounds = {t: [mean_d(images[g[0]]), mean_d(images[g[-1]])] for t, g in groups.items()}
    return groups, bounds


def mean_d(rec):
    return sum(d["score"] for d in rec["dets"]) / len(rec["dets"])


def check_provenance(rec):
    for key in ("image", "label"):
        p = rec[key]
        if not p.startswith(VERIFIED_ROOT):
            raise SystemExit(f"{key} outside the verified export: {p}")
        if any(b in p for b in BANNED):
            raise SystemExit(f"{key} points at a retired tree: {p}")


def check_polygons(label_path):
    for line in Path(label_path).read_text().split("\n"):
        if line.strip() and len(line.split()) <= 5:
            raise SystemExit(f"box-format label, expected polygons: {label_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = json.loads(TIERS.read_text())["images"]
    rng = random.Random(args.seed)
    paired = set(v1_paired_stems(images))

    chosen, per_site, bounds_all, paired_kept = [], {}, {}, []
    for site in SITES:
        pool = [s for s, r in images.items() if r["site"] == site and r["dets"]]
        groups, bounds = site_terciles(pool, images)
        bounds_all[site] = bounds
        per_site[site] = {}
        for tier in TERCILES:
            members = list(groups[tier])
            if len(members) < DRAW[tier]:
                raise SystemExit(f"{site}/{tier} holds {len(members)}, needs {DRAW[tier]}")
            seeds = [s for s in members if s in paired]
            rng.shuffle(members)
            pick = seeds[: DRAW[tier]]
            pick += [s for s in members if s not in pick][: DRAW[tier] - len(pick)]
            paired_kept += [s for s in pick if s in paired]
            per_site[site][tier] = sorted(pick)
            chosen += pick

    img_dir, lbl_dir = OUT_DS / "test/images", OUT_DS / "test/labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    inst_tiers = Counter()
    for stem in chosen:
        rec = images[stem]
        check_provenance(rec)
        check_polygons(rec["label"])
        inst_tiers.update(d["tier"] for d in rec["dets"])
        name = f"PW_{rec['site']}_{stem}"
        for dst, src in ((img_dir / f"{name}.jpg", rec["image"]),
                         (lbl_dir / f"{name}.txt", rec["label"])):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    (OUT_DS / "data.yaml").write_text(
        f"path: {OUT_DS}\ntrain: test/images\nval: test/images\n"
        f"test: test/images\nnc: 1\nnames: ['penguin']\n")

    manifest = {
        "seed": args.seed,
        "design": "40 per site, hardness terciles by rank within site, 13/13/14",
        "annotation_version": "v2",
        "provenance": "Roboflow user-verified export, " + VERIFIED_ROOT.rstrip("/"),
        "sites": SITES,
        "n_images": len(chosen),
        "n_instances": sum(len(images[s]["dets"]) for s in chosen),
        "instance_tier_counts": dict(inst_tiers),
        "per_site_draw": per_site,
        "per_site_tercile_mean_D_bounds": bounds_all,
        "per_site_pool_size": {s: sum(1 for _, r in images.items()
                                      if r["site"] == s and r["dets"]) for s in SITES},
        "v1_paired_candidates": sorted(paired),
        "v1_paired_surviving": sorted(paired_kept),
        "v1_paired_note": ("The v1 annotations are invalid, so a surviving pairing is a "
                           "convenience for inspection and carries no delta claim."),
        "v1_test_pw_removed_from_p5_r1_gold": json.loads(V1_MANIFEST.read_text())["test_pw"],
        "dataset_root": str(OUT_DS),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"{len(chosen)} images, {manifest['n_instances']} instances, "
          f"{len(paired_kept)} of {len(paired)} v1 pairings kept")
    print(f"instance tiers: {dict(inst_tiers)}")
    print(f"wrote {OUT_MANIFEST} and {OUT_DS}")


if __name__ == "__main__":
    main()
