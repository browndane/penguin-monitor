#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Rebuild the R3 split trees on the five user-verified Penguin-Watch sites.

The trees on disk were assembled from pw_yolo_staging, which points every site
at penguin-watch-unverified, so both the imagery and the labels came from the
retired tree. This script rebuilds them from the verified Roboflow export using
the image and label paths recorded in pw_v2_tiers.json.

p5_r3_random holds a 70/15/15 draw over the 9,968 verified images the 200-image
gold does not use. p5_r3_groupaware holds DAMOa, HALFb and HALFc entire as test,
and splits LOCKb and PETEe 82/18 into train and validation so the train-to-val
ratio matches the random split's 70:15.

    python prep_18_r3_rebuild_verified.py --seed 42
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

GOLD = Path.home() / "bench_any/runs/p5_r1_gold"
TIERS = GOLD / "pw_v2_tiers.json"
GOLD200 = GOLD / "gold_manifest_pw_v2_200.json"
DATASETS = Path.home() / "bench_any/datasets"
RANDOM_DS = DATASETS / "p5_r3_random"
GROUP_DS = DATASETS / "p5_r3_groupaware"
HELD_OUT = ["DAMOa", "HALFb", "HALFc"]
TRAIN_SITES = ["LOCKb", "PETEe"]
VERIFIED_ROOT = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch/"
DATA_YAML = ("train: train/images\nval: valid/images\ntest: test/images\n"
             "nc: 1\nnames:\n- penguin\n")


def gold_stems():
    m = json.loads(GOLD200.read_text())
    return {s for site in m["per_site_draw"].values()
            for tier in site.values() for s in tier}


def write_tree(root, splits, images):
    if root.exists():
        shutil.rmtree(root)
    for split, stems in splits.items():
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for stem in stems:
            rec = images[stem]
            if not rec["image"].startswith(VERIFIED_ROOT):
                raise SystemExit(f"unverified source for {stem}: {rec['image']}")
            name = f"PW_{rec['site']}_{stem}"
            (img_dir / f"{name}.jpg").symlink_to(rec["image"])
            (lbl_dir / f"{name}.txt").symlink_to(rec["label"])
    (root / "data.yaml").write_text(DATA_YAML)


def counts(splits, images):
    return {s: {"n": len(v), "sites": dict(Counter(images[x]["site"] for x in v)),
                "labelled": sum(1 for x in v if images[x]["dets"])}
            for s, v in splits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = json.loads(TIERS.read_text())["images"]
    gold = gold_stems()
    rng = random.Random(args.seed)

    pool = sorted(s for s in images if s not in gold)
    rng.shuffle(pool)
    n = len(pool)
    n_train, n_val = round(n * 0.70), round(n * 0.15)
    rnd_splits = {"train": pool[:n_train],
                  "valid": pool[n_train:n_train + n_val],
                  "test": pool[n_train + n_val:]}

    held = sorted(s for s, r in images.items() if r["site"] in HELD_OUT)
    rest = sorted(s for s, r in images.items() if r["site"] in TRAIN_SITES)
    rng.shuffle(rest)
    cut = round(len(rest) * 0.82)
    grp_splits = {"train": rest[:cut], "valid": rest[cut:], "test": held}

    write_tree(RANDOM_DS, rnd_splits, images)
    write_tree(GROUP_DS, grp_splits, images)

    report = {
        "seed": args.seed,
        "source": "pw_v2_tiers.json, Roboflow user-verified export of five sites",
        "gold_excluded_from_random": len(gold),
        "p5_r3_random": counts(rnd_splits, images),
        "p5_r3_groupaware": counts(grp_splits, images),
        "groupaware_gold_in_train": sorted(gold & set(grp_splits["train"])),
        "groupaware_note": ("The site holdout separates the 120 DAMOa/HALFb/HALFc gold "
                            "frames. LOCKb and PETEe gold frames remain in the group-aware "
                            "training pool, so report group-aware results on the held-out "
                            "sites only."),
    }
    out = GOLD / "r3_rebuild_verified.json"
    out.write_text(json.dumps(report, indent=1))
    for tree in ("p5_r3_random", "p5_r3_groupaware"):
        print(tree, {k: (v["n"], v["labelled"]) for k, v in report[tree].items()})
    print(f"gold frames inside group-aware train: {len(report['groupaware_gold_in_train'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
