#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build group-aware Penguin-Watch splits for the transfer experiments.

prep_25 (pw_inat_style) shuffles all pairs together, so every site leaks across
train/val/test. This script partitions by site: each site lives in exactly one
split. Two configurations, mirroring the two agreed experiments:

  --config small_sites_test  (E2)
      train+valid: LOCKb, PETEe (the two big dense colonies, ~50% empty)
      test:        DAMOa, HALFb, HALFc (three small sparse colonies, <2% empty)
      Tests transfer to unseen camera deployments. Cannot test FP-on-empties
      because the test set has 7 empties in 727 frames.

  --config holdout_petee     (E3)
      train+valid: LOCKb, DAMOa, HALFb, HALFc
      test:        PETEe (387 frames, 47% empty, unseen)
      Tests transfer AND keeps empties in the test set, so FP-on-empties is
      answerable.

Within the non-test pool, images are shuffled (seed 42) and sliced 88/12 into
train/valid (mirrors prep_25's 70/20 of the non-test portion ~ 88/12 of the
whole non-test pool). The test pool is untouched by the shuffle. This keeps the
group boundary hard: no site crosses train/val into test.

Output: ~/bench_any/datasets/pw_groupaware_<config>/
        with data.yaml, train/valid/test images+labels symlinked from staging.

Usage:
    python prep_27_pw_groupaware_split.py --config small_sites_test
    python prep_27_pw_groupaware_split.py --config holdout_petee
    python prep_27_pw_groupaware_split.py --config small_sites_test --check
"""
from __future__ import annotations
import argparse, os, random, shutil
from pathlib import Path

STAGING = Path(os.path.expanduser("~/bench_any/datasets/pw_yolo_staging"))
OUTROOT = Path(os.path.expanduser("~/bench_any/datasets"))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG")
SEED = 42

CONFIGS = {
    "small_sites_test": {
        "train_val_sites": ["LOCKb", "PETEe"],
        "test_sites": ["DAMOa", "HALFb", "HALFc"],
        "train_frac": 0.88,  # of the train_val pool; rest is valid
    },
    "holdout_petee": {
        "train_val_sites": ["LOCKb", "DAMOa", "HALFb", "HALFc"],
        "test_sites": ["PETEe"],
        "train_frac": 0.88,
    },
}


def site_pairs(sites):
    """Return list of (img_path, lbl_path) for the given site list, skipping
    images without a label file (staging always writes one, but be safe)."""
    pairs = []
    for s in sites:
        sdir = STAGING / s
        if not sdir.is_dir():
            raise SystemExit(f"staging site missing: {s}")
        idir = sdir / "images"
        ldir = sdir / "labels"
        for img in sorted(idir.iterdir()):
            if img.suffix not in IMG_EXTS:
                continue
            lbl = ldir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            pairs.append((img, lbl))
    return pairs


def build(cfg_name, overwrite=False):
    cfg = CONFIGS[cfg_name]
    out = OUTROOT / f"pw_groupaware_{cfg_name}"
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    tv_pairs = site_pairs(cfg["train_val_sites"])
    te_pairs = site_pairs(cfg["test_sites"])
    random.seed(SEED)
    random.shuffle(tv_pairs)
    n_train = int(round(len(tv_pairs) * cfg["train_frac"]))
    train = tv_pairs[:n_train]
    valid = tv_pairs[n_train:]
    test = te_pairs
    splits = {"train": train, "valid": valid, "test": test}
    for split in splits:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        for img, lbl in items:
            oi = out / split / "images" / img.name
            ol = out / split / "labels" / lbl.name
            if not oi.exists():
                oi.symlink_to(img.resolve())
            if not ol.exists():
                ol.symlink_to(lbl.resolve())
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: 1\n"
        f"names: ['penguin']\n"
    )
    # report
    print(f"[built] {out}")
    for split in ("train", "valid", "test"):
        items = splits[split]
        n_emp = sum(1 for _, lbl in items if lbl.stat().st_size == 0)
        n = len(items)
        print(f"  {split:<6} {n:>5}  empty={n_emp} ({100*n_emp/n if n else 0:.1f}%)")
    print(f"  train_val_sites={cfg['train_val_sites']}  test_sites={cfg['test_sites']}")


def check(cfg_name):
    cfg = CONFIGS[cfg_name]
    out = OUTROOT / f"pw_groupaware_{cfg_name}"
    if not out.is_dir():
        raise SystemExit(f"{out} not built")
    bad = 0
    tv = set(cfg["train_val_sites"])
    te = set(cfg["test_sites"])
    # every image in train/valid must be from a train_val site; test from a test site
    site_of = {}
    for s in tv | te:
        for img in (STAGING / s / "images").iterdir():
            site_of[img.name] = s
    for split, allowed in (("train", tv), ("valid", tv), ("test", te)):
        d = out / split / "images"
        for img in d.iterdir():
            if img.name.startswith("."):
                continue
            s = site_of.get(img.name)
            if s is None:
                print(f"  PARITY FAIL {split}: unknown site for {img.name}")
                bad += 1
            elif s not in allowed:
                print(f"  PARITY FAIL {split}: {img.name} is from {s}, allowed={sorted(allowed)}")
                bad += 1
    if bad:
        raise SystemExit(f"parity FAILED: {bad} site-leak violations")
    print("[check] parity OK: no site crosses its allowed split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check(args.config)
    else:
        build(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
