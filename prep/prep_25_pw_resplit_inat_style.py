#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build the Penguin-Watch training split, mirroring the iNat veryclean protocol.

Pools all 10,168 frames across the five verified PW sub-colonies in
``pw_yolo_staging`` and random-splits 70/20/10 into train/valid/test, gold
included, matching how ``african-penguin-inaturalist-sam-veryclean`` was split
(6072/1733/856 there). A single class (penguin), YOLO polygon labels already
in staging. Filenames keep their sub-colony prefix so provenance survives the
shuffle. Fixed seed for reproducibility. Images are symlinked, labels are
symlinked too (they already exist as real files in staging).

Output: ~/bench_any/datasets/pw_inat_style/
  train/images, train/labels, valid/images, valid/labels, test/images, test/labels
  data.yaml  (train: train/images, val: valid/images, test: test/images, nc:1)

Usage:
    python prep_25_pw_resplit_inat_style.py [--seed 42] [--overwrite]
"""
from __future__ import annotations
import os, sys, argparse, random
from pathlib import Path

STAGING = Path(os.path.expanduser("~/bench_any/datasets/pw_yolo_staging"))
OUT = Path(os.path.expanduser("~/bench_any/datasets/pw_inat_style"))
SUBCOLONIES = ["DAMOa", "HALFb", "HALFc", "LOCKb", "PETEe"]
RATIO = (0.70, 0.20, 0.10)  # train / valid / test
IMG_EXTS = (".jpg", ".png", ".JPG", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.overwrite and any(OUT.rglob("*.txt")):
        print(f"[skip] {OUT} already populated; pass --overwrite to rebuild")
        return
    if args.overwrite and OUT.exists():
        import shutil
        shutil.rmtree(OUT)

    # Collect (image_path, label_path) per sub-colony.
    pairs = []
    for sub in SUBCOLONIES:
        sdir = STAGING / sub
        if not sdir.is_dir():
            print(f"[warn] missing sub-colony {sub}", file=sys.stderr)
            continue
        for img in sorted(sdir.joinpath("images").iterdir()):
            if img.suffix not in IMG_EXTS:
                continue
            lbl = sdir.joinpath("labels", img.stem + ".txt")
            if not lbl.is_file():
                continue
            pairs.append((img, lbl))

    print(f"[pool] {len(pairs)} (image, label) pairs across {len(SUBCOLONIES)} sub-colonies")

    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(round(n * RATIO[0]))
    n_valid = int(round(n * RATIO[1]))
    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test": pairs[n_train + n_valid:],
    }
    for k, v in splits.items():
        print(f"[split] {k}: {len(v)}")

    for split in splits:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        for img, lbl in items:
            img_link = OUT / split / "images" / img.name
            lbl_link = OUT / split / "labels" / lbl.name
            if not img_link.exists():
                img_link.symlink_to(img.resolve())
            if not lbl_link.exists():
                lbl_link.symlink_to(lbl.resolve())

    # data.yaml — mirror inat veryclean exactly (valid dir named "valid")
    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: {OUT}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: 1\n"
        f"names: ['penguin']\n"
    )
    print(f"[done] {OUT}")
    for split in splits:
        ni = len(list((OUT / split / "images").iterdir()))
        print(f"  {split}: {ni} images")


if __name__ == "__main__":
    main()
