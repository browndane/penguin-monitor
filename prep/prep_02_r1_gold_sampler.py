#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Script 2: Gold-subset sampler + R1 gold dataset.

Builds ~/bench_any/datasets/p5_r1_gold/ (YOLO format):
- TEST = 100 held-out images: 50 PW (round-robin across 16 sub-colony codes,
  uniform within each) + 50 iNat (uniform over the 8,661 labeled images
  EXCLUDING the 3 BouldersPenguin files).
- TRAIN/VALID from the NON-GOLD remainder:
  * ALL PW non-gold images -> train.
  * iNat non-gold: 90% train / 10% valid.
- Writes data.yaml (nc=1, names=[penguin], train/val/test paths).
- Writes manifest ~/bench_any/runs/p5_r1_gold/gold_manifest.json.
- Symlinks images; copies label files.

Seed = numpy.random.default_rng(seed=42).
"""
from __future__ import annotations
import os, sys, json, shutil, argparse
from pathlib import Path
import numpy as np
import yaml

PW_STAGING = Path(os.path.expanduser("~/bench_any/datasets/pw_yolo_staging"))
INAT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/african-penguin-inaturalist-sam-veryclean")
OUT = Path(os.path.expanduser("~/bench_any/datasets/p5_r1_gold"))
MAN_DIR = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold"))
SEED = 42
BOULDERS = ["BouldersPenguin2.jpg", "BouldersPenguin3.jpg", "BouldersPenguin4.jpg"]
LABEL_EXT = ".txt"
IMG_EXT = ".jpg"

def list_pw_staging(nonempty_only=False):
    """Return list of (image_path, label_path, subcolony, filename) for PW staging.
    If nonempty_only=True, only include images whose label file has >=1 line
    (i.e. has at least one penguin annotation). Used for the gold TEST set so
    the held-out test actually evaluates detection on frames with penguins.
    """
    out = []
    if not PW_STAGING.exists():
        return out
    for sub in sorted(os.listdir(PW_STAGING)):
        sub_img_dir = PW_STAGING / sub / "images"
        sub_lbl_dir = PW_STAGING / sub / "labels"
        if not sub_img_dir.is_dir():
            continue
        for img in sorted(os.listdir(sub_img_dir)):
            if not img.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(img)[0]
            lbl = sub_lbl_dir / (stem + LABEL_EXT)
            if nonempty_only:
                # only include if label exists and has >=1 non-empty line
                if not lbl.exists():
                    continue
                try:
                    if lbl.read_text().strip() == "":
                        continue
                except Exception:
                    continue
            out.append((sub_img_dir / img, lbl, sub, img))
    return out

def list_inat_labeled():
    """Return list of (image_path, label_path, split, filename) for iNat, excluding Boulders."""
    out = []
    for split in ["train", "valid", "test"]:
        img_dir = INAT / split / "images"
        lbl_dir = INAT / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(os.listdir(img_dir)):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if img in BOULDERS or os.path.splitext(img)[0] in [os.path.splitext(b)[0] for b in BOULDERS]:
                continue
            stem = os.path.splitext(img)[0]
            lbl = lbl_dir / (stem + LABEL_EXT)
            out.append((img_dir / img, lbl, split, img))
    return out

def place(path_pair, dst_img, dst_lbl, symlink=True):
    """Place image (symlink) and label (copy) into YOLO dir."""
    src_img, src_lbl = path_pair
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    # image: symlink to source
    if not dst_img.exists():
        os.symlink(src_img, dst_img)
    # label: copy so we can later edit independently if needed (cheap, text)
    if not dst_lbl.exists():
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw_test_n", type=int, default=50)
    ap.add_argument("--inat_test_n", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    MAN_DIR.mkdir(parents=True, exist_ok=True)
    for s in ["train", "valid", "test"]:
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    # --- enumerate PW staging ---
    # ALL PW images (incl empty-label) go to train; gold TEST sampled from non-empty only.
    pw_all = list_pw_staging(nonempty_only=False)
    pw_nonempty = list_pw_staging(nonempty_only=True)
    pw = pw_all  # default pool for train = all PW frames
    # group non-empty by subcolony (for gold test round-robin)
    pw_by_sub = {}
    for rec in pw_nonempty:
        pw_by_sub.setdefault(rec[2], []).append(rec)
    subs_sorted = sorted(pw_by_sub.keys())
    for s in subs_sorted:
        pw_by_sub[s].sort(key=lambda r: r[3])  # deterministic by filename
    print(f"PW staging: {len(pw_all)} total images, {len(pw_nonempty)} non-empty-label across {len(subs_sorted)} sub-colonies")

    # --- enumerate iNat labeled (excluding Boulders) ---
    inat = list_inat_labeled()
    inat.sort(key=lambda r: r[3])
    print(f"iNat labeled (excl Boulders): {len(inat)}")

    # --- TEST: 50 PW round-robin across 16 sub-colonies ---
    pw_test = []
    # round-robin: cycle through sub-colonies, pick one uniform-random each round, until 50
    rng_local = np.random.default_rng(SEED)
    # Build per-sub candidate lists (indices) so we can mark selected
    sub_candidates = {s: list(range(len(pw_by_sub[s]))) for s in subs_sorted}
    selected_per_sub = {s: set() for s in subs_sorted}
    order = list(subs_sorted)
    # ensure we shuffle the order so round-robin doesn't always start at BAILa deterministically
    order_idx = list(range(len(order)))
    rng_local.shuffle(order_idx)
    order = [order[i] for i in order_idx]
    need = args.pw_test_n
    rounds = 0
    while len(pw_test) < need:
        progressed = False
        for s in order:
            if len(pw_test) >= need:
                break
            avail = [i for i in sub_candidates[s] if i not in selected_per_sub[s]]
            if not avail:
                continue
            pick = int(rng_local.integers(0, len(avail)))
            idx = avail[pick]
            selected_per_sub[s].add(idx)
            pw_test.append(pw_by_sub[s][idx])
            progressed = True
        if not progressed:
            break
        rounds += 1
        if rounds > 1000:
            break
    print(f"PW test selected: {len(pw_test)}; per-sub distribution: "
          f"{ {s: len(selected_per_sub[s]) for s in subs_sorted} }")

    # --- TEST: 50 iNat uniform ---
    inat_idx = list(range(len(inat)))
    rng_local.shuffle(inat_idx)
    inat_test_idx = inat_idx[:args.inat_test_n]
    inat_test = [inat[i] for i in inat_test_idx]
    print(f"iNat test selected: {len(inat_test)}")

    # mark selected filenames for exclusion from train/val
    pw_test_names = set(r[3] for r in pw_test)
    inat_test_names = set(r[3] for r in inat_test)

    # --- PW train = all non-gold PW (all frames, incl empty-label, minus gold test) ---
    pw_train = [r for r in pw_all if r[3] not in pw_test_names]
    print(f"PW train (non-gold, all frames): {len(pw_train)}")

    # --- iNat non-gold: 90% train / 10% valid (seeded) ---
    inat_rest_idx = [i for i in range(len(inat)) if inat[i][3] not in inat_test_names]
    inat_rest_idx_arr = np.array(inat_rest_idx, dtype=np.int64)
    rng_local.shuffle(inat_rest_idx_arr)
    n_valid_inat = int(round(0.10 * len(inat_rest_idx_arr)))
    inat_valid_idx = inat_rest_idx_arr[:n_valid_inat]
    inat_train_idx = inat_rest_idx_arr[n_valid_inat:]
    inat_valid = [inat[i] for i in inat_valid_idx]
    inat_train = [inat[i] for i in inat_train_idx]
    print(f"iNat train={len(inat_train)} valid={len(inat_valid)} test={len(inat_test)}")

    # --- place files ---
    def place_set(records, split):
        for (src_img, src_lbl, _src_split, fname) in records:
            dst_img = OUT / split / "images" / fname
            dst_lbl = OUT / split / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT)
            place((src_img, src_lbl), dst_img, dst_lbl)

    place_set(pw_train + inat_train, "train")
    place_set(inat_valid, "valid")
    place_set(pw_test + inat_test, "test")

    # counts
    def count(split):
        img_dir = OUT / split / "images"
        lbl_dir = OUT / split / "labels"
        ni = len([f for f in os.listdir(img_dir) if not f.startswith(".")])
        nl = len([f for f in os.listdir(lbl_dir) if not f.startswith(".")])
        return ni, nl
    cnt = {s: count(s) for s in ["train", "valid", "test"]}

    # --- data.yaml ---
    yaml_data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["penguin"],
    }
    with open(OUT / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    # --- manifest ---
    pw_sub_membership = {}
    for rec in pw_all:
        pw_sub_membership[rec[3]] = rec[2]
    manifest = {
        "seed": SEED,
        "test_n": args.pw_test_n + args.inat_test_n,
        "test_pw": [r[3] for r in pw_test],
        "test_inat": [r[3] for r in inat_test],
        "pw_subcolony_membership": pw_sub_membership,
        "excluded_boulderspenguin": BOULDERS,
        "counts": {
            "pw_total": len(pw_all),
            "pw_nonempty_label": len(pw_nonempty),
            "inat_total_excl_boulders": len(inat),
            "pw_test": len(pw_test),
            "inat_test": len(inat_test),
            "pw_train": len(pw_train),
            "inat_train": len(inat_train),
            "inat_valid": len(inat_valid),
            "train_img": cnt["train"][0], "train_lbl": cnt["train"][1],
            "valid_img": cnt["valid"][0], "valid_lbl": cnt["valid"][1],
            "test_img": cnt["test"][0], "test_lbl": cnt["test"][1],
        },
        "pw_test_per_subcolony": {s: len(selected_per_sub[s]) for s in subs_sorted},
        "paths": {
            "root": str(OUT),
            "data_yaml": str(OUT / "data.yaml"),
            "pw_staging": str(PW_STAGING),
            "inat": str(INAT),
        },
    }
    mf = MAN_DIR / "gold_manifest.json"
    with open(mf, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {mf}")
    print(f"Counts: {cnt}")
    print(f"Test PW={len(pw_test)} iNat={len(inat_test)} total={len(pw_test)+len(inat_test)}")

if __name__ == "__main__":
    main()
