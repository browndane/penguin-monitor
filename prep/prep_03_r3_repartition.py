#!/usr/bin/env python
"""Script 3: R3 group-aware + random-sensitivity repartition (PW only).

3a. Group-aware ~/bench_any/datasets/p5_r3_groupaware/:
    permutation(16) -> 8 train / 3 val / 5 test WHOLE sub-colonies.
    Copy (symlink) PW YOLO images+labels (from script-1 staging) into
    train/val/test by sub-colony membership. Write data.yaml + manifest.

3b. Random-sensitivity ~/bench_any/datasets/p5_r3_random/:
    random per-image 70/15/15 over ALL 81,941 PW frames (seed 42),
    ignoring sub-colony. Write data.yaml + manifest.

Seed = numpy.random.default_rng(seed=42) for BOTH (same seed per spec).
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import yaml

PW_STAGING = Path(os.path.expanduser("~/bench_any/datasets/pw_yolo_staging"))
OUT_GA = Path(os.path.expanduser("~/bench_any/datasets/p5_r3_groupaware"))
OUT_RN = Path(os.path.expanduser("~/bench_any/datasets/p5_r3_random"))
MAN_GA = Path(os.path.expanduser("~/bench_any/runs/p5_r3_groupaware"))
MAN_RN = Path(os.path.expanduser("~/bench_any/runs/p5_r3_random"))
SEED = 42
LABEL_EXT = ".txt"
IMG_EXT = ".jpg"

def list_pw_staging():
    """Return list of (image_path, label_path, subcolony, filename) for PW staging."""
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
            out.append((sub_img_dir / img, lbl, sub, img))
    return out

def place(src_img, src_lbl, dst_img, dst_lbl):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not dst_img.exists():
        os.symlink(src_img, dst_img)
    if not dst_lbl.exists():
        if src_lbl.exists():
            import shutil
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("")

def write_yaml(root: Path):
    yaml_data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["penguin"],
    }
    with open(root / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

def count_split(root, split):
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    ni = len([f for f in os.listdir(img_dir) if not f.startswith(".")]) if img_dir.is_dir() else 0
    nl = len([f for f in os.listdir(lbl_dir) if not f.startswith(".")]) if lbl_dir.is_dir() else 0
    return ni, nl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga_train", type=int, default=8)
    ap.add_argument("--ga_val", type=int, default=3)
    ap.add_argument("--ga_test", type=int, default=5)
    ap.add_argument("--rn_train_pct", type=float, default=0.70)
    ap.add_argument("--rn_val_pct", type=float, default=0.15)
    args = ap.parse_args()

    pw = list_pw_staging()
    print(f"PW staging total: {len(pw)}")
    # group by subcolony
    pw_by_sub = {}
    for rec in pw:
        pw_by_sub.setdefault(rec[2], []).append(rec)
    subs = sorted(pw_by_sub.keys())
    print(f"Sub-colonies: {len(subs)} -> {subs}")
    assert len(subs) == 16, f"expected 16 sub-colonies, got {len(subs)}"

    # =================== 3a GROUP-AWARE ===================
    print("\n=== 3a Group-aware ===")
    OUT_GA.mkdir(parents=True, exist_ok=True)
    MAN_GA.mkdir(parents=True, exist_ok=True)
    for s in ["train", "valid", "test"]:
        (OUT_GA / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT_GA / s / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(16)
    n_train = args.ga_train; n_val = args.ga_val; n_test = args.ga_test
    assert n_train + n_val + n_test == 16, "train+val+test must equal 16"
    train_subs = [subs[perm[i]] for i in range(0, n_train)]
    val_subs = [subs[perm[i]] for i in range(n_train, n_train + n_val)]
    test_subs = [subs[perm[i]] for i in range(n_train + n_val, 16)]
    print(f"train_subs ({len(train_subs)}): {train_subs}")
    print(f"val_subs ({len(val_subs)}): {val_subs}")
    print(f"test_subs ({len(test_subs)}): {test_subs}")

    # place
    for s in train_subs:
        for (src_img, src_lbl, sub, fname) in pw_by_sub[s]:
            place(src_img, src_lbl, OUT_GA / "train" / "images" / fname,
                  OUT_GA / "train" / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT))
    for s in val_subs:
        for (src_img, src_lbl, sub, fname) in pw_by_sub[s]:
            place(src_img, src_lbl, OUT_GA / "valid" / "images" / fname,
                  OUT_GA / "valid" / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT))
    for s in test_subs:
        for (src_img, src_lbl, sub, fname) in pw_by_sub[s]:
            place(src_img, src_lbl, OUT_GA / "test" / "images" / fname,
                  OUT_GA / "test" / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT))

    write_yaml(OUT_GA)
    ga_counts = {s: count_split(OUT_GA, s) for s in ["train", "valid", "test"]}
    print(f"Group-aware counts: {ga_counts}")
    ga_manifest = {
        "seed": SEED,
        "train_subs": train_subs,
        "val_subs": val_subs,
        "test_subs": test_subs,
        "counts": {
            "train_img": ga_counts["train"][0], "train_lbl": ga_counts["train"][1],
            "valid_img": ga_counts["valid"][0], "valid_lbl": ga_counts["valid"][1],
            "test_img": ga_counts["test"][0], "test_lbl": ga_counts["test"][1],
            "total_img": sum(c[0] for c in ga_counts.values()),
        },
        "paths": {"root": str(OUT_GA), "data_yaml": str(OUT_GA / "data.yaml")},
    }
    with open(MAN_GA / "group_split_manifest.json", "w") as f:
        json.dump(ga_manifest, f, indent=2)
    print(f"Manifest -> {MAN_GA / 'group_split_manifest.json'}")

    # =================== 3b RANDOM-SENSITIVITY ===================
    print("\n=== 3b Random-sensitivity ===")
    OUT_RN.mkdir(parents=True, exist_ok=True)
    MAN_RN.mkdir(parents=True, exist_ok=True)
    for s in ["train", "valid", "test"]:
        (OUT_RN / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT_RN / s / "labels").mkdir(parents=True, exist_ok=True)
    rng2 = np.random.default_rng(SEED)
    idx = np.arange(len(pw))
    rng2.shuffle(idx)
    n_train = int(round(args.rn_train_pct * len(pw)))
    n_val = int(round(args.rn_val_pct * len(pw)))
    n_test = len(pw) - n_train - n_val
    print(f"random split: train={n_train} val={n_val} test={n_test} total={len(pw)}")
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    for i in train_idx:
        (src_img, src_lbl, sub, fname) = pw[i]
        place(src_img, src_lbl, OUT_RN / "train" / "images" / fname,
              OUT_RN / "train" / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT))
    for i in val_idx:
        (src_img, src_lbl, sub, fname) = pw[i]
        place(src_img, src_lbl, OUT_RN / "valid" / "images" / fname,
              OUT_RN / "valid" / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT))
    for i in test_idx:
        (src_img, src_lbl, sub, fname) = pw[i]
        place(src_img, src_lbl, OUT_RN / "test" / "images" / fname,
              OUT_RN / "test" / "labels" / (os.path.splitext(fname)[0] + LABEL_EXT))
    write_yaml(OUT_RN)
    rn_counts = {s: count_split(OUT_RN, s) for s in ["train", "valid", "test"]}
    print(f"Random counts: {rn_counts}")
    rn_manifest = {
        "seed": SEED,
        "pct": {"train": args.rn_train_pct, "val": args.rn_val_pct, "test": 1.0 - args.rn_train_pct - args.rn_val_pct},
        "counts": {
            "train_img": rn_counts["train"][0], "train_lbl": rn_counts["train"][1],
            "valid_img": rn_counts["valid"][0], "valid_lbl": rn_counts["valid"][1],
            "test_img": rn_counts["test"][0], "test_lbl": rn_counts["test"][1],
            "total_img": sum(c[0] for c in rn_counts.values()),
        },
        "paths": {"root": str(OUT_RN), "data_yaml": str(OUT_RN / "data.yaml")},
    }
    with open(MAN_RN / "random_split_manifest.json", "w") as f:
        json.dump(rn_manifest, f, indent=2)
    print(f"Manifest -> {MAN_RN / 'random_split_manifest.json'}")

if __name__ == "__main__":
    main()
