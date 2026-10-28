#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build the two small-site ablation arms as symlink trees off pw_inat_style.

Companion to the empty-label background-mining ablation (see
/home/user/.claude/plans/ComputerVisionResearchBench/2026-08-03_pw-empty-label-background-mining-ablation.md
"Small-site design" section).

Design (load-bearing):
  Penguin core = the three smallest sites (HALFc, HALFb, DAMOa), ~0% empties.
  Background pool = empty frames from the two large sites (LOCKb, PETEe) only.
  No nonempty frame from LOCKb/PETEe enters any arm, so no penguin signal leaks
  from the large sites. The split each frame lives in is the split prep_25
  already assigned it; we do NOT reshuffle, so the test split is identical
  across arms and disjoint from train.

Two arms, shared eval surface:
  Arm 1 (no-background control):  pw_smallsite_core
      train  = core nonempty frames (HALFc+HALFb+DAMOa nonempty in prep_25 train)
      valid  = core nonempty frames (same sites, prep_25 valid)
  Arm 2 (with-background):        pw_smallsite_withbg
      train  = Arm 1 train  +  LOCKb/PETEe empty frames in prep_25 train
      valid  = Arm 1 valid  (unchanged; fitness ignores empties in both arms)

Test slices are NOT built into the train trees; eval runs against the existing
pw_inat_style test split, partitioned by stratum:
  Obj 1 = 93 core nonempty test frames (HALFc 30 + HALFb 29 + DAMOa 34)
  Obj 2 = 470 LOCKb/PETEe empty test frames (LOCKb 287 + PETEe 183)
The split ids are emitted to <out>/<run_tag>_test_strata_ids.json so the eval
script can subset predictions without a second dataset build.

An empty label is a 0-byte .txt (genuine background frame; verified). Labels are
symlinks to pw_yolo_staging, so size must be read with follow_symlinks=True.

Build method (mirrors prep_26): symlink image+label from pw_inat_style into the
arm dir, resolving to the real staging bytes. No copytree, no reshuffle. The
nonempty core frames are byte-identical to pw_inat_style in the same splits.

Usage:
    python prep_28_pw_smallsite_arms.py            # build both arms
    python prep_28_pw_smallsite_arms.py --check     # assert parity + no leak
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

SRC = Path(os.path.expanduser("~/bench_any/datasets/pw_inat_style"))
OUTROOT = Path(os.path.expanduser("~/bench_any/datasets"))
SPLITS = ("train", "valid", "test")

CORE_SITES = ("HALFc", "HALFb", "DAMOa")
BG_SITES = ("LOCKb", "PETEe")

ARMS = {
    "core": {"out": OUTROOT / "pw_smallsite_core", "with_background": False},
    "withbg": {"out": OUTROOT / "pw_smallsite_withbg", "with_background": True},
}


def site_of(name: str) -> str | None:
    # filenames are PW_<SITE>_rest; site is the second underscore field
    if not name.startswith("PW_"):
        return None
    rest = name[3:]
    return rest.split("_", 1)[0] if "_" in rest else rest


def is_empty_label(lbl: Path) -> bool:
    # 0-byte .txt == background frame. Labels are symlinks; dereference.
    return lbl.is_symlink() and lbl.stat().st_size == 0 or (
        lbl.is_file() and lbl.stat().st_size == 0
    )


def split_files(split: str):
    """Yield (img_name, stem, is_empty, site) for every image in a split."""
    s_img = SRC / split / "images"
    s_lbl = SRC / split / "labels"
    for img in sorted(s_img.iterdir()):
        if img.name.startswith("."):
            continue
        stem = img.stem
        lbl = s_lbl / f"{stem}.txt"
        if not lbl.exists():
            # treat missing label as empty (prep_25 always writes one; be safe)
            empty = True
        else:
            empty = lbl.stat().st_size == 0  # follows symlinks by default
        site = site_of(img.name)
        yield img.name, stem, empty, site


def build_arm(arm_key: str, overwrite: bool = False) -> dict:
    arm = ARMS[arm_key]
    out = arm["out"]
    with_bg = arm["with_background"]
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
        shutil.rmtree(out)

    counts = {s: {"total": 0, "kept": 0, "empty": 0, "core_ne": 0, "bg_empty": 0} for s in SPLITS}
    test_strata = {"nonempty_core": [], "empty_large": []}

    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        s_img = SRC / split / "images"
        s_lbl = SRC / split / "labels"
        for img_name, stem, empty, site in split_files(split):
            counts[split]["total"] += 1
            if empty:
                counts[split]["empty"] += 1

            keep = False
            if site in CORE_SITES and not empty:
                # core nonempty: enters BOTH arms (train/valid) and Obj 1 test slice
                keep = split in ("train", "valid")
                counts[split]["core_ne"] += 1
                if split == "test":
                    test_strata["nonempty_core"].append(stem)
            elif site in BG_SITES and empty:
                # large-site empty background: Arm 2 train only
                counts[split]["bg_empty"] += 1
                if with_bg and split == "train":
                    keep = True
                if split == "test":
                    test_strata["empty_large"].append(stem)
            # everything else (large-site nonempty, core empty) is dropped from train/valid

            if keep:
                oi = out / split / "images" / img_name
                ol = out / split / "labels" / f"{stem}.txt"
                if not oi.exists():
                    oi.symlink_to((s_img / img_name).resolve())
                if not ol.exists():
                    ol.symlink_to((s_lbl / f"{stem}.txt").resolve())
                counts[split]["kept"] += 1

    # test points at the ORIGINAL pw_inat_style test split (1016 frames, both
    # strata), not the arm's empty test/ dir. The arm dirs have no test images
    # of their own; the two objectives are answered by subsetting this real
    # test split's predictions via test_strata_ids.json. Per-epoch fitness rides
    # on the arm's own val (core nonempty, 173), so early stopping ignores
    # empties in both arms. Wrappers eval on test (cfg.get("test") or "val").
    pw_test = SRC / "test" / "images"
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: {pw_test}\n"
        f"nc: 1\n"
        f"names: ['penguin']\n"
    )
    # test strata ids are shared across arms; write once per arm dir for convenience
    (out / "test_strata_ids.json").write_text(json.dumps(test_strata, indent=2))
    return {"arm": arm_key, "out": str(out), "with_bg": with_bg, "counts": counts, "test_strata": {k: len(v) for k, v in test_strata.items()}}


def check() -> None:
    """Parity + no-leak guarantees the ablation's comparability rests on:
      - no LOCKb/PETEe nonempty frame in any arm's train/valid
      - no empty frame in core arm's train/valid
      - core arm's train/valid are a subset of withbg's (core is the control)
      - test strata sizes match the plan (93 nonempty core, 470 empty large)
    """
    bad = 0
    for arm_key, arm in ARMS.items():
        out = arm["out"]
        if not out.is_dir():
            raise SystemExit(f"{arm_key}: {out} not built")
        for split in ("train", "valid"):
            d = out / split / "images"
            for img in d.iterdir():
                if img.name.startswith("."):
                    continue
                site = site_of(img.name)
                lbl = out / split / "labels" / f"{img.stem}.txt"
                is_empty = lbl.stat().st_size == 0 if lbl.exists() else True
                if site in BG_SITES and not is_empty:
                    print(f"  PARITY FAIL {arm_key}/{split}: large-site NONEMPTY leaked: {img.name}")
                    bad += 1
                if arm_key == "core" and is_empty:
                    print(f"  PARITY FAIL {arm_key}/{split}: empty frame in no-background arm: {img.name}")
                    bad += 1
    # core train/valid must be a subset of withbg train/valid
    for split in ("train", "valid"):
        core = {p.name for p in (ARMS["core"]["out"] / split / "images").iterdir() if not p.name.startswith(".")}
        bg = {p.name for p in (ARMS["withbg"]["out"] / split / "images").iterdir() if not p.name.startswith(".")}
        if not core.issubset(bg):
            leaked = core - bg
            print(f"  PARITY FAIL: core/{split} not subset of withbg/{split}: {len(leaked)} frames")
            bad += len(leaked)
    # test strata sizes
    ts = json.loads((ARMS["core"]["out"] / "test_strata_ids.json").read_text())
    if len(ts["nonempty_core"]) != 93:
        print(f"  PARITY FAIL: Obj 1 test slice = {len(ts['nonempty_core'])}, expected 93")
        bad += 1
    if len(ts["empty_large"]) != 470:
        print(f"  PARITY FAIL: Obj 2 test slice = {len(ts['empty_large'])}, expected 470")
        bad += 1
    if bad:
        raise SystemExit(f"parity check FAILED: {bad} violations")
    print("[check] parity OK: no large-site nonempty leaked, no empty in core arm, "
          "core subset of withbg, test strata 93/470")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    for arm_key in ("core", "withbg"):
        r = build_arm(arm_key, overwrite=args.overwrite)
        c = r["counts"]
        print(f"\n[built] {r['out']}  (with_background={r['with_bg']})")
        print(f"{'split':<7} {'kept':>6} {'core_ne':>8} {'bg_empty':>9}")
        for s in SPLITS:
            print(f"{s:<7} {c[s]['kept']:>6} {c[s]['core_ne']:>8} {c[s]['bg_empty']:>9}")
        print(f"  test_strata: {r['test_strata']}")
    print("\nrun --check to verify parity")


if __name__ == "__main__":
    main()
