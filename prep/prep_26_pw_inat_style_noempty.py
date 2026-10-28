#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build pw_inat_style_noempty: pw_inat_style with empty-label pairs removed.

Companion to the empty-label background-mining ablation (see
/home/user/.claude/plans/ComputerVisionResearchBench/2026-08-03_pw-empty-label-background-mining-ablation.md).

Design constraint (load-bearing): the derivative must share the EXACT same
nonempty images in the EXACT same splits as pw_inat_style. We achieve this by
copying pw_inat_style's directory tree and unlinking only the empty-label pairs.
We do NOT re-run prep_25 with a filter, because that reshuffles val/test and
breaks the byte-identical control on the nonempty images.

An empty label is a 0-byte .txt (genuine background frame; verified: contiguous
sequential runs at LOCKb/PETEe, consumed as Detections.empty() by rfdetr/yolo).

Usage:
    python prep_26_pw_inat_style_noempty.py            # build
    python prep_26_pw_inat_style_noempty.py --check    # assert split parity
"""
from __future__ import annotations
import argparse, shutil, os
from pathlib import Path

SRC = Path(os.path.expanduser("~/bench_any/datasets/pw_inat_style"))
OUT = Path(os.path.expanduser("~/bench_any/datasets/pw_inat_style_noempty"))
SPLITS = ("train", "valid", "test")


def is_empty_label(lbl: Path) -> bool:
    # 0-byte .txt == background frame. (size 0, not "missing"; missing is skipped
    # by prep_25 and never reaches pw_inat_style.)
    return lbl.is_file() and lbl.stat().st_size == 0


def build(overwrite: bool = False) -> None:
    if OUT.exists():
        if not overwrite:
            raise SystemExit(f"{OUT} exists; pass --overwrite to rebuild")
        shutil.rmtree(OUT)
    if not SRC.is_dir():
        raise SystemExit(f"source missing: {SRC}")

    src_counts: dict[str, dict[str, int]] = {}
    out_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        s_img = SRC / split / "images"
        s_lbl = SRC / split / "labels"
        if not s_img.is_dir() or not s_lbl.is_dir():
            raise SystemExit(f"source split incomplete: {split}")
        o_img = OUT / split / "images"
        o_lbl = OUT / split / "labels"
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)
        src_counts[split] = {"total": 0, "empty": 0, "kept": 0}
        out_counts[split] = {"total": 0, "empty": 0, "kept": 0}
        # copytree the whole split subtree once, then prune empties in place.
        for entry in s_img.iterdir():
            if entry.name.startswith("."):
                continue
            stem = entry.stem
            lbl = s_lbl / f"{stem}.txt"
            src_counts[split]["total"] += 1
            if is_empty_label(lbl):
                src_counts[split]["empty"] += 1
                continue  # drop the empty pair
            # keep: symlink image + label (resolve to the real staging files)
            oi = o_img / entry.name
            ol = o_lbl / lbl.name
            if not oi.exists():
                oi.symlink_to(entry.resolve())
            if not ol.exists():
                ol.symlink_to(lbl.resolve())
            src_counts[split]["kept"] += 1
            out_counts[split]["kept"] += 1

    # data.yaml mirrors prep_25 exactly except the path
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: 1\n"
        f"names: ['penguin']\n"
    )
    print(f"[built] {OUT}")
    print(f"{'split':<7} {'src_total':>9} {'src_empty':>9} {'kept':>6}")
    tot_src = tot_empty = tot_kept = 0
    for split in SPLITS:
        c = src_counts[split]
        print(f"{split:<7} {c['total']:>9} {c['empty']:>9} {c['kept']:>6}")
        tot_src += c["total"]; tot_empty += c["empty"]; tot_kept += c["kept"]
    print(f"{'TOTAL':<7} {tot_src:>9} {tot_empty:>9} {tot_kept:>6}")


def check() -> None:
    """Assert every nonempty image in SRC appears in the same split in OUT.
    This is the parity guarantee the ablation's comparability rests on."""
    if not OUT.is_dir():
        raise SystemExit(f"{OUT} not built; run without --check first")
    bad = 0
    for split in SPLITS:
        s_img = SRC / split / "images"
        s_lbl = SRC / split / "labels"
        o_img = OUT / split / "images"
        o_lbl = OUT / split / "labels"
        for entry in s_img.iterdir():
            if entry.name.startswith("."):
                continue
            lbl = s_lbl / f"{entry.stem}.txt"
            if is_empty_label(lbl):
                # must be ABSENT from OUT
                if (o_img / entry.name).exists() or (o_lbl / lbl.name).exists():
                    print(f"  PARITY FAIL {split}: empty pair kept in OUT: {entry.name}")
                    bad += 1
            else:
                # must be PRESENT in OUT in the same split
                if not (o_img / entry.name).exists() or not (o_lbl / lbl.name).exists():
                    print(f"  PARITY FAIL {split}: nonempty pair missing from OUT: {entry.name}")
                    bad += 1
    if bad:
        raise SystemExit(f"parity check FAILED: {bad} mismatches")
    print("[check] parity OK: every nonempty image is in the same split; no empty image leaked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        build(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
