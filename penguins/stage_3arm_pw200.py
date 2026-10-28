#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Stage the 200 PW v2-gold frames + Oxford citizen-science clicks into the
folder layout the two generator scripts expect:

    <STAGE>/<site>/<site>.json     (Oxford clicks, filtered to the 200 GT imNames)
    <STAGE>/<site>/<imName>.JPG    (symlink to the v2-gold image)

Run on each host before launching point_to_sam21_coco.py (SAM2) or
point_to_sam3_coco.py (SAM3).  Idempotent: re-running only rewrites JSONs.

Usage:  python3 penguins/stage_3arm_pw200.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

GT = Path.home() / "bench_any/runs/obj2_counting/sam2refine_pw-v2-200/gt_coco.json"
# Image dirs searched in order; the 200 v2-gold frames span several split dirs.
IMG_DIRS = [
    Path.home() / "bench_any/datasets/pw_smallsite_withbg/train/images",
    Path.home() / "bench_any/datasets/pw_smallsite_withbg/valid/images",
    Path.home() / "bench_any/datasets/pw_smallsite_withbg/test/images",
    Path.home() / "bench_any/datasets/pw_inat_style/train/images",
    Path.home() / "bench_any/datasets/pw_inat_style/valid/images",
    Path.home() / "bench_any/datasets/pw_inat_style/test/images",
    Path.home() / "bench_any/datasets/pw_yolo_staging/PETEe/images",
    Path.home() / "bench_any/datasets/p5_r3_groupaware/train/images",
]
OXFORD = Path("/tmp/oxford_pw_annotations/CompleteAnnotations_2016-07-11")
STAGE = Path.home() / "bench_any/runs/p5_r1_gold/pw200_zeroshot/stage_3arm"
SITES = ["DAMOa", "HALFb", "HALFc", "LOCKb", "PETEe"]


def main() -> None:
    gt = json.loads(GT.read_text())
    # Map site -> set of imNames in the 200-frame GT set.
    by_site: dict[str, set[str]] = {s: set() for s in SITES}
    for im in gt["images"]:
        parts = im["file_name"].replace(".jpg", "").split("_")
        site, imName = parts[1], "_".join(parts[2:])
        if site in by_site:
            by_site[site].add(imName)

    STAGE.mkdir(parents=True, exist_ok=True)
    total_json = total_img = 0
    for site in SITES:
        sdir = STAGE / site
        sdir.mkdir(parents=True, exist_ok=True)
        ox = json.loads((OXFORD / f"{site}.json").read_text())
        gt_names = by_site[site]
        # Keep only dots entries whose imName is in the 200-frame GT set.
        kept = [e for e in ox["dots"] if e["imName"] in gt_names]
        (sdir / f"{site}.json").write_text(json.dumps({"dots": kept}))
        total_json += len(kept)
        # Symlink images as <imName>.JPG (generators append .JPG).
        for imName in gt_names:
            dst = sdir / f"{imName}.JPG"
            if dst.exists():
                total_img += 1
                continue
            src = None
            for d in IMG_DIRS:
                cand = d / f"PW_{site}_{imName}.jpg"
                if cand.exists():
                    src = cand
                    break
            if src is None:
                print(f"WARN missing image PW_{site}_{imName}.jpg", flush=True)
                continue
            dst.symlink_to(src)
            total_img += 1
        print(f"{site}: {len(kept)} click entries, {len(gt_names)} images", flush=True)

    print(f"STAGE done: {total_json} click entries, {total_img} images at {STAGE}", flush=True)


if __name__ == "__main__":
    main()
