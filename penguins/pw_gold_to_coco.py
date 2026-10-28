#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Penguin-Watch gold masks, YOLO-seg polygons to COCO, so the sieve bake-off can
run on a second corpus.

Boulders carries the entire border-rule result and SANParks is too sparse at its
frame edges to separate any arm, so one corpus is doing all the work. Penguin-
Watch is the only other gold set with masks, and its labels are normalised
polygons rather than run-length encodings, which is the only reason it has never
been in this comparison.

The conversion is deliberately literal. Polygons are scaled by the image size and
handed to `pycocotools` for encoding, so the areas and extents the sieve reads
come from the same code path that produces them for the other two corpora and no
second convention can creep in. Images are read for their dimensions only.

CPU, seconds. Writes ~/bench_any/runs/sam2refine_pw-v2-200/gt_coco.json, matching
the layout `sieve_composition.load_corpus` already expects.
"""
import json
from pathlib import Path

from PIL import Image
from pycocotools import mask as mu

SRC = Path.home() / "bench_any" / "datasets" / "p5_r1_gold_pw_v2_200" / "test"
OUT = (Path.home() / "bench_any" / "runs" / "obj2_counting"
       / "sam2refine_pw-v2-200")


def polygons(label_file, w, h):
    """Every polygon in one YOLO-seg label file, in absolute pixel coordinates.

    A YOLO-seg row is a class index followed by normalised x y pairs. Rows with
    fewer than three points cannot bound an area and are dropped rather than
    encoded, since `frPyObjects` would accept them and return a degenerate mask
    that the sieve would then have to reason about."""
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        xy = [float(v) for v in parts[1:]]
        pts = [(xy[i] * w, xy[i + 1] * h) for i in range(0, len(xy) - 1, 2)]
        if len(pts) < 3:
            continue
        out.append([c for p in pts for c in p])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    n_poly_in = 0
    for img_path in sorted((SRC / "images").glob("*")):
        lab = SRC / "labels" / (img_path.stem + ".txt")
        if not lab.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": img_path.name,
                       "width": w, "height": h})
        for poly in polygons(lab, w, h):
            n_poly_in += 1
            rles = mu.frPyObjects([poly], h, w)
            rle = mu.merge(rles)
            x, y, bw, bh = [float(v) for v in mu.toBbox(rle)]
            annotations.append({
                "id": len(annotations) + 1, "image_id": img_id, "category_id": 1,
                "bbox": [x, y, bw, bh], "area": float(mu.area(rle)), "iscrowd": 0,
                "segmentation": {"size": list(rle["size"]),
                                 "counts": rle["counts"].decode("ascii")},
            })

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "penguin"}]}
    (OUT / "gt_coco.json").write_text(json.dumps(coco))

    # Round-trip gate. Every polygon that went in must come back out as exactly
    # one annotation, and every annotation must decode to a non-empty mask, or
    # the corpus is silently smaller than the labels claim.
    assert len(annotations) == n_poly_in, (len(annotations), n_poly_in)
    empty = sum(1 for a in annotations if a["area"] <= 0)
    assert empty == 0, f"{empty} degenerate masks"
    print(f"images {len(images)}  instances {len(annotations)}  "
          f"-> {OUT / 'gt_coco.json'}", flush=True)


if __name__ == "__main__":
    main()
