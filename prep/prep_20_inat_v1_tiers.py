#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build a boulders-style tiers sidecar for the 50-image iNaturalist half of the v1 gold.

`gold200_tiers.json` holds 200 manifest keys, 50 each from Penguin-Watch, iNaturalist,
SANParks and Boulders. The Penguin-Watch 50 dangle since the 2026-07-31 detach. It is a
tier-assignment sidecar, keyed by absolute path and carrying only idx/tier/score per
detection, while `prep_13_supervised_reeval.py` needs a manifest keyed by image filename
with an RLE mask per instance. Passing the source through unchanged wrote a file that
joined to zero predictions and then crashed on the absent geometry. This rasterises the
Roboflow polygons for the 50 iNaturalist frames and writes the boulders format, as
`prep_19_pw200_tiers.py` does for Penguin-Watch.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

SRC = os.path.expanduser("~/bench_any/runs/p5_r1_gold/gold200_tiers.json")
IMGS = Path(os.path.expanduser("~/bench_any/datasets/p5_r1_gold/test/images"))
OUT = os.path.expanduser("~/bench_any/runs/p5_r1_gold/inat_v1_50_tiers.json")
# tab:percorpus_v1, cited by the inat_v1_50 entry in prep_13.CORPORA.
EXPECTED = {"easy": 51, "medium": 331, "hard": 12}


def encode(mask_bool):
    rle = mu.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def polys_from_label(path, w, h):
    out = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        xy = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        out.append(np.round(xy).astype(np.int32))
    return out


def main():
    src = json.load(open(SRC))["manifest"]
    keep = {k: v for k, v in src.items() if Path(k).parent == IMGS and os.path.exists(k)}
    manifest = {}
    counts = {"easy": 0, "medium": 0, "hard": 0}
    dropped = []
    n_inst = 0
    for path, rec in sorted(keep.items()):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        label = Path(path.replace("/images/", "/labels/")).with_suffix(".txt")
        polys = polys_from_label(label, w, h)
        dets = []
        for det in rec:
            i = det["idx"]
            if i >= len(polys):
                dropped.append((Path(path).name, i, "idx past polygon count"))
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [polys[i]], 1)
            if m.sum() == 0:
                dropped.append((Path(path).name, i, "polygon rasterises empty"))
                continue
            ys, xs = np.nonzero(m)
            # XYXY, as prep_13.build_gt_coco converts to COCO XYWH itself.
            dets.append({"idx": i, "tier": det["tier"], "score": det["score"],
                         "rle": encode(m.astype(bool)),
                         "bbox": [float(xs.min()), float(ys.min()),
                                  float(xs.max() + 1), float(ys.max() + 1)],
                         "area": float(m.sum()),
                         "width": w, "height": h})
            counts[det["tier"]] += 1
            n_inst += 1
        manifest[Path(path).name] = dets
    json.dump({"corpus": "inat_v1_50", "tier_counts": counts, "total": n_inst,
               "n_images": len(manifest), "n_instances": n_inst,
               "manifest": manifest}, open(OUT, "w"))
    print(f"images {len(manifest)} instances {n_inst} tiers {counts} -> {OUT}")
    for name, i, why in dropped:
        print(f"  DROPPED {name} idx {i}: {why}")
    if counts != EXPECTED:
        raise SystemExit(f"tier counts {counts} do not match tab:percorpus_v1 {EXPECTED}")


if __name__ == "__main__":
    main()
