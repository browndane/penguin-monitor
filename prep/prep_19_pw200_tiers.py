#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Build a boulders-style tiers sidecar for the 200-image Penguin-Watch v2 gold.

`pw_v2_tiers.json` is keyed by stem and carries only idx/tier/score per detection,
while `prep_13_supervised_reeval.py` expects a manifest keyed by the image filename
with an RLE mask per instance. This rasterises the Roboflow polygons for the 200
gold frames and writes `pw_v2_200_tiers.json` in the boulders format.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

TIERS = os.path.expanduser("~/bench_any/runs/p5_r1_gold/pw_v2_tiers.json")
GOLD = Path(os.path.expanduser("~/bench_any/datasets/p5_r1_gold_pw_v2_200/test/images"))
OUT = os.path.expanduser("~/bench_any/runs/p5_r1_gold/pw_v2_200_tiers.json")


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
    src = json.load(open(TIERS))["images"]
    manifest = {}
    counts = {"easy": 0, "medium": 0, "hard": 0}
    n_inst = 0
    for link in sorted(GOLD.iterdir()):
        stem = link.name[len("PW_"):].split("_", 1)[1].rsplit(".", 1)[0]
        rec = src[stem]
        img = cv2.imread(str(link))
        h, w = img.shape[:2]
        polys = polys_from_label(rec["label"], w, h)
        dets = []
        for det in rec["dets"]:
            i = det["idx"]
            if i >= len(polys):
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [polys[i]], 1)
            if m.sum() == 0:
                continue
            ys, xs = np.nonzero(m)
            dets.append({"idx": i, "tier": det["tier"], "score": det["score"],
                         "rle": encode(m.astype(bool)),
                         "bbox": [float(xs.min()), float(ys.min()),
                                  float(xs.max() + 1), float(ys.max() + 1)],
                         "area": float(m.sum()),
                         "width": w, "height": h})
            counts[det["tier"]] += 1
            n_inst += 1
        manifest[link.name] = dets
    json.dump({"corpus": "pw_v2_200", "tier_counts": counts, "total": n_inst,
               "n_images": len(manifest), "n_instances": n_inst,
               "manifest": manifest}, open(OUT, "w"))
    print(f"images {len(manifest)} instances {n_inst} tiers {counts} -> {OUT}")


if __name__ == "__main__":
    main()
