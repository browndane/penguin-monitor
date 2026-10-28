#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Score difficulty tiers over the Roboflow user-verified Penguin-Watch export.

Same definition as prep_09 (paper L170-179), reimplemented against YOLO-seg polygons
because the verified tree never enters FiftyOne:

    D_i = 0.40*S_i + 0.35*O_i + 0.25*Dens_i
    S_i    = 1 - min(Area_i / (0.04*W*H), 1)
    O_i    = max_{j!=i} IoU(bbox_i, bbox_j)
    Dens_i = |{j : ||c_j - c_i|| <= 3*sqrt(Area_i)}| / N

Tier cuts are the paper's fixed constants, not re-fitted: easy < 0.30, medium < 0.58,
hard above. Writes one record per image with its per-instance tiers.

    python prep_15_pw_v2_tiers.py
"""
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

VERIFIED = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch")
SITES = ["DAMOa", "HALFb", "HALFc", "LOCKb", "PETEe"]
OUT = Path.home() / "bench_any/runs/p5_r1_gold/pw_v2_tiers.json"
W_SIZE, W_OVER, W_DENS = 0.40, 0.35, 0.25
EASY_MAX, MED_MAX = 0.30, 0.58


def tier_of(score):
    if score < EASY_MAX:
        return "easy"
    if score < MED_MAX:
        return "medium"
    return "hard"


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def bbox_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score_image(label_path, w, h):
    """Return a list of {tier, score} in label-file order, or [] if unlabelled."""
    areas, bboxes, centers = [], [], []
    for line in Path(label_path).read_text().split("\n"):
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.array([float(v) for v in parts[1:]], dtype=float).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        areas.append(polygon_area(coords))
        x0, y0 = coords[:, 0].min(), coords[:, 1].min()
        x1, y1 = coords[:, 0].max(), coords[:, 1].max()
        bboxes.append([x0, y0, x1, y1])
        centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    if not areas:
        return []

    n = float(len(areas))
    cen = np.array(centers)
    out = []
    for i, area in enumerate(areas):
        s = 1.0 - min(area / (0.04 * w * h), 1.0)
        o = max((bbox_iou(bboxes[i], bboxes[j]) for j in range(len(areas)) if j != i),
                default=0.0)
        r = 3.0 * np.sqrt(area)
        dens = float((np.linalg.norm(cen - cen[i], axis=1) <= r).sum()) / n
        d = W_SIZE * s + W_OVER * o + W_DENS * dens
        out.append({"idx": i, "tier": tier_of(d), "score": round(d, 5)})
    return out


def main():
    records, counts = {}, Counter()
    for site in SITES:
        for split in ("train", "valid", "test"):
            for img in sorted(glob.glob(str(VERIFIED / site / split / "images" / "*"))):
                lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                if not os.path.exists(lbl):
                    continue
                with Image.open(img) as im:
                    w, h = im.size
                dets = score_image(lbl, w, h)
                stem = os.path.basename(img).split("_JPG.rf.")[0]
                records[stem] = {"site": site, "split": split, "image": img,
                                 "label": lbl, "dets": dets}
                counts.update(d["tier"] for d in dets)
        print(f"{site} done, {len(records)} images so far", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"tier_cuts": [EASY_MAX, MED_MAX],
                               "totals": dict(counts),
                               "images": records}, indent=1))
    print("images:", len(records), "instances:", sum(counts.values()))
    print("tiers:", dict(counts))
    print("written:", OUT)


if __name__ == "__main__":
    main()
