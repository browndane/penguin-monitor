#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Obj-2: SAM2-prompted COUNT comparison on iNat test, per density bucket.

Uses the SAM2-prompted masks already on disk (sam3_sam2refine/sam2_masks.json,
856 iNat test images, 6807 preds) plus the iNat test GT. Runs count_decompose
per image, aggregates by density bucket (1, 2-5, 6-20, 21+), writes a row to
sit beside the supervised models in inat_fixed05_table.txt. CPU only; no GPU.
Self-reports to FINDINGS section 15 + touches _inat_sam2_count_done.
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from PIL import Image
from pycocotools import mask as mu

from count_decompose import _decompose_image

RUNS = Path(os.path.expanduser("~/bench_any/runs/obj2_counting"))
D = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")
INAT_IMG = D / "african-penguin-inaturalist-sam-veryclean" / "test" / "images"
INAT_LBL = D / "african-penguin-inaturalist-sam-veryclean" / "test" / "labels"
SAM2_MASKS = RUNS / "sam3_sam2refine" / "sam2_masks.json"
FINDINGS = RUNS / "FINDINGS.md"
SENTINEL = RUNS / "_inat_sam2_count_done"

# density buckets per the first-pass density fit
def bucket(n):
    if n == 1: return "1"
    if 2 <= n <= 5: return "2-5"
    if 6 <= n <= 20: return "6-20"
    return "21+"


def load_gt():
    """{fn: [rle]} from YOLO labels."""
    out = {}
    for lf in sorted(INAT_LBL.glob("*.txt")):
        imgs = list(INAT_IMG.glob(lf.stem + ".*"))
        if not imgs:
            continue
        w, h = Image.open(imgs[0]).size
        rles = []
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            xy = [float(v) for v in p[1:]]
            poly = [xy[i] * (w if i % 2 == 0 else h) for i in range(len(xy))]
            rles.append(mu.merge(mu.frPyObjects([poly], h, w)))
        out[imgs[0].name] = rles
    return out


def main():
    log = lambda m: print(m, flush=True)
    log("=== inat sam2 count start ===")
    truth = load_gt()
    import re
    # SAM2 masks carry hashed image_ids: int(concatenated digits of filename).
    # e.g. "101388612_169319975.jpg" -> 101388612169319975. Rebuild id->fn.
    files = sorted(INAT_IMG.glob("*.jpg"))
    fn_to_id = {}
    for f in files:
        try:
            fn_to_id[f.name] = int("".join(re.findall(r"\d+", f.name)))
        except ValueError:
            continue
    iid_to_fn = {v: k for k, v in fn_to_id.items()}
    preds = json.loads(SAM2_MASKS.read_text())
    log(f"mapped {len(iid_to_fn)} ids; {len(set(a['image_id'] for a in preds) & set(iid_to_fn))} "
        f"of {len(set(a['image_id'] for a in preds))} mask ids covered")
    # bucket preds by image
    per_fn = defaultdict(list)
    for p in preds:
        fn = iid_to_fn.get(p["image_id"])
        if fn is None:
            continue
        s = p["segmentation"]
        r = dict(s)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode("utf-8")
        per_fn[fn].append(r)
    log(f"iNat test: {len(truth)} GT images, {sum(len(v) for v in truth.values())} GT instances")
    log(f"SAM2-prompted preds: {len(preds)} over {len(per_fn)} images")

    # aggregate per density bucket
    agg = defaultdict(lambda: defaultdict(float))
    n_img = defaultdict(int)
    ngt = defaultdict(int)
    total_mae = 0
    total_sgn = 0
    all_missed = 0
    all_spurious = 0
    all_gt = 0
    for fn, gts in truth.items():
        b = bucket(len(gts))
        p = per_fn.get(fn, [])
        ev = _decompose_image(p, gts, 0.5, 0.5)
        ev.pop("_fate", None)
        for k, v in ev.items():
            agg[b][k] += v
        n_img[b] += 1
        ngt[b] += len(gts)
        total_mae += abs(len(p) - len(gts))
        total_sgn += len(p) - len(gts)
        all_missed += ev["missed"]
        all_spurious += ev["spurious"]
        all_gt += len(gts)

    n = len(truth)
    overall = {"MAE": total_mae / n, "signed": total_sgn / n,
               "missed_pct": 100.0 * all_missed / all_gt,
               "spurious_pct": 100.0 * all_spurious / all_gt}
    log(f"overall: MAE={overall['MAE']:.2f} signed={overall['signed']:+.2f} "
        f"missed={overall['missed_pct']:.1f}% spurious={overall['spurious_pct']:.1f}%")

    out = ["\n## 15. SAM 2-prompted counting on iNat test (Obj-2 density paper)\n\n",
           "SAM 3 boxes prompted SAM 2 for masks on the 856-image iNat test split; ",
           "count decomposition against GT, per density bucket. This lands the ",
           "zero-shot SAM 2-prompted row beside the six supervised models on the ",
           "density curve. CPU pass over masks already on disk; no new GPU run.\n\n",
           "| bucket | n_imgs | n_gt | matched | missed | merged | split | spurious | signed bias |\n",
           "|---|---|---|---|---|---|---|---|---|\n"]
    for b in ("1", "2-5", "6-20", "21+"):
        a = agg[b]
        sb = (a.get("spurious", 0) - a.get("missed", 0)) / ngt[b] if ngt[b] else 0.0
        out.append(f"| {b} | {n_img[b]} | {ngt[b]} | {int(a.get('matched',0))} | "
                   f"{int(a.get('missed',0))} | {int(a.get('merged',0))} | "
                   f"{int(a.get('split',0))} | {int(a.get('spurious',0))} | {sb:+.2f} |\n")
    out.append(f"\nOverall (all buckets): MAE {overall['MAE']:.2f}, signed bias "
               f"{overall['signed']:+.2f}, missed {overall['missed_pct']:.1f}%, "
               f"spurious {overall['spurious_pct']:.1f}%.\n")
    out.append("\nCompare against the supervised rows in `inat_fixed05_table.txt`: "
               "SAM 3 zero-shot MAE 1.01 (signed -0.54, missed 9.6%, spurious 4.8%). "
               "The SAM 2-prompted row shows whether mask refinement changes the "
               "count picture versus the SAM 3 box count.\n")
    with open(FINDINGS, "a") as f:
        f.write("".join(out))
    SENTINEL.write_text("done\n")
    log("=== inat sam2 count complete ===")


if __name__ == "__main__":
    main()
