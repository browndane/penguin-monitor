"""Compute paper difficulty tiers (L170-179) for p5_gold200 GT -> SIDECAR JSON manifest.

D_i = 0.40*S_i + 0.35*O_i + 0.25*D_i
  S_i = 1 - min(Area_i / (0.04 * W * H), 1)         (size; smaller = harder)
  O_i = max_{j!=i} IoU(b_i, b_j)                    (overlap; via mask bbox)
  D_i = (1/N) * |{j : ||c_j - c_i||_2 <= r_i}|      (local density; N = per-image count)
  r_i = 3 * sqrt(Area_i)

Tiers: Easy < 0.30, Medium 0.30-0.58, Hard >= 0.58  (~40/40/20 split).

SIDECAR design (no FiftyOne mask writes): writes ~/bench_any/runs/p5_r1_gold/gold200_tiers.json
keyed by sample filepath -> [{"idx":det_index, "tier":"easy"|"medium"|"hard", "score":float}, ...].
The eval (prep_10) reads tiers from this manifest; GT masks stay untouched in FiftyOne.
"""
import json, os
from pathlib import Path
import numpy as np
import fiftyone as fo

DS = "p5_gold200"
MAN = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold/gold200_tiers.json"))
W_OVER, O_OVER, D_OVER = 0.40, 0.35, 0.25
EASY_MAX, MED_MAX = 0.30, 0.58


def mask_bbox_abs(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)]


def bbox_iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1-ix0); ih = max(0.0, iy1-iy0)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def tier_of(score):
    if score < EASY_MAX: return "easy"
    elif score < MED_MAX: return "medium"
    else: return "hard"


def main():
    ds = fo.load_dataset(DS)
    if "metadata" not in ds.get_field_schema() or ds.first().metadata is None:
        print("computing metadata..."); ds.compute_metadata()
    tier_counts = {"easy":0,"medium":0,"hard":0}
    manifest = {}
    n_imgs = len(ds)
    for i, s in enumerate(ds.iter_samples()):
        meta = s.metadata
        w, h = meta.width, meta.height
        gt = s.ground_truth
        if gt is None or not gt.detections:
            manifest[s.filepath] = []
            if i % 20 == 0: print(f"  {i}/{n_imgs} (no GT)", flush=True)
            continue
        dets = gt.detections
        # per-det info
        areas = []; bboxes = []; centers = []
        for d in dets:
            m = d.mask
            area = float(m.sum()) if m is not None else 0.0
            areas.append(area)
            bb_abs = mask_bbox_abs(m) if m is not None else None
            bboxes.append(bb_abs)
            if bb_abs is not None:
                cx = (bb_abs[0]+bb_abs[2])/2.0; cy = (bb_abs[1]+bb_abs[3])/2.0
            else:
                br = d.bounding_box; cx = (br[0]+br[2]/2)*w; cy = (br[1]+br[3]/2)*h
            centers.append((cx, cy))
        N = float(len(dets))
        per_det = []
        for idx, d in enumerate(dets):
            area = areas[idx]
            S_i = 1.0 - min(area/(0.04*w*h), 1.0) if (w*h)>0 else 1.0
            # overlap
            O_i = 0.0
            bi = bboxes[idx]
            if bi is not None:
                for j in range(len(dets)):
                    if j == idx: continue
                    bj = bboxes[j]
                    if bj is None: continue
                    O_i = max(O_i, bbox_iou(bi, bj))
            # density
            r_i = 3*np.sqrt(area) if area > 0 else 0.0
            cx, cy = centers[idx]
            cnt = 0
            for j in range(len(dets)):
                if j == idx: continue
                dist = np.hypot(centers[j][0]-cx, centers[j][1]-cy)
                if dist <= r_i: cnt += 1
            D_i = (cnt/N) if N > 0 else 0.0
            score = W_OVER*S_i + O_OVER*O_i + D_OVER*D_i
            t = tier_of(score)
            tier_counts[t] += 1
            per_det.append({"idx": idx, "tier": t, "score": float(score)})
        manifest[s.filepath] = per_det
        if i % 20 == 0:
            print(f"  {i}/{n_imgs} done", flush=True)
    MAN.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tier_counts": tier_counts, "total": sum(tier_counts.values()), "manifest": manifest}, open(MAN,"w"), indent=2)
    total = sum(tier_counts.values())
    print(f"\ntier distribution: {tier_counts}  (total {total})")
    print(f"pct: { {k: round(100*v/total,1) if total else 0 for k,v in tier_counts.items()} }")
    print(f"-> {MAN}")


if __name__ == "__main__":
    main()
