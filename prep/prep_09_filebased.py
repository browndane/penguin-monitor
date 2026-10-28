# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Phase 1 prep_09 (file-based): compute paper difficulty tiers (L170-179) for a YOLO
gold corpus -> SIDECAR JSON manifest WITH gold RLEs.

D_i = 0.40*S_i + 0.35*O_i + 0.25*delta_i
  S_i = 1 - min(Area_i / (0.04 * W * H), 1)         (size; smaller = harder)
  O_i = max_{j!=i} IoU(b_i, b_j)                    (overlap; via mask bbox)
  delta_i = (1/N) * |{j : ||c_j - c_i||_2 <= r_i}|  (local density; N = per-image count)
  r_i = 3 * sqrt(Area_i)

Tiers: Easy < 0.30, Medium 0.30-0.58, Hard >= 0.58  (report AS IS, do NOT force 40/40/20).

File-based (no FiftyOne): reads YOLO labels `class cx cy w h x1 y1 x2 y2 ...` (coords
normalised 0-1), rasterizes polygons via cv2.fillPoly at NATIVE res (coords * W,H) to
compute Area, encodes to COCO RLE IMMEDIATELY (never holds full-res mask stack; ~17GB
for Boulders 123 imgs x 769 inst x 5184x3456). Image dims from cv2.imread (NOT fo metadata).

Output: ~/bench_any/runs/p5_r1_v2/<corpus>/<corpus>_tiers.json keyed by FILENAME ->
[{"idx", "tier", "score", "rle", "bbox":[x,y,w,h], "area", "width", "height"}, ...].
The eval (prep_10) and sweep (prep_11) load these pre-computed RLEs, no re-rasterize.
"""
import json, os, sys, argparse
from pathlib import Path
import numpy as np
import cv2
import pycocotools.mask as mu

W_OVER, O_OVER, D_OVER = 0.40, 0.35, 0.25
EASY_MAX, MED_MAX = 0.30, 0.58


def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def rasterize_yolo_polygon(parts, W, H):
    """parts: list of floats [cx,cy,w,h, x1,y1, x2,y2, ...] (normalised 0-1).
    Polygon vertices are the x1,y1,x2,y2,... pairs. Rasterize via fillPoly at native res.
    Returns (mask_bool[H,W], area_px, bbox_abs [x0,y0,w,h])."""
    n = len(parts)
    if n < 4 + 4:  # need cx,cy,w,h + at least one (x,y) pair
        return None
    # polygon vertices start at index 4 (after cx,cy,w,h); pairs
    verts = parts[4:]
    if len(verts) % 2 != 0 or len(verts) < 2:
        return None
    pts = []
    for k in range(0, len(verts), 2):
        x = float(verts[k]) * W
        y = float(verts[k + 1]) * H
        pts.append([x, y])
    if len(pts) < 3:
        # degenerate: fall back to bbox rect (cx,cy,w,h)
        cx, cy, bw, bh = parts[0], parts[1], parts[2], parts[3]
        x0 = (cx - bw / 2) * W
        y0 = (cy - bh / 2) * H
        x1 = (cx + bw / 2) * W
        y1 = (cy + bh / 2) * H
        pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_np], 1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    area = float(mask.sum())
    bbox = [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    return mask.astype(bool), area, bbox


def bbox_iou(a, b):
    # a, b = [x0,y0,x1,y1]
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tier_of(score):
    if score < EASY_MAX:
        return "easy"
    elif score < MED_MAX:
        return "medium"
    else:
        return "hard"


def parse_yolo_label(path):
    """Return list of per-line parts (lists of floats)."""
    dets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 5:
                continue
            cls = int(float(toks[0]))
            nums = [float(t) for t in toks[1:]]
            dets.append((cls, nums))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--images", required=True, help="images dir")
    ap.add_argument("--labels", required=True, help="labels dir")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    tier_counts = {"easy": 0, "medium": 0, "hard": 0}
    manifest = {}
    n_imgs = len(img_files)
    n_inst = 0

    for i, imgp in enumerate(img_files):
        fname = imgp.name
        lblp = lbl_dir / (imgp.stem + ".txt")
        img = cv2.imread(str(imgp))
        if img is None:
            manifest[fname] = []
            print(f"  {i}/{n_imgs} {fname} IMG READ FAIL", flush=True)
            continue
        H, W = img.shape[:2]
        del img
        if not lblp.exists():
            manifest[fname] = []
            if i % 20 == 0:
                print(f"  {i}/{n_imgs} {fname} (no label)", flush=True)
            continue
        dets = parse_yolo_label(lblp)
        # rasterize each, encode RLE immediately, collect per-det metadata
        per_det = []
        rles = []
        areas = []
        bboxes_xyxy = []
        centers = []
        for cls_i, nums in dets:
            r = rasterize_yolo_polygon(nums, W, H)
            if r is None:
                continue
            mask_bool, area, bbox_xywh = r
            rle = encode_mask(mask_bool)
            del mask_bool
            x0, y0, w, h = bbox_xywh
            bbox_xyxy = [x0, y0, x0 + w, y0 + h]
            rles.append(rle)
            areas.append(area)
            bboxes_xyxy.append(bbox_xyxy)
            centers.append((x0 + w / 2.0, y0 + h / 2.0))
        N = float(len(rles))
        n_inst += len(rles)
        for idx in range(len(rles)):
            area = areas[idx]
            S_i = 1.0 - min(area / (0.04 * W * H), 1.0) if (W * H) > 0 else 1.0
            O_i = 0.0
            bi = bboxes_xyxy[idx]
            for j in range(len(rles)):
                if j == idx:
                    continue
                O_i = max(O_i, bbox_iou(bi, bboxes_xyxy[j]))
            r_i = 3 * np.sqrt(area) if area > 0 else 0.0
            cx, cy = centers[idx]
            cnt = 0
            for j in range(len(rles)):
                if j == idx:
                    continue
                dist = np.hypot(centers[j][0] - cx, centers[j][1] - cy)
                if dist <= r_i:
                    cnt += 1
            delta_i = (cnt / N) if N > 0 else 0.0
            score = W_OVER * S_i + O_OVER * O_i + D_OVER * delta_i
            t = tier_of(score)
            tier_counts[t] += 1
            per_det.append({
                "idx": idx,
                "tier": t,
                "score": float(score),
                "rle": rles[idx],
                "bbox": bboxes_xyxy[idx],  # [x0,y0,x1,y1]
                "area": float(area),
                "width": int(W),
                "height": int(H),
            })
        manifest[fname] = per_det
        if i % 20 == 0:
            print(f"  {i}/{n_imgs} {fname} done ({len(rles)} inst)", flush=True)

    json.dump({
        "corpus": args.corpus,
        "tier_counts": tier_counts,
        "total": sum(tier_counts.values()),
        "n_images": n_imgs,
        "n_instances": n_inst,
        "manifest": manifest,
    }, open(out, "w"))
    total = sum(tier_counts.values())
    print(f"\ntier distribution: {tier_counts}  (total {total})", flush=True)
    print(f"pct: { {k: round(100 * v / total, 1) if total else 0 for k, v in tier_counts.items()} }", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
