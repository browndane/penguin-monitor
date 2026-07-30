#!/usr/bin/env python3
"""Phase C2: controlled NMS toggle on a single trained YOLOv11x-seg checkpoint.

The CrossParadigm paper (P1) claims that "removing non-maximum-suppression dampens
occlusion-driven failures and exposes size as the dominant residual factor," citing
occlusion-failure correlation 0.231 -> 0.074 and size-failure correlation 0.345 -> 0.605.
That claim is currently OBSERVATIONAL across architectures (YOLOv11x-seg NMS-based vs
RT-DETR/RF-DETR NMS-free), so "removing NMS" is confounded with backbone, training, and
everything else. The user's locked decision #7: run a REAL toggle on the SAME trained
checkpoint, NMS on vs NMS off (iou=1.0, a no-op), and refit the failure correlations
both ways. This isolates the NMS post-processing from the architecture.

GPU eval only (inference, no retraining). Uses the trained YOLOv11x-seg checkpoint that
survives at ~/bench_any/runs/segment/african-penguin_yolo11x-seg_..._e150.../weights/best.pt
on the 856-image iNaturalist-veryclean test split.

conf=0.001 matches Ultralytics' val() default (the paper's eval protocol): near-zero
threshold keeps every candidate so the PR curve is complete, the COCO-eval convention.
NMS is controlled SOLELY by the iou argument: iou=0.7 (the val default) => NMS on;
iou=1.0 => NMS is a no-op (no two distinct boxes reach IoU 1.0, nothing suppressed).
conf is held IDENTICAL across both arms so the only varying factor is post-processing NMS.

Usage:
    nohup taskset -c 0-5 ~/.virtualenvs/cvML/bin/python penguins/nms_toggle_refit.py \
        --out penguins/research/c2_nms_toggle_2026-07-15.json \
        > ~/c2_nms_toggle.log 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

BENCH = "/home/vorse/bench_any/runs"
RUN_DIR = f"{BENCH}/segment/african-penguin_yolo11x-seg_b0.98_e150_i640_be64_fair_grid"
CKPT = f"{RUN_DIR}/weights/best.pt"
# Shared 856-image GT (cat_id=1).
GT_PATH = f"{BENCH}/segment/african-penguin_eomt-large_b2_e150_i640_be64_fair_grid/_bench_gt_coco.json"
DATASET = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/african-penguin-inaturalist-sam-veryclean"
TEST_IMG_DIR = f"{DATASET}/test/images"

# Difficulty factor constants (must match penguins/ap_difficulty.py exactly).
A_REF = 0.04   # reference area fraction for the size factor (ap_difficulty L52)
N_MAX = 8.0    # neighbour-count normaliser for the density factor (ap_difficulty L53)


def _box_iou_xyxy(a, b):
    """IoU of two xyxy boxes."""
    x1, y1, x2, y2 = a
    x1b, y1b, x2b, y2b = b
    ix1, iy1 = max(x1, x1b), max(y1, y1b)
    ix2, iy2 = min(x2, x2b), min(y2, y2b)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def difficulty_factors(gt_boxes_xyxy: List[Tuple[float, float, float, float]],
                        W: int, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-GT-instance (size, occlusion, density) factors, matching penguins/ap_difficulty.py.

    S_i = 1 - min(area / (W*H) / A_REF, 1)        # small objects -> high size difficulty
    O_i = max IoU with any other GT box in image  # overlapping GT -> occlusion
    D_i = min(neighbour_count / N_MAX, 1)           # crowded -> density
    """
    n = len(gt_boxes_xyxy)
    S = np.zeros(n)
    O = np.zeros(n)
    D = np.zeros(n)
    areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in gt_boxes_xyxy]
    rel_areas = [a / (W * H) for a in areas]
    cx = [(b[0] + b[2]) / 2 for b in gt_boxes_xyxy]
    cy = [(b[1] + b[3]) / 2 for b in gt_boxes_xyxy]
    diags = [((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5 for b in gt_boxes_xyxy]
    for i in range(n):
        S[i] = 1.0 - min(rel_areas[i] / A_REF, 1.0)
        max_iou = 0.0
        for j in range(n):
            if i != j:
                max_iou = max(max_iou, _box_iou_xyxy(gt_boxes_xyxy[i], gt_boxes_xyxy[j]))
        O[i] = max_iou
        sr = 2.0 * diags[i]
        nb = 1 + sum(1 for j in range(n) if i != j and
                     ((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2) ** 0.5 <= sr)
        D[i] = min(nb / N_MAX, 1.0)
    return S, O, D


def matched_pred(gt_xyxy, preds_xyxy, preds_scores, iou_thr=0.5):
    """Best-overlapping pred for a GT box (IoU >= iou_thr), else missed (idx=-1)."""
    best_iou, best_idx = iou_thr - 1e-9, -1
    for k, p in enumerate(preds_xyxy):
        iou = _box_iou_xyxy(gt_xyxy, p)
        if iou > best_iou:
            best_iou, best_idx = iou, k
    if best_idx >= 0 and best_iou >= iou_thr:
        return best_idx, best_iou
    return -1, 0.0


def run_inference(ckpt, img_dir, iou_thr_nms, imgsz=640, conf=0.001, max_det=1000):
    """Run YOLO predict; return dict image_stem -> list of ((x1,y1,x2,y2), score).

    stream=True processes one image at a time and frees its GPU tensors before the next,
    which is essential for the NMS-off arm: iou=1.0 + conf=0.001 lets thousands of raw
    candidate boxes survive per image, and batch-decoding masks for all of them OOMs a
    15.5 GB card. Streaming keeps peak memory to one image's worth.
    """
    from ultralytics import YOLO
    model = YOLO(ckpt)
    res = model.predict(source=img_dir, imgsz=imgsz, conf=conf, iou=iou_thr_nms,
                        device=0, verbose=False, save=False, save_txt=False, save_json=False,
                        retina_masks=False, agnostic_nms=False, max_det=max_det, stream=True)
    out: Dict[str, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    for r in res:
        stem = os.path.splitext(os.path.basename(r.path))[0]
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            for k in range(len(xyxy)):
                boxes.append(((float(xyxy[k, 0]), float(xyxy[k, 1]),
                               float(xyxy[k, 2]), float(xyxy[k, 3])), float(sc[k])))
        out[stem] = boxes
    return out


def failure_correlations(gt_by_stem, preds_by_stem, wh, iou_match=0.5):
    """For each GT instance: matched by a pred (detected) or not (failure=1).
    Returns per-factor Pearson r with failure + detection accounting.
    """
    from scipy import stats
    S_list, O_list, D_list, fail = [], [], [], []
    n_preds_total = 0
    n_detected = 0
    for stem, gts in gt_by_stem.items():
        W, H = wh[stem]
        S, O, D = difficulty_factors(gts, W, H)
        preds = preds_by_stem.get(stem, [])
        n_preds_total += len(preds)
        pred_xyxy = [p[0] for p in preds]
        pred_sc = [p[1] for p in preds]
        for i, g in enumerate(gts):
            idx, _ = matched_pred(g, pred_xyxy, pred_sc, iou_match)
            if idx >= 0:
                n_detected += 1
            S_list.append(S[i]); O_list.append(O[i]); D_list.append(D[i])
            fail.append(0.0 if idx >= 0 else 1.0)
    S_arr, O_arr, D_arr, fail_arr = map(np.array, (S_list, O_list, D_list, fail))
    return {
        "n_gt": int(len(S_arr)),
        "n_detected": int(n_detected),
        "detect_rate": round(n_detected / len(S_arr), 4) if len(S_arr) else 0.0,
        "n_preds_total": int(n_preds_total),
        "pearson_r_size_failure": round(float(stats.pearsonr(S_arr, fail_arr)[0]), 4),
        "pearson_r_occlusion_failure": round(float(stats.pearsonr(O_arr, fail_arr)[0]), 4),
        "pearson_r_density_failure": round(float(stats.pearsonr(D_arr, fail_arr)[0]), 4),
    }


def load_gt(gt_path):
    """GT coco -> (dict stem -> list of xyxy, dict stem -> (W,H))."""
    coco = json.load(open(gt_path))
    id2img = {im["id"]: im for im in coco["images"]}
    by_stem: Dict[str, List[Tuple[float, float, float, float]]] = {}
    wh: Dict[str, Tuple[int, int]] = {}
    for a in coco["annotations"]:
        im = id2img[a["image_id"]]
        stem = os.path.splitext(im["file_name"])[0]
        x, y, w, h = a["bbox"]
        by_stem.setdefault(stem, []).append((float(x), float(y), float(x + w), float(y + h)))
        wh[stem] = (int(im["width"]), int(im["height"]))
    return by_stem, wh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--img_dir", default=TEST_IMG_DIR)
    ap.add_argument("--out", default="penguins/research/c2_nms_toggle_2026-07-15.json")
    ap.add_argument("--iou_match", type=float, default=0.5,
                    help="IoU threshold for a GT to count as detected")
    ap.add_argument("--nms_on_iou", type=float, default=0.7,
                    help="NMS IoU for the on-arm (Ultralytics val default)")
    ap.add_argument("--nms_off_iou", type=float, default=1.0,
                    help="NMS IoU for the off-arm (1.0 = no-op)")
    args = ap.parse_args()

    print(f"[c2] ckpt: {args.ckpt}", flush=True)
    print(f"[c2] img_dir: {args.img_dir}", flush=True)
    assert os.path.isfile(args.ckpt), f"checkpoint missing: {args.ckpt}"
    assert os.path.isdir(args.img_dir), f"img dir missing: {args.img_dir}"

    gt_by_stem, wh = load_gt(GT_PATH)
    print(f"[c2] loaded GT: {len(gt_by_stem)} images, "
          f"{sum(len(v) for v in gt_by_stem.values())} GT instances", flush=True)

    print(f"[c2] NMS ON (iou={args.nms_on_iou}, conf=0.001) ...", flush=True)
    preds_on = run_inference(args.ckpt, args.img_dir, iou_thr_nms=args.nms_on_iou)
    print(f"[c2]   {sum(len(v) for v in preds_on.values())} total predictions", flush=True)

    print(f"[c2] NMS OFF (iou={args.nms_off_iou}, conf=0.001) ...", flush=True)
    preds_off = run_inference(args.ckpt, args.img_dir, iou_thr_nms=args.nms_off_iou)
    print(f"[c2]   {sum(len(v) for v in preds_off.values())} total predictions", flush=True)

    print("[c2] refitting failure correlations (NMS ON) ...", flush=True)
    res_on = failure_correlations(gt_by_stem, preds_on, wh, args.iou_match)
    print("[c2] refitting failure correlations (NMS OFF) ...", flush=True)
    res_off = failure_correlations(gt_by_stem, preds_off, wh, args.iou_match)

    result = {
        "checkpoint": args.ckpt,
        "n_gt_images": len(gt_by_stem),
        "n_gt_instances": sum(len(v) for v in gt_by_stem.values()),
        "iou_match": args.iou_match,
        "nms_on": res_on,
        "nms_off": res_off,
        "paper_claim_p1_L252": {
            "occlusion_failure_cross_arch": "0.231 (NMS-based) -> 0.074 (NMS-free)",
            "size_failure_cross_arch": "0.345 (NMS-based) -> 0.605 (NMS-free)",
            "note": "cross-architecture observational; this C2 run is the controlled "
                    "same-checkpoint toggle (conf held at 0.001 both arms, only NMS iou varies)",
        },
        "interpretation": {
            "nms_off_drops_occlusion_corr_more_than_size": (
                (res_off["pearson_r_occlusion_failure"] < res_on["pearson_r_occlusion_failure"])
                and (res_off["pearson_r_size_failure"] >= res_on["pearson_r_size_failure"])
            ),
            "pred_count_inflation_nms_off_vs_on": res_off["n_preds_total"] - res_on["n_preds_total"],
            "detect_rate_on": res_on["detect_rate"],
            "detect_rate_off": res_off["detect_rate"],
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[c2] wrote {args.out}", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
