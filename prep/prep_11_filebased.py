# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Phase 1 prep_11 (file-based): SAM3 zero-shot text-prompt predictions on a YOLO gold
corpus, sanity-sieve sweep, final preds for eval.

Pipeline (all file-based, no FiftyOne):
  1. Run SAM3 text-prompt "a penguin" once per image at SCORE=0.25, MASK_THR=0.5.
     Save RAW outputs (masks as COCO RLE, scores, boxes) per image to {corpus}_sam3_raw.json.
     SAM3 is the expensive part; sweep reuses raw outputs, never re-runs SAM3.
  2. Sweep sieve grid:
       min_area in {32, 64, 128}
       nms_iou in {0.7, 0.8, 0.9}
       max_dim_frac in {0.9, 0.95, 1.0}
       SCORE in [0.25, 0.28] (secondary; raw outputs already filtered at 0.25, so 0.28 is a
         post-filter on saved scores; 0.25 = no extra filter).
     Objective: AP@0.5 with recall >= 0.95 floor. Compute AP@0.5 cheaply (numpy IoU +
     confidence sort vs v2 gold RLEs from prep_09 tiers sidecar). Among configs meeting
     recall>=0.95 floor, pick max AP@0.5. Save sweep table to {corpus}_sieve_sweep.json.
  3. Apply chosen sieve config to raw outputs -> final preds (masks as COCO RLE + scores +
     boxes), save to {corpus}_sam3_pred.json.

SAM3 API (verified prep_11b): text-prompt via proc(images=im, text="a penguin",
return_tensors="pt") then mdl(**inputs); bf16; LONG_EDGE_CAP=2448 for >4096px (Boulders
5184px); rescale masks to native via INTER_NEAREST.
post_process_instance_segmentation(out, threshold=0.25, mask_threshold=0.5,
target_sizes=[(H,W)]). HF_HOME=/mnt/LinuxBackup/sam3.

SCORE=0.25 FIXED (do NOT raise). User verified: at 0.25 FPs are low-confidence and NO
true penguins missed; at 0.35 true positives get cut. Sieve (area/NMS/dim-frac) cuts
FPs, not SCORE.
"""
import os, sys, argparse, time, json, copy
os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")

from pathlib import Path
import numpy as np
import cv2
import pycocotools.mask as mu

LONG_EDGE_CAP = 2448
SCORE = 0.25
MASK_THR = 0.5

# Sieve grid
MIN_AREAS = [32, 64, 128]
NMS_IOUS = [0.7, 0.8, 0.9]
MAX_DIM_FRACS = [0.9, 0.95, 1.0]
SCORES = [0.25, 0.28]
RECALL_FLOOR = 0.95


def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def decode_mask(rle):
    m = mu.decode(rle)
    if isinstance(m, np.ndarray):
        return m.astype(bool)
    return np.asarray(m, dtype=bool)


def get_sam3():
    import torch
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    mdl = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda").eval()
    return proc, mdl


def sam3_text_predict(proc, mdl, img_np, score, mask_thr, prompt="a penguin"):
    import torch
    from PIL import Image
    H0, W0 = img_np.shape[:2]
    long_edge = max(W0, H0)
    if long_edge > LONG_EDGE_CAP:
        scale = LONG_EDGE_CAP / long_edge
        Wds, Hds = int(W0 * scale), int(H0 * scale)
        img_ds = cv2.resize(img_np, (Wds, Hds))
    else:
        img_ds = img_np
        Wds, Hds = W0, H0
    im = Image.fromarray(img_ds[:, :, ::-1]).convert("RGB")
    with torch.inference_mode():
        inputs = proc(images=im, text=prompt, return_tensors="pt").to("cuda")
        out = mdl(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=score, mask_threshold=mask_thr, target_sizes=[(Hds, Wds)])[0]
    masks_ds = res["masks"].cpu().numpy().astype(bool) if "masks" in res else np.zeros((0, Hds, Wds), bool)
    sc = res["scores"].float().cpu().numpy() if "scores" in res else np.zeros(0)
    boxes = res["boxes"].float().cpu().numpy() if "boxes" in res else np.zeros((0, 4))
    # rescale masks to original size
    if len(masks_ds) and (Wds != W0 or Hds != H0):
        masks = np.stack([cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks_ds])
    else:
        masks = masks_ds
    # rescale boxes to original
    if len(boxes) and (Wds != W0 or Hds != H0):
        sx, sy = W0 / Wds, H0 / Hds
        boxes = boxes * np.array([sx, sy, sx, sy])
    return masks, sc, boxes


def sieve_apply(masks_rle, scores, boxes, min_area, nms_iou, max_dim_frac, score_floor, W, H):
    """Apply sieve to raw SAM3 outputs (RLE-encoded masks). Returns keep indices.
    gates: area >= min_area, max-dim <= max_dim_frac*dim, NMS by score, score >= score_floor.
    Uses pycocotools RLE ops (mu.area, mu.toBbox) — NO full-res mask decoding in the sweep."""
    n = len(scores)
    if n == 0:
        return []
    score_ok = scores >= score_floor
    # area + max-dim gates via RLE ops (fast, no decode)
    rle_list = [masks_rle[i] for i in range(n) if score_ok[i]]
    idx_map = [i for i in range(n) if score_ok[i]]
    if not rle_list:
        return []
    areas = np.array(mu.area(rle_list))
    bboxes = np.array(mu.toBbox(rle_list))  # [x, y, w, h]
    area_ok = areas >= min_area
    dim_ok = (bboxes[:, 2] <= max_dim_frac * W) & (bboxes[:, 3] <= max_dim_frac * H)
    # NMS by score (box IoU, fast)
    order = np.argsort(-scores)
    suppressed = np.zeros(n, bool)
    for idx in order:
        if suppressed[idx] or not score_ok[idx]:
            continue
        for jdx in order:
            if jdx == idx or suppressed[jdx] or not score_ok[jdx]:
                continue
            if _iou(boxes[idx], boxes[jdx]) > nms_iou:
                suppressed[jdx] = True
    keep = [i for i in range(n) if score_ok[i] and area_ok[idx_map.index(i)] and dim_ok[idx_map.index(i)] and not suppressed[i]]
    return keep


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_iou_np(a, b):
    """IoU of two bool masks (same shape)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def compute_ap50_recall(gold_per_img, pred_per_img, iou_thr=0.5):
    """Cheap AP@0.5 + recall over all images.
    gold_per_img: {fname: [rle_dict, ...]}
    pred_per_img: {fname: [{"rle","score","box"}, ...]}
    Returns (ap50_percent, recall).
    Uses pycocotools.mask.iou (RLE-based, fast) — NO full-res mask decoding.
    AP@0.5: gather all (score, tp/fp) across images, sort desc, compute all-point
    precision-recall area at IoU 0.5. Recall = TPs / total GT.
    """
    all_scores = []
    all_tp = []
    all_fp = []
    n_gt = 0
    for fname, glds in gold_per_img.items():
        n_gt += len(glds)
        preds = pred_per_img.get(fname, [])
        if not preds:
            continue
        # match preds to golds greedily by score (IoU >= iou_thr) via mu.iou
        pred_sorted = sorted(preds, key=lambda p: -p["score"])
        gold_used = [False] * len(glds)
        # batch IoU: all preds vs all golds in one call
        dt_rles = [p["rle"] for p in pred_sorted]
        gt_rles = glds
        iscrowd = [0] * len(gt_rles)
        iou_mat = mu.iou(dt_rles, gt_rles, iscrowd)  # [n_preds, n_golds]
        for pi, p in enumerate(pred_sorted):
            best_iou = 0.0
            best_g = -1
            for gi in range(len(glds)):
                if gold_used[gi]:
                    continue
                iou = float(iou_mat[pi, gi])
                if iou > best_iou:
                    best_iou = iou
                    best_g = gi
            if best_g >= 0 and best_iou >= iou_thr:
                gold_used[best_g] = True
                all_tp.append(1); all_fp.append(0); all_scores.append(p["score"])
            else:
                all_fp.append(1); all_tp.append(0); all_scores.append(p["score"])
    if n_gt == 0:
        return 0.0, 0.0
    tp = np.array(all_tp); fp = np.array(all_fp); sc = np.array(all_scores)
    order = np.argsort(-sc)
    tp = tp[order]; fp = fp[order]
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / n_gt
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    n_tp = int(cum_tp[-1]) if len(cum_tp) else 0
    rec = n_tp / n_gt
    return ap * 100.0, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--tiers", required=True, help="prep_09 tiers JSON (gold RLEs)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-sam3", action="store_true", help="skip SAM3, reuse existing raw JSON")
    ap.add_argument("--prompt", default="a penguin", help="SAM3 text prompt (default 'a penguin', the paper's single-prompt baseline)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # prompt-tag the output filenames when a non-default prompt is used, so the
    # default "a penguin" raw is never clobbered by a different prompt's run.
    ptag = "" if args.prompt == "a penguin" else "_" + args.prompt.replace(" ", "_")
    raw_path = out_dir / f"{args.corpus}_sam3_raw{ptag}.json"
    sweep_path = out_dir / f"{args.corpus}_sieve_sweep{ptag}.json"
    pred_path = out_dir / f"{args.corpus}_sam3_pred{ptag}.json"

    # load gold tiers + RLEs
    tiers_data = json.load(open(args.tiers))
    gold_manifest = tiers_data["manifest"]
    gold_per_img = {fn: [e["rle"] for e in dets] for fn, dets in gold_manifest.items()}

    img_dir = Path(args.images)
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

    # --- Step 1: SAM3 raw predictions ---
    if args.skip_sam3 and raw_path.exists():
        print(f"SKIP SAM3; reusing {raw_path}", flush=True)
        raw = json.load(open(raw_path))
    else:
        proc, mdl = get_sam3()
        raw = {}
        t0 = time.time()
        for i, imgp in enumerate(img_files):
            fname = imgp.name
            img = cv2.imread(str(imgp))
            if img is None:
                raw[fname] = []
                print(f"  {i}/{len(img_files)} {fname} IMG FAIL", flush=True)
                continue
            H, W = img.shape[:2]
            masks, sc, boxes = sam3_text_predict(proc, mdl, img, SCORE, MASK_THR, prompt=args.prompt)
            del img
            per_img = []
            for k in range(len(masks)):
                rle = encode_mask(masks[k])
                per_img.append({
                    "rle": rle,
                    "score": float(sc[k]),
                    "box": [float(x) for x in boxes[k]],
                })
            raw[fname] = per_img
            del masks, sc, boxes
            if i % 10 == 0:
                print(f"  {i}/{len(img_files)} {fname} SAM3 -> {len(per_img)} raw preds ({time.time()-t0:.1f}s)", flush=True)
        json.dump(raw, open(raw_path, "w"))
        print(f"SAM3 raw done in {time.time()-t0:.1f}s -> {raw_path}", flush=True)

    # --- Step 2: sieve sweep ---
    print("\n=== sieve sweep ===", flush=True)
    sweep_results = []
    best = None
    for min_area in MIN_AREAS:
        for nms_iou in NMS_IOUS:
            for max_dim_frac in MAX_DIM_FRACS:
                for score_floor in SCORES:
                    pred_per_img = {}
                    for fname, raw_preds in raw.items():
                        dets = gold_manifest.get(fname, [])
                        if dets:
                            W = dets[0]["width"]; H = dets[0]["height"]
                        else:
                            # need dims; skip if unknown
                            continue
                        scores = np.array([p["score"] for p in raw_preds])
                        boxes = np.array([p["box"] for p in raw_preds]) if len(raw_preds) else np.zeros((0, 4))
                        rles = [p["rle"] for p in raw_preds]
                        keep = sieve_apply(rles, scores, boxes, min_area, nms_iou, max_dim_frac, score_floor, W, H)
                        pred_per_img[fname] = [{"rle": rles[k], "score": float(scores[k]), "box": boxes[k].tolist()} for k in keep]
                    ap50, rec = compute_ap50_recall(gold_per_img, pred_per_img, iou_thr=0.5)
                    cfg = {"min_area": min_area, "nms_iou": nms_iou, "max_dim_frac": max_dim_frac, "score_floor": score_floor}
                    entry = {**cfg, "ap50": round(ap50, 2), "recall": round(rec, 4), "meets_floor": rec >= RECALL_FLOOR}
                    sweep_results.append(entry)
                    flag = " *" if entry["meets_floor"] else ""
                    print(f"  {cfg} -> AP50={ap50:.2f} rec={rec:.4f}{flag}", flush=True)
                    if entry["meets_floor"] and (best is None or ap50 > best["ap50"]):
                        best = entry
    json.dump({"corpus": args.corpus, "sweep": sweep_results, "best": best}, open(sweep_path, "w"), indent=2)
    print(f"\nbest: {best}", flush=True)
    print(f"-> {sweep_path}", flush=True)

    # --- Step 3: final preds with chosen config ---
    if best is None:
        print("WARNING: no config met recall floor; using min_area=64 nms_iou=0.8 max_dim_frac=1.0 score=0.25", flush=True)
        best = {"min_area": 64, "nms_iou": 0.8, "max_dim_frac": 1.0, "score_floor": 0.25}
    final = {}
    n_preds = 0
    for fname, raw_preds in raw.items():
        dets = gold_manifest.get(fname, [])
        if dets:
            W = dets[0]["width"]; H = dets[0]["height"]
        else:
            continue
        scores = np.array([p["score"] for p in raw_preds])
        boxes = np.array([p["box"] for p in raw_preds]) if len(raw_preds) else np.zeros((0, 4))
        rles = [p["rle"] for p in raw_preds]
        keep = sieve_apply(rles, scores, boxes, best["min_area"], best["nms_iou"], best["max_dim_frac"], best["score_floor"], W, H)
        final[fname] = [{"rle": rles[k], "score": float(scores[k]), "box": boxes[k].tolist()} for k in keep]
        n_preds += len(keep)
    json.dump({"corpus": args.corpus, "chosen_config": best, "n_preds": n_preds, "preds": final}, open(pred_path, "w"))
    print(f"final preds: {n_preds} -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
