# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# Shared post-eval toolkit for bench_any.py and its *_any_tune.py wrappers.
#
# Goal: provide ONE framework-agnostic implementation of the metrics that every
# tuner should report, so the summary table compares apples to apples instead
# of relying on each upstream's quirky logger.
#
# Functions:
#   coco_eval(gt_json, pred_json, iou_type)         -> 12-stat COCOeval dict
#   coco_eval_full(gt_json, bbox_pred, segm_pred)   -> merged bench-schema dict
#   masks_to_box_preds(segm_pred_json, out_json)    -> derives bbox preds from masks
#   yolo_test_split_to_coco_gt(data_yaml, out_json) -> COCO GT for test (image_id=stem)
#   semseg_extra(conf_mat)                          -> mIoU/mAcc/fwIoU/pixel_acc plus foreground-only variants
#   boundary_iou_mean(pred_label_map, gt_label_map, num_classes, dilation_ratio=0.02)
#   model_params_M(model)                           -> float (millions of parameters)
#   time_fps(callable_fn, n_warmup=3, n_iter=20)    -> mean FPS
#   sam2_seg_from_boxes(images_dir, bbox_pred_json, sam2_ckpt, sam2_cfg, out_json)
#
# Every public function is defensive: failures return zeros / empty dicts and
# log a single warning so a missing predictions JSON never crashes a 5-epoch
# smoke test. The bench-schema keys produced here are:
#   box_mAP50_95, box_mAP50, box_mAP75, box_mAP_S, box_mAP_M, box_mAP_L,
#   seg_mAP50_95, seg_mAP50, seg_mAP75, seg_mAP_S, seg_mAP_M, seg_mAP_L,
#   box_AR_100, seg_AR_100, mIoU, mAcc, fwIoU, pixel_acc, boundary_IoU,
#   params_M, fps_eval.

from __future__ import annotations
import os, sys, json, time, warnings, math, tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---- bench-schema metric keys this module can fill --------------------------
# P/R/F1 are derived from pycocotools' PR arrays at IoU=0.5 (averaged over
# categories, IoU=0.5 slice, area=all, maxDets=100) so non-YOLO frameworks
# can populate the same columns YOLO/Ultralytics already fills natively.
BOX_AP_KEYS = ["box_P", "box_R", "box_F1",
               "box_mAP50_95", "box_mAP50", "box_mAP75",
               "box_mAP_S",    "box_mAP_M", "box_mAP_L", "box_AR_100"]
SEG_AP_KEYS = ["seg_P", "seg_R", "seg_F1",
               "seg_mAP50_95", "seg_mAP50", "seg_mAP75",
               "seg_mAP_S",    "seg_mAP_M", "seg_mAP_L", "seg_AR_100"]
# SAM2-prompted segmentation keys: a 1:1 rename of SEG_AP_KEYS (seg_* -> sam2_*).
# Used so SAM2-derived masks land in their own column family instead of
# colliding with native mask-head seg_* metrics.
SAM2_AP_KEYS = ["sam2_" + k[len("seg_"):] for k in SEG_AP_KEYS]
SEMSEG_KEYS = [
    "mIoU", "mAcc", "fwIoU", "pixel_acc", "boundary_IoU",
    "fg_mIoU", "fg_mAcc", "fg_fwIoU", "fg_pixel_acc", "fg_boundary_IoU",
    "fg_seg_P", "fg_seg_R", "fg_seg_F1",
]
PERF_KEYS   = ["params_M", "fps_eval"]

# Mapping COCOeval.stats[0..11] -> (mAP, mAP50, mAP75, mAP_S, mAP_M, mAP_L,
#                                   AR_1, AR_10, AR_100, AR_S, AR_M, AR_L).
_COCO_STAT_ORDER = ["mAP50_95", "mAP50", "mAP75", "mAP_S", "mAP_M", "mAP_L",
                    "AR_1",     "AR_10", "AR_100", "AR_S", "AR_M", "AR_L"]

# Per-image detection cap for torchmetrics (used by dinov3_m2f). Raising the
# top cap to 1000 ensures crowded images with hundreds of penguins aren't
# truncated. torchmetrics accepts arbitrary thresholds so [1,10,1000] is fine.
#
# NOTE: do NOT use this constant for pycocotools COCOeval — pycocotools'
# _summarize(1) hard-codes maxDets=100 in its default call so 100 must always
# be present in ev.params.maxDets or stats[0] (mAP50_95) returns -1 → 0.
# COCOeval maxDets is set explicitly to [1,10,100] in coco_eval() below.
# The prediction-truncation limit (how many boxes land in predictions.json) is
# controlled separately by max_det=1000 in the .val() / inference calls in
# each wrapper, which is the actual mechanism that lets dense scenes be scored.
MAX_DETS = [1, 10, 1000]

# ---- AP-Difficulty core (ported from penguins/ap_difficulty.py L51-58) -----
# Pure annotation-geometry difficulty scoring (size / occlusion / density) that
# stratifies instances into Easy/Medium/Hard tiers for per-tier mAP eval. The
# core math runs in *unit* (normalised) coordinate space; see
# `compute_difficulty_from_coco_gt` for the COCO-GT (pixel-bbox) entry point.
A_REF = 0.04          # Reference area fraction (Eq. 1) — dimensionless (unit space)
N_MAX = 8             # Max-neighbour normalisation (Eq. 3)
ALPHA = 0.40          # Size weight      (Eq. 4)
BETA  = 0.35          # Occlusion weight (Eq. 4)
GAMMA = 0.25          # Density weight   (Eq. 4)
TAU_1 = 0.30          # Easy/Medium boundary   (Eq. 5)
TAU_2 = 0.58          # Medium/Hard boundary   (Eq. 5)
DIFFICULTY_TIERS = ("Easy", "Medium", "Hard")


def _warn(msg: str) -> None:
    print(f"[bench_eval] ⚠️ {msg}", file=sys.stderr)


def _zero_box() -> Dict[str, float]:
    return {k: 0.0 for k in BOX_AP_KEYS}


def _zero_seg() -> Dict[str, float]:
    return {k: 0.0 for k in SEG_AP_KEYS}


def _zero_sam2() -> Dict[str, float]:
    return {k: 0.0 for k in SAM2_AP_KEYS}


def _polygon_area(pts: List[Tuple[float, float]]) -> float:
    """Compute the actual mathematical area of a polygon using the Shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def safe_image_id(stem: str) -> int:
    """Ultralytics `save_json=True` writes image_id = int(stem) when the stem is
    purely numeric, otherwise the raw string. Our COCO GT must agree. We always
    emit ints (pycocotools requires ints): numeric stems become themselves,
    non-numeric stems hash to a deterministic 31-bit int."""
    try:
        return int(stem)
    except ValueError:
        # Stable 31-bit hash (avoids the random salt of builtin hash()).
        import hashlib as _h
        return int(_h.md5(stem.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


# =============================================================================
# 1. COCO ground-truth builder (image_id = stem-derived int; class 0-indexed)
# =============================================================================
def yolo_split_to_coco_gt(data_yaml: str, split: str,
                          out_json: str,
                          one_indexed_cats: bool = False) -> Tuple[int, List[str], Dict[str, int]]:
    """Build a COCO GT json for one split of a YOLO dataset.

    Returns (n_images, class_names, stem_to_image_id). The image_id mapping is
    returned so callers can rewrite their framework's predictions JSON to use
    matching ids before calling coco_eval.

    Polygon labels are turned into both a `segmentation` (polygon) and a `bbox`
    (tight enclosing rect); rect labels get only a `bbox`. category_id is
    0-indexed by default so it matches Ultralytics' `save_json=True` output
    (set one_indexed_cats=True for RF-DETR / pycocotools-COCO-style)."""
    import cv2  # noqa
    import yaml as _yaml
    try:
        cfg = _yaml.safe_load(open(data_yaml))
    except Exception as e:
        _warn(f"cannot read {data_yaml}: {e}")
        return 0, [], {}
    names = cfg.get("names")
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names)]
    else:
        class_names = list(names or [])
    split_rel = cfg.get(split) or cfg.get("valid" if split == "val" else split)
    if not split_rel:
        _warn(f"split '{split}' not in {data_yaml}")
        return 0, class_names, {}
    root = Path(cfg.get("path", Path(data_yaml).parent))
    img_dir = (Path(root) / split_rel).resolve() if not Path(split_rel).is_absolute() else Path(split_rel)
    if not img_dir.is_dir():
        _warn(f"{img_dir} is not a directory")
        return 0, class_names, {}

    cat_off = 1 if one_indexed_cats else 0
    cats = [{"id": i + cat_off, "name": n, "supercategory": "object"}
            for i, n in enumerate(class_names)]
    images, anns = [], []
    stem_to_id: Dict[str, int] = {}
    ann_id = 1
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts: continue
        im = cv2.imread(str(img_path))
        if im is None:
            _warn(f"cv2 cannot read {img_path}"); continue
        h, w = im.shape[:2]
        iid = safe_image_id(img_path.stem)
        stem_to_id[img_path.stem] = iid
        images.append({"id": iid, "file_name": img_path.name, "width": w, "height": h})
        # YOLO labels live under .../labels/... mirroring .../images/...
        parts = list(img_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            lbl = Path(*parts).with_suffix(".txt")
        else:
            lbl = img_path.with_suffix(".txt")
        if not lbl.is_file(): continue
        with open(lbl) as fh:
            for line in fh:
                ps = line.strip().split()
                if len(ps) < 5: continue
                cid = int(ps[0])
                if not (0 <= cid < len(class_names)): continue
                coords = [float(x) for x in ps[1:]]
                if len(coords) == 4:
                    xc, yc, bw, bh = coords
                    x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                    bw_p, bh_p = bw * w, bh * h
                    anns.append({"id": ann_id, "image_id": iid,
                                 "category_id": cid + cat_off,
                                 "bbox": [x1, y1, bw_p, bh_p],
                                 "area": bw_p * bh_p,
                                 "iscrowd": 0, "segmentation": []})
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    anns.append({"id": ann_id, "image_id": iid,
                                 "category_id": cid + cat_off,
                                 "bbox": [x1, y1, x2 - x1, y2 - y1],
                                 "area": _polygon_area(pts),
                                 "iscrowd": 0,
                                 "segmentation": [[c for p in pts for c in p]]})
                ann_id += 1
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump({"images": images, "annotations": anns, "categories": cats}, fh)
    return len(images), class_names, stem_to_id


def fix_yolo_pred_image_ids(pred_json: str, gt_json: str, out_json: str) -> int:
    """Ultralytics save_json=True may write string image_ids for non-numeric
    stems, but pycocotools needs int ids. Rewrite predictions to match GT ids.
    Returns number of predictions fixed (0 on error)."""
    if not (Path(pred_json).is_file() and Path(gt_json).is_file()):
        return 0
    try:
        with open(gt_json) as f:
            gt = json.load(f)
        with open(pred_json) as f:
            preds = json.load(f)

        # Build stem -> int mapping from GT
        stem_to_int = {}
        for img in gt.get("images", []):
            stem = Path(img["file_name"]).stem
            stem_to_int[stem] = img["id"]
            # Also map the raw id in case it matches
            if isinstance(img["id"], int):
                stem_to_int[str(img["id"])] = img["id"]

        # Remap prediction image_ids
        fixed_count = 0
        for pred in preds:
            orig_id = pred["image_id"]
            # Try direct lookup first
            if orig_id in stem_to_int:
                pred["image_id"] = stem_to_int[orig_id]
                fixed_count += 1
            # Try converting to stem
            elif isinstance(orig_id, str):
                stem = Path(orig_id).stem if "/" in str(orig_id) else orig_id
                if stem in stem_to_int:
                    pred["image_id"] = stem_to_int[stem]
                    fixed_count += 1

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(preds, f)
        return fixed_count
    except Exception as e:
        _warn(f"fix_yolo_pred_image_ids failed: {e}")
        return 0


def remap_pred_categories(pred_json: str, gt_json: str, out_json: str) -> Tuple[int, str]:
    """Detect and apply a clean +1 / -1 category_id offset between predictions
    and GT. Returns (n_remapped, mode) where mode in
    {'identity','pred-1','pred+1','no-match','error'}.
    Writes a remapped copy of preds to out_json only on 'pred-1'/'pred+1';
    on 'identity'/'no-match'/'error' the caller should keep the original pred_json.
    Defensive: never raises; logs a warning and returns (0, 'error') on failure."""
    if not (Path(pred_json).is_file() and Path(gt_json).is_file()):
        return 0, "error"
    try:
        with open(pred_json) as f:
            preds = json.load(f)
        with open(gt_json) as f:
            gt = json.load(f)
        if not preds:
            return 0, "identity"
        pred_cats = {int(p["category_id"]) for p in preds if "category_id" in p}
        gt_cats = {int(c["id"]) for c in gt.get("categories", [])}
        if not gt_cats:
            _warn("remap_pred_categories: GT has no categories")
            return 0, "no-match"
        if not pred_cats:
            return 0, "identity"

        is_identity_valid = pred_cats <= gt_cats

        shifted_minus = {c - 1 for c in pred_cats}
        is_minus_valid = all(c >= 0 for c in shifted_minus) and shifted_minus <= gt_cats

        shifted_plus = {c + 1 for c in pred_cats}
        is_plus_valid = shifted_plus <= gt_cats

        if is_identity_valid and is_minus_valid:
            # Tie-breaker: If predictions don't start at 0, and min is >= 1, and shift-minus is valid,
            # it is highly likely that they are 1-indexed predictions mapped to a 0-indexed GT.
            if not (all(c - 1 in gt_cats for c in pred_cats) and min(pred_cats) >= 1):
                return 0, "identity"
        elif is_identity_valid:
            return 0, "identity"

        # Try -1 offset (e.g. 1-indexed pred vs 0-indexed GT). Only accept a
        # clean set match/subset after shifting; a loose intersection is unsafe
        # on one-class datasets because unrelated COCO classes can be shifted
        # into the project class and inflate/deflate AP.
        if is_minus_valid:
            for p in preds:
                p["category_id"] = int(p["category_id"]) - 1
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(preds, f)
            print(f"   🔧 remapped category_id (pred-1): {sorted(pred_cats)} -> {sorted(shifted_minus)}")
            return len(preds), "pred-1"

        # Try +1 offset (e.g. 0-indexed pred vs 1-indexed GT)
        if is_plus_valid:
            for p in preds:
                p["category_id"] = int(p["category_id"]) + 1
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(preds, f)
            print(f"   🔧 remapped category_id (pred+1): {sorted(pred_cats)} -> {sorted(shifted_plus)}")
            return len(preds), "pred+1"
        _warn(f"remap_pred_categories: no-match. pred_cats={sorted(pred_cats)} "
              f"gt_cats={sorted(gt_cats)}")
        return 0, "no-match"
    except Exception as e:
        _warn(f"remap_pred_categories failed: {e}")
        return 0, "error"


# =============================================================================
# 2. COCOeval wrappers
# =============================================================================
def coco_eval(gt_json: str, pred_json: str, iou_type: str = "bbox",
              quiet: bool = True) -> Dict[str, float]:
    """Run pycocotools COCOeval and return the 12 standard stats, prefixed by
    iou_type ('bbox' / 'segm'). Returns zeros on any error."""
    prefix = "box_" if iou_type == "bbox" else "seg_"
    blank = {prefix + k: 0.0 for k in _COCO_STAT_ORDER}
    # P/R/F1 are also part of this function's return schema (added in v2 so
    # non-YOLO frameworks can populate them); ensure error paths emit zeros
    # for them as well, otherwise downstream `out.get(...)` lookups crash.
    blank[prefix + "P"]  = 0.0
    blank[prefix + "R"]  = 0.0
    blank[prefix + "F1"] = 0.0
    if not (Path(gt_json).is_file() and Path(pred_json).is_file()):
        _warn(f"missing GT or pred JSON for {iou_type}: gt={gt_json} pred={pred_json}")
        return blank
    try:
        from pycocotools.coco import COCO  # noqa
        from pycocotools.cocoeval import COCOeval  # noqa
    except Exception as e:
        _warn(f"pycocotools not importable ({e}); {iou_type} = 0")
        return blank
    # Empty predictions array crashes pycocotools' loadRes; bail with zeros.
    try:
        with open(pred_json) as fh:
            preds = json.load(fh)
        if not preds:
            _warn(f"{pred_json} is empty; {iou_type} = 0")
            return blank
    except Exception as e:
        _warn(f"cannot read {pred_json}: {e}"); return blank

    if quiet:
        # COCOeval is chatty — silence the stdout dump while preserving warnings.
        import contextlib, io
        _ctx = contextlib.redirect_stdout(io.StringIO())
    else:
        import contextlib as _c
        _ctx = _c.nullcontext()
    try:
        with _ctx:
            gt = COCO(gt_json)
            # Clinical fix: pycocotools.COCO.loadRes() raises Exception if results
            # contain image_ids or category_ids not present in the GT set. This
            # happens when a model evaluates on 'val' (e.g. 8 imgs) but we call
            # post-hoc eval against 'test' (e.g. 5 imgs), or if a model predicts
            # extra COCO classes not in the project dataset. Filter to valid IDs.
            valid_ids = set(gt.getImgIds())
            valid_cats = set(gt.getCatIds())
            filtered_preds = []
            for p in preds:
                try:
                    # Coerce to int; pycocotools expects integer IDs.
                    iid = int(p.get("image_id", -1))
                    cid = int(p.get("category_id", -1))
                    if iid in valid_ids and cid in valid_cats:
                        # Copy to avoid modifying the original list if reused.
                        p_new = {k: v for k, v in p.items()}
                        p_new["image_id"] = iid
                        p_new["category_id"] = cid
                        filtered_preds.append(p_new)
                except (ValueError, TypeError):
                    continue
            if not filtered_preds:
                _warn(f"COCOeval({iou_type}): no predictions matched GT images/categories")
                return blank
            # Use the filtered list directly (loadRes accepts either filename or list)
            dt = gt.loadRes(filtered_preds)
            ev = COCOeval(gt, dt, iouType=iou_type)
            # MUST keep [1,10,100]: pycocotools._summarize(1) hard-codes
            # maxDets=100; if 100 is absent stats[0]=mAP50_95 returns -1→0.
            # Dense-image truncation is handled upstream by max_det=1000 in
            # .val() / inference calls, not here.
            ev.params.maxDets = [1, 10, 100]
            # Fail loud if someone propagates MAX_DETS=[1,10,1000] (torchmetrics
            # only) into COCOeval — _summarize(1) hard-codes maxDets=100, so its
            # absence silently zeroes mAP50_95. Better a crash than a 0.0 headline.
            assert list(ev.params.maxDets) == [1, 10, 100], (
                f"ev.params.maxDets must be [1,10,100] (got {list(ev.params.maxDets)}); "
                "MAX_DETS=[1,10,1000] is torchmetrics-only, never COCOeval.")
            ev.evaluate(); ev.accumulate(); ev.summarize()
            stats = list(ev.stats)
            ev_eval = ev.eval  # precision/recall arrays (filled by accumulate())
        print(f"   ✓ COCOeval({iou_type}): {len(filtered_preds)}/{len(preds)} predictions evaluated")
    except Exception as e:
        _warn(f"COCOeval({iou_type}) failed: {e}")
        try:
            gt_data = json.load(open(gt_json))
            print(f"   ✗ GT had {len(gt_data.get('images', []))} images, pred had {len(preds)} entries")
        except Exception: pass
        return blank
    out = {}
    for i, k in enumerate(_COCO_STAT_ORDER):
        v = float(stats[i]) if i < len(stats) else 0.0
        # pycocotools returns -1 when a size bucket has no GT/dets; report 0
        # so the bench summary doesn't render negatives.
        out[prefix + k] = max(0.0, v)
    # ---- P / R / F1 derived from COCOeval PR arrays at IoU=0.5 -------------
    # Matches Ultralytics' headline P/R semantics: precision and recall
    # averaged over categories at the F1-optimal confidence point. The
    # COCOeval `precision` array has shape (T, R, K, A, M) where T=IoU
    # thresholds, R=recall thresholds (101), K=categories, A=area ranges,
    # M=maxDets. We slice T=IoU0.5, A=all (idx 0), M=maxDets=100 (idx -1)
    # and pick the recall point that maximises F1 per-category, then average.
    try:
        import numpy as _np
        P_arr = ev_eval.get("precision") if isinstance(ev_eval, dict) else None
        R_arr = ev_eval.get("recall")    if isinstance(ev_eval, dict) else None
        if P_arr is not None and R_arr is not None and P_arr.size > 0:
            # IoU=0.5 is the first threshold in COCOeval's default iouThrs.
            p50 = P_arr[0, :, :, 0, -1]          # (R, K)
            r_thr = _np.asarray(ev.params.recThrs)  # (R,)
            f1 = _np.zeros(p50.shape[1], dtype=float)
            p_pick = _np.zeros(p50.shape[1], dtype=float)
            r_pick = _np.zeros(p50.shape[1], dtype=float)
            for k_idx in range(p50.shape[1]):
                pk = p50[:, k_idx]
                # pycocotools writes -1 for "no detections / no GT" cells.
                valid = pk >= 0
                if not valid.any():
                    continue
                pv = _np.where(valid, pk, 0.0)
                rv = r_thr  # recall thresholds align with axis R
                denom = pv + rv
                f1k = _np.where(denom > 0, 2 * pv * rv / _np.maximum(denom, 1e-12), 0.0)
                best = int(_np.argmax(f1k))
                f1[k_idx]   = float(f1k[best])
                p_pick[k_idx] = float(pv[best])
                r_pick[k_idx] = float(rv[best])
            # Average only over categories that had any valid prediction.
            mask = f1 > 0
            if mask.any():
                out[prefix + "P"]  = float(p_pick[mask].mean())
                out[prefix + "R"]  = float(r_pick[mask].mean())
                out[prefix + "F1"] = float(f1[mask].mean())
            else:
                out[prefix + "P"] = out[prefix + "R"] = out[prefix + "F1"] = 0.0
        else:
            out[prefix + "P"] = out[prefix + "R"] = out[prefix + "F1"] = 0.0
    except Exception as _e:
        _warn(f"P/R/F1 derivation from COCOeval failed ({_e}); reporting zeros")
        out[prefix + "P"] = out[prefix + "R"] = out[prefix + "F1"] = 0.0
    return out


# ============================================================================
# AP-Difficulty: per-instance difficulty + per-tier COCOeval
# (ported from penguins/ap_difficulty.py L100-223 + L665-767). All geometry runs
# in *unit* (normalised) coordinate space; the COCO-GT entry point normalises
# pixel bboxes first. No image decoding — pure annotation geometry.
# ============================================================================

def _unit_bbox(x: float, y: float, w: float, h: float) -> tuple:
    """Normalised bbox as (x1, y1, x2, y2, area, diag) in unit space.

    COCO GT stores ``bbox=[x,y,w,h]`` in *pixels*; callers normalise by the
    image's width/height first (w_n=w/W, h_n=h/H, cx_n=(x+w/2)/W, cy_n=(y+h/2)/H)
    so A_REF=0.04 is dimensionless. Doing geometry in pixel space here would make
    size_factor = 1 - min((w*h)/A_REF, 1) ≈ 1 for every box → all-"Hard" tiers
    and a meaningless Easy−Hard delta.
    """
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    area = w * h
    diag = math.sqrt(w * w + h * h)
    return x1, y1, x2, y2, area, diag


def _unit_iou(a, b) -> float:
    """IoU of two unit-space (x1,y1,x2,y2,area,diag) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[4] + b[4] - inter
    return inter / union if union > 0 else 0.0


def _unit_center_distance(a, b) -> float:
    """Euclidean distance between two unit-space box centres."""
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def compute_difficulty_from_coco_gt(gt_json: str) -> list:
    """For every GT annotation, compute S_i (Eq.1), O_i (Eq.2), D_i (Eq.3),
    the combined difficulty_score (Eq.4), and difficulty_tier (Eq.5).

    Reads the COCO GT JSON directly: ``bbox=[x,y,w,h]`` in *pixels*,
    ``images[].width/height`` populated. The FIRST step is to normalise each box
    by its image's width/height into unit space, then run all geometry (size,
    IoU, density search_radius=2*diag_n, center distance) in unit space — matching
    ap_difficulty.py which works in YOLO normalised coords. Do NOT mix pixel and
    unit space: A_REF=0.04 is dimensionless, so pixel-space area makes
    size_factor≈1 → every instance lands in one tier and the Easy−Hard delta is
    silently meaningless (the worst failure mode — plausible-looking numbers).

    Returns a list of dicts ``{ann_id, image_id, size_factor, occlusion_factor,
    density_factor, difficulty_score, difficulty_tier}`` ordered as the GT's
    annotations appear (ann_id is the GT ``id`` field).
    """
    with open(gt_json) as fh:
        gt = json.load(fh)
    # image_id -> (width, height) so each box is normalised by its own image.
    dims = {im["id"]: (float(im.get("width") or 0), float(im.get("height") or 0))
            for im in gt.get("images", [])}
    # Group boxes per image (unit space) for occlusion/density, and keep a flat
    # ordered list mirroring ap_difficulty's ann-counter loop.
    per_image: Dict[int, list] = {}
    flat: list = []
    for ann in gt.get("annotations", []):
        img_id = ann["image_id"]
        W, H = dims.get(img_id, (0.0, 0.0))
        x, y, w, h = ann["bbox"]
        if W > 0 and H > 0:
            wn, hn = w / W, h / H
            cxn, cyn = (x + w / 2.0) / W, (y + h / 2.0) / H
            # unit-space (x1,y1,x2,y2) centred on the normalised centre
            box = _unit_bbox(cxn - wn / 2.0, cyn - hn / 2.0, wn, hn)
        else:
            # No image dims on file: cannot normalise safely; skip this ann.
            # (bench_any writes W/H into the GT, so this is a defensive guard.)
            box = None
        entry = {
            "ann_id": ann.get("id"),
            "image_id": img_id,
            "box": box,                # unit-space (x1,y1,x2,y2,area,diag) or None
            "size_factor": 0.0,
            "occlusion_factor": 0.0,
            "density_factor": 0.0,
            "difficulty_score": 0.0,
            "difficulty_tier": "Easy",
        }
        per_image.setdefault(img_id, []).append(entry)
        flat.append(entry)

    # Eq.1-4 — geometry in unit space, per image (occlusion/density are within-image).
    for img_id, items in per_image.items():
        boxes = [it["box"] for it in items]
        for i, it in enumerate(items):
            bi = boxes[i]
            if bi is None:
                it["difficulty_tier"] = "Easy"  # un-normalisable → cannot score → default
                continue
            # Size factor (Eq.1): area is already relative in unit space.
            it["size_factor"] = 1.0 - min(bi[4] / A_REF, 1.0)
            # Occlusion factor (Eq.2): max pairwise IoU with siblings.
            max_iou = 0.0
            for j in range(len(boxes)):
                if i != j and boxes[j] is not None:
                    max_iou = max(max_iou, _unit_iou(bi, boxes[j]))
            it["occlusion_factor"] = max_iou
            # Density factor (Eq.3): neighbours within 2*diagonal (unit space).
            search_radius = 2.0 * bi[5]
            neighbour_count = sum(
                1 for j in range(len(boxes))
                if i != j and boxes[j] is not None
                and _unit_center_distance(bi, boxes[j]) < search_radius
            )
            it["density_factor"] = min(neighbour_count / N_MAX, 1.0)
            # Combined difficulty (Eq.4).
            it["difficulty_score"] = (
                ALPHA * it["size_factor"]
                + BETA * it["occlusion_factor"]
                + GAMMA * it["density_factor"]
            )
            # Eq.5 tier.
            if it["difficulty_score"] < TAU_1:
                it["difficulty_tier"] = "Easy"
            elif it["difficulty_score"] < TAU_2:
                it["difficulty_tier"] = "Medium"
            else:
                it["difficulty_tier"] = "Hard"
    # Strip the working "box" key before returning.
    for it in flat:
        it.pop("box", None)
    return flat


def stratified_coco_eval(gt_json: str, pred_json: str, difficulty_list: list,
                         iou_type: str = "segm") -> Dict[str, Dict[str, float]]:
    """Per-tier pycocotools eval. Ports ap_difficulty.stratified_evaluation
    (L665-767) with iou_type selectable ('bbox'|'segm'). The GT subset per tier
    keeps ``images`` intact and filters only ``annotations`` (ap_difficulty
    L729-740); loadRes re-keys predictions against the subset.

    Returns ``{'aggregate': {...}, 'Easy': {...}, 'Medium': {...}, 'Hard': {...}}``
    where each tier dict carries mAP50_95 / mAP50 / mAP75 / AR_100 / instances,
    plus a top-level ``'delta'`` dict
    ``{'mAP50_95': Easy-Hard, 'mAP50': Easy-Hard, 'mAP75': Easy-Hard,
    'AR_100': Easy-Hard}`` (delta IS a dict keyed by metric — NOT a scalar;
    callers extract ``tiers['delta']['mAP50_95']`` for the scalar diff_delta_*).

    pycocotools gotchas honoured (see the maxDets note below, and ap_difficulty):
      * ev.params.maxDets stays [1,10,100] (pycocotools default) inside every
        per-tier COCOeval — never propagate MAX_DETS=[1,10,1000] here.
        _summarize(1) hard-codes maxDets=100; if 100 is absent, stats[0]
        (mAP50_95) returns -1 → clamped to 0, silently zeroing the main metric.
      * predictions are filtered to the subset GT's valid image + category IDs
        before loadRes (coco_eval's clinical fix) so loadRes never sees a pred
        whose image_id/category_id is absent from the tier subset (it would
        otherwise KeyError on loadRes).
      * stats[8] is named AR_100 (bench schema name), not ap_difficulty's 'recall'.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as _e:
        _warn(f"pycocotools unavailable ({_e}); stratified_coco_eval returning zeros")
        zero = {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AR_100": 0.0, "instances": 0}
        return {t: dict(zero) for t in ("aggregate",) + DIFFICULTY_TIERS} | \
               {"delta": {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "AR_100")}}

    with open(gt_json) as fh:
        gt_data = json.load(fh)
    with open(pred_json) as fh:
        preds = json.load(fh)
    if not isinstance(preds, list) or not preds:
        _warn("stratified_coco_eval: empty predictions → all tiers zero")
        zero = {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AR_100": 0.0, "instances": 0}
        return {t: dict(zero) for t in ("aggregate",) + DIFFICULTY_TIERS} | \
               {"delta": {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "AR_100")}}

    # Map ann_id → tier, then the per-tier annotation IDs (ap_difficulty L692-700).
    ann_id_to_tier = {}
    for it in difficulty_list:
        if it.get("ann_id") is not None:
            ann_id_to_tier[it["ann_id"]] = it["difficulty_tier"]
    all_ann_ids = list(ann_id_to_tier.keys())

    results: Dict[str, Dict[str, float]] = {}
    for tier_name in ("aggregate",) + DIFFICULTY_TIERS:
        if tier_name == "aggregate":
            eval_ann_ids = all_ann_ids
        else:
            eval_ann_ids = [aid for aid, t in ann_id_to_tier.items() if t == tier_name]
        if not eval_ann_ids:
            _warn(f"stratified_coco_eval: no annotations in tier '{tier_name}'")
            results[tier_name] = {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0,
                                 "AR_100": 0.0, "instances": 0}
            continue

        # Per-tier GT subset: keep images intact, filter annotations only.
        eval_id_set = set(eval_ann_ids)
        tier_gt = {
            "images": gt_data["images"],
            "categories": gt_data["categories"],
            "annotations": [a for a in gt_data["annotations"] if a["id"] in eval_id_set],
        }
        tier_gt_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(tier_gt, tier_gt_tmp)
        tier_gt_tmp.close()
        try:
            tier_coco_gt = COCO(tier_gt_tmp.name)
            # Clinical pred filter (coco_eval): keep preds whose image_id + cat_id
            # exist in the subset, so loadRes never KeyErrors on an out-of-tier pred.
            valid_img_ids = set(tier_coco_gt.getImgIds())
            valid_cat_ids = set(tier_coco_gt.getCatIds())
            tier_preds = [p for p in preds
                          if p.get("image_id") in valid_img_ids
                          and p.get("category_id") in valid_cat_ids]
            if not tier_preds:
                _warn(f"stratified_coco_eval: no valid preds for tier '{tier_name}'")
                results[tier_name] = {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0,
                                     "AR_100": 0.0, "instances": len(eval_ann_ids)}
                continue
            tier_coco_dt = tier_coco_gt.loadRes(tier_preds)
            ev = COCOeval(tier_coco_gt, tier_coco_dt, iou_type)
            # CRITICAL: keep the pycocotools default maxDets=[1,10,100].
            # Do NOT set ev.params.maxDets = MAX_DETS here.
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            # Clamp pycocotools -1 (empty size/tier bucket) to 0 — matches
            # the main coco_eval path at L429 so a tier with no GT/dets reads
            # 0.0, not a negative that flips the Easy-Hard delta sign.
            results[tier_name] = {
                "mAP50_95": max(0.0, float(ev.stats[0])),
                "mAP50": max(0.0, float(ev.stats[1])),
                "mAP75": max(0.0, float(ev.stats[2])),
                "AR_100": max(0.0, float(ev.stats[8])),
                "instances": len(eval_ann_ids),
            }
        finally:
            try:
                os.unlink(tier_gt_tmp.name)
            except OSError:
                pass

    # delta = Easy - Hard per metric (ap_difficulty L762-765); dict, keyed by metric.
    easy = results.get("Easy", {})
    hard = results.get("Hard", {})
    results["delta"] = {
        k: float(easy.get(k, 0.0) - hard.get(k, 0.0))
        for k in ("mAP50_95", "mAP50", "mAP75", "AR_100")
    }
    return results


def coco_eval_full(gt_json: str,
                   bbox_pred_json: Optional[str] = None,
                   segm_pred_json: Optional[str] = None) -> Dict[str, float]:
    """Run both bbox + segm COCOeval against a single GT and return the merged
    bench-schema dict (box_mAP*..., seg_mAP*..., box_AR_100, seg_AR_100). Pass
    None for either pred-json to skip that iou_type.

    v3: Automatically calls remap_pred_categories for both in-place so 0-indexed
    predictions match 1-indexed GTs (typical for M2F/EoMT/RFDETR)."""
    out = {}
    out.update(_zero_box()); out.update(_zero_seg())
    if bbox_pred_json and Path(bbox_pred_json).is_file():
        remap_pred_categories(bbox_pred_json, gt_json, bbox_pred_json)
        out.update({k: v for k, v in coco_eval(gt_json, bbox_pred_json, "bbox").items()
                    if k in BOX_AP_KEYS})
    if segm_pred_json and Path(segm_pred_json).is_file():
        remap_pred_categories(segm_pred_json, gt_json, segm_pred_json)
        out.update({k: v for k, v in coco_eval(gt_json, segm_pred_json, "segm").items()
                    if k in SEG_AP_KEYS})
    return out


# =============================================================================
# 3. mask -> bbox: derive box predictions from segmentation predictions.
#    Used so seg-only models (EoMT / M2F / RF-DETR-Seg) get an honest
#    cross-task box mAP without re-running their heads.
# =============================================================================
def masks_to_box_preds(segm_pred_json: str, out_json: str) -> int:
    """Read a COCO-format predictions file containing `segmentation` (RLE or
    polygon) entries and write a sibling bbox predictions file with `bbox`
    derived as the tight enclosing rectangle of each mask. Returns the number
    of predictions written (0 on error)."""
    if not Path(segm_pred_json).is_file():
        _warn(f"masks_to_box_preds: {segm_pred_json} not found"); return 0
    try:
        from pycocotools import mask as _mu  # noqa
    except Exception as e:
        _warn(f"pycocotools.mask not importable ({e})"); return 0
    try:
        with open(segm_pred_json) as fh:
            preds = json.load(fh)
    except Exception as e:
        _warn(f"cannot read {segm_pred_json}: {e}"); return 0
    if isinstance(preds, dict):
        for key in ("annotations", "predictions", "results"):
            if isinstance(preds.get(key), list):
                preds = preds[key]
                break
    if not isinstance(preds, list):
        _warn(f"masks_to_box_preds: {segm_pred_json} is {type(preds).__name__}, expected list/dict-with-list")
        return 0
    import numpy as _np
    out = []
    dropped_no_image_id = 0
    dropped_no_mask_or_box = 0
    dropped_bad_box = 0
    for p in preds:
        if not isinstance(p, dict):
            continue
        # If the prediction already has a bbox, use it directly.
        bbox = p.get("bbox")
        if bbox is None:
            seg = p.get("segmentation")
            if seg is None:
                dropped_no_mask_or_box += 1
                continue
            try:
                if isinstance(seg, dict):  # RLE
                    rle = seg
                    # Some frameworks (M2F/EoMT dump) store 'counts' as ASCII string;
                    # pycocotools.mask.toBbox expects bytes for compressed RLE.
                    if "counts" in rle and isinstance(rle["counts"], str):
                        rle = {**rle, "counts": rle["counts"].encode("ascii")}

                    # Some frameworks emit list-of-list polygons in dict form; normalise.
                    if "counts" in rle and isinstance(rle["counts"], list):
                        rle = _mu.frPyObjects(rle, rle["size"][0], rle["size"][1])
                    res = _mu.toBbox(rle)
                    if hasattr(res, "reshape"):
                        bbox = res.reshape(-1).tolist()
                    else:
                        bbox = res.tolist() if hasattr(res, "tolist") else res
                else:  # polygon list-of-lists [[x1,y1,x2,y2,...], [...]]
                    all_pts = [pt for poly in seg for pt in poly]
                    if len(all_pts) >= 6:
                        xs = all_pts[0::2]; ys = all_pts[1::2]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            except Exception:
                dropped_bad_box += 1
                continue

        if bbox:
            # bbox must be [x, y, w, h] list. toBbox can return np.ndarray.
            if isinstance(bbox, _np.ndarray):
                bbox = bbox.tolist()
            if isinstance(bbox, list) and len(bbox) == 4:
                if "image_id" not in p:
                    dropped_no_image_id += 1
                    continue
                entry = {"image_id": p["image_id"], "category_id": p["category_id"],
                         "bbox": bbox, "score": p.get("score", 1.0)}
                out.append(entry)
            else:
                dropped_bad_box += 1
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh)
    if not out:
        _warn("masks_to_box_preds: wrote 0 boxes "
              f"(rows={len(preds)}, no_image_id={dropped_no_image_id}, "
              f"no_mask_or_box={dropped_no_mask_or_box}, bad_box={dropped_bad_box})")
    return len(out)


# =============================================================================
# 4. Semantic-segmentation metrics
# =============================================================================
def semseg_extra(conf_mat) -> Dict[str, float]:
    """Compute mIoU / mAcc / fwIoU / pixel_acc (+ pixel-level seg_P/R/F1) from
    a square confusion matrix (rows = GT class, cols = predicted class). All
    classes with zero GT pixels are excluded from mIoU/mAcc/P/R/F1 averages so
    a class absent from the test split doesn't drag the mean to 0.

    seg_P/R/F1 are pixel-level (not instance-level) for semantic-seg models —
    they fill the same CSV column YOLO-seg fills with its instance-level P/R/F1
    so the schema stays uniform. The fg_* columns repeat the same calculation
    with class 0 excluded, which is the paper-facing view for sparse foreground
    datasets where background dominates pixel_acc."""
    import numpy as _np
    cm = _np.asarray(conf_mat, dtype=_np.float64)
    zero = {"mIoU": 0.0, "mAcc": 0.0, "fwIoU": 0.0, "pixel_acc": 0.0,
            "seg_P": 0.0, "seg_R": 0.0, "seg_F1": 0.0,
            "fg_mIoU": 0.0, "fg_mAcc": 0.0, "fg_fwIoU": 0.0,
            "fg_pixel_acc": 0.0, "fg_boundary_IoU": 0.0,
            "fg_seg_P": 0.0, "fg_seg_R": 0.0, "fg_seg_F1": 0.0}
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.sum() == 0:
        return zero
    tp = _np.diag(cm)
    gt = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    iou = tp / _np.maximum(gt + pred - tp, 1e-12)
    acc = tp / _np.maximum(gt, 1e-12)
    # Per-class precision / recall / F1 from the same confusion matrix.
    prec = tp / _np.maximum(pred, 1e-12)
    rec  = tp / _np.maximum(gt,   1e-12)
    f1   = 2 * prec * rec / _np.maximum(prec + rec, 1e-12)
    present = gt > 0
    mIoU = float(iou[present].mean()) if present.any() else 0.0
    mAcc = float(acc[present].mean()) if present.any() else 0.0
    seg_P  = float(prec[present].mean()) if present.any() else 0.0
    seg_R  = float(rec[present].mean())  if present.any() else 0.0
    seg_F1 = float(f1[present].mean())   if present.any() else 0.0
    fw = gt / _np.maximum(gt.sum(), 1e-12)
    fwIoU = float((fw * iou).sum())
    pixel_acc = float(tp.sum() / max(1.0, cm.sum()))

    cls = _np.arange(cm.shape[0])
    fg = present & (cls > 0)
    fg_mIoU = float(iou[fg].mean()) if fg.any() else 0.0
    fg_mAcc = float(acc[fg].mean()) if fg.any() else 0.0
    fg_seg_P  = float(prec[fg].mean()) if fg.any() else 0.0
    fg_seg_R  = float(rec[fg].mean())  if fg.any() else 0.0
    fg_seg_F1 = float(f1[fg].mean())   if fg.any() else 0.0
    fg_gt = gt.copy()
    if len(fg_gt):
        fg_gt[0] = 0.0
    fg_fw = fg_gt / _np.maximum(fg_gt.sum(), 1e-12)
    fg_fwIoU = float((fg_fw * iou).sum()) if fg.any() else 0.0
    fg_pixel_acc = float(tp[1:].sum() / max(1.0, gt[1:].sum())) if len(tp) > 1 else 0.0
    return {"mIoU": mIoU, "mAcc": mAcc, "fwIoU": fwIoU, "pixel_acc": pixel_acc,
            "seg_P": seg_P, "seg_R": seg_R, "seg_F1": seg_F1,
            "fg_mIoU": fg_mIoU, "fg_mAcc": fg_mAcc, "fg_fwIoU": fg_fwIoU,
            "fg_pixel_acc": fg_pixel_acc, "fg_boundary_IoU": 0.0,
            "fg_seg_P": fg_seg_P,
            "fg_seg_R": fg_seg_R, "fg_seg_F1": fg_seg_F1}


def boundary_iou_mean(pred_label: "Any", gt_label: "Any",
                      num_classes: int, dilation_ratio: float = 0.02) -> float:
    """Mean Boundary-IoU (Cheng et al., 2021) across classes for a single
    image pair. Boundary is defined as the symmetric difference between the
    mask and its erosion by `dilation_ratio * image_diagonal` pixels.
    Inputs are HxW integer label maps."""
    try:
        import numpy as _np
        import cv2 as _cv2
    except Exception:
        return 0.0
    pred = _np.asarray(pred_label); gt = _np.asarray(gt_label)
    if pred.shape != gt.shape or pred.ndim != 2: return 0.0
    h, w = pred.shape
    d = max(1, int(round(dilation_ratio * (h * h + w * w) ** 0.5)))
    kernel = _np.ones((3, 3), _np.uint8)
    ious = []
    # COCO/mIoU standard: background (class 0) is excluded from the mean.
    # We iterate over all classes present in the ground truth label map.
    present_classes = [c for c in _np.unique(gt) if c > 0]
    if not present_classes: return 0.0
    for c in present_classes:
        gt_c = (gt == c).astype(_np.uint8)
        pr_c = (pred == c).astype(_np.uint8)
        # Even if not in pred, we must count it in the denominator of the mean.
        gt_b = gt_c - _cv2.erode(gt_c, kernel, iterations=d)
        pr_b = pr_c - _cv2.erode(pr_c, kernel, iterations=d)
        if gt_b.sum() == 0: 
            # If the object is too small for the boundary thickness, fallback to mask IoU
            gt_b, pr_b = gt_c, pr_c
        inter = _np.logical_and(gt_b, pr_b).sum()
        union = _np.logical_or(gt_b, pr_b).sum()
        ious.append(inter / max(1.0, union))
    return float(sum(ious) / len(ious)) if ious else 0.0


# =============================================================================
# 5. Performance: params + inference FPS
# =============================================================================
def model_params_M(model) -> float:
    """Total parameter count in millions; 0.0 if the object is not a torch
    nn.Module (e.g. a YOLO() wrapper has .model)."""
    try:
        import torch  # noqa
        m = getattr(model, "model", model)
        if not hasattr(m, "parameters"): return 0.0
        n = sum(p.numel() for p in m.parameters())
        return n / 1e6
    except Exception as e:
        _warn(f"model_params_M failed: {e}"); return 0.0


def time_fps(forward_fn, n_warmup: int = 3, n_iter: int = 20) -> float:
    """Call `forward_fn()` n_warmup + n_iter times and return mean iterations
    per second over the timed window. Synchronises CUDA when available so the
    timing reflects the GPU side, not just kernel launch. `forward_fn` is the
    user's own zero-arg closure (so this helper stays framework-agnostic)."""
    try:
        import torch  # noqa
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    try:
        for _ in range(max(0, n_warmup)): forward_fn()
        if has_cuda:
            import torch as _t; _t.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, n_iter)): forward_fn()
        if has_cuda:
            import torch as _t; _t.cuda.synchronize()
        dt = time.perf_counter() - t0
        return float(n_iter / dt) if dt > 0 else 0.0
    except Exception as e:
        _warn(f"time_fps failed: {e}"); return 0.0


# =============================================================================
# 5b. Detector / instance-seg confusion matrix (paper figure)
#     Mirrors Ultralytics' confusion_matrix.png: each GT box is matched 1-to-1
#     to its highest-IoU predicted box (above iou_thr and above conf_thr) of
#     any class; pairs vote at [gt_class, pred_class]; unmatched GTs vote at
#     [gt_class, background]; unmatched preds vote at [background, pred_class].
#     The result is an (N+1)x(N+1) matrix where the last row/col is "bg".
# =============================================================================
def detector_confusion_matrix(gt_json: str, pred_json: str, class_names: List[str],
                              iou_thr: float = 0.5, conf_thr: float = 0.25):
    """Build a confusion matrix from COCO-format GT and predictions.

    Returns (cm, labels) where ``cm`` is a NumPy array of shape (N+1, N+1) and
    ``labels`` is ``list(class_names) + ['background']``. Empty matrix on any
    failure so callers can rely on .size to decide whether to plot.
    """
    try:
        import numpy as _np
    except Exception as e:
        _warn(f"detector_confusion_matrix: numpy unavailable ({e})")
        return None, list(class_names) + ["background"]
    if not (Path(gt_json).is_file() and Path(pred_json).is_file()):
        return _np.zeros((0, 0)), list(class_names) + ["background"]
    try:
        with open(gt_json) as fh:
            gt = json.load(fh)
        with open(pred_json) as fh:
            preds = json.load(fh)
    except Exception as e:
        _warn(f"detector_confusion_matrix: json read failed ({e})")
        return _np.zeros((0, 0)), list(class_names) + ["background"]

    n = max(1, len(class_names))
    cm = _np.zeros((n + 1, n + 1), dtype=_np.int64)
    bg = n  # background index

    # COCO category_id -> contiguous class index 0..N-1.
    cats = sorted({c["id"] for c in gt.get("categories", [])})
    if not cats:
        cats = list(range(n))
    cat_to_idx = {c: i for i, c in enumerate(cats[:n])}

    # Group GT and predictions by image_id.
    gt_by_img: dict = {}
    for ann in gt.get("annotations", []):
        gt_by_img.setdefault(int(ann["image_id"]), []).append(ann)
    pred_by_img: dict = {}
    for p in preds:
        if float(p.get("score", 1.0)) < conf_thr:
            continue
        pred_by_img.setdefault(int(p["image_id"]), []).append(p)

    def _xywh_to_xyxy(b):
        x, y, w, h = b
        return x, y, x + w, y + h

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    all_img_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())
    for img_id in all_img_ids:
        gts = gt_by_img.get(img_id, [])
        prs = sorted(pred_by_img.get(img_id, []),
                     key=lambda p: -float(p.get("score", 0.0)))
        gt_used = [False] * len(gts)
        pr_used = [False] * len(prs)
        # Greedy IoU matching (highest-score pred first picks its best GT).
        for pi, p in enumerate(prs):
            best_iou = iou_thr
            best_gi = -1
            pbox = _xywh_to_xyxy(p["bbox"])
            for gi, g in enumerate(gts):
                if gt_used[gi]:
                    continue
                gbox = _xywh_to_xyxy(g["bbox"])
                iou = _iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou; best_gi = gi
            if best_gi >= 0:
                gt_used[best_gi] = True
                pr_used[pi] = True
                gi = cat_to_idx.get(int(gts[best_gi]["category_id"]), 0)
                pj = cat_to_idx.get(int(p["category_id"]), 0)
                cm[gi, pj] += 1
        # Unmatched GTs -> predicted as background.
        for gi, used in enumerate(gt_used):
            if used:
                continue
            ci = cat_to_idx.get(int(gts[gi]["category_id"]), 0)
            cm[ci, bg] += 1
        # Unmatched predictions -> background was matched as that class.
        for pi, used in enumerate(pr_used):
            if used:
                continue
            cj = cat_to_idx.get(int(prs[pi]["category_id"]), 0)
            cm[bg, cj] += 1
    return cm, list(class_names) + ["background"]


# =============================================================================
# 6. Optional: SAM2 prompted segmentation from detector boxes
#    Lets pure detectors (RT-DETR, YOLO-det) get a seg_mAP* column for
#    completeness. Clearly labeled as "(SAM2)" in the bench summary because the
#    mask quality is SAM2's, not the detector's.
# =============================================================================
def sam2_seg_from_boxes(images_dir: str, bbox_pred_json: str,
                         sam2_ckpt: str, sam2_cfg: str,
                         gt_json: str, out_json: str) -> Dict[str, float]:
    """Read bbox predictions, prompt SAM2 with each box on the matching image,
    encode the returned mask as COCO RLE, and run a segm COCOeval against
    `gt_json`. Returns the sam2_* metric dict (zeros if SAM2 isn't installed).

    NB: results are reported under the sam2_* family so SAM2-prompted masks
    never overwrite a model's native seg_* mask-head columns (e.g. YOLO-seg
    reports both native seg_* and prompted sam2_*)."""
    blank = _zero_sam2()
    try:
        # Add SAM2 to Python path
        import sys
        sam2_root = os.environ.get("SAM2_ROOT", "/home/user/sam2")
        if sam2_root not in sys.path:
            sys.path.insert(0, sam2_root)

        # Initialize Hydra to point to SAM2 config dir BEFORE importing sam2
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=f"{sam2_root}/sam2/configs", version_base="1.2")

        from sam2.build_sam import build_sam2  # noqa
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa
        from pycocotools import mask as _mu  # noqa
        import numpy as _np, cv2 as _cv2, torch as _t
    except Exception as e:
        _warn(f"SAM2 stack not importable ({e}); seg_via_sam2 disabled"); return blank
    if not (Path(bbox_pred_json).is_file() and Path(gt_json).is_file()):
        _warn("sam2_seg_from_boxes: bbox_pred_json or gt_json missing"); return blank
    if not Path(sam2_ckpt).is_file():
        _warn(f"sam2 checkpoint not found: {sam2_ckpt}"); return blank

    # build_sam2 expects a relative config path from the Hydra config dir
    # If sam2_cfg is absolute, convert to relative
    cfg_path = Path(sam2_cfg)
    if cfg_path.is_absolute():
        # Extract relative path: /home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
        # becomes: sam2.1/sam2.1_hiera_l.yaml
        try:
            cfg_rel = str(cfg_path.relative_to(f"{sam2_root}/sam2/configs"))
        except ValueError:
            cfg_rel = sam2_cfg  # fallback to original
    else:
        cfg_rel = sam2_cfg

    try:
        device = "cuda" if _t.cuda.is_available() else "cpu"
        sam2 = build_sam2(cfg_rel, sam2_ckpt, device=device)
        predictor = SAM2ImagePredictor(sam2)
    except Exception as e:
        _warn(f"SAM2 build failed: {e}"); return blank

    with open(bbox_pred_json) as fh:
        preds = json.load(fh)
    with open(gt_json) as fh:
        gt = json.load(fh)
    id_to_fname = {im["id"]: im["file_name"] for im in gt.get("images", [])}

    # Bucket predictions by image so we set the predictor's image once per file.
    by_image: Dict[int, List[dict]] = {}
    for p in preds:
        by_image.setdefault(p["image_id"], []).append(p)

    # All boxes of an image go through SAM2 in one prompt batch; chunked so a
    # crowded image (hundreds of penguins) can't blow up decoder VRAM.
    # cvBench speed port: BENCH_SAM2_BOX_CHUNK=64 batches the per-box prompt
    # loop (was one predict() call per box in fp32). BENCH_SAM2_BF16=1 runs the
    # prompt under autocast(bf16) — ~0.004 sam2_mAP50 cost, 3.85x on top of the
    # batching (measured BASE 2.03s -> PORT 0.53s on 479 box prompts / 5 images,
    # 479/479 masks, mAP parity). Set BENCH_SAM2_BF16=0 or BENCH_SAM2_BOX_CHUNK=1
    # to restore the exact old per-box fp32 numbers.
    import os as _os
    chunk = int(_os.environ.get("BENCH_SAM2_BOX_CHUNK", "64"))
    autocast = (device == "cuda") and _os.environ.get("BENCH_SAM2_BF16", "1") == "1"

    out_preds = []
    for iid, ps in by_image.items():
        fname = id_to_fname.get(iid)
        if not fname:
            continue
        img_path = Path(images_dir) / fname
        if not img_path.is_file():
            continue
        try:
            img = _cv2.imread(str(img_path))
            if img is None: continue
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            with _t.inference_mode(), _t.autocast("cuda", dtype=_t.bfloat16, enabled=autocast):
                predictor.set_image(img)
        except Exception as e:
            _warn(f"SAM2.set_image failed on {img_path.name}: {e}"); continue
        boxes = _np.array([[p["bbox"][0], p["bbox"][1],
                            p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
                           for p in ps], dtype=_np.float32)
        for i in range(0, len(ps), chunk):
            blk, pblk = boxes[i:i + chunk], ps[i:i + chunk]
            try:
                with _t.inference_mode(), _t.autocast("cuda", dtype=_t.bfloat16, enabled=autocast):
                    masks, _scores, _ = predictor.predict(box=blk, multimask_output=False)
            except Exception:
                continue
            if masks is None or len(masks) == 0: continue
            # predict() squeezes dim 0, so a single box returns (1,H,W) and N
            # boxes return (N,1,H,W); one mask per box either way.
            masks = _np.asarray(masks).reshape(-1, *_np.asarray(masks).shape[-2:])
            for p, m in zip(pblk, masks):
                rle = _mu.encode(_np.asfortranarray(m.astype(_np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                out_preds.append({"image_id": iid, "category_id": p["category_id"],
                                  "segmentation": rle, "score": float(p.get("score", 1.0))})
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out_preds, fh)
    # Run a segm COCOeval (it emits seg_* keys) and remap into the sam2_* family
    # so SAM2-prompted masks don't overwrite a model's native seg_* columns.
    seg_res = coco_eval(gt_json, out_json, "segm")
    sam2_res = {"sam2_" + k[len("seg_"):]: v for k, v in seg_res.items() if k in SEG_AP_KEYS}
    # Guard the native-vs-prompted routing: every returned key must be sam2_*.
    # A slip here lets a SAM2-prompted score masquerade as a native seg_* score.
    assert all(k.startswith("sam2_") for k in sam2_res), (
        f"sam2_seg_from_boxes must return only sam2_* keys; got {sorted(sam2_res)}")
    return sam2_res
