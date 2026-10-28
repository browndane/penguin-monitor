#!/usr/bin/env python
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Script 1: COCO -> YOLO converter for Penguin-Watch.

For each of the 16 PW sub-colony dirs, read _annotations.coco.json,
decode each annotation's RLE segmentation with pycocotools.mask.decode,
convert non-empty masks to YOLO polygon format (normalized by image w/h).
If a mask decodes to >1 connected component or is degenerate, fall back
to a 4-corner bbox rectangle polygon (normalized). Skip empty masks.

Output into ~/bench_any/datasets/pw_yolo_staging/<subcolony>/{images,labels}.
Sub-colony code is kept as a prefix in every image filename so the
repartitioner can group by sub-colony. Images are symlinked; labels are real files.

Usage:
    python prep_01_pw_coco_to_yolo.py [--overwrite]
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
import cv2
# Prevent OpenCV thread oversubscription when running parallel workers.
# Each worker process should use a single OpenCV thread; the parallelism comes
# from the ProcessPoolExecutor, not from OpenCV's internal threading.
cv2.setNumThreads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# The raw unverified tree was deleted 2026-07-31; this is the verified Roboflow export.
SRC = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/penguin-watch")
STAGING = Path(os.path.expanduser("~/bench_any/datasets/pw_yolo_staging"))
# forced lowercase .jpg extension for YOLO friendliness; original is .JPG
OUT_EXT = ".jpg"
LABEL_EXT = ".txt"
CLASS_IDX = 0  # penguin -> single class

def decode_rle(seg):
    """Return uint8 HxW mask (0/1) from a COCO RLE seg dict, or None on failure."""
    try:
        s = dict(seg)
        counts = s.get("counts")
        # pycocotools.mask.decode requires bytes counts
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        rle = {"size": s["size"], "counts": counts}
        m = maskUtils.decode(rle)
        if m.ndim != 2 or m.size == 0:
            return None
        return m
    except Exception:
        return None

def mask_to_polygon(mask, w, h):
    """Return a normalized polygon list [x1,y1,x2,y2,...] for the largest
    external contour of mask, or None if degenerate/empty.
    If >1 external contour exists (multi-component), signal ('multi', None)
    so the caller falls back to bbox rectangle.
    Optimized: uses cv2.findContours with CHAIN_APPROX_TC89_L1 (faster than SIMPLE)
    and skips approxPolyDP simplification (TC89 already compresses well).
    """
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return ("empty", None)
    if len(contours) > 1:
        return ("multi", None)
    cnt = contours[0]
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 3:
        return ("degen", None)
    # TC89_L1 already compresses; only simplify if extremely large
    if pts.shape[0] > 500:
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float64)
        if approx.shape[0] >= 3:
            pts = approx
    poly = []
    for x, y in pts:
        poly.append(x / w)
        poly.append(y / h)
    poly = [min(1.0, max(0.0, v)) for v in poly]
    return ("poly", poly)

def bbox_rect_polygon(bbox, w, h):
    """4-corner rectangle polygon normalized, from COCO bbox [x,y,w,h]."""
    x, y, bw, bh = bbox
    x1, y1 = x, y
    x2, y2 = x + bw, y + bh
    # normalized corners clockwise
    return [
        x1 / w, y1 / h,
        x2 / w, y1 / h,
        x2 / w, y2 / h,
        x1 / w, y2 / h,
    ]

def fmt_line(poly):
    parts = [str(CLASS_IDX)]
    for v in poly:
        parts.append(f"{v:.8f}")
    return " ".join(parts)

def process_subcolony(sub, overwrite=False):
    sub_dir = SRC / sub
    jf = sub_dir / "_annotations.coco.json"
    if not jf.is_file():
        return {"sub": sub, "error": "no json"}
    out_img_dir = STAGING / sub / "images"
    out_lbl_dir = STAGING / sub / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(jf) as f:
        data = json.load(f)
    imgs = data.get("images", [])
    anns = data.get("annotations", [])
    # index anns by image_id
    anns_by_img = {}
    for a in anns:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    n_img = len(imgs)
    n_ann = len(anns)
    n_lbl_written = 0
    n_empty_lbl = 0  # image had >=1 ann but all masks empty -> we still write empty label? NO, skip
    n_img_no_ann = 0
    zero_mask_images = []  # images where NO valid mask produced (no anns OR all anns empty)
    counts = {"images": n_img, "anns": n_ann, "labels_written": 0,
              "bbox_fallback": 0, "multi_comp": 0, "empty_mask": 0, "iscrowd_skip": 0,
              "degen": 0, "symlink_created": 0, "symlink_exists": 0}
    for im in imgs:
        im_id = im["id"]
        fname = im["file_name"]
        w = im["width"]; h = im["height"]
        stem = os.path.splitext(fname)[0]
        new_name = f"PW_{sub}_{stem}{OUT_EXT}"
        new_lbl = f"PW_{sub}_{stem}{LABEL_EXT}"
        # symlink image
        src_img = sub_dir / fname
        dst_img = out_img_dir / new_name
        if not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
                counts["symlink_created"] += 1
            except FileExistsError:
                counts["symlink_exists"] += 1
        else:
            counts["symlink_exists"] += 1
        # build label lines
        lines = []
        img_anns = anns_by_img.get(im_id, [])
        if not img_anns:
            n_img_no_ann += 1
            zero_mask_images.append(new_name)
            # write empty label file so YOLO has a matching file? YOLO expects label per image;
            # write empty file to keep 1:1 mapping.
            (out_lbl_dir / new_lbl).write_text("")
            continue
        any_valid = False
        for a in img_anns:
            if a.get("iscrowd", 0) == 1:
                counts["iscrowd_skip"] += 1
                continue
            seg = a.get("segmentation")
            if not isinstance(seg, dict):
                # not RLE; skip (shouldn't happen for PW)
                counts["empty_mask"] += 1
                continue
            m = decode_rle(seg)
            if m is None or m.sum() == 0:
                counts["empty_mask"] += 1
                continue
            kind, poly = mask_to_polygon(m, w, h)
            if kind == "empty":
                counts["empty_mask"] += 1
                continue
            if kind == "multi":
                counts["multi_comp"] += 1
                # fallback to bbox rectangle polygon
                poly = bbox_rect_polygon(a["bbox"], w, h)
                lines.append(fmt_line(poly))
                counts["bbox_fallback"] += 1
                any_valid = True
                continue
            if kind == "degen":
                counts["degen"] += 1
                poly = bbox_rect_polygon(a["bbox"], w, h)
                lines.append(fmt_line(poly))
                counts["bbox_fallback"] += 1
                any_valid = True
                continue
            # kind == poly
            lines.append(fmt_line(poly))
            any_valid = True
        if any_valid:
            (out_lbl_dir / new_lbl).write_text("\n".join(lines) + "\n")
            n_lbl_written += 1
            counts["labels_written"] += 1
        else:
            # all anns for this image produced nothing valid -> empty label
            (out_lbl_dir / new_lbl).write_text("")
            zero_mask_images.append(new_name)
    elapsed = time.time() - t0
    return {
        "sub": sub, "n_img": n_img, "n_ann": n_ann,
        "labels_with_content": n_lbl_written,
        "images_no_ann": n_img_no_ann,
        "zero_mask_images_n": len(zero_mask_images),
        "zero_mask_images": zero_mask_images[:20],  # cap for report
        "counts": counts, "elapsed_s": round(elapsed, 1),
    }

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="*", default=None, help="subset of sub-colonies to process")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="parallel workers (one per sub-colony)")
    args = ap.parse_args()
    STAGING.mkdir(parents=True, exist_ok=True)
    subs = args.subs or sorted([d for d in os.listdir(SRC) if (SRC / d / "_annotations.coco.json").is_file()])
    print(f"Processing {len(subs)} sub-colonies -> {STAGING} (workers={min(args.workers, len(subs))})", flush=True)
    results = []
    # parallel across sub-colonies; each is independent (own JSON, own output dir)
    n_workers = min(args.workers, len(subs))
    # use spawn to be safe (no shared state; each worker re-imports)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        future_to_sub = {ex.submit(process_subcolony, sub, args.overwrite): sub for sub in subs}
        for fut in as_completed(future_to_sub):
            sub = future_to_sub[fut]
            try:
                r = fut.result()
                print(f"  [{sub}] imgs={r.get('n_img')} anns={r.get('n_ann')} labels={r.get('labels_with_content')} "
                      f"zero_mask={r.get('zero_mask_images_n')} t={r.get('elapsed_s')}s", flush=True)
                results.append(r)
            except Exception as e:
                print(f"  [{sub}] ERROR: {e}", flush=True)
                results.append({"sub": sub, "error": str(e)})
    # sort results by sub-colony name for deterministic manifest
    results.sort(key=lambda r: r.get("sub", ""))
    # write summary manifest
    mf = STAGING / "prep_01_manifest.json"
    with open(mf, "w") as f:
        json.dump({"source": str(SRC), "staging": str(STAGING),
                    "subcolonies": results}, f, indent=2)
    print(f"\nManifest -> {mf}", flush=True)
    # totals
    tot_img = sum(r.get("n_img",0) for r in results)
    tot_ann = sum(r.get("n_ann",0) for r in results)
    tot_lbl = sum(r.get("labels_with_content",0) for r in results)
    tot_fb = sum(r.get("counts",{}).get("bbox_fallback",0) for r in results)
    tot_multi = sum(r.get("counts",{}).get("multi_comp",0) for r in results)
    tot_empty = sum(r.get("counts",{}).get("empty_mask",0) for r in results)
    print(f"TOTAL imgs={tot_img} anns={tot_ann} labels_with_content={tot_lbl} "
          f"bbox_fallback={tot_fb} multi_comp={tot_multi} empty_mask={tot_empty}", flush=True)

if __name__ == "__main__":
    main()
