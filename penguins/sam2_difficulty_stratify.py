#!/usr/bin/env python
"""Difficulty-stratify the SAM2-prompted masks (_bench_sam2_seg.json) per paradigm.

Companion to difficulty_stratify_posthoc.py: that tool stratifies the NATIVE
pred file per run dir. This one stratifies the SAM2-prompted seg file so
tab:ablation row 764 ("+ SAM2-prompted dual-mask") in P1 cvml_paper.tex can be
populated with real Easy/Med/Hard mAP50 numbers.

Pure CPU (pycocotools eval only). Reads the same frozen run dirs as Chunk B.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

# Reuse the frozen run dirs + difficulty machinery from the bench harness.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_eval as _be
from difficulty_stratify_posthoc import discover_gt_pred  # same per-framework gt discovery as Chunk B

# 7 paradigms with SAM2-prompted seg files (the 6 ablation paradigms + rtdetr-3,
# which P2 tab:cross also needs as a pure-detector SAM2-prompted row).
PARADIGMS = [
    ("yolo11x-seg",      "YOLOv11x"),
    ("yolo26x-seg",      "YOLOv26x"),
    ("rtdetr-3",         "RT-DETR-v3"),
    ("rtdetr-4",         "RT-DETR-v4"),
    ("rf-detr-seg-large","RF-DETR"),
    ("eomt-large",       "EoMT"),
    ("dinov3-m2f-vith16plus", "DINOv3-M2F"),
]

RUNS_ROOT = Path.home() / "bench_any" / "runs"
OUT_CSV = Path.home() / "bench_any" / "runs" / "segment" / "research_outputs" / "difficulty_stratified_sam2_150e.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def _find_run_dir(tag: str) -> Path | None:
    seg_hits = sorted(RUNS_ROOT.glob(f"segment/african-penguin_*{tag}*e150*"))
    if seg_hits:
        return seg_hits[0]
    det_hits = sorted(RUNS_ROOT.glob(f"detect/african-penguin_*{tag}*e150*"))
    if det_hits:
        return det_hits[0]
    return None


def main() -> int:
    rows: list[dict] = []
    for tag, label in PARADIGMS:
        rd = _find_run_dir(tag)
        if rd is None:
            print(f"[sam2-stratify] {tag}: run dir not found", file=sys.stderr)
            continue
        # GT lives at run-dir root (RF-DETR/EoMT/DINOv3/RT-DETR) OR under test/
        # (YOLO). Reuse Chunk B's discover_gt_pred so RT-DETR's _rtdetrv4_coco/
        # instances_test.json is found too.
        gt, _native_pred = discover_gt_pred(rd, "test")
        if gt is None or not gt.exists():
            print(f"[sam2-stratify] {tag}: gt not found in {rd.name}", file=sys.stderr)
            rows.append(dict(model_tag=tag, paradigm=label, status="no_gt"))
            continue
        sam2_pred = rd / "_bench_sam2_seg.json"
        if not sam2_pred.exists():
            sam2_pred = rd / "test" / "_bench_sam2_seg.json"  # YOLO keeps test artifacts here
        if not sam2_pred.exists():
            # Detect runs (RT-DETR) keep sam2 seg at run-dir root.
            sam2_pred = rd / "_bench_sam2_seg.json"
        # pycocotools loadRes needs a 'bbox' field (and 'area') even for segm
        # preds; the SAM2-prompted seg file carries only segmentation+score.
        # Materialise bbox/area from the RLE into a temp file.
        sam2_pred = _materialise_bbox(sam2_pred, tag)
        if not sam2_pred.exists():
            print(f"[sam2-stratify] {tag}: no _bench_sam2_seg.json in {rd.name}", file=sys.stderr)
            rows.append(dict(model_tag=tag, paradigm=label, status="no_sam2_pred"))
            continue
        dlist = _be.compute_difficulty_from_coco_gt(str(gt))
        with _be._suppress_stdout() if hasattr(_be, "_suppress_stdout") else _nullctx():
            tiers = _be.stratified_coco_eval(str(gt), str(sam2_pred), dlist,
                                             iou_type="segm")
        agg = tiers["aggregate"]; e = tiers["Easy"]; m = tiers["Medium"]; h = tiers["Hard"]
        rows.append(dict(
            model_tag=tag, paradigm=label, status="ok",
            agg_mAP50=agg["mAP50"], agg_mAP50_95=agg["mAP50_95"],
            easy_mAP50=e["mAP50"], medium_mAP50=m["mAP50"], hard_mAP50=h["mAP50"],
            easy_mAP50_95=e["mAP50_95"], medium_mAP50_95=m["mAP50_95"], hard_mAP50_95=h["mAP50_95"],
            delta_mAP50=e["mAP50"]-h["mAP50"], delta_mAP50_95=e["mAP50_95"]-h["mAP50_95"],
            easy_n=e["instances"], medium_n=m["instances"], hard_n=h["instances"],
            gt_json=str(gt), sam2_pred_json=str(sam2_pred),
        ))
        print(f"[sam2-stratify] {tag:24s} agg={agg['mAP50']:.3f} E={e['mAP50']:.2f} M={m['mAP50']:.2f} H={h['mAP50']:.2f}")

    if rows:
        keys = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"\n[sam2-stratify] wrote {OUT_CSV}")
    return 0


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


_TMP_DIR = Path("/dev/shm/sam2_stratify")


def _materialise_bbox(sam2_json: Path, tag: str) -> Path:
    """Add bbox+area (derived from RLE mask) to SAM2 seg preds so loadRes works.
    Cached in /dev/shm keyed by tag+mtime; reuses pycocotools mask.toBbox."""
    from pycocotools import mask as maskUtils
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = _TMP_DIR / f"_sam2_bbox_{tag}.json"
    if out.exists() and out.stat().st_mtime >= sam2_json.stat().st_mtime:
        return out
    with open(sam2_json) as fh:
        preds = json.load(fh)
    for p in preds:
        seg = p["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:  # RLE (counts str/bytes = compressed)
            # counts already a compressed-RLE string -> use directly, not frPyObjects.
            if isinstance(seg["counts"], (str, bytes)):
                rle = {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}
            else:
                rle = maskUtils.frPyObjects(seg, *seg["size"])
            bbox = maskUtils.toBbox(rle).tolist()  # [x,y,w,h] (1D for single RLE)
            bbox = bbox[0] if isinstance(bbox[0], list) else bbox
            p["bbox"] = [float(x) for x in bbox]
            area = maskUtils.area(rle).tolist()
            if isinstance(area, list):
                area = area[0] if area and isinstance(area[0], list) else area
            p["area"] = float(area) if not isinstance(area, list) else float(area[0])
        elif isinstance(seg, list):  # polygon
            import numpy as np
            xs, ys = [], []
            for poly in seg:
                xs += poly[0::2]; ys += poly[1::2]
            p["bbox"] = [float(min(xs)), float(min(ys)),
                         float(max(xs) - min(xs)), float(max(ys) - min(ys))]
            p["area"] = float((max(xs)-min(xs)) * (max(ys)-min(ys)))
        else:
            p["bbox"] = [0.0, 0.0, 0.0, 0.0]; p["area"] = 0.0
    with open(out, "w") as fh:
        json.dump(preds, fh)
    print(f"[sam2-stratify] materialised bbox for {len(preds)} preds -> {out}")
    return out


if __name__ == "__main__":
    sys.exit(main())
