#!/usr/bin/env python3
"""
Sanity sieve for raw SAM-3 candidate masks in dense penguin colonies.
====================================================================

Three geometric tests, applied *before* class-agnostic NMS so nonsense masks cannot
suppress real ones via IoU overlap. Spec: Elsevier *Ecological Informatics* draft,
``elsarticle-template-harv.tex`` §ssec:sieve (lines ~220-238).

Failure modes targeted (verbatim from the draft):
  * Hyper-fragmented masks   -- mask pieces too small to correspond to a real penguin
                                 (dot-sized fragments triggered on rocks or guano).
  * Thin-strip merge masks    -- >90% of one bbox dimension occupied by a thin strip
                                 that is actually a sliver of merged neighbours.
  * Edge-stub masks            -- sub-flipper stubs clipped by the image border.

Tests (a mask is REJECTED iff any holds):
  1. Area:                  area < A_min,                 A_min = 324 px (18 x 18).
  2. Dimension-fraction:     max-dim fraction rho >= rho_max,   rho_max = 1.0
                            (a mask that spans the whole of one bbox dimension is a
                             strip / merge artefact).
  3. Edge-stub:             within 4 px of image border AND area < 8 * A_min.
                            This kills sub-flipper stubs without penalising legitimate
                             edge penguins that happen to touch the border.

After sieving, class-agnostic NMS at IoU 0.55 deduplicates the surviving boxes.

Pure geometry on a mask / bbox + image shape. No torch, no I/O.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (spec, §ssec:sieve)
# ---------------------------------------------------------------------------
A_MIN: float = 324.0          # px -- 18 x 18
RHO_MAX: float = 0.9          # dimension-fraction threshold (tightened from 1.0, which rejected only full-frame-spanning masks)
EDGE_BORDER: float = 4.0      # px -- "within 4 px of image border"
EDGE_STUB_AREA_K: float = 8.0  # reject if area < K * A_min near the border
NMS_IOU: float = 0.55         # class-agnostic NMS threshold (post-sieve)


@dataclass
class SieveStats:
    """Per-test rejection counters for a single image's masks."""
    n_in: int = 0
    n_reject_area: int = 0
    n_reject_dim: int = 0
    n_reject_edge: int = 0
    n_reject_nms: int = 0
    n_out: int = 0

    @property
    def n_rejected_pre_nms(self) -> int:
        return self.n_reject_area + self.n_reject_dim + self.n_reject_edge


# ---------------------------------------------------------------------------
# Mask representation
# ---------------------------------------------------------------------------
# A mask is a 2-D uint8/bool array (H x W) OR a tight bbox + area. We support both
# because SAM-3 emits dense masks but the sieve's tests only need: area, bbox extent,
# and (for the edge test) the mask's reach toward each image border.
#
# Tight bbox encodings are accepted for callers that only have boxes (e.g. a detector
# whose masks were discarded). In that case the "area" used by tests 1 and 3 is the
# bbox area, and the edge reach is the bbox itself -- a conservative over-estimate
# that rejects *more* aggressively, which is the safe direction for a sieve.


def _mask_area(mask: np.ndarray) -> int:
    """Pixel count of the foreground (non-zero) region of a single mask."""
    return int(np.count_nonzero(mask))


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Tight bbox of a single 2-D mask as (x1, y1, x2, y2) inclusive pixel coords.

    Returns (-1, -1, -1, -1) for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (-1, -1, -1, -1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    """Inclusive-pixel bbox area (a 1-px-wide line has area 1)."""
    if x2 < x1 or y2 < y1:
        return 0
    return (x2 - x1 + 1) * (y2 - y1 + 1)


def _bbox_from_any(mask: np.ndarray | None, bbox: Tuple[int, int, int, int] | None
                   ) -> Tuple[int, int, int, int]:
    """Resolve to a (x1, y1, x2, y2) bbox. Mask wins if supplied and non-empty."""
    if mask is not None:
        bb = _mask_bbox(mask)
        if bb[0] >= 0:
            return bb
    if bbox is not None:
        return bbox
    return (-1, -1, -1, -1)


def _area_from_any(mask: np.ndarray | None, bbox: Tuple[int, int, int, int] | None,
                   area: int | None) -> int:
    """Resolve to a pixel area. Explicit area wins, then mask count, then bbox area."""
    if area is not None and area > 0:
        return int(area)
    if mask is not None and mask.size > 0:
        a = _mask_area(mask)
        if a > 0:
            return a
    if bbox is not None:
        return _bbox_area(*bbox)
    return 0


# ---------------------------------------------------------------------------
# The three tests
# ---------------------------------------------------------------------------
def reject_area(area: int, a_min: float = A_MIN) -> bool:
    """Test 1: hyper-fragmented / dot-sized fragment."""
    return area < a_min


def reject_dimension(bbox: Tuple[int, int, int, int], img_wh: Tuple[int, int],
                     rho_max: float = RHO_MAX) -> bool:
    """Test 2: thin-strip / merge artefact (spec, verbatim).

    Reject if either bounding-box axis exceeds a fraction ``rho_max`` of the
    corresponding image dimension. With ``rho_max = 0.9`` the test rejects any mask
    whose bounding box spans more than 90% of one image axis, catching thin-strip
    merge artefacts stitched across a row of penguins. The earlier default of 1.0
    rejected only masks spanning the full frame and was effectively a no-op.
    """
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        return True  # degenerate bbox -> reject
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    img_w, img_h = img_wh
    if img_w <= 0 or img_h <= 0:
        return False
    # Fraction of each image axis the bbox spans.
    fx = w / float(img_w)
    fy = h / float(img_h)
    rho = max(fx, fy)
    return rho >= rho_max


def reject_edge_stub(bbox: Tuple[int, int, int, int], img_wh: Tuple[int, int],
                     area: int, a_min: float = A_MIN,
                     border: float = EDGE_BORDER,
                     area_k: float = EDGE_STUB_AREA_K) -> bool:
    """Test 3: edge-stub -- within ``border`` px of the image edge AND small.

    A mask that touches the image border is *kept* unless it is also tiny (area <
    ``area_k * a_min``). This kills sub-flipper stubs clipped by the border without
    penalising legitimate edge penguins that happen to touch the border.
    """
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        return True
    img_w, img_h = img_wh
    near_edge = (x1 < border or y1 < border or
                 x2 > (img_w - 1) - border or y2 > (img_h - 1) - border)
    if not near_edge:
        return False
    return area < area_k * a_min


def mask_passes(mask: np.ndarray | None,
                img_wh: Tuple[int, int],
                bbox: Tuple[int, int, int, int] | None = None,
                area: int | None = None,
                a_min: float = A_MIN,
                rho_max: float = RHO_MAX,
                border: float = EDGE_BORDER,
                area_k: float = EDGE_STUB_AREA_K) -> bool:
    """Return True iff a single mask survives all three sieve tests.

    ``img_wh`` is (width, height) of the source image. Pass a dense ``mask`` for the
    tightest geometry; if only a ``bbox`` (+ optional ``area``) is available, the
    tests run on the box (conservatively -- rejects more, safe for a sieve).
    """
    bb = _bbox_from_any(mask, bbox)
    if bb[0] < 0:
        return False  # empty mask
    a = _area_from_any(mask, bbox, area)
    if reject_area(a, a_min):
        return False
    if reject_dimension(bb, img_wh, rho_max):
        return False
    if reject_edge_stub(bb, img_wh, a, a_min, border, area_k):
        return False
    return True


# ---------------------------------------------------------------------------
# Class-agnostic NMS (post-sieve deduplication)
# ---------------------------------------------------------------------------
def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    """IoU of two (x1, y1, x2, y2) boxes (inclusive pixel coords)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 < ax1 or ay2 < ay1 or bx2 < bx1 or by2 < by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    a_area = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    b_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def class_agnostic_nms(boxes: List[Tuple[int, int, int, int]],
                       scores: List[float],
                       iou_thr: float = NMS_IOU) -> List[int]:
    """Greedy NMS over boxes of a single (class-agnostic) category.

    Returns the indices of kept boxes, highest score first. Each suppressed box
    contributes to ``SieveStats.n_reject_nms`` in the caller.
    """
    if not boxes:
        return []
    order = list(np.argsort(scores)[::-1])
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            if _iou(boxes[i], boxes[j]) < iou_thr:
                rest.append(j)
        order = rest
    return keep


# ---------------------------------------------------------------------------
# Full sieve pipeline (apply tests -> NMS) over a batch of masks
# ---------------------------------------------------------------------------
def sieve_masks(masks: List[np.ndarray | None],
                img_wh: Tuple[int, int],
                scores: List[float] | None = None,
                bboxes: List[Tuple[int, int, int, int] | None] | None = None,
                areas: List[int | None] | None = None,
                a_min: float = A_MIN,
                rho_max: float = RHO_MAX,
                border: float = EDGE_BORDER,
                area_k: float = EDGE_STUB_AREA_K,
                nms_iou: float = NMS_IOU,
                apply_nms: bool = True,
                ) -> Tuple[List[int], SieveStats]:
    """Run the full sieve over one image's masks.

    Each entry in ``masks`` is a 2-D array (or None if only a bbox is available).
    ``bboxes`` / ``areas`` are optional per-mask accompaniments. ``scores`` are the
    detector's confidence scores, used only for NMS ordering.

    Returns (kept_indices, stats). ``kept_indices`` are indices into the input
    lists, in the order they survive (NMS order if ``apply_nms`` else input order).
    """
    n = len(masks)
    if bboxes is None:
        bboxes = [None] * n
    if areas is None:
        areas = [None] * n
    if scores is None:
        scores = [0.0] * n
    stats = SieveStats(n_in=n)

    survivors: List[int] = []
    for idx, (m, bb, ar) in enumerate(zip(masks, bboxes, areas)):
        resolved_bb = _bbox_from_any(m, bb)
        if resolved_bb[0] < 0:
            # empty mask counts as an area rejection
            stats.n_reject_area += 1
            continue
        a = _area_from_any(m, bb, ar)

        # Apply the three tests, in spec order, with mutually-exclusive accounting
        # (a mask rejected by test 1 is not also counted under test 3).
        if reject_area(a, a_min):
            stats.n_reject_area += 1
            continue
        if reject_dimension(resolved_bb, img_wh, rho_max):
            stats.n_reject_dim += 1
            continue
        if reject_edge_stub(resolved_bb, img_wh, a, a_min, border, area_k):
            stats.n_reject_edge += 1
            continue
        survivors.append(idx)

    if apply_nms and len(survivors) > 1:
        surv_boxes = [_bbox_from_any(masks[i], bboxes[i]) for i in survivors]
        surv_scores = [scores[i] for i in survivors]
        keep_local = class_agnostic_nms(surv_boxes, surv_scores, nms_iou)
        n_suppressed = len(survivors) - len(keep_local)
        stats.n_reject_nms = n_suppressed
        survivors = [survivors[k] for k in keep_local]

    stats.n_out = len(survivors)
    return survivors, stats


__all__ = [
    "A_MIN", "RHO_MAX", "EDGE_BORDER", "EDGE_STUB_AREA_K", "NMS_IOU",
    "SieveStats", "reject_area", "reject_dimension", "reject_edge_stub",
    "mask_passes", "class_agnostic_nms", "sieve_masks",
]


# ---------------------------------------------------------------------------
# CLI — run the sanity sieve over a COCO-format predictions JSON.
# ---------------------------------------------------------------------------
def _cli_main() -> None:
    """Run the sanity sieve over a COCO predictions JSON in-place per image.

    Reads ``--predictions`` (list of COCO preds, each with ``segmentation`` RLE
    or polygon, or ``bbox`` only) and ``--gt`` (a COCO ``instances`` JSON whose
    ``images`` list gives per-image width/height). Applies ``sieve_masks`` per
    image (area / dimension / edge-stub tests + class-agnostic NMS) and writes a
    filtered predictions JSON to ``--out`` plus an optional per-image stats CSV.
    Exits 0 with a summary on stdout.

    Usage:
        python penguins/sanity_sieve.py \
            --predictions ~/bench_any/runs/segment/<run>/predictions.json \
            --gt          ~/bench_any/runs/segment/<run>/_bench_gt_coco.json \
            --out         /tmp/preds_sieved.json \
            [--a-min 324 --rho-max 0.9 --nms-iou 0.5 --no-nms --out-stats s.csv]
    """
    import argparse
    import csv
    import json
    from collections import defaultdict
    from pathlib import Path

    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pycocotools is required: pip install pycocotools") from exc

    ap = argparse.ArgumentParser(description=_cli_main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="COCO predictions JSON (RLE/polygon/bbox).")
    ap.add_argument("--gt", required=True, help="COCO instances JSON with per-image width/height.")
    ap.add_argument("--out", required=True, help="Output filtered predictions JSON path.")
    ap.add_argument("--out-stats", default="", help="Optional per-image stats CSV path.")
    ap.add_argument("--a-min", type=float, default=A_MIN, help=f"Min mask area (default {A_MIN}).")
    ap.add_argument("--rho-max", type=float, default=RHO_MAX,
                     help=f"Max bbox dimension-fraction (default {RHO_MAX}).")
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU,
                     help=f"NMS IoU threshold (default {NMS_IOU}).")
    ap.add_argument("--no-nms", action="store_true", help="Skip class-agnostic NMS after sieve.")
    args = ap.parse_args()

    preds = json.loads(Path(args.predictions).read_text())
    gt = json.loads(Path(args.gt).read_text())
    img_wh = {im["id"]: (int(im["width"]), int(im["height"])) for im in gt["images"]}

    by_image = defaultdict(list)
    for p in preds:
        by_image[p["image_id"]].append(p)

    kept_preds: list = []
    n_in_total = n_out_total = n_reject_nms_total = 0
    rows = []
    for image_id, plist in by_image.items():
        wh = img_wh.get(image_id)
        if wh is None:
            # Unknown image — keep as-is (cannot run geometric tests without size).
            kept_preds.extend(plist)
            continue
        masks: list = []
        bboxes: list = []
        scores: list = []
        for p in plist:
            seg = p.get("segmentation")
            if isinstance(seg, dict) and "counts" in seg:
                rle = {k: v for k, v in seg.items()}
                if isinstance(rle.get("counts"), str):
                    rle["counts"] = rle["counts"].encode("ascii", "ignore")
                try:
                    masks.append(coco_mask.decode(rle).astype(bool))
                except Exception:
                    masks.append(None)
            elif isinstance(seg, list) and seg:
                H, W = wh[1], wh[0]
                try:
                    rles = coco_mask.frPyObjects(seg, H, W)
                    m = coco_mask.decode(rles)
                    masks.append(m.astype(bool).max(axis=2) if m.ndim == 3 else m.astype(bool))
                except Exception:
                    masks.append(None)
            else:
                masks.append(None)
            bboxes.append(tuple(int(round(c)) for c in p["bbox"]) if p.get("bbox") else None)
            scores.append(float(p.get("score", 0.0)))
        kept_idx, stats = sieve_masks(masks, img_wh=wh, scores=scores, bboxes=bboxes,
                                      a_min=args.a_min, rho_max=args.rho_max,
                                      nms_iou=args.nms_iou, apply_nms=not args.no_nms)
        for i in kept_idx:
            kept_preds.append(plist[i])
        n_in_total += stats.n_in
        n_out_total += len(kept_idx)
        n_reject_nms_total += stats.n_reject_nms
        rows.append({"image_id": image_id, "n_in": stats.n_in, "n_out": len(kept_idx),
                     "n_reject_nms": stats.n_reject_nms})

    Path(args.out).write_text(json.dumps(kept_preds))
    if args.out_stats:
        with open(args.out_stats, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "n_in", "n_out", "n_reject_nms"])
            w.writeheader(); w.writerows(rows)
    print(f"[sieve] {n_in_total} in -> {n_out_total} kept "
          f"({n_reject_nms_total} removed by NMS). wrote {args.out}")


if __name__ == "__main__":
    _cli_main()
