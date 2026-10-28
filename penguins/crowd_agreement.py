#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""
Crowd-sourced agreement filter for SAM-3 masks on Penguin-Watch images.
======================================================================

Reuses the 1-8 citizen-scientist dot annotations per bird (originally collected
for count supervision, Arteta et al. 2016) as a *free* agreement signal on the
zero-shot SAM-3 masks. For each mask ``m_i`` with confidence ``s_i``, the number
``a(m_i)`` of *distinct* annotators who placed at least one dot inside ``m_i`` is
computed; the mask is kept iff:

    a(m_i) >= k                  OR
    s_i       >= tau_bypass      OR
    (no clicks on the image at all).

Defaults (spec, §ssec:agreement): ``k = 3``, ``tau_bypass = 0.5``.

The two bypass conditions ensure:
  (i)  SAM-3 is trusted on click-sparse frames where it is highly confident, and
  (ii) images that simply received no citizen-science attention are not silently
       emptied of their masks.

The agreement filter is *optional* -- disabled on corpora that lack dot
annotations (caller passes ``clicks_per_image = None`` or an empty list, which
triggers the "no clicks" bypass for every mask).

Pure numpy. No torch, no I/O.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (spec, §ssec:agreement)
# ---------------------------------------------------------------------------
K_DEFAULT: int = 3            # distinct-annotator agreement threshold
TAU_BYPASS_DEFAULT: float = 0.5  # confidence bypass threshold


# ---------------------------------------------------------------------------
# Click / mask geometry
# ---------------------------------------------------------------------------
# A "click" is a single (x, y) dot by one annotator. Each click carries the id of
# the annotator who placed it so that ``a(m_i)`` counts DISTINCT annotators, not
# distinct clicks. We accept clicks as either:
#   * a list of (x, y, annotator_id) tuples, or
#   * three parallel arrays (xs, ys, annotator_ids).
# A "mask" is a 2-D uint8/bool array OR a tight bbox (x1, y1, x2, y2). The
# point-in-mask test falls back to point-in-bbox when only a bbox is available
# (a conservative over-estimate: more clicks count as inside, so ``a`` can only
# increase -- masks are KEPT more readily, the safe direction for an agreement
# filter that is meant to veto low-agreement junk, not to invent agreement).


def _clicks_to_arrays(clicks: Sequence[Tuple[float, float, int]] | None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise a click list to (xs[N], ys[N], annotator_ids[N]) int arrays.

    Returns three empty arrays when ``clicks`` is None or empty.
    """
    if not clicks:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    xs = np.fromiter((c[0] for c in clicks), dtype=np.float64, count=len(clicks))
    ys = np.fromiter((c[1] for c in clicks), dtype=np.float64, count=len(clicks))
    ids = np.fromiter((c[2] for c in clicks), dtype=np.int64, count=len(clicks))
    return xs, ys, ids


def point_in_mask(x: float, y: float,
                  mask: np.ndarray | None = None,
                  bbox: Tuple[int, int, int, int] | None = None) -> bool:
    """True iff integer-rounded (x, y) lies inside the mask (or its bbox fallback)."""
    ix, iy = int(round(x)), int(round(y))
    if mask is not None and mask.size > 0:
        h, w = mask.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            return bool(mask[iy, ix])
        return False
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return bool(x1 <= ix <= x2 and y1 <= iy <= y2)
    return False


def annotator_agreement(mask: np.ndarray | None,
                        xs: np.ndarray, ys: np.ndarray, annotator_ids: np.ndarray,
                        bbox: Tuple[int, int, int, int] | None = None) -> int:
    """``a(m_i)`` -- count of DISTINCT annotators with >=1 dot inside the mask."""
    if xs.size == 0:
        return 0
    inside: set[int] = set()
    # Vectorise the geometric test where possible, then count distinct ids.
    if mask is not None and mask.size > 0:
        h, w = mask.shape[:2]
        ixs = np.clip(np.rint(xs).astype(np.int64), 0, w - 1)
        iys = np.clip(np.rint(ys).astype(np.int64), 0, h - 1)
        hits = mask[iys, ixs] > 0
        for keep, aid in zip(hits.tolist(), annotator_ids.tolist()):
            if keep:
                inside.add(int(aid))
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        ixs = np.rint(xs).astype(np.int64)
        iys = np.rint(ys).astype(np.int64)
        hits = (ixs >= x1) & (ixs <= x2) & (iys >= y1) & (iys <= y2)
        for keep, aid in zip(hits.tolist(), annotator_ids.tolist()):
            if keep:
                inside.add(int(aid))
    return len(inside)


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------
@dataclass
class AgreementStats:
    """Per-image accounting of the agreement filter."""
    n_in: int = 0
    n_keep_agreement: int = 0       # kept by a(m_i) >= k
    n_keep_bypass_conf: int = 0      # kept by s_i >= tau_bypass
    n_keep_bypass_noclicks: int = 0  # kept because the image had no clicks
    n_drop: int = 0

    @property
    def n_out(self) -> int:
        return self.n_keep_agreement + self.n_keep_bypass_conf + self.n_keep_bypass_noclicks


def keep_mask(a_mi: int, s_i: float, k: int = K_DEFAULT,
              tau_bypass: float = TAU_BYPASS_DEFAULT,
              image_has_clicks: bool = True) -> bool:
    """Single-mask decision per the spec equation."""
    if a_mi >= k:
        return True
    if s_i >= tau_bypass:
        return True
    if not image_has_clicks:
        return True
    return False


def filter_by_agreement(masks: List[np.ndarray | None],
                        scores: List[float],
                        clicks_per_image: Sequence[Tuple[float, float, int]] | None,
                        bboxes: List[Tuple[int, int, int, int] | None] | None = None,
                        k: int = K_DEFAULT,
                        tau_bypass: float = TAU_BYPASS_DEFAULT,
                        ) -> Tuple[List[int], AgreementStats]:
    """Apply the agreement filter to one image's masks.

    ``clicks_per_image`` is the flat list of (x, y, annotator_id) clicks for the
    whole image (any click inside a given mask counts toward that mask's
    ``a(m_i)``). Pass None or [] for a click-free image -- every mask is then kept
    via the "no clicks" bypass.

    Returns (kept_indices, stats). ``kept_indices`` preserve input order.
    """
    n = len(masks)
    if bboxes is None:
        bboxes = [None] * n
    xs, ys, ids = _clicks_to_arrays(clicks_per_image)
    image_has_clicks = xs.size > 0
    stats = AgreementStats(n_in=n)

    kept: List[int] = []
    for idx, (m, s_i, bb) in enumerate(zip(masks, scores, bboxes)):
        a_mi = annotator_agreement(m, xs, ys, ids, bb) if image_has_clicks else 0
        if a_mi >= k:
            stats.n_keep_agreement += 1
            kept.append(idx)
        elif s_i >= tau_bypass:
            stats.n_keep_bypass_conf += 1
            kept.append(idx)
        elif not image_has_clicks:
            stats.n_keep_bypass_noclicks += 1
            kept.append(idx)
        else:
            stats.n_drop += 1
    return kept, stats


__all__ = [
    "K_DEFAULT", "TAU_BYPASS_DEFAULT", "AgreementStats",
    "annotator_agreement", "point_in_mask", "keep_mask", "filter_by_agreement",
]


# ---------------------------------------------------------------------------
# CLI — apply the crowd-sourced agreement filter over a COCO predictions JSON.
# ---------------------------------------------------------------------------
def _cli_main() -> None:
    """Apply the crowd-sourced agreement filter to a COCO predictions JSON.

    Reads ``--predictions`` (COCO preds with ``segmentation`` RLE/polygon or
    ``bbox``, plus ``score``) and ``--clicks`` (a JSON mapping ``image_id`` to a
    list of ``{"x", "y", "annotator_id"}`` dot annotations — the Penguin-Watch
    citizen-scientist clicks). Runs ``filter_by_agreement`` per image: a mask is
    kept iff >= ``--k`` distinct annotators placed a dot inside it, OR its score
    >= ``--tau-bypass`` (high-confidence bypass), OR the image had no clicks at
    all (no-clicks bypass). Writes a filtered predictions JSON to ``--out``.

    Usage:
        python penguins/crowd_agreement.py \
            --predictions ~/bench_any/runs/segment/<run>/predictions.json \
            --clicks      /path/to/penguin_watch_clicks.json \
            --out         /tmp/preds_agreed.json \
            [--k 2 --tau-bypass 0.5]
    """
    import argparse
    import json
    from collections import defaultdict
    from pathlib import Path

    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pycocotools is required: pip install pycocotools") from exc

    ap = argparse.ArgumentParser(description=_cli_main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="COCO predictions JSON.")
    ap.add_argument("--clicks", required=True,
                    help="JSON: image_id -> [{'x','y','annotator_id'}, ...] (str keys OK).")
    ap.add_argument("--out", required=True, help="Output filtered predictions JSON path.")
    ap.add_argument("--k", type=int, default=K_DEFAULT, help=f"Min distinct annotators (default {K_DEFAULT}).")
    ap.add_argument("--tau-bypass", type=float, default=TAU_BYPASS_DEFAULT,
                    help=f"Score bypass threshold (default {TAU_BYPASS_DEFAULT}).")
    args = ap.parse_args()

    preds = json.loads(Path(args.predictions).read_text())
    clicks_raw = json.loads(Path(args.clicks).read_text())
    # Normalise keys to int (JSON object keys are strings).
    clicks = {int(k): v for k, v in clicks_raw.items()}

    by_image = defaultdict(list)
    for p in preds:
        by_image[p["image_id"]].append(p)

    kept_preds: list = []
    n_in = n_out = 0
    for image_id, plist in by_image.items():
        img_clicks = clicks.get(int(image_id), [])
        clicks_tuples = [(float(c["x"]), float(c["y"]), int(c["annotator_id"]))
                         for c in img_clicks]
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
                # Need image H/W; skip mask, fall back to bbox for point-in-box.
                masks.append(None)
            else:
                masks.append(None)
            bboxes.append(tuple(int(round(c)) for c in p["bbox"]) if p.get("bbox") else None)
            scores.append(float(p.get("score", 0.0)))
        kept_idx, stats = filter_by_agreement(
            masks, scores=scores, clicks_per_image=clicks_tuples or None,
            bboxes=bboxes, k=args.k, tau_bypass=args.tau_bypass)
        for i in kept_idx:
            kept_preds.append(plist[i])
        n_in += stats.n_in
        n_out += len(kept_idx)

    Path(args.out).write_text(json.dumps(kept_preds))
    print(f"[agreement] {n_in} in -> {n_out} kept. wrote {args.out}")


if __name__ == "__main__":
    _cli_main()
