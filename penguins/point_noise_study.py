#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Point-noise sanity study: what crowd-annotator sloppiness costs SAM 2.

WHY
---
The crowdsourced-point pipeline in
``Fish/YOLO/point_crowdsource_annotate/point_to_sam21_coco.py`` turns raw
Penguin Watch click coordinates into masks: clicks within ``MERGE_RADIUS``
pixels are grouped as one penguin, a group is trusted only once
``MIN_ANNOTATORS`` distinct clickers agree on it, and the group's medoid
point is the SAM 2 prompt. That pipeline has never been checked against a
corpus with real ground truth, so it is unknown how much mask quality a
sloppy crowd actually costs, or whether the agreement threshold buys
anything. This script answers that on the boulders-penguin gold corpus,
which has hand-drawn polygons instead of crowd clicks: it manufactures
synthetic "annotators" by jittering each ground-truth instance's centre by
a controlled amount and adding a controlled rate of spurious off-target
clicks, then runs the same merge-and-threshold-and-prompt pipeline and
scores the resulting masks against the real polygon.

Two independent knobs are swept: how far each synthetic annotator's click
drifts from the true centre (``--noise_levels``, a fraction of the
instance's own size) and how often an annotator clicks nothing at all
(``--extra_click_rates``, spurious clicks per real instance). A third axis,
the agreement threshold (``--min_annotators``), does not need its own SAM 2
passes -- the merged point groups and their SAM 2 masks are computed once
per (noise, extra-click) pair, and the threshold only changes which of
those already-computed groups are kept, so the sweep is add-on-top-only,
not multiplicative in inference cost.

REUSE, NOT REWRITE
-------------------
- ``count_decompose.load_gold`` parses the boulders-penguin YOLO polygon
  labels into ground-truth RLEs, boxes and per-image (h, w) -- unchanged.
- ``sanity_sieve.mask_passes`` is the same geometric sieve used on the
  real SAM-3 pipeline in this repo, applied here to the SAM 2 masks coming
  out of the noisy-point simulation.
- The SAM 2.1 predictor init (Hydra config dir -> build_sam2 ->
  SAM2ImagePredictor) and the constants MERGE_RADIUS=60,
  MIN_ANNOTATORS=4, MAX_MASK_FRACTION=0.12, MIN_MASK_AREA=384, and the
  ``pick_best_mask`` / ``merge_points`` / ``medoid`` logic, are carried
  over verbatim from ``point_to_sam21_coco.py`` (that script is not
  importable as a module -- it loads the model and walks a folder list at
  import time -- so the small pure functions are copied here rather than
  imported, with the same names and the same behaviour).

INPUT
-----
The boulders-penguin gold corpus (``--corpus``, default
``/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train``),
with ``images/`` and ``labels/`` subdirectories of YOLO-polygon labels.

OUTPUT
------
One CSV row per (noise_level, extra_click_rate, min_annotators) combination
at ``--out`` (default
``~/bench_any/runs/obj2_counting/point_noise/results.csv``), with recall
and precision of the recovered masks against ground truth, mean mask IoU
on the recovered instances, and the sanity-sieve rejection rate. Reports
how much of the crowd's sloppiness the merge-and-agreement step actually
absorbs, and where it stops working.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_decompose import load_gold  # noqa: E402
from sanity_sieve import mask_passes  # noqa: E402

# ---------------------------------------------------------------------------
# Constants carried over verbatim from point_to_sam21_coco.py
# ---------------------------------------------------------------------------
MERGE_RADIUS = 60          # clicks within this distance = same penguin
MAX_MASK_FRACTION = 0.12   # reject masks covering > 12% of image
MIN_MASK_AREA = 384        # reject tiny fragment masks

SAM2_ROOT = "/home/user/sam2"
SAM2_CONFIGS_ROOT = f"{SAM2_ROOT}/sam2/configs"
SAM2_CKPT = os.environ.get(
    "SAM2_CKPT", f"{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = os.environ.get(
    "SAM2_CONFIG", f"{SAM2_CONFIGS_ROOT}/sam2.1/sam2.1_hiera_l.yaml")

DEFAULT_CORPUS = (
    "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/"
    "boulders-penguin/train")
DEFAULT_OUT = os.path.expanduser(
    "~/bench_any/runs/obj2_counting/point_noise/results.csv")

DEFAULT_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8]
DEFAULT_EXTRA_CLICK_RATES = [0.0, 0.25, 0.5]
DEFAULT_MIN_ANNOTATORS_SWEEP = [1, 2, 4, 6]
N_ANNOTATORS = 6


# ---------------------------------------------------------------------------
# point_to_sam21_coco.py logic, copied verbatim (that script is a
# top-level walk-and-load driver, not an importable module)
# ---------------------------------------------------------------------------
def pick_best_mask(masks, scores, img_h, img_w):
    """From SAM2's multi-mask output, pick the highest-scoring mask that
    passes sanity checks. Returns (mask, score) or (None, None)."""
    img_area = img_h * img_w
    best_mask = None
    best_score = -1
    for mask, score in zip(masks, scores):
        area = int(mask.sum())
        if area / img_area > MAX_MASK_FRACTION:
            continue
        if area < MIN_MASK_AREA:
            continue
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask, best_score


def merge_points_indexed(points, radius):
    """Same greedy distance-based grouping as point_to_sam21_coco.merge_points,
    adapted to return groups of *indices* into ``points`` rather than copies
    of the points themselves, so caller-side metadata (which annotator, which
    ground-truth instance a click was jittered from) survives the merge."""
    groups = []
    centroids = []
    for idx, pt in enumerate(points):
        p = np.array(pt, dtype=np.float64)
        merged = False
        for i, c in enumerate(centroids):
            if np.linalg.norm(p - c) <= radius:
                groups[i].append(idx)
                centroids[i] = np.mean([points[j] for j in groups[i]], axis=0)
                merged = True
                break
        if not merged:
            groups.append([idx])
            centroids.append(p.copy())
    return groups


def medoid_index(idxs, points):
    """Index (into ``points``, from ``idxs``) closest to the group centroid."""
    arr = np.array([points[i] for i in idxs], dtype=np.float64)
    c = arr.mean(axis=0)
    dists = np.linalg.norm(arr - c, axis=1)
    return idxs[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# SAM 2 predictor init -- mirrors sam2_video_propagate._build_sam2_video_predictor,
# swapped for the single-image predictor.
# ---------------------------------------------------------------------------
def build_sam2_image_predictor(sam2_cfg: str, sam2_ckpt: str, device: str):
    if SAM2_ROOT not in sys.path:
        sys.path.insert(0, SAM2_ROOT)
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=SAM2_CONFIGS_ROOT, version_base="1.2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg_path = Path(sam2_cfg)
    if cfg_path.is_absolute():
        try:
            cfg_rel = str(cfg_path.relative_to(SAM2_CONFIGS_ROOT))
        except ValueError:
            cfg_rel = sam2_cfg
    else:
        cfg_rel = sam2_cfg
    sam2_model = build_sam2(cfg_rel, sam2_ckpt, device=device)
    return SAM2ImagePredictor(sam2_model)


# ---------------------------------------------------------------------------
# Gold-corpus loading -- reuses count_decompose.load_gold for the geometry,
# and mirrors its exact sorted-glob-with-skip loop to recover the matching
# image path, which load_gold itself does not return (it was written only
# to hand back RLEs for a downstream prediction match, not to drive
# inference).
# ---------------------------------------------------------------------------
def load_gold_with_paths(labels_dir, images_dir):
    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({}, f)
        empty_pred_path = f.name
    try:
        geom = load_gold(labels_dir, images_dir, empty_pred_path, min_score=1e9)
    finally:
        os.unlink(empty_pred_path)

    img_paths = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem = lbl.stem
        img = next((p for p in images_dir.glob(stem + ".*")), None)
        if img is None:
            continue
        img_paths.append(img)

    if len(img_paths) != len(geom):
        sys.exit(f"gold loader mismatch: {len(img_paths)} image paths vs "
                  f"{len(geom)} geometry rows -- labels/images out of sync")
    return [(*g, p) for g, p in zip(geom, img_paths)]


# ---------------------------------------------------------------------------
# Synthetic crowd simulation
# ---------------------------------------------------------------------------
def simulate_clicks(gt_boxes, w, h, noise_level, extra_click_rate, n_annotators, rng):
    """Manufacture synthetic annotator clicks around each GT instance.

    Returns (points, annotator_idx, origin_idx) as parallel lists.
    ``origin_idx`` is the GT instance index a click was jittered from, or
    -1 for a spurious click with no real target.
    """
    points, annotator_idx, origin_idx = [], [], []
    n_gt = len(gt_boxes)
    for a in range(n_annotators):
        for gi, (x, y, bw, bh) in enumerate(gt_boxes):
            cx, cy = x + bw / 2.0, y + bh / 2.0
            scale_ref = max((bw * bh) ** 0.5, 1.0)
            jitter = rng.normal(0.0, noise_level * scale_ref, size=2)
            px = float(np.clip(cx + jitter[0], 0, w - 1))
            py = float(np.clip(cy + jitter[1], 0, h - 1))
            points.append([px, py])
            annotator_idx.append(a)
            origin_idx.append(gi)

        n_extra = int(rng.poisson(extra_click_rate * n_gt)) if n_gt > 0 else 0
        for _ in range(n_extra):
            px = float(rng.uniform(0, w - 1))
            py = float(rng.uniform(0, h - 1))
            points.append([px, py])
            annotator_idx.append(a)
            origin_idx.append(-1)

    return points, annotator_idx, origin_idx


def group_target(idxs, origin_idx):
    """Majority-vote GT instance index for a merged group, or -1 if the
    group is made entirely of spurious clicks."""
    votes = [origin_idx[i] for i in idxs if origin_idx[i] >= 0]
    if not votes:
        return -1
    vals, counts = np.unique(votes, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def n_distinct_annotators(idxs, annotator_idx):
    return len({annotator_idx[i] for i in idxs})


# ---------------------------------------------------------------------------
# Per-image, per-(noise, extra) processing: compute every merged group's
# SAM 2 mask once, so the min_annotators sweep afterwards is a filter, not
# a re-run.
# ---------------------------------------------------------------------------
def process_image(predictor, gt_rles, gt_boxes, hw,
                   noise_level, extra_click_rate, n_annotators, rng):
    """Simulate clicks and run SAM 2 for one image at one (noise, extra)
    setting. Assumes ``predictor.set_image`` was already called by the
    caller for the *current* image -- the embedding depends only on
    pixels, not on the simulated points, so it is computed once per image
    and reused across the whole noise/extra-click sweep rather than
    recomputed on every combination (a 5x3=15x redundant-embedding trap
    for an 18-megapixel corpus otherwise)."""
    h, w = hw
    points, annotator_idx, origin_idx = simulate_clicks(
        gt_boxes, w, h, noise_level, extra_click_rate, n_annotators, rng)
    if not points:
        return []

    groups = merge_points_indexed(points, MERGE_RADIUS)

    records = []
    for idxs in groups:
        rep_idx = medoid_index(idxs, points)
        rep = points[rep_idx]
        input_point = np.array([[int(round(rep[0])), int(round(rep[1]))]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask, score = pick_best_mask(masks, scores, h, w)

        target = group_target(idxs, origin_idx)
        n_ann = n_distinct_annotators(idxs, annotator_idx)

        if mask is None:
            records.append(dict(n_annotators=n_ann, target=target,
                                 sieve_pass=False, has_mask=False, iou=0.0))
            continue

        sieve_ok = bool(mask_passes(mask, (w, h)))

        iou = 0.0
        if target >= 0:
            pred_rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            pred_rle["counts"] = pred_rle["counts"].decode("utf-8")
            iou = float(mask_utils.iou([pred_rle], [gt_rles[target]], [0])[0][0])

        records.append(dict(n_annotators=n_ann, target=target,
                             sieve_pass=sieve_ok, has_mask=True, iou=iou))

    return records


# ---------------------------------------------------------------------------
# Aggregation across the min_annotators sweep, given one (noise, extra)
# combo's already-computed group records for the whole corpus.
# ---------------------------------------------------------------------------
def aggregate(records, n_gt_total, min_annotators):
    kept = [r for r in records if r["n_annotators"] >= min_annotators]
    n_groups_kept = len(kept)
    passing = [r for r in kept if r["sieve_pass"]]
    n_sieve_reject = n_groups_kept - len(passing)

    real = [r for r in passing if r["target"] >= 0]
    spurious = [r for r in passing if r["target"] < 0]

    recovered_targets = {r["target"] for r in real if r["iou"] > 0}
    recall = len(recovered_targets) / n_gt_total if n_gt_total else 0.0
    recall_iou50 = (len({r["target"] for r in real if r["iou"] >= 0.5})
                     / n_gt_total if n_gt_total else 0.0)
    mean_iou = float(np.mean([r["iou"] for r in real])) if real else 0.0
    n_accepted = len(real) + len(spurious)
    precision = len(real) / n_accepted if n_accepted else 0.0
    sieve_reject_rate = n_sieve_reject / n_groups_kept if n_groups_kept else 0.0

    return dict(
        n_groups_total=len(records),
        n_groups_kept=n_groups_kept,
        n_accepted_real=len(real),
        n_accepted_spurious=len(spurious),
        recall=round(recall, 4),
        recall_iou50=round(recall_iou50, 4),
        mean_iou=round(mean_iou, 4),
        precision=round(precision, 4),
        sieve_reject_rate=round(sieve_reject_rate, 4),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                     help="dataset root with images/ and labels/ (default: "
                          "boulders-penguin train split)")
    ap.add_argument("--labels_dir", default=None,
                     help="override the labels dir (default: <corpus>/labels)")
    ap.add_argument("--images_dir", default=None,
                     help="override the images dir (default: <corpus>/images)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                     help="only process the first N images (0 = all)")
    ap.add_argument("--n_annotators", type=int, default=N_ANNOTATORS,
                     help="synthetic annotators per image (default 6)")
    ap.add_argument("--noise_levels", type=float, nargs="+",
                     default=DEFAULT_NOISE_LEVELS,
                     help="click jitter std, as a fraction of instance scale")
    ap.add_argument("--extra_click_rates", type=float, nargs="+",
                     default=DEFAULT_EXTRA_CLICK_RATES,
                     help="expected spurious clicks per real instance")
    ap.add_argument("--min_annotators", type=int, nargs="+",
                     default=DEFAULT_MIN_ANNOTATORS_SWEEP,
                     help="agreement thresholds to sweep (filter only, free)")
    ap.add_argument("--sam2_config", default=SAM2_CONFIG)
    ap.add_argument("--sam2_ckpt", default=SAM2_CKPT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir) if args.labels_dir else Path(args.corpus) / "labels"
    images_dir = Path(args.images_dir) if args.images_dir else Path(args.corpus) / "images"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        sys.exit(f"missing labels/images dir: {labels_dir} / {images_dir}")

    gold = load_gold_with_paths(labels_dir, images_dir)
    if args.limit > 0:
        gold = gold[: args.limit]
    if not gold:
        sys.exit("no gold images found")

    print(f"[point_noise_study] {len(gold)} images, "
          f"{sum(len(g[1]) for g in gold)} GT instances, "
          f"noise_levels={args.noise_levels}, "
          f"extra_click_rates={args.extra_click_rates}, "
          f"min_annotators={args.min_annotators}")

    predictor = build_sam2_image_predictor(args.sam2_config, args.sam2_ckpt, args.device)

    from PIL import Image
    rng = np.random.default_rng(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["noise_level", "extra_click_rate", "min_annotators", "n_images",
                  "n_gt_instances"] + list(aggregate([], 1, 1).keys())

    n_gt_total = sum(len(g[1]) for g in gold)

    # image decode + SAM 2 embedding is the expensive step on an
    # 18-megapixel corpus, so the outer loop is over images (decode once,
    # embed once) and the noise/extra-click sweep runs inside it, reusing
    # the cached embedding for every combination on that image.
    records_by_combo = {(nl, ecr): []
                         for nl in args.noise_levels
                         for ecr in args.extra_click_rates}

    for i, (pred_rles, gt_rles, gt_boxes, hw, img_path) in enumerate(gold, 1):
        image_arr = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image_arr)
        for noise_level in args.noise_levels:
            for extra_click_rate in args.extra_click_rates:
                recs = process_image(
                    predictor, gt_rles, gt_boxes, hw,
                    noise_level, extra_click_rate, args.n_annotators, rng)
                records_by_combo[(noise_level, extra_click_rate)].extend(recs)
        print(f"[point_noise_study] embedded {i}/{len(gold)} images", flush=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for noise_level in args.noise_levels:
            for extra_click_rate in args.extra_click_rates:
                all_records = records_by_combo[(noise_level, extra_click_rate)]
                for min_ann in args.min_annotators:
                    agg = aggregate(all_records, n_gt_total, min_ann)
                    row = dict(noise_level=noise_level,
                               extra_click_rate=extra_click_rate,
                               min_annotators=min_ann,
                               n_images=len(gold),
                               n_gt_instances=n_gt_total,
                               **agg)
                    writer.writerow(row)
                    print(f"  noise={noise_level:.2f} extra={extra_click_rate:.2f} "
                          f"min_ann={min_ann}  recall={agg['recall']:.3f} "
                          f"mean_iou={agg['mean_iou']:.3f} "
                          f"precision={agg['precision']:.3f} "
                          f"sieve_reject={agg['sieve_reject_rate']:.3f}")

    print(f"\n[point_noise_study] wrote {out_path}")


if __name__ == "__main__":
    main()
