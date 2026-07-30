#!/usr/bin/env python3
"""
Standalone post-hoc difficulty-stratified eval over FROZEN run dirs.
=====================================================================

P1 (ECCV 1a benchmark) missing piece. Reads existing gt/pred COCO JSON from
``~/bench_any/runs/{detect,segment}/<run_name>/`` and fills manuscript
Table~``tab:cross`` (cross-paradigm Easy/Medium/Hard mask-mAP$_{50}$ table) with
real-data numbers, **without invoking the orchestrator's subprocess machinery**.

The difficulty-stratification capability is built, bug-fixed and
smoke-verified, but it had never been run on the real 125-epoch
african-penguin corpus. This script makes that run eval-only:

  * difficulty is computed from GT geometry alone
    (``bench_eval.compute_difficulty_from_coco_gt`` -- Eq.1-5 of the paper).
  * per-tier COCOeval is ``bench_eval.stratified_coco_eval`` (pycocotools under
    the hood; ``maxDets=[1,10,100]`` honoured, as in ``bench_eval``).
  * gt/pred discovery reuses the orchestrator's exact ``_rank`` preference so
    the file selected for each framework is the same one the in-orchestrator
    ``--difficulty_stratify`` block would select (BUG 4's two-layer fix).
  * bbox-vs-seg gating reuses ``bench_any._difficulty_iou_gating`` so
    dinov3_semseg is skipped (no instance boxes), pure detectors score bbox
    only, native mask-head seg models score both.

The script is **read-only over run dirs** and writes a single new CSV
(``difficulty_stratified_125e.csv`` by default). It never launches a subprocess,
never trains, and never writes inside a run dir.

Usage
-----
    ~/cvML/bin/python penguins/difficulty_stratify_posthoc.py \\
        --runs-root ~/bench_any/runs \\
        --model-tags yolo11x-seg yolo26x-seg rtdetr-3 rtdetr-4 \\
                     rf-detr-seg-large rf-detr-seg-xlarge \\
                     dinov3-m2f-smallplus dinov3-m2f-vith16plus \\
                     eomt-large \\
        --out difficulty_stratified_125e.csv

    # default --model-tags is the six-paradigm tab:cross set (see DEFAULT_TAGS).
    ~/cvML/bin/python penguins/difficulty_stratify_posthoc.py --out /tmp/x.csv

Run on the 1-epoch AnimalP smokes (functional check, not real numbers):
    ~/cvML/bin/python penguins/difficulty_stratify_posthoc.py \\
        --runs-root ~/bench_any/runs \\
        --model-tags yolo11s-seg rtdetr-3 rf-detr-seg-medium dinov3-m2f-smallplus \\
        --out /tmp/diff_smoke.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect C-level + Python-level stdout to /dev/null.

    pycocotools' COCOeval.summarize() prints ~12 lines per tier and ignores
    Python's sys.stdout (it uses C printf via the underlying lib). So we dup2
    the fd too. Stderr is left untouched so real errors still surface.
    """
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        saved_py = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stdout = saved_py
    finally:
        os.dup2(saved, 1)
        os.close(devnull)

# --- benchmark libs (lightweight; no torch at import time) ----------------
# ``bench_eval`` provides the two compute primitives; ``bench_any`` provides
# the bbox/seg gating matrix that matches the in-orchestrator path exactly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bench_eval as _be  # noqa: E402
import bench_any as _ba    # noqa: E402  (import is safe: argparse is under __main__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RUNS_ROOT = Path.home() / "bench_any" / "runs"

# Six-paradigm cross-paradigm set from P1's ``tab:cross`` (manuscript uses the
# x/large variants; the small smokes stand in for the same frameworks locally).
DEFAULT_TAGS: List[str] = [
    "yolo11x-seg", "yolo26x-seg",
    "rtdetr-3", "rtdetr-4",
    "rf-detr-seg-large", "rf-detr-seg-xlarge",
    "dinov3-m2f-smallplus", "dinov3-m2f-vith16plus",
    "eomt-large",
]

# NMS-free flag per paradigm family, keyed by a prefix of the model_tag.
# Verbatim from tab:cross: YOLOv11s/l-seg = No; RT-DETR, RF-DETR-Seg, EoMT,
# DINOv3-Mask2Former = Yes. Match by longest-prefix first so "yolo11x-seg"
# and "yolo26x-seg" both hit the "yolo" rule.
NMS_FREE_PREFIXES: List[Tuple[str, bool]] = [
    ("yolo",    False),   # NMS-based one-stage
    ("rtdetr",  True),    # NMS-free query-based
    ("rf-detr", True),    # NMS-free
    ("eomt",    True),    # encoder-only mask transformer, NMS-free
    ("dinov3-m2f", True), # decoder-based, NMS-free
    ("dinov3-semseg", None),  # skipped (no instance boxes)
]


@dataclass
class TierRow:
    """One row of the output CSV: a single (model_tag, iou_type) pair."""
    model_tag: str
    framework: str
    task: str
    iou_type: str            # 'bbox' | 'segm'
    nms_free: Optional[bool] = None  # True/False/None (None = skipped/unknown)
    gt_json: str = ""
    pred_json: str = ""
    n_instances: int = 0
    # Per-tier mAP -- emit BOTH mAP50_95 (what the orchestrator stores) AND
    # mAP50 (what tab:cross actually prints). tab:cross prints mAP50.
    agg_mAP50_95: float = 0.0
    agg_mAP50: float = 0.0
    agg_mAP75: float = 0.0
    agg_AR_100: float = 0.0
    easy_mAP50_95: float = 0.0
    easy_mAP50: float = 0.0
    easy_mAP75: float = 0.0
    easy_AR_100: float = 0.0
    easy_n: int = 0
    medium_mAP50_95: float = 0.0
    medium_mAP50: float = 0.0
    medium_mAP75: float = 0.0
    medium_AR_100: float = 0.0
    medium_n: int = 0
    hard_mAP50_95: float = 0.0
    hard_mAP50: float = 0.0
    hard_mAP75: float = 0.0
    hard_AR_100: float = 0.0
    hard_n: int = 0
    # delta = Easy - Hard per metric (ap_difficulty L762-765). tab:cross prints
    # the mAP50 delta as $\Delta$.
    delta_mAP50_95: float = 0.0
    delta_mAP50: float = 0.0
    delta_mAP75: float = 0.0
    delta_AR_100: float = 0.0
    status: str = "ok"       # 'ok' | 'skipped' | 'no_gt' | 'no_pred' | 'error'


# ---------------------------------------------------------------------------
# gt/pred discovery -- mirrors bench_any's --difficulty_stratify block exactly
# ---------------------------------------------------------------------------
def _rank_candidate(p: Path, eval_split: str) -> Tuple:
    """Rank a gt/pred candidate path (lower tuple wins).

    Re-implementation of the in-orchestrator ``_rank`` (bench_any.py
    ``--difficulty_stratify`` block) so this script selects the SAME file the
    orchestrator would, preserving BUG 4's two-layer fix:
      * YOLO ID-fixed pred file (_bench_predictions_catfixed > _fixed) wins,
        because raw predictions.json has string image_ids that don't match the
        gt's int safe_image_id hashes (0 overlap -> 0 mAP, BUG 4 Layer 1).
      * DINOv3-M2F / EoMT: ``_bench_predictions_resolved.json`` wins over
        ``_bench_predictions_fixed.json``. Both exist on disk for these
        frameworks, but the wrapper writes ``_fixed`` (via
        ``fix_yolo_pred_image_ids``) BEFORE ``coco_eval_full`` runs
        ``remap_pred_categories``; that remap overwrites ``_resolved`` in place
        with the category_id corrected to match the 1-indexed GT, leaving
        ``_fixed`` stale (0-indexed category_id -> 0 matches -> agg mAP = 0).
        ``_resolved`` is the post-remap file the orchestrator actually scored.
      * eval_split in path/name wins (avoid train-split gt/pred).
      * shallower path (closer to run root) wins.
      * name is a deterministic tie-break.
    """
    name = p.name.lower()
    path = str(p).lower()
    sp = (eval_split or "").lower()
    if name == "_bench_predictions_resolved.json":
        fixed = 0
    elif "catfixed" in name:
        fixed = 0
    elif name.startswith("_bench_predictions_fixed"):
        fixed = 1
    else:
        fixed = 2
    in_split = bool(sp) and (sp in path or name.endswith(f"_{eval_split}.json")
                            or name == f"predictions_{eval_split}.json"
                            or name == f"instances_{eval_split}.json")
    return (fixed, 0 if in_split else 1, len(p.parts), name)


def discover_gt_pred(run_dir: Path, eval_split: str = "test"
                     ) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the gt/pred COCO JSON pair under ``run_dir``.

    Per-framework layouts (from bench_any.py difficulty block):
      - RF-DETR:            <run>/_bench_gt_coco.json + predictions_test.json       (root)
      - RT-DETR-external:  <run>/_rtdetrv2_repo_coco/instances_{split}.json
                          + <run>/predictions_{split}.json                           (gt in subdir)
      - YOLO:              <run>/<split>/_bench_gt_coco.json
                          + <run>/<split>/_bench_predictions_catfixed.json          (split subdir)
      - DINOv3-M2F / EoMT: <run>/_bench_gt_coco.json
                          + <run>/_bench_predictions_fixed.json                     (root)
    """
    gt_cands = sorted(
        list(run_dir.rglob("_bench_gt_coco.json"))
        + list(run_dir.rglob("instances_*.json"))
        + list(run_dir.rglob("*gt*coco*.json")),
        key=lambda p: _rank_candidate(p, eval_split),
    )
    pred_cands = sorted(
        list(run_dir.rglob("predictions*.json"))
        + list(run_dir.rglob("_bench_predictions_*.json"))
        + list(run_dir.rglob("*pred*coco*.json")),
        key=lambda p: _rank_candidate(p, eval_split),
    )
    # Dedup preserving rank order.
    seen: set = set()
    gts: List[Path] = []
    for p in gt_cands:
        if p not in seen:
            seen.add(p); gts.append(p)
    seen = set()
    preds: List[Path] = []
    for p in pred_cands:
        if p not in seen:
            seen.add(p); preds.append(p)
    return (gts[0] if gts else None, preds[0] if preds else None)


# ---------------------------------------------------------------------------
# Manifest + tag helpers
# ---------------------------------------------------------------------------
def read_manifest(run_dir: Path) -> Optional[dict]:
    """Load ``bench_manifest.json`` from a run dir (None if missing/corrupt)."""
    mf = run_dir / "bench_manifest.json"
    if not mf.is_file():
        return None
    try:
        with open(mf) as fh:
            return json.load(fh)
    except Exception:
        return None


def nms_free_for_tag(model_tag: str) -> Optional[bool]:
    """NMS-free flag for a paradigm, by longest-prefix match on the tag.

    Returns None for dinov3_semseg (skipped -- no instance boxes). The
    ``NMS_FREE_PREFIXES`` list is checked longest-prefix-first so
    ``dinov3-m2f`` does not shadow under a shorter rule.
    """
    tag = model_tag.lower()
    # Longest-prefix-first: sort by prefix length descending.
    for prefix, flag in sorted(NMS_FREE_PREFIXES, key=lambda kv: -len(kv[0])):
        if tag.startswith(prefix):
            return flag
    return None


def discover_run_dirs(runs_root: Path, model_tags: List[str]) -> Dict[str, Path]:
    """Map each requested model_tag to a run dir under ``runs_root``.

    Scans ``{runs_root}/detect`` and ``{runs_root}/segment``. A run dir matches
    a tag iff the run-name stem (``<dataset>_<model_stem>_<cfg_tag>``) contains
    the tag as a segment. If multiple dirs match a tag (e.g. b0.98 and b1
    smokes, or e1 and e125), prefer -- in order -- the highest epoch count,
    then the largest batch, then lexicographically. This picks the real
    125-epoch run over a 1-epoch smoke when both exist.
    """
    out: Dict[str, Path] = {}
    if not runs_root.is_dir():
        return out
    candidates: Dict[str, List[Path]] = {t: [] for t in model_tags}
    for task_dir in (runs_root / "detect", runs_root / "segment"):
        if not task_dir.is_dir():
            continue
        for run_dir in task_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = run_dir.name.lower()
            for tag in model_tags:
                if tag.lower() in name:
                    candidates[tag].append(run_dir)

    def _epoch_key(p: Path) -> int:
        # Parse ``_e<N>_`` from the run name; bigger N wins. 0 if absent.
        import re
        m = re.search(r"_e(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    def _batch_key(p: Path) -> float:
        import re
        m = re.search(r"_b([\d.]+)_", p.name)
        return float(m.group(1)) if m else 0.0

    for tag, dirs in candidates.items():
        if not dirs:
            continue
        # Highest epoch, then largest batch, then lexicographic name (stable).
        dirs.sort(key=lambda p: (-_epoch_key(p), -_batch_key(p), p.name))
        out[tag] = dirs[0]
    return out


# ---------------------------------------------------------------------------
# Per-run difficulty-stratified eval
# ---------------------------------------------------------------------------
def _tiers_to_row(tag: str, framework: str, task: str, iou_type: str,
                  gt_json: str, pred_json: str, tiers: Dict) -> TierRow:
    """Flatten a stratified_coco_eval result dict into a TierRow."""
    nms = nms_free_for_tag(tag)
    agg = tiers.get("aggregate", {})
    ez = tiers.get("Easy", {})
    md = tiers.get("Medium", {})
    hd = tiers.get("Hard", {})
    dl = tiers.get("delta", {})
    return TierRow(
        model_tag=tag, framework=framework, task=task, iou_type=iou_type,
        nms_free=nms, gt_json=gt_json, pred_json=pred_json,
        n_instances=int(agg.get("instances", 0)),
        agg_mAP50_95=float(agg.get("mAP50_95", 0.0)),
        agg_mAP50=float(agg.get("mAP50", 0.0)),
        agg_mAP75=float(agg.get("mAP75", 0.0)),
        agg_AR_100=float(agg.get("AR_100", 0.0)),
        easy_mAP50_95=float(ez.get("mAP50_95", 0.0)),
        easy_mAP50=float(ez.get("mAP50", 0.0)),
        easy_mAP75=float(ez.get("mAP75", 0.0)),
        easy_AR_100=float(ez.get("AR_100", 0.0)),
        easy_n=int(ez.get("instances", 0)),
        medium_mAP50_95=float(md.get("mAP50_95", 0.0)),
        medium_mAP50=float(md.get("mAP50", 0.0)),
        medium_mAP75=float(md.get("mAP75", 0.0)),
        medium_AR_100=float(md.get("AR_100", 0.0)),
        medium_n=int(md.get("instances", 0)),
        hard_mAP50_95=float(hd.get("mAP50_95", 0.0)),
        hard_mAP50=float(hd.get("mAP50", 0.0)),
        hard_mAP75=float(hd.get("mAP75", 0.0)),
        hard_AR_100=float(hd.get("AR_100", 0.0)),
        hard_n=int(hd.get("instances", 0)),
        delta_mAP50_95=float(dl.get("mAP50_95", 0.0)),
        delta_mAP50=float(dl.get("mAP50", 0.0)),
        delta_mAP75=float(dl.get("mAP75", 0.0)),
        delta_AR_100=float(dl.get("AR_100", 0.0)),
    )


def stratify_run(run_dir: Path, model_tag: str,
                 eval_split: str = "test") -> List[TierRow]:
    """Run the post-hoc difficulty stratify on one run dir.

    Returns one TierRow per iou_type the framework is gated for (bbox, segm),
    or a single 'skipped' row if the framework has no instance boxes
    (dinov3_semseg). The difficulty list is computed ONCE from the gt JSON and
    reused across iou_types (the gt is the same; only the pred iou_type differs).
    """
    mf = read_manifest(run_dir)
    framework = (mf or {}).get("framework", "")
    task = (mf or {}).get("task", "")
    do_box, do_seg = _ba._difficulty_iou_gating(framework, task)

    gt_path, pred_path = discover_gt_pred(run_dir, eval_split)
    if gt_path is None:
        return [TierRow(model_tag=model_tag, framework=framework, task=task,
                       iou_type="-", gt_json="", pred_json="",
                       n_instances=0, status="no_gt")]
    if pred_path is None:
        return [TierRow(model_tag=model_tag, framework=framework, task=task,
                       iou_type="-", gt_json=str(gt_path), pred_json="",
                       n_instances=0, status="no_pred")]

    gt_json = str(gt_path)
    pred_json = str(pred_path)

    # Compute the per-instance difficulty list ONCE from the gt geometry.
    dlist = _be.compute_difficulty_from_coco_gt(gt_json)

    rows: List[TierRow] = []
    if not do_box and not do_seg:
        # dinov3_semseg: no instance boxes -> skip entirely (matches gating).
        rows.append(TierRow(model_tag=model_tag, framework=framework, task=task,
                            iou_type="-", gt_json=gt_json, pred_json=pred_json,
                            n_instances=len(dlist), nms_free=None,
                            status="skipped"))
        return rows

    iou_types: List[str] = []
    if do_box:
        iou_types.append("bbox")
    if do_seg:
        iou_types.append("segm")
    for iou_type in iou_types:
        try:
            # Silence pycocotools' COCOeval.summarize() (prints ~12 lines per
            # tier x 4 tiers x iou_types) so the script's own per-run [stratify]
            # lines stay readable. Errors still surface via the except arm.
            with _suppress_stdout():
                tiers = _be.stratified_coco_eval(gt_json, pred_json, dlist,
                                                  iou_type=iou_type)
            rows.append(_tiers_to_row(model_tag, framework, task, iou_type,
                                      gt_json, pred_json, tiers))
        except Exception as e:  # noqa: BLE001 -- report, don't abort the batch
            rows.append(TierRow(model_tag=model_tag, framework=framework,
                                task=task, iou_type=iou_type,
                                gt_json=gt_json, pred_json=pred_json,
                                n_instances=len(dlist), status=f"error:{e}"))
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
# Column order: identity first, then the tab:cross-ready mask-mAP50 columns,
# then the full per-tier mAP50_95 / mAP75 / AR_100 detail for both iou_types.
CSV_COLUMNS: List[str] = [
    "model_tag", "framework", "task", "iou_type", "nms_free", "status",
    "n_instances",
    # tab:cross uses mask mAP50 (the segm row's easy/hard/delta/agg columns).
    "agg_mAP50", "easy_mAP50", "hard_mAP50", "delta_mAP50",
    # Full detail (both iou_types) for the appendix / reproducibility.
    "agg_mAP50_95", "agg_mAP75", "agg_AR_100",
    "easy_mAP50_95", "easy_mAP75", "easy_AR_100", "easy_n",
    "medium_mAP50_95", "medium_mAP50", "medium_mAP75", "medium_AR_100", "medium_n",
    "hard_mAP50_95", "hard_mAP75", "hard_AR_100", "hard_n",
    "delta_mAP50_95", "delta_mAP75", "delta_AR_100",
    "gt_json", "pred_json",
]


def write_csv(rows: List[TierRow], out_path: Path) -> None:
    """Write the stratified rows to ``out_path`` (one row per iou_type)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"[stratify] wrote {len(rows)} rows -> {out_path}")


def print_tabcross_summary(rows: List[TierRow]) -> None:
    """Print a tab:cross-style mask-mAP50 summary to stdout.

    Picks the ``segm`` row for native mask-head models and the ``bbox`` row
    for pure detectors (which have no segm iou_type under gating). This is
    the cross-task comparison the manuscript's tab:cross makes -- detectors
    use SAM2-prompted masks in the real run, but the post-hoc stratify here
    scores the predictions the run dir actually contains (bbox for detectors).
    """
    # Prefer segm when present; else bbox.
    by_tag: Dict[str, TierRow] = {}
    for r in rows:
        if r.status != "ok":
            continue
        if r.model_tag not in by_tag:
            by_tag[r.model_tag] = r
        else:
            # segm wins over bbox for the summary line.
            if r.iou_type == "segm" and by_tag[r.model_tag].iou_type != "segm":
                by_tag[r.model_tag] = r
    if not by_tag:
        print("[stratify] no 'ok' rows to summarise")
        return
    print("\n=== tab:cross-style summary (mAP50, %) ===")
    print(f"{'Model':<26} {'NMS-free':<9} {'Easy':>6} {'Hard':>6} {'Δ':>6} {'Agg.':>6}")
    for tag, r in by_tag.items():
        nms = "-" if r.nms_free is None else ("Yes" if r.nms_free else "No")
        print(f"{tag:<26} {nms:<9} {r.easy_mAP50*100:6.1f} {r.hard_mAP50*100:6.1f} "
              f"{r.delta_mAP50*100:6.1f} {r.agg_mAP50*100:6.1f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT),
                    help=f"runs root (default {DEFAULT_RUNS_ROOT})")
    ap.add_argument("--model-tags", nargs="+", default=DEFAULT_TAGS,
                    help="model tags to stratify (default: six-paradigm tab:cross set)")
    ap.add_argument("--eval-split", default="test",
                    help="split name for gt/pred selection (default test)")
    ap.add_argument("--out", default="difficulty_stratified_125e.csv",
                    help="output CSV path (default difficulty_stratified_125e.csv)")
    ap.add_argument("--no-summary", action="store_true",
                    help="suppress the stdout tab:cross-style summary")
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser()
    if not runs_root.is_dir():
        print(f"[stratify] runs root not found: {runs_root}", file=sys.stderr)
        return 2

    discovered = discover_run_dirs(runs_root, args.model_tags)
    if not discovered:
        print(f"[stratify] no run dirs matching any of {args.model_tags} "
              f"under {runs_root}", file=sys.stderr)
        return 3

    print(f"[stratify] {len(discovered)}/{len(args.model_tags)} tags found:")
    for tag, rd in discovered.items():
        print(f"  {tag:<28} -> {rd}")
    missing = [t for t in args.model_tags if t not in discovered]
    for t in missing:
        print(f"  {t:<28} -> (not found)")

    all_rows: List[TierRow] = []
    for tag in args.model_tags:
        if tag not in discovered:
            all_rows.append(TierRow(model_tag=tag, framework="", task="",
                                    iou_type="-", status="no_run_dir"))
            continue
        rows = stratify_run(discovered[tag], tag, eval_split=args.eval_split)
        all_rows.extend(rows)
        for r in rows:
            tag_nms = "-" if r.nms_free is None else ("Yes" if r.nms_free else "No")
            print(f"[stratify] {tag:<26} {r.iou_type:<5} nms={tag_nms:<4} "
                  f"status={r.status} n={r.n_instances} "
                  f"agg50={r.agg_mAP50:.3f} E={r.easy_mAP50:.3f} "
                  f"H={r.hard_mAP50:.3f} Δ={r.delta_mAP50:.3f}")

    out_path = Path(args.out).expanduser()
    write_csv(all_rows, out_path)
    if not args.no_summary:
        print_tabcross_summary(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
