#!/usr/bin/env python3
"""Unit tests for penguins/difficulty_stratify_posthoc.py.

Run: ~/cvML/bin/python penguins/test_difficulty_stratify_posthoc.py
(or ``~/cvML/bin/python -m pytest penguins/ -q`` once pytest is installed).

Tests are pure + a synthetic end-to-end run: they build a tiny COCO gt/pred in
a tmp dir (no access to ~/bench_any/runs, no orchestrator, no torch).
"""

import json
import sys
import os
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from difficulty_stratify_posthoc import (
    _rank_candidate, nms_free_for_tag, discover_gt_pred,
    discover_run_dirs, _tiers_to_row, stratify_run, TierRow, CSV_COLUMNS,
)

# Repo root on sys.path so the script's own `import bench_any` works in
# stratify_run (which imports bench_any at call time via the module-level
# import the script already did).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# _rank_candidate
# ---------------------------------------------------------------------------
class TestRankCandidate(unittest.TestCase):
    def test_catfixed_beats_fixed_beats_raw(self):
        # YOLO: catfixed is the most-correct file (BUG 4 Layer 1 fix).
        r_cat = _rank_candidate(Path("run/test/_bench_predictions_catfixed.json"), "test")
        r_fix = _rank_candidate(Path("run/test/_bench_predictions_fixed.json"), "test")
        r_raw = _rank_candidate(Path("run/test/predictions.json"), "test")
        self.assertLess(r_cat[0], r_fix[0])
        self.assertLess(r_fix[0], r_raw[0])

    def test_resolved_beats_fixed_m2f_eomt(self):
        # BUG 7 (2026-07-04): for M2F/EoMT, _bench_predictions_resolved.json is
        # the post-remap (category_id corrected to match 1-indexed GT) file the
        # orchestrator actually scored; _bench_predictions_fixed.json is a stale
        # pre-remap copy (category_id=0 -> 0 matches -> 0 mAP). _resolved must
        # rank above _fixed so post-hoc stratify picks the correct file.
        r_res = _rank_candidate(Path("run/_bench_predictions_resolved.json"), "test")
        r_fix = _rank_candidate(Path("run/_bench_predictions_fixed.json"), "test")
        self.assertLess(r_res[0], r_fix[0],
                        "BUG 7 regression: _resolved must outrank _fixed for M2F/EoMT")
        # _resolved and catfixed never coexist (M2F/EoMT vs YOLO), so both at
        # rank 0 is fine — they are disjoint across paradigms.
        r_cat = _rank_candidate(Path("run/test/_bench_predictions_catfixed.json"), "test")
        self.assertEqual(r_res[0], r_cat[0], "_resolved and catfixed share rank 0 (disjoint paradigms)")

    def test_in_split_beats_out_of_split(self):
        r_in = _rank_candidate(Path("run/test/instances_test.json"), "test")
        r_out = _rank_candidate(Path("run/train/instances_train.json"), "test")
        # Both are 'instances_*' (fixed rank 2), but in_split wins on field[1].
        self.assertEqual(r_in[0], r_out[0])
        self.assertLess(r_in[1], r_out[1])

    def test_shallower_path_beats_deeper(self):
        # Same name, different depth -> shallower wins (fewer path parts).
        r_shallow = _rank_candidate(Path("run/predictions_test.json"), "test")
        r_deep = _rank_candidate(Path("run/sub/dir/predictions_test.json"), "test")
        self.assertEqual(r_shallow[0], r_deep[0])
        self.assertEqual(r_shallow[1], r_deep[1])
        self.assertLess(r_shallow[2], r_deep[2])


# ---------------------------------------------------------------------------
# nms_free_for_tag
# ---------------------------------------------------------------------------
class TestNmsFreeForTag(unittest.TestCase):
    def test_yolo_is_nms_based(self):
        self.assertFalse(nms_free_for_tag("yolo11x-seg"))
        self.assertFalse(nms_free_for_tag("yolo26x-seg"))

    def test_nms_free_paradigms(self):
        self.assertTrue(nms_free_for_tag("rtdetr-3"))
        self.assertTrue(nms_free_for_tag("rtdetr-4"))
        self.assertTrue(nms_free_for_tag("rf-detr-seg-large"))
        self.assertTrue(nms_free_for_tag("eomt-large"))
        self.assertTrue(nms_free_for_tag("dinov3-m2f-smallplus"))
        self.assertTrue(nms_free_for_tag("dinov3-m2f-vith16plus"))

    def test_semseg_skipped(self):
        # dinov3_semseg -> None (no instance boxes -> skip entirely).
        self.assertIsNone(nms_free_for_tag("dinov3-semseg-vits16plus"))

    def test_longest_prefix_wins(self):
        # 'dinov3-m2f' must not be shadowed by a shorter 'dinov3' rule; this
        # is a guard that the prefix list is checked longest-first.
        self.assertTrue(nms_free_for_tag("dinov3-m2f-smallplus"))


# ---------------------------------------------------------------------------
# discover_gt_pred (per-framework layout)
# ---------------------------------------------------------------------------
class TestDiscoverGtPred(unittest.TestCase):
    def test_yolo_split_subdir_layout(self):
        # YOLO: gt + catfixed pred under <run>/test/
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "test").mkdir()
            gt = run / "test" / "_bench_gt_coco.json"; gt.write_text("{}")
            cat = run / "test" / "_bench_predictions_catfixed.json"; cat.write_text("[]")
            raw = run / "test" / "predictions.json"; raw.write_text("[]")
            g, p = discover_gt_pred(run, "test")
            self.assertEqual(g, gt)
            self.assertEqual(p, cat)   # catfixed wins over raw

    def test_rfdetr_root_layout(self):
        # RF-DETR: gt + predictions_test.json at run root.
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            gt = run / "_bench_gt_coco.json"; gt.write_text("{}")
            pred = run / "predictions_test.json"; pred.write_text("[]")
            g, p = discover_gt_pred(run, "test")
            self.assertEqual(g, gt)
            self.assertEqual(p, pred)

    def test_rtdetr_subdir_gt_layout(self):
        # RT-DETR-external: gt under _rtdetrv2_repo_coco/, pred at root.
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "_rtdetrv2_repo_coco").mkdir()
            gt = run / "_rtdetrv2_repo_coco" / "instances_test.json"; gt.write_text("{}")
            train_gt = run / "_rtdetrv2_repo_coco" / "instances_train.json"; train_gt.write_text("{}")
            pred = run / "predictions_test.json"; pred.write_text("[]")
            g, p = discover_gt_pred(run, "test")
            self.assertEqual(g, gt)   # test split wins over train
            self.assertEqual(p, pred)

    def test_m2f_root_layout(self):
        # DINOv3-M2F / EoMT: gt + _bench_predictions_fixed.json at root.
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            gt = run / "_bench_gt_coco.json"; gt.write_text("{}")
            pred = run / "_bench_predictions_fixed.json"; pred.write_text("[]")
            g, p = discover_gt_pred(run, "test")
            self.assertEqual(g, gt)
            self.assertEqual(p, pred)

    def test_no_pred_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "_bench_gt_coco.json").write_text("{}")
            g, p = discover_gt_pred(run, "test")
            self.assertIsNotNone(g)
            self.assertIsNone(p)


# ---------------------------------------------------------------------------
# discover_run_dirs (epoch/batch preference)
# ---------------------------------------------------------------------------
class TestDiscoverRunDirs(unittest.TestCase):
    def test_highest_epoch_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seg = root / "segment"
            seg.mkdir()
            (seg / "AnimalP_yolo11x-seg_b4_e1_i640").mkdir()
            (seg / "AnimalP_yolo11x-seg_b4_e125_i640").mkdir()
            (seg / "AnimalP_yolo11x-seg_b4_e50_i640").mkdir()
            d = discover_run_dirs(root, ["yolo11x-seg"])
            self.assertIn("yolo11x-seg", d)
            self.assertEqual(d["yolo11x-seg"].name, "AnimalP_yolo11x-seg_b4_e125_i640")

    def test_batch_tiebreak(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seg = root / "segment"
            seg.mkdir()
            # Same epoch, different batch -> larger batch wins.
            (seg / "AnimalP_rtdetr-3_b4_e125_i640").mkdir()
            (seg / "AnimalP_rtdetr-3_b8_e125_i640").mkdir()
            d = discover_run_dirs(root, ["rtdetr-3"])
            self.assertEqual(d["rtdetr-3"].name, "AnimalP_rtdetr-3_b8_e125_i640")

    def test_missing_tag_not_returned(self):
        with tempfile.TemporaryDirectory() as td:
            d = discover_run_dirs(Path(td), ["yolo11x-seg"])
            self.assertEqual(d, {})


# ---------------------------------------------------------------------------
# _tiers_to_row (flattening)
# ---------------------------------------------------------------------------
class TestTiersToRow(unittest.TestCase):
    def test_flatten_and_delta(self):
        tiers = {
            "aggregate": {"mAP50_95": 0.50, "mAP50": 0.60, "mAP75": 0.55, "AR_100": 0.70, "instances": 100},
            "Easy":      {"mAP50_95": 0.80, "mAP50": 0.90, "mAP75": 0.85, "AR_100": 0.95, "instances": 40},
            "Medium":    {"mAP50_95": 0.50, "mAP50": 0.60, "mAP75": 0.55, "AR_100": 0.70, "instances": 35},
            "Hard":      {"mAP50_95": 0.20, "mAP50": 0.30, "mAP75": 0.25, "AR_100": 0.40, "instances": 25},
            "delta":     {"mAP50_95": 0.60, "mAP50": 0.60, "mAP75": 0.60, "AR_100": 0.55},
        }
        r = _tiers_to_row("eomt-large", "eomt", "segment", "segm", "g.json", "p.json", tiers)
        self.assertEqual(r.model_tag, "eomt-large")
        self.assertEqual(r.iou_type, "segm")
        self.assertTrue(r.nms_free)            # eomt -> NMS-free
        self.assertEqual(r.n_instances, 100)
        self.assertAlmostEqual(r.agg_mAP50, 0.60)
        self.assertAlmostEqual(r.easy_mAP50, 0.90)
        self.assertAlmostEqual(r.hard_mAP50, 0.30)
        self.assertEqual(r.easy_n, 40)
        self.assertEqual(r.hard_n, 25)
        # delta is Easy - Hard (the dict's delta, passed through).
        self.assertAlmostEqual(r.delta_mAP50, 0.60)


# ---------------------------------------------------------------------------
# stratify_run end-to-end on a synthetic COCO gt/pred
# ---------------------------------------------------------------------------
def _bbox_to_polygon(x, y, w, h):
    """COCO polygon for a bbox rectangle (8 coords = 4 points)."""
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _build_synthetic_coco(tmp):
    """A tiny but valid COCO gt + matching predictions.

    3 images (640x480), 9 annotations sized to spread across difficulty tiers
    (some large+isolated = Easy, some tiny+overlapping = Hard). Predictions match
    the gt boxes exactly so mAP is non-zero. Returns (gt_path, pred_path).
    """
    images = [
        {"id": 1, "width": 640, "height": 480, "file_name": "i1.jpg"},
        {"id": 2, "width": 640, "height": 480, "file_name": "i2.jpg"},
        {"id": 3, "width": 640, "height": 480, "file_name": "i3.jpg"},
    ]
    cats = [{"id": 1, "name": "penguin", "supercategory": "bird"}]
    anns = []
    preds = []
    aid = 1
    # image 1: 3 large, well-separated boxes (Easy-ish)
    for (x, y, w, h) in [(50, 50, 120, 120), (250, 50, 120, 120), (450, 50, 120, 120)]:
        anns.append({"id": aid, "image_id": 1, "category_id": 1,
                     "bbox": [x, y, w, h], "area": w * h,
                     "segmentation": [_bbox_to_polygon(x, y, w, h)], "iscrowd": 0})
        preds.append({"image_id": 1, "category_id": 1, "bbox": [x, y, w, h],
                      "score": 0.95, "segmentation": {"counts": [], "size": [480, 640]}})
        aid += 1
    # image 2: 3 medium, slightly overlapping (Medium-ish)
    for (x, y, w, h) in [(100, 100, 60, 60), (130, 130, 60, 60), (200, 200, 60, 60)]:
        anns.append({"id": aid, "image_id": 2, "category_id": 1,
                     "bbox": [x, y, w, h], "area": w * h,
                     "segmentation": [_bbox_to_polygon(x, y, w, h)], "iscrowd": 0})
        preds.append({"image_id": 2, "category_id": 1, "bbox": [x, y, w, h],
                      "score": 0.90, "segmentation": {"counts": [], "size": [480, 640]}})
        aid += 1
    # image 3: 3 tiny, tightly clustered boxes (Hard-ish)
    for (x, y, w, h) in [(300, 300, 20, 20), (310, 310, 20, 20), (320, 320, 20, 20)]:
        anns.append({"id": aid, "image_id": 3, "category_id": 1,
                     "bbox": [x, y, w, h], "area": w * h,
                     "segmentation": [_bbox_to_polygon(x, y, w, h)], "iscrowd": 0})
        preds.append({"image_id": 3, "category_id": 1, "bbox": [x, y, w, h],
                      "score": 0.85, "segmentation": {"counts": [], "size": [480, 640]}})
        aid += 1
    gt = {"images": images, "categories": cats, "annotations": anns}
    gt_path = tmp / "_bench_gt_coco.json"
    pred_path = tmp / "_bench_predictions_fixed.json"
    with open(gt_path, "w") as fh:
        json.dump(gt, fh)
    with open(pred_path, "w") as fh:
        json.dump(preds, fh)
    return gt_path, pred_path


def _write_manifest(run, framework, task):
    mf = {"framework": framework, "task": task,
          "model_tag": run.name, "eval_split": "test"}
    with open(run / "bench_manifest.json", "w") as fh:
        json.dump(mf, fh)


class TestStratifyRun(unittest.TestCase):
    def test_detector_bbox_only(self):
        # Pure detector (task=detect, framework=rtdetr) -> 1 row, iou_type=bbox.
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _build_synthetic_coco(run)
            _write_manifest(run, "rtdetr_external", "detect")
            rows = stratify_run(run, "rtdetr-3")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].iou_type, "bbox")
            self.assertEqual(rows[0].status, "ok")
            self.assertEqual(rows[0].nms_free, True)
            self.assertGreater(rows[0].n_instances, 0)

    def test_mask_head_seg_both_iou_types(self):
        # Native mask-head seg (task=segment, framework=eomt) -> 2 rows.
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _build_synthetic_coco(run)
            _write_manifest(run, "eomt", "segment")
            rows = stratify_run(run, "eomt-large")
            self.assertEqual(len(rows), 2)
            self.assertEqual({r.iou_type for r in rows}, {"bbox", "segm"})
            for r in rows:
                self.assertEqual(r.status, "ok")
                self.assertTrue(r.nms_free)

    def test_dinov3_semseg_skipped(self):
        # dinov3_semseg -> skipped (no instance boxes).
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _build_synthetic_coco(run)
            _write_manifest(run, "dinov3_semseg", "segment")
            rows = stratify_run(run, "dinov3-semseg-vits16plus")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].status, "skipped")
            self.assertIsNone(rows[0].nms_free)

    def test_no_pred_returns_no_pred_row(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "_bench_gt_coco.json").write_text(
                json.dumps({"images": [], "categories": [], "annotations": []}))
            _write_manifest(run, "eomt", "segment")
            rows = stratify_run(run, "eomt-large")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].status, "no_pred")

    def test_delta_is_easy_minus_hard(self):
        # On the synthetic set, delta_mAP50 must equal easy_mAP50 - hard_mAP50
        # (the stratified_coco_eval contract: delta dict = Easy - Hard).
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _build_synthetic_coco(run)
            _write_manifest(run, "rtdetr_external", "detect")
            rows = stratify_run(run, "rtdetr-3")
            r = rows[0]
            self.assertAlmostEqual(r.delta_mAP50,
                                   r.easy_mAP50 - r.hard_mAP50, places=5)


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
class TestCsvColumns(unittest.TestCase):
    def test_tabcross_columns_present(self):
        # The columns tab:cross needs (mask mAP50) must be in the CSV header.
        for col in ["model_tag", "nms_free", "easy_mAP50", "hard_mAP50",
                    "delta_mAP50", "agg_mAP50"]:
            self.assertIn(col, CSV_COLUMNS, f"missing {col} in CSV_COLUMNS")


if __name__ == "__main__":
    unittest.main()
