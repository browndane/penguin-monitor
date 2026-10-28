#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Unit tests for penguins/sanity_sieve.py. Run: ~/cvML/bin/python -m pytest penguins/ -q
(or plain ``python penguins/test_sanity_sieve.py``)."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sanity_sieve import (
    A_MIN, RHO_MAX, EDGE_BORDER, EDGE_STUB_AREA_K, NMS_IOU,
    reject_area, reject_dimension, reject_edge_stub,
    mask_passes, class_agnostic_nms, sieve_masks, SieveStats,
)


def _mask_from_bbox(x1, y1, x2, y2, wh=(640, 480)):
    """Build a filled-rect mask for a bbox in an image of size wh=(W,H)."""
    w, h = wh
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2 + 1, x1:x2 + 1] = 1
    return m


class TestRejectArea(unittest.TestCase):
    def test_below_amin_rejected(self):
        self.assertTrue(reject_area(100))
        self.assertTrue(reject_area(323))   # just under

    def test_at_amin_kept(self):
        self.assertFalse(reject_area(324))   # exactly A_MIN -> not < A_MIN

    def test_above_amin_kept(self):
        self.assertFalse(reject_area(325))
        self.assertFalse(reject_area(10000))


class TestRejectDimension(unittest.TestCase):
    def test_full_image_strip_rejected(self):
        # bbox spans the whole width of a 640x480 image -> fx = 1.0 >= rho_max
        bb = (0, 100, 639, 110)   # 640 wide, 11 tall -- a thin strip
        self.assertTrue(reject_dimension(bb, (640, 480)))

    def test_small_box_kept(self):
        bb = (100, 100, 150, 150)   # 51x51, fx ~ 0.08
        self.assertFalse(reject_dimension(bb, (640, 480)))

    def test_degenerate_bbox_rejected(self):
        self.assertTrue(reject_dimension((10, 10, 5, 5), (640, 480)))


class TestRejectEdgeStub(unittest.TestCase):
    def test_small_near_border_rejected(self):
        # 5x5 box touching left border, area 25 < 8 * 324
        bb = (0, 100, 4, 104)
        self.assertTrue(reject_edge_stub(bb, (640, 480), area=25))

    def test_large_near_border_kept(self):
        # 100x100 box touching border, area 10000 >= 8 * 324 = 2592
        bb = (0, 100, 99, 199)
        self.assertFalse(reject_edge_stub(bb, (640, 480), area=10000))

    def test_small_far_from_border_kept(self):
        bb = (300, 300, 304, 304)   # not near any border
        self.assertFalse(reject_edge_stub(bb, (640, 480), area=25))


class TestMaskPasses(unittest.TestCase):
    def test_clean_central_mask_passes(self):
        m = _mask_from_bbox(100, 100, 150, 150)   # 51x51, area 2601, central
        self.assertTrue(mask_passes(m, (640, 480)))

    def test_tiny_fragment_rejected(self):
        m = _mask_from_bbox(100, 100, 105, 105)   # 6x6, area 36 < 324
        self.assertFalse(mask_passes(m, (640, 480)))

    def test_edge_stub_rejected(self):
        m = _mask_from_bbox(0, 100, 4, 104)   # touches border, tiny
        self.assertFalse(mask_passes(m, (640, 480)))

    def test_edge_full_penguin_kept(self):
        # A big mask that touches the border but is large enough.
        m = _mask_from_bbox(0, 100, 99, 199)
        self.assertTrue(mask_passes(m, (640, 480)))

    def test_empty_mask_rejected(self):
        m = np.zeros((480, 640), dtype=np.uint8)
        self.assertFalse(mask_passes(m, (640, 480)))


class TestClassAgnosticNMS(unittest.TestCase):
    def test_overlap_suppressed(self):
        boxes = [(100, 100, 200, 200), (102, 102, 202, 202), (400, 400, 500, 500)]
        scores = [0.9, 0.8, 0.7]
        keep = class_agnostic_nms(boxes, scores, iou_thr=0.55)
        # First two overlap heavily -> only the higher-scored first survives.
        self.assertEqual(keep, [0, 2])

    def test_no_overlap_all_kept(self):
        boxes = [(0, 0, 10, 10), (100, 100, 110, 110), (200, 200, 210, 210)]
        scores = [0.5, 0.6, 0.4]
        keep = class_agnostic_nms(boxes, scores, iou_thr=0.55)
        self.assertEqual(keep, [1, 0, 2])   # score-descending

    def test_empty(self):
        self.assertEqual(class_agnostic_nms([], []), [])


class TestSieveMasks(unittest.TestCase):
    def test_full_pipeline_counts(self):
        masks = [
            _mask_from_bbox(100, 100, 150, 150),   # 0: clean, kept
            _mask_from_bbox(105, 105, 155, 155),   # 1: IoU 0.685 with 0 -> NMSed by 0
            _mask_from_bbox(10, 10, 12, 12),         # 2: tiny fragment, rejected (area)
            _mask_from_bbox(0, 100, 29, 129),        # 3: edge-stub, area 900 in [324, 2592), near border
            _mask_from_bbox(0, 0, 639, 10),         # 4: full-width strip, rejected (dim)
        ]
        scores = [0.9, 0.85, 0.7, 0.6, 0.5]
        kept, stats = sieve_masks(masks, (640, 480), scores=scores)
        # Tests apply in spec order with mutually-exclusive accounting:
        #   mask 2 -> area (36 < 324)
        #   mask 3 -> edge-stub (area 900 passes area test, near border, < 8*A_min)
        #   mask 4 -> dimension (spans full image width)
        self.assertEqual(stats.n_reject_area, 1)
        self.assertEqual(stats.n_reject_edge, 1)
        self.assertEqual(stats.n_reject_dim, 1)
        # 0 and 1 overlap heavily -> NMS drops one.
        self.assertEqual(stats.n_reject_nms, 1)
        self.assertEqual(stats.n_out, 1)
        self.assertEqual(kept, [0])

    def test_nms_off(self):
        masks = [
            _mask_from_bbox(100, 100, 150, 150),
            _mask_from_bbox(120, 120, 170, 170),
        ]
        kept, stats = sieve_masks(masks, (640, 480), scores=[0.9, 0.8], apply_nms=False)
        self.assertEqual(stats.n_reject_nms, 0)
        self.assertEqual(len(kept), 2)

    def test_empty_input(self):
        kept, stats = sieve_masks([], (640, 480))
        self.assertEqual(kept, [])
        self.assertEqual(stats.n_in, 0)


class TestSieveBoundaryEdges(unittest.TestCase):
    """A-AUDIT-1 edge cases: exact-constant boundary conditions."""

    def test_mask_area_exactly_amin_kept(self):
        # 18x18 mask -> area == A_MIN (324). Spec: reject only if area < A_MIN.
        # Place it in the image interior so the edge-stub test does not fire.
        m = _mask_from_bbox(100, 100, 117, 117)  # inclusive bbox -> 18x18 = 324 px
        self.assertEqual(int(m.sum()), 324)
        self.assertTrue(mask_passes(m, (640, 480)))

    def test_mask_area_one_below_amin_rejected(self):
        m = _mask_from_bbox(100, 100, 116, 117)  # 17x18 = 306 px < 324
        self.assertEqual(int(m.sum()), 306)
        self.assertFalse(mask_passes(m, (640, 480)))

    def test_mask_touching_edge_within_border_band_rejected_if_small(self):
        # x1=3 is strictly within the 4px border band (3 < 4) -> near_edge True.
        # Small area (< 8*A_min=2592) -> edge-stub REJECTED.
        bb = (3, 100, 200, 300)
        self.assertTrue(reject_edge_stub(bb, (640, 480), area=25))      # small -> rejected
        self.assertFalse(reject_edge_stub(bb, (640, 480), area=10000))  # large -> kept

    def test_mask_at_exactly_border_px_is_not_near_edge(self):
        # The "within 4 px" test is strict: x1 < border. x1 == EDGE_BORDER (4)
        # is NOT near-edge, so the edge-stub test does not fire even for a tiny
        # mask. This is the documented boundary: a stub starting exactly at 4px
        # from the border is treated as an interior mask.
        bb = (4, 100, 8, 103)  # x1 == border, tiny area
        self.assertFalse(reject_edge_stub(bb, (640, 480), area=25))

    def test_mask_just_outside_border_band_kept_regardless(self):
        # x1=5 is OUTSIDE the 4px border band -> not near edge -> edge test does not fire.
        bb = (5, 100, 8, 103)  # tiny but NOT near edge
        self.assertFalse(reject_edge_stub(bb, (640, 480), area=25))


if __name__ == "__main__":
    unittest.main()
