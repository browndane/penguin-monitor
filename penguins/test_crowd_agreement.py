#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Unit tests for penguins/crowd_agreement.py. Run: ~/cvML/bin/python -m pytest penguins/ -q
(or plain ``python penguins/test_crowd_agreement.py``)."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from crowd_agreement import (
    K_DEFAULT, TAU_BYPASS_DEFAULT,
    annotator_agreement, point_in_mask, keep_mask, filter_by_agreement,
)


def _rect_mask(x1, y1, x2, y2, wh=(640, 480)):
    w, h = wh
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2 + 1, x1:x2 + 1] = 1
    return m


class TestAnnotatorAgreement(unittest.TestCase):
    def test_distinct_annotators_counted_once(self):
        # Three clicks inside, but only 2 distinct annotator ids.
        clicks = [(110, 110, 1), (120, 120, 1), (130, 130, 2)]
        xs = np.array([c[0] for c in clicks])
        ys = np.array([c[1] for c in clicks])
        ids = np.array([c[2] for c in clicks])
        m = _rect_mask(100, 100, 200, 200)
        self.assertEqual(annotator_agreement(m, xs, ys, ids), 2)

    def test_clicks_outside_count_zero(self):
        clicks = [(0, 0, 1), (10, 10, 2)]
        xs = np.array([c[0] for c in clicks]); ys = np.array([c[1] for c in clicks])
        ids = np.array([c[2] for c in clicks])
        m = _rect_mask(100, 100, 200, 200)
        self.assertEqual(annotator_agreement(m, xs, ys, ids), 0)

    def test_no_clicks_zero(self):
        m = _rect_mask(100, 100, 200, 200)
        self.assertEqual(annotator_agreement(m, np.array([]), np.array([]), np.array([])), 0)

    def test_bbox_fallback(self):
        # No mask; clicks inside the bbox should count.
        clicks = [(110, 110, 1), (150, 150, 3)]
        xs = np.array([c[0] for c in clicks]); ys = np.array([c[1] for c in clicks])
        ids = np.array([c[2] for c in clicks])
        self.assertEqual(annotator_agreement(None, xs, ys, ids, bbox=(100, 100, 200, 200)), 2)


class TestPointInMask(unittest.TestCase):
    def test_inside_dense(self):
        m = _rect_mask(100, 100, 200, 200)
        self.assertTrue(point_in_mask(150, 150, mask=m))
        self.assertFalse(point_in_mask(50, 50, mask=m))

    def test_inside_bbox(self):
        self.assertTrue(point_in_mask(150, 150, bbox=(100, 100, 200, 200)))
        self.assertFalse(point_in_mask(50, 50, bbox=(100, 100, 200, 200)))


class TestKeepMask(unittest.TestCase):
    def test_agreement_threshold(self):
        self.assertTrue(keep_mask(a_mi=3, s_i=0.1))
        self.assertFalse(keep_mask(a_mi=2, s_i=0.1))

    def test_confidence_bypass(self):
        self.assertTrue(keep_mask(a_mi=0, s_i=0.5))
        self.assertTrue(keep_mask(a_mi=0, s_i=0.9))
        self.assertFalse(keep_mask(a_mi=0, s_i=0.49))

    def test_no_clicks_bypass(self):
        # Image has no clicks -> kept even with low agreement AND low confidence.
        self.assertTrue(keep_mask(a_mi=0, s_i=0.1, image_has_clicks=False))

    def test_dropped_when_low_both(self):
        self.assertFalse(keep_mask(a_mi=1, s_i=0.2, image_has_clicks=True))


class TestFilterByAgreement(unittest.TestCase):
    def test_kept_by_agreement_kept_by_conf_kept_by_noclicks_dropped(self):
        masks = [
            _rect_mask(100, 100, 200, 200),   # 0: 3 annotators click inside -> kept (agreement)
            _rect_mask(300, 300, 400, 400),    # 1: no clicks but score 0.9 -> kept (conf bypass)
            _rect_mask(10, 10, 60, 60),         # 2: low agreement, low score -> dropped
        ]
        scores = [0.2, 0.9, 0.1]
        # 3 distinct annotators click inside mask 0.
        clicks = [(110, 110, 1), (120, 120, 2), (130, 130, 3)]
        kept, stats = filter_by_agreement(masks, scores, clicks)
        self.assertIn(0, kept)
        self.assertIn(1, kept)
        self.assertNotIn(2, kept)
        self.assertEqual(stats.n_keep_agreement, 1)
        self.assertEqual(stats.n_keep_bypass_conf, 1)
        self.assertEqual(stats.n_drop, 1)

    def test_no_clicks_on_image_keeps_all(self):
        masks = [_rect_mask(100, 100, 200, 200), _rect_mask(300, 300, 400, 400)]
        kept, stats = filter_by_agreement(masks, [0.1, 0.2], clicks_per_image=None)
        self.assertEqual(kept, [0, 1])
        self.assertEqual(stats.n_keep_bypass_noclicks, 2)

    def test_k_threshold_boundary(self):
        # Exactly k=3 distinct annotators -> kept by agreement.
        masks = [_rect_mask(100, 100, 200, 200)]
        clicks = [(110, 110, 1), (120, 120, 2), (130, 130, 3)]
        kept, _ = filter_by_agreement(masks, [0.0], clicks)
        self.assertEqual(kept, [0])

    def test_two_distinct_annotators_below_k_dropped(self):
        masks = [_rect_mask(100, 100, 200, 200)]
        clicks = [(110, 110, 1), (120, 120, 2)]   # only 2 distinct < k=3
        kept, _ = filter_by_agreement(masks, [0.1], clicks)
        self.assertEqual(kept, [])

    def test_empty_input(self):
        kept, stats = filter_by_agreement([], [], None)
        self.assertEqual(kept, [])
        self.assertEqual(stats.n_in, 0)


class TestAgreementAuditEdges(unittest.TestCase):
    """A-AUDIT-1 edge cases: 1 annotator + 1 click inside mask (k=3 reject
    unless tau_bypass), and zero-clicks bypass keeps ALL regardless of score."""

    def test_one_annotator_one_click_inside_low_score_dropped(self):
        # k=3 distinct-annotator threshold; 1 annotator (any clicks) is far below.
        masks = [_rect_mask(100, 100, 200, 200)]
        clicks = [(150, 150, 1)]   # 1 annotator, 1 click inside
        kept, stats = filter_by_agreement(masks, [0.1], clicks)
        self.assertEqual(kept, [])
        self.assertEqual(stats.n_drop, 1)

    def test_one_annotator_one_click_high_score_bypassed(self):
        masks = [_rect_mask(100, 100, 200, 200)]
        clicks = [(150, 150, 1)]
        # score >= tau_bypass (0.5) -> kept via confidence bypass, NOT agreement.
        kept, stats = filter_by_agreement(masks, [0.6], clicks)
        self.assertEqual(kept, [0])
        self.assertEqual(stats.n_keep_bypass_conf, 1)
        self.assertEqual(stats.n_keep_agreement, 0)

    def test_zero_clicks_bypass_keeps_all_even_low_score(self):
        # No clicks anywhere on the image -> every mask kept, score irrelevant.
        masks = [_rect_mask(100, 100, 200, 200), _rect_mask(300, 300, 400, 400)]
        kept, stats = filter_by_agreement(masks, [0.01, 0.0], clicks_per_image=[])
        self.assertEqual(kept, [0, 1])
        self.assertEqual(stats.n_keep_bypass_noclicks, 2)
        self.assertEqual(stats.n_drop, 0)

    def test_score_exactly_tau_bypass_kept(self):
        # s_i == tau_bypass (0.5) satisfies s_i >= tau_bypass -> kept.
        self.assertTrue(keep_mask(a_mi=0, s_i=0.5))


if __name__ == "__main__":
    unittest.main()
