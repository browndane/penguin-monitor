#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""
Unit tests for temporal_coverage.py — synthetic EXIF fixtures covering the three verdict branches:
multi-week (CJS-feasible), sub-week multi-day (Lincoln-Petersen only), and single-day (infeasible).

Run: ~/cvML/bin/python penguins/test_temporal_coverage.py
(unittest, not pytest, which this project's environment does not carry.)
"""
import datetime
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temporal_coverage as tc  # noqa: E402


def _make_jpeg_with_exif(path: Path, dt: datetime.datetime) -> None:
    """Write a minimal JPEG with an EXIF DateTime tag set to `dt`."""
    from PIL import Image
    exif = Image.Exif()
    exif[tc.DATETIME_TAG] = dt.strftime("%Y:%m:%d %H:%M:%S")
    Image.new("RGB", (8, 8), color="black").save(path, exif=exif, format="JPEG")


class TestParseExifDateTime(unittest.TestCase):
    def test_valid(self):
        d = tc.parse_exif_datetime("2023:01:12 18:50:00")
        self.assertEqual(d, datetime.datetime(2023, 1, 12, 18, 50, 0))

    def test_empty(self):
        self.assertIsNone(tc.parse_exif_datetime(""))

    def test_garbage(self):
        self.assertIsNone(tc.parse_exif_datetime("not a date"))

    def test_wrong_format(self):
        self.assertIsNone(tc.parse_exif_datetime("2023-01-12 18:50:00"))


class TestAnalyze(unittest.TestCase):
    def _dates(self, *ymd):
        return [datetime.datetime(*d) for d in ymd]

    def test_cjs_feasible_multi_week(self):
        dates = self._dates(
            (2023, 1, 12, 9, 0, 0), (2023, 1, 13, 9, 0, 0), (2023, 4, 21, 9, 0, 0),
        )
        rep = tc.analyze(dates, camera_subdirs=False, root=Path("/tmp"))
        self.assertEqual(rep["verdict"], "cjs_feasible")
        self.assertEqual(rep["n_days"], 3)
        self.assertGreaterEqual(rep["span_days"], 7)
        self.assertIn("enough occasions", rep["reason"])

    def test_lincoln_petersen_only_sub_week(self):
        # Two days, but only 3 days apart — multi-day but <1 week.
        dates = self._dates((2023, 1, 12, 9, 0, 0), (2023, 1, 15, 9, 0, 0))
        rep = tc.analyze(dates, camera_subdirs=False, root=Path("/tmp"))
        self.assertEqual(rep["verdict"], "lincoln_petersen_only")
        self.assertEqual(rep["n_days"], 2)
        self.assertLess(rep["span_days"], 7)
        self.assertIn("Lincoln-Petersen", rep["reason"])

    def test_single_occasion(self):
        dates = self._dates((2023, 1, 12, 9, 0, 0), (2023, 1, 12, 14, 0, 0))
        rep = tc.analyze(dates, camera_subdirs=False, root=Path("/tmp"))
        self.assertEqual(rep["verdict"], "single_occasion")
        self.assertEqual(rep["n_days"], 1)
        self.assertIn("single day", rep["reason"])

    def test_no_data(self):
        rep = tc.analyze([], camera_subdirs=False, root=Path("/tmp"))
        self.assertEqual(rep["verdict"], "no_data")
        self.assertEqual(rep["n_images"], 0)

    def test_week_key_stable(self):
        self.assertEqual(tc.week_key(datetime.date(2023, 1, 12)),
                         tc.week_key(datetime.date(2023, 1, 13)))  # same ISO week


class TestEndToEnd(unittest.TestCase):
    """Round-trip: write synthetic JPEGs with EXIF, scan them, check the verdict."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tempcov_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_cam(self, cam: str, name: str, dt: datetime.datetime) -> Path:
        cdir = Path(self.tmp) / cam
        cdir.mkdir(parents=True, exist_ok=True)
        p = cdir / name
        _make_jpeg_with_exif(p, dt)
        return p

    def test_multi_camera_cjs_feasible(self):
        # Two cameras, captures spanning Jan–Apr 2023 — CJS-feasible.
        self._write_cam("Lighthouse", "a.JPG", datetime.datetime(2023, 1, 12, 9, 0, 0))
        self._write_cam("Lighthouse", "b.JPG", datetime.datetime(2023, 1, 13, 9, 0, 0))
        self._write_cam("Slipway", "c.JPG", datetime.datetime(2023, 4, 21, 9, 0, 0))
        rep = tc.analyze(
            [d for _, d in tc.image_dates(Path(self.tmp), recursive=True)],
            camera_subdirs=True, root=Path(self.tmp),
        )
        self.assertEqual(rep["verdict"], "cjs_feasible")
        self.assertEqual(rep["n_days"], 3)
        self.assertIn("Lighthouse", rep["per_camera"])
        self.assertIn("Slipway", rep["per_camera"])
        self.assertEqual(rep["per_camera"]["Lighthouse"]["n_days"], 2)
        self.assertEqual(rep["per_camera"]["Slipway"]["n_days"], 1)

    def test_single_day_infeasible(self):
        # All captures on one day → single-occasion.
        self._write_cam("Cam1", "a.JPG", datetime.datetime(2023, 1, 12, 8, 0, 0))
        self._write_cam("Cam1", "b.JPG", datetime.datetime(2023, 1, 12, 14, 0, 0))
        rep = tc.analyze(
            [d for _, d in tc.image_dates(Path(self.tmp), recursive=True)],
            camera_subdirs=True, root=Path(self.tmp),
        )
        self.assertEqual(rep["verdict"], "single_occasion")
        self.assertEqual(rep["n_days"], 1)


class TestMainCLI(unittest.TestCase):
    def test_missing_dir_returns_1(self):
        rc = tc.main(["--dataset", "/nonexistent/path/that/does/not/exist"])
        self.assertEqual(rc, 1)

    def test_json_output_shape(self):
        tmp = tempfile.mkdtemp(prefix="tempcov_cli_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        _make_jpeg_with_exif(Path(tmp) / "x.JPG", datetime.datetime(2023, 1, 12, 9, 0, 0))
        _make_jpeg_with_exif(Path(tmp) / "y.JPG", datetime.datetime(2023, 4, 21, 9, 0, 0))
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            rc = tc.main(["--dataset", tmp, "--json"])
        finally:
            sys.stdout = real_stdout
        self.assertEqual(rc, 0)
        import json
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["verdict"], "cjs_feasible")
        self.assertEqual(rep["n_days"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
