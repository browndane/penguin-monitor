#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""
temporal_coverage.py — Obj 4 Phase 0 temporal-coverage checker (CJS feasibility gate).

Walks a dataset directory, reads EXIF DateTime (tag 306) from every JPEG via Pillow, buckets
captures by day and by week, reports occasion count, span (first→last date), per-camera-location
breakdown (if the directory structure encodes camera locations as subdirs), and emits a clear
verdict — "CJS-feasible (≥2 occasions over ≥1 week)" vs "single-occasion → pivot to
Lincoln-Petersen".

Why this exists: CJS (Cormack-Jolly-Seber) apparent-survival estimation needs ≥2 occasions of
repeated observations to construct encounter histories. Single-occasion data can only support
closed-population Lincoln-Petersen (weaker, still novel). This tool is the make-or-break gate
for whether Obj 4 can even do CJS on a given dataset. Pure read-only metadata inspection — no
training, no orchestrator, no `~/bench_any/runs/` writes.

Usage:
    ~/cvML/bin/python penguins/temporal_coverage.py --dataset <dir> [--camera-subdirs] [--json]
    ~/cvML/bin/python penguins/temporal_coverage.py --dataset /mnt/.../CampusPenguin --camera-subdirs

Exit codes: 0 = tool ran (see verdict in output); 1 = dataset dir not found; 2 = no images with EXIF.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
    from PIL.ExifTags import Base
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


# ---------------------------------------------------------------------------
# EXIF date extraction
# ---------------------------------------------------------------------------
# EXIF DateTime tag id 306 (0x0132) = "DateTime" = "YYYY:MM:DD HH:MM:SS".
DATETIME_TAG = 306


def parse_exif_datetime(s: str) -> Optional[datetime.datetime]:
    """Parse an EXIF DateTime string 'YYYY:MM:DD HH:MM:SS' → datetime; None on failure."""
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s.strip(), "%Y:%m:%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def image_dates(root: Path, recursive: bool = True) -> List[Tuple[Path, datetime.datetime]]:
    """Walk `root`, return [(path, capture_datetime)] for every JPEG with a usable EXIF DateTime."""
    if Image is None:
        raise RuntimeError("Pillow is required (pip install pillow) — not in the active venv?")
    out: List[Tuple[Path, datetime.datetime]] = []
    globber = root.rglob if recursive else root.glob
    for p in globber("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        try:
            with Image.open(p) as im:
                exif = im.getexif()
        except Exception:
            continue
        dt = parse_exif_datetime(exif.get(DATETIME_TAG, ""))
        if dt is not None:
            out.append((p, dt))
    return out


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------
def week_key(d: datetime.date) -> str:
    """ISO year + ISO week number — a stable week bucket key."""
    iy, iw, _ = d.isocalendar()
    return f"{iy}-W{iw:02d}"


def analyze(dates: List[datetime.datetime], camera_subdirs: bool, root: Path) -> Dict:
    """Build the coverage report dict from a list of capture datetimes."""
    if not dates:
        return {
            "verdict": "no_data",
            "n_images": 0,
            "reason": "no JPEGs with a usable EXIF DateTime were found",
        }

    days = {d.date() for d in dates}
    weeks = {week_key(d.date()) for d in dates}
    sorted_days = sorted(days)
    span_days = (sorted_days[-1] - sorted_days[0]).days
    span_weeks = span_days / 7.0

    # Per-camera breakdown if the dataset encodes cameras as top-level subdirs.
    per_camera: Dict[str, Dict] = {}
    if camera_subdirs:
        cam_buckets: Dict[str, List[datetime.datetime]] = collections.defaultdict(list)
        for p, d in image_dates(root, recursive=True):
            # Camera = the top-level subdir under root.
            try:
                cam = p.relative_to(root).parts[0]
            except ValueError:
                cam = "(root)"
            cam_buckets[cam].append(d)
        for cam, cam_dates in cam_buckets.items():
            cam_days = {d.date() for d in cam_dates}
            cam_weeks = {week_key(d.date()) for d in cam_dates}
            per_camera[cam] = {
                "n_images": len(cam_dates),
                "n_days": len(cam_days),
                "n_weeks": len(cam_weeks),
                "first": min(cam_dates).isoformat(),
                "last": max(cam_dates).isoformat(),
            }

    # Verdict: CJS needs ≥2 occasions over ≥1 week. Conservative reading: ≥2 distinct days AND
    # span ≥7 days. If only one day (or all captures within a single day), it's single-occasion.
    n_days = len(days)
    if n_days >= 2 and span_days >= 7:
        verdict = "cjs_feasible"
        reason = (
            f"{n_days} capture days over a {span_days}-day span "
            f"({len(weeks)} ISO weeks) — enough occasions for CJS encounter histories"
        )
    elif n_days >= 2 and span_days < 7:
        verdict = "lincoln_petersen_only"
        reason = (
            f"{n_days} capture days but only a {span_days}-day span (<1 week) — "
            "closed-population Lincoln-Petersen is the viable estimator; CJS needs ≥1 week"
        )
    else:
        verdict = "single_occasion"
        reason = (
            "all captures on a single day — no repeated occasions; CJS infeasible, "
            "Lincoln-Petersen also infeasible (needs ≥2 capture occasions)"
        )

    return {
        "verdict": verdict,
        "n_images": len(dates),
        "n_days": n_days,
        "n_weeks": len(weeks),
        "first": min(dates).isoformat(),
        "last": max(dates).isoformat(),
        "span_days": span_days,
        "span_weeks": round(span_weeks, 2),
        "per_camera": per_camera,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dataset", required=True, help="dataset root dir to scan")
    ap.add_argument("--camera-subdirs", action="store_true",
                    help="treat top-level subdirs as camera locations (CampusPenguin layout)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON on stdout")
    ap.add_argument("--no-recursive", action="store_true",
                    help="scan only the top level (not subdirs)")
    args = ap.parse_args(argv)

    root = Path(args.dataset).expanduser()
    if not root.is_dir():
        print(f"[temporal] dataset dir not found: {root}", file=sys.stderr)
        return 1

    found = image_dates(root, recursive=not args.no_recursive)
    dates = [d for _, d in found]
    if not dates:
        print(f"[temporal] no JPEGs with a usable EXIF DateTime under {root}", file=sys.stderr)
        return 2

    report = analyze(dates, args.camera_subdirs, root)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"[temporal] dataset: {root}")
        print(f"  images with EXIF date: {report['n_images']}")
        print(f"  capture days: {report['n_days']}   ISO weeks: {report['n_weeks']}")
        print(f"  first: {report['first']}   last: {report['last']}")
        print(f"  span: {report['span_days']} days ({report['span_weeks']} weeks)")
        if report.get("per_camera"):
            print("  per camera-location:")
            for cam, c in sorted(report["per_camera"].items()):
                print(f"    {cam:<28} n={c['n_images']:<4} days={c['n_days']:<3} "
                      f"weeks={c['n_weeks']:<3} {c['first'][:10]} → {c['last'][:10]}")
        verdict_label = {
            "cjs_feasible": "CJS-FEASIBLE",
            "lincoln_petersen_only": "LINCOLN-PETERSEN ONLY (CJS infeasible — span <1 week)",
            "single_occasion": "SINGLE OCCASION (CJS + Lincoln-Petersen both infeasible)",
            "no_data": "NO DATA",
        }.get(report["verdict"], report["verdict"])
        print(f"  VERDICT: {verdict_label}")
        print(f"  reason:  {report['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
