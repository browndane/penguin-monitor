#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# bench_dataset_stats.py — ground-truth dataset profiler for the bench pipeline.
#
# Why this exists
# ---------------
# The penguin datasets are *crowded*: a single image can hold several hundred
# instances. That has two practical consequences the benchmark cares about:
#   1) detection-metric caps (pycocotools maxDets / torchmetrics
#      max_detection_thresholds) silently truncate dense images unless raised,
#      and
#   2) you simply want a feel for how many objects per image you're dealing with
#      on train / val / test before trusting any mAP number.
#
# This module reads the YOLO ground-truth labels (no model, no images decoded —
# YOLO coords are normalised so areas are computed directly) and reports, per
# split: image/instance counts, the full instances-per-image distribution
# (min / median / mean / max + percentiles), how many images exceed the 100 /
# 300 / 1000 detection caps, per-class instance counts, and normalised
# instance-area stats. Results are printed verbosely and written as JSON + CSV
# under ~/bench_any/dataset_stats/<dataset-name>/.
#
# Standalone:
#   python bench_dataset_stats.py --data <dataset-dir-with-data.yaml> [--out DIR]
#
# Library (used by bench_any.py at startup):
#   from bench_dataset_stats import summarize_dataset
#   summarize_dataset(data_dir, out_dir=None, print_summary=True)

from __future__ import annotations
import os, sys, json, csv, math, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Detection caps the benchmark applies downstream; we report how many images
# would be truncated at each (pycocotools/torchmetrics default top cap is 100,
# Ultralytics' default max_det is 300, and bench_eval.MAX_DETS raises it to 1000).
_CAP_BUCKETS = (100, 300, 1000)


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,100]) over an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _dist_stats(vals: List[float]) -> Dict[str, float]:
    """min / p25 / median / mean / p75 / p90 / p95 / p99 / max / std for a list."""
    if not vals:
        return {k: 0.0 for k in
                ("min", "p25", "median", "mean", "p75", "p90", "p95", "p99", "max", "std")}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    return {
        "min": float(s[0]),
        "p25": _percentile(s, 25),
        "median": _percentile(s, 50),
        "mean": float(mean),
        "p75": _percentile(s, 75),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "max": float(s[-1]),
        "std": float(math.sqrt(var)),
    }


def _polygon_area_norm(coords: List[float]) -> float:
    """Shoelace area of a normalised polygon (coords are x0,y0,x1,y1,...)."""
    n = len(coords) // 2
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = coords[2 * i], coords[2 * i + 1]
        x_j, y_j = coords[2 * j], coords[2 * j + 1]
        area += x_i * y_j - x_j * y_i
    return abs(area) / 2.0


def _label_for_image(ip: Path) -> Path:
    """Map a YOLO image path to its label .txt (images/ -> labels/ swap)."""
    parts = list(ip.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return ip.with_suffix(".txt")


def _resolve_split_images(cfg: dict, data_dir: Path, split_rel) -> List[Path]:
    """Resolve a data.yaml split entry to a list of image paths (dir or list-file)."""
    root = Path(cfg.get("path", data_dir)).expanduser()
    if not root.is_absolute():
        root = (Path(data_dir) / root).resolve()
    img_dir = (root / split_rel).resolve() if not Path(split_rel).is_absolute() else Path(split_rel)
    if img_dir.is_dir():
        return [p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    if img_dir.is_file():
        try:
            return [Path(l.strip()) for l in img_dir.read_text().splitlines() if l.strip()]
        except Exception:
            return []
    return []


def _class_names(cfg: dict) -> List[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in (names or [])]


def _split_stats(cfg: dict, data_dir: Path, split_rel, class_names: List[str]) -> Dict:
    """Per-split GT statistics from YOLO labels (no image decoding)."""
    img_paths = _resolve_split_images(cfg, data_dir, split_rel)
    per_image_counts: List[int] = []
    per_class_counts: Dict[int, int] = {}
    areas: List[float] = []          # normalised instance area (fraction of image)
    n_bbox = 0
    n_poly = 0
    n_empty = 0                      # images with a label file but 0 instances
    n_missing = 0                    # images with no label file at all
    for ip in img_paths:
        lbl = _label_for_image(ip)
        if not lbl.is_file():
            n_missing += 1
            per_image_counts.append(0)
            continue
        cnt = 0
        try:
            with open(lbl) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    cnt += 1
                    fields = ln.split()
                    try:
                        cls = int(float(fields[0]))
                    except (ValueError, IndexError):
                        cls = -1
                    per_class_counts[cls] = per_class_counts.get(cls, 0) + 1
                    coords = [float(x) for x in fields[1:]] if len(fields) > 1 else []
                    if len(coords) == 4:
                        n_bbox += 1
                        areas.append(max(0.0, coords[2]) * max(0.0, coords[3]))
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        n_poly += 1
                        areas.append(_polygon_area_norm(coords))
        except Exception:
            pass
        per_image_counts.append(cnt)
        if cnt == 0:
            n_empty += 1

    counts_f = [float(c) for c in per_image_counts]
    n_images = len(per_image_counts)
    n_instances = int(sum(per_image_counts))

    # How many images exceed each downstream detection cap.
    cap_exceed = {str(cap): int(sum(1 for c in per_image_counts if c > cap))
                  for cap in _CAP_BUCKETS}

    # Per-class with human-readable names where available.
    per_class_named = {}
    for cls, c in sorted(per_class_counts.items()):
        name = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
        per_class_named[name] = int(c)

    return {
        "n_images": n_images,
        "n_images_with_instances": int(sum(1 for c in per_image_counts if c > 0)),
        "n_images_empty_label": n_empty,
        "n_images_missing_label": n_missing,
        "n_instances": n_instances,
        "label_type": {"bbox": n_bbox, "polygon": n_poly},
        "instances_per_image": _dist_stats(counts_f),
        "images_exceeding_cap": cap_exceed,
        "instance_area_norm": _dist_stats(areas),
        "per_class_instances": per_class_named,
    }


def summarize_dataset(data_dir, out_dir: Optional[str] = None,
                      print_summary: bool = True) -> Dict:
    """Profile train/val(valid)/test GT for a YOLO dataset.

    Returns the full stats dict. Writes <name>_dataset_stats.json and
    <name>_dataset_stats.csv into out_dir (defaults to
    ~/bench_any/dataset_stats/<dataset-name>/). Never raises — returns {} on a
    fundamental failure (e.g. missing data.yaml) so callers can ignore it."""
    data_dir = Path(data_dir).resolve()
    yaml_path = data_dir / "data.yaml"
    try:
        import yaml as _yaml
        with open(yaml_path) as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception as e:
        if print_summary:
            print(f"[dataset_stats] ⚠️ cannot read {yaml_path}: {e}")
        return {}

    class_names = _class_names(cfg)
    # Canonical split order; 'valid' is the Roboflow spelling of 'val'.
    split_keys = [("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test")]
    splits: Dict[str, Dict] = {}
    seen_canonical = set()
    for key, canonical in split_keys:
        if canonical in seen_canonical:
            continue
        rel = cfg.get(key)
        if not rel:
            continue
        st = _split_stats(cfg, data_dir, rel, class_names)
        splits[canonical] = st
        seen_canonical.add(canonical)

    result = {
        "dataset": data_dir.name,
        "data_dir": str(data_dir),
        "classes": class_names,
        "detection_caps_reported": list(_CAP_BUCKETS),
        "splits": splits,
    }

    # Persist.
    if out_dir is None:
        out_dir = Path.home() / "bench_any" / "dataset_stats" / data_dir.name
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{data_dir.name}_dataset_stats.json", "w") as fh:
            json.dump(result, fh, indent=2)
        _write_csv(out_dir / f"{data_dir.name}_dataset_stats.csv", result)
    except Exception as e:
        if print_summary:
            print(f"[dataset_stats] ⚠️ failed to write stats to {out_dir}: {e}")

    if print_summary:
        _print_summary(result, out_dir)
    return result


def _write_csv(path: Path, result: Dict) -> None:
    """One row per split with the headline numbers (per-class kept in JSON)."""
    fields = [
        "split", "n_images", "n_images_with_instances", "n_images_empty_label",
        "n_images_missing_label", "n_instances",
        "ipi_min", "ipi_median", "ipi_mean", "ipi_p95", "ipi_p99", "ipi_max", "ipi_std",
        "imgs_gt_100", "imgs_gt_300", "imgs_gt_1000",
        "label_bbox", "label_polygon",
        "area_norm_min", "area_norm_median", "area_norm_mean", "area_norm_max",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for split, st in result.get("splits", {}).items():
            ipi = st["instances_per_image"]
            area = st["instance_area_norm"]
            cap = st["images_exceeding_cap"]
            w.writerow({
                "split": split,
                "n_images": st["n_images"],
                "n_images_with_instances": st["n_images_with_instances"],
                "n_images_empty_label": st["n_images_empty_label"],
                "n_images_missing_label": st["n_images_missing_label"],
                "n_instances": st["n_instances"],
                "ipi_min": round(ipi["min"], 3),
                "ipi_median": round(ipi["median"], 3),
                "ipi_mean": round(ipi["mean"], 3),
                "ipi_p95": round(ipi["p95"], 3),
                "ipi_p99": round(ipi["p99"], 3),
                "ipi_max": round(ipi["max"], 3),
                "ipi_std": round(ipi["std"], 3),
                "imgs_gt_100": cap.get("100", 0),
                "imgs_gt_300": cap.get("300", 0),
                "imgs_gt_1000": cap.get("1000", 0),
                "label_bbox": st["label_type"]["bbox"],
                "label_polygon": st["label_type"]["polygon"],
                "area_norm_min": round(area["min"], 6),
                "area_norm_median": round(area["median"], 6),
                "area_norm_mean": round(area["mean"], 6),
                "area_norm_max": round(area["max"], 6),
            })


def _print_summary(result: Dict, out_dir: Path) -> None:
    name = result.get("dataset", "?")
    print("\n" + "=" * 78)
    print(f"📊 DATASET GROUND-TRUTH STATS — {name}")
    print(f"   classes ({len(result.get('classes', []))}): {result.get('classes', [])}")
    print("=" * 78)
    for split, st in result.get("splits", {}).items():
        ipi = st["instances_per_image"]
        cap = st["images_exceeding_cap"]
        area = st["instance_area_norm"]
        print(f"\n── {split.upper()} "
              f"── {st['n_images']} images, {st['n_instances']} instances "
              f"({st['n_images_with_instances']} labelled, "
              f"{st['n_images_empty_label']} empty, "
              f"{st['n_images_missing_label']} missing-label)")
        print(f"   instances/image:  min={ipi['min']:.0f}  median={ipi['median']:.1f}  "
              f"mean={ipi['mean']:.1f}  p95={ipi['p95']:.0f}  p99={ipi['p99']:.0f}  "
              f"max={ipi['max']:.0f}  std={ipi['std']:.1f}")
        print(f"   crowded images:   >100={cap.get('100', 0)}  "
              f">300={cap.get('300', 0)}  >1000={cap.get('1000', 0)}")
        print(f"   label types:      bbox={st['label_type']['bbox']}  "
              f"polygon={st['label_type']['polygon']}")
        print(f"   inst. area (norm): min={area['min']:.5f}  median={area['median']:.5f}  "
              f"mean={area['mean']:.5f}  max={area['max']:.5f}")
        pc = st["per_class_instances"]
        if pc:
            shown = ", ".join(f"{k}={v}" for k, v in list(pc.items())[:8])
            more = "" if len(pc) <= 8 else f"  (+{len(pc) - 8} more)"
            print(f"   per-class:        {shown}{more}")
    print("\n" + "-" * 78)
    print(f"📝 Saved: {out_dir}/{name}_dataset_stats.json")
    print(f"          {out_dir}/{name}_dataset_stats.csv")
    print("-" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Profile YOLO ground-truth (instances/image, areas, per-class) "
                    "across train/val/test.")
    ap.add_argument("--data", required=True,
                    help="YOLO dataset dir containing data.yaml")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: ~/bench_any/dataset_stats/<name>/)")
    ap.add_argument("--quiet", action="store_true", help="Suppress the printed summary")
    args = ap.parse_args()
    res = summarize_dataset(args.data, out_dir=args.out, print_summary=not args.quiet)
    if not res:
        sys.exit(1)


if __name__ == "__main__":
    main()
