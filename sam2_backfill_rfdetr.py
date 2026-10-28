#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""One-shot SAM2 backfill for RF-DETR run dirs that already have predictions.

Reads each run dir's predictions_test.json + _bench_gt_coco.json, runs SAM2,
and appends sam2_* columns to the results.csv (updates in-place, idempotent).
"""

import os, sys, json, csv
from pathlib import Path

# Add bench dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_eval as _be

SAM2_CKPT = os.environ.get(
    "SAM2_CKPT", "/home/user/sam2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = os.environ.get(
    "SAM2_CONFIG", "/home/user/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

RUNS_ROOT = Path.home() / "bench_any" / "runs" / "segment"


def resolve_test_images_dir(data_yaml_path: str) -> Path | None:
    """Resolve the test images directory from a dataset data.yaml."""
    import yaml
    try:
        cfg = yaml.safe_load(open(data_yaml_path))
        root = Path(cfg.get("path", Path(data_yaml_path).parent))
        if not root.is_absolute():
            root = (Path(data_yaml_path).parent / root).resolve()
        split = cfg.get("test") or cfg.get("val")
        if not split:
            return None
        return (root / split).resolve() if not Path(split).is_absolute() else Path(split)
    except Exception as e:
        print(f"   ⚠️ Cannot resolve test images: {e}")
        return None


def backfill_one(run_dir: Path) -> bool:
    """Run SAM2 on existing predictions, merge sam2_* into results.csv. Idempotent."""
    pred_json = run_dir / "predictions_test.json"
    gt_json = run_dir / "_bench_gt_coco.json"
    results_csv = run_dir / "results.csv"
    manifest_json = run_dir / "bench_manifest.json"

    if not pred_json.is_file():
        print(f"   ⚠️ No predictions_test.json — skipping")
        return False
    if not gt_json.is_file():
        print(f"   ⚠️ No _bench_gt_coco.json — skipping")
        return False
    if not results_csv.is_file():
        print(f"   ⚠️ No results.csv — skipping")
        return False

    # Check if SAM2 already run (idempotent)
    with open(results_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        existing_keys = set(reader.fieldnames or [])
    already_done = any(k.startswith("sam2_") for k in existing_keys)
    if already_done:
        # Check if values are non-zero
        last = rows[-1] if rows else {}
        if float(last.get("sam2_mAP50_95", 0) or 0) > 0:
            print(f"   ✅ SAM2 already done (sam2_mAP50_95={last['sam2_mAP50_95']})")
            return True
        print(f"   🔄 SAM2 columns present but zero — re-running")

    # Resolve data dir from manifest
    data_dir = None
    if manifest_json.is_file():
        try:
            with open(manifest_json) as fh:
                manifest = json.load(fh)
            data_dir = manifest.get("data") or manifest.get("data_dir")
        except Exception:
            pass

    images_dir = None
    if data_dir:
        data_yaml = Path(data_dir) / "data.yaml"
        if data_yaml.is_file():
            images_dir = resolve_test_images_dir(str(data_yaml))
    if images_dir is None or not images_dir.is_dir():
        print(f"   ⚠️ Cannot resolve test images dir — skipping SAM2")
        return False

    print(f"   🧪 Running SAM2 on {pred_json} …")
    sam_out = run_dir / "_bench_sam2_seg.json"
    sam_metrics = _be.sam2_seg_from_boxes(
        str(images_dir), str(pred_json),
        SAM2_CKPT, SAM2_CONFIG,
        str(gt_json), str(sam_out))

    if not sam_metrics or all(v == 0.0 for v in sam_metrics.values()):
        print(f"   ⚠️ SAM2 returned all zeros — check logs above")
        return False

    # Merge into results.csv
    last_row = rows[-1].copy() if rows else {}
    for k, v in sam_metrics.items():
        last_row[k] = v

    # Preserve original field order, append sam2_* keys
    from bench_schema import SAM2_METRIC_KEYS
    new_fieldnames = list(reader.fieldnames) if reader.fieldnames else list(last_row.keys())
    for k in SAM2_METRIC_KEYS:
        if k not in new_fieldnames:
            new_fieldnames.append(k)

    with open(results_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=new_fieldnames)
        w.writeheader()
        w.writerow(last_row)

    print(f"   ✅ SAM2 backfilled: sam2_mAP50_95={last_row.get('sam2_mAP50_95', 0):.4f}")
    return True


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else None

    if targets:
        run_dirs = [RUNS_ROOT / t for t in targets]
    else:
        # Auto-discover all RF-DETR run dirs with predictions
        run_dirs = sorted(
            d for d in RUNS_ROOT.iterdir()
            if d.is_dir() and "rf-detr" in d.name and (d / "predictions_test.json").is_file()
        )

    if not run_dirs:
        print("No RF-DETR run dirs found with predictions_test.json")
        sys.exit(1)

    print(f"Backfilling SAM2 for {len(run_dirs)} RF-DETR run(s):")
    for rd in run_dirs:
        print(f"  {rd.name}")

    ok = 0
    for rd in run_dirs:
        print(f"\n📁 {rd.name}")
        try:
            if backfill_one(rd):
                ok += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print(f"\nDone: {ok}/{len(run_dirs)} succeeded")

    # Hint: re-harvest
    if ok > 0:
        print("\nTo update benchmark_results.csv, re-run the orchestrator harvest:")
        print(f"  python3 bench_any.py --data <dataset> --models rf-detr-seg-large.pt rf-detr-seg-xlarge.pt --epochs 0 --imgsz 640 --batch_sz 2 --batch_effective 32 --seg_head sam2 --report benchmark_results.csv --aug_policy fair --inference_only")


if __name__ == "__main__":
    main()
