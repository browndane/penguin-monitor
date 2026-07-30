# Benchmark orchestrator: routes each --models entry to the matching *_any_tune.py
# subprocess and aggregates their results.csv into one CSV + compact summary.
#
# USAGE:
#   nohup python bench_any.py --data /path/to/yolo_dataset \
#       --models yolo11x-seg.pt rtdetr-x.pt rfdetr-seg-medium \
#                dinov3-m2f-smallplus eomt-large dinov3-semseg-vits16plus \
#       --epochs 50 --imgsz 640 --batch_sz 4 > ~/bench.out 2>&1 & disown
# OR auto batch:
# nohup python bench_any.py --data /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/RoboflowDatasets/SEGMENT/penguin-seg-combined/ --models ~/yolo_general/models/yolo11x-seg.pt ~/bench_any/detr_weights/rtdetr-x.pt ~/bench_any/detr_weights/rf-detr-seg-large.pt dinov3-m2f-smallplus eomt-large dinov3-semseg-vits16plus --epochs 150 --imgsz 640 --batch_sz 0.8 --batch_effective 32 > ~/bench2.out 2>&1 & disown


import os, sys, csv, argparse, subprocess, hashlib, json, time
from pathlib import Path

from bench_schema import (
    ALL_METRIC_KEYS, PERF_KEYS, RUN_BATCH_KEYS, SUPPORT_KEYS, zero_metrics,
    DIFFICULTY_METRIC_KEYS,
)
from bench_model_registry import (
    expand_model_sets,
    model_set_lines,
    pick_framework,
    registry_lines,
    resolve_local_model,
)
from bench_aug import ultralytics_aug_args, write_aug_grid

HOME_BENCH = Path.home() / "bench_any"
DEFAULT_DINOV3_WEIGHTS_DIR = HOME_BENCH / "dinov3_weights"
RESERVED_MODEL_TOKENS = {
    "yolo", "rtdetr", "rtdetr_external", "rfdetr",
    "dinov3_m2f", "dinov3_semseg", "eomt",
}
# Difficulty-stratified columns (Step 2) are analytical — derived post-hoc from
# existing predictions, not scored model metrics — so exclude them from the
# all-zero/metric_status check (a model with valid box/seg mAP but zero
# difficulty columns must NOT read as "all_zero").
SCORE_KEYS = set(ALL_METRIC_KEYS) - set(PERF_KEYS + SUPPORT_KEYS + DIFFICULTY_METRIC_KEYS)
SUMMARY_COL_W = 11
SUMMARY_PERF_COLS = [("params_M", "params_M"), ("fps_eval", "fps_eval")]
SUMMARY_SPECS = {
    "semantic": [
        ("fg_mIoU", "fg_mIoU"),
        ("fg_mAcc", "fg_mAcc"),
        ("fg_pix_acc", "fg_pixel_acc"),
        ("fg_bIoU", "fg_boundary_IoU"),
        ("mIoU", "mIoU"),
        ("mAcc", "mAcc"),
        ("fwIoU", "fwIoU"),
        ("pixel_acc", "pixel_acc"),
        ("boundary_IoU", "boundary_IoU"),
    ] + SUMMARY_PERF_COLS,
    "instance-seg": [
        ("seg_mAP50_95", "seg_mAP50_95"),
        ("seg_mAP50", "seg_mAP50"),
        ("seg_mAP75", "seg_mAP75"),
        ("seg_mAP_S", "seg_mAP_S"),
        ("seg_mAP_M", "seg_mAP_M"),
        ("seg_mAP_L", "seg_mAP_L"),
        ("seg_AR_100", "seg_AR_100"),
        ("box_mAP50_95", "box_mAP50_95"),
        ("box_mAP50", "box_mAP50"),
        ("box_mAP75", "box_mAP75"),
    ] + SUMMARY_PERF_COLS,
    "detect": [
        ("box_mAP50_95", "box_mAP50_95"),
        ("box_mAP50", "box_mAP50"),
        ("box_mAP75", "box_mAP75"),
        ("box_mAP_S", "box_mAP_S"),
        ("box_mAP_M", "box_mAP_M"),
        ("box_mAP_L", "box_mAP_L"),
        ("box_AR_100", "box_AR_100"),
        ("seg_mAP50_95", "seg_mAP50_95"),
        ("seg_mAP50", "seg_mAP50"),
        ("seg_mAP75", "seg_mAP75"),
    ] + SUMMARY_PERF_COLS,
}

def build_config_tag(args, batch_sz):
    """Compact human-readable fingerprint of the hyperparameters that change a
    run's results. Folding this into the run-dir name keeps a hyperparameter
    sweep (different batch / lr / epochs / imgsz / aug) in *separate* run dirs so
    re-running never silently overwrites a prior run's weights, graphs, and
    results.csv (and resume only ever matches a truly identical config)."""
    parts = [f"b{batch_sz:g}", f"e{args.epochs}", f"i{args.imgsz}"]
    if getattr(args, "batch_effective", 0) and args.batch_effective > 0:
        parts.append(f"be{args.batch_effective}")
    if getattr(args, "lr", 0) and args.lr > 0:
        parts.append(f"lr{args.lr:g}")
    if getattr(args, "aug_policy", "native") and args.aug_policy != "native":
        parts.append(args.aug_policy)
    if getattr(args, "aug_grid", False):
        parts.append("grid")
    if getattr(args, "tune", False):
        parts.append("tune")
    user_tag = getattr(args, "run_tag", "") or ""
    if user_tag.strip():
        parts.append(user_tag.strip().replace(" ", "-"))
    return "_".join(parts)


def build_run_name(data_dir, model_name, cfg_tag=None):
    """Return <dataset>_<model_stem>[_<cfg_tag>] for the run directory.
    Dataset name is capped at 15 chars for readability (full name is still
    in bench_manifest.json). A child wrapper launched by the orchestrator
    inherits the exact run name via the BENCH_RUN_NAME env var, so parent
    and child always agree on the dir."""
    override = os.environ.get("BENCH_RUN_NAME")
    if override:
        return override
    ds_name = Path(data_dir).resolve().name[:15]
    raw = f"{ds_name}_{Path(model_name).stem}"
    if cfg_tag:
        raw = f"{raw}_{cfg_tag}"
    return raw


_CKPT_GLOBS = [
    "checkpoint_*.pth", "*.ckpt", "**/*.ckpt", "**/checkpoint_*.pth",
    "weights/last.pt", "weights/best.pt", "weights/last.pth",
    "weights/best.pth", "**/*.safetensors",
]


def _dir_has_checkpoints(d: Path) -> bool:
    """Return True if *d* contains at least one checkpoint-like file."""
    return any(p for g in _CKPT_GLOBS for p in d.glob(g))


def find_trained_run_dir(task, data_dir, model_entry):
    """When --inference_only, scan for an existing trained run dir whose
    dataset + model stem match, sorted by checkpoint count (newest-ckpt
    mtime breaks ties). Matches by dir-name prefix — no manifest parsing.

    Returns the resolved run_name, or None if no trained dir is found."""
    runs_root = HOME_BENCH / "runs" / task
    if not runs_root.is_dir():
        return None
    prefix = f"{Path(data_dir).resolve().name[:15]}_{Path(model_entry).stem}_"
    best = None
    best_score = (-1, -1)
    for d in runs_root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        ckpts = [p for g in _CKPT_GLOBS for p in d.glob(g)]
        n = len(ckpts)
        if n == 0:
            continue
        newest = max(p.stat().st_mtime for p in ckpts)
        score = (n, newest)
        if score > best_score:
            best_score = score
            best = d.name
    return best


def harvest_results_csv(task, run_name):
    # Every sub-tuner persists results under ~/bench_any/runs/<task>/<run_name>/.
    p = HOME_BENCH / "runs" / task / run_name / "results.csv"
    if p.is_file():
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            last = rows[-1]
            out = {k: float(last.get(k, 0.0) or 0.0) for k in ALL_METRIC_KEYS}
            # Optional per-model batch bookkeeping the wrapper may report back.
            # Only honour non-empty, numeric values so a blank column cannot
            # silently override the orchestrator-resolved batch.
            for k in RUN_BATCH_KEYS:
                raw = last.get(k)
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    out[k] = float(raw)
                except (TypeError, ValueError):
                    continue
            return out
    return None


def count_dataset_support(data_dir, split_order=None):
    """Count images and instance-annotations in the evaluation split.
    Counts one annotation per non-empty line of each YOLO label .txt file.
    Returns (split_key, n_images, n_instances); zeros on any failure so callers
    never crash on exotic dataset layouts."""
    try:
        import yaml as _yaml
        with open(Path(data_dir) / "data.yaml") as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception:
        return "", 0, 0
    root = Path(cfg.get("path", data_dir)).expanduser()
    if not root.is_absolute():
        root = (Path(data_dir) / root).resolve()
    split_key = ""
    split_rel = None
    for key in (split_order or ("test", "val", "valid")):
        if cfg.get(key):
            split_key, split_rel = key, cfg.get(key)
            break
    if not split_rel:
        return "", 0, 0
    img_dir = (root / split_rel).resolve() if not Path(split_rel).is_absolute() else Path(split_rel)
    if not img_dir.is_dir():
        # Some data.yamls point at a .txt list file instead of a directory.
        if img_dir.is_file():
            try:
                img_paths = [Path(l.strip()) for l in img_dir.read_text().splitlines() if l.strip()]
            except Exception:
                return split_key, 0, 0
        else:
            return split_key, 0, 0
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    n_images = len(img_paths)
    n_instances = 0
    for ip in img_paths:
        # YOLO convention: replace .../images/... with .../labels/... and .ext with .txt.
        parts = list(ip.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            lbl = Path(*parts).with_suffix(".txt")
        else:
            lbl = ip.with_suffix(".txt")
        if lbl.is_file():
            try:
                with open(lbl) as fh:
                    n_instances += sum(1 for ln in fh if ln.strip())
            except Exception:
                pass
    return split_key, n_images, n_instances


def eval_split_order(framework):
    # Bench contract: use the same held-out split for every framework.
    # Wrappers whose upstream code only has a validation loader map YOLO test
    # into that loader internally when test exists.
    return ("test", "val", "valid")


def parse_batch_overrides(specs, parser):
    overrides = []
    for group in specs or []:
        for spec in group.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if "=" not in spec:
                parser.error("--batch_sz_override entries must be TOKEN=VALUE")
            token, value = (s.strip() for s in spec.split("=", 1))
            if not token:
                parser.error("--batch_sz_override TOKEN cannot be empty")
            try:
                value = float(value)
            except ValueError:
                parser.error(f"--batch_sz_override value must be numeric: {spec}")
            if value < 0:
                parser.error(f"--batch_sz_override value must be >= 0: {spec}")
            overrides.append((token.lower(), value))
    return overrides


def resolve_batch_sz(batch_values, models, index, entry, framework, overrides):
    if len(batch_values) == 1:
        batch_sz = batch_values[0]
        source = "global"
    else:
        batch_sz = batch_values[index]
        source = "per-model"

    entry_l = entry.lower()
    stem_l = Path(entry).stem.lower()
    for token, value in overrides:
        if token in ("*", "all") or token == framework or token in entry_l or token in stem_l:
            batch_sz = value
            source = f"override:{token}"
    return batch_sz, source


def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _git_snapshot(script_dir):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(script_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(
            ["git", "-C", str(script_dir), "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": None}


def _read_report_header(report_path):
    try:
        if not report_path.exists() or report_path.stat().st_size == 0:
            return []
        with open(report_path, newline="") as fh:
            return next(csv.reader(fh), [])
    except Exception:
        return []


def _metric_status(metrics, rc):
    if metrics is None:
        return "missing_results"
    if not any(float(metrics.get(k, 0.0) or 0.0) > 0.0 for k in SCORE_KEYS):
        return "all_zero"
    return "ok" if rc == 0 else f"partial_rc_{rc}"


def _difficulty_iou_gating(framework: str, task: str):
    """Which iou_types difficulty-stratify should run for a framework/task.

    Returns ``(do_box, do_seg)``. Pure detectors emit bbox-only predictions → box
    only. Semantic-seg (dinov3_semseg) produces no instance boxes → neither (skip
    entirely — its pred glob is empty anyway). Native mask-head seg models emit
    both → both. The dinov3_semseg check comes FIRST because its task is
    "segment" (so the segment branch would otherwise wrongly claim seg-capable).
    """
    if framework == "dinov3_semseg":
        return (False, False)  # no instance boxes → skip entirely
    if task == "detect":       # rtdetr-3/rtdetr-4 (external), detector rf-detr
        return (True, False)   # bbox-only predictions
    if task == "segment":      # yolo-seg, rf-detr-seg, eomt, dinov3-m2f
        return (True, True)
    # Unknown task: default to box-only (bbox predictions are universal).
    return (True, False)


def _redacted_args(args):
    out = dict(vars(args))
    for key in ("hub_token",):
        if out.get(key):
            out[key] = "<redacted>"
    return out


def _write_manifest(path, payload):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[bench] ⚠️ could not write manifest {path}: {e}")


def _dinov3_dir_kind(path_like):
    """Classify a --dinov3_dir value without assuming one wrapper's semantics."""
    if not path_like:
        return "", None
    p = Path(path_like).expanduser()
    if not p.exists():
        return "missing", p
    if (p / "hubconf.py").is_file() or (p / "dinov3" / "hubconf.py").is_file():
        return "torch_hub", p
    if (p / "config.json").is_file():
        return "hf_model", p
    if p.is_dir() and list(p.rglob("dinov3_*.pth")):
        return "pth_weights_dir", p
    if p.is_dir():
        return "dir", p
    return "file", p


def _format_nonzero_metrics(metrics):
    if metrics is None:
        return ""
    text = "  ".join(
        f"{k}={v:.4f}" for k, v in metrics.items()
        if k in SCORE_KEYS and float(v or 0.0) > 0.0
    )
    return text or "all score metrics = 0.0000"


def _quiet_child_env(env):
    env.setdefault("TQDM_DISABLE", "1")
    # TQDM_DISABLE is ignored by any tqdm bar constructed with an explicit
    # `disable=` kwarg (e.g. the DINOv3-M2F upstream training/eval bars), so it
    # alone left m2f runs writing ~0.7M progress redraws (50 MB logs). mininterval
    # is never passed explicitly, so this cap still lands: one refresh / 30 s.
    env.setdefault("TQDM_MININTERVAL", "30")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("HF_HUB_VERBOSITY", "error")
    env.setdefault("DATASETS_VERBOSITY", "error")


def _clear_stale_outputs(run_dir):
    for stale_name in ("results.csv", "UNSUPPORTED.txt"):
        stale = run_dir / stale_name
        if stale.exists():
            try:
                stale.unlink()
            except Exception:
                pass


def _summary_groups(rows):
    groups = {"semantic": [], "instance-seg": [], "detect": []}
    unsupported = []
    for row in rows:
        if row.get("status") == "unsupported" or row.get("metric_status") == "unsupported":
            unsupported.append(row)
        elif row["task"] == "segment" and row["framework"] == "dinov3_semseg":
            groups["semantic"].append(row)
        elif row["task"] == "segment":
            groups["instance-seg"].append(row)
        else:
            groups["detect"].append(row)
    return groups, unsupported


def _render_value(row, key):
    value = float(row.get(key, 0.0) or 0.0)
    if key in ("params_M", "fps_eval"):
        return f"{'-':>{SUMMARY_COL_W}}" if value <= 0.0 else f"{value:>{SUMMARY_COL_W}.1f}"
    return f"{value:>{SUMMARY_COL_W}.4f}"


def print_summary(rows):
    groups, unsupported_rows = _summary_groups(rows)
    for kind, items in groups.items():
        if not items:
            continue
        specs = SUMMARY_SPECS[kind]
        print(f"\n  [{kind}]")
        head = f"  {'model':<32} {'fw':<12}"
        for header, _ in specs:
            head += f" {header:>{SUMMARY_COL_W}}"
        head += f" {'N_img':>7} {'N_inst':>7} {'status':>8} {'metric_status':>14}"
        print(head)
        for row in items:
            line = f"  {row['model_tag']:<32} {row['framework']:<12}"
            for _, key in specs:
                line += " " + _render_value(row, key)
            line += f" {int(row.get('n_images', 0)):>7d} {int(row.get('n_instances', 0)):>7d}"
            line += f" {str(row.get('status', '')):>8} {str(row.get('metric_status', '')):>14}"
            print(line)

    if unsupported_rows:
        print("\n  [unsupported]")
        for row in unsupported_rows:
            print(f"  {row['model_tag']:<32} {row['framework']:<16} runtime not wired; "
                  f"artifact downloaded, metrics intentionally not scored")


def validate_model_args(args, parser):
    try:
        args.models = list(args.models) + expand_model_sets(args.model_set)
    except KeyError as e:
        parser.error(str(e))
    if not args.data:
        parser.error("--data is required unless --list_models or --list_model_sets is used")
    if not args.models:
        parser.error("provide at least one --models entry or --model_set")
    if len(args.batch_sz) not in (1, len(args.models)):
        parser.error("--batch_sz must contain either one value or exactly one value per --models entry")

    bad_tokens = [m for m in args.models if Path(str(m)).name.lower() in RESERVED_MODEL_TOKENS]
    if bad_tokens:
        parser.error(
            "these are framework/route names, not runnable model entries: "
            + ", ".join(bad_tokens)
            + ". Run --list_models and use catalog entries such as rtdetr-2.pt, "
              "rtdetr-3.pt, rtdetr-4.pt, rf-detr-seg-large.pt, or dinov3-semseg-vitb16."
        )


def prepare_model_specs(models, parser):
    specs = []
    for entry in models:
        resolved_entry, resolve_source = resolve_local_model(entry)
        route_entry = resolved_entry if resolve_source in ("path", "missing_path") else entry
        framework, _, script, task_for, project_for = pick_framework(route_entry)
        if not script:
            msg = f"{entry} matched registry route '{framework}', but no runnable harness is registered yet"
            if resolved_entry != entry:
                msg += f" (resolved local weight: {resolved_entry})"
            parser.error(msg)
        specs.append({
            "entry": entry,
            "resolved_entry": resolved_entry,
            "resolve_source": resolve_source,
            "route_entry": route_entry,
            "framework": framework,
            "script": script,
            "task_for": task_for,
            "project_for": project_for,
        })
    return specs


def main():
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    ap = argparse.ArgumentParser(description="Plug-in benchmark orchestrator")
    ap.add_argument("--data", help="YOLO dataset dir (data.yaml inside)")
    ap.add_argument("--models", nargs='*', default=[])
    ap.add_argument("--model_set", action="append", default=[],
                    help="Append a named registry model set. Use --list_model_sets to view names.")
    ap.add_argument("--list_models", action="store_true",
                    help="Print declarative model registry routes and exit.")
    ap.add_argument("--list_model_sets", action="store_true",
                    help="Print named model sets and exit.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch_sz", type=float, nargs="+", default=[0.8],
                    help="Micro-batch size per GPU. One value applies to every model; alternatively pass one "
                         "value per --models entry. If <1 (e.g. 0.8), frameworks that support fractional "
                         "auto-batch may auto-size it to fit VRAM. If >=1, it is used as the literal "
                         "micro-batch.")
    ap.add_argument("--batch_sz_override", action="append", default=[],
                    help="Per-framework/model micro-batch override. Format TOKEN=VALUE; may be repeated or "
                         "comma-separated. TOKEN matches a framework name (yolo, rtdetr, rfdetr, eomt, "
                         "dinov3_m2f, dinov3_semseg) or any substring of the model path/stem. Last match wins.")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.0,
                    help="Override learning rate forwarded to every tuner's --lr. "
                         "0 = leave each tuner's own default (recommended). "
                         "Useful when a tiny micro-batch (e.g. batch_sz=2) makes the "
                         "framework default LR too aggressive (YOLO/EoMT divergence).")
    ap.add_argument("--grad_accum_steps", type=int, default=0,
                    help="Forwarded to RF-DETR and EoMT tuners (both accept it; yolo/rtdetr/dinov3 don't).")
    ap.add_argument("--batch_effective", type=int, default=8,
                    help="Target effective batch size (gradient-noise floor). Forwarded as the\n"
                         "native knob per framework: --nbs for yolo/rtdetr (Ultralytics auto-accumulates),\n"
                         "--batch_effective for rfdetr/eomt/dinov3_m2f (each computes its own grad_accum).\n"
                         "dinov3_semseg has no upstream accumulation support and will run at micro-bs.\n"
                         "Set 0 to disable (let each tuner use its own default).")
    ap.add_argument("--eval_every", type=int, default=10,
                    help="Run full eval every K epochs in non-Ultralytics tuners that support it "
                         "(currently rtdetr_external; will fan out to rfdetr/eomt/dinov3 in phase 2). "
                         "Ultralytics tuners ignore this. Final epoch is always evaluated.")
    ap.add_argument("--no_compile_eomt", action='store_true',
                    help="Pass --no_compile to the EoMT tuner (disables torch.compile/inductor; "
                         "saves ~1-2GB on 16GB GPUs at ~10-20%% throughput cost).")
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="native = each framework default; off/fair = normalized Ultralytics YOLO/RT-DETR knobs.")
    ap.add_argument("--tune_aug", action="store_true",
                    help="Include the same small augmentation search space in YOLO/Ultralytics RT-DETR tuning.")
    ap.add_argument("--aug_grid", action="store_true",
                    help="Write a quick original-vs-policy augmentation audit grid under ~/bench_any/aug_audit.")
    ap.add_argument("--report", default="benchmark_results.csv")
    ap.add_argument("--run_tag", default="",
                    help="Optional extra label appended to every run-dir name (after the auto "
                         "hyperparameter fingerprint b<batch>_e<epochs>_i<imgsz>...). Use it to keep "
                         "otherwise-identical configs in separate run dirs (e.g. --run_tag seedB) so "
                         "repeat runs never overwrite each other's weights/graphs/results.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--inference_only", action='store_true')
    ap.add_argument("--tune", action='store_true')
    ap.add_argument("--tune_iter", type=int, default=5)
    ap.add_argument("--dinov3_weights", default=os.environ.get("DINOV3_WEIGHTS", ""))
    ap.add_argument("--dinov3_dir",
                    default=os.environ.get("DINOV3_DIR", str(DEFAULT_DINOV3_WEIGHTS_DIR)),
                    help="DINOv3 .pth weights dir for semseg; defaults to ~/bench_any/dinov3_weights.")
    ap.add_argument("--hub_token",      default=os.environ.get("HF_TOKEN", ""))
    # --- v2: cross-task fairness knobs --------------------------------------
    # When a pure detector (RT-DETR, YOLO-det) finishes, optionally prompt a
    # promptable segmenter (SAM2 / SAM3) with the predicted boxes to get masks
    # and report a seg_mAP* column. The result is clearly cross-task in the
    # summary because the mask quality is the promptor's, not the detector's.
    ap.add_argument("--seg_head",
                    choices=["off", "sam2", "sam3"],
                    default="off",
                    help="How to obtain seg_mAP for a detector-only model (RT-DETR / YOLO-det). "
                         "'sam2' = SAM2-prompted (needs --sam2_ckpt + --sam2_config). "
                         "'sam3' = SAM3-prompted via sam3_coco_annotator.py. "
                         "Seg-capable tuners ignore this flag.")
    ap.add_argument("--difficulty_stratify", action="store_true",
                    help="Post-hoc AP-Difficulty eval: split GT into Easy/Medium/Hard tiers and "
                         "report box/seg mAP50_95 per tier + Easy-Hard delta. Reads the run's "
                         "existing _bench_gt_coco.json + predictions*.json; no retraining. "
                         "dinov3-semseg is auto-skipped (semantic seg has no instance boxes); "
                         "pure detectors (RT-DETR) get box_* tiers only; seg models get both.")
    ap.add_argument("--cross_model_plots", action="store_true",
                    help="After the model loop, write paper-ready cross-model comparison figures "
                         "(seg/box mAP bars, native-vs-SAM2 paired bars, difficulty tiers, FPS-vs-mAP "
                         "Pareto, Easy-Hard delta) to <report parent>/cross_model_figures/. Reads the "
                         "just-written report CSV by column name; no-ops on partial runs. Multi-seed "
                         "rows (--run_tag seedN) aggregate to mean ± std.")
    ap.add_argument("--sam2_ckpt",
                    default=os.environ.get("SAM2_CKPT",
                                           "/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt"),
                    help="Path to a SAM2 .pt checkpoint (default: sam2.1_hiera_large.pt).")
    ap.add_argument("--sam2_config",
                    default=os.environ.get("SAM2_CONFIG",
                                           "/home/vorse/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
                    help="SAM2 hydra config (default: sam2.1_hiera_l.yaml).")
    ap.add_argument("--verbose", action='store_true', help="Don't suppress upstream tqdm/HF banners.")
    # Resume is ON by default across all wrappers: every *_any_tune.py accepts --resume
    # and will pick up its last checkpoint (Ultralytics last.pt, HF checkpoint-last,
    # RF-DETR checkpoint.pth, EoMT Lightning ckpt, dinov3_semseg last.pt). Use
    # --no-resume to force fresh runs (and wipe stale results.csv).
    ap.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=True,
                    help="Forward --resume to every sub-tuner so interrupted runs continue from the last "
                         "checkpoint. Use --no-resume to start fresh (clears stale results.csv).")
    args = ap.parse_args()
    if args.list_models:
        print("\n".join(registry_lines()))
        return
    if args.list_model_sets:
        print("\n".join(model_set_lines()))
        return
    validate_model_args(args, ap)
    batch_overrides = parse_batch_overrides(args.batch_sz_override, ap)

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data).resolve()
    if not (data_dir / "data.yaml").exists():
        print(f"❌ {data_dir}/data.yaml not found"); sys.exit(1)

    default_split, default_n_images, default_n_instances = count_dataset_support(data_dir)
    print(f"[bench] eval support: {default_n_images} images / {default_n_instances} instances "
          f"(split={default_split or 'unknown'})")
    # Profile the ground-truth across train/val/test once per run so the log
    # shows the instances-per-image distribution (these datasets are crowded —
    # hundreds of penguins per image — which is why the detection caps were
    # raised). Stats are also written to ~/bench_any/dataset_stats/<name>/.
    # Defensive: never let a profiling hiccup abort the benchmark.
    try:
        import bench_dataset_stats as _bds
        _bds.summarize_dataset(data_dir, print_summary=True)
    except Exception as _e_ds:
        print(f"[bench] ⚠️ dataset stats skipped: {_e_ds}")
    if args.aug_grid or args.aug_policy != "native":
        grid = write_aug_grid(data_dir, args.aug_policy, HOME_BENCH / "aug_audit")
        if grid:
            print(f"[bench] aug audit grid: {grid}")
    if args.aug_policy != "native":
        print("[bench] aug_policy controls YOLO and Ultralytics RT-DETR knobs; "
              "external/non-Ultralytics wrappers keep their native loaders for now.")

    prepared_models = prepare_model_specs(args.models, ap)

    report_path = HOME_BENCH / Path(args.report).name if not Path(args.report).is_absolute() else Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Metrics-only CSV: run bookkeeping (status/return_code/metric_status/
    # manifest_path/run_dir) lives in bench_manifest.json, not here. We still
    # build the full in-memory row (incl. those fields) so print_summary keeps
    # working; DictWriter(extrasaction="ignore") drops them from disk.
    # train_time_s records the wall-clock training+eval time per model so the
    # CSV reports timing beyond fps_eval.
    header = (["model_tag", "framework", "task", "eval_split", "epochs", "imgsz",
               "batch_sz", "batch_effective", "train_time_s"]
              + ALL_METRIC_KEYS)
    existing_header = _read_report_header(report_path)
    write_header = existing_header != header
    if existing_header and write_header:
        print(f"[bench] ⚠️ report header changed; appending a fresh header row to {report_path}")
    fh = open(report_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
    if write_header: writer.writeheader()

    rows = []
    for model_idx, spec in enumerate(prepared_models):
        entry = spec["entry"]
        resolved_entry = spec["resolved_entry"]
        resolve_source = spec["resolve_source"]
        route_entry = spec["route_entry"]
        name = spec["framework"]
        script = spec["script"]
        task_for = spec["task_for"]
        project_for = spec["project_for"]
        task = task_for(route_entry.lower())
        eval_split, n_images, n_instances = count_dataset_support(data_dir, eval_split_order(name))
        batch_sz, batch_source = resolve_batch_sz(
            args.batch_sz, args.models, model_idx, entry, name, batch_overrides)
        # Fold the distinguishing hyperparameters into the run name so a sweep
        # over batch/lr/epochs/imgsz/aug lands in separate run dirs instead of
        # overwriting a previous run's weights/graphs/results.csv.
        # inference-only: find the matching trained run dir (same dataset +
        # model stem, has checkpoints) instead of computing a new config tag
        # that would almost certainly differ (different epochs/batch/aug/etc.)
        # and create a separate, empty dir with no checkpoints to load.
        if args.inference_only:
            # Try exact config-tag match first, then fall back to prefix
            # scan.  Without exact-match preference, find_trained_run_dir
            # picks the dir with most checkpoints when multiple configs
            # exist for the same dataset+model (e.g. b1_e15 vs b0.98_e1).
            exact_cfg = build_config_tag(args, batch_sz)
            exact_name = build_run_name(data_dir, entry, exact_cfg)
            exact_dir = HOME_BENCH / "runs" / task / exact_name
            if exact_dir.is_dir() and _dir_has_checkpoints(exact_dir):
                run_name = exact_name
                print(f"[bench] inference-only: resolved trained run dir {run_name} (exact match)")
            else:
                trained_run = find_trained_run_dir(task, data_dir, entry)
                if trained_run:
                    run_name = trained_run
                    print(f"[bench] inference-only: resolved trained run dir {run_name}")
                else:
                    print(f"[bench] ⚠️  inference-only: no trained run dir found for "
                          f"{Path(data_dir).name}_{Path(entry).stem}_* — will use pretrained weights")
                    run_name = exact_name
        else:
            cfg_tag = build_config_tag(args, batch_sz)
            run_name = build_run_name(data_dir, entry, cfg_tag)
        print(f"\n[bench] === {entry}  (fw={name}  task={task}  batch_sz={batch_sz:g} "
              f"[{batch_source}]  eval_split={eval_split or 'unknown'}  run={run_name}) ===")
        if resolved_entry != entry:
            print(f"[bench] resolved model: {entry} -> {resolved_entry} ({resolve_source})")
        elif resolve_source == "unresolved" and name in ("yolo", "rtdetr", "rfdetr"):
            print(f"[bench] ⚠️ local weights not found for '{entry}'. Passing it through may let the "
                  f"{name} backend download/resolve weights by name.")
        elif resolve_source == "missing_path":
            print(f"[bench] ⚠️ explicit model path does not exist: {entry}")

        cmd = [sys.executable, str(script_dir / script),
               "--data", str(data_dir), "--model", resolved_entry, "--task", task,
               "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
               "--batch_sz", str(batch_sz), "--patience", str(args.patience)]
        if args.inference_only or args.epochs <= 0:
            cmd.append("--inference_only")
        if args.resume:
            cmd.append("--resume")
        if args.lr > 0: cmd += ["--lr", str(args.lr)]
        if args.tune and name in ("yolo", "rtdetr"):
            cmd += ["--tune", "--tune_iter", str(args.tune_iter)]
            if args.tune_aug:
                cmd += ["--tune_aug"]
        if name in ("yolo", "rtdetr"):
            cmd += ultralytics_aug_args(args.aug_policy)
        # Forward the unified --aug_policy flag to EVERY wrapper. Each wrapper
        # ignores it when "native" (its pre-bench behavior is preserved); when
        # "off"/"fair" the shared bench_aug.shared_albumentations_policy() list
        # becomes the sole source of training-time augmentation for that fw.
        cmd += ["--aug_policy", args.aug_policy]
        # Announce the routing decision before launching the subprocess so the
        # one [bench] aug_policy=… -> applied|native|unsupported line appears
        # per model even if the child wrapper crashes early.
        _expected_status = "native" if args.aug_policy == "native" else "applied"
        print(f"[bench] aug_policy={args.aug_policy}: {name} -> {_expected_status}")
        if args.grad_accum_steps > 0 and name in ("rfdetr", "eomt"):
            cmd += ["--grad_accum_steps", str(args.grad_accum_steps)]
        # Standardised effective-batch floor across frameworks. Each translates this
        # to its own native knob (Ultralytics nbs / RF-DETR+EoMT+M2F grad_accum).
        if args.batch_effective > 0:
            if name in ("yolo", "rtdetr", "rtdetr_external"):
                cmd += ["--nbs", str(args.batch_effective)]
            elif name in ("rfdetr", "eomt", "dinov3_m2f"):
                cmd += ["--batch_effective", str(args.batch_effective)]
            elif name == "dinov3_semseg":
                print(f"[bench] ⚠️ {entry}: dinov3_semseg has no grad-accumulation hook; "
                      f"--batch_effective={args.batch_effective} ignored (effective batch = {batch_sz}).")
        # Per-epoch eval cadence (currently only honored by rtdetr_external; other
        # wrappers will pick this up in phase 2).
        if args.eval_every and name == "rtdetr_external":
            cmd += ["--eval_every", str(int(args.eval_every))]
        if args.no_compile_eomt and name == "eomt":
            cmd += ["--no_compile"]
        if name in ("dinov3_semseg", "eomt"):
            if args.dinov3_weights:
                cmd += ["--dinov3_weights", args.dinov3_weights]
            if args.dinov3_dir:
                dino_kind, dino_path = _dinov3_dir_kind(args.dinov3_dir)
                if name == "eomt" and dino_kind not in ("hf_model", "missing"):
                    print(f"[bench] ⚠️ {entry}: not forwarding --dinov3_dir={dino_path} to EoMT "
                          f"because it is {dino_kind}, not an HF model dir with config.json.")
                else:
                    cmd += ["--dinov3_dir", args.dinov3_dir]
        if name in ("dinov3_m2f", "eomt") and args.hub_token:
            cmd += ["--hub_token", args.hub_token]
        # Forward prompted-seg flag to pure detectors only. Seg-capable tuners
        # ignore this (they already have a real mask head). For backward
        # compatibility with existing tuners the chosen head is passed as the
        # legacy `--seg_via_sam2` flag when seg_head == "sam2"; sam3 is opt-in
        # and goes through the new `--seg_head sam3` flag if/when tuners pick
        # it up. Tuners that don't recognise the new flag will exit non-zero,
        # so we only forward what each tuner is known to accept today.
        is_pure_detector = task == "detect" and name in ("rtdetr", "rtdetr_external", "yolo", "rfdetr")
        # YOLO-seg and RF-DETR-seg can additionally report SAM2-prompted masks
        # (sam2_*) alongside their native mask heads (seg_*), so forward the
        # prompted-seg flags even for the segment task. Other native-head seg
        # wrappers stay seg-only.
        sam2_eligible = is_pure_detector or (name in ("yolo", "rfdetr", "dinov3_m2f", "eomt") and task == "segment")
        if args.seg_head != "off" and sam2_eligible:
            if args.seg_head == "sam2":
                cmd += ["--seg_via_sam2"]
                if args.sam2_ckpt:   cmd += ["--sam2_ckpt",   args.sam2_ckpt]
                if args.sam2_config: cmd += ["--sam2_config", args.sam2_config]
            elif args.seg_head == "sam3":
                # Not yet wired in yolo/rtdetr tuners — warn loudly so the user
                # knows the seg_mAP* columns will stay '-' for this run.
                print(f"[bench] ⚠️ {entry}: --seg_head sam3 requested but tuner "
                      f"'{script}' has no sam3 hook yet; seg_mAP* will be blank.")
        if args.extra: cmd += list(args.extra)

        env = os.environ.copy()
        env["WANDB_PROJECT"] = project_for(route_entry.lower())
        env["WANDB_NAME"] = run_name
        # Hand the fully-resolved run name (incl. hyperparameter fingerprint) to
        # the child wrapper so its run dir matches what the orchestrator harvests.
        env["BENCH_RUN_NAME"] = run_name
        env["BENCH_EVAL_DIR"] = str(Path(__file__).resolve().parent)

        # When resuming, preserve results.csv so the child appends to its history
        # instead of restarting metrics from epoch 0; otherwise clear stale outputs.
        stale_run_dir = HOME_BENCH / "runs" / task / run_name
        if not args.resume:
            _clear_stale_outputs(stale_run_dir)

        if not args.verbose:
            _quiet_child_env(env)

        # Ensure the run directory exists before spawn so Ultralytics/wandb
        # subdirs (runs/, wandb/) land inside the run dir instead of cluttering
        # the script root. Non-Ultralytics wrappers that set their own CWD
        # (EoMT, M2F) override this via their launch scripts; this only affects
        # yolo/rtdetr/rtdetr_external wrappers that run in-process.
        child_cwd = stale_run_dir
        child_cwd.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        rc = subprocess.call(cmd, env=env, cwd=str(child_cwd))
        t1 = time.time()
        if rc != 0: print(f"[bench] ⚠️ {entry} subprocess rc={rc}; recording zeros")

        raw_metrics = harvest_results_csv(task, run_name)
        run_dir_path = HOME_BENCH / "runs" / task / run_name
        unsupported_marker = run_dir_path / "UNSUPPORTED.txt"
        unsupported = unsupported_marker.is_file()
        metric_status = "unsupported" if unsupported else _metric_status(raw_metrics, rc)
        if raw_metrics is None:
            m = zero_metrics()
            nonzero = ""
        else:
            m = raw_metrics
            nonzero = _format_nonzero_metrics(m)
        # Inject dataset support (computed once, identical for every row).
        m["n_images"] = float(n_images)
        m["n_instances"] = float(n_instances)

        # Paper figures: harmonize whatever the wrapper produced (Ultralytics
        # results.csv + PNGs, Lightning metrics.csv, RF-DETR TB events, or our
        # own history.csv from rtdetr_external) into <run_dir>/graphs/ with
        # uniform train/val loss, accuracy curve, PR, and (when possible)
        # confusion-matrix vector PDFs. All failures are swallowed so plotting
        # never breaks a benchmark run.
        try:
            import bench_plots as _bp
            # Pick up any coco GT/pred json the wrapper saved so PR/CM plots
            # can be produced without re-running eval.
            gt_candidates = list(run_dir_path.glob("_bench_gt_coco.json")) \
                            + list(run_dir_path.glob("*gt*coco*.json"))
            pred_candidates = list(run_dir_path.glob("predictions*.json")) \
                              + list(run_dir_path.glob("*pred*coco*.json"))
            gt_json = str(gt_candidates[0]) if gt_candidates else None
            pred_json = str(pred_candidates[0]) if pred_candidates else None
            # Class names: read from data.yaml.
            class_names = []
            try:
                import yaml as _yaml
                with open(data_dir / "data.yaml") as _fh:
                    _cfg = _yaml.safe_load(_fh) or {}
                _names = _cfg.get("names") or []
                if isinstance(_names, dict):
                    class_names = [_names[k] for k in sorted(_names.keys(), key=lambda x: int(x))]
                else:
                    class_names = list(_names)
            except Exception:
                class_names = []
            # Try detector CM only when both jsons exist.
            cm = None
            cm_labels = None
            if gt_json and pred_json and class_names:
                try:
                    import bench_eval as _be
                    cm, cm_labels = _be.detector_confusion_matrix(
                        gt_json, pred_json, list(class_names),
                        iou_thr=0.5, conf_thr=0.25,
                    )
                except Exception as _e:
                    print(f"[bench_plots] detector CM skipped: {_e}")
            figs = _bp.harmonize_run(
                run_dir_path, name,
                title=f"{Path(entry).stem} ({data_dir.name})",
                gt_coco_json=gt_json, pred_coco_json=pred_json,
                class_names=cm_labels or class_names or None,
                cm=cm,
            )
            if figs:
                print(f"[bench] 📈 figures for {entry}: {sorted(figs.keys())}")
        except Exception as _e:
            print(f"[bench] ⚠️ harmonize_run failed for {entry}: {_e}")

        # ---- v3: difficulty-stratified post-hoc eval (Step 3) -----------------
        # Self-discovers gt/pred so it does NOT depend on the plots block's
        # gt_json/pred_json (those are unbound if `import bench_plots` threw above).
        # Reads existing run artifacts; no retraining. Gated per-framework via
        # _difficulty_iou_gating (dinov3_semseg skipped — no instance boxes).
        if getattr(args, "difficulty_stratify", False):
            try:
                import bench_eval as _be
                # Self-discover gt/pred COCO JSON for the post-hoc difficulty
                # stratify. Wrappers write these to DIFFERENT places, so a root
                # glob alone misses most of them:
                #   - RF-DETR:  <run>/_bench_gt_coco.json + predictions_test.json   (root)
                #   - RT-DETR-external (v3 repo route): <run>/_rtdetrv2_repo_coco/instances_{split}.json
                #               + <run>/predictions_{split}.json                    (gt in subdir)
                #   - YOLO:     <run>/<split>/_bench_gt_coco.json (split subdir)
                #               + <run>/<split>/_bench_predictions_catfixed.json    (the ID+category-
                #                 fixed file; Ultralytics' raw <split>/predictions.json uses STRING
                #                 image_ids that don't match the gt's int safe_image_id hashes —
                #                 bench_eval.fix_yolo_pred_image_ids rewrites them to _bench_predictions_fixed.json,
                #                 then category-fixed to _bench_predictions_catfixed.json).
                # Rank preference (lowest tuple wins):
                #   0. YOLO ID-fixed pred file (_bench_predictions_catfixed > _fixed) — only these
                #      match the gt's int image_ids; raw predictions.json has str ids → 0 overlap.
                #   1. eval_split in the path OR name (avoid train-split gt/pred winning).
                #   2. shallower path (closer to run root).
                #   3. name (deterministic tie-break).
                _sp = (eval_split or "").lower()

                def _rank(p: Path):
                    name = p.name.lower()
                    path = str(p).lower()
                    # Category-canonicalized preds are the final, most-correct file
                    # for every framework. Two such files exist across wrappers:
                    #   - YOLO: _bench_predictions_catfixed.json  (ID-fixed → cat-fixed)
                    #   - DINOv3-M2F: _bench_predictions_resolved.json (cat 0→1)
                    # `_bench_predictions_fixed.json` is ID-fixed but NOT cat-fixed
                    # for DINOv3-M2F (still category_id=0 vs gt=1) → zero overlap
                    # → silently zeros every tier. Prefer the canonical files first.
                    if "catfixed" in name or name.endswith("_predictions_resolved.json"):
                        fixed = 0
                    elif name.startswith("_bench_predictions_fixed"):
                        fixed = 1
                    else:
                        fixed = 2
                    in_split = bool(_sp) and (_sp in path or name.endswith(f"_{eval_split}.json")
                                              or name == f"predictions_{eval_split}.json"
                                              or name == f"instances_{eval_split}.json")
                    return (fixed, 0 if in_split else 1, len(p.parts), name)

                _gt_cands = sorted(
                    list(run_dir_path.rglob("_bench_gt_coco.json"))
                    + list(run_dir_path.rglob("instances_*.json"))
                    + list(run_dir_path.rglob("*gt*coco*.json")),
                    key=_rank,
                )
                _pred_cands = sorted(
                    list(run_dir_path.rglob("predictions*.json"))
                    + list(run_dir_path.rglob("_bench_predictions_*.json"))
                    + list(run_dir_path.rglob("*pred*coco*.json")),
                    key=_rank,
                )
                # Dedup while preserving rank order.
                _seen = set(); _gts = []
                for _p in _gt_cands:
                    if _p not in _seen: _seen.add(_p); _gts.append(_p)
                _seen = set(); _preds = []
                for _p in _pred_cands:
                    if _p not in _seen: _seen.add(_p); _preds.append(_p)
                _gt_json = str(_gts[0]) if _gts else None
                # Pred selection with category-overlap validation: a ranked-best
                # pred whose category_ids don't overlap the GT's (BUG 11 corollary:
                # DINOv3-M2F _fixed has cat 0, gt cat 1) silently zeros every
                # tier. Fall through ranked candidates until one overlaps.
                _pred_json = None
                if _gt_json and _preds:
                    try:
                        with open(_gt_json) as _gf:
                            _gt_cats = {c.get("id") for c in json.load(_gf).get("categories", [])}
                    except Exception:
                        _gt_cats = None
                    for _pc in _preds:
                        if _gt_cats is None:
                            _pred_json = str(_pc); break
                        try:
                            with open(_pc) as _pf:
                                _pp = json.load(_pf)
                            _pcats = {p.get("category_id") for p in _pp} if isinstance(_pp, list) else set()
                        except Exception:
                            _pcats = set()
                        if _pcats & _gt_cats:
                            _pred_json = str(_pc); break
                    if _pred_json is None:
                        _pred_json = str(_preds[0])  # last resort: keep prior behavior
                if _gt_json and _pred_json:
                    _do_box, _do_seg = _difficulty_iou_gating(name, task)
                    dlist = _be.compute_difficulty_from_coco_gt(_gt_json)
                    if _do_box:
                        tiers = _be.stratified_coco_eval(_gt_json, _pred_json, dlist, iou_type="bbox")
                        for t in ("Easy", "Medium", "Hard"):
                            m[f"box_mAP50_95_{t}"] = tiers[t]["mAP50_95"]
                        m["diff_delta_box"] = tiers["delta"]["mAP50_95"]  # scalar, not dict
                    if _do_seg:
                        tiers = _be.stratified_coco_eval(_gt_json, _pred_json, dlist, iou_type="segm")
                        for t in ("Easy", "Medium", "Hard"):
                            m[f"seg_mAP50_95_{t}"] = tiers[t]["mAP50_95"]
                        m["diff_delta_seg"] = tiers["delta"]["mAP50_95"]  # scalar, not dict
                    print(f"[bench] 📊 difficulty-stratified {entry}: "
                          f"box={'' if _do_box else 'skip '}seg={'' if _do_seg else 'skip '}"
                          f"(delta box={m.get('diff_delta_box', 0.0):.3f}, "
                          f"seg={m.get('diff_delta_seg', 0.0):.3f})")
                else:
                    print(f"[bench] ⚠️ difficulty-stratify skipped for {entry}: "
                          f"missing gt/pred (gt={bool(_gt_json)}, pred={bool(_pred_json)})")
            except Exception as _e:
                print(f"[bench] ⚠️ difficulty-stratify failed for {entry}: {_e}")

        manifest_path = run_dir_path / "bench_manifest.json"
        run_status = "unsupported" if unsupported else (
            "ok" if rc == 0 and raw_metrics is not None else ("partial" if raw_metrics else "failed"))
        manifest = {
            "model_entry": entry,
            "model_entry_resolved": resolved_entry,
            "model_entry_resolve_source": resolve_source,
            "model_tag": Path(entry).stem,
            "framework": name,
            "task": task,
            "script": script,
            "command": cmd,
            "return_code": int(rc),
            "status": run_status,
            "metric_status": metric_status,
            "duration_sec": round(t1 - t0, 3),
            "eval_split": eval_split,
            "eval_n_images": int(n_images),
            "eval_n_instances": int(n_instances),
            "started_unix": t0,
            "ended_unix": t1,
            "data_dir": str(data_dir),
            "data_yaml_sha256": _sha256_file(data_dir / "data.yaml"),
            "args": _redacted_args(args),
            "python": sys.version,
            "python_executable": sys.executable,
            "bench_git": _git_snapshot(script_dir),
        }
        _write_manifest(manifest_path, manifest)

        run_dir = str(run_dir_path)
        # Prefer the wrapper-reported micro/effective batch when present; otherwise
        # fall back to the orchestrator-resolved values. Pop the harvested keys so
        # they don't double-write through **m below.
        csv_batch_sz = m.pop("batch_sz", batch_sz)
        csv_batch_effective = m.pop("batch_effective", args.batch_effective)
        row = {"model_tag": Path(entry).stem, "framework": name, "task": task,
               "eval_split": eval_split, "epochs": args.epochs, "imgsz": args.imgsz,
               "batch_sz": csv_batch_sz, "batch_effective": csv_batch_effective,
               "train_time_s": round(t1 - t0, 3),
               "status": run_status, "return_code": int(rc),
               "metric_status": metric_status, "manifest_path": str(manifest_path),
               "run_dir": run_dir, **m}
        rows.append(row); writer.writerow(row); fh.flush()
        if unsupported:
            print(f"[bench] {entry}: unsupported runtime; artifact ready, metrics intentionally not scored")
        else:
            print(f"[bench] {entry}: {nonzero or 'no metrics captured'}")

    fh.close()
    print(f"\n[bench] Wrote {len(rows)} rows → {report_path}\n[bench] Summary:")
    print_summary(rows)

    # Step 4: optional cross-model comparison figures over the aggregated report.
    # Reads report_path (just written + closed) by column name; only emitted when
    # the relevant columns/rows are present, so it no-ops cleanly on partial runs.
    if getattr(args, "cross_model_plots", False) and rows:
        try:
            import bench_plots as _bp
            _figs = _bp.plot_cross_model_comparison(str(report_path))
            if _figs:
                print(f"[bench] cross-model figures: {len(_figs)} written")
        except Exception as _e:
            print(f"[bench] ⚠️ cross-model plots skipped: {_e}")


if __name__ == "__main__":
    main()
