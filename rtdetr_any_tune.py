# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# USAGE (Train):       python3 rtdetr_any_tune.py --data /path/to/dataset --epochs 100 --batch_sz 8 --model rtdetr-x.pt
# USAGE (Resume):      python3 rtdetr_any_tune.py --data /path/to/dataset --resume
# USAGE (Tune+Train):  python3 rtdetr_any_tune.py --data /path/to/dataset --tune --tune_iter 10 --epochs 20
#
# Mirrors yolo_any_tune.py exactly; swaps YOLO -> RTDETR. Detection-only (RT-DETR has no
# segmentation head in Ultralytics). For RT-DETRv4 / RT-DETRv2 just point --model at the
# corresponding .pt once the weights load via Ultralytics' RTDETR class.

import numpy as np, os, yaml, logging, argparse, gc, sys, shutil, atexit, tempfile, hashlib
from ultralytics import RTDETR
from pathlib import Path
from bench_schema import zero_metrics

# Ray is only reachable on the --tune path, so it is imported there. A top-level
# import kills every plain training run in a venv without a ray wheel.
def _load_ray():
    global ray, tune, OptunaSearch
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
np.random.seed(42)


class NoPlotLimitWarningFilter(logging.Filter):
    def filter(self, record): return 'Limiting validation plots' not in record.getMessage()
logging.getLogger("ultralytics").addFilter(NoPlotLimitWarningFilter())


def optional_aug_kwargs(args):
    out = {}
    for key in ("mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    return out

TEMP_DIR_PATH = None
def cleanup_temp_dir():
    global TEMP_DIR_PATH
    if TEMP_DIR_PATH and TEMP_DIR_PATH.exists():
        shutil.rmtree(TEMP_DIR_PATH, ignore_errors=True); TEMP_DIR_PATH = None


def normalize_dataset_layout(original_data_yaml, dataset_name):
    global TEMP_DIR_PATH
    original_yaml_path = Path(original_data_yaml).resolve()
    original_data_dir = original_yaml_path.parent
    shm_dir = Path('/dev/shm')
    temp_base = shm_dir if shm_dir.exists() else Path(tempfile.gettempdir())
    TEMP_DIR_PATH = temp_base / f"rtdetr_norm_{dataset_name}_{os.getpid()}"
    TEMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    atexit.register(cleanup_temp_dir)

    with open(original_yaml_path, 'r') as f: original_yaml_data = yaml.safe_load(f)
    new_yaml_data = {'names': original_yaml_data.get('names', []), 'nc': original_yaml_data.get('nc', 0)}

    for split in ['train', 'val', 'test']:
        if split not in original_yaml_data: continue
        source_img_dir = original_data_dir / original_yaml_data[split]
        source_lbl_dir = Path(str(source_img_dir).replace('images', 'labels'))
        if not source_img_dir.is_dir(): continue
        target_img_dir = TEMP_DIR_PATH / split / 'images'
        target_lbl_dir = TEMP_DIR_PATH / split / 'labels'
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_file in source_img_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                os.symlink(img_file.resolve(), target_img_dir / img_file.name)
        if source_lbl_dir.is_dir():
            for lbl_file in source_lbl_dir.glob('*.txt'):
                os.symlink(lbl_file.resolve(), target_lbl_dir / lbl_file.name)
        new_yaml_data[split] = f"{split}/images"

    new_yaml_path = TEMP_DIR_PATH / 'data.yaml'
    with open(new_yaml_path, 'w') as f: yaml.dump(new_yaml_data, f, default_flow_style=False, sort_keys=False)
    return str(TEMP_DIR_PATH), str(new_yaml_path)


def print_detailed_metrics(metrics, class_names):
    box = metrics.box
    print(f"\n--- Overall Metrics ---\nmAP50-95(B): {box.map:.4f}\nmAP50(B): {box.map50:.4f}")
    if hasattr(metrics, 'fitness'): print(f"Fitness: {metrics.fitness:.4f}")
    if box.maps is None: return
    header = f"{'Class':<20} {'N':<8} {'P':<10} {'R':<10} {'mAP50':<10} {'mAP75':<10} {'mAP50-95':<10}"
    print(f"{header}\n{'-' * len(header)}")
    for i, cn in enumerate(class_names):
        n = int(metrics.nt_per_class[i]) if i < len(metrics.nt_per_class) else 0
        p = box.p[i] if i < len(box.p) else 0.0
        r = box.r[i] if i < len(box.r) else 0.0
        m50 = box.ap50[i] if i < len(box.ap50) else 0.0
        m75 = float(box.all_ap[i, 5]) if hasattr(box, 'all_ap') and box.all_ap is not None and i < len(box.all_ap) and box.all_ap.shape[1] > 5 else 0.0
        m = box.maps[i] if i < len(box.maps) else 0.0
        print(f"{cn:<20} {n:<8} {p:<10.4f} {r:<10.4f} {m50:<10.4f} {m75:<10.4f} {m:<10.4f}")


if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    # Args
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to YOLO dataset directory containing data.yaml")
    ap.add_argument("--model", default="rtdetr-x.pt", help="RT-DETR weights / cfg (rtdetr-l.pt, rtdetr-x.pt, rtdetrv4.pt, ...)")
    ap.add_argument("--task", default="detect", choices=["detect"], help="RT-DETR is detection-only")
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument('--freeze', default=0, type=int)
    ap.add_argument("--batch_sz", default=0.8, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--lr", default=5e-5, type=float)
    ap.add_argument("--lrf", default=0.01, type=float)
    ap.add_argument("--opti", default="AdamW")
    ap.add_argument("--nbs", default=64, type=int)
    ap.add_argument("--cos_lr", action='store_true')
    ap.add_argument("--threads", default=8, type=int)
    ap.add_argument("--dropout", default=0.0, type=float)
    ap.add_argument("--weight_decay", default=0.0001, type=float)
    ap.add_argument("--momentum", default=0.9, type=float)
    ap.add_argument("--warmup_momentum", default=0.8, type=float)
    ap.add_argument("--iou", default=0.7, type=float)
    ap.add_argument("--hsv_h", default=0.0, type=float)
    ap.add_argument("--hsv_s", default=0.0, type=float)
    ap.add_argument("--hsv_v", default=0.0, type=float)
    ap.add_argument("--cutmix", default=None, type=float)
    ap.add_argument("--copy_paste", default=None, type=float)
    ap.add_argument("--mosaic", default=None, type=float)
    ap.add_argument("--mixup", default=None, type=float)
    ap.add_argument("--fliplr", default=None, type=float)
    ap.add_argument("--flipud", default=None, type=float)
    ap.add_argument("--erasing", default=None, type=float)
    ap.add_argument('--scale', default=0.2, type=float)
    ap.add_argument('--translate', default=0.1, type=float)
    ap.add_argument('--shear', default=0, type=float)
    ap.add_argument('--perspective', default=0.0, type=float)
    ap.add_argument('--degrees', default=10.0, type=float)
    ap.add_argument("--multi_scale", action='store_true')
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--tune", action='store_true')
    ap.add_argument("--tune_iter", default=5, type=int)
    ap.add_argument("--tune_aug", action='store_true')
    ap.add_argument("--inference_only", action='store_true')
    # --- v2 bench_eval hooks ----------------------------------------------
    # RT-DETR is detection-only; --seg_via_sam2 turns on a SAM2-prompted seg
    # post-pass to fill the seg_mAP* cross-task columns honestly (labeled
    # '(SAM2)' in the bench summary so it's not confused with native masks).
    ap.add_argument("--seg_via_sam2", action='store_true')
    ap.add_argument("--sam2_ckpt",   default=os.environ.get("SAM2_CKPT", ""))
    ap.add_argument("--sam2_config", default=os.environ.get("SAM2_CONFIG", ""))
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, feed the shared cross-framework Albumentations "
                         "pipeline into Ultralytics RT-DETR via the `augmentations=` kwarg.")
    args = ap.parse_args()
    if args.aug_policy in ("off", "fair"):
        from bench_aug import shared_albumentations_policy
        _SHARED_AUG = shared_albumentations_policy(args.aug_policy, args.imgsz, framework="rtdetr")
        print(f"[bench] aug_policy={args.aug_policy}: rtdetr -> applied")
    else:
        _SHARED_AUG = None
        print(f"[bench] aug_policy={args.aug_policy}: rtdetr -> native")

    # Setup
    ORIGINAL_DATA_DIR = Path(args.data).resolve()
    if not ORIGINAL_DATA_DIR.is_dir(): print(f"❌ Data directory not found"); sys.exit(1)
    DATA_NAME = ORIGINAL_DATA_DIR.name
    ORIGINAL_DATA_YAML = ORIGINAL_DATA_DIR / 'data.yaml'
    if not ORIGINAL_DATA_YAML.exists(): print(f"❌ data.yaml not found"); sys.exit(1)

    ACTIVE_DATA_DIR, ACTIVE_DATA_YAML = normalize_dataset_layout(ORIGINAL_DATA_YAML, DATA_NAME)
    ACTIVE_ROOT_DIR = Path(ACTIVE_DATA_DIR).parent
    TASK = args.task
    BATCH_SIZE = int(args.batch_sz) if args.batch_sz >= 1 else args.batch_sz

    MODEL_STEM = Path(args.model).stem
    if MODEL_STEM.endswith(f"_{DATA_NAME}"): MODEL_STEM = MODEL_STEM[:-(len(DATA_NAME) + 1)]
    _env_run = os.environ.get("BENCH_RUN_NAME")
    RUN_NAME = _env_run if _env_run else f"{DATA_NAME[:15]}_{MODEL_STEM}"
    os.environ.setdefault('WANDB_PROJECT', "RTDETR-detect")

    for val_dir_name in ['val', 'test']:
        val_cleanup_dir = Path(ACTIVE_ROOT_DIR) / val_dir_name
        if val_cleanup_dir.exists(): shutil.rmtree(val_cleanup_dir, ignore_errors=True)

    with open(ACTIVE_DATA_YAML) as f: CLASSES = yaml.load(f, Loader=yaml.FullLoader)['names']

    # Tune
    if args.tune:
        _load_ray()
        def train_objective(tune_config):
            m = RTDETR(args.model)
            m.train(
                data=ACTIVE_DATA_YAML, task=TASK, imgsz=args.imgsz, epochs=args.epochs,
                patience=args.patience, batch=BATCH_SIZE, optimizer=args.opti,
                project=os.path.join(ACTIVE_ROOT_DIR, 'ray_tune_runs'),
                name=f"{RUN_NAME}_trial", exist_ok=True, verbose=False, pretrained=True,
                close_mosaic=min(10, args.epochs // 10), iou=args.iou,
                hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
                degrees=tune_config.get("degrees", args.degrees) if args.aug_policy == "native" else 0.0,
                translate=tune_config.get("translate", args.translate) if args.aug_policy == "native" else 0.0,
                shear=args.shear if args.aug_policy == "native" else 0.0,
                perspective=args.perspective if args.aug_policy == "native" else 0.0,
                scale=tune_config.get("scale", args.scale) if args.aug_policy == "native" else 0.0,
                erasing=args.erasing if args.erasing is not None else 0.05,
                lrf=args.lrf, warmup_momentum=args.warmup_momentum, cos_lr=False,
                multi_scale=args.multi_scale, deterministic=False,
                weight_decay=tune_config["weight_decay"], momentum=tune_config["momentum"],
                dropout=tune_config["dropout"], lr0=tune_config["lr0"],
                **({"augmentations": _SHARED_AUG} if _SHARED_AUG is not None else {}),
                **optional_aug_kwargs(args)
            )
            del m; gc.collect()

        search_space = {
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "momentum": tune.uniform(0.85, 0.98),
            "dropout": tune.uniform(0.0, 0.2),
            "lr0": tune.loguniform(5e-5, 5e-3),
        }
        if args.tune_aug and args.aug_policy == "native":
            search_space.update({
                "scale": tune.uniform(0.0, 0.20),
                "translate": tune.uniform(0.0, 0.10),
                "degrees": tune.uniform(0.0, 10.0),
            })
        tune_metric = "metrics/mAP50-95(B)"
        tuner = tune.Tuner(
            tune.with_resources(train_objective, {"gpu": 1}), param_space=search_space,
            tune_config=tune.TuneConfig(metric=tune_metric, mode="max",
                                        search_alg=OptunaSearch(), num_samples=args.tune_iter),
            run_config=ray.train.RunConfig(name=f"tune_{RUN_NAME}",
                                           storage_path=os.path.join(ACTIVE_ROOT_DIR, 'ray_results')),
        )
        results_grid = tuner.fit()
        best = results_grid.get_best_result(metric=tune_metric, mode="max")
        print(f"✅ Best {tune_metric}: {best.metrics[tune_metric]:.4f}")
        args.lr = best.config['lr0']; args.weight_decay = best.config['weight_decay']
        args.momentum = best.config['momentum']; args.dropout = best.config['dropout']
        args.scale = best.config.get("scale", args.scale)
        args.translate = best.config.get("translate", args.translate)
        args.degrees = best.config.get("degrees", args.degrees)
        if ray.is_initialized(): ray.shutdown()

    # Train / inference
    final_model = None; final_proj_name = ""
    if not args.inference_only:
        model_path_to_use = args.model; is_resuming_run = False
        if args.resume and not args.tune:
            last_pt_path = Path.home() / 'bench_any' / 'runs' / TASK / RUN_NAME / 'weights' / 'last.pt'
            if os.path.exists(last_pt_path):
                model_path_to_use = last_pt_path; is_resuming_run = True
            else:
                # Resume is ON by default in bench_any, but a fresh run simply
                # has no last.pt yet. Don't abort (that made rtdetr entries exit
                # instantly with rc=1 so a later model looked like it "ran
                # first"); just start training from the pretrained weights.
                print("ℹ Resume requested but no checkpoint yet; starting a fresh run.")
        model = RTDETR(model_path_to_use)
        # `--threads 0` means auto, not "no workers" (see yolo_any_tune.py). -1
        # forces single-process. cvBench speed port.
        WORKERS = 0 if args.threads < 0 else (args.threads or min(8, os.cpu_count() or 8))
        # Bench: per-epoch mAP75/mAP50_95 sidecar for paper-figure history.csv.
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import bench_plots as _bp
            _bp.install_ultra_map75_callback(model)
        except Exception as _e:
            print(f"[rtdetr] mAP75 callback skipped: {_e}")
        model.train(
            data=ACTIVE_DATA_YAML, project=os.path.join(ACTIVE_ROOT_DIR, 'runs', TASK),
            name=RUN_NAME, exist_ok=True, task=TASK, optimizer=args.opti,
            imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
            batch=BATCH_SIZE, lr0=args.lr, lrf=args.lrf, workers=WORKERS, nbs=args.nbs,
            cos_lr=args.cos_lr, close_mosaic=min(10, max(1, args.epochs // 30)),
            dropout=args.dropout, weight_decay=args.weight_decay, momentum=args.momentum,
            warmup_momentum=args.warmup_momentum, iou=args.iou,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            freeze=args.freeze if args.freeze > 0 else None,
            degrees=args.degrees if args.aug_policy == "native" else 0.0,
            translate=args.translate if args.aug_policy == "native" else 0.0,
            shear=args.shear if args.aug_policy == "native" else 0.0,
            perspective=args.perspective if args.aug_policy == "native" else 0.0,
            scale=args.scale if args.aug_policy == "native" else 0.0,
            erasing=args.erasing if args.erasing is not None else 0.05,
            multi_scale=args.multi_scale, pretrained=not is_resuming_run,
            deterministic=False, show_conf=False, show_boxes=False, show_labels=False,
            **({"augmentations": _SHARED_AUG} if _SHARED_AUG is not None else {}),
            **optional_aug_kwargs(args)
        )
        final_proj_name = model.trainer.save_dir
        final_model = RTDETR(os.path.join(final_proj_name, 'weights', 'best.pt'))
    else:
        # inference-only: load trained checkpoint from the run dir
        weights_dir = Path.home() / 'bench_any' / 'runs' / TASK / RUN_NAME / 'weights'
        ckpt_path = weights_dir / 'best.pt'
        if not ckpt_path.is_file():
            ckpt_path = weights_dir / 'last.pt'
        if ckpt_path.is_file():
            final_model = RTDETR(str(ckpt_path))
            print(f"🔁 inference-only: loading trained checkpoint {ckpt_path}")
        else:
            final_model = RTDETR(args.model)
            print(f"⚠️  inference-only: no checkpoint at {weights_dir}; using pretrained weights")
        final_proj_name = os.path.join(ACTIVE_ROOT_DIR, 'runs', TASK, RUN_NAME)

    # Final val + persist results.csv (read by bench_any.py)
    if final_model and final_proj_name:
        with open(ACTIVE_DATA_YAML) as f: active_yaml_config = yaml.safe_load(f)
        # All persistent outputs go under ~/bench_any/ — never into the project / dataset dir.
        bench_dir = Path.home() / 'bench_any'
        primary_save_dir = bench_dir / 'runs' / TASK / RUN_NAME
        final_metrics = {}
        for split in ('val', 'test'):
            if split not in active_yaml_config: continue
            print(f"\n🧪 Validating final model on '{split}' split...")
            metrics = final_model.val(data=ACTIVE_DATA_YAML, split=split,
                                      project=str(ACTIVE_ROOT_DIR), name=f"{split}_{RUN_NAME}",
                                      save=True, save_json=True, plots=False, deterministic=False,
                                      # Crowded penguin scenes can hold several hundred
                                      # instances; Ultralytics' default max_det=300 caps
                                      # both the native metrics and predictions.json fed
                                      # to COCOeval. Raise to 1000 (matches bench_eval.MAX_DETS).
                                      max_det=1000,
                                      show_conf=False, show_boxes=False, show_labels=False)
            print_detailed_metrics(metrics, CLASSES)
            if split == 'test':
                b = metrics.box
                m75 = float(np.nanmean(b.all_ap[:, 5])) if getattr(b, 'all_ap', None) is not None and b.all_ap.shape[1] > 5 else 0.0
                f1 = (2*b.mp*b.mr/(b.mp+b.mr)) if (b.mp + b.mr) > 0 else 0.0
                # Ensure we initialize with ALL keys from the bench_any schema so that
                # harvest_results_csv doesn't report 0.0 for missing keys that should be hyphenated.
                final_metrics = zero_metrics()
                final_metrics.update(dict(box_P=float(b.mp), box_R=float(b.mr), box_F1=float(f1),
                                     box_mAP50=float(b.map50), box_mAP75=m75, box_mAP50_95=float(b.map)))
                # ---- v2: bench_eval post-pass (S/M/L + AR_100 + optional SAM2) ----
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    import bench_eval as _be
                    # Ultralytics may append -N suffix if dir exists; find the actual dir.
                    val_dir_candidates = sorted(Path(ACTIVE_ROOT_DIR).glob(f"{split}_{RUN_NAME}*"),
                                                key=lambda p: p.stat().st_mtime, reverse=True)
                    val_dir = val_dir_candidates[0] if val_dir_candidates else Path(ACTIVE_ROOT_DIR) / f"{split}_{RUN_NAME}"
                    pred_jsons = sorted(val_dir.glob("predictions*.json"))
                    if pred_jsons:
                        pred_json = pred_jsons[0]
                        gt_json = val_dir / "_bench_gt_coco.json"
                        _be.yolo_split_to_coco_gt(ACTIVE_DATA_YAML, split, str(gt_json),
                                                  one_indexed_cats=False)
                        # Fix image_id mismatch between YOLO predictions and GT
                        fixed_pred = val_dir / "_bench_predictions_fixed.json"
                        n_fixed = _be.fix_yolo_pred_image_ids(str(pred_json), str(gt_json), str(fixed_pred))
                        if n_fixed > 0:
                            print(f"   🔧 Fixed {n_fixed} prediction image_ids to match GT")
                            pred_json = fixed_pred
                        # Auto-correct category_id offset (Ultralytics often emits
                        # 1-indexed cat ids while our GT is 0-indexed). Critical
                        # for SAM2 path too: SAM2 copies category_id verbatim.
                        cat_fixed = val_dir / "_bench_predictions_catfixed.json"
                        n_cf, cf_mode = _be.remap_pred_categories(
                            str(pred_json), str(gt_json), str(cat_fixed))
                        if n_cf > 0:
                            print(f"   🔧 Remapped {n_cf} prediction category_ids "
                                  f"(mode={cf_mode}) to match GT")
                            pred_json = cat_fixed
                        extra = _be.coco_eval_full(str(gt_json), str(pred_json), None)
                        # Use the shared COCOeval pass as the canonical bench
                        # metric so Ultralytics RT-DETR is comparable with the
                        # external and DINO/RF-DETR wrappers.
                        for k, v in extra.items():
                            if k in _be.BOX_AP_KEYS or k in _be.SEG_AP_KEYS:
                                final_metrics[k] = v
                        if args.seg_via_sam2 and args.sam2_ckpt and args.sam2_config:
                            try:
                                cfg_yaml = yaml.safe_load(open(ACTIVE_DATA_YAML))
                                _root = Path(cfg_yaml.get("path", Path(ACTIVE_DATA_YAML).parent))
                                _split = cfg_yaml.get("test") or cfg_yaml.get("val")
                                _imgs = (_root / _split).resolve() if not Path(_split).is_absolute() else Path(_split)
                                sam_out = val_dir / "_bench_sam2_seg.json"
                                sam_metrics = _be.sam2_seg_from_boxes(
                                    str(_imgs), str(pred_json),
                                    args.sam2_ckpt, args.sam2_config,
                                    str(gt_json), str(sam_out))
                                for k, v in sam_metrics.items():
                                    final_metrics[k] = v
                            except Exception as _e:
                                print(f"⚠️ SAM2 seg post-pass failed: {_e}")
                    else:
                        print(f"⚠️ no predictions.json under {val_dir}; S/M/L will be 0.")
                except Exception as _e:
                    print(f"⚠️ bench_eval post-pass failed: {_e}")
                # ---- v2: params + eval FPS -------------------------------
                try:
                    import bench_eval as _be2, torch as _t
                    final_metrics["params_M"] = _be2.model_params_M(final_model)
                    inner = getattr(final_model, "model", final_model)
                    if hasattr(inner, "eval") and _t.cuda.is_available():
                        inner = inner.cuda().eval()
                        x = _t.zeros(1, 3, args.imgsz, args.imgsz,
                                     device="cuda", dtype=next(inner.parameters()).dtype)
                        with _t.no_grad():
                            final_metrics["fps_eval"] = _be2.time_fps(lambda: inner(x))
                except Exception as _e:
                    print(f"⚠️ params/fps probe failed: {_e}")

        # Copy results out + write results.csv — single target under ~/bench_any/.
        try:
            import csv
            primary_save_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(final_proj_name, primary_save_dir, dirs_exist_ok=True)
            for split in ('val', 'test'):
                tdir = Path(ACTIVE_ROOT_DIR) / f"{split}_{RUN_NAME}"
                if tdir.exists(): shutil.copytree(tdir, primary_save_dir / split, dirs_exist_ok=True)
            if final_metrics:
                # Report the actual micro-batch + realised effective batch so the
                # bench CSV reflects the real per-model batch (fractional auto-batch
                # is resolved inside Ultralytics, so leave batch_sz to fallback).
                if BATCH_SIZE >= 1:
                    final_metrics["batch_sz"] = int(BATCH_SIZE)
                    final_metrics["batch_effective"] = max(int(args.nbs), int(BATCH_SIZE))
                else:
                    final_metrics["batch_effective"] = int(args.nbs)
                with open(primary_save_dir / 'results.csv', 'w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(final_metrics.keys())); w.writeheader(); w.writerow(final_metrics)
            print(f"✅ Results copied to: {primary_save_dir}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    print("🎉 Complete!")
