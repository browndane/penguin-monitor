# USAGE (Train): python3 yolo_any_tune.py --data /path/to/dataset --epochs 100 --batch_sz 8 --model yolov8l-seg.pt
# USAGE (Resume): python3 yolo_any_tune.py --data /path/to/dataset --resume
# USAGE (Tune & Train): python3 yolo_any_tune.py --data /path/to/dataset --tune --tune_iter 10 --epochs 20
# USAGE (Advanced): python3 yolo_any_tune.py --data /mnt/LinuxBackup/FoundationDatasets/oiv7-yolo-allfindrefine-penguin --patience 20 --batch_sz 8 --tune --tune_iter 1 --epochs 11 --model /home/vorse/yolo_general/models/penguin/yolo11x-seg_penguin.pt --nbs 8

import numpy as np, os, yaml, logging, argparse, gc, sys, shutil, atexit, tempfile, cv2, random, hashlib
import albumentations as A
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from bench_schema import zero_metrics

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["OMP_NUM_THREADS"] = "1"

np.random.seed(42)


class NoPlotLimitWarningFilter(logging.Filter):
    def filter(self, record): return 'Limiting validation plots' not in record.getMessage()

logging.getLogger("ultralytics").addFilter(NoPlotLimitWarningFilter())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TEMP_DIR_PATH = None

def cleanup_temp_dir():
    global TEMP_DIR_PATH
    if TEMP_DIR_PATH and TEMP_DIR_PATH.exists():
        shutil.rmtree(TEMP_DIR_PATH, ignore_errors=True)
        TEMP_DIR_PATH = None


def plot_ground_truth_standalone(data_yaml_path, max_plots=20):
    """Plot ground truth with organized folder structure"""
    print("🔍 Plotting ground truth annotations...")

    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    class_names = config.get('names', [])
    dataset_root = Path(data_yaml_path).parent
    bench_dir = Path.home() / 'bench_any'

    # Create organized output directory
    output_base = bench_dir / "gt_plots" / dataset_root.name

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

    for split in ['train', 'val', 'test']:
        if split not in config: continue

        split_path = dataset_root / config[split]
        labels_path = Path(str(split_path).replace('images', 'labels'))
        if not split_path.exists() or not labels_path.exists(): continue

        # Create split subfolder
        split_output_dir = output_base / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Get images and shuffle
        imgs = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png')) + list(split_path.glob('*.jpeg'))
        random.shuffle(imgs)

        for i, img_path in enumerate(imgs[:max_plots//3]):
            img = cv2.imread(str(img_path))
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            label_path = labels_path / f"{img_path.stem}.txt"
            if not label_path.exists(): continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img_rgb)
            ax.set_title(f"GT {split}: {img_path.name}")

            bbox_count = 0
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue

                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    if class_id >= len(class_names): continue

                    color = colors[class_id % len(colors)]

                    if len(coords) == 4:  # Bounding box
                        x_c, y_c, w_n, h_n = coords
                        x1, y1 = (x_c - w_n/2)*w, (y_c - h_n/2)*h
                        rect = patches.Rectangle((x1, y1), w_n*w, h_n*h, linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1-5, class_names[class_id], bbox=dict(facecolor=color, alpha=0.7), fontsize=8, color='white')
                        bbox_count += 1
                    elif len(coords) >= 6 and len(coords) % 2 == 0:  # Polygon
                        points = [(coords[j]*w, coords[j+1]*h) for j in range(0, len(coords), 2)]
                        polygon = patches.Polygon(points, linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                        ax.add_patch(polygon)
                        cx, cy = sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points)
                        ax.text(cx, cy, class_names[class_id], bbox=dict(facecolor=color, alpha=0.7), fontsize=8, color='white', ha='center')
                        bbox_count += 1

            if bbox_count == 0:
                ax.text(w//2, h//2, "No annotations", ha='center', va='center', bbox=dict(facecolor='yellow', alpha=0.7), fontsize=12)

            ax.axis('off')
            plt.tight_layout()

            output_path = split_output_dir / f"{img_path.stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_path} ({bbox_count} annotations)")

    print(f"🎉 GT plots saved to: {output_base}")

def normalize_dataset_layout(original_data_yaml, dataset_name):
    global TEMP_DIR_PATH
    original_yaml_path = Path(original_data_yaml).resolve()
    original_data_dir = original_yaml_path.parent

    shm_dir = Path('/dev/shm')
    temp_base = shm_dir if shm_dir.exists() else Path(tempfile.gettempdir())
    TEMP_DIR_PATH = temp_base / f"yolo_norm_{dataset_name}_{os.getpid()}"
    TEMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    atexit.register(cleanup_temp_dir)

    with open(original_yaml_path, 'r') as f:
        original_yaml_data = yaml.safe_load(f)

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
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_yaml_data, f, default_flow_style=False, sort_keys=False)

    return str(TEMP_DIR_PATH), str(new_yaml_path)

def print_detailed_metrics(metrics, class_names, task):
    metric_box = metrics.seg if task == 'segment' else metrics.box
    task_letter = 'M' if task == 'segment' else 'B'

    print(f"\n--- Overall Metrics ---\nmAP50-95({task_letter}): {metric_box.map:.4f}\nmAP50({task_letter}): {metric_box.map50:.4f}")
    if hasattr(metrics, 'fitness'): print(f"Fitness: {metrics.fitness:.4f}")

    if metric_box.maps is not None:
        header = f"{'Class':<20} {'N':<8} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP75':<12} {'mAP50-95':<12}"
        print(f"{header}\n{'-' * len(header)}")

        for i, class_name in enumerate(class_names):
            instances = int(metrics.nt_per_class[i]) if i < len(metrics.nt_per_class) else 0
            precision = metric_box.p[i] if i < len(metric_box.p) else 0.0
            recall = metric_box.r[i] if i < len(metric_box.r) else 0.0
            map50 = metric_box.ap50[i] if i < len(metric_box.ap50) else 0.0
            # mAP75 = all_ap[:, 5] (IoU thresholds 0.5..0.95 step 0.05 -> index 5 is 0.75)
            if hasattr(metric_box, 'all_ap') and metric_box.all_ap is not None and i < len(metric_box.all_ap) and metric_box.all_ap.shape[1] > 5:
                map75 = float(metric_box.all_ap[i, 5])
            else:
                map75 = 0.0
            map50_95 = metric_box.maps[i] if i < len(metric_box.maps) else 0.0

            print(f"{class_name:<20} {instances:<8} {precision:<12.4f} {recall:<12.4f} {map50:<12.4f} {map75:<12.4f} {map50_95:<12.4f}")

_AUG_POLICY_FOR_YOLO = "native"  # set from args in __main__

def get_custom_augmentations():
    # When the bench --aug_policy is off/fair, defer entirely to the shared
    # cross-framework Albumentations pipeline so YOLO/Ultralytics RT-DETR see
    # exactly the same training-time augmentation distribution as EoMT,
    # RF-DETR, DINOv3-M2F, and DINOv3-SemSeg.
    if _AUG_POLICY_FOR_YOLO in ("off", "fair"):
        from bench_aug import shared_albumentations_policy
        return shared_albumentations_policy(_AUG_POLICY_FOR_YOLO, 640, framework="yolo")
    return [
        A.OneOf([
            A.MedianBlur(blur_limit=(5, 9), p=0.25),
            A.MotionBlur(blur_limit=(5, 9), p=0.5),
            A.UnsharpMask(blur_limit=(5, 9), p=0.5),
        ], p=0.15),
        A.OneOf([
            A.Compose([
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=(-70, 20),
                                     val_shift_limit=(-70, 10), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
                A.CLAHE(p=0.5),
            ]),
        ], p=0.015),
        A.OneOf([
            A.GlassBlur(sigma=0.5, max_delta=2, iterations=1, mode="fast", p=1.0),
            A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
            A.ImageCompression(quality_range=(40, 60), p=1.0),
        ], p=0.015),
        A.OneOf([
            A.RandomRain(slant_range=(-10, 10), drop_length=30, drop_width=1,
                         drop_color=(160, 160, 160), blur_value=9,
                         brightness_coefficient=0.9, rain_type='drizzle', p=0.5),
            A.RandomFog(fog_coef_range=(0.15, 0.3), alpha_coef=0.1, p=1.0),
        ], p=0.05),
    ]


def optional_aug_kwargs(args):
    out = {}
    for key in ("mosaic", "mixup", "fliplr", "flipud"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    return out

if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    # Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to YOLO dataset directory")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    ap.add_argument("--task", default="segment", choices=["detect", "segment"])
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument('--freeze', default=0, type=int) #freeze 8 works well
    ap.add_argument("--batch_sz", default=0, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--lrf", default=0.01, type=float)
    ap.add_argument("--opti", default="AdamW")
    ap.add_argument("--nbs", default=64, type=int)
    ap.add_argument("--cos_lr", action='store_true')
    ap.add_argument("--threads", default=8, type=int)
    ap.add_argument("--dropout", default=0.2, type=float)
    ap.add_argument("--weight_decay", default=0.001, type=float)
    ap.add_argument("--momentum", default=0.96, type=float)
    ap.add_argument("--warmup_momentum", default=0.78, type=float)
    ap.add_argument("--iou", default=0.7, type=float)
    ap.add_argument("--hsv_h", default=0.0, type=float)
    ap.add_argument("--hsv_s", default=0.0, type=float)
    ap.add_argument("--hsv_v", default=0.0, type=float)
    ap.add_argument("--cutmix", default=0.0, type=float)
    ap.add_argument("--copy_paste", default=0.0, type=float)
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
    ap.add_argument("--visualize", action='store_true')
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--tune", action='store_true')
    ap.add_argument("--tune_iter", default=5, type=int)
    ap.add_argument("--tune_aug", action='store_true')
    ap.add_argument("--inference_only", action='store_true')
    ap.add_argument("--plot_gt", action='store_true')
    # --- v2 bench_eval hooks ------------------------------------------------
    # Opt-in SAM2-prompted segmentation for detection-only YOLO variants so
    # `seg_mAP*` columns get filled fairly via a frozen foundation segmenter.
    # Ignored for YOLO-seg (already has a native mask head).
    ap.add_argument("--seg_via_sam2", action='store_true',
                    help="(detect-only) After val, prompt SAM2 with predicted boxes to compute seg mAP.")
    ap.add_argument("--sam2_ckpt",   default=os.environ.get("SAM2_CKPT", ""))
    ap.add_argument("--sam2_config", default=os.environ.get("SAM2_CONFIG", ""))
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, replace get_custom_augmentations() with the shared "
                         "cross-framework Albumentations pipeline from bench_aug.py.")
    args = ap.parse_args()
    _AUG_POLICY_FOR_YOLO = args.aug_policy  # noqa: F841 (rebound below)
    globals()["_AUG_POLICY_FOR_YOLO"] = args.aug_policy
    _aug_status = "applied" if args.aug_policy in ("off", "fair") else "native"
    print(f"[bench] aug_policy={args.aug_policy}: yolo -> {_aug_status}")

    # Setup
    ORIGINAL_DATA_DIR = Path(args.data).resolve()
    if not ORIGINAL_DATA_DIR.is_dir(): print(f"❌ Error: Data directory not found"); sys.exit(1)
    DATA_NAME = ORIGINAL_DATA_DIR.name
    ORIGINAL_DATA_YAML = ORIGINAL_DATA_DIR / 'data.yaml'
    if not ORIGINAL_DATA_YAML.exists(): print(f"❌ Error: data.yaml not found"); sys.exit(1)

    # Plot GT Mode - Independent execution
    if args.plot_gt:
        plot_ground_truth_standalone(str(ORIGINAL_DATA_YAML))
        print("✅ Ground truth plotting complete")
        sys.exit(0)

    # Continue with training/inference
    ACTIVE_DATA_DIR, ACTIVE_DATA_YAML = normalize_dataset_layout(ORIGINAL_DATA_YAML, DATA_NAME)
    ACTIVE_ROOT_DIR = Path(ACTIVE_DATA_DIR).parent
    TASK = args.task
    BATCH_SIZE = int(args.batch_sz) if args.batch_sz >= 1 else args.batch_sz

    # Build descriptive run name: <dataset>_<model_stem>
    MODEL_STEM = Path(args.model).stem
    if MODEL_STEM.endswith(f"_{DATA_NAME}"): MODEL_STEM = MODEL_STEM[:-(len(DATA_NAME) + 1)]
    _env_run = os.environ.get("BENCH_RUN_NAME")
    RUN_NAME = _env_run if _env_run else f"{DATA_NAME[:15]}_{MODEL_STEM}"
    os.environ.setdefault('WANDB_PROJECT', f"YOLO-{TASK}")

    # Clean up any existing validation directories from previous runs
    for val_dir_name in ['val', 'test']:
        val_cleanup_dir = Path(ACTIVE_ROOT_DIR) / val_dir_name
        if val_cleanup_dir.exists():
            shutil.rmtree(val_cleanup_dir, ignore_errors=True)

    with open(ACTIVE_DATA_YAML) as file:
        CLASSES = yaml.load(file, Loader=yaml.FullLoader)['names']

    # Hyperparameter tuning
    if args.tune:
        def train_objective(tune_config):
            model_tune = YOLO(args.model)
            model_tune.train(
                data=ACTIVE_DATA_YAML, task=TASK, imgsz=args.imgsz, epochs=args.epochs,
                patience=args.patience, batch=BATCH_SIZE, optimizer=args.opti,
                project=os.path.join(ACTIVE_ROOT_DIR, 'ray_tune_runs'),
                name=f"{RUN_NAME}_trial", exist_ok=True, verbose=False, pretrained=True,
                close_mosaic=min(10, args.epochs // 10), iou=args.iou, hsv_h=args.hsv_h,
                hsv_v=args.hsv_v, hsv_s=args.hsv_s, cutmix=args.cutmix, copy_paste=args.copy_paste, copy_paste_mode="mixup",
                degrees=tune_config.get("degrees", args.degrees) if args.aug_policy == "native" else 0.0,
                translate=tune_config.get("translate", args.translate) if args.aug_policy == "native" else 0.0,
                shear=args.shear if args.aug_policy == "native" else 0.0,
                perspective=args.perspective if args.aug_policy == "native" else 0.0,
                scale=tune_config.get("scale", args.scale) if args.aug_policy == "native" else 0.0,
                erasing=args.erasing if args.erasing is not None else 0.05,
                lrf=args.lrf, warmup_momentum=args.warmup_momentum, cos_lr=False,
                multi_scale=args.multi_scale, weight_decay=tune_config["weight_decay"],
                momentum=tune_config["momentum"], dropout=tune_config["dropout"], lr0=tune_config["lr0"],
                augmentations=get_custom_augmentations(), **optional_aug_kwargs(args)
            )
            del model_tune; gc.collect()

        search_space = {
            "weight_decay": tune.loguniform(1e-5, 1e-3), "momentum": tune.uniform(0.85, 0.98),
            "lr0": tune.loguniform(5e-4, 5e-3)
        }
        if args.tune_aug and args.aug_policy == "native":
            search_space.update({
                "scale": tune.uniform(0.0, 0.20),
                "translate": tune.uniform(0.0, 0.10),
                "degrees": tune.uniform(0.0, 10.0),
            })

        tune_metric = f"metrics/mAP50-95({'M' if TASK == 'segment' else 'B'})"

        tuner = tune.Tuner(
            tune.with_resources(train_objective, {"gpu": 1}), param_space=search_space,
            tune_config=tune.TuneConfig(metric=tune_metric, mode="max", search_alg=OptunaSearch(), num_samples=args.tune_iter),
            run_config=ray.train.RunConfig(name=f"tune_{RUN_NAME}", storage_path=os.path.join(ACTIVE_ROOT_DIR, 'ray_results'))
        )

        results_grid = tuner.fit()
        best_result = results_grid.get_best_result(metric=tune_metric, mode="max")

        print(f"✅ Best {tune_metric}: {best_result.metrics[tune_metric]:.4f}")
        args.lr = best_result.config['lr0']
        args.weight_decay = best_result.config['weight_decay']
        args.momentum = best_result.config['momentum']
        args.scale = best_result.config.get("scale", args.scale)
        args.translate = best_result.config.get("translate", args.translate)
        args.degrees = best_result.config.get("degrees", args.degrees)

        if ray.is_initialized(): ray.shutdown()

    # Training or inference
    final_model = None
    final_proj_name = ""

    if not args.inference_only:
        model_path_to_use = args.model
        is_resuming_run = False

        if args.resume and not args.tune:
            last_pt_path = Path.home() / 'bench_any' / 'runs' / TASK / RUN_NAME / 'weights' / 'last.pt'
            if os.path.exists(last_pt_path):
                model_path_to_use = last_pt_path
                is_resuming_run = True
            else:
                # Resume is ON by default in bench_any, but a fresh run simply
                # has no last.pt yet. Don't abort (that made yolo entries exit
                # instantly with rc=1 so the next model looked like it "ran
                # first"); just start training from the pretrained weights.
                print("ℹ️ Resume requested but no checkpoint yet; starting a fresh run.")

        model = YOLO(model_path_to_use)

        # Bench: log per-epoch mAP75 (and mAP50_95) to a sidecar so the paper
        # figures show full IoU breakdown per-epoch, not just the final point.
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import bench_plots as _bp
            _bp.install_ultra_map75_callback(model)
        except Exception as _e:
            print(f"[yolo] mAP75 callback skipped: {_e}")

        model.train(
            data=ACTIVE_DATA_YAML, project=os.path.join(ACTIVE_ROOT_DIR, 'runs', TASK),
            name=RUN_NAME, exist_ok=True, task=TASK, optimizer=args.opti,
            imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
            batch=BATCH_SIZE, lr0=args.lr, lrf=args.lrf, workers=args.threads, nbs=args.nbs,
            cos_lr=args.cos_lr, close_mosaic=min(10, args.epochs // 30),
            dropout=args.dropout, weight_decay=args.weight_decay, momentum=args.momentum,
            warmup_momentum=args.warmup_momentum,
            iou=args.iou, hsv_h=args.hsv_h, hsv_v=args.hsv_v, hsv_s=args.hsv_s,
            freeze=args.freeze if args.freeze > 0 else None,
            cutmix=args.cutmix, copy_paste=args.copy_paste, copy_paste_mode="mixup",
            degrees=args.degrees if args.aug_policy == "native" else 0.0,
            translate=args.translate if args.aug_policy == "native" else 0.0,
            shear=args.shear if args.aug_policy == "native" else 0.0,
            perspective=args.perspective if args.aug_policy == "native" else 0.0,
            scale=args.scale if args.aug_policy == "native" else 0.0,
            erasing=args.erasing if args.erasing is not None else 0.05, multi_scale=args.multi_scale,
            visualize=args.visualize, pretrained=not is_resuming_run, augmentations=get_custom_augmentations(),
            show_conf=False, show_boxes=False, show_labels=False, **optional_aug_kwargs(args)
        )

        final_proj_name = model.trainer.save_dir
        final_model = YOLO(os.path.join(final_proj_name, 'weights', 'best.pt'))
    else:
        # inference-only: load trained checkpoint from the run dir
        weights_dir = Path.home() / 'bench_any' / 'runs' / TASK / RUN_NAME / 'weights'
        ckpt_path = weights_dir / 'last.pt'
        if not ckpt_path.is_file():
            ckpt_path = weights_dir / 'best.pt'
        if ckpt_path.is_file():
            final_model = YOLO(str(ckpt_path))
            print(f"🔁 inference-only: loading trained checkpoint {ckpt_path}")
        else:
            final_model = YOLO(args.model)
            print(f"⚠️  inference-only: no checkpoint at {weights_dir}; using pretrained weights")
        final_proj_name = os.path.join(ACTIVE_ROOT_DIR, 'runs', TASK, RUN_NAME)

    # Final validation and predictions
    if final_model and final_proj_name:
        with open(ACTIVE_DATA_YAML) as f:
            active_yaml_config = yaml.safe_load(f)

        # All persistent outputs go under ~/bench_any/ — never into the project / dataset dir.
        bench_dir = Path.home() / 'bench_any'
        primary_save_dir = bench_dir / 'runs' / TASK / RUN_NAME

        # Validation with metrics using temp directory first
        final_metrics = {}
        for split in ('val', 'test'):
            if split not in active_yaml_config: continue
            print(f"\n🧪 Validating final model on '{split}' split...")
            # save_json=True makes Ultralytics emit predictions.json (COCO format)
            # under <project>/<name>/ — required by bench_eval.coco_eval_full so
            # we can compute S/M/L AP, AR_100, and mask→box cross-task box AP.
            metrics = final_model.val(data=ACTIVE_DATA_YAML, split=split,
                                      project=str(ACTIVE_ROOT_DIR), name=f"{split}_{RUN_NAME}",
                                      save=True, save_json=True, plots=False,
                                      visualize=args.visualize,
                                      # Crowded penguin scenes can hold several hundred
                                      # instances; Ultralytics' default max_det=300 caps
                                      # both the native metrics and the predictions.json
                                      # we feed to COCOeval. Raise to 1000 so dense
                                      # images aren't truncated (matches bench_eval.MAX_DETS).
                                      max_det=1000,
                                      show_conf=False, show_boxes=False, show_labels=False)
            print_detailed_metrics(metrics, CLASSES, TASK)
            if split == 'test':
                b = metrics.box; s = getattr(metrics, 'seg', None)
                def _m75(x):
                    a = getattr(x, 'all_ap', None)
                    return float(np.nanmean(a[:, 5])) if a is not None and a.shape[1] > 5 else 0.0
                def _f1(p, r): return float(2*p*r/(p+r)) if (p + r) > 0 else 0.0
                # Ensure we initialize with ALL keys from the bench_any schema so that
                # harvest_results_csv doesn't report 0.0 for missing keys that should be hyphenated.
                final_metrics = zero_metrics()
                final_metrics.update(dict(
                    box_P=float(b.mp), box_R=float(b.mr), box_F1=_f1(float(b.mp), float(b.mr)),
                    box_mAP50=float(b.map50), box_mAP75=_m75(b), box_mAP50_95=float(b.map),
                    seg_P=float(s.mp) if s else 0.0, seg_R=float(s.mr) if s else 0.0,
                    seg_F1=_f1(float(s.mp), float(s.mr)) if s else 0.0,
                    seg_mAP50=float(s.map50) if s else 0.0,
                    seg_mAP75=_m75(s) if s else 0.0,
                    seg_mAP50_95=float(s.map) if s else 0.0,
                ))
                # ---- v2: framework-agnostic post-eval via bench_eval ---------
                # Fills box_mAP_S/M/L + box_AR_100 (always) and seg_mAP_S/M/L
                # + seg_AR_100 (YOLO-seg) using pycocotools COCOeval on the
                # predictions.json Ultralytics just wrote. For YOLO-det with
                # --seg_via_sam2, also prompts SAM2 with predicted boxes to
                # produce a seg_mAP* column. All steps degrade to zeros on
                # failure (e.g. pycocotools missing) without breaking the run.
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
                        # Auto-correct category_id offset (Ultralytics may emit
                        # 1-indexed cat ids while our GT is 0-indexed). Defensive:
                        # no-op when sets already match; warns on no-match.
                        cat_fixed = val_dir / "_bench_predictions_catfixed.json"
                        n_cf, cf_mode = _be.remap_pred_categories(
                            str(pred_json), str(gt_json), str(cat_fixed))
                        if n_cf > 0:
                            print(f"   🔧 Remapped {n_cf} prediction category_ids "
                                  f"(mode={cf_mode}) to match GT")
                            pred_json = cat_fixed
                        # YOLO predictions.json contains both bbox and (for
                        # -seg models) segmentation entries in one file, so we
                        # use it for both iou types.
                        is_seg = TASK == "segment"
                        bbox_pred = str(pred_json)
                        segm_pred = str(pred_json) if is_seg else None
                        extra = _be.coco_eval_full(str(gt_json), bbox_pred, segm_pred)
                        # Use the shared COCOeval pass as the canonical bench
                        # metric so YOLO rows are comparable with non-Ultralytics
                        # wrappers. Ultralytics' native numbers are still printed
                        # above in its own run log.
                        for k, v in extra.items():
                            if k in _be.BOX_AP_KEYS or k in _be.SEG_AP_KEYS:
                                final_metrics[k] = v
                        # Mask→box cross-fill is redundant for YOLO-seg (we
                        # already have native box mAP from Ultralytics), but
                        # we run it anyway as a sanity check if the native
                        # box block came back all-zero (e.g. seg-only export).
                        if is_seg and float(b.map) == 0.0:
                            box_pred = val_dir / "_bench_box_from_masks.json"
                            n = _be.masks_to_box_preds(str(pred_json), str(box_pred))
                            if n > 0:
                                bx = _be.coco_eval(str(gt_json), str(box_pred), "bbox")
                                for k, v in bx.items():
                                    if k in _be.BOX_AP_KEYS:
                                        final_metrics.setdefault(k, 0.0)
                                        if final_metrics[k] == 0.0:
                                            final_metrics[k] = v
                        # SAM2-prompted seg AP. For pure-detector YOLO this is
                        # the only seg signal; for YOLO-seg it is reported in the
                        # sam2_* family alongside the native seg_* mask head, so a
                        # single row carries both native and foundation seg.
                        if args.seg_via_sam2 and TASK in ("detect", "segment"):
                            try:
                                cfg_yaml = yaml.safe_load(open(ACTIVE_DATA_YAML))
                                _root = Path(cfg_yaml.get("path", Path(ACTIVE_DATA_YAML).parent))
                                _split = cfg_yaml.get("test") or cfg_yaml.get("val")
                                _imgs = (_root / _split).resolve() if not Path(_split).is_absolute() else Path(_split)
                            except Exception:
                                _imgs = None
                            if args.sam2_ckpt and args.sam2_config and _imgs:
                                sam_out = val_dir / "_bench_sam2_seg.json"
                                sam_metrics = _be.sam2_seg_from_boxes(
                                    str(_imgs), str(pred_json),
                                    args.sam2_ckpt, args.sam2_config,
                                    str(gt_json), str(sam_out))
                                for k, v in sam_metrics.items():
                                    final_metrics[k] = v
                            else:
                                print("⚠️ --seg_via_sam2 set but --sam2_ckpt/--sam2_config "
                                      "missing or test images dir not resolvable; skipping.")
                    else:
                        print(f"⚠️ no predictions.json under {val_dir}; "
                              f"S/M/L AP will be 0 for this row.")
                except Exception as _e:
                    print(f"⚠️ bench_eval post-pass failed: {_e}")
                # ---- v2: params + eval FPS -----------------------------------
                try:
                    import bench_eval as _be2, torch as _t
                    final_metrics["params_M"] = _be2.model_params_M(final_model)
                    # One-shot FPS probe at the configured imgsz on a random
                    # tensor. Doesn't touch the model state; uses no_grad.
                    inner = getattr(final_model, "model", final_model)
                    if hasattr(inner, "eval") and _t.cuda.is_available():
                        inner = inner.cuda().eval()
                        x = _t.zeros(1, 3, args.imgsz, args.imgsz,
                                     device="cuda", dtype=next(inner.parameters()).dtype)
                        with _t.no_grad():
                            final_metrics["fps_eval"] = _be2.time_fps(lambda: inner(x))
                except Exception as _e:
                    print(f"⚠️ params/fps probe failed: {_e}")

        # Save results — single target under ~/bench_any/ (nothing in project / dataset).
        try:
            primary_save_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(final_proj_name, primary_save_dir, dirs_exist_ok=True)
            for split in ('val', 'test'):
                temp_val_dir = Path(ACTIVE_ROOT_DIR) / f"{split}_{RUN_NAME}"
                if temp_val_dir.exists():
                    shutil.copytree(temp_val_dir, primary_save_dir / split, dirs_exist_ok=True)

            # Write results.csv (consumed by bench_any.py orchestrator)
            if final_metrics:
                # Report the actual micro-batch and the realised effective batch
                # (Ultralytics auto-accumulates micro_bs up to nbs) so the bench
                # CSV shows the real per-model batch, not the orchestrator arg.
                # When batch_sz is fractional (auto-batch), the literal micro-batch
                # is chosen inside Ultralytics and unknown here, so we leave it for
                # the orchestrator to fall back on its resolved value.
                if BATCH_SIZE >= 1:
                    final_metrics["batch_sz"] = int(BATCH_SIZE)
                    final_metrics["batch_effective"] = max(int(args.nbs), int(BATCH_SIZE))
                else:
                    final_metrics["batch_effective"] = int(args.nbs)
                import csv as _csv
                with open(primary_save_dir / 'results.csv', 'w', newline='') as fh:
                    w = _csv.DictWriter(fh, fieldnames=list(final_metrics.keys()))
                    w.writeheader(); w.writerow(final_metrics)

            print(f"✅ Results copied to: {primary_save_dir}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    cleanup_temp_dir()
    print("🎉 Complete!")
