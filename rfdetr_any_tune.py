# USAGE (Train):  python3 rfdetr_any_tune.py --data /path/to/dataset --epochs 100 --batch_sz 4 --model rfdetr-seg-medium
# USAGE (Resume): python3 rfdetr_any_tune.py --data /path/to/dataset --resume
#
# Mirrors yolo_any_tune.py CLI/structure but uses Roboflow's `rfdetr` package. Input is a
# YOLO-style dataset (data.yaml + train/val/test image+label dirs); converted internally to
# the COCO layout RF-DETR expects. Defaults to a segmentation variant for fair comparison
# with yolo*-seg; pass a model name with 'det' (e.g. 'rfdetr-base') to force detection-only.

import os, sys, yaml, argparse, shutil, atexit, tempfile, hashlib, json, csv, glob, subprocess, logging
from pathlib import Path
from bench_schema import zero_metrics

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TEMP_DIR_PATH = None
def cleanup_temp_dir():
    global TEMP_DIR_PATH
    if TEMP_DIR_PATH and TEMP_DIR_PATH.exists():
        shutil.rmtree(TEMP_DIR_PATH, ignore_errors=True); TEMP_DIR_PATH = None


def cleanup_stale_shm():
    """Remove leftover rfdetr_* dirs in /dev/shm whose owning pid is no longer alive.
    Previous OOM crashes in this benchmark sweep leave behind big symlink trees + run
    outputs that fill /dev/shm; this keeps the sweep self-healing."""
    shm = Path('/dev/shm')
    if not shm.exists(): return
    for d in shm.glob('rfdetr_*'):
        try:
            pid = int(d.name.rsplit('_', 1)[-1])
        except ValueError:
            continue
        try:
            os.kill(pid, 0)  # alive?
        except OSError:
            shutil.rmtree(d, ignore_errors=True)


def _yolo_to_coco_split(img_dir, lbl_dir, class_names, json_path, image_id_start=1, ann_id_start=1):
    """Convert one YOLO split (images+labels dirs) to a COCO json. Returns (next_img_id, next_ann_id)."""
    import cv2
    # Hashed image IDs from bench_eval to match Ultralytics/COCOeval expectations.
    # Sequential IDs 1, 2, 3... cause mismatches in post-hoc evaluation.
    try:
        from bench_eval import safe_image_id
    except ImportError:
        def safe_image_id(s): return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF

    images, annotations = [], []
    cats = [{"id": i, "name": n, "supercategory": "object"} for i, n in enumerate(class_names)]
    img_id = image_id_start - 1
    ann_id = ann_id_start
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']: continue
        im = cv2.imread(str(img_path))
        if im is None: continue
        h, w = im.shape[:2]
        img_id = safe_image_id(img_path.stem)
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in open(lbl):
                parts = line.strip().split()
                if len(parts) < 5: continue
                cid = int(parts[0]); coords = [float(x) for x in parts[1:]]
                if cid >= len(class_names): continue
                if len(coords) == 4:
                    xc, yc, bw, bh = coords
                    x1, y1 = (xc - bw/2)*w, (yc - bh/2)*h
                    bbox = [x1, y1, bw*w, bh*h]
                    annotations.append({"id": ann_id, "image_id": img_id, "category_id": cid,
                                        "bbox": bbox, "area": bw*w*bh*h, "iscrowd": 0,
                                        "segmentation": []})
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    pts = [(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)]
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    bbox = [x1, y1, x2-x1, y2-y1]
                    seg = [[c for p in pts for c in p]]
                    annotations.append({"id": ann_id, "image_id": img_id, "category_id": cid,
                                        "bbox": bbox, "area": (x2-x1)*(y2-y1), "iscrowd": 0,
                                        "segmentation": seg})
                ann_id += 1
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({"images": images, "annotations": annotations, "categories": cats}, f)
    return img_id, ann_id


def yolo_to_coco_layout(yolo_data_yaml, dataset_name, eval_split_as_valid=False):
    """Convert a YOLO dataset (data.yaml) into the COCO layout RF-DETR expects:
    <out>/{train,valid,test}/_annotations.coco.json + flat image files.
    Images are symlinked from source to keep this fast and free of disk pressure.

    eval_split_as_valid=True is for third-party trainers that only evaluate a
    validation dataloader. In that mode, YOLO test is mapped into the generated
    `valid/` directory when test exists, so every bench row evaluates the same
    held-out split.
    """
    global TEMP_DIR_PATH
    yp = Path(yolo_data_yaml).resolve(); cfg_base = yp.parent
    cfg = yaml.safe_load(open(yp))
    root = Path(cfg.get("path", cfg_base)).expanduser()
    if not root.is_absolute():
        root = (cfg_base / root).resolve()
    class_names = cfg.get('names') if isinstance(cfg.get('names'), list) else list(cfg['names'].values())
    cleanup_stale_shm()
    shm = Path('/dev/shm')
    base = shm if shm.exists() else Path(tempfile.gettempdir())
    TEMP_DIR_PATH = base / f"rfdetr_coco_{dataset_name}_{os.getpid()}"
    TEMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    atexit.register(cleanup_temp_dir)

    # Map YOLO split keys -> generated COCO dirs. Some upstream wrappers only
    # evaluate "valid"; for those, put YOLO test into valid when test exists.
    if eval_split_as_valid and cfg.get("test"):
        split_dirs = {'train': 'train', 'test': 'valid'}
    else:
        split_dirs = {'train': 'train', 'val': 'valid', 'valid': 'valid', 'test': 'test'}
    next_img, next_ann = 1, 1
    for split_key, dest_name in split_dirs.items():
        if split_key not in cfg: continue
        src_rel = Path(cfg[split_key])
        src_img = src_rel if src_rel.is_absolute() else root / src_rel
        # Swap only the 'images' path *component* (mirrors bench_eval). A blind
        # str.replace also rewrites any parent dir containing the substring
        # "images" (e.g. /data/my_images/train/images), silently pointing at a
        # non-existent label dir so every annotation is dropped → 0 mAP.
        _parts = list(src_img.parts)
        if "images" in _parts:
            _parts[_parts.index("images")] = "labels"
            src_lbl = Path(*_parts)
        else:
            src_lbl = src_img
        if not src_img.is_dir(): continue
        dest = TEMP_DIR_PATH / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        for img_path in src_img.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']:
                link = dest / img_path.name
                if not link.exists(): os.symlink(img_path.resolve(), link)
        next_img, next_ann = _yolo_to_coco_split(src_img, src_lbl, class_names,
                                                  dest / "_annotations.coco.json",
                                                  next_img, next_ann)
    return TEMP_DIR_PATH, class_names


def pick_rfdetr_class(name):
    """Map a model-name token to the appropriate rfdetr class. Defaults to the
    segmentation variant; pass 'det' / 'detect' in the name to force detection-only."""
    import rfdetr as _rf
    n = name.lower()
    is_seg = "seg" in n or not ("-det" in n or n.endswith("det") or "detect" in n)
    # NB: order matters — check 'xlarge' before 'large' so substring match doesn't shadow it
    for size, attr_det, attr_seg in [
        ("xlarge", "RFDETRXLarge", "RFDETRSegXLarge"),
        ("nano",   "RFDETRNano",   "RFDETRSegNano"),
        ("small",  "RFDETRSmall",  "RFDETRSegSmall"),
        ("medium", "RFDETRMedium", "RFDETRSegMedium"),
        ("large",  "RFDETRLarge",  "RFDETRSegLarge"),
    ]:
        if size in n: return getattr(_rf, attr_seg if is_seg else attr_det), is_seg
    # default base
    return (_rf.RFDETRSegMedium if is_seg else _rf.RFDETRBase), is_seg


class _RFDinoPretrainNoiseFilter(logging.Filter):
    NOISY_PARTS = (
        "Using a different number of positional encodings than DINOv2",
        "Using patch size 12 instead of 14",
        "Using patch size 16 instead of 14",
        "Checkpoint has 90 classes but model is configured for 1",
        "Model is not optimized for inference",
        "load_pretrain_weights: checkpoint lacks args.num_queries / args.group_detr",
    )

    def filter(self, record):
        msg = record.getMessage()
        return not any(part in msg for part in self.NOISY_PARTS)


def install_rfdetr_log_filter():
    """Suppress duplicate upstream DINOv2 init warnings after our clearer note."""
    for logger_name in ("rf-detr", "rfdetr"):
        lg = logging.getLogger(logger_name)
        if not any(isinstance(f, _RFDinoPretrainNoiseFilter) for f in lg.filters):
            lg.addFilter(_RFDinoPretrainNoiseFilter())


def quiet_lightning_tips():
    """Silence Lightning's promotional info tips without hiding warnings/errors."""
    try:
        import lightning.fabric.utilities.rank_zero as _brz  # type: ignore
        import lightning.pytorch.utilities.rank_zero as _lrz  # type: ignore
        import pytorch_lightning.utilities.rank_zero as _prz  # type: ignore
        _brz.rank_zero_info = lambda *a, **k: None
        _lrz.rank_zero_info = lambda *a, **k: None
        _prz.rank_zero_info = lambda *a, **k: None
    except Exception:
        pass


def rfdetr_variant_defaults(model_cls):
    cfg_cls = getattr(model_cls, "_model_config_class", None)
    fields = getattr(cfg_cls, "model_fields", {}) or {}

    def _default(name, fallback=None):
        field = fields.get(name)
        return getattr(field, "default", fallback) if field is not None else fallback

    return {
        "patch_size": _default("patch_size"),
        "resolution": _default("resolution"),
        "pretrain_weights": _default("pretrain_weights"),
        "encoder": _default("encoder"),
    }


def harvest_metrics_from_tb(out_dir):
    """Read the latest events.out.tfevents under out_dir and return dict of test/val metrics."""
    # Ensure we initialize with ALL keys from the bench_any schema so that
    # harvest_results_csv doesn't report 0.0 for missing keys that should be hyphenated.
    metrics = zero_metrics()
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator # noqa
    except Exception as e:
        print(f"⚠️ tensorboard not importable ({e}); metrics = 0"); return metrics
    cands = []
    for r, _, files in os.walk(out_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                fp = os.path.join(r, f); cands.append((os.path.getmtime(fp), fp))
    if not cands:
        print(f"⚠️ no tfevents under {out_dir}"); return metrics
    cands.sort(reverse=True)
    ea = EventAccumulator(cands[0][1], size_guidance={"scalars": 0}); ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    def _last(*names):
        for n in names:
            if n in tags:
                evs = ea.Scalars(n)
                if evs: return float(evs[-1].value)
        return 0.0
    metrics["box_mAP50"]    = _last("test/mAP_50",    "val/mAP_50")
    metrics["box_mAP75"]    = _last("test/mAP_75",    "val/mAP_75")
    metrics["box_mAP50_95"] = _last("test/mAP_50_95", "val/mAP_50_95")
    metrics["box_P"]        = _last("test/precision", "val/precision")
    metrics["box_R"]        = _last("test/recall",    "val/recall")
    metrics["box_F1"]       = _last("test/F1",        "val/F1")
    metrics["seg_mAP50"]    = _last("test/segm_mAP_50",    "val/segm_mAP_50",
                                    "test/segm/mAP_50",     "val/segm/mAP_50")
    metrics["seg_mAP75"]    = _last("test/segm_mAP_75",    "val/segm_mAP_75",
                                    "test/segm/mAP_75",     "val/segm/mAP_75",
                                    "test/mAP_segm_75",     "val/mAP_segm_75")
    metrics["seg_mAP50_95"] = _last("test/segm_mAP_50_95", "val/segm_mAP_50_95",
                                    "test/segm/mAP_50_95",  "val/segm/mAP_50_95")
    # Size-bucketed AP (COCO small/medium/large). RF-DETR logs these alongside
    # the headline mAPs via faster-coco-eval, under both 'test/' and 'val/'.
    # Tag spelling matches the existing siblings: detection uses 'mAP_<size>'
    # and segmentation uses 'segm_mAP_<size>'. Any missing tag falls through
    # _last() to 0.0, so this is safe on older runs too.
    metrics["box_mAP_S"]    = _last("test/mAP_small",       "val/mAP_small",
                                    "test/mAP_s",           "val/mAP_s")
    metrics["box_mAP_M"]    = _last("test/mAP_medium",      "val/mAP_medium",
                                    "test/mAP_m",           "val/mAP_m")
    metrics["box_mAP_L"]    = _last("test/mAP_large",       "val/mAP_large",
                                    "test/mAP_l",           "val/mAP_l")
    metrics["seg_mAP_S"]    = _last("test/segm_mAP_small",  "val/segm_mAP_small",
                                    "test/segm_mAP_s",      "val/segm_mAP_s")
    metrics["seg_mAP_M"]    = _last("test/segm_mAP_medium", "val/segm_mAP_medium",
                                    "test/segm_mAP_m",      "val/segm_mAP_m")
    metrics["seg_mAP_L"]    = _last("test/segm_mAP_large",  "val/segm_mAP_large",
                                    "test/segm_mAP_l",      "val/segm_mAP_l")
    return metrics


def _install_bench_aug_policy_for_rfdetr(policy: str) -> None:
    """Thin shared-policy hook for RF-DETR.

    Wraps ``rfdetr.datasets.coco.build_coco`` so the returned ``CocoDetection``
    instance is tagged with its ``image_set`` and ``CocoDetection.__getitem__``
    is monkey-patched to push the shared Albumentations pipeline through the
    train split's (image, target) pair BEFORE rfdetr's own ``_transforms`` run.
    Eval/val splits stay untouched. Logs one ``[bench] aug_policy=…`` line.
    """
    if policy == "native":
        print(f"[bench] aug_policy={policy}: rfdetr -> native")
        return
    try:
        from bench_aug import shared_albumentations_policy
        from rfdetr.datasets import coco as _rfcoco
        import albumentations as _A
        import numpy as _np
        import torch as _torch
        from PIL import Image as _PIL
    except Exception as e:
        print(f"[bench] aug_policy={policy}: rfdetr -> unsupported ({e})")
        return
    pipeline = shared_albumentations_policy(policy, 640)
    if not pipeline:
        print(f"[bench] aug_policy={policy}: rfdetr -> applied (empty)")
        return

    _orig_build = _rfcoco.build_coco
    _orig_getitem = _rfcoco.CocoDetection.__getitem__

    def _patched_getitem(self, idx):
        if getattr(self, "_bench_image_set", "") != "train":
            return _orig_getitem(self, idx)
        img, target = super(_rfcoco.CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        try:
            arr = _np.array(img.convert("RGB") if isinstance(img, _PIL.Image) else img)
            boxes_xyxy = target.get("boxes")
            labels = target.get("labels")
            masks = target.get("masks")
            kwargs = {"image": arr}
            extra = {}
            if boxes_xyxy is not None and len(boxes_xyxy) > 0:
                bb = boxes_xyxy.tolist()
                extra["bbox_params"] = _A.BboxParams(format="pascal_voc",
                                                    label_fields=["labels"],
                                                    min_visibility=0.1)
                kwargs["bboxes"] = bb
                kwargs["labels"] = labels.tolist() if labels is not None else [0] * len(bb)
            if masks is not None and masks.shape[0] > 0:
                kwargs["masks"] = [m.numpy().astype(_np.uint8) for m in masks]
            t = _A.Compose(pipeline, **extra)
            aug = t(**kwargs)
            img = _PIL.fromarray(aug["image"])
            if "bboxes" in aug:
                kept_b = aug["bboxes"]; kept_l = aug.get("labels", [])
                if len(kept_b):
                    target["boxes"] = _torch.as_tensor(kept_b, dtype=_torch.float32).reshape(-1, 4)
                    target["labels"] = _torch.as_tensor(kept_l, dtype=_torch.int64)
                    target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * \
                                     (target["boxes"][:, 3] - target["boxes"][:, 1])
                    target["iscrowd"] = _torch.zeros(len(kept_b), dtype=_torch.int64)
                else:
                    target["boxes"] = _torch.zeros((0, 4), dtype=_torch.float32)
                    target["labels"] = _torch.zeros((0,), dtype=_torch.int64)
                    target["area"] = _torch.zeros((0,), dtype=_torch.float32)
                    target["iscrowd"] = _torch.zeros((0,), dtype=_torch.int64)
            if "masks" in aug and masks is not None:
                if aug["masks"]:
                    target["masks"] = _torch.from_numpy(
                        _np.stack([m.astype(bool) for m in aug["masks"]], axis=0))
                else:
                    target["masks"] = _torch.zeros((0, arr.shape[0], arr.shape[1]), dtype=_torch.bool)
        except Exception:
            pass  # On any albumentations error, fall through with original (img, target).
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def _patched_build(image_set, args_, resolution):
        ds = _orig_build(image_set, args_, resolution)
        try:
            ds._bench_image_set = str(image_set).split("_")[0]
        except Exception:
            pass
        return ds

    _rfcoco.CocoDetection.__getitem__ = _patched_getitem
    _rfcoco.build_coco = _patched_build
    print(f"[bench] aug_policy={policy}: rfdetr -> applied")


# When this module is imported by sibling *_any_tune.py wrappers (e.g.
# dinov3_m2f_any_tune.py / eomt_any_tune.py reusing yolo_to_coco_layout),
# the CLI + training block below MUST NOT execute at import time — otherwise
# importing the helpers triggers a full RF-DETR run with the parent process's
# sys.argv (this is exactly what caused the bug where bench_any.py routing
# dinov3-m2f-* would still download rf-detr-seg-small.pt and start training).
# Guard everything below behind the standard __main__ check.
if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    # Ensure faster-coco-eval is installed for metrics. Upstream rfdetr uses it
    # as the backend for torchmetrics.MeanAveragePrecision.
    try:
        import faster_coco_eval
    except ImportError:
        print("📦 [rfdetr] pip install faster-coco-eval")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-coco-eval"])

    # Args (mirrors yolo_any_tune.py shape; some flags are no-ops for RF-DETR but accepted for consistency)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to YOLO dataset directory containing data.yaml")
    ap.add_argument("--model", default="rfdetr-seg-medium",
                    help="rfdetr / rfdetr-base / rfdetr-{nano,small,medium,large,xlarge} / rfdetr-seg-{nano,small,medium,large,xlarge}")
    ap.add_argument("--task", default="segment", choices=["detect", "segment"])
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--batch_sz", default=4, type=float,
                    help="RF-DETR int batch_size. Float<1 (Ultralytics-style) -> 'auto' (RF-DETR probes the model on the GPU and picks the largest safe micro-batch, then sets grad_accum_steps so safe_micro * grad_accum ~= auto_batch_target_effective=16 per device).")
    ap.add_argument("--grad_accum_steps", default=0, type=int,
                    help="Gradient accumulation steps. 0 = let RF-DETR decide (default=4 with explicit batch; auto-tuned when --batch_sz<1 -> 'auto'). >0 overrides; effective_batch = batch_size * grad_accum_steps * devices.")
    ap.add_argument("--batch_effective", default=0, type=int,
                    help="Target effective batch (shared bench floor). When >0 and --batch_sz is an explicit int and --grad_accum_steps unset, computes grad_accum=ceil(N/batch_sz). Ignored on the auto path (--batch_sz<1) since RF-DETR self-tunes to its own target=16.")
    ap.add_argument("--imgsz", default=624, type=int, help="RF-DETR default resolution is 576 (multiples of 56)")
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--threads", default=4, type=int)
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--inference_only", action='store_true', help="Skip training; eval given weights via train(epochs=0,run_test=True)")
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, monkey-patch rfdetr.datasets.coco.CocoDetection.__getitem__ "
                         "so the train split passes through the shared Albumentations pipeline before "
                         "rfdetr's own _transforms run. Eval split is left untouched.")
    # Opt-in SAM2-prompted segmentation for detection-only RF-DETR variants so
    # the bench summary can report seg mAP for pure detectors.  seg variants
    # (rf-detr-seg-*) also accept it to report sam2_* alongside native seg_*.
    ap.add_argument("--seg_via_sam2", action='store_true',
                    help="(detect-only or seg): after validation, prompt SAM2 with predicted boxes to compute seg mAP.")
    ap.add_argument("--sam2_ckpt", default=os.environ.get("SAM2_CKPT", ""))
    ap.add_argument("--sam2_config", default=os.environ.get("SAM2_CONFIG", ""))
    args = ap.parse_args()
    _install_bench_aug_policy_for_rfdetr(args.aug_policy)

    ORIG_DATA = Path(args.data).resolve()
    if not ORIG_DATA.is_dir(): print(f"❌ Data directory not found"); sys.exit(1)
    DATA_YAML = ORIG_DATA / 'data.yaml'
    if not DATA_YAML.exists(): print(f"❌ data.yaml not found"); sys.exit(1)
    DATA_NAME = ORIG_DATA.name

    print(f"🔄 Converting YOLO → COCO under /dev/shm …")
    COCO_DIR, CLASSES = yolo_to_coco_layout(DATA_YAML, DATA_NAME)
    print(f"✅ COCO layout ready at {COCO_DIR} (classes: {CLASSES})")

    # Resolve run dir
    MODEL_STEM = Path(args.model).stem
    # bench_any.py forwards the fully-resolved run name (incl. hyperparameter
    # fingerprint) via BENCH_RUN_NAME so distinct configs don't overwrite each
    # other; standalone runs derive it locally.
    _env_run = os.environ.get("BENCH_RUN_NAME")
    RUN_NAME = _env_run if _env_run else f"{DATA_NAME[:15]}_{MODEL_STEM}"
    shm = Path('/dev/shm'); base = shm if shm.exists() else Path(tempfile.gettempdir())
    TASK_DIR = "segment" if args.task == "segment" else "detect"
    RUN_DIR = base / f"rfdetr_runs_{os.getpid()}" / TASK_DIR / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('WANDB_PROJECT', f"RFDETR-{TASK_DIR}")
    os.environ.setdefault('WANDB_DIR', str(RUN_DIR))

    # Build model
    ModelCls, is_seg = pick_rfdetr_class(args.model)
    print(f"🧠 Using {ModelCls.__name__}  (seg={is_seg})")
    install_rfdetr_log_filter()
    defaults = rfdetr_variant_defaults(ModelCls)
    ps = defaults.get("patch_size")
    if ps and int(ps) != 14:
        print(f"ℹ️ RF-DETR {ModelCls.__name__} is a patch-{ps} published checkpoint "
              f"({defaults.get('pretrain_weights')}); keeping that architecture so the "
              "RF-DETR weights load. Suppressing duplicate upstream DINOv2 patch-14 warnings.")
    if len(CLASSES) == 1:
        print("ℹ️ RF-DETR single-class COCO eval: normalizing predicted category_id to 1 "
              "to match the generated 1-indexed GT categories.")

    # RF-DETR wants an int batch_size, OR the literal "auto". When set to "auto" RF-DETR
    # (rfdetr/detr.py:584) probes the actual model with a forward+backward pass on the
    # target device and returns:
    #   safe_micro_batch              -> config.batch_size
    #   recommended_grad_accum_steps  -> config.grad_accum_steps
    # such that safe_micro * grad_accum ~= auto_batch_target_effective (default 16) per
    # device. So passing Ultralytics-style fractional --batch_sz <1 forwards to "auto" and
    # RF-DETR self-sizes both micro-batch AND grad-accum to keep effective batch sane on
    # whatever GPU is available — this is the recommended path for the bench sweep.
    bs = int(args.batch_sz) if args.batch_sz >= 1 else "auto"

    # Free any leftover CUDA / shared-memory state from a previous run in the same
    # benchmark sweep before allocating the new model.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache(); _torch.cuda.ipc_collect()
    except Exception: pass

    init_kwargs = {}
    # When resuming, point at last checkpoint in the bench_any save dir.
    # All persistent outputs land under ~/bench_any/ — never the project / dataset dir.
    bench_dir = Path.home() / 'bench_any'
    PRIMARY = bench_dir / 'runs' / TASK_DIR / RUN_NAME
    if args.resume:
        last = PRIMARY / 'checkpoint_last.pth'
        if last.exists():
            init_kwargs["pretrain_weights"] = str(last)
    # When inferencing only, look for a trained checkpoint in the
    # matching run dir.  bench_any forwards --resume so the dir name
    # carries the trained hyperparameter fingerprint (not e0).  Fall
    # back to the --model pretrained weights file only when no trained
    # checkpoint is found — otherwise inference-only silently scores
    # fresh pretrained weights ~30 mAP points below the trained result.
    if not init_kwargs and args.inference_only:
        # Priority: checkpoint_best_total.pth (what training's final
        # Trainer.test() loads), then last.ckpt (Lightning final state),
        # then checkpoint_*.pth by mtime.
        best_total = PRIMARY / "checkpoint_best_total.pth"
        if best_total.exists():
            init_kwargs["pretrain_weights"] = str(best_total)
            print(f"🔁 inference-only: loading best-total checkpoint {best_total.name}")
        if not init_kwargs:
            last_ckpt = PRIMARY / "last.ckpt"
            if last_ckpt.exists():
                init_kwargs["pretrain_weights"] = str(last_ckpt)
                print(f"🔁 inference-only: loading trained checkpoint {last_ckpt.name}")
        if not init_kwargs:
            ckpts = sorted(PRIMARY.rglob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
            if not ckpts:
                ckpts = sorted(PRIMARY.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime,
                               reverse=True)
            if ckpts:
                init_kwargs["pretrain_weights"] = str(ckpts[0])
                print(f"🔁 inference-only: loading trained checkpoint {ckpts[0].name}")
    # If --model is an explicit file path (e.g. ~/yolo_general/models/rf-detr-seg-xlarge.pt),
    # honour it as the pretrained weights. Otherwise let rfdetr download the default for the
    # selected class into its own cache. Resume / inference-only checkpoint takes precedence.
    if not init_kwargs and Path(args.model).expanduser().is_file():
        init_kwargs["pretrain_weights"] = str(Path(args.model).expanduser().resolve())
        print(f"📦 Using user-supplied weights: {init_kwargs['pretrain_weights']}")
    if not init_kwargs:
        print(f"⚠️ No checkpoint or pretrained weights found; rfdetr will use its own defaults.")
    model = ModelCls(**init_kwargs)

    # RF-DETR uses a pretrained-specific positional-encoding tied to (resolution // patch_size).
    # Overriding `resolution` re-initialises the PE and DESTROYS pretrained backbone
    # compatibility (metrics collapse to 0). Do NOT pass `resolution=` — let rfdetr use the
    # config default for the chosen variant. `--imgsz` is informational only here.
    print(f"ℹ️ --imgsz is informational for RF-DETR; using its built-in pretrained resolution.")

    train_kwargs = dict(
        dataset_dir=str(COCO_DIR), output_dir=str(RUN_DIR),
        batch_size=bs, lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.threads,
        tensorboard=True, run_test=True,
    )
    # Only forward grad_accum_steps when explicitly set (>0). Leaving it unset lets the
    # RF-DETR auto-batch path overwrite the config default (4) with the probe-derived
    # value, which is what we want for --batch_sz<1. Useful manual override: when the
    # user pins a low micro-batch (e.g. --batch_sz 2 for rfdetr-seg-large on 16GB),
    # pass --grad_accum_steps 8 to keep the effective batch at 16.
    if args.grad_accum_steps > 0:
        train_kwargs["grad_accum_steps"] = args.grad_accum_steps
        print(f"📐 grad_accum_steps={args.grad_accum_steps} (effective batch ≈ {args.grad_accum_steps} × micro-bs)")
    elif args.batch_effective > 0 and isinstance(bs, int):
        # Bench-level effective-batch floor; only meaningful when micro-bs is explicit.
        _ga = max(1, (args.batch_effective + bs - 1) // bs)
        train_kwargs["grad_accum_steps"] = _ga
        print(f"📐 grad_accum_steps={_ga} (auto from --batch_effective={args.batch_effective}, micro_bs={bs}, effective≈{_ga * bs})")

    quiet_lightning_tips()
    if args.inference_only or int(args.epochs) <= 0:
        print("🧪 RF-DETR eval-only: skipping trainer and generating COCOeval predictions")
    else:
        _ga = args.grad_accum_steps if args.grad_accum_steps > 0 else 'auto/default'
        print(f"🚂 Training {args.epochs} epochs (patience={args.patience}, bs={bs}, grad_accum={_ga}) …")
        model.train(epochs=args.epochs, early_stopping=True,
                    early_stopping_patience=args.patience, **train_kwargs)

    # Harvest + persist.  In training mode the TB events live under
    # RUN_DIR (/dev/shm/rfdetr_runs_<PID>/...).  In inference-only mode
    # RUN_DIR is a fresh empty directory (different PID), so we read the
    # persisted events from PRIMARY (~/bench_any/runs/...) instead and
    # fall back to RUN_DIR when PRIMARY has none (first-ever run).
    _tb_src = str(RUN_DIR)
    if args.inference_only or int(args.epochs) <= 0:
        if list(PRIMARY.glob("events.out.tfevents*")):
            _tb_src = str(PRIMARY)
        else:
            print("⚠️ No persisted TB events in PRIMARY; will fall back to pycocotools headline mAP")
    metrics = harvest_metrics_from_tb(_tb_src)
    # ---- v2: params + eval FPS via the live model object ----------------
    # RF-DETR's `model` wrapper exposes `.model` (the underlying nn.Module)
    # plus `.predict` for a forward pass; using both keeps the timing on the
    # real evaluation graph (post-fusion, post-quant) rather than a dummy.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bench_eval as _be, torch as _t, torch.nn as _nn
        # `model` is an RFDETR wrapper; `model.model` is a `ModelContext`
        # (Lightning-style wrapper) which is NOT an nn.Module. We need to drill
        # down to the actual LWDETR core. Try a few known attribute paths and
        # fall back to a generic search for the first nn.Module attribute.
        inner = None
        for path in (("model", "model"), ("model", "lit_model", "model"),
                     ("model", "module"), ("model", "network"), ("model",)):
            cand = model
            try:
                for a in path: cand = getattr(cand, a)
            except AttributeError:
                cand = None
            if isinstance(cand, _nn.Module):
                inner = cand; break
        if inner is None:
            # Last resort: scan attributes of model.model for any nn.Module.
            ctx = getattr(model, "model", None)
            if ctx is not None:
                for a in dir(ctx):
                    if a.startswith("_"): continue
                    try: v = getattr(ctx, a)
                    except Exception: continue
                    if isinstance(v, _nn.Module): inner = v; break
        if inner is not None:
            metrics["params_M"] = sum(p.numel() for p in inner.parameters()) / 1e6
        else:
            print("⚠️ rfdetr params probe: no nn.Module found under model.model")
        # FPS probe via the public RF-DETR predict() API on a real test image.
        # Raw forward on the inner LWDETR fails (DETRs need targets/preproc),
        # so use the high-level wrapper which handles preprocessing + postproc.
        try:
            cfg_yaml = yaml.safe_load(open(str(Path(args.data) / "data.yaml")))
            _root = Path(cfg_yaml.get("path", Path(args.data)))
            if not _root.is_absolute():
                _root = (Path(args.data) / _root).resolve()
            _split = cfg_yaml.get("test") or cfg_yaml.get("val")
            img_path = None
            if _split:
                _img_dir = (_root / _split).resolve() if not Path(_split).is_absolute() else Path(_split)
                if _img_dir.is_dir():
                    _exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                    for p in sorted(_img_dir.iterdir()):
                        if p.suffix.lower() in _exts:
                            img_path = p; break
            if img_path is None:
                print("⚠️ rfdetr fps probe skipped: no test image resolvable")
            else:
                # RF-DETR's DINOv2 backbone needs input divisible by 24 (patch=12).
                # Pre-resize to a safe multiple of 24 to avoid the
                # "Backbone requires input shape to be divisible by 24" error.
                try:
                    from PIL import Image as _Image
                    _img = _Image.open(str(img_path)).convert("RGB")
                    _sz = max(24, (int(args.imgsz) // 24) * 24)
                    _img = _img.resize((_sz, _sz), _Image.BILINEAR)
                    metrics["fps_eval"] = float(_be.time_fps(
                        lambda: model.predict(_img),
                        n_warmup=3, n_iter=10))
                except Exception as _e3:
                    # Fallback: try the raw path; RF-DETR may handle the resize itself.
                    try:
                        metrics["fps_eval"] = float(_be.time_fps(
                            lambda: model.predict(str(img_path)),
                            n_warmup=3, n_iter=10))
                    except Exception as _e4:
                        print(f"⚠️ rfdetr fps probe skipped: {_e3} / fallback: {_e4}")
        except Exception as _e2:
            print(f"⚠️ rfdetr fps probe skipped: {_e2}")
    except Exception as _e:
        print(f"⚠️ params/fps probe failed: {_e}")

    # ---- Add S/M/L and AR_100 metrics via pycocotools ---------
    # RF-DETR's TensorBoard doesn't log S/M/L, so we compute them from a
    # predictions JSON. The upstream Lightning trainer does NOT write one for
    # the test split, so we generate it ourselves by running model.predict()
    # over every test image and packing results in COCO results format.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bench_eval as _be
        gt_json = RUN_DIR / "_bench_gt_coco.json"
        # RF-DETR uses 1-indexed categories
        _be.yolo_split_to_coco_gt(str(Path(args.data) / "data.yaml"), "test", str(gt_json),
                                 one_indexed_cats=True)

        pred_json = RUN_DIR / "predictions_test.json"
        # (Re)generate when file is missing OR empty/invalid. The previous
        # guard `if not pred_json.is_file()` accepted a stale 2-byte "[]"
        # from a prior failed run and short-circuited everything downstream.
        _need_gen = True
        if pred_json.is_file():
            try:
                with open(pred_json) as _fh: _existing = json.load(_fh)
                _need_gen = not (isinstance(_existing, list) and len(_existing) > 0)
            except Exception:
                _need_gen = True
        if _need_gen:
            # Generate predictions_test.json by predicting on every test image.
            # `supervision.Detections` objects from RF-DETR have .xyxy /
            # .confidence / .class_id and (for seg models) .mask.
            try:
                _cfg = yaml.safe_load(open(str(Path(args.data) / "data.yaml")))
                _root = Path(_cfg.get("path", Path(args.data)))
                if not _root.is_absolute():
                    _root = (Path(args.data) / _root).resolve()
                _split = _cfg.get("test") or _cfg.get("val")
                _img_dir = (_root / _split).resolve() if _split and not Path(_split).is_absolute() else (Path(_split) if _split else None)
                _exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                # Build filename -> image_id map from GT
                with open(gt_json) as f: _gt = json.load(f)
                _name_to_id = {im["file_name"]: im["id"] for im in _gt.get("images", [])}
                imgs = sorted([p for p in _img_dir.iterdir() if p.suffix.lower() in _exts]) if _img_dir and _img_dir.is_dir() else []
                results = []
                try:
                    from pycocotools import mask as _mask_util  # noqa
                    _have_rle = True
                except Exception:
                    _have_rle = False
                # RF-DETR's DINOv2 backbone requires input shape divisible by
                # 24 (patch=12). Passing a raw path/jpg of arbitrary size can
                # crash internally; the fps-probe block above resizes via PIL
                # before calling predict — mirror that here so we don't lose
                # every prediction to a silent shape error.
                try:
                    from PIL import Image as _Image
                    _have_pil = True
                except Exception:
                    _have_pil = False
                _sz = max(24, (int(args.imgsz) // 24) * 24)
                # Lower confidence threshold so we don't drop borderline
                # predictions that COCOeval would still credit at IoU=0.5.
                # RF-DETR's default is ~0.5; COCOeval prefers all detections.
                _predict_kwargs_options = [
                    {"threshold": 0.01},      # supervision-style kwarg
                    {"conf": 0.01},           # alt name
                    {},                        # fallback: model defaults
                ]
                _err_first = None
                _err_count = 0
                _resolved_kwargs = None
                for ip in imgs:
                    img_id = _name_to_id.get(ip.name)
                    if img_id is None: continue
                    # Load + resize via PIL (RF-DETR accepts PIL.Image directly).
                    try:
                        if _have_pil:
                            _img = _Image.open(str(ip)).convert("RGB").resize((_sz, _sz), _Image.BILINEAR)
                            _src = _img
                        else:
                            _src = str(ip)
                    except Exception as _e_load:
                        _err_first = _err_first or f"load:{_e_load}"
                        _err_count += 1
                        continue
                    det = None
                    if _resolved_kwargs is not None:
                        try: det = model.predict(_src, **_resolved_kwargs)
                        except Exception as _e_p:
                            _err_first = _err_first or f"predict:{_e_p}"
                            _err_count += 1
                            continue
                    else:
                        # First image: discover which kwarg name works.
                        for _kw in _predict_kwargs_options:
                            try:
                                det = model.predict(_src, **_kw)
                                _resolved_kwargs = _kw; break
                            except TypeError:
                                continue
                            except Exception as _e_p:
                                _err_first = _err_first or f"predict:{_e_p}"
                                break
                        if det is None:
                            _err_count += 1
                            continue
                    xyxy = getattr(det, "xyxy", None)
                    conf = getattr(det, "confidence", None)
                    cls  = getattr(det, "class_id", None)
                    masks = getattr(det, "mask", None)
                    if xyxy is None or len(xyxy) == 0: continue
                    # Predictions are in resized (sz,sz) coords; scale back to
                    # original image size so they line up with GT boxes.
                    try:
                        _orig_w, _orig_h = (_Image.open(str(ip)).size if _have_pil else (_sz, _sz))
                    except Exception:
                        _orig_w, _orig_h = (_sz, _sz)
                    _sx = float(_orig_w) / float(_sz)
                    _sy = float(_orig_h) / float(_sz)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                        x1, x2 = x1 * _sx, x2 * _sx
                        y1, y2 = y1 * _sy, y2 * _sy
                        pred_cat = 1 if len(CLASSES) == 1 else (int(cls[i]) if cls is not None else 1)
                        entry = {"image_id": int(img_id),
                                 "category_id": pred_cat,
                                 "bbox": [x1, y1, x2 - x1, y2 - y1],
                                 "score": float(conf[i]) if conf is not None else 1.0}
                        if is_seg and masks is not None and _have_rle and i < len(masks):
                            import numpy as _np
                            try:
                                import cv2 as _cv2
                                m_small = (masks[i].astype(_np.uint8) > 0).astype(_np.uint8)
                                m_full = _cv2.resize(m_small, (_orig_w, _orig_h),
                                                     interpolation=_cv2.INTER_NEAREST)
                            except Exception:
                                m_full = (masks[i].astype(_np.uint8) > 0).astype(_np.uint8)
                            m_full = _np.asfortranarray(m_full)
                            rle = _mask_util.encode(m_full)
                            if isinstance(rle.get("counts"), bytes):
                                rle["counts"] = rle["counts"].decode("ascii")
                            entry["segmentation"] = rle
                        results.append(entry)
                with open(pred_json, "w") as f: json.dump(results, f)
                print(f"   📝 Generated predictions_test.json with {len(results)} entries "
                      f"(kwargs={_resolved_kwargs}, errors={_err_count}"
                      + (f", first_error={_err_first}" if _err_first else "") + ")")
            except Exception as _e:
                print(f"⚠️ Could not generate predictions_test.json: {_e}")

        if pred_json.is_file():
            # Fix image_id mismatch if needed (should be no-op since we used GT image_ids).
            fixed_pred = RUN_DIR / "_bench_predictions_fixed.json"
            n_fixed = _be.fix_yolo_pred_image_ids(str(pred_json), str(gt_json), str(fixed_pred))
            if n_fixed > 0:
                print(f"   🔧 Fixed {n_fixed} prediction image_ids to match GT")
                pred_json = fixed_pred
            # Defensive cat-id remap (RF-DETR is 1-indexed, GT we built is 1-indexed,
            # but just in case predict() returns 0-indexed class_id).
            cat_fixed = RUN_DIR / "_bench_predictions_catfixed.json"
            n_cf, cf_mode = _be.remap_pred_categories(str(pred_json), str(gt_json), str(cat_fixed))
            if n_cf > 0:
                print(f"   🔧 Remapped {n_cf} prediction category_ids (mode={cf_mode}) to match GT")
                pred_json = cat_fixed
            # RF-DETR always has box predictions; seg only if is_seg
            bbox_pred = str(pred_json)
            segm_pred = str(pred_json) if is_seg else None
            extra = _be.coco_eval_full(str(gt_json), bbox_pred, segm_pred)
            # Only fill S/M/L and AR_100 unconditionally (don't overwrite the
            # main mAP values from TensorBoard). Fill P/R/F1 from COCOeval PR
            # arrays only when not already provided by TensorBoard harvest —
            # RF-DETR's logger doesn't emit P/R/F1, so this is the only source.
            for k, v in extra.items():
                if k.endswith(("_S", "_M", "_L", "_AR_100")):
                    metrics[k] = v
                elif k.endswith(("_P", "_R", "_F1")):
                    if float(metrics.get(k, 0.0) or 0.0) == 0.0:
                        metrics[k] = v
                elif k.endswith(("_mAP50", "_mAP75", "_mAP50_95")):
                    # In normal training, TensorBoard carries headline mAP and
                    # COCOeval backfills gaps. In eval-only mode there is no
                    # TensorBoard file, so this path supplies all headline mAPs.
                    if float(metrics.get(k, 0.0) or 0.0) == 0.0:
                        metrics[k] = v
            print(f"   📊 Added S/M/L + AR_100 metrics")

            # SAM2-prompted seg AP.  For pure-detector RF-DETR this is the
            # only seg signal; for RF-DETR-seg it is reported in the sam2_*
            # family alongside native seg_* mask-head columns, so a single
            # row carries both native and foundation-prompted seg quality.
            if args.seg_via_sam2:
                try:
                    _cfg_yaml = yaml.safe_load(open(str(Path(args.data) / "data.yaml")))
                    _root = Path(_cfg_yaml.get("path", Path(args.data)))
                    if not _root.is_absolute():
                        _root = (Path(args.data) / _root).resolve()
                    _split = _cfg_yaml.get("test") or _cfg_yaml.get("val")
                    _imgs = (_root / _split).resolve() if _split and not Path(_split).is_absolute() else Path(_split) if _split else None
                except Exception:
                    _imgs = None
                if args.sam2_ckpt and args.sam2_config and _imgs:
                    sam_out = RUN_DIR / "_bench_sam2_seg.json"
                    sam_metrics = _be.sam2_seg_from_boxes(
                        str(_imgs), str(pred_json),
                        args.sam2_ckpt, args.sam2_config,
                        str(gt_json), str(sam_out))
                    for k, v in sam_metrics.items():
                        metrics[k] = v
                else:
                    print("⚠️ --seg_via_sam2 set but --sam2_ckpt/--sam2_config "
                          "missing or test images dir not resolvable; skipping.")
        else:
            print(f"⚠️ predictions_test.json not found at {pred_json}")
    except Exception as _e:
        print(f"⚠️ S/M/L metrics failed: {_e}")

    print("\n--- Final test metrics (TensorBoard) ---")
    for k, v in metrics.items(): print(f"  {k:<14} {v:.4f}")

    # Report the actual micro-batch + realised effective batch so the bench CSV
    # shows the real per-model batch. When micro-bs is "auto" (fractional), the
    # literal value is chosen inside RF-DETR and unknown here, so we leave both
    # for the orchestrator to fall back on its resolved values.
    if isinstance(bs, int):
        metrics["batch_sz"] = int(bs)
        _resolved_ga = int(train_kwargs.get("grad_accum_steps", 1) or 1)
        metrics["batch_effective"] = int(bs * _resolved_ga)

    try:
        PRIMARY.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(RUN_DIR), str(PRIMARY), dirs_exist_ok=True)
        with open(PRIMARY / 'results.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(metrics.keys())); w.writeheader(); w.writerow(metrics)
        print(f"✅ Results copied to: {PRIMARY}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

    print("🎉 Complete!")
