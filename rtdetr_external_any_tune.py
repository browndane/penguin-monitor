# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
# External RT-DETR wrapper for versions that are not loadable by Ultralytics.
#
# rtdetr-2: Hugging Face Transformers RTDetrV2ForObjectDetection; supports
#           smoke fine-tuning and COCOeval on the YOLO test split.
# rtdetr-3: official RT-DETRv3 is PaddleDetection-only. In this Python 3.14
#           venv Paddle is unavailable, so the runnable path is explicitly
#           marked as a PyTorch compatibility route using the upstream v3 config
#           metadata plus converted .pdparams tensors.
# rtdetr-4: downloads the official RT-DETRv4 checkpoint to detr_weights when
#           gdown is available, then smoke fine-tunes and evaluates.

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from bench_schema import zero_metrics

# ---------------------------------------------------------------------------
# Best-checkpoint selection, shared by all three routes.
#
# Every route used to keep only weights/last.pth and score the in-memory model
# at the end of training, so the published number was whatever the last epoch
# happened to be. yolo_any_tune and rtdetr_any_tune both publish their best.pt,
# so the bench was comparing a best checkpoint against a last epoch. These two
# helpers close that gap with one implementation rather than three: the periodic
# COCO eval each route already runs supplies the selection signal.
# ---------------------------------------------------------------------------
BENCH_BEST_CKPT = "best.pth"
BENCH_BEST_METRIC = "box_mAP50_95"


def bench_track_best(run_dir, model, ep_metrics, best_state, source_label):
    """Save weights/best.pth whenever a periodic eval improves the fitness."""
    if not ep_metrics:
        return
    import torch
    score = float(ep_metrics.get(BENCH_BEST_METRIC) or 0.0)
    if score <= best_state["score"]:
        return
    best_state["score"] = score
    weights_dir = Path(run_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), BENCH_BEST_METRIC: score,
                "epoch": best_state.get("epoch"),
                "source": f"bench_any_{source_label}"},
               weights_dir / BENCH_BEST_CKPT)
    print(f"🏅 new best {BENCH_BEST_METRIC}={score:.4f}; saved weights/{BENCH_BEST_CKPT}",
          flush=True)


def bench_load_best(run_dir, model, source_label):
    """Load weights/best.pth before the final eval. Returns True when it lands."""
    import torch
    path = Path(run_dir) / "weights" / BENCH_BEST_CKPT
    if not path.is_file():
        print(f"⚠️ {source_label}: no weights/{BENCH_BEST_CKPT}; the reported "
              f"metrics are the LAST epoch, not the selected checkpoint")
        return False
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        print(f"🏅 {source_label}: scoring the selected checkpoint "
              f"({BENCH_BEST_METRIC}={float(state.get(BENCH_BEST_METRIC, 0.0)):.4f})")
        return True
    except Exception as exc:
        print(f"⚠️ {source_label}: best-checkpoint load failed ({exc}); the "
              f"reported metrics are the LAST epoch")
        return False


HOME_BENCH = Path.home() / "bench_any"
WEIGHTS_DIR = HOME_BENCH / "detr_weights"
RUNTIMES_DIR = HOME_BENCH / "runtimes"
RTDETRV4_REPO = RUNTIMES_DIR / "RT-DETRv4"
RTDETRV4_GIT = "https://github.com/RT-DETRs/RT-DETRv4.git"
RTDETRV3_REPO = RUNTIMES_DIR / "RT-DETRv3"
RTDETRV3_GIT = "https://github.com/clxia12/RT-DETRv3.git"


DOWNLOADS = {
    "rtdetr-2": {
        "kind": "hf",
        "model_id": "PekingU/rtdetr_v2_r50vd",
        "local_dir": WEIGHTS_DIR / "rtdetr-2_hf",
    },
    "rtdetr-2-pth": {
        "kind": "local",
        "path": WEIGHTS_DIR / "rtdetrv2_r50vd_6x_coco_ema.pth",
        "variant": "r50",
        "config": "rtv2_r50vd_6x_coco.yml",
    },
    "rtdetr-3": {
        "kind": "gdrive",
        "file_id": "1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk",
        "path": WEIGHTS_DIR / "rtdetrv3_r18vd_6x_coco.pdparams",
        "variant": "r18",
        "config": "rtv2_r18vd_120e_coco.yml",
        "official_config": "rtdetrv3_r18vd_6x_coco.yml",
    },
    "rtdetr-3-r18": {
        "kind": "local",
        "path": WEIGHTS_DIR / "rtdetrv3_r18vd_6x_coco.pdparams",
        "variant": "r18",
        "config": "rtv2_r18vd_120e_coco.yml",
        "official_config": "rtdetrv3_r18vd_6x_coco.yml",
    },
    "rtdetr-3-r34": {
        "kind": "local",
        "path": WEIGHTS_DIR / "rtdetrv3_r34vd_6x.pdparams",
        "variant": "r34",
        "config": "rtv2_r34vd_120e_coco.yml",
        "official_config": "rtdetrv3_r34vd_6x_coco.yml",
    },
    "rtdetr-3-r50": {
        "kind": "local",
        "path": WEIGHTS_DIR / "rtdetrv3_r50vd_6x.pdparams",
        "variant": "r50",
        "config": "rtv2_r50vd_6x_coco.yml",
        "official_config": "rtdetrv3_r50vd_6x_coco.yml",
    },
    "rtdetr-4": {
        "kind": "gdrive",
        "file_id": "1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z",
        "path": WEIGHTS_DIR / "rtdetrv4_s_coco.pth",
        "runtime": "RT-DETRs/RT-DETRv4 PyTorch repo",
    },
}


def _model_key(model: str) -> str:
    stem = Path(str(model)).stem.lower().replace("_", "-").replace("rt-detr", "rtdetr")
    if "rtdetrv2" in stem:
        return "rtdetr-2-pth"
    if "rtdetr-v2" in stem or "rtdetr-2-pth" in stem:
        return "rtdetr-2-pth"
    if "rtdetr-2" in stem:
        return "rtdetr-2"
    if "rtdetrv3" in stem or "rtdetr-v3" in stem or "rtdetr-3" in stem:
        if "r50" in stem:
            return "rtdetr-3-r50"
        if "r34" in stem:
            return "rtdetr-3-r34"
        if "r18" in stem:
            return "rtdetr-3-r18"
        return "rtdetr-3"
    if "rtdetrv4" in stem or "rtdetr-v4" in stem or "rtdetr-4" in stem:
        return "rtdetr-4"
    return stem


def build_run_name(data_dir, model_name):
    return f"{Path(data_dir).resolve().name[:15]}_{Path(model_name).stem}"


def ensure_download(model_key: str) -> str:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    spec = DOWNLOADS[model_key]
    if spec["kind"] == "hf":
        from huggingface_hub import snapshot_download
        local_dir = Path(spec["local_dir"])
        snapshot_download(repo_id=spec["model_id"], local_dir=str(local_dir))
        return str(local_dir)
    if spec["kind"] == "local":
        path = Path(spec["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{model_key} expects local weights at {path}")
        return str(path)
    if spec["kind"] == "url":
        path = Path(spec["path"])
        if not path.is_file() or path.stat().st_size == 0:
            import requests
            tmp = path.with_suffix(path.suffix + ".tmp")
            with requests.get(spec["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(path)
        return str(path)
    if spec["kind"] == "gdrive":
        path = Path(spec["path"])
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
        try:
            import gdown  # type: ignore
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown  # type: ignore
        url = f"https://drive.google.com/uc?id={spec['file_id']}"
        gdown.download(url=url, output=str(path), quiet=False)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"gdown did not create {path}")
        return str(path)
    raise ValueError(model_key)


def ensure_importable(module_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(module_name)
        return
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or module_name])
    __import__(module_name)


def ensure_rtdetrv4_repo() -> Path:
    if (RTDETRV4_REPO / "train.py").is_file() and (RTDETRV4_REPO / "engine").is_dir():
        return RTDETRV4_REPO
    RUNTIMES_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", RTDETRV4_GIT, str(RTDETRV4_REPO)])
    return RTDETRV4_REPO


def ensure_rtdetrv3_repo() -> Path:
    cfg_dir = RTDETRV3_REPO / "configs" / "rtdetrv3"
    if (RTDETRV3_REPO / "tools" / "train.py").is_file() and cfg_dir.is_dir():
        return RTDETRV3_REPO
    RUNTIMES_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", RTDETRV3_GIT, str(RTDETRV3_REPO)])
    return RTDETRV3_REPO


def paddle_available() -> bool:
    try:
        __import__("paddle")
        return True
    except Exception:
        return False


def _load_rtdetrv3_official_config(official_config_name: str) -> tuple[Path, dict]:
    repo = ensure_rtdetrv3_repo()
    cfg_path = repo / "configs" / "rtdetrv3" / official_config_name
    if not cfg_path.is_file():
        raise RuntimeError(f"RT-DETRv3 official config not found: {cfg_path}")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg_path, cfg


def yolo_image_paths(data_dir: Path, split: str):
    cfg = yaml.safe_load(open(data_dir / "data.yaml")) or {}
    root = Path(cfg.get("path", data_dir)).expanduser()
    if not root.is_absolute():
        root = (data_dir / root).resolve()
    rel = cfg.get(split) or cfg.get("valid" if split == "val" else split)
    if not rel:
        return [], []
    img_dir = Path(rel) if Path(rel).is_absolute() else root / rel
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts]) if img_dir.is_dir() else []
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return paths, list(names)


def choose_eval_split(data_dir: Path) -> str:
    try:
        cfg = yaml.safe_load(open(data_dir / "data.yaml")) or {}
    except Exception:
        return "test"
    for split in ("test", "val", "valid"):
        if cfg.get(split):
            return split
    return "test"


def images_dir_for_split(data_dir: Path, split: str) -> Path | None:
    try:
        cfg = yaml.safe_load(open(data_dir / "data.yaml")) or {}
    except Exception:
        return None
    root = Path(cfg.get("path", data_dir)).expanduser()
    if not root.is_absolute():
        root = (data_dir / root).resolve()
    rel = cfg.get(split) or cfg.get("valid" if split == "val" else split)
    if not rel:
        return None
    return Path(rel) if Path(rel).is_absolute() else (root / rel).resolve()


def label_path_for_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def yolo_annotations_for_image(img_path: Path, image_id: int):
    from PIL import Image
    w, h = Image.open(img_path).size
    anns = []
    lbl = label_path_for_image(img_path)
    if not lbl.is_file():
        return anns
    for ann_id, line in enumerate(open(lbl), start=1):
        ps = line.strip().split()
        if len(ps) < 5:
            continue
        coords = [float(x) for x in ps[1:]]
        if len(coords) == 4:
            xc, yc, bw, bh = coords
            x, y, ww, hh = (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]; ys = coords[1::2]
            x, y = min(xs) * w, min(ys) * h
            ww, hh = (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h
        else:
            continue
        if ww <= 1 or hh <= 1:
            continue
        anns.append({"id": ann_id, "image_id": image_id, "category_id": 0,
                     "bbox": [x, y, ww, hh], "area": ww * hh, "iscrowd": 0})
    return anns


class YoloDetectionDataset:
    def __init__(self, data_dir: Path, split: str, processor, aug_pipeline=None):
        self.paths, self.names = yolo_image_paths(data_dir, split)
        self.processor = processor
        # Shared cross-framework Albumentations pipeline (or None). Only applied
        # to the train split — the eval-split dataset is constructed with
        # aug_pipeline=None so val/test images are never augmented.
        self.aug_pipeline = aug_pipeline

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import torch
        import numpy as np
        from PIL import Image
        import bench_eval as _be

        p = self.paths[idx]
        image = Image.open(p).convert("RGB")
        image_id = _be.safe_image_id(p.stem)
        anns = yolo_annotations_for_image(p, image_id)

        # ---- bench: shared Albumentations pre-transform (image + bboxes) ----
        if self.aug_pipeline is not None and anns:
            try:
                import albumentations as A
                bboxes = [a["bbox"] for a in anns]
                cats = [a["category_id"] for a in anns]
                t = A.Compose(
                    self.aug_pipeline,
                    bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"],
                                             min_visibility=0.1),
                )
                arr = np.array(image)
                aug = t(image=arr, bboxes=bboxes, category_ids=cats)
                image = Image.fromarray(aug["image"])
                new_anns = []
                for ann_id, (bb, cid) in enumerate(zip(aug["bboxes"], aug["category_ids"]), start=1):
                    x, y, ww, hh = bb
                    if ww <= 1 or hh <= 1:
                        continue
                    new_anns.append({"id": ann_id, "image_id": image_id, "category_id": int(cid),
                                     "bbox": [float(x), float(y), float(ww), float(hh)],
                                     "area": float(ww * hh), "iscrowd": 0})
                anns = new_anns
            except Exception:
                pass

        enc = self.processor(
            images=image,
            annotations={"image_id": image_id, "annotations": anns},
            return_tensors="pt",
        )
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": enc["labels"][0],
            "path": p,
            "image_id": image_id,
            "orig_size": torch.tensor([image.height, image.width]),
        }


def collate(batch):
    import torch
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
        "paths": [x["path"] for x in batch],
        "image_ids": [x["image_id"] for x in batch],
        "orig_sizes": torch.stack([x["orig_size"] for x in batch]),
    }


def _progress_log_every(total_steps: int) -> int:
    """Pick a sensible step interval for intra-epoch progress logging.

    Aim for ~4-5 progress lines per epoch so the user sees movement without
    flooding the log. Floor at 10 steps so very tiny epochs still log.
    """
    if total_steps <= 0:
        return 50
    # ~5 lines/epoch => total_steps/5; clamp to [10, 2000].
    every = max(10, total_steps // 5)
    return min(every, 2000)


def _log_step(prefix: str, ep: int, total_ep: int, step: int, total_steps: int,
              loss_running: float, ep_t0: float):
    """Print a single intra-epoch progress line with ETA. Flushed immediately."""
    elapsed = time.time() - ep_t0
    rate = step / max(1e-9, elapsed)
    remaining = max(0, total_steps - step)
    eta = remaining / max(1e-9, rate)
    pct = 100.0 * step / max(1, total_steps)
    print(f"  {prefix} ep {ep + 1}/{total_ep} step {step}/{total_steps} ({pct:.0f}%) "
          f"loss={loss_running:.4f} {rate:.2f} it/s elapsed={elapsed:.0f}s eta={eta:.0f}s",
          flush=True)


def _read_class_names(data_dir: Path) -> list:
    """Best-effort class-name list from YOLO data.yaml. Empty list on failure."""
    try:
        with open(Path(data_dir) / "data.yaml") as fh:
            cfg = yaml.safe_load(fh) or {}
        names = cfg.get("names") or []
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        return list(names)
    except Exception:
        return []


def _emit_paper_figures(run_dir: Path, gt_json, pred_json, class_names, title: str):
    """End-of-run paper figures. All failures swallowed so they never break a run."""
    try:
        import bench_plots as _bp
        import bench_eval as _be
        cm = None
        labels = None
        if gt_json is not None and pred_json is not None and class_names:
            try:
                cm, labels = _be.detector_confusion_matrix(
                    str(gt_json), str(pred_json), list(class_names),
                    iou_thr=0.5, conf_thr=0.25,
                )
            except Exception as e:
                print(f"[bench_plots] CM build failed: {e}")
        out = _bp.plot_all(
            run_dir,
            gt_coco_json=str(gt_json) if gt_json is not None else None,
            pred_coco_json=str(pred_json) if pred_json is not None else None,
            cm=cm, class_names=labels, title=title,
        )
        if out:
            print(f"📈 paper figures: {sorted(out.keys())}")
    except Exception as e:
        print(f"[bench_plots] plot_all failed: {e}")


def _resolve_micro_bs_static(args, default: int = 4) -> int:
    """Resolve --batch_sz to an integer without needing a live model.

    Used by config-writing paths (RT-DETRv2 repo / RT-DETRv4) where we can't
    probe VRAM. Fractional values (<1) map to `default` (capped by
    --batch_sz_max). Values >=1 are taken literally (also capped).
    """
    val = float(getattr(args, "batch_sz", 1) or 0)
    cap = int(getattr(args, "batch_sz_max", 16) or 16)
    if val >= 1:
        return max(1, min(int(val), cap))
    return max(1, min(default, cap))


def _resolve_micro_bs(args, model, device, processor) -> int:
    """Resolve --batch_sz into an integer micro-batch.

    - val >= 1   : literal micro-batch (clamped to >=1 and to --batch_sz_max).
    - 0 < val < 1: fraction of GPU VRAM to target; probe by doubling until the
                   next step would exceed (fraction * total_vram) or OOM.
    - val <= 0   : fall back to literal 1.
    """
    import torch
    val = float(getattr(args, "batch_sz", 1) or 0)
    cap = int(getattr(args, "batch_sz_max", 16) or 16)
    if val >= 1:
        return max(1, min(int(val), cap))
    if val <= 0 or device == "cpu" or not torch.cuda.is_available():
        return 1

    # Fractional auto-batch: probe forward+backward at increasing bs.
    total_vram = torch.cuda.get_device_properties(0).total_memory
    target = int(total_vram * val)
    H = W = int(args.imgsz)
    bs = 1
    best = 1
    model_was_training = model.training
    model.train()
    while bs <= cap:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 3, H, W, device=device)
            labels = [{"class_labels": torch.zeros(1, dtype=torch.long, device=device),
                       "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device)} for _ in range(bs)]
            out = model(pixel_values=x, labels=labels)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated()
            if peak > target:
                break
            best = bs
            # Heuristic: project next step's peak; stop if it would blow target.
            if peak * 2 > target:
                break
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    if not model_was_training:
        model.eval()
    torch.cuda.empty_cache()
    print(f"[rtdetr_external] auto micro-batch: bs={best} "
          f"(target={val:.2f}*VRAM, cap={cap})")
    return max(1, best)


def train_v2(args, model_dir: str, run_dir: Path):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
    import bench_eval as _be
    import bench_plots as _bp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    # `--threads 0` means auto, not "no workers" (see yolo_any_tune.py). -1
    # forces single-process. cvBench speed port.
    workers = 0 if args.threads < 0 else (args.threads or min(8, os.cpu_count() or 8))
    processor.size = {"height": int(args.imgsz), "width": int(args.imgsz)}
    model = RTDetrV2ForObjectDetection.from_pretrained(
        model_dir, num_labels=1, ignore_mismatched_sizes=True,
        id2label={0: "object"}, label2id={"object": 0},
    ).to(device)

    # Resume / inference-only: load trained checkpoint from the run dir. The two
    # want different files — a resume must continue from the last epoch, an
    # eval-only pass must score the selected checkpoint.
    _last_ckpt = run_dir / "weights" / "last.pth"
    _best_ckpt = run_dir / "weights" / BENCH_BEST_CKPT
    ckpt_path = _best_ckpt if (args.inference_only and _best_ckpt.is_file()) else _last_ckpt
    if (args.resume or args.inference_only) and ckpt_path.is_file():
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"🔁 loaded trained checkpoint {ckpt_path}")
    elif args.inference_only:
        print(f"⚠️  inference-only: no checkpoint at {ckpt_path}; using pretrained weights")

    # Shared cross-framework Albumentations pipeline (only for train; eval is never augmented).
    _aug_pipe = None
    _aug_status = "native"
    _policy = getattr(args, "aug_policy", "native")
    if _policy in ("off", "fair"):
        try:
            from bench_aug import shared_albumentations_policy
            _aug_pipe = shared_albumentations_policy(_policy, args.imgsz)
            _aug_status = "applied"
        except Exception as _e:
            _aug_status = "unsupported"
            print(f"[bench] rtdetr_external: shared_albumentations_policy import failed: {_e}")
    elif _policy == "native":
        # This loader has no augmentation of its own; inherit Ultralytics' defaults
        # so the native arm is comparable to the other frameworks' native arms.
        try:
            from bench_aug import ultralytics_native_albumentations
            _aug_pipe = ultralytics_native_albumentations() or None
            _aug_status = "ultralytics-native" if _aug_pipe else "unsupported"
        except Exception as _e:
            _aug_status = "unsupported"
            print(f"[bench] rtdetr_external: ultralytics_native_albumentations failed: {_e}")
    print(f"[bench] aug_policy={getattr(args, 'aug_policy', 'native')}: rtdetr_external -> {_aug_status}")
    train_ds = YoloDetectionDataset(Path(args.data), "train", processor, aug_pipeline=_aug_pipe)
    eval_split = choose_eval_split(Path(args.data))
    test_ds = YoloDetectionDataset(Path(args.data), eval_split, processor)
    bs = _resolve_micro_bs(args, model, device, processor)
    accum = max(1, int(getattr(args, "nbs", 0) or 0) // max(1, bs)) if int(getattr(args, "nbs", 0) or 0) > bs else 1
    eff_bs = bs * accum
    eval_every = max(1, int(getattr(args, "eval_every", 10) or 10))
    # Clamp eval_every so short runs still yield ≥4 mAP points on the curve.
    eval_every = min(eval_every, max(1, int(args.epochs) // 4))
    # Best-checkpoint selection state; the periodic eval above is the signal.
    _bench_route = str(args.model)
    _best_state = {"score": -1.0, "epoch": None}
    _no_improve = 0  # bench_any: epochs since the last best-metric improvement (patience)
    print(f"🚂 RT-DETRv2 train epochs={int(args.epochs)} train_images={len(train_ds)} "
          f"eval_split={eval_split} eval_images={len(test_ds)} batch={bs} "
          f"grad_accum={accum} effective_batch={eff_bs} eval_every={eval_every}", flush=True)

    gt_json = run_dir / "_bench_gt_coco.json"
    _be.yolo_split_to_coco_gt(str(Path(args.data) / "data.yaml"), eval_split, str(gt_json), one_indexed_cats=False)
    pred_json = run_dir / f"predictions_{eval_split}.json"

    val_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=workers,
                            persistent_workers=workers > 0, collate_fn=collate) \
        if len(test_ds) else None

    wb_run = _bp.maybe_init_wandb(
        config={"framework": "rtdetr_external", "route": "v2",
                "epochs": int(args.epochs), "batch": bs, "effective_batch": eff_bs,
                "imgsz": int(args.imgsz), "lr": float(args.lr)},
        extra_tags=["rtdetr-2"],
    )

    def _run_full_eval():
        """COCO eval over the val split. Writes pred_json and returns metric dict + fps."""
        preds_ = []
        if val_loader is None:
            with open(pred_json, "w") as fh: json.dump([], fh)
            return zero_metrics(), 0.0
        model.eval()
        t0 = time.time()
        n_img = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(pixel_values=batch["pixel_values"].to(device))
                results = processor.post_process_object_detection(
                    out, threshold=float(args.conf), target_sizes=batch["orig_sizes"].to(device)
                )
                for image_id, res in zip(batch["image_ids"], results):
                    for score, box in zip(res["scores"].detach().cpu().tolist(),
                                          res["boxes"].detach().cpu().tolist()):
                        x1, y1, x2, y2 = box
                        if x2 <= x1 or y2 <= y1:
                            continue
                        preds_.append({"image_id": int(image_id), "category_id": 0,
                                       "bbox": [x1, y1, x2 - x1, y2 - y1],
                                       "score": float(score)})
                n_img += len(results)
        fps_ = n_img / max(1e-9, time.time() - t0)
        with open(pred_json, "w") as fh: json.dump(preds_, fh)
        m_ = zero_metrics()
        m_.update(_be.coco_eval_full(str(gt_json), str(pred_json), None))
        return m_, fps_

    def _val_loss_pass():
        """Forward-only loss pass over the val split. Returns mean loss or None."""
        if val_loader is None:
            return None
        model.eval()
        total_v = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                labels_ = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in lab.items()}
                           for lab in batch["labels"]]
                out = model(pixel_values=batch["pixel_values"].to(device), labels=labels_)
                total_v += float(out.loss.detach().cpu())
                n_v += 1
        return total_v / max(1, n_v) if n_v else None

    if not args.inference_only and int(args.epochs) > 0 and len(train_ds):
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers,
                            persistent_workers=workers > 0, pin_memory=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        model.train()
        total_steps = len(loader)
        log_every = _progress_log_every(total_steps)
        print(f"   train_loader: {total_steps} steps/epoch, logging every {log_every} steps", flush=True)
        for ep in range(int(args.epochs)):
            ep_t0 = time.time()
            total = 0.0
            n_steps = 0
            model.train()
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                labels = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in lab.items()}
                          for lab in batch["labels"]]
                out = model(pixel_values=batch["pixel_values"].to(device), labels=labels)
                loss = out.loss / accum
                loss.backward()
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                total += float(loss.detach().cpu()) * accum
                n_steps += 1
                if n_steps % log_every == 0 or n_steps == total_steps:
                    _log_step("RT-DETRv2", ep, int(args.epochs), n_steps, total_steps,
                              total / max(1, n_steps), ep_t0)
            # Flush any trailing micro-batches.
            if n_steps % accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            train_loss = total / max(1, n_steps)
            val_loss = _val_loss_pass()
            # Periodic full COCO eval (and always on the final epoch).
            do_full = ((ep + 1) % eval_every == 0) or (ep + 1 == int(args.epochs))
            ep_metrics = None
            if do_full:
                ep_metrics, _ = _run_full_eval()
                _best_state["epoch"] = ep + 1
                _prev_score = _best_state["score"]
                bench_track_best(run_dir, model, ep_metrics, _best_state, _bench_route)
                # bench_any: wire --patience (was a dead arg). Count full-eval
                # epochs since the last improvement on BENCH_BEST_METRIC and stop
                # early once patience is exceeded — matches YOLO's epoch-based
                # patience semantics. A non-positive patience disables it.
                if _best_state["score"] > _prev_score:
                    _no_improve = 0
                else:
                    _no_improve += 1
                    if int(args.patience) > 0 and _no_improve >= int(args.patience):
                        print(f"🛑 early stopping: no {BENCH_BEST_METRIC} improvement "
                              f"for {_no_improve} eval epochs (patience={int(args.patience)})")
                        break
            elapsed = time.time() - ep_t0
            print(f"epoch {ep + 1}/{int(args.epochs)}: steps={n_steps} "
                  f"train_loss={train_loss:.4f}"
                  + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
                  + (f" box_mAP50_95={ep_metrics.get('box_mAP50_95', 0.0):.4f}" if ep_metrics else "")
                  + f" elapsed={elapsed:.1f}s",
                  flush=True)
            row = {"epoch": ep + 1, "elapsed_s": elapsed, "lr": float(args.lr),
                   "train_loss": train_loss, "val_loss": val_loss}
            if ep_metrics:
                for k in ("box_mAP50", "box_mAP50_95", "box_mAP75", "seg_mAP50", "seg_mAP50_95", "seg_mAP75"):
                    v = ep_metrics.get(k)
                    if v is not None: row[k] = float(v)
            _bp.append_history(run_dir, row)
            _bp.wandb_log_safe(wb_run, row, step=ep + 1)

    # Persist the last epoch. This route read weights/last.pth for --resume at
    # L563 but never wrote it, so a resume silently restarted from the
    # pretrained weights and reported the shortfall nowhere. Routes 2 and 3
    # already do this; route 1 was the odd one out.
    #
    # Guarded on inference_only: an eval-only pass has best.pth loaded into
    # `model`, so an unguarded save would overwrite the real last epoch with the
    # selected checkpoint and destroy the only copy of the final weights.
    if not args.inference_only:
        _weights_dir = run_dir / "weights"
        _weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "source": "bench_any_rtdetrv2_hf"}, _weights_dir / "last.pth")

    # Final eval (also writes pred_json) — needed for metrics + final figures.
    # Scored on the selected checkpoint, not the last epoch.
    bench_load_best(run_dir, model, _bench_route)
    metrics, fps = _run_full_eval()
    if args.seg_via_sam2:
        img_dir = images_dir_for_split(Path(args.data), eval_split)
        if args.sam2_ckpt and args.sam2_config and img_dir and img_dir.is_dir():
            print("🎯 SAM2 masks from RT-DETRv2 boxes")
            sam_out = run_dir / "_bench_sam2_seg.json"
            sam_metrics = _be.sam2_seg_from_boxes(
                str(img_dir), str(pred_json),
                args.sam2_ckpt, args.sam2_config,
                str(gt_json), str(sam_out),
            )
            metrics.update(sam_metrics)
        else:
            print("⚠️ --seg_via_sam2 set but SAM2 paths or eval image dir are missing; skipping.")
    metrics["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    metrics["fps_eval"] = fps
    print(f"📊 RT-DETRv2 box_mAP50={metrics.get('box_mAP50', 0.0):.4f} "
          f"box_mAP50_95={metrics.get('box_mAP50_95', 0.0):.4f} "
          f"seg_mAP50_95={metrics.get('seg_mAP50_95', 0.0):.4f} fps={fps:.1f}")

    # Paper figures (vector PDFs) + W&B summary.
    _emit_paper_figures(run_dir, gt_json, pred_json,
                        _read_class_names(Path(args.data)) or ["object"],
                        title=f"RT-DETRv2 ({Path(args.data).name})")
    _bp.wandb_log_safe(wb_run, {"final/" + k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
    _bp.wandb_finish_safe(wb_run)
    return metrics


def _write_rtdetrv4_config(repo: Path, data_dir: Path, run_dir: Path, args, eval_split: str) -> tuple[Path, Path]:
    import bench_eval as _be

    ann_dir = run_dir / "_rtdetrv4_coco"
    ann_dir.mkdir(parents=True, exist_ok=True)
    train_gt = ann_dir / "instances_train.json"
    eval_gt = ann_dir / f"instances_{eval_split}.json"
    _be.yolo_split_to_coco_gt(str(data_dir / "data.yaml"), "train", str(train_gt), one_indexed_cats=False)
    _be.yolo_split_to_coco_gt(str(data_dir / "data.yaml"), eval_split, str(eval_gt), one_indexed_cats=False)

    train_paths, names = yolo_image_paths(data_dir, "train")
    eval_paths, _ = yolo_image_paths(data_dir, eval_split)
    if not train_paths:
        raise RuntimeError("RT-DETRv4: no train images found")
    if not eval_paths:
        raise RuntimeError(f"RT-DETRv4: no {eval_split} images found")

    bs = _resolve_micro_bs_static(args)
    imgsz = int(args.imgsz)
    cfg = {
        "__include__": [str(repo / "configs/rtv4/rtv4_hgnetv2_s_coco.yml")],
        "output_dir": str(run_dir / "_rtdetrv4_out"),
        "num_classes": max(1, len(names)),
        "remap_mscoco_category": False,
        "eval_spatial_size": [imgsz, imgsz],
        "use_amp": False,
        "use_ema": False,
        "sync_bn": False,
        "epoches": int(args.epochs),
        "print_freq": 999999,
        "HGNetv2": {"pretrained": False},
        "train_dataloader": {
            "total_batch_size": bs,
            "num_workers": 0,
            "drop_last": False,
            "shuffle": True,
            "dataset": {
                "img_folder": str(train_paths[0].parent),
                "ann_file": str(train_gt),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [imgsz, imgsz]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                    "policy": None,
                },
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": imgsz,
                "base_size_repeat": None,
                "stop_epoch": None,
                "mixup_prob": 0.0,
                "mixup_epochs": [0, 0],
            },
        },
        "val_dataloader": {
            "total_batch_size": bs,
            "num_workers": 0,
            "drop_last": False,
            "shuffle": False,
            "dataset": {
                "img_folder": str(eval_paths[0].parent),
                "ann_file": str(eval_gt),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [imgsz, imgsz]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ],
                    "policy": None,
                },
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": imgsz,
                "base_size_repeat": None,
                "stop_epoch": None,
                "mixup_prob": 0.0,
                "mixup_epochs": [0, 0],
            },
        },
    }
    cfg_path = run_dir / "_rtdetrv4_config.yml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return cfg_path, eval_gt


def _load_matching_state(model, ckpt_path: str) -> tuple[int, int]:
    import torch
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "ema" in state and isinstance(state["ema"], dict):
        src = state["ema"].get("module", state["ema"])
    elif isinstance(state, dict) and "model" in state:
        src = state["model"]
    else:
        src = state
    cur = model.state_dict()
    matched = {k: v for k, v in src.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    model.load_state_dict(matched, strict=False)
    return len(matched), len(cur)


def _write_rtdetrv2_repo_config(
    repo: Path,
    data_dir: Path,
    run_dir: Path,
    args,
    eval_split: str,
    variant: str,
    config_name: str,
) -> tuple[Path, Path]:
    import bench_eval as _be

    ann_dir = run_dir / "_rtdetrv2_repo_coco"
    ann_dir.mkdir(parents=True, exist_ok=True)
    train_gt = ann_dir / "instances_train.json"
    eval_gt = ann_dir / f"instances_{eval_split}.json"
    _be.yolo_split_to_coco_gt(str(data_dir / "data.yaml"), "train", str(train_gt), one_indexed_cats=False)
    _be.yolo_split_to_coco_gt(str(data_dir / "data.yaml"), eval_split, str(eval_gt), one_indexed_cats=False)

    train_paths, names = yolo_image_paths(data_dir, "train")
    eval_paths, _ = yolo_image_paths(data_dir, eval_split)
    if not train_paths:
        raise RuntimeError("RT-DETR repo route: no train images found")
    if not eval_paths:
        raise RuntimeError(f"RT-DETR repo route: no {eval_split} images found")

    depth = {"r18": 18, "r34": 34, "r50": 50}[variant]
    bs = _resolve_micro_bs_static(args)
    imgsz = int(args.imgsz)
    dataloader_common = {
        "total_batch_size": bs,
        "num_workers": 0,
        "drop_last": False,
        "collate_fn": {
            "type": "BatchImageCollateFunction",
            "base_size": imgsz,
            "base_size_repeat": None,
            "stop_epoch": None,
            "mixup_prob": 0.0,
            "mixup_epochs": [0, 0],
        },
    }
    cfg = {
        "__include__": [str(repo / "configs/rtv2" / config_name)],
        "output_dir": str(run_dir / "_rtdetrv2_repo_out"),
        "num_classes": max(1, len(names)),
        "remap_mscoco_category": False,
        "eval_spatial_size": [imgsz, imgsz],
        "use_amp": False,
        "use_ema": False,
        "sync_bn": False,
        "epoches": int(args.epochs),
        "print_freq": 999999,
        "PResNet": {
            "depth": depth,
            "pretrained": False,
            "freeze_at": -1,
            "freeze_norm": False,
        },
        "train_dataloader": {
            **dataloader_common,
            "shuffle": True,
            "dataset": {
                "img_folder": str(train_paths[0].parent),
                "ann_file": str(train_gt),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [imgsz, imgsz]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                    "policy": None,
                },
            },
        },
        "val_dataloader": {
            **dataloader_common,
            "shuffle": False,
            "dataset": {
                "img_folder": str(eval_paths[0].parent),
                "ann_file": str(eval_gt),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [imgsz, imgsz]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ],
                    "policy": None,
                },
            },
        },
    }
    cfg_path = run_dir / f"_rtdetrv2_repo_{variant}_config.yml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return cfg_path, eval_gt


def _sanitize_rt_repo_dataloaders(cfg) -> None:
    for loader_name in ("train_dataloader", "val_dataloader"):
        loader_cfg = cfg.yaml_cfg.get(loader_name)
        if not isinstance(loader_cfg, dict):
            continue
        collate_cfg = loader_cfg.get("collate_fn")
        if isinstance(collate_cfg, dict):
            collate_cfg.pop("scales", None)


def _rtdetrv3_key_candidates(key: str) -> list[str]:
    key = key.replace("._mean", ".running_mean").replace("._variance", ".running_var")
    out = [key]
    for stage, idx in (("res2", 0), ("res3", 1), ("res4", 2), ("res5", 3)):
        prefix = f"backbone.{stage}."
        if prefix in key:
            mapped = key.replace(prefix, f"backbone.res_layers.{idx}.blocks.")
            mapped = re.sub(rf"{stage}([a-z])\.", lambda m: str(ord(m.group(1)) - ord("a")) + ".", mapped)
            out.append(mapped)
    if key.startswith("transformer."):
        suffix = key[len("transformer."):]
        out.append("decoder." + suffix)
        if key.startswith("transformer.decoder."):
            out.append("decoder.decoder." + key[len("transformer.decoder."):])
        if key.startswith("transformer.dec_bbox_head."):
            out.append("decoder.dec_bbox_head." + key[len("transformer.dec_bbox_head."):])
        if key.startswith("transformer.dec_score_head."):
            out.append("decoder.dec_score_head." + key[len("transformer.dec_score_head."):])
        if key.startswith("transformer.enc_bbox_head.0."):
            out.append("decoder.enc_bbox_head." + key[len("transformer.enc_bbox_head.0."):])
        if key.startswith("transformer.query_pos_head."):
            out.append("decoder.query_pos_head." + key[len("transformer.query_pos_head."):])
    return list(dict.fromkeys(out))


def _load_rtdetrv3_paddle_state(model, ckpt_path: str) -> tuple[int, int, int]:
    import torch

    with open(ckpt_path, "rb") as fh:
        state = pickle.load(fh, encoding="latin1")
    cur = model.state_dict()
    matched = {}
    transposed = 0
    for src_key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        tensor = torch.as_tensor(value)
        for dst_key in _rtdetrv3_key_candidates(src_key):
            if dst_key not in cur:
                continue
            if tuple(cur[dst_key].shape) == tuple(tensor.shape):
                matched[dst_key] = tensor.to(dtype=cur[dst_key].dtype)
                break
            if tensor.ndim == 2 and tuple(cur[dst_key].shape) == tuple(tensor.t().shape):
                matched[dst_key] = tensor.t().contiguous().to(dtype=cur[dst_key].dtype)
                transposed += 1
                break
    model.load_state_dict(matched, strict=False)
    return len(matched), len(cur), transposed


def _fix_rtdetrv4_eval_tensors(model, device):
    enc = getattr(model, "encoder", None)
    if enc is not None:
        for idx in getattr(enc, "use_encoder_idx", []):
            name = f"pos_embed{idx}"
            value = getattr(enc, name, None)
            if hasattr(value, "to"):
                setattr(enc, name, value.to(device))
    dec = getattr(model, "decoder", None)
    for name in ("anchors", "valid_mask"):
        value = getattr(dec, name, None)
        if hasattr(value, "to"):
            setattr(dec, name, value.to(device))


def train_rtdetrv2_repo(
    args,
    ckpt_path: str,
    run_dir: Path,
    variant: str,
    config_name: str,
    source_kind: str,
    official_config_name: str | None = None,
):
    import torch
    import bench_eval as _be

    if source_kind == "paddle":
        official_cfg_path, official_cfg = _load_rtdetrv3_official_config(official_config_name or "")
        official_layers = (official_cfg.get("RTDETRTransformerv3") or {}).get("num_decoder_layers")
        official_noises = (official_cfg.get("RTDETRTransformerv3") or {}).get("num_noises")
        official_resnet = (official_cfg.get("ResNet") or {}).get("depth")
        print(f"RT-DETRv3 official config: {official_cfg_path}")
        if not paddle_available():
            print(
                "⚠️ Official RT-DETRv3 runtime is PaddleDetection, but Paddle is not importable in this "
                f"Python {sys.version_info.major}.{sys.version_info.minor} venv. "
                "Using PyTorch compatibility route instead: converted .pdparams weights loaded into the "
                "RT-DETRv2-style PyTorch runtime. This is runnable and comparable, but not the exact "
                "official RT-DETRv3 architecture with DINOv3Head/PPYOLOEHead/o2m branch."
            )
        print(
            "RT-DETRv3 config metadata: "
            f"ResNet depth={official_resnet or variant}, decoder_layers={official_layers}, "
            f"num_noises={official_noises}"
        )

    ensure_importable("calflops")
    repo = ensure_rtdetrv4_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from engine.core import YAMLConfig

    data_dir = Path(args.data).resolve()
    eval_split = choose_eval_split(data_dir)
    cfg_path, gt_json = _write_rtdetrv2_repo_config(repo, data_dir, run_dir, args, eval_split, variant, config_name)
    cfg = YAMLConfig(str(cfg_path))
    _sanitize_rt_repo_dataloaders(cfg)
    model = cfg.model
    if source_kind == "paddle":
        matched, total, transposed = _load_rtdetrv3_paddle_state(model, ckpt_path)
        print(f"RT-DETRv3-{variant.upper()} transferred {matched}/{total} tensors from {ckpt_path} ({transposed} transposed)")
    else:
        matched, total = _load_matching_state(model, ckpt_path)
        print(f"RT-DETRv2-{variant.upper()} transferred {matched}/{total} tensors from {ckpt_path}")

    # Inference-only: overwrite pretrained weights with the trained
    # checkpoint from the run dir (same pattern as train_v4 L1234-1241).
    if args.inference_only:
        # Score the selected checkpoint when one exists, else the last epoch.
        trained_ckpt = next((p for p in (run_dir / "weights" / BENCH_BEST_CKPT,
                                         run_dir / "weights" / "last.pth")
                             if p.is_file()), run_dir / "weights" / "last.pth")
        if trained_ckpt.is_file():
            state = torch.load(str(trained_ckpt), map_location="cpu")
            model.load_state_dict(state["model"], strict=False)
            print(f"🔁 inference-only: loaded trained checkpoint {trained_ckpt}")
        else:
            print(f"⚠️  inference-only: no trained checkpoint at {trained_ckpt}; using pretrained weights")

    import bench_plots as _bp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    lr = float(args.lr) if float(args.lr) > 0 else 1e-4
    train_paths, _ = yolo_image_paths(data_dir, "train")
    eval_paths, _ = yolo_image_paths(data_dir, eval_split)
    route_name = "RT-DETRv3-" + variant.upper() if source_kind == "paddle" else "RT-DETRv2-" + variant.upper()
    bs = _resolve_micro_bs_static(args)
    eval_every = max(1, int(getattr(args, "eval_every", 10) or 10))
    # Clamp eval_every so short runs still yield ≥4 mAP points on the curve.
    eval_every = min(eval_every, max(1, int(args.epochs) // 4))
    # Best-checkpoint selection state; the periodic eval above is the signal.
    _bench_route = str(args.model)
    _best_state = {"score": -1.0, "epoch": None}
    _no_improve = 0  # bench_any: epochs since the last best-metric improvement (patience)
    print(f"🚂 {route_name} train epochs={int(args.epochs)} train_images={len(train_paths)} "
          f"eval_split={eval_split} eval_images={len(eval_paths)} batch={bs} lr={lr:g} "
          f"eval_every={eval_every}", flush=True)

    pred_json = run_dir / f"predictions_{eval_split}.json"
    val_loader = cfg.val_dataloader

    wb_run = _bp.maybe_init_wandb(
        config={"framework": "rtdetr_external", "route": "v2_repo" if source_kind != "paddle" else "v3_compat",
                "variant": variant, "epochs": int(args.epochs), "batch": bs,
                "imgsz": int(args.imgsz), "lr": lr},
        extra_tags=[route_name],
    )

    def _run_full_eval():
        preds_ = []
        model.eval()
        _fix_rtdetrv4_eval_tensors(model, device)
        postprocessor.eval()
        t0 = time.time()
        n_img = 0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(device)
                targets_d = [{k: v.to(device) for k, v in t.items()} for t in targets]
                orig_sizes = torch.stack([t["orig_size"] for t in targets_d], dim=0).to(device)
                outputs = model(samples)
                results = postprocessor(outputs, orig_sizes)
                for target, result in zip(targets_d, results):
                    image_id = int(target["image_id"].item())
                    scores = result["scores"].detach().cpu().tolist()
                    labels_ = result["labels"].detach().cpu().tolist()
                    boxes = result["boxes"].detach().cpu().tolist()
                    for score, label, box in zip(scores, labels_, boxes):
                        if float(score) < float(args.conf):
                            continue
                        x1, y1, x2, y2 = box
                        if x2 <= x1 or y2 <= y1:
                            continue
                        preds_.append({"image_id": image_id, "category_id": int(label),
                                       "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                       "score": float(score)})
                n_img += len(results)
        fps_ = n_img / max(1e-9, time.time() - t0)
        with open(pred_json, "w") as fh: json.dump(preds_, fh)
        m_ = zero_metrics()
        m_.update(_be.coco_eval_full(str(gt_json), str(pred_json), None))
        return m_, fps_

    def _val_loss_pass(ep_idx: int):
        """Forward-only loss pass on val. Uses the repo's criterion."""
        try:
            model.eval()
            criterion.eval() if hasattr(criterion, "eval") else None
            total_v = 0.0
            n_v = 0
            with torch.no_grad():
                for step_v, (samples, targets) in enumerate(val_loader):
                    samples = samples.to(device)
                    targets_d = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    outputs = model(samples, targets=targets_d)
                    loss_dict = criterion(
                        outputs, targets_d,
                        epoch=ep_idx, step=step_v,
                        global_step=ep_idx * max(1, len(val_loader)) + step_v,
                        epoch_step=max(1, len(val_loader)),
                    )
                    total_v += float(sum(v.detach().cpu() for v in loss_dict.values()))
                    n_v += 1
            return total_v / max(1, n_v) if n_v else None
        except Exception as e:
            print(f"[bench_plots] val_loss skipped ({route_name}): {e}")
            return None

    if not args.inference_only and int(args.epochs) > 0:
        train_loader = cfg.train_dataloader
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
        total_steps = len(train_loader)
        log_every = _progress_log_every(total_steps)
        print(f"   train_loader: {total_steps} steps/epoch, logging every {log_every} steps", flush=True)
        for ep in range(int(args.epochs)):
            ep_t0 = time.time()
            train_loader.set_epoch(ep)
            model.train()
            criterion.train()
            total_loss = 0.0
            n_steps = 0
            for step, (samples, targets) in enumerate(train_loader):
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(samples, targets=targets)
                loss_dict = criterion(
                    outputs, targets,
                    epoch=ep, step=step,
                    global_step=ep * len(train_loader) + step,
                    epoch_step=len(train_loader),
                )
                loss = sum(loss_dict.values())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.detach().cpu())
                n_steps += 1
                if n_steps % log_every == 0 or n_steps == total_steps:
                    _log_step(route_name, ep, int(args.epochs), n_steps, total_steps,
                              total_loss / max(1, n_steps), ep_t0)
            train_loss = total_loss / max(1, n_steps)
            val_loss = _val_loss_pass(ep)
            do_full = ((ep + 1) % eval_every == 0) or (ep + 1 == int(args.epochs))
            ep_metrics = None
            if do_full:
                ep_metrics, _ = _run_full_eval()
                _best_state["epoch"] = ep + 1
                _prev_score = _best_state["score"]
                bench_track_best(run_dir, model, ep_metrics, _best_state, _bench_route)
                # bench_any: wire --patience (was a dead arg). Count full-eval
                # epochs since the last improvement on BENCH_BEST_METRIC and stop
                # early once patience is exceeded — matches YOLO's epoch-based
                # patience semantics. A non-positive patience disables it.
                if _best_state["score"] > _prev_score:
                    _no_improve = 0
                else:
                    _no_improve += 1
                    if int(args.patience) > 0 and _no_improve >= int(args.patience):
                        print(f"🛑 early stopping: no {BENCH_BEST_METRIC} improvement "
                              f"for {_no_improve} eval epochs (patience={int(args.patience)})")
                        break
            elapsed = time.time() - ep_t0
            print(f"epoch {ep + 1}/{int(args.epochs)}: steps={n_steps} "
                  f"train_loss={train_loss:.4f}"
                  + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
                  + (f" box_mAP50_95={ep_metrics.get('box_mAP50_95', 0.0):.4f}" if ep_metrics else "")
                  + f" elapsed={elapsed:.1f}s",
                  flush=True)
            row = {"epoch": ep + 1, "elapsed_s": elapsed, "lr": lr,
                   "train_loss": train_loss, "val_loss": val_loss}
            if ep_metrics:
                for k in ("box_mAP50", "box_mAP50_95", "box_mAP75", "seg_mAP50", "seg_mAP50_95", "seg_mAP75"):
                    v = ep_metrics.get(k)
                    if v is not None: row[k] = float(v)
            _bp.append_history(run_dir, row)
            _bp.wandb_log_safe(wb_run, row, step=ep + 1)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    source_label = "rtdetrv3_pytorch_compat" if source_kind == "paddle" else "rtdetrv2_pytorch_repo"
    # Only after real training. Under --inference_only, L1101-1109 has already
    # loaded best.pth into `model`, so saving here would overwrite the last
    # epoch with the selected checkpoint and lose the final weights for good.
    if not args.inference_only:
        torch.save({"model": model.state_dict(), "source": f"bench_any_{source_label}"}, weights_dir / "last.pth")

    print(f"🧪 {route_name} eval on {eval_split}: {len(eval_paths)} images")
    bench_load_best(run_dir, model, route_name)
    metrics, fps = _run_full_eval()
    if args.seg_via_sam2:
        img_dir = images_dir_for_split(data_dir, eval_split)
        if args.sam2_ckpt and args.sam2_config and img_dir and img_dir.is_dir():
            print(f"🎯 SAM2 masks from {route_name} boxes")
            sam_out = run_dir / "_bench_sam2_seg.json"
            metrics.update(_be.sam2_seg_from_boxes(
                str(img_dir), str(pred_json),
                args.sam2_ckpt, args.sam2_config,
                str(gt_json), str(sam_out),
            ))
        else:
            print("⚠️ --seg_via_sam2 set but SAM2 paths or eval image dir are missing; skipping.")
    metrics["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    metrics["fps_eval"] = fps
    print(f"📊 {route_name} box_mAP50={metrics.get('box_mAP50', 0.0):.4f} "
          f"box_mAP50_95={metrics.get('box_mAP50_95', 0.0):.4f} "
          f"seg_mAP50_95={metrics.get('seg_mAP50_95', 0.0):.4f} fps={fps:.1f}")
    if source_kind == "paddle":
        (run_dir / "RUNTIME_NOTE.txt").write_text(
            "RT-DETRv3 official upstream is PaddleDetection. This run used the PyTorch compatibility "
            "route because Paddle is not available in the current Python 3.14 venv.\n"
            f"official_config={official_config_name}\n"
            f"checkpoint={ckpt_path}\n"
        )

    _emit_paper_figures(run_dir, gt_json, pred_json,
                        _read_class_names(data_dir) or ["object"],
                        title=f"{route_name} ({data_dir.name})")
    _bp.wandb_log_safe(wb_run, {"final/" + k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
    _bp.wandb_finish_safe(wb_run)
    return metrics


def train_v4(args, ckpt_path: str, run_dir: Path):
    import torch
    import bench_eval as _be
    import bench_plots as _bp

    ensure_importable("calflops")
    repo = ensure_rtdetrv4_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from engine.core import YAMLConfig

    data_dir = Path(args.data).resolve()
    eval_split = choose_eval_split(data_dir)
    cfg_path, gt_json = _write_rtdetrv4_config(repo, data_dir, run_dir, args, eval_split)
    cfg = YAMLConfig(str(cfg_path))
    _sanitize_rt_repo_dataloaders(cfg)
    model = cfg.model
    matched, total = _load_matching_state(model, ckpt_path)
    print(f"RT-DETRv4-S transferred {matched}/{total} tensors from {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)

    # inference-only: load trained checkpoint from the run dir, preferring the
    # selected checkpoint over the last epoch.
    ckpt_path = next((p for p in (run_dir / "weights" / BENCH_BEST_CKPT,
                                  run_dir / "weights" / "last.pth")
                      if p.is_file()), run_dir / "weights" / "last.pth")
    if args.inference_only and ckpt_path.is_file():
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"🔁 inference-only: loading trained checkpoint {ckpt_path}")
    elif args.inference_only:
        print(f"⚠️  inference-only: no checkpoint at {ckpt_path}; using pretrained weights")

    bs = _resolve_micro_bs_static(args)
    lr = float(args.lr) if float(args.lr) > 0 else 5e-5
    train_paths, _ = yolo_image_paths(data_dir, "train")
    eval_paths, _ = yolo_image_paths(data_dir, eval_split)
    eval_every = max(1, int(getattr(args, "eval_every", 10) or 10))
    # Clamp eval_every so short runs still yield ≥4 mAP points on the curve.
    eval_every = min(eval_every, max(1, int(args.epochs) // 4))
    # Best-checkpoint selection state; the periodic eval above is the signal.
    _bench_route = str(args.model)
    _best_state = {"score": -1.0, "epoch": None}
    _no_improve = 0  # bench_any: epochs since the last best-metric improvement (patience)
    print(f"🚂 RT-DETRv4-S train epochs={int(args.epochs)} train_images={len(train_paths)} "
          f"eval_split={eval_split} eval_images={len(eval_paths)} batch={bs} lr={lr:g} "
          f"eval_every={eval_every}", flush=True)

    pred_json = run_dir / f"predictions_{eval_split}.json"
    val_loader = cfg.val_dataloader

    wb_run = _bp.maybe_init_wandb(
        config={"framework": "rtdetr_external", "route": "v4",
                "epochs": int(args.epochs), "batch": bs,
                "imgsz": int(args.imgsz), "lr": lr},
        extra_tags=["rtdetr-4"],
    )

    def _run_full_eval():
        preds_ = []
        model.eval()
        _fix_rtdetrv4_eval_tensors(model, device)
        postprocessor.eval()
        t0 = time.time()
        n_img = 0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(device)
                targets_d = [{k: v.to(device) for k, v in t.items()} for t in targets]
                orig_sizes = torch.stack([t["orig_size"] for t in targets_d], dim=0).to(device)
                outputs = model(samples)
                results = postprocessor(outputs, orig_sizes)
                for target, result in zip(targets_d, results):
                    image_id = int(target["image_id"].item())
                    scores = result["scores"].detach().cpu().tolist()
                    labels_ = result["labels"].detach().cpu().tolist()
                    boxes = result["boxes"].detach().cpu().tolist()
                    for score, label, box in zip(scores, labels_, boxes):
                        if float(score) < float(args.conf):
                            continue
                        x1, y1, x2, y2 = box
                        if x2 <= x1 or y2 <= y1:
                            continue
                        preds_.append({"image_id": image_id, "category_id": int(label),
                                       "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                       "score": float(score)})
                n_img += len(results)
        fps_ = n_img / max(1e-9, time.time() - t0)
        with open(pred_json, "w") as fh: json.dump(preds_, fh)
        m_ = zero_metrics()
        m_.update(_be.coco_eval_full(str(gt_json), str(pred_json), None))
        return m_, fps_

    def _val_loss_pass(ep_idx: int):
        try:
            model.eval()
            criterion.eval() if hasattr(criterion, "eval") else None
            total_v = 0.0
            n_v = 0
            with torch.no_grad():
                for step_v, (samples, targets) in enumerate(val_loader):
                    samples = samples.to(device)
                    targets_d = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    outputs = model(samples, targets=targets_d)
                    loss_dict = criterion(
                        outputs, targets_d,
                        epoch=ep_idx, step=step_v,
                        global_step=ep_idx * max(1, len(val_loader)) + step_v,
                        epoch_step=max(1, len(val_loader)),
                    )
                    total_v += float(sum(v.detach().cpu() for v in loss_dict.values()))
                    n_v += 1
            return total_v / max(1, n_v) if n_v else None
        except Exception as e:
            print(f"[bench_plots] val_loss skipped (RT-DETRv4-S): {e}")
            return None

    if not args.inference_only and int(args.epochs) > 0:
        train_loader = cfg.train_dataloader
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
        total_steps = len(train_loader)
        log_every = _progress_log_every(total_steps)
        print(f"   train_loader: {total_steps} steps/epoch, logging every {log_every} steps", flush=True)
        for ep in range(int(args.epochs)):
            ep_t0 = time.time()
            train_loader.set_epoch(ep)
            model.train()
            criterion.train()
            total_loss = 0.0
            n_steps = 0
            for step, (samples, targets) in enumerate(train_loader):
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(samples, targets=targets)
                loss_dict = criterion(
                    outputs, targets,
                    epoch=ep, step=step,
                    global_step=ep * len(train_loader) + step,
                    epoch_step=len(train_loader),
                )
                loss = sum(loss_dict.values())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.detach().cpu())
                n_steps += 1
                if n_steps % log_every == 0 or n_steps == total_steps:
                    _log_step("RT-DETRv4-S", ep, int(args.epochs), n_steps, total_steps,
                              total_loss / max(1, n_steps), ep_t0)
            train_loss = total_loss / max(1, n_steps)
            val_loss = _val_loss_pass(ep)
            do_full = ((ep + 1) % eval_every == 0) or (ep + 1 == int(args.epochs))
            ep_metrics = None
            if do_full:
                ep_metrics, _ = _run_full_eval()
                _best_state["epoch"] = ep + 1
                _prev_score = _best_state["score"]
                bench_track_best(run_dir, model, ep_metrics, _best_state, _bench_route)
                # bench_any: wire --patience (was a dead arg). Count full-eval
                # epochs since the last improvement on BENCH_BEST_METRIC and stop
                # early once patience is exceeded — matches YOLO's epoch-based
                # patience semantics. A non-positive patience disables it.
                if _best_state["score"] > _prev_score:
                    _no_improve = 0
                else:
                    _no_improve += 1
                    if int(args.patience) > 0 and _no_improve >= int(args.patience):
                        print(f"🛑 early stopping: no {BENCH_BEST_METRIC} improvement "
                              f"for {_no_improve} eval epochs (patience={int(args.patience)})")
                        break
            elapsed = time.time() - ep_t0
            print(f"epoch {ep + 1}/{int(args.epochs)}: steps={n_steps} "
                  f"train_loss={train_loss:.4f}"
                  + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
                  + (f" box_mAP50_95={ep_metrics.get('box_mAP50_95', 0.0):.4f}" if ep_metrics else "")
                  + f" elapsed={elapsed:.1f}s",
                  flush=True)
            row = {"epoch": ep + 1, "elapsed_s": elapsed, "lr": lr,
                   "train_loss": train_loss, "val_loss": val_loss}
            if ep_metrics:
                for k in ("box_mAP50", "box_mAP50_95", "box_mAP75", "seg_mAP50", "seg_mAP50_95", "seg_mAP75"):
                    v = ep_metrics.get(k)
                    if v is not None: row[k] = float(v)
            _bp.append_history(run_dir, row)
            _bp.wandb_log_safe(wb_run, row, step=ep + 1)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Only after real training — see the same guard on the v2/v3 route: an
    # eval-only pass holds best.pth in `model` and would clobber the last epoch.
    if not args.inference_only:
        torch.save({"model": model.state_dict(), "source": "bench_any_rtdetr4"}, weights_dir / "last.pth")

    print(f"🧪 RT-DETRv4-S eval on {eval_split}: {len(eval_paths)} images")
    bench_load_best(run_dir, model, "RT-DETRv4-S")
    metrics, fps = _run_full_eval()
    if args.seg_via_sam2:
        img_dir = images_dir_for_split(data_dir, eval_split)
        if args.sam2_ckpt and args.sam2_config and img_dir and img_dir.is_dir():
            print("🎯 SAM2 masks from RT-DETRv4-S boxes")
            sam_out = run_dir / "_bench_sam2_seg.json"
            metrics.update(_be.sam2_seg_from_boxes(
                str(img_dir), str(pred_json),
                args.sam2_ckpt, args.sam2_config,
                str(gt_json), str(sam_out),
            ))
        else:
            print("⚠️ --seg_via_sam2 set but SAM2 paths or eval image dir are missing; skipping.")
    metrics["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    metrics["fps_eval"] = fps
    print(f"📊 RT-DETRv4-S box_mAP50={metrics.get('box_mAP50', 0.0):.4f} "
          f"box_mAP50_95={metrics.get('box_mAP50_95', 0.0):.4f} "
          f"seg_mAP50_95={metrics.get('seg_mAP50_95', 0.0):.4f} fps={fps:.1f}")

    _emit_paper_figures(run_dir, gt_json, pred_json,
                        _read_class_names(data_dir) or ["object"],
                        title=f"RT-DETRv4-S ({data_dir.name})")
    _bp.wandb_log_safe(wb_run, {"final/" + k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
    _bp.wandb_finish_safe(wb_run)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="detect", choices=["detect"])
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch_sz", type=float, default=1)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--inference_only", action="store_true")
    ap.add_argument("--nbs", type=int, default=0,
                    help="Target effective batch (gradient accumulation). 0 = disabled.")
    ap.add_argument("--batch_sz_max", type=int, default=16,
                    help="Hard ceiling for fractional/auto micro-batch sizing.")
    ap.add_argument("--eval_every", type=int, default=10,
                    help="Run full COCO eval every K epochs (final epoch always evaluated). "
                         "Per-epoch val-loss + history.csv are written regardless.")
    ap.add_argument("--seg_via_sam2", action="store_true")
    ap.add_argument("--sam2_ckpt", default="")
    ap.add_argument("--sam2_config", default="")
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, inject the shared Albumentations pipeline as a pre-transform "
                         "into YoloDetectionDataset.__getitem__ (image + bboxes via A.Compose with "
                         "BboxParams(format='coco')). Eval split is never augmented.")
    args, _ = ap.parse_known_args()

    data_dir = Path(args.data).resolve()
    # bench_any.py forwards the fully-resolved run name (incl. hyperparameter
    # fingerprint) via BENCH_RUN_NAME so distinct configs land in separate run
    # dirs and the orchestrator can find results.csv where it expects them.
    _env_run = os.environ.get("BENCH_RUN_NAME")
    run_name = _env_run if _env_run else build_run_name(data_dir, args.model)
    run_dir = HOME_BENCH / "runs" / "detect" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    unsupported_path = run_dir / "UNSUPPORTED.txt"
    if unsupported_path.exists():
        unsupported_path.unlink()
    key = _model_key(args.model)
    if key not in DOWNLOADS:
        raise SystemExit(f"unknown external RT-DETR model '{args.model}'")

    model_path = Path(str(args.model)).expanduser()
    if model_path.is_file() and DOWNLOADS[key]["kind"] != "hf":
        artifact = str(model_path.resolve())
    else:
        artifact = ensure_download(key)
    print(f"✅ {key} artifact ready: {artifact}")

    if key == "rtdetr-2":
        metrics = train_v2(args, artifact, run_dir)
    elif key == "rtdetr-2-pth":
        spec = DOWNLOADS[key]
        metrics = train_rtdetrv2_repo(args, artifact, run_dir, spec["variant"], spec["config"], "torch")
    elif key.startswith("rtdetr-3"):
        spec = DOWNLOADS[key]
        metrics = train_rtdetrv2_repo(
            args, artifact, run_dir, spec["variant"], spec["config"], "paddle",
            official_config_name=spec.get("official_config"),
        )
    elif key == "rtdetr-4":
        metrics = train_v4(args, artifact, run_dir)
    else:
        runtime = DOWNLOADS[key].get("runtime", "external runtime")
        metrics = zero_metrics()
        unsupported_path.write_text(
            f"{key} checkpoint downloaded to {artifact}\n"
            f"Required runtime not wired in this venv: {runtime}\n"
            "This is not an Ultralytics .pt model; do not route it through rtdetr_any_tune.py.\n"
        )
        print(unsupported_path.read_text().strip())

    # Report the resolved micro-batch and realised effective batch so the bench
    # CSV reflects the real per-model batch. _resolve_micro_bs_static mirrors the
    # literal/cap logic used during training (exact for batch_sz>=1; for fractional
    # auto-batch it reports the static default, which the orchestrator treats as a
    # fallback). Only set when not an unsupported stub.
    if not unsupported_path.exists():
        try:
            _micro = _resolve_micro_bs_static(args)
            _nbs = int(getattr(args, "nbs", 0) or 0)
            metrics["batch_sz"] = int(_micro)
            metrics["batch_effective"] = max(_nbs, int(_micro)) if _nbs > _micro else int(_micro)
        except Exception as _e:
            print(f"⚠️ batch reporting skipped: {_e}")

    with open(run_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    print(f"✅ Wrote {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
