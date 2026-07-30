from __future__ import annotations

import random
from pathlib import Path

# All knobs disabled on both policies — when policy != "native", augmentation is
# the sole responsibility of the shared Albumentations list (see
# shared_albumentations_policy below) so every framework sees the same pipeline.
ULTRALYTICS_AUG = {
    "off": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "shear": 0.0, "perspective": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "fliplr": 0.0, "flipud": 0.0, "erasing": 0.0,
    },
    "fair": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "shear": 0.0, "perspective": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "fliplr": 0.0, "flipud": 0.0, "erasing": 0.0,
    },
}


def ultralytics_aug_args(policy: str) -> list[str]:
    if policy == "native":
        return []
    vals = ULTRALYTICS_AUG[policy]
    out = []
    for key, value in vals.items():
        out += [f"--{key}", str(value)]
    return out


def shared_albumentations_policy(policy: str, imgsz: int, task: str | None = None,
                                  framework: str | None = None) -> list:
    """Single source of truth for the cross-framework Albumentations pipeline.

    policy="off"    -> []
    policy="fair"   -> a small, fixed list (<=6 transforms) safe for both
                       bboxes and instance/semantic masks. The same list is
                       used by YOLO-seg, RT-DETR det, RF-DETR seg, EoMT,
                       DINOv3-M2F, and DINOv3-SemSeg so that every framework
                       sees the same training-time augmentation distribution.
    policy="native" -> []  (caller keeps its own defaults)

    NOTE: Ultralytics' external `augmentations=` hook applies image-only ops
    and does NOT propagate geometric transforms back into the polygon/bbox
    targets. That causes a batch["bboxes"] vs batch["cls"] size mismatch on
    YOLO-seg when an A.Affine drops/clips instances. So when the caller is
    Ultralytics YOLO/RT-DETR, we omit A.Affine and keep only photometric +
    horizontal-flip ops. Non-Ultralytics wrappers (RF-DETR, EoMT, M2F,
    semseg, rtdetr_external) go through loaders that sync masks/boxes with
    Albumentations targets, so they keep the full pipeline.
    """
    if policy in ("native", "off"):
        return []
    if policy != "fair":
        raise ValueError(f"unknown aug policy: {policy!r}")
    try:
        import albumentations as A
        import cv2
    except Exception:
        return []
    random.seed(0)
    ultralytics_path = (framework in ("yolo", "rtdetr"))
    pipeline = [A.HorizontalFlip(p=0.5)]
    if not ultralytics_path:
        pipeline.append(A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ))
    pipeline += [
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20,
                             val_shift_limit=20, p=0.5),
        A.MotionBlur(blur_limit=5, p=0.10),
        A.GaussNoise(p=0.10),
        A.ImageCompression(quality_range=(60, 95), p=0.10),
    ]
    return pipeline


def write_aug_grid(data_dir: str | Path, policy: str, out_dir: str | Path, n: int = 8) -> Path | None:
    try:
        import cv2
        import numpy as np
        import yaml
    except Exception as e:
        print(f"[bench] ⚠️ aug grid skipped: {e}")
        return None

    data_dir = Path(data_dir)
    with open(data_dir / "data.yaml") as fh:
        cfg = yaml.safe_load(fh) or {}
    root = Path(cfg.get("path", data_dir)).expanduser()
    if not root.is_absolute():
        root = (data_dir / root).resolve()
    split = cfg.get("train")
    if not split:
        return None
    img_dir = root / split if not Path(split).is_absolute() else Path(split)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        return None

    def aug(img):
        if policy != "fair":
            return img
        import albumentations as A
        t = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.90, 1.10), rotate=(-5, 5),
                     border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20, p=0.7),
        ])
        return t(image=img)["image"]

    tiles = []
    for p in images[:n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        vis = np.hstack([img, aug(img.copy())])
        cv2.putText(vis, p.stem[:24], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(vis)
    if not tiles:
        return None

    rows = []
    for i in range(0, len(tiles), 2):
        row = tiles[i:i + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{data_dir.name}_aug_{policy}.jpg"
    cv2.imwrite(str(out), grid)
    return out
