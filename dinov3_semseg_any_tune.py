# Wraps Raessan/semantic_segmentation_dinov3 (lightweight ASPP head on a frozen DINOv3
# backbone, semantic seg). Mirrors yolo_any_tune.py CLI.
# USAGE: python3 dinov3_semseg_any_tune.py --data /path/to/dataset --epochs 20 --batch_sz 8 --model dinov3-semseg-vits16plus
# DINOv3 weights: pass --dinov3_weights /path/to.pth (or set $DINOV3_WEIGHTS).

import os, sys, yaml, argparse, shutil, atexit, tempfile, hashlib, csv, subprocess, json
from pathlib import Path
import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_URL = "https://github.com/Raessan/semantic_segmentation_dinov3.git"
REPO_BASE = Path(os.environ.get("BENCH_ANY", str(Path.home() / "bench_any")))
REPO_CACHE = REPO_BASE / "semantic_segmentation_dinov3"

BACKBONES = {
    # Order matters: the matcher does `if k in name`, so longer/more specific
    # keys must precede their prefixes (e.g. vits16plus before vits16, vith16plus
    # before vith16). Full DINOv3 paper Table 2 / Table 11 set.
    "vits16plus":     "dinov3_vits16plus",
    "vits16":         "dinov3_vits16",
    "vitb16":         "dinov3_vitb16",
    "vitl16":         "dinov3_vitl16",
    "vith16plus":     "dinov3_vith16plus",
    "vith16":         "dinov3_vith16plus",
    "vit7b16":        "dinov3_vit7b16",
    "convnext_tiny":  "dinov3_convnext_tiny",
    "convnext_small": "dinov3_convnext_small",
    "convnext_base":  "dinov3_convnext_base",
    "convnext_large": "dinov3_convnext_large",
    # Friendly aliases.
    "smallplus":      "dinov3_vits16plus",
    "small":          "dinov3_vits16plus",
    "base":           "dinov3_vitb16",
    "large":          "dinov3_vitl16",
    "huge":           "dinov3_vith16plus",
    "giant":          "dinov3_vit7b16",
}

TEMP_DIR_PATH = None
def cleanup_temp_dir():
    global TEMP_DIR_PATH
    if TEMP_DIR_PATH and TEMP_DIR_PATH.exists():
        shutil.rmtree(TEMP_DIR_PATH, ignore_errors=True); TEMP_DIR_PATH = None


def _ensure_imports(modules, label=""):
    missing = [spec for imp, spec in modules
               if subprocess.run([sys.executable, "-c", f"import {imp}"], capture_output=True).returncode]
    if not missing: return
    print(f"📦 [{label}] pip install {' '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing]); return
    except subprocess.CalledProcessError:
        for s in missing:
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", s])
            except subprocess.CalledProcessError as e: print(f"⚠️ pip install {s} rc={e.returncode}")


def ensure_repo():
    REPO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_CACHE / ".git").exists():
        print(f"📥 Cloning {REPO_URL} → {REPO_CACHE}")
        subprocess.check_call(["git", "clone", "--depth=1", "--recurse-submodules", REPO_URL, str(REPO_CACHE)])
    _ensure_imports([("omegaconf", "omegaconf"), ("fvcore", "fvcore"), ("iopath", "iopath"),
                     ("tqdm", "tqdm"), ("PIL", "Pillow"), ("cv2", "opencv-python")], label="dinov3-semseg")
    return REPO_CACHE


def resolve_dinov3_weights(dino_model, repo_cache, explicit="", explicit_dir=""):
    """Resolve a DINOv3 backbone .pth.

    Search order:
      1. explicit `--dinov3_weights` (file)             - highest priority
      2. `--dinov3_dir` / $DINOV3_DIR                   - shared weights dir
      3. $BENCH_ANY/dinov3_weights                       - sibling of repo cache
      4. ~/dinov3_weights                                - legacy default
      5. <repo_cache>/dinov3_weights                    - in-repo fallback
    """
    if explicit and Path(explicit).expanduser().is_file():
        return str(Path(explicit).expanduser().resolve())
    bench_any = Path(os.environ.get("BENCH_ANY", str(Path.home() / "bench_any")))
    candidates = []
    if explicit_dir: candidates.append(Path(explicit_dir).expanduser())
    candidates += [bench_any / "dinov3_weights",
                   Path.home() / "dinov3_weights",
                   Path(repo_cache) / "dinov3_weights"]
    for d in candidates:
        if d.is_dir():
            cands = sorted(d.rglob(f"{dino_model}*.pth"))
            if cands: return str(cands[0].resolve())
    return ""


def available_dinov3_weights(repo_cache, explicit_dir=""):
    bench_any = Path(os.environ.get("BENCH_ANY", str(Path.home() / "bench_any")))
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir).expanduser())
    candidates += [
        bench_any / "dinov3_weights",
        Path.home() / "dinov3_weights",
        Path(repo_cache) / "dinov3_weights",
    ]
    found = {}
    for d in candidates:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("dinov3_*.pth")):
            stem = p.name.split("_pretrain", 1)[0]
            found.setdefault(stem, str(p.resolve()))
    return found


def resolve_dinov3_hub_dir(repo_dir, explicit_dir=""):
    """Return a torch.hub local-source directory containing hubconf.py.

    Older bench commands use --dinov3_dir for both the facebookresearch/dinov3
    checkout and a plain directory of .pth weights. Only the former is valid as
    torch.hub's repo_or_dir; the latter remains useful as a weight search path.
    """
    default_hub = Path(repo_dir) / "dinov3"
    if explicit_dir:
        p = Path(explicit_dir).expanduser()
        if (p / "hubconf.py").is_file():
            return str(p.resolve()), ""
        if (p / "dinov3" / "hubconf.py").is_file():
            return str((p / "dinov3").resolve()), ""
        if p.is_dir():
            return str(default_hub), (
                f"⚠️ Ignoring --dinov3_dir={p} as a torch.hub source; it has no hubconf.py. "
                "Using it only as a DINOv3 .pth weight search directory."
            )
    return str(default_hub), ""


def pick_backbone(name):
    n = name.lower()
    for k, bb in BACKBONES.items():
        if k in n: return bb
    return BACKBONES["vits16plus"]


def _seg_id_to_rgb(sid): return (sid & 0xFF, (sid >> 8) & 0xFF, (sid >> 16) & 0xFF)


def _seg_id_to_bgr(sid):
    r, g, b = _seg_id_to_rgb(sid)
    return (b, g, r)


def yolo_to_coco_panoptic(data_yaml, dataset_name):
    """Rasterise YOLO polygons into COCO panoptic PNGs + per-split JSONs in
    <root>/{train2017,val2017}/<file>.jpg + <root>/annotations/panoptic_*."""
    global TEMP_DIR_PATH
    yp = Path(data_yaml).resolve(); root = yp.parent
    cfg = yaml.safe_load(open(yp))
    raw_names = cfg.get('names')
    class_names = raw_names if isinstance(raw_names, list) else list(raw_names.values())

    shm = Path('/dev/shm'); base = shm if shm.exists() else Path(tempfile.gettempdir())
    TEMP_DIR_PATH = base / f"dinov3_semseg_{dataset_name}_{os.getpid()}"
    TEMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    atexit.register(cleanup_temp_dir)
    (TEMP_DIR_PATH / "annotations").mkdir(exist_ok=True)

    # Upstream prepends 'background' (id 0) at runtime; start cat ids at 1.
    categories = [{"id": i + 1, "name": n, "supercategory": "object", "isthing": 1}
                  for i, n in enumerate(class_names)]
    eval_key = 'test' if cfg.get('test') else ('valid' if cfg.get('valid') else 'val')
    splits = {'train': 'train2017', eval_key: 'val2017'}
    seen = set()
    for split_key, dst in splits.items():
        if split_key not in cfg or dst in seen: continue
        seen.add(dst)
        src_img = (root / cfg[split_key]).resolve()
        # Swap only the 'images' path *component* (mirrors bench_eval). A blind
        # str.replace corrupts paths whose parent dirs also contain "images",
        # silently yielding empty masks (all-background GT).
        _parts = list(src_img.parts)
        if "images" in _parts:
            _parts[_parts.index("images")] = "labels"
            src_lbl = Path(*_parts)
        else:
            src_lbl = src_img
        if not src_img.is_dir(): continue
        img_out = TEMP_DIR_PATH / dst
        pan_out = TEMP_DIR_PATH / "annotations" / f"panoptic_{dst}"
        img_out.mkdir(parents=True, exist_ok=True); pan_out.mkdir(parents=True, exist_ok=True)

        images, panop_anns = [], []
        next_seg_id = 1
        for img_id, img_path in enumerate(sorted(src_img.iterdir()), start=1):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'): continue
            link = img_out / img_path.name
            if not link.exists(): os.symlink(img_path.resolve(), link)
            im = cv2.imread(str(img_path))
            if im is None: continue
            h, w = im.shape[:2]
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            pan_png = np.zeros((h, w, 3), dtype=np.uint8)
            segments_info = []
            lbl_path = src_lbl / f"{img_path.stem}.txt"
            if lbl_path.exists():
                for line in open(lbl_path):
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cid = int(parts[0]); coords = [float(x) for x in parts[1:]]
                    if cid >= len(class_names): continue
                    if len(coords) >= 6 and len(coords) % 2 == 0:
                        pts = np.array([(coords[i] * w, coords[i + 1] * h)
                                        for i in range(0, len(coords), 2)], dtype=np.int32)
                        seg_id = next_seg_id; next_seg_id += 1
                        cv2.fillPoly(pan_png, [pts], _seg_id_to_bgr(seg_id))
                        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                        segments_info.append({"id": seg_id, "category_id": cid + 1,
                                              "area": int(cv2.contourArea(pts)),
                                              "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0})
                    elif len(coords) == 4:
                        xc, yc, bw, bh = coords
                        x1 = int(max(0, (xc - bw / 2) * w)); y1 = int(max(0, (yc - bh / 2) * h))
                        x2 = int(min(w, (xc + bw / 2) * w)); y2 = int(min(h, (yc + bh / 2) * h))
                        seg_id = next_seg_id; next_seg_id += 1
                        cv2.rectangle(pan_png, (x1, y1), (x2, y2), _seg_id_to_bgr(seg_id), thickness=-1)
                        segments_info.append({"id": seg_id, "category_id": cid + 1,
                                              "area": (x2 - x1) * (y2 - y1),
                                              "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0})
            cv2.imwrite(str(pan_out / f"{img_path.stem}.png"), pan_png)
            panop_anns.append({"image_id": img_id, "file_name": f"{img_path.stem}.png",
                               "segments_info": segments_info})

        with open(TEMP_DIR_PATH / "annotations" / f"panoptic_{dst}.json", "w") as f:
            json.dump({"images": images, "annotations": panop_anns, "categories": categories}, f)
    return TEMP_DIR_PATH, class_names


# Auto-generated runner that mirrors the upstream notebook cells, with checkpoint
# save/resume + mIoU/pixel-acc eval + results.csv write.
RUNNER_SRC = r'''import os, sys, csv
import multiprocessing as _mp
# Py3.14 changed Linux multiprocessing default from "fork" → "forkserver".
# Standard spawn method for CUDA safety under Python 3.14.
try:
    if sys.platform.startswith("linux"):
        _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
import numpy as np, torch, torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config.config as cfg
from src.dataset_coco import DatasetCOCOPanoptic
from src.model_backbone import DinoBackbone
from src.model_head import ASPPDecoder
from src.loss_focal_dice import SemanticLoss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--inference_only", action='store_true')
    ap.add_argument("--aug_policy", default="native", choices=["native", "off", "fair"],
                    help="When != native, replace train_set.transform with the shared "
                         "cross-framework Albumentations pipeline (semantic + instance "
                         "masks via additional_targets). Eval is left untouched.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "last.pth"

    train_set = DatasetCOCOPanoptic(cfg.COCO_ROOT, "train", cfg.IMG_SIZE, cfg.PATCH_SIZE,
                                    cfg.PROB_AUGMENT_TRAINING, cfg.IMG_MEAN, cfg.IMG_STD)
    val_set   = DatasetCOCOPanoptic(cfg.COCO_ROOT, "val",   cfg.IMG_SIZE, cfg.PATCH_SIZE,
                                    cfg.PROB_AUGMENT_VALID, cfg.IMG_MEAN, cfg.IMG_STD)

    # ---- bench_any: shared cross-framework Albumentations policy (train only) ---
    # Replaces train_set.transform's geometric+photometric ops with the shared
    # pipeline so this runner sees the same augmentation distribution as YOLO,
    # RT-DETR, RF-DETR, EoMT, and DINOv3-M2F. Eval (val_set) is left untouched.
    _BENCH_AUG_POLICY = os.environ.get("BENCH_AUG_POLICY", "native")
    if _BENCH_AUG_POLICY in ("off", "fair"):
        try:
            import sys as _sys
            _be_dir = os.environ.get("BENCH_EVAL_DIR")
            if _be_dir and _be_dir not in _sys.path: _sys.path.insert(0, _be_dir)
            import albumentations as A
            from bench_aug import shared_albumentations_policy as _sap
            _pipe = _sap(_BENCH_AUG_POLICY, cfg.IMG_SIZE)
            if _pipe:
                train_set.transform = A.Compose(
                    _pipe,
                    additional_targets={'semantic': 'mask', 'instance': 'mask'},
                )
            # Force augment_prob=1.0 so the shared pipeline is the only source of
            # augmentation and is always applied (each individual A.* keeps its p=).
            train_set.augment_prob = 1.0
            print(f"[bench_any] dinov3_semseg shared aug applied (policy={_BENCH_AUG_POLICY})")
        except Exception as _e:
            print(f"[bench_any] dinov3_semseg shared aug unavailable: {_e}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.threads, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, num_workers=args.threads, shuffle=False)
    NC = len(train_set.class_names)

    dino = torch.hub.load(repo_or_dir=cfg.DINOV3_DIR, model=cfg.DINO_MODEL,
                          source="local", weights=cfg.DINO_WEIGHTS)
    backbone = DinoBackbone(dino, cfg.MODEL_TO_NUM_LAYERS[cfg.DINO_MODEL]).to(device)
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval()
    head = ASPPDecoder(num_classes=NC, in_ch=cfg.MODEL_TO_EMBED_DIM[cfg.DINO_MODEL],
                       target_size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE)).to(device)
    # Resume: old checkpoints were a bare head state_dict (so --resume only
    # warm-started the weights and the loop below still re-ran the FULL epoch
    # budget from epoch 0, throwing away optimizer momentum). New checkpoints wrap
    # {head, opt, epoch} so a crashed run continues mid-training from the last
    # completed epoch. We stash the blob here and restore the optimizer once it
    # exists (a few lines down).
    _start_ep = 0
    _ckpt_blob = None
    if (args.resume or args.inference_only) and ckpt.is_file():
        _ckpt_blob = torch.load(str(ckpt), map_location=device)
        if isinstance(_ckpt_blob, dict) and "head" in _ckpt_blob:
            head.load_state_dict(_ckpt_blob["head"])
            _start_ep = int(_ckpt_blob.get("epoch", 0))
            print(f"📦 Loaded {ckpt} (resume from epoch {_start_ep})")
        else:
            # Back-compat: bare state_dict from an older run — weights only.
            head.load_state_dict(_ckpt_blob)
            _ckpt_blob = None
            print(f"📦 Loaded {ckpt} (legacy weights-only; optimizer/epoch not recoverable)")

    opt = optim.Adam(head.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    if _ckpt_blob is not None and "opt" in _ckpt_blob:
        try:
            opt.load_state_dict(_ckpt_blob["opt"]); print("📦 Optimizer state restored")
        except Exception as _e_opt:
            print(f"⚠️ optimizer state restore failed ({_e_opt}); continuing with fresh Adam state")
    loss_fn = SemanticLoss(dice_weight=cfg.WEIGHT_LOSS_DICE, focal_weight=cfg.WEIGHT_LOSS_FOCAL,
                           target_size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE), ignore_index=None)


    def eval_miou():
        """Walk the val set and build a full HxW confusion matrix, then defer to
        bench_eval.semseg_extra for the headline scalars (mIoU/mAcc/fwIoU/pixel_acc)
        and to bench_eval.boundary_iou_mean for the boundary AP averaged across
        images. All metrics share one forward pass so this is no slower than the
        old mIoU-only loop."""
        head.eval()
        cm = np.zeros((NC, NC), dtype=np.int64)
        biou_sum = 0.0; biou_n = 0
        # bench_eval lives in the parent agent's project dir (where bench_any.py is).
        # Parent injects that path via $BENCH_EVAL_DIR so this runner — which lives
        # under the third-party DINOv3-semseg repo — can still import it.
        try:
            import sys as _sys, os as _os
            from pathlib import Path as _Path
            _be_dir = _os.environ.get("BENCH_EVAL_DIR")
            if _be_dir:
                _sys.path.insert(0, _be_dir)
            # Sibling fallback
            _sib = str(_Path(__file__).resolve().parent)
            if _sib not in _sys.path:
                _sys.path.insert(0, _sib)
            import bench_eval as _be  # type: ignore
        except Exception:
            _be = None
        with torch.no_grad():
            for images, sem_t, _ in tqdm(val_loader, desc="val"):
                images = images.to(device, dtype=torch.float)
                sem_t = sem_t.to(device, dtype=torch.long)
                logits = head(backbone(images))
                logits = torch.nn.functional.interpolate(logits, size=sem_t.shape[-2:],
                                                         mode="bilinear", align_corners=False)
                pred = logits.argmax(dim=1)
                # Confusion matrix update: rows=GT, cols=pred. bincount on the
                # flattened (NC * gt + pred) index is the fastest way to get this
                # without per-class loops.
                gt_flat = sem_t.view(-1).cpu().numpy()
                pr_flat = pred.view(-1).cpu().numpy()
                valid = (gt_flat >= 0) & (gt_flat < NC) & (pr_flat >= 0) & (pr_flat < NC)
                if valid.any():
                    idx = NC * gt_flat[valid].astype(np.int64) + pr_flat[valid].astype(np.int64)
                    cm += np.bincount(idx, minlength=NC * NC).reshape(NC, NC)
                # Boundary-IoU averaged per-image (bench_eval helper is class-aware).
                if _be is not None:
                    for i in range(len(sem_t)):
                        gt_np = sem_t[i].cpu().numpy()
                        pr_np = pred[i].cpu().numpy()
                        try:
                            biou_sum += float(_be.boundary_iou_mean(pr_np, gt_np, NC)); biou_n += 1
                        except Exception:
                            pass
        if _be is not None:
            extra = _be.semseg_extra(cm)
        else:
            # Fallback: mIoU + pixel_acc only, matching the old return contract.
            tp = np.diag(cm); gt = cm.sum(1); pr = cm.sum(0)
            iou = tp / np.maximum(gt + pr - tp, 1e-12)
            m = gt > 0
            extra = {"mIoU": float(iou[m].mean()) if m.any() else 0.0,
                     "mAcc": 0.0, "fwIoU": 0.0,
                     "pixel_acc": float(tp.sum() / max(1, cm.sum()))}
        extra["boundary_IoU"] = (biou_sum / biou_n) if biou_n else 0.0
        # boundary_iou_mean intentionally excludes background, so expose the same
        # value through the explicit foreground column for paper tables.
        extra["fg_boundary_IoU"] = extra["boundary_IoU"]
        return extra


    if not args.inference_only and _start_ep >= args.epochs:
        print(f"✅ Already trained {_start_ep}/{args.epochs} epochs; skipping to final eval.")
    if not args.inference_only and _start_ep < args.epochs:
        for ep in range(_start_ep, args.epochs):
            head.train(); tloss = 0.0; bi = 0
            for bi, (images, sem_t, _) in enumerate(tqdm(train_loader, desc=f"ep{ep+1}/{args.epochs}")):
                images = images.to(device, dtype=torch.float); sem_t = sem_t.to(device, dtype=torch.long)
                opt.zero_grad()
                losses = loss_fn(head(backbone(images)), sem_t); losses[0].backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
                opt.step(); tloss += losses[0].item()
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep + 1}, str(ckpt))
            _avg = tloss / max(1, bi + 1)
            print(f"epoch {ep+1}: train_loss={_avg:.4f}")
            # Per-epoch history.csv for paper figures (bench_plots picks it up).
            try:
                _gd = out_dir / "graphs"; _gd.mkdir(parents=True, exist_ok=True)
                _hp = _gd / "history.csv"
                import csv as _csv
                _cols = ["epoch", "train_loss", "lr"]
                _new = not _hp.is_file()
                with open(_hp, "a", newline="") as _fh:
                    _w = _csv.writer(_fh)
                    if _new: _w.writerow(_cols)
                    _w.writerow([ep + 1, f"{_avg:.6f}", f"{opt.param_groups[0]['lr']:g}"])
            except Exception as _e:
                print(f"  ⚠️ history.csv append skipped: {_e}")

    extra = eval_miou()
    miou = float(extra.get("mIoU", 0.0)); pacc = float(extra.get("pixel_acc", 0.0))
    print(f"📊 mIoU={miou:.4f}  mAcc={extra.get('mAcc',0.0):.4f}  fwIoU={extra.get('fwIoU',0.0):.4f}  "
          f"pixel_acc={pacc:.4f}  boundary_IoU={extra.get('boundary_IoU',0.0):.4f}")
    print(f"📊 foreground: mIoU={extra.get('fg_mIoU',0.0):.4f}  mAcc={extra.get('fg_mAcc',0.0):.4f}  "
          f"pixel_acc={extra.get('fg_pixel_acc',0.0):.4f}  boundary_IoU={extra.get('fg_boundary_IoU',0.0):.4f}")
    metrics = {k: 0.0 for k in ("box_P","box_R","box_F1","box_mAP50","box_mAP75","box_mAP50_95",
                                "seg_P","seg_R","seg_F1","seg_mAP50","seg_mAP75","seg_mAP50_95",
                                "box_mAP_S", "box_mAP_M", "box_mAP_L", "box_AR_100",
                                "seg_mAP_S", "seg_mAP_M", "seg_mAP_L", "seg_AR_100",
                                "mIoU", "mAcc", "fwIoU", "pixel_acc", "boundary_IoU",
                                "fg_mIoU", "fg_mAcc", "fg_fwIoU", "fg_pixel_acc", "fg_boundary_IoU",
                                "fg_seg_P", "fg_seg_R", "fg_seg_F1",
                                "params_M", "fps_eval", "n_images", "n_instances")}
    metrics["mIoU"]         = miou
    metrics["mAcc"]         = float(extra.get("mAcc", 0.0))
    metrics["fwIoU"]        = float(extra.get("fwIoU", 0.0))
    metrics["pixel_acc"]    = pacc
    metrics["boundary_IoU"] = float(extra.get("boundary_IoU", 0.0))
    metrics["fg_mIoU"]         = float(extra.get("fg_mIoU", 0.0))
    metrics["fg_mAcc"]         = float(extra.get("fg_mAcc", 0.0))
    metrics["fg_fwIoU"]        = float(extra.get("fg_fwIoU", 0.0))
    metrics["fg_pixel_acc"]    = float(extra.get("fg_pixel_acc", 0.0))
    metrics["fg_boundary_IoU"] = float(extra.get("fg_boundary_IoU", 0.0))
    # Pixel-level P/R/F1 averaged over present classes (mirrors mIoU semantics).
    # These come straight from `bench_eval.semseg_extra` (v3+); the previous
    # `2*miou*pacc/(miou+pacc)` shortcut produced an F1 that didn't match its own
    # P / R (which were left at 0), making the row look inconsistent.
    metrics["seg_P"]        = float(extra.get("seg_P",  0.0))
    metrics["seg_R"]        = float(extra.get("seg_R",  0.0))
    metrics["seg_F1"]       = float(extra.get("seg_F1", 0.0))
    metrics["fg_seg_P"]     = float(extra.get("fg_seg_P",  0.0))
    metrics["fg_seg_R"]     = float(extra.get("fg_seg_R",  0.0))
    metrics["fg_seg_F1"]    = float(extra.get("fg_seg_F1", 0.0))
    # Params + FPS: count head + frozen backbone together (this is the deployed
    # inference graph). FPS probed on a single zero tensor at val resolution.
    # Re-import bench_eval at module scope (the earlier import lived inside
    # eval_miou()'s scope, so referencing it here used to raise NameError).
    try:
        import sys as _sys, os as _os
        from pathlib import Path as _Path
        _be_dir = _os.environ.get("BENCH_EVAL_DIR")
        if _be_dir and _be_dir not in _sys.path: _sys.path.insert(0, _be_dir)
        _sib = str(_Path(__file__).resolve().parent)
        if _sib not in _sys.path: _sys.path.insert(0, _sib)
        try:
            import bench_eval as _be2  # type: ignore
        except Exception:
            _be2 = None
        if _be2 is not None:
            n_p = sum(p.numel() for p in head.parameters()) + sum(p.numel() for p in backbone.parameters())
            metrics["params_M"] = float(n_p) / 1e6
            if torch.cuda.is_available():
                x = torch.zeros(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device,
                                dtype=next(head.parameters()).dtype)
                head.eval(); backbone.eval()
                metrics["fps_eval"] = float(_be2.time_fps(lambda: head(backbone(x))))
    except Exception as _e:
        print(f"⚠️ params/fps probe failed: {_e}")
    # Report the actual batch. This wrapper trains a lightweight ASPP head with a
    # plain loop (no gradient accumulation), so effective batch == micro-batch.
    try:
        metrics["batch_sz"] = int(cfg.BATCH_SIZE)
        metrics["batch_effective"] = int(cfg.BATCH_SIZE)
    except Exception as _e:
        print(f"⚠️ batch reporting skipped: {_e}")
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        w.writeheader(); w.writerow(metrics)
    print(f"✅ Wrote {out_dir / 'results.csv'}")
    # Append final eval row (mIoU/pixel_acc/etc.) to history.csv so the
    # accuracy_curve panel shows the final measurement instead of being blank.
    try:
        _gd = out_dir / "graphs"; _gd.mkdir(parents=True, exist_ok=True)
        _hp = _gd / "history.csv"
        import csv as _csv
        _ep_now = args.epochs
        _new_keys = ["epoch", "mIoU", "pixel_acc", "fg_mIoU", "boundary_IoU"]
        # Read existing rows + extend header.
        _rows = []; _hdr = None
        if _hp.is_file():
            with open(_hp, newline="") as _f:
                _r = _csv.DictReader(_f); _hdr = _r.fieldnames or []
                _rows = list(_r)
        _hdr2 = list(dict.fromkeys((_hdr or []) + _new_keys))
        _final = {"epoch": _ep_now,
                  "mIoU": f"{miou:.6f}", "pixel_acc": f"{pacc:.6f}",
                  "fg_mIoU": f"{float(extra.get('fg_mIoU', 0.0)):.6f}",
                  "boundary_IoU": f"{float(extra.get('boundary_IoU', 0.0)):.6f}"}
        # Merge final into the row matching this epoch (if it exists), else append.
        _merged = False
        for _r in _rows:
            if str(_r.get("epoch", "")) == str(_ep_now):
                _r.update(_final); _merged = True; break
        if not _merged:
            _rows.append(_final)
        with open(_hp, "w", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=_hdr2, extrasaction="ignore")
            _w.writeheader(); _w.writerows(_rows)
    except Exception as _e:
        print(f"⚠️ history.csv final append skipped: {_e}")

if __name__ == "__main__":
    main()
'''


def patch_config(repo_dir, coco_root, batch_size, lr, epochs, dino_dir, dino_weights, dino_model):
    """Rewrite config/config.py with our paths line-wise so the file stays valid Python."""
    cfg_path = Path(repo_dir) / "config" / "config.py"
    text = cfg_path.read_text()
    repls = {
        "COCO_ROOT":     repr(str(coco_root)),
        "DINOV3_DIR":    repr(str(dino_dir)),
        "DINO_WEIGHTS":  repr(str(dino_weights)),
        "DINO_MODEL":    repr(str(dino_model)),
        "BATCH_SIZE":    str(int(batch_size)),
        "LEARNING_RATE": str(float(lr)),
        "NUM_EPOCHS":    str(int(epochs)),
        "SAVE_MODEL":    "False",
    }
    new_lines = []
    for line in text.splitlines():
        s = line.lstrip(); replaced = False
        for k, v in repls.items():
            if s.startswith(f"{k} =") or s.startswith(f"{k}="):
                new_lines.append(f"{k} = {v}"); replaced = True; break
        if not replaced: new_lines.append(line)
    cfg_path.write_text("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "win32":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    # ---------------------------------------------------------------- Args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="dinov3-semseg-vits16plus")
    ap.add_argument("--task", default="segment", choices=["segment"])
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--batch_sz", default=8, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--threads", default=4, type=int)
    ap.add_argument("--resume", action='store_true')
    ap.add_argument("--inference_only", action='store_true')
    ap.add_argument("--tune", action='store_true')
    ap.add_argument("--tune_iter", default=5, type=int)
    ap.add_argument("--repo_dir", default="")
    ap.add_argument("--dinov3_dir", default=os.environ.get("DINOV3_DIR", ""),
                    help="facebookresearch/dinov3 checkout (torch.hub local source).")
    ap.add_argument("--dinov3_weights", default=os.environ.get("DINOV3_WEIGHTS", ""),
                    help="DINOv3 backbone .pth (or set $DINOV3_WEIGHTS).")
    ap.add_argument("--aug_policy", choices=["native", "off", "fair"], default="native",
                    help="When != native, the templated runner replaces train_set.transform "
                         "with the shared cross-framework Albumentations pipeline.")
    args = ap.parse_args()
    _semseg_aug_status = "applied" if args.aug_policy in ("off", "fair") else "native"
    print(f"[bench] aug_policy={args.aug_policy}: dinov3_semseg -> {_semseg_aug_status}")
    if args.repo_dir: REPO_CACHE = Path(args.repo_dir).expanduser().resolve()

    ORIG_DATA = Path(args.data).resolve()
    if not ORIG_DATA.is_dir(): print("❌ Data directory not found"); sys.exit(1)
    DATA_YAML = ORIG_DATA / 'data.yaml'
    if not DATA_YAML.exists(): print("❌ data.yaml not found"); sys.exit(1)
    DATA_NAME = ORIG_DATA.name

    print("🔄 Converting YOLO → COCO panoptic …")
    COCO_ROOT, CLASSES = yolo_to_coco_panoptic(DATA_YAML, DATA_NAME)
    print(f"✅ COCO panoptic ready (classes: {CLASSES})")

    REPO_DIR = ensure_repo()
    DINO_MODEL = pick_backbone(args.model)
    DINOV3_DIR, hub_warn = resolve_dinov3_hub_dir(REPO_DIR, args.dinov3_dir)
    if hub_warn:
        print(hub_warn)
    DINOV3_WEIGHTS = resolve_dinov3_weights(DINO_MODEL, REPO_CACHE, args.dinov3_weights, args.dinov3_dir)
    if not DINOV3_WEIGHTS:
        available = available_dinov3_weights(REPO_CACHE, args.dinov3_dir)
        scanned_dirs = []
        for d in [
            Path(args.dinov3_dir).expanduser() if args.dinov3_dir else None,
            Path.home() / "bench_any" / "dinov3_weights",
            Path.home() / "dinov3_weights",
            Path(REPO_CACHE) / "dinov3_weights",
        ]:
            if d and d not in scanned_dirs:
                scanned_dirs.append(d)
        print(f"❌ DINOv3 weights not found for requested backbone: {DINO_MODEL}")
        print(f"   Expected filename pattern: {DINO_MODEL}*.pth")
        print("   Searched directories:")
        for d in scanned_dirs:
            print(f"     {d}")
        if available:
            print("   Available local DINOv3 .pth weights:")
            for model_name, path in sorted(available.items()):
                print(f"     {model_name}: {path}")
        print("   The available local weights decide which semseg aliases can run.")
        print("   For gated Meta DINOv3 weights, download from "
              "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and place the "
              f"file in /home/vorse/bench_any/dinov3_weights/ as {DINO_MODEL}*.pth, or pass "
              "--dinov3_weights /full/path/to/file.pth.")
        print("   Use a dinov3-semseg-* alias whose backbone is listed above, or pass "
              "--dinov3_weights directly.")
        sys.exit(1)
    print(f"✅ Using DINOv3 weights: {DINOV3_WEIGHTS}")
    print(f"🔗 Using DINOv3 torch.hub source: {DINOV3_DIR}")

    bs = max(1, int(args.batch_sz)) if args.batch_sz >= 1 else 1
    patch_config(REPO_DIR, COCO_ROOT, bs, args.lr, args.epochs,
                 DINOV3_DIR, DINOV3_WEIGHTS, DINO_MODEL)

    MODEL_STEM = Path(args.model).stem
    _env_run = os.environ.get("BENCH_RUN_NAME")
    RUN_NAME = _env_run if _env_run else f"{DATA_NAME[:15]}_{MODEL_STEM}"
    TASK_DIR = "segment"
    # RUN_DIR on disk (not /dev/shm): checkpoints survive run, no tmpfs pressure.
    # Routed under ~/bench_any/ so nothing lands in the project / dataset directory.
    RUN_DIR = Path.home() / 'bench_any' / 'runs' / TASK_DIR / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('WANDB_PROJECT', "DINOv3-SemSeg")

    runner = Path(REPO_DIR) / "_bench_runner.py"
    runner.write_text(RUNNER_SRC)
    cmd = [sys.executable, str(runner),
           "--epochs", str(args.epochs), "--batch_size", str(bs), "--lr", str(args.lr),
           "--threads", str(args.threads), "--out_dir", str(RUN_DIR)]
    if args.resume: cmd.append("--resume")
    if args.inference_only: cmd.append("--inference_only")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_DIR}:{env.get('PYTHONPATH', '')}"
    # Let the templated runner script import bench_eval from this project dir,
    # which is not on PYTHONPATH when launched via subprocess from REPO_DIR.
    env["BENCH_EVAL_DIR"] = str(Path(__file__).resolve().parent)
    env["BENCH_AUG_POLICY"] = args.aug_policy
    print(f"🚂 dinov3-semseg")
    rc = subprocess.call(cmd, cwd=str(REPO_DIR), env=env)
    if rc != 0: print(f"⚠️ runner rc={rc}; checking for partial results.csv")

    # RUN_DIR is already under ~/bench_any/, so no separate mirror is needed.
    print(f"✅ Results: {RUN_DIR}")

    print("🎉 Complete!")
    if rc != 0 and not (RUN_DIR / "results.csv").is_file():
        sys.exit(rc)
