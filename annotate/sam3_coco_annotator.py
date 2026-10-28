# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""
Auto Object Segmentation: If `<folder>/_annotations.coco.json` exists → use it as prompts (bboxes → centroids → points, else keypoints/point). Else --prompt fallback
python sam3_coco_annotator.py <image_dir> [--prompt "a <thing>"] [--auto-splits]
python point_crowdsource_annotate/sam3_coco_annotator.py /mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/RoboflowDatasets/RF100-VL/penguin_seg/ --auto-splits --prompt "a penguin"


Notes:
  * `--class-name` is auto-derived from --prompt ("a penguin" → "penguin") and from the input JSON's categories in box/point modes; pass it to override.
  * Existing `_annotations.coco.json` is renamed to `.jsonbk` (kept if already there); pass `--overwrite` to wipe the active JSON and restart (backup is always preserved).
  * SAM 3 weights cache: /mnt/LinuxBackup/sam3 (~2 GB). OOM frames retried on CPU.
"""

import argparse
import json
import os
import sys
import time

# Redirect HuggingFace caches + tame CUDA allocator BEFORE torch/transformers import.
SAM3_CACHE_DIR = "/mnt/LinuxBackup/sam3"
os.makedirs(SAM3_CACHE_DIR, exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", SAM3_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(SAM3_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(SAM3_CACHE_DIR, "hub"))

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def _ensure_sam3_transformers():
    try:
        from transformers import Sam3Model, Sam3Processor  # noqa: F401
        return
    except Exception:
        pass
    print("Installing transformers from GitHub (needed for SAM 3)...", flush=True)
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "git+https://github.com/huggingface/transformers",
    ])


_ensure_sam3_transformers()

import torch
from torchvision.ops import nms as tv_nms
from transformers import Sam3Model, Sam3Processor


MODEL_CANDIDATES = ["facebook/sam3.1", "facebook/sam3_1", "facebook/sam3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="SAM 3 → COCO auto-labeller (text / boxes / points prompts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_dir", help="Directory containing images.")
    p.add_argument("--output", default=None,
                   help="Output COCO JSON path (default: <image_dir>/_annotations.coco.json)")
    p.add_argument("--ext", default=".jpg,.jpeg,.png,.JPG,.JPEG,.PNG",
                   help="Comma-separated image extensions.")

    p.add_argument("--auto-splits", action="store_true",
                   help="Treat image_dir as a parent of split subfolders (train/ valid/ test/); "
                        "each is processed in place, mode inferred from its JSON if any else "
                        "text mode with --prompt.")

    p.add_argument("--dry-run", action="store_true", help="With --auto-splits: print the plan and exit without loading SAM 3.")
    p.add_argument("--overwrite", action="store_true",
                   help="Start from scratch: ignore any existing sibling _annotations.coco.json "
                        "for mode-detection (so --prompt forces text mode), and delete the active "
                        "output JSON before writing. The .jsonbk backup is always preserved.")

    p.add_argument("--prompt", default=None,
                   help="Text prompt (text mode). If passed, forces text mode and ignores any "
                        "sibling _annotations.coco.json. Example: --prompt \"a penguin\".")
    p.add_argument("--class-name", default=None,
                   help="COCO category name (default: derived from --prompt, or from the "
                        "input JSON's categories[0].name in box/point mode).")

    p.add_argument("--score", type=float, default=0.25, help="SAM 3 confidence threshold.")
    p.add_argument("--mask-threshold", type=float, default=0.3, help="Mask-logits binarisation. Lower is less broken")
    p.add_argument("--max-dim-frac", type=float, default=1.0,
                   help="Reject masks whose bbox W or H > frac * image dim.")
    p.add_argument("--min-area", type=int, default=144, help="Reject masks smaller than this.")
    p.add_argument("--nms", type=float, default=0.7, help="Class-agnostic NMS IoU (text mode).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sam3():
    for cand in MODEL_CANDIDATES:
        try:
            print(f"Trying {cand} on {DEVICE} (cache={os.environ['HF_HUB_CACHE']})...")
            proc = Sam3Processor.from_pretrained(cand)
            mdl = Sam3Model.from_pretrained(cand, dtype=DTYPE).to(DEVICE).eval()
            print(f"SAM 3 loaded: {cand} ({DTYPE}).")
            return mdl, proc, cand
        except Exception as e:
            print(f"  {cand} unavailable ({type(e).__name__}: {str(e).splitlines()[0]})")
    raise RuntimeError(f"Could not load any SAM 3 checkpoint from {MODEL_CANDIDATES}.")


def derive_class_name(prompt):
    """'a penguin' / 'an orca' / 'the king penguin' → 'penguin' / 'orca' / 'king penguin'."""
    if not prompt:
        return "object"
    s = prompt.strip()
    low = s.lower()
    for art in ("a ", "an ", "the "):
        if low.startswith(art):
            return s[len(art):].strip() or "object"
    return s or "object"


def passes_sanity(mask_np, img_h, img_w, min_area, max_dim_frac):
    """Reject masks that are too small or that span too much of either axis."""
    if int(mask_np.sum()) < min_area:
        return False
    ys, xs = np.where(mask_np > 0)
    if ys.size == 0:
        return False
    bb_h = int(ys.max() - ys.min() + 1)
    bb_w = int(xs.max() - xs.min() + 1)
    return (bb_w / img_w <= max_dim_frac) and (bb_h / img_h <= max_dim_frac)


def discover_images(image_dir, ext_csv):
    exts = tuple(e.strip() for e in ext_csv.split(",") if e.strip())
    out = [fn for fn in os.listdir(image_dir)
           if fn.endswith(exts) and os.path.isfile(os.path.join(image_dir, fn))]
    return sorted(out)


def load_prompt_coco(path):
    """Load a COCO JSON → (images, boxes_per_img_id, points_per_img_id, categories)."""
    with open(path) as f:
        coco = json.load(f)
    boxes_per_img, points_per_img = {}, {}
    for a in coco.get("annotations", []):
        img_id = a.get("image_id")
        if a.get("bbox"):
            x, y, w, h = a["bbox"][:4]
            boxes_per_img.setdefault(img_id, []).append([x, y, x + w, y + h])
        kpts = a.get("keypoints")
        if kpts:
            # COCO keypoints = [x1,y1,v1, x2,y2,v2, ...]; v=0 means unlabeled.
            for i in range(0, len(kpts), 3):
                if i + 2 < len(kpts) and kpts[i + 2] > 0:
                    points_per_img.setdefault(img_id, []).append([kpts[i], kpts[i + 1]])
        elif a.get("point") and len(a["point"]) >= 2:
            points_per_img.setdefault(img_id, []).append([a["point"][0], a["point"][1]])
    return (coco.get("images", []), boxes_per_img, points_per_img,
            coco.get("categories", []))


def cuda_empty():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def mask_to_coco_ann(ann_id, img_id, mask_np, score=None):
    """Encode a uint8 [H,W] mask as a COCO annotation dict."""
    rle = mask_util.encode(np.asfortranarray(mask_np))
    rle["counts"] = rle["counts"].decode("utf-8")
    ann = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "segmentation": rle,
        "bbox": mask_util.toBbox(rle).tolist(),
        "area": int(mask_util.area(rle)),
        "iscrowd": 0,
    }
    if score is not None:
        ann["score"] = float(score)
    return ann


def text_forward_with_oom_fallback(model, processor, inputs, score_thr, mask_thr, filename):
    """Run SAM 3 forward + post_process_instance_segmentation; retry on CPU on CUDA-OOM."""
    try:
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=score_thr, mask_threshold=mask_thr,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        return results
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] {filename}: retrying post-process on CPU.", flush=True)
        cuda_empty()
        try:
            with torch.inference_mode():
                outputs = model(**inputs)
            outputs = outputs.to("cpu")
            cuda_empty()
            return processor.post_process_instance_segmentation(
                outputs, threshold=score_thr, mask_threshold=mask_thr,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
        except Exception as e2:
            print(f"  [SKIP] {filename}: GPU+CPU both failed ({type(e2).__name__}: {e2})", flush=True)
            return None


# ---------------------------------------------------------------------------
# Per-image handlers (one returns appended-annotation count + log-tail string)
# ---------------------------------------------------------------------------
def process_text(model, processor, pil_image, img_id, ann_id, coco, args, filename):
    w, h = pil_image.size
    inputs = processor(images=pil_image, text=args.prompt, return_tensors="pt").to(DEVICE)
    results = text_forward_with_oom_fallback(
        model, processor, inputs, args.score, args.mask_threshold, filename
    )
    del inputs
    cuda_empty()
    if results is None:
        return ann_id, 0, "(skipped)"

    masks, scores, boxes = results["masks"], results["scores"], results["boxes"]
    n_raw = int(masks.shape[0])

    # Sanity filter BEFORE NMS so nonsense masks can't suppress real ones.
    sane = []  # list of (mask_np, box_tensor, score_tensor)
    for i in range(n_raw):
        m_np = masks[i].detach().cpu().numpy().astype(np.uint8)
        if m_np.ndim == 3:
            m_np = m_np[0]
        if passes_sanity(m_np, h, w, args.min_area, args.max_dim_frac):
            sane.append((m_np, boxes[i], scores[i]))
    n_sanity_drop = n_raw - len(sane)

    # Class-agnostic NMS.
    if sane:
        bf = torch.stack([s[1] for s in sane]).detach().float().cpu()
        sf = torch.stack([s[2] for s in sane]).detach().float().cpu()
        keep = tv_nms(bf, sf, args.nms).tolist()
    else:
        keep = []
    n_nms_drop = len(sane) - len(keep)

    s_min, s_max, n_good = 1.0, 0.0, 0
    for i in keep:
        m_np, _, s_t = sane[i]
        score = float(s_t.item()) if hasattr(s_t, "item") else float(s_t)
        s_min, s_max = min(s_min, score), max(s_max, score)
        coco["annotations"].append(mask_to_coco_ann(ann_id, img_id, m_np, score))
        ann_id += 1
        n_good += 1

    del masks, scores, boxes
    cuda_empty()

    score_str = f"score={s_min:.2f}–{s_max:.2f}" if n_good else "no detections"
    log = (f"(raw={n_raw}, sanity-drop={n_sanity_drop}, "
           f"nms-drop={n_nms_drop}) ({score_str})")
    return ann_id, n_good, log


def process_prompted(model, processor, pil_image, img_id, ann_id, coco, args, filename,
                     boxes=None, points=None):
    """Box- or point-prompted mode: one mask per prompt, no score/NMS."""
    w, h = pil_image.size
    if boxes is None:
        r = max(8.0, 0.03 * min(w, h))
        boxes = [[max(0.0, float(px) - r), max(0.0, float(py) - r),
                  min(float(w) - 1.0, float(px) + r),
                  min(float(h) - 1.0, float(py) + r)]
                 for px, py in points]
    inputs = processor(images=pil_image, input_boxes=[boxes],
                       return_tensors="pt").to(DEVICE)
    
    # Processor returns input_boxes/points as float32, but model is in bf16 (DTYPE).
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and v.is_floating_point() and v.dtype != DTYPE:
            inputs[k] = v.to(DTYPE)
    try:
        with torch.inference_mode():
            outputs = model(**inputs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [SKIP-OOM] {filename}: {e}", flush=True)
        cuda_empty()
        return ann_id, 0, "(skipped)"

    pred_masks = (outputs.pred_masks if hasattr(outputs, "pred_masks")
                  else outputs["pred_masks"]).detach().to("cpu").float()

    if hasattr(processor, "post_process_masks"):
        resized = processor.post_process_masks(
            pred_masks,
            original_sizes=inputs["original_sizes"].to("cpu"),
            reshaped_input_sizes=inputs["reshaped_input_sizes"].to("cpu"),
        )[0].squeeze(1).numpy().astype(np.uint8)
    else:
        binm = (pred_masks > args.mask_threshold).numpy().astype(np.uint8)
        binm = binm.reshape(-1, binm.shape[-2], binm.shape[-1])
        resized = np.stack([
            np.array(Image.fromarray(m * 255).resize((w, h), Image.NEAREST)) > 0
            for m in binm
        ]).astype(np.uint8)

    del outputs, inputs
    cuda_empty()

    n_raw = int(resized.shape[0])
    n_good, n_sanity_drop = 0, 0
    for idx in range(n_raw):
        m_np = resized[idx]
        if not passes_sanity(m_np, h, w, args.min_area, args.max_dim_frac):
            n_sanity_drop += 1
            continue
        coco["annotations"].append(mask_to_coco_ann(ann_id, img_id, m_np))
        ann_id += 1
        n_good += 1
    return ann_id, n_good, f"(prompts={n_raw}, sanity-drop={n_sanity_drop})"


# ----------------------------------------------------------------------------------------------------------------------
# Split planning + atomic rename (backing up of json annotations as jsonbk (also prevents upload wrong json to Roboflow)
# ----------------------------------------------------------------------------------------------------------------------
def plan_from_json(json_path, has_text_fallback):
    """Infer mode (boxes → points → text) from a COCO JSON's annotation contents."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        return None, f"unreadable ({type(e).__name__}: {e})"
    anns = data.get("annotations", [])
    n_imgs = len(data.get("images", []))
    n_box = sum(1 for a in anns if a.get("bbox"))
    n_pts = sum(1 for a in anns if a.get("keypoints") or a.get("point"))
    if n_box:
        return "boxes", f"{n_imgs} images, {n_box} bboxes"
    if n_pts:
        return "points", f"{n_imgs} images, {n_pts} point-anns"
    if has_text_fallback:
        return "text", f"{n_imgs} images (no prompts in JSON; will use --prompt)"
    return None, f"{n_imgs} images, no bbox/point prompts and no --prompt fallback"


def iter_split_folders(root, ext_csv):
    """Yield (subfolder, json_path_or_None) for each immediate subfolder that has
    either an `_annotations.coco.json` or at least one image with one of `ext_csv`."""
    exts = tuple(e.strip() for e in ext_csv.split(",") if e.strip())
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        jp = os.path.join(sub, "_annotations.coco.json")
        if os.path.isfile(jp):
            yield sub, jp
            continue
        try:
            has_img = any(fn.endswith(exts) and os.path.isfile(os.path.join(sub, fn))
                          for fn in os.listdir(sub))
        except OSError:
            has_img = False
        if has_img:
            yield sub, None


def atomic_install(final_path, tmp_path, overwrite=False):
    """
    Move `tmp_path` onto `final_path`. If `final_path` already exists, first
    rename it to `final_path + 'bk'` — but skip the rename if a backup already
    exists (keep the older backup, just overwrite the active JSON).
    If `overwrite=True`, delete the active JSON first so the run starts from
    scratch. The `.jsonbk` backup is always preserved.
    """
    bk_path = final_path + "bk"
    if overwrite:
        if os.path.exists(final_path):
            os.remove(final_path)
            print(f"  [--overwrite] removed {os.path.basename(final_path)} "
                  f"(backup {os.path.basename(bk_path)} preserved)")
    elif os.path.exists(final_path):
        if os.path.exists(bk_path):
            print(f"  [backup exists] {os.path.basename(bk_path)} kept; overwriting active JSON.")
            os.remove(final_path)
        else:
            os.rename(final_path, bk_path)
            print(f"  Renamed old {os.path.basename(final_path)} → {os.path.basename(bk_path)}")
    os.rename(tmp_path, final_path)


# ---------------------------------------------------------------------------
# Per-folder runner (one call = one output JSON)
# ---------------------------------------------------------------------------
def run_folder(args, model, processor, model_id, image_dir, output_path,
               mode, prompt_json_path, label_prefix=""):
    """
    Build the work list for `image_dir`, run SAM 3, write `output_path`
    (via a `.new` temp + atomic rename so the existing JSON is preserved as
    `.jsonbk`). Returns (done, ann_id-1, skipped).
    """
    json_cats = []
    if mode in ("boxes", "points"):
        imgs, boxes_per_img, points_per_img, json_cats = load_prompt_coco(prompt_json_path)
        work = [(im["id"], im["file_name"]) for im in imgs
                if os.path.exists(os.path.join(image_dir, im["file_name"]))]
        src_count = len(imgs)
        prompt_count = sum(len(v) for v in (
            boxes_per_img if mode == "boxes" else points_per_img).values())
        mode_label = "Boxes→points-prompt" if mode == "boxes" else "Points-prompt"
        print(f"{label_prefix}{mode_label} mode: "
              f"{len(work)}/{src_count} images matched on disk, {prompt_count} prompts.")
        
        # Fallback: JSON references none of the images on disk (e.g. stale filenames in the input COCO).
        if not work:
            disk_imgs = discover_images(image_dir, args.ext)
            if disk_imgs and args.prompt:
                print(f"{label_prefix}No JSON images matched on disk; "
                      f"falling back to text-prompt mode on {len(disk_imgs)} "
                      f"disk image(s) with prompt={args.prompt!r}.")
                mode = "text"
                boxes_per_img, points_per_img = {}, {}
                work = list(enumerate(disk_imgs, start=1))
    else:
        boxes_per_img, points_per_img = {}, {}
        filenames = discover_images(image_dir, args.ext)
        work = list(enumerate(filenames, start=1))
        print(f"{label_prefix}Text-prompt mode: prompt={args.prompt!r}, score>={args.score}, "
              f"{len(work)} images.")

    if not work:
        print(f"{label_prefix}No images to process; skipping.")
        return 0, 0, 0

    # Effective class name: explicit --class-name wins; else input JSON's categories[0].name (box/points); else derived from --prompt.
    if args.class_name:
        class_name = args.class_name
    elif mode in ("boxes", "points") and json_cats:
        class_name = json_cats[0].get("name") or derive_class_name(args.prompt)
    else:
        class_name = derive_class_name(args.prompt)
    # Stash on args so nested helpers can read it if ever needed.
    args.class_name = class_name

    coco = {
        "info": {
            "year": time.strftime("%Y"), "version": "1.0",
            "description": f"{model_id} auto-labelled ({mode}-prompt; class={class_name})",
        },
        "licenses": [{"id": 1, "name": "", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": class_name}],
    }

    ann_id, done, skipped = 1, 0, 0
    total = len(work)
    start = time.time()

    for img_id, rel_name in work:
        filepath = os.path.join(image_dir, rel_name)
        if not os.path.exists(filepath):
            skipped += 1
            continue

        pil_image = Image.open(filepath).convert("RGB")
        w, h = pil_image.size
        coco["images"].append({"id": img_id, "file_name": rel_name, "width": w, "height": h})

        if mode == "text":
            ann_id, n_good, log_tail = process_text(
                model, processor, pil_image, img_id, ann_id, coco, args, rel_name
            )
        elif mode == "boxes":
            # Convert each bbox to its centroid (point) so the sanity sieve filters spurious masks
            boxes = boxes_per_img.get(img_id, [])
            if not boxes:
                n_good, log_tail = 0, "(no box prompts)"
            else:
                centroids = [[(x1 + x2) * 0.5, (y1 + y2) * 0.5]
                             for x1, y1, x2, y2 in boxes]
                ann_id, n_good, log_tail = process_prompted(
                    model, processor, pil_image, img_id, ann_id, coco, args, rel_name,
                    points=centroids,
                )
        else:  # points
            pts = points_per_img.get(img_id, [])
            if not pts:
                n_good, log_tail = 0, "(no point prompts)"
            else:
                ann_id, n_good, log_tail = process_prompted(
                    model, processor, pil_image, img_id, ann_id, coco, args, rel_name,
                    points=pts,
                )

        done += 1
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"{label_prefix}[{done}/{total}] {rel_name} | {n_good} kept {log_tail} | "
            f"{ann_id - 1} total | {rate:.2f} img/s | ETA: {eta / 60:.1f} min",
            flush=True,
        )

    # Write to a temp path, then atomically install (rename-old-to-.jsonbk, new → active).
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".new"
    with open(tmp_path, "w") as f:
        json.dump(coco, f)
    atomic_install(output_path, tmp_path, overwrite=getattr(args, "overwrite", False))

    elapsed = time.time() - start
    print(f"{label_prefix}Done: {done} images, {ann_id - 1} annotations in {elapsed / 60:.1f} min")
    print(f"{label_prefix}Skipped: {skipped}")
    print(f"{label_prefix}Saved to: {output_path}")
    return done, ann_id - 1, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if not os.path.isdir(args.image_dir):
        sys.exit(f"Image dir not found: {args.image_dir}")

    # ----- Auto-splits: iterate split subfolders (JSON-bearing or image-only) -----
    if args.auto_splits:
        splits = list(iter_split_folders(args.image_dir, args.ext))
        if not splits:
            print(f"--auto-splits: no split subfolders under {args.image_dir}; "
                  f"falling back to single-folder mode on image_dir itself.")
            args.auto_splits = False  # fall through to single-folder branch below

    if args.auto_splits:
        has_text_fallback = bool(args.prompt)
        plans = []  # (sub, jp_or_None, mode, detail)
        for sub, jp in splits:
            # --overwrite or explicit --prompt: ignore sibling JSON and use text mode.
            if args.overwrite or args.prompt is not None:
                jp = None
            if jp is not None:
                mode, detail = plan_from_json(jp, has_text_fallback)
            else:
                n_imgs = len(discover_images(sub, args.ext))
                if has_text_fallback:
                    mode, detail = "text", f"{n_imgs} images (no JSON; --prompt={args.prompt!r})"
                else:
                    mode, detail = None, f"{n_imgs} images, no JSON and no --prompt fallback"
            plans.append((sub, jp, mode, detail))

        print(f"\n--auto-splits: found {len(splits)} split folder(s) under {args.image_dir}")
        for sub, jp, mode, detail in plans:
            name = os.path.basename(sub)
            tag = "SKIP" if mode is None else mode
            print(f"  [{tag:>6}] {name}: {detail}")
        if args.dry_run:
            print("\n--dry-run: not loading SAM 3, exiting.")
            return

        usable = [p for p in plans if p[2] is not None]
        if not usable:
            sys.exit("No splits have a usable prompt source; exiting.")

        model, processor, model_id = load_sam3()
        grand_start = time.time()
        grand_done = grand_anns = 0
        for i, (sub, jp, mode, _detail) in enumerate(usable, start=1):
            name = os.path.basename(sub)
            prefix = f"[split {i}/{len(usable)} {name}] "
            print(f"\n{'=' * 60}\n{prefix}starting ({mode}-prompt)\n{'=' * 60}")
            output_path = os.path.join(sub, "_annotations.coco.json")
            # Boxes/points prompt is the active output path; run_folder reads it into memory first, then writes `.new`, then renames the original to `.jsonbk`.
            done, anns, _ = run_folder(
                args, model, processor, model_id, sub, output_path, mode, jp, prefix
            )
            grand_done += done
            grand_anns += anns

        print(f"\n{'=' * 60}")
        print(f"All {len(usable)} splits done: {grand_done} images, {grand_anns} annotations "
              f"in {(time.time() - grand_start) / 60:.1f} min.")
        return

    # ----- Single-folder mode -----
    # Same rule as per-split in --auto-splits: if `_annotations.coco.json`
    # sits next to the images, auto-detect boxes/points from it; else text.
    # Exception: --overwrite means "start from scratch", so we intentionally
    # ignore any sibling JSON for mode-detection and fall through to text.
    output_path = args.output or os.path.join(args.image_dir, "_annotations.coco.json")
    sibling_json = os.path.join(args.image_dir, "_annotations.coco.json")
    has_text_fallback = bool(args.prompt)
    # Explicit --prompt wins over sibling JSON (same semantics as --overwrite):
    # boxes/points mode skips score/NMS gates and amplifies any noise in the
    # input JSON, which is rarely what the user wants when they typed --prompt.
    ignore_sibling = args.overwrite or args.prompt is not None
    if os.path.isfile(sibling_json) and not ignore_sibling:
        mode, detail = plan_from_json(sibling_json, has_text_fallback)
        prompt_json = sibling_json if mode in ("boxes", "points") else None
    else:
        n_imgs = len(discover_images(args.image_dir, args.ext))
        if has_text_fallback:
            mode, detail, prompt_json = "text", f"{n_imgs} images (no JSON; --prompt={args.prompt!r})", None
        else:
            mode, detail, prompt_json = None, f"{n_imgs} images, no JSON and no --prompt fallback", None
    tag = "SKIP" if mode is None else mode
    print(f"[{tag:>6}] {os.path.basename(args.image_dir.rstrip('/'))}: {detail}")
    if args.dry_run:
        print("\n--dry-run: not loading SAM 3, exiting.")
        return
    if mode is None:
        sys.exit("No usable prompt source (no JSON, no --prompt); exiting.")

    model, processor, model_id = load_sam3()
    run_folder(args, model, processor, model_id, args.image_dir, output_path,
               mode, prompt_json)


if __name__ == "__main__":
    main()
