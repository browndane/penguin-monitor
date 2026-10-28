#!/usr/bin/env python3
# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Time the SAM2 mask-refinement step of the semi-automated annotation pipeline.

This measures the per-image cost the SAM2-refine step adds: prompt SAM2's image
predictor with each detector box on an image, get back a tight mask. It does NOT
measure the human expert-verification step (that needs a person; cite literature
for it in the paper). Output: per-image ms (mean +/- std), cold-start (first
image, includes model load + CUDA warmup) vs warm, and per-box ms.

Honest scope (for P1 CrossParadigm...tex annotation-efficiency subsection):
this is the automation's contribution — the part the detector-proposal ->
SAM2-refine pipeline adds on top of a person reviewing proposals.

Run with the full-stack venv (has sam2 + hydra + cuda):
  ~/.virtualenvs/cvML/bin/python penguins/sam2_refine_timing.py \
    --images_dir <test/images> --boxes_json <RF-DETR _bench_predictions_fixed.json> \
    --n 100 --out penguins/research/sam2_refine_timing.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

_SAM2_ROOT = "/home/user/sam2"
if _SAM2_ROOT not in sys.path:
    sys.path.insert(0, _SAM2_ROOT)


def _load_boxes_by_image(boxes_json, gt_json):
    """Return {image_filename: [[x1,y1,x2,y2], ...]} from a COCO-style pred file.

    `boxes_json` is a flat COCO annotation list: [{image_id, bbox:[x,y,w,h], ...}].
    `gt_json` is the COCO GT with the images array for image_id -> file_name.
    bbox xywh -> xyxy."""
    gt = json.load(open(gt_json))
    if isinstance(gt, dict):
        id2name = {im["id"]: im["file_name"] for im in gt.get("images", [])}
    else:
        id2name = {im["id"]: im["file_name"] for im in gt if isinstance(im, dict) and "id" in im}
    d = json.load(open(boxes_json))
    anns = d if isinstance(d, list) else d.get("annotations", [])
    by_img = {}
    for a in anns:
        fn = id2name.get(a["image_id"])
        if fn is None:
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(fn, []).append([float(x), float(y), float(x + w), float(y + h)])
    return by_img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True, help="dir of test images")
    ap.add_argument("--boxes_json", required=True,
                    help="COCO-format bbox preds (xywh) to prompt SAM2 with")
    ap.add_argument("--gt_json", required=True,
                    help="COCO GT json with images array (for image_id -> file_name)")
    ap.add_argument("--sam2_ckpt", default=None, help="override SAM2 ckpt (default: sam2_video_propagate constant)")
    ap.add_argument("--sam2_cfg", default=None, help="override SAM2 cfg path (default: sam2_video_propagate constant)")
    ap.add_argument("--n", type=int, default=100, help="number of images to time")
    ap.add_argument("--out", required=True, help="output JSON with timing stats")
    args = ap.parse_args()

    # Reuse sam2_video_propagate's SAM2 constants (the validated config/ckpt
    # paths + configs root) so this script matches the demo pipeline exactly.
    # sam2_video_propagate is a sibling module in penguins/.
    import sam2_video_propagate as _svp
    sam2_ckpt = args.sam2_ckpt or _svp.SAM2_CKPT
    sam2_cfg = args.sam2_cfg or _svp.SAM2_CONFIG
    configs_root = _svp.SAM2_CONFIGS_ROOT

    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=configs_root, version_base="1.2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import numpy as np
    import cv2
    import torch

    cfg_rel = str(Path(sam2_cfg).relative_to(configs_root)) if Path(sam2_cfg).is_absolute() else sam2_cfg

    boxes_by_img = _load_boxes_by_image(args.boxes_json, args.gt_json)
    images_dir = Path(args.images_dir)
    # Only images that have boxes + exist on disk.
    candidates = [(images_dir / fn, boxes) for fn, boxes in boxes_by_img.items()
                  if (images_dir / fn).is_file()]
    if not candidates:
        print(f"[timing] no images with boxes found in {images_dir}")
        return
    candidates = candidates[: args.n]
    print(f"[timing] timing SAM2 refine over {len(candidates)} images, "
          f"{sum(len(b) for _, b in candidates)} total boxes")

    sam2 = build_sam2(cfg_rel, sam2_ckpt, device="cuda")
    predictor = SAM2ImagePredictor(sam2)

    per_image_ms = []
    per_box_ms = []
    cold_ms = None
    for idx, (img_path, boxes) in enumerate(candidates):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        predictor.set_image(rgb)
        # Prompt with all boxes on this image at once (batched point prompts via
        # box arg). Each box -> one mask. This is the exact annotation-refine op.
        import numpy as _np
        box_arr = _np.array(boxes, dtype=_np.float32) if boxes else _np.zeros((0, 4), _np.float32)
        masks, _, _ = predictor.predict(box=box_arr, multimask_output=False)
        if masks.ndim == 4:
            masks = masks
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        per_image_ms.append(dt_ms)
        per_box_ms.append(dt_ms / max(1, len(boxes)))
        if idx == 0:
            cold_ms = dt_ms
    warm = per_image_ms[1:] if len(per_image_ms) > 1 else per_image_ms

    def _stats(xs):
        xs = list(xs)
        n = len(xs)
        if n == 0:
            return {"n": 0}
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        return {"n": n, "mean_ms": round(mean, 2),
                "std_ms": round(var ** 0.5, 2),
                "min_ms": round(min(xs), 2), "max_ms": round(max(xs), 2)}

    out = {
        "scope": {
            "what": "SAM2 image-predictor box-prompted mask refine per image",
            "not_measured": "human expert-verification step (needs a person; cite lit)",
            "n_images": len(per_image_ms),
            "n_boxes_total": sum(len(b) for _, b in candidates[: len(per_image_ms)]),
            "sam2_ckpt": sam2_ckpt,
            "sam2_cfg": sam2_cfg,
        },
        "per_image_ms": _stats(per_image_ms),
        "per_image_warm_ms": _stats(warm),
        "cold_start_ms": cold_ms,
        "per_box_ms": _stats(per_box_ms),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[timing] wrote {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
