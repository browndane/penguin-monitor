"""Generate the 3 evaluable prediction rows on p5_gold200 (mask mAP@0.5 re-eval).

Rows (pred field names):
  - sam3_pred       : SAM3 zero-shot text-prompt "a penguin" + sanity sieve (score 0.25)
  - yolo11x_pred    : YOLOv11x-seg supervised ckpt (penguin-seg-combined_yolo11x-seg/weights/best.pt)
  - rtdetr_sam3_pred: RT-DETR-x boxes (penguin-seg-combined_rtdetr-x/weights/best.pt) + SAM3 box-prompt masks

All predictions stored as fo.Detections with `mask` (binary, image-res) and `confidence`,
keyed to the existing p5_gold200 samples (match by filepath). Use --row to pick one.

Sanity sieve (SAM3 row): min_area=64, nms_iou=0.8, max_dim_frac=1.0 (same as the autolabel pass).

GPU: SAM3 in bfloat16, downsample long edge to 2448 for >4096px images to avoid OOM
(same OOM-avoidance as prep_04). YOLOv11x-seg runs native ultralytics (retina_masks=True).
RT-DETR gives boxes; SAM3 box-prompt turns boxes into masks.
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")

from pathlib import Path
import numpy as np
import cv2
import fiftyone as fo

DS = "p5_gold200"
YOLO_CKPT = "/home/vorse/bench_any/runs/segment/penguin-seg-combined_yolo11x-seg/weights/best.pt"
RTDETR_CKPT = "/home/vorse/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"
LONG_EDGE_CAP = 2448  # downsample for SAM3 on huge frames to avoid OOM
SCORE = 0.25
MASK_THR = 0.5


def sieve(masks, scores, boxes, min_area=64, nms_iou=0.8, max_dim_frac=1.0):
    """Sanity sieve: drop tiny, heavily-overlapping, or full-frame masks. Returns keep idx."""
    keep = []
    n = len(masks)
    if n == 0:
        return keep
    areas = np.array([m.sum() for m in masks])
    H, W = masks[0].shape
    # area gate
    area_ok = areas >= min_area
    # max-dim gate (mask spans more than max_dim_frac of image in either axis)
    dim_ok = np.ones(n, bool)
    for i, m in enumerate(masks):
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            dim_ok[i] = False
            continue
        if (xs.max()-xs.min()+1) > max_dim_frac*W or (ys.max()-ys.min()+1) > max_dim_frac*H:
            dim_ok[i] = False
    # NMS by score: sort desc, suppress lower-scoring overlapping boxes
    order = np.argsort(-scores)
    suppressed = np.zeros(n, bool)
    for idx in order:
        if suppressed[idx]:
            continue
        for jdx in order:
            if jdx == idx or suppressed[jdx]:
                continue
            if _iou(boxes[idx], boxes[jdx]) > nms_iou:
                suppressed[jdx] = True
    for i in range(n):
        if area_ok[i] and dim_ok[i] and not suppressed[i]:
            keep.append(i)
    return keep


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0.0, x1-x0); ih = max(0.0, y1-y0)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def mask_to_detection(mask, score, W, H):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(x1-x0+1, 1), max(y1-y0+1, 1)
    return fo.Detection(
        label="penguin",
        bounding_box=[x0/W, y0/H, bw/W, bh/H],
        mask=mask,
        confidence=float(score),
    )


def get_sam3():
    import torch
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    mdl = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda").eval()
    return proc, mdl


def sam3_text_predict(proc, mdl, img_np, score, mask_thr):
    """SAM3 text-prompt 'a penguin', return masks [N,H,W] bool, scores [N], boxes [N,4] abs.
    Mirrors the working prep_04 pattern (text= kwarg, mdl(**inputs), target_sizes)."""
    import torch
    from PIL import Image
    H0, W0 = img_np.shape[:2]
    long_edge = max(W0, H0)
    if long_edge > LONG_EDGE_CAP:
        scale = LONG_EDGE_CAP / long_edge
        Wds, Hds = int(W0*scale), int(H0*scale)
        img_ds = cv2.resize(img_np, (Wds, Hds))
    else:
        img_ds = img_np
        Wds, Hds = W0, H0
    im = Image.fromarray(img_ds[:,:,::-1]).convert("RGB")
    with torch.inference_mode():
        inputs = proc(images=im, text="a penguin", return_tensors="pt").to("cuda")
        out = mdl(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=score, mask_threshold=mask_thr, target_sizes=[(Hds, Wds)])[0]
    masks_ds = res["masks"].cpu().numpy().astype(bool) if "masks" in res else np.zeros((0, Hds, Wds), bool)
    sc = res["scores"].float().cpu().numpy() if "scores" in res else np.zeros(0)
    boxes = res["boxes"].float().cpu().numpy() if "boxes" in res else np.zeros((0,4))
    # rescale masks to original size
    if len(masks_ds) and (Wds != W0 or Hds != H0):
        masks = np.stack([cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks_ds])
    else:
        masks = masks_ds
    # rescale boxes to original
    if len(boxes) and (Wds != W0 or Hds != H0):
        sx, sy = W0/Wds, H0/Hds
        boxes = boxes * np.array([sx, sy, sx, sy])
    return masks, sc, boxes


def sam3_box_predict(proc, mdl, img_np, boxes_abs, mask_thr):
    """SAM3 box-prompt on given boxes -> masks [N,H,W] bool, scores [N].
    API verified 2026-07-18 (prep_11b): forward(input_boxes=[boxes.tolist()]) +
    cast floats to model dtype (bfloat16) + post_process_instance_segmentation
    returns {scores, boxes, masks} at target_sizes. No .predict method exists."""
    import torch
    from PIL import Image
    H0, W0 = img_np.shape[:2]
    long_edge = max(W0, H0)
    if long_edge > LONG_EDGE_CAP:
        scale = LONG_EDGE_CAP / long_edge
        Wds, Hds = int(W0*scale), int(H0*scale)
        img_ds = cv2.resize(img_np, (Wds, Hds))
        boxes = boxes_abs * scale
    else:
        img_ds = img_np; Wds, Hds = W0, H0
        boxes = boxes_abs
    im = Image.fromarray(img_ds[:,:,::-1]).convert("RGB")
    inputs = proc(images=im, input_boxes=[boxes.tolist()], return_tensors="pt").to("cuda")
    mdtype = next(mdl.parameters()).dtype
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(mdtype)
    with torch.inference_mode():
        out = mdl(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.25, mask_threshold=mask_thr, target_sizes=[(Hds, Wds)])[0]
    masks_ds = res["masks"].cpu().numpy().astype(bool)
    sc = res["scores"].float().cpu().numpy() if "scores" in res else np.ones(len(masks_ds))
    if len(masks_ds) and (Wds != W0 or Hds != H0):
        masks = np.stack([cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks_ds])
    else:
        masks = masks_ds
    return masks, sc


def _ensure_field(ds, name):
    if name not in ds.get_field_schema():
        ds.add_sample_field(name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)


def row_sam3(ds):
    proc, mdl = get_sam3()
    t0 = time.time()
    _ensure_field(ds, "sam3_pred")
    for i, s in enumerate(ds.iter_samples(progress=True)):
        img = cv2.imread(s.filepath)
        if img is None:
            continue
        H, W = img.shape[:2]
        masks, sc, boxes = sam3_text_predict(proc, mdl, img, SCORE, MASK_THR)
        if len(masks):
            boxes_abs = boxes  # already abs original
            keep = sieve(masks, sc, boxes_abs)
            dets = [mask_to_detection(masks[k], sc[k], W, H) for k in keep]
            dets = [d for d in dets if d is not None]
        else:
            dets = []
        s["sam3_pred"] = fo.Detections(detections=dets)
        s.save()
    print(f"SAM3 row done in {time.time()-t0:.1f}s")


def row_yolo11x(ds):
    from ultralytics import YOLO
    model = YOLO(YOLO_CKPT)
    t0 = time.time()
    _ensure_field(ds, "yolo11x_pred")
    for s in ds.iter_samples(progress=True):
        res = model(s.filepath, conf=0.001, iou=0.7, imgsz=640, max_det=10000, retina_masks=True, verbose=False)
        r = res[0]
        H, W = r.orig_shape
        dets = []
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy().astype(bool)  # [N, h, w] at imgsz? -> need orig
            # ultralytics masks at orig res when retina_masks=True? masks are at imgsz; rescale
            mh, mw = masks.shape[1], masks.shape[2]
            for i, m in enumerate(masks):
                if mh != H or mw != W:
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                conf = float(r.boxes.conf[i].cpu().numpy()) if r.boxes.conf is not None else 1.0
                d = mask_to_detection(m, conf, W, H)
                if d is not None:
                    dets.append(d)
        s["yolo11x_pred"] = fo.Detections(detections=dets)
        s.save()
    print(f"YOLOv11x row done in {time.time()-t0:.1f}s")


def row_rtdetr_sam3(ds):
    from ultralytics import RTDETR
    import torch
    rtdetr = RTDETR(RTDETR_CKPT)
    proc, mdl = get_sam3()
    t0 = time.time()
    _ensure_field(ds, "rtdetr_sam3_pred")
    for s in ds.iter_samples(progress=True):
        img = cv2.imread(s.filepath)
        if img is None:
            continue
        H, W = img.shape[:2]
        # RT-DETR boxes
        res = rtdetr(s.filepath, conf=0.25, iou=0.7, imgsz=640, verbose=False)
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            s["rtdetr_sam3_pred"] = fo.Detections(detections=[])
            s.save(); continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)  # [N,4] abs
        # SAM3 box-prompt
        masks, sc = sam3_box_predict(proc, mdl, img, boxes, MASK_THR)
        dets = []
        for i in range(len(masks)):
            d = mask_to_detection(masks[i], sc[i] if i < len(sc) else 1.0, W, H)
            if d is not None:
                dets.append(d)
        s["rtdetr_sam3_pred"] = fo.Detections(detections=dets)
        s.save()
    print(f"RT-DETR+SAM3 row done in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=["sam3", "yolo11x", "rtdetr_sam3"])
    args = ap.parse_args()
    ds = fo.load_dataset(DS)
    if args.row == "sam3":
        row_sam3(ds)
    elif args.row == "yolo11x":
        row_yolo11x(ds)
    elif args.row == "rtdetr_sam3":
        row_rtdetr_sam3(ds)

if __name__ == "__main__":
    main()
