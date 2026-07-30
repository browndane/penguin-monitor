"""Empirically verify SAM3 box-prompt API on ONE gold-200 image before the full run.
Casts inputs to model dtype (bfloat16) to fix the dtype mismatch. Prints text only."""
import os
os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")
import numpy as np, cv2
import fiftyone as fo

ds = fo.load_dataset("p5_gold200")
s = ds.first()
img = cv2.imread(s.filepath)
H0, W0 = img.shape[:2]
print(f"img: {s.filepath.split('/')[-1]}  shape={img.shape}", flush=True)

from ultralytics import RTDETR
RTDETR_CKPT = "/home/vorse/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"
rtdetr = RTDETR(RTDETR_CKPT)
r = rtdetr(s.filepath, conf=0.25, iou=0.7, imgsz=640, verbose=False)[0]
if r.boxes is None or len(r.boxes) == 0:
    g = s.ground_truth.detections[0]
    bb = g.bounding_box
    boxes = np.array([[bb[0]*W0, bb[1]*H0, (bb[0]+bb[2])*W0, (bb[1]+bb[3])*H0]])
else:
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
print(f"RT-DETR boxes: {len(boxes)}  first: {boxes[0].tolist()}", flush=True)

import torch
from transformers import Sam3Model, Sam3Processor
from PIL import Image
proc = Sam3Processor.from_pretrained("facebook/sam3")
mdl = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda").eval()
mdtype = next(mdl.parameters()).dtype
print(f"SAM3 loaded, model dtype={mdtype}", flush=True)

LONG_EDGE_CAP = 2448
long_edge = max(W0, H0)
if long_edge > LONG_EDGE_CAP:
    scale = LONG_EDGE_CAP / long_edge
    Wds, Hds = int(W0*scale), int(H0*scale)
    img_ds = cv2.resize(img, (Wds, Hds)); bxs = boxes * scale
else:
    img_ds = img; Wds, Hds = W0, H0; bxs = boxes
im = Image.fromarray(img_ds[:,:,::-1]).convert("RGB")

def to_model_dtype(batch):
    for k, v in batch.items():
        if torch.is_floating_point(v):
            batch[k] = v.to(mdtype)
    return batch

# box-prompt via forward(input_boxes=...)
print("=== attempt: proc(input_boxes=[bxs]) + cast + mdl(**inputs) ===", flush=True)
try:
    inputs = proc(images=im, input_boxes=[bxs.tolist()], return_tensors="pt")
    inputs = inputs.to("cuda")
    inputs = to_model_dtype(inputs)
    with torch.inference_mode():
        out = mdl(**inputs)
    print(f"out type: {type(out).__name__}", flush=True)
    print(f"out fields: {[a for a in dir(out) if not a.startswith('_') and not callable(getattr(out,a,None))][:30]}", flush=True)
    # Sam3ImageSegmentationOutput - find masks
    for attr in ["pred_masks", "masks", "segmentation_logits", "logits"]:
        if hasattr(out, attr):
            v = getattr(out, attr)
            print(f"  out.{attr}: type={type(v).__name__} shape={tuple(v.shape) if hasattr(v,'shape') else 'n/a'}", flush=True)
    # try post_process with target_sizes
    try:
        res = proc.post_process_instance_segmentation(out, threshold=0.25, mask_threshold=0.5, target_sizes=[(Hds, Wds)])[0]
        print(f"  post_process keys: {list(res.keys())}", flush=True)
        if "masks" in res:
            mk = res["masks"]
            print(f"  masks: shape={tuple(mk.shape)} sum={int(mk.sum())}", flush=True)
        if "scores" in res:
            print(f"  scores: {res['scores']}", flush=True)
        if "boxes" in res:
            print(f"  boxes: {res['boxes']}", flush=True)
    except Exception as e:
        print(f"  post_process FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("BOX-PROMPT API OK", flush=True)
except Exception as e:
    print(f"BOX-PROMPT FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    import traceback; traceback.print_exc()

print("=== DONE ===", flush=True)
