"""Phase 2 prep_13 (file-based): supervised re-eval of 6 real trained ckpts AS-IS
on the 2 v2 gold corpora, same metric as Phase 1 (pycocotools mask mAP@0.5,
iscrowd-ignore per-tier, agg=mean of 3 tiers).

Reuse from Phase 1 (UNCHANGED):
  - build_gt_coco / build_pred_coco / eval_tier / encode_mask from prep_10
  - Tier sidecars {boulders,sanparks}_tiers.json (pre-computed; do NOT recompute)

6 ckpts, all trained on african-penguin-inaturalist-sam-veryclean (NEVER saw gold):
  1. rf-detr-seg-xlarge  (native mask)  checkpoint_best_total.pth  paper 0.932
  2. rf-detr-seg-large   (native mask)  checkpoint_best_total.pth  paper 0.904
  3. rtdetr-4            (boxes->SAM3)  weights/last.pth            paper 0.868
  4. rtdetr-3            (boxes->SAM3)  weights/last.pth            paper 0.826
  5. eomt-large          (native mask)  last.ckpt                   paper 0.834
  6. yolo11x-seg         (native mask)  weights/best.pt            paper 0.581

Prediction method:
  - Native mask heads: run inference at the model's native path, extract masks,
    rescale to original res via INTER_NEAREST, encode COCO RLE IMMEDIATELY.
    yolo11x: Ultralytics model(img, conf=0.001, iou=0.7, imgsz=640,
            retina_masks=True, max_det=10000); masks.data rescaled INTER_NEAREST.
    rf-detr-seg: rfdetr ModelCls(pretrain_weights=ckpt).predict(PIL img at
            (sz,sz), threshold=0.01); det.mask rescaled INTER_NEAREST to orig.
    eomt-large: manual MCI build from ckpt hparams + state_dict; forward via
            resize_and_pad_imgs_instance_panoptic + self(imgs); last layer;
            interpolate to 640; revert; topk(100) at 640; masks>0; INTER_NEAREST
            to original; encode RLE per-mask.
  - RT-DETR (rtdetr-4, rtdetr-3): Ultralytics RTDETR(ckpt).predict(conf=0.25,
            iou=0.7, imgsz=640) -> xyxy boxes; then SAM3 box-prompt (proc with
            input_boxes=[boxes.tolist()], cast floats to bf16, mdl(**inputs),
            post_process_instance_segmentation(threshold=0.25, mask_threshold=0.5,
            target_sizes=[(Hds,Wds)])). LONG_EDGE_CAP=2448 for >4096px. SAM3
            loaded ONCE, reused across both RT-DETR ckpts + both corpora.

Outputs per corpus:
  ~/bench_any/runs/p5_r1_v2/<corpus>/<ckpt>_pred.json   (COCO-RLE preds by filename)
  ~/bench_any/runs/p5_r1_v2/<corpus>/<ckpt>_reeval.json (per-tier + agg)
Manifest:
  ~/bench_any/runs/p5_r1_v2/manifest_phase2.json (ckpt paths, params, versions)

Metric invariants (SAME as prep_10, UNCHANGED eval_tier):
  - pycocotools mask COCOeval, IoU 0.5, maxDets=[1,10,100].
  - Per-tier via iscrowd-ignore (GT in tier iscrowd=0 active, else 1 ignore).
  - Read precision[0,:,:,0,-1] to dodge _summarize-hardcodes-100 trap.
  - copy.deepcopy(pred_list) before every loadRes (loadRes mutates preds).
  - Agg = arithmetic mean of 3 tier mAPs.
"""
import os, sys, argparse, time, json, copy, tempfile, hashlib
from pathlib import Path

import numpy as np
import cv2
import pycocotools.mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

TIERS = ["easy", "medium", "hard"]

# Paper in-distribution seg mAP50 (iNat val; OOD gold divergence expected)
PAPER = {
    "rf-detr-seg-xlarge": 0.932,
    "rf-detr-seg-large": 0.904,
    "rtdetr-4": 0.868,
    "rtdetr-3": 0.826,
    "eomt-large": 0.834,
    "yolo11x-seg": 0.581,
}

LONG_EDGE_CAP = 2448  # SAM3 OOM cap for >4096px images

# --- Checkpoint registry (absolute paths) ---
CKPT = {
    "rf-detr-seg-xlarge": os.path.expanduser(
        "~/bench_any/runs/segment/african-penguin_rf-detr-seg-xlarge_b1_e125_i640_be32_fair_grid/checkpoint_best_total.pth"),
    "rf-detr-seg-large": os.path.expanduser(
        "~/bench_any/runs/segment/african-penguin_rf-detr-seg-large_b2_e125_i640_be32_fair_grid/checkpoint_best_total.pth"),
    "rtdetr-4": os.path.expanduser(
        "~/bench_any/runs/detect/african-penguin_rtdetr-4_b8_e150_i640_be64_fair_grid/weights/last.pt"),
    "rtdetr-3": os.path.expanduser(
        "~/bench_any/runs/detect/african-penguin_rtdetr-3_b8_e150_i640_be64_fair_grid/weights/last.pt"),
    "eomt-large": os.path.expanduser(
        "~/bench_any/runs/segment/african-penguin_eomt-large_b2_e125_i640_be32_fair_grid/african-penguin_eomt-large_b2_e125_i640_be32_fair_grid/version_0/checkpoints/last.ckpt"),
    "yolo11x-seg": os.path.expanduser(
        "~/bench_any/runs/segment/african-penguin_yolo11x-seg_b0.98_e150_i640_be64_fair_grid/weights/best.pt"),
    # --- COMBINED-training ckpts (saw Boulders/SANParks envs during training; gold held out) ---
    "yolo11x-seg-combined": os.path.expanduser(
        "~/bench_any/runs/segment/penguin-seg-combined_yolo11x-seg/weights/best.pt"),
    "rf-detr-seg-large-combined": os.path.expanduser(
        "~/bench_any/runs/segment/penguin-seg-combined_rf-detr-seg-large/checkpoint_best_total.pth"),
    "eomt-large-combined": os.path.expanduser(
        "~/bench_any/runs/segment/penguin-seg-combined_eomt-large/penguin-seg-combined_eomt-large/version_0/checkpoints/epoch=71-step=6264.ckpt"),
    "rtdetr-x-combined": os.path.expanduser(
        "~/bench_any/runs/detect/penguin-seg-combined_rtdetr-x/weights/best.pt"),
}

CORPORA = {
    "boulders": {
        "images": "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/boulders-penguin/train/images",
        "tiers": os.path.expanduser("~/bench_any/runs/p5_r1_v2/boulders/boulders_tiers.json"),
    },
    "sanparks": {
        "images": "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/sanparks-algoa/train/images",
        "tiers": os.path.expanduser("~/bench_any/runs/p5_r1_v2/sanparks/sanparks_tiers.json"),
    },
}


# ============================================================================
# REUSED from prep_10 (UNCHANGED eval_tier / build_gt_coco / build_pred_coco)
# ============================================================================
def encode_mask(mask_bool):
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mu.encode(m)
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def build_gt_coco(tiers_manifest, tier):
    """Build COCO GT json for a tier: GT in tier -> iscrowd=0 active; else iscrowd=1 ignore."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    for fname, dets in tiers_manifest.items():
        img_id += 1
        if not dets:
            images.append({"id": img_id, "width": 0, "height": 0, "file_name": fname})
            continue
        W = dets[0]["width"]; H = dets[0]["height"]
        images.append({"id": img_id, "width": W, "height": H, "file_name": fname})
        for e in dets:
            t = e["tier"]
            rle = e["rle"]
            x0, y0, x1, y1 = e["bbox"]
            bw = x1 - x0; bh = y1 - y0
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(e["area"]),
                "iscrowd": 0 if t == tier else 1,
                "_tier": t,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "penguin"}]}


def build_pred_coco(preds_per_img, tiers_manifest):
    """Build COCO pred json from {filename: [{rle, score, box}, ...]}."""
    fname_to_id = {}
    img_id = 0
    for fname in tiers_manifest.keys():
        img_id += 1
        fname_to_id[fname] = img_id
    preds = []
    for fname, plist in preds_per_img.items():
        iid = fname_to_id.get(fname)
        if iid is None:
            continue
        for p in plist:
            preds.append({
                "image_id": iid,
                "category_id": 1,
                "segmentation": p["rle"],
                "score": float(p["score"]),
            })
    return preds


def eval_tier(gt_dict, pred_list):
    """Run COCOeval at iou 0.5, maxDet 100. Return mAP@0.5 (percent) or None.
    UNCHANGED from prep_10: reads precision[0,:,:,0,-1], deepcopy before loadRes."""
    if not pred_list:
        return None
    pred_list = copy.deepcopy(pred_list)
    gtf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(gt_dict, gtf); gtf.close()
    try:
        coco_gt = COCO(gtf.name)
        coco_dt = coco_gt.loadRes(pred_list)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [1, 10, 100]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
        pr = ev.eval["precision"][0, :, 0, 0, -1]
        valid = pr[pr > -1]
        ap = float(np.mean(valid)) if len(valid) else 0.0
        return ap * 100.0
    except Exception as e:
        print(f"  COCOeval ERR: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return None
    finally:
        os.unlink(gtf.name)


def _rle_to_bytes(rle):
    """Convert an RLE dict with str counts to one with bytes counts (pycocotools
    internal format for mu.iou). Cheap: just encode the str to bytes."""
    rc = dict(rle)
    if isinstance(rc.get("counts"), str):
        rc["counts"] = rc["counts"].encode("ascii")
    return rc


def quick_recall(tiers_manifest, preds_per_img):
    """Greedy IoU-0.5 recall over all images. Uses pycocotools RLE-based IoU
    (mu.iou) which operates on RLEs directly (no full-res decode) — same result
    as prep_10's decode-based version but orders of magnitude faster on 5K images."""
    n_gt = 0; n_tp = 0
    for fname, dets in tiers_manifest.items():
        glds = [e["rle"] for e in dets]
        n_gt += len(glds)
        preds = preds_per_img.get(fname, [])
        if not preds or not glds:
            continue
        gt_rles = [_rle_to_bytes(r) for r in glds]
        pr_rles = [_rle_to_bytes(p["rle"]) for p in preds]
        pr_scores = np.array([p["score"] for p in preds])
        # IoU matrix [n_preds, n_gt] via RLE ops
        if len(pr_rles) and len(gt_rles):
            iou_mat = mu.iou(pr_rles, gt_rles, [0] * len(gt_rles))
        else:
            iou_mat = np.zeros((len(pr_rles), len(gt_rles)))
        order = np.argsort(-pr_scores)
        used = [False] * len(glds)
        for idx in order:
            best_g = -1; best_iou = 0.0
            for gi in range(len(glds)):
                if used[gi]:
                    continue
                v = float(iou_mat[idx, gi])
                if v > best_iou:
                    best_iou = v; best_g = gi
            if best_g >= 0 and best_iou >= 0.5:
                used[best_g] = True; n_tp += 1
    return (n_tp / n_gt) if n_gt > 0 else 0.0


# ============================================================================
# Per-ckpt prediction builders. Each returns {filename: [{rle, score, box}, ...]}
# ============================================================================

def _img_files(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts])


def predict_yolo11x(ckpt_path, img_dir):
    """YOLO11x-seg native mask. Ultralytics predict at imgsz=640, retina_masks=False
    (masks at imgsz to avoid OOM on 5K images), max_det=10000, conf=0.001, iou=0.7.
    Rescale masks INTER_NEAREST to original. This matches RF-DETR/EoMT which predict
    at internal res then upscale."""
    from ultralytics import YOLO
    model = YOLO(ckpt_path)
    preds = {}
    files = _img_files(img_dir)
    for i, ip in enumerate(files):
        r = model.predict(str(ip), conf=0.001, iou=0.7, imgsz=640,
                          retina_masks=False, max_det=10000, verbose=False)
        per = []
        if r and r[0].masks is not None and r[0].masks.data is not None:
            H0, W0 = r[0].orig_shape
            masks_t = r[0].masks.data  # (N, h_pred, w_pred) at imgsz (640)
            confs = r[0].boxes.conf if r[0].boxes is not None else None
            xyxy = r[0].boxes.xyxy if r[0].boxes is not None else None
            mdet = masks_t.shape[0]
            # rescale each mask to orig via INTER_NEAREST, encode RLE immediately
            for k in range(mdet):
                m_small = masks_t[k].detach().cpu().numpy().astype(np.uint8)
                m_full = cv2.resize(m_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
                rle = encode_mask(m_full.astype(bool))
                sc = float(confs[k].item()) if confs is not None else 1.0
                bx = xyxy[k].detach().cpu().numpy().tolist() if xyxy is not None else [0,0,0,0]
                per.append({"rle": rle, "score": sc, "box": [float(x) for x in bx]})
        preds[ip.name] = per
        if i % 20 == 0:
            print(f"  yolo11x {i}/{len(files)} {ip.name} -> {len(per)} preds", flush=True)
        del r
    del model
    return preds


def predict_rfdetr_seg(model_tag, ckpt_path, img_dir):
    """RF-DETR-seg native mask. Build model with pretrain_weights=ckpt, predict at
    (sz,sz), rescale masks INTER_NEAREST to original. sz = (640//24)*24 = 624."""
    import rfdetr as _rf
    from PIL import Image
    # variant resolution
    variant = "xlarge" if "xlarge" in model_tag else "large"
    if variant == "xlarge":
        ModelCls = _rf.RFDETRSegXLarge
    else:
        ModelCls = _rf.RFDETRSegLarge
    model = ModelCls(pretrain_weights=str(ckpt_path))
    sz = max(24, (640 // 24) * 24)  # 624
    files = _img_files(img_dir)
    preds = {}
    # discover threshold kwarg
    resolved_kw = None
    for kw in [{"threshold": 0.01}, {"conf": 0.01}, {}]:
        try:
            t = model.predict(Image.open(str(files[0])).convert("RGB").resize((sz, sz), Image.BILINEAR), **kw)
            resolved_kw = kw; break
        except (TypeError, Exception):
            continue
    if resolved_kw is None:
        resolved_kw = {}
    for i, ip in enumerate(files):
        orig = Image.open(str(ip)).convert("RGB")
        W0, H0 = orig.size
        img_resized = orig.resize((sz, sz), Image.BILINEAR)
        det = model.predict(img_resized, **resolved_kw)
        xyxy = getattr(det, "xyxy", None)
        conf = getattr(det, "confidence", None)
        masks = getattr(det, "mask", None)
        per = []
        if xyxy is not None and len(xyxy) > 0:
            sx = float(W0) / float(sz)
            sy = float(H0) / float(sz)
            for j in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy
                sc = float(conf[j]) if conf is not None else 1.0
                if masks is not None and j < len(masks):
                    m_small = (masks[j].astype(np.uint8) > 0).astype(np.uint8)
                    m_full = cv2.resize(m_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    rle = encode_mask(m_full.astype(bool))
                    per.append({"rle": rle, "score": sc, "box": [x1, y1, x2, y2]})
        preds[ip.name] = per
        if i % 20 == 0:
            print(f"  rfdetr-{variant} {i}/{len(files)} {ip.name} -> {len(per)} preds", flush=True)
        del det
    del model
    return preds


def predict_eomt(ckpt_path, img_dir):
    """EoMT-large native mask. Manual build from ckpt hparams + state_dict.
    Forward: resize_and_pad_imgs -> model -> last layer -> interpolate 640 ->
    revert -> topk(100) at 640 res -> masks>0 -> INTER_NEAREST to original -> RLE.
    Build once, predict all images."""
    sys.path.insert(0, "/home/vorse/bench_any/eomt")
    import torch
    import torch.nn.functional as F
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_instance import MaskClassificationInstance

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    img_size = tuple(hp.get("img_size", [640, 640]))
    num_classes = int(hp.get("num_classes", 1))

    # Build network by walking the class_path dict (advisor guidance).
    # EoMT needs num_classes and ViT needs img_size, but the saved network spec
    # omits both (they're injected by the Lightning CLI at build time). Inject
    # them here so the manual build matches the trained architecture.
    def build_component(spec, depth=0):
        cls_path = spec["class_path"]
        init_args = dict(spec.get("init_args", {}))
        resolved = {}
        for k, v in init_args.items():
            if isinstance(v, dict) and "class_path" in v:
                resolved[k] = build_component(v, depth + 1)
            else:
                resolved[k] = v
        # inject missing required args based on class
        cls_name = cls_path.rsplit(".", 1)[-1]
        if cls_name == "ViT" and "img_size" not in resolved:
            resolved["img_size"] = img_size
        if cls_name == "EoMT" and "num_classes" not in resolved:
            resolved["num_classes"] = num_classes
        mod_name, cls_name = cls_path.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        return cls(**resolved)

    network = build_component(hp["network"])
    # Build MCI with the same hyperparams (minus network which we built)
    mci_kwargs = {k: v for k, v in hp.items() if k != "network" and k != "_instantiator"}
    # ckpt_path/delta_weights/load_ckpt_class_head are build-time flags, not __init__ args we want
    for bad in ("ckpt_path", "delta_weights", "load_ckpt_class_head"):
        mci_kwargs.pop(bad, None)
    model = MaskClassificationInstance(network=network, **mci_kwargs)
    # Load trained state (strip 'network.' prefix? MCI stores network under self.network)
    sd = ck["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval()
    topk = int(hp.get("eval_top_k_instances", 100))

    files = _img_files(img_dir)
    preds = {}
    t0 = time.time()
    with torch.inference_mode():
        for i, ip in enumerate(files):
            # Load image as (C,H,W) tensor uint8, as EoMT dataset returns
            img_np = cv2.imread(str(ip))  # BGR
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            H0, W0 = img_np.shape[:2]
            # resize_and_pad_imgs_instance_panotropic calls PIL.Image.fromarray which
            # requires uint8, NOT float. Pass uint8 for the resize/pad, then convert
            # the padded output to float for the model forward pass.
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()  # uint8
            img_sizes = [img_t[0].shape[-2:]]
            transformed = model.resize_and_pad_imgs_instance_panoptic(img_t)
            # LightningModule.forward divides by 255 internally; pass float [0,255]
            # as eval_step does (dataset returns uint8 [0,255]).
            transformed = transformed.float()
            mask_logits_per_layer, class_logits_per_layer = model(transformed)
            mask_logits = mask_logits_per_layer[-1]
            class_logits = class_logits_per_layer[-1]
            mask_logits = F.interpolate(mask_logits, img_size, mode="bilinear")
            # revert_resize_and_pad interpolates logits to original resolution, which
            # OOMs on 18MP Boulders images (5184x3456) with 100+ queries.  Do the
            # revert on a capped resolution, then upscale masks on CPU via INTER_NEAREST.
            # PRESERVE ASPECT RATIO when capping: the revert function calls
            # scale_img_size_instance_panoptic(img_sizes[i]) to determine the crop
            # region (removing padding). If we pass a square (1280,1280) for a
            # non-square original, the crop includes padding -> masks are shifted.
            CAP = 1280  # cap longest side; mask upscaling is INTER_NEAREST anyway
            capped_sizes = []
            for s in img_sizes:
                h, w = s[0], s[1]
                long_side = max(h, w)
                if long_side > CAP:
                    f = CAP / long_side
                    capped_sizes.append((round(h * f), round(w * f)))
                else:
                    capped_sizes.append((h, w))
            mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, capped_sizes)
            per = []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (torch.arange(scores.shape[-1], device=model.device)
                          .unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1))
                topk_scores, topk_indices = scores.flatten(0, 1).topk(topk, sorted=False)
                labels = labels[topk_indices]
                topk_indices = topk_indices // scores.shape[-1]
                ml = mask_logits[j][topk_indices]
                masks = ml > 0
                mask_scores = (ml.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                final_scores = topk_scores * mask_scores
                keep = (final_scores > 1e-3).nonzero(as_tuple=True)[0].tolist()
                for k in keep:
                    m = masks[k].detach().cpu().numpy().astype(np.uint8)
                    if m.sum() == 0:
                        continue
                    # rescale to capped then to original via INTER_NEAREST
                    m_full = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    rle = encode_mask(m_full.astype(bool))
                    ys, xs = np.where(m_full > 0)
                    if len(xs) == 0:
                        continue
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    per.append({"rle": rle, "score": float(final_scores[k].item()),
                                "box": [float(x1), float(y1), float(x2), float(y2)]})
            preds[ip.name] = per
            del img_t, transformed, mask_logits, class_logits, masks, ml
            if i % 10 == 0:
                print(f"  eomt {i}/{len(files)} {ip.name} -> {len(per)} preds ({time.time()-t0:.1f}s)", flush=True)
    del model, network
    torch.cuda.empty_cache()
    return preds


def get_sam3_box():
    """Load SAM3 model + processor for box-prompt. Use explicit cache_dir so
    DINOv3 (default HF cache) and SAM3 (/mnt/LinuxBackup/sam3) don't collide."""
    import torch
    from transformers import Sam3Model, Sam3Processor
    cache = "/mnt/LinuxBackup/sam3/hub"
    proc = Sam3Processor.from_pretrained("facebook/sam3", cache_dir=cache)
    mdl = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16, cache_dir=cache).to("cuda").eval()
    return proc, mdl


def sam3_box_predict(proc, mdl, img_np, boxes_xyxy, score=0.25, mask_thr=0.5):
    """SAM3 box-prompt: given an image + N xyxy boxes, return masks at original res.
    LONG_EDGE_CAP=2448 for >4096px. Cast floats to bf16 (memory note)."""
    import torch
    from PIL import Image
    H0, W0 = img_np.shape[:2]
    long_edge = max(W0, H0)
    if long_edge > LONG_EDGE_CAP:
        scale = LONG_EDGE_CAP / long_edge
        Wds, Hds = int(W0 * scale), int(H0 * scale)
        img_ds = cv2.resize(img_np, (Wds, Hds))
        # scale boxes to downsampled coords
        boxes_ds = np.array(boxes_xyxy, dtype=np.float64) * scale if len(boxes_xyxy) else np.zeros((0, 4))
    else:
        img_ds = img_np
        Wds, Hds = W0, H0
        boxes_ds = np.array(boxes_xyxy, dtype=np.float64) if len(boxes_xyxy) else np.zeros((0, 4))
    im = Image.fromarray(img_ds[:, :, ::-1]).convert("RGB")
    if len(boxes_ds) == 0:
        return np.zeros((0, H0, W0), bool), np.zeros(0)
    inputs = proc(images=im, input_boxes=[boxes_ds.tolist()], return_tensors="pt").to("cuda")
    mdtype = next(mdl.parameters()).dtype
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(mdtype)
    with torch.inference_mode():
        out = mdl(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=score, mask_threshold=mask_thr, target_sizes=[(Hds, Wds)])[0]
    masks_ds = res["masks"].cpu().numpy().astype(bool) if "masks" in res else np.zeros((0, Hds, Wds), bool)
    sc = res["scores"].float().cpu().numpy() if "scores" in res else np.zeros(0)
    # rescale masks to original
    if len(masks_ds) and (Wds != W0 or Hds != H0):
        masks = np.stack([cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
                          for m in masks_ds])
    else:
        masks = masks_ds
    return masks, sc


RTDETRV4_REPO = Path.home() / "bench_any" / "runtimes" / "RT-DETRv4"
RTDETRV4_CFG_BASE = "configs/rtv4/rtv4_hgnetv2_s_coco.yml"
RTDETRV3_CFG_BASE = "configs/rtv2/rtv2_r18vd_120e_coco.yml"


def _rtdetrv4_fix_eval_tensors(model, device):
    """Move non-parameter tensors (pos_embed, anchors, valid_mask) to device.
    Without this, model(samples) crashes because these buffers live on CPU."""
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


def _build_rtdetr_external(ckpt_path, num_classes=1, imgsz=640):
    """Build an RT-DETRv4 or RT-DETRv3 model from an external-runtime checkpoint.

    These checkpoints were trained by the external RT-DETR runtime (saved as
    {"model": state_dict, "source": ...}), not by Ultralytics, so Ultralytics'
    RTDETR(ckpt) cannot load them. We build the model from the runtime's YAML
    config (minimal, no dataloaders/teacher), load the state_dict, and return
    (model, postprocessor) ready for per-image inference.
    """
    import torch
    import yaml
    import sys as _sys

    stem = str(ckpt_path).lower()
    is_v3 = "rtdetr-3" in stem or "rtdetrv3" in stem or "rtdetr-v3" in stem
    if is_v3:
        cfg_base = RTDETRV4_REPO / RTDETRV3_CFG_BASE
    else:
        cfg_base = RTDETRV4_REPO / RTDETRV4_CFG_BASE

    if str(RTDETRV4_REPO) not in _sys.path:
        _sys.path.insert(0, str(RTDETRV4_REPO))
    from engine.core import YAMLConfig

    cfg_dict = {
        "__include__": [str(cfg_base)],
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "eval_spatial_size": [imgsz, imgsz],
    }
    if is_v3:
        cfg_dict["PResNet"] = {"depth": 18, "pretrained": False, "freeze_at": -1, "freeze_norm": False}
    else:
        cfg_dict["HGNetv2"] = {"pretrained": False}

    cfg_path = RTDETRV4_REPO / f"_prep13_{'v3' if is_v3 else 'v4'}_inline.yml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg_dict, fh, sort_keys=False)

    cfg = YAMLConfig(str(cfg_path))
    model = cfg.model
    postprocessor = cfg.postprocessor

    state = torch.load(str(ckpt_path), map_location="cpu")
    src = state["model"] if isinstance(state, dict) and "model" in state else state
    cur = model.state_dict()
    matched = {k: v for k, v in src.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    model.load_state_dict(matched, strict=False)
    tag = "RT-DETRv3-R18" if is_v3 else "RT-DETRv4-S"
    print(f"  [{tag}] loaded {len(matched)}/{len(cur)} tensors from {Path(ckpt_path).name}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    postprocessor = postprocessor.to(device).eval()
    _rtdetrv4_fix_eval_tensors(model, device)
    return model, postprocessor, device


def predict_rtdetr_sam3(ckpt_path, img_dir, proc, mdl):
    """RT-DETR box inference (external runtime) -> xyxy boxes, then SAM3 box-prompt.
    conf=0.25, iou=0.7 (NMS), imgsz=640. Boxes -> SAM3 -> masks at original res.

    Loads RT-DETRv4/v3 checkpoints trained by the external runtime, not Ultralytics.
    Preprocessing matches the runtime's val pipeline: Resize(640) + ConvertPILImage
    (float32, scale=True = divide by 255). The postprocessor returns top-300 queries
    with sigmoid scores and xyxy boxes scaled to original image size; we filter by
    conf=0.25 and apply NMS(iou=0.7) before passing boxes to SAM3."""
    import torch
    import torchvision

    model, postprocessor, device = _build_rtdetr_external(ckpt_path)
    imgsz = 640
    files = _img_files(img_dir)
    preds = {}
    t0 = time.time()
    for i, ip in enumerate(files):
        img_np = cv2.imread(str(ip))  # BGR
        if img_np is None:
            preds[ip.name] = []
            continue
        H0, W0 = img_np.shape[:2]
        # Preprocess: BGR -> RGB, resize to 640, /255, CHW
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (imgsz, imgsz))
        tensor = torch.from_numpy(img_resized.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(tensor)
            # RT-DETRv4 postprocessor convention is [W, H]: it multiplies
            # normalised xyxy boxes by orig_target_sizes.repeat(1,2) = [W,H,W,H],
            # applied to [x0,y0,x1,y1] (postprocessor.py:55; dataset.py:80 uses
            # torch.tensor([w,h])). Passing [H,W] swaps axes -> boxes land in
            # wrong regions -> SAM3 segments nothing matching gold (1.1 agg).
            orig_sizes = torch.tensor([[W0, H0]], device=device)
            results = postprocessor(outputs, orig_sizes)
        r0 = results[0]
        scores = r0["scores"].detach().cpu()
        boxes = r0["boxes"].detach().cpu()
        # Filter by confidence
        keep = scores >= 0.25
        scores_f = scores[keep]
        boxes_f = boxes[keep]
        # NMS at iou=0.7 (RT-DETR is one-to-one, NMS rarely removes boxes, but
        # we match the original Ultralytics iou=0.7 contract for parity)
        if len(boxes_f) > 0:
            keep_idx = torchvision.ops.nms(boxes_f, scores_f, 0.7)
            boxes_f = boxes_f[keep_idx]
            scores_f = scores_f[keep_idx]
        boxes = [[float(v) for v in b] for b in boxes_f]
        per = []
        if len(boxes):
            masks, sc = sam3_box_predict(proc, mdl, img_np, boxes)
            for k in range(len(masks)):
                rle = encode_mask(masks[k])
                per.append({"rle": rle, "score": float(sc[k]) if k < len(sc) else 1.0,
                            "box": boxes[k] if k < len(boxes) else [0,0,0,0]})
        preds[ip.name] = per
        del img_np, tensor
        if i % 10 == 0:
            print(f"  rtdetr+sam3 {i}/{len(files)} {ip.name} -> {len(per)} preds ({time.time()-t0:.1f}s)", flush=True)
    del model, postprocessor
    import torch as _t; _t.cuda.empty_cache()
    return preds


def predict_rtdetr_ultralytics_sam3(ckpt_path, img_dir, proc, mdl):
    """RT-DETR (Ultralytics RTDETR ckpt) -> boxes -> SAM3 box-prompt masks.

    Mirrors predict_rtdetr_sam3 but loads via `ultralytics.RTDETR(ckpt)` instead
    of the external-runtime builder. Ultralytics handles its own resize/postproc,
    so boxes come back in original image coords (no coord-fix needed). Boxes are
    then fed to the SAME sam3_box_predict used by the external path. Returns the
    SAME shape as predict_rtdetr_sam3: {filename: [{rle, score, box}, ...]}.
    """
    import torch
    from ultralytics import RTDETR
    files = _img_files(img_dir)
    print(f"[rtdetr-ultralytics+sam3] {len(files)} images from {img_dir}", flush=True)
    model = RTDETR(ckpt_path)
    preds = {}
    for i, ip in enumerate(files):
        t0 = time.time()
        img_np = cv2.imread(str(ip))  # BGR, same as external path
        if img_np is None:
            preds[ip.name] = []
            continue
        r = model.predict(str(ip), conf=0.25, imgsz=640, verbose=False, retina_masks=False)
        if not r or len(r) == 0 or r[0].boxes is None or len(r[0].boxes) == 0:
            preds[ip.name] = []
            del img_np
            continue
        b = r[0].boxes.xyxy.cpu().numpy()
        s = r[0].boxes.conf.cpu().numpy()
        keep = s >= 0.25
        boxes = b[keep]
        scores = s[keep]
        per = []
        if len(boxes) > 0:
            masks, sc = sam3_box_predict(proc, mdl, img_np, boxes.tolist())
            # SAM3 instance-seg post-processor can return MORE masks than input
            # boxes (it splits one box into multiple instances), so len(masks) may
            # exceed len(boxes) and boxes[k] would overrun. The `box` field is
            # informational only (eval core uses rle+score, never box), but keep
            # it correct: derive each mask's box from the mask itself via
            # cv2.boundingRect (always available, always matches the mask). This
            # never drops masks and never indexes out of bounds.
            for k in range(len(masks)):
                rle = encode_mask(masks[k])
                mk = masks[k]
                if mk.any():
                    ys, xs = np.where(mk)
                    bx = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                else:
                    bx = [float(v) for v in boxes[k]] if k < len(boxes) else [0.0, 0.0, 0.0, 0.0]
                per.append({"rle": rle,
                            "score": float(sc[k]) if k < len(sc) else float(scores[k]) if k < len(scores) else 1.0,
                            "box": bx})
        preds[ip.name] = per
        del img_np
        if i % 10 == 0 or i == len(files) - 1:
            print(f"  rtdetr-uly+sam3 {i+1}/{len(files)} {ip.name} -> {len(per)} preds ({time.time()-t0:.1f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return preds


# ============================================================================
# Orchestration
# ============================================================================

def run_eval_for_ckpt(ckpt_tag, corpus_name, preds_per_img, tiers_manifest, out_dir):
    """Run per-tier eval on a ckpt's preds; write <ckpt>_reeval.json."""
    pred_list = build_pred_coco(preds_per_img, tiers_manifest)
    print(f"  [{ckpt_tag}/{corpus_name}] COCO preds: {len(pred_list)}", flush=True)
    tier_maps = {}
    for t in TIERS:
        gt_dict = build_gt_coco(tiers_manifest, t)
        n_active = sum(1 for a in gt_dict["annotations"] if a["iscrowd"] == 0)
        ap_t = eval_tier(gt_dict, pred_list)
        tier_maps[t] = ap_t
        print(f"  [{ckpt_tag}/{corpus_name}] {t}: {n_active} active GT -> {ap_t:.1f}%" if ap_t is not None else f"  {t}: None", flush=True)
    valid = [tier_maps[t] for t in TIERS if tier_maps[t] is not None]
    agg = float(np.mean(valid)) if len(valid) == 3 else None
    rec = quick_recall(tiers_manifest, preds_per_img)
    print(f"  [{ckpt_tag}/{corpus_name}] AGG: {agg:.1f}% recall: {rec:.4f}" if agg else "  AGG: None", flush=True)
    result = {
        "ckpt_tag": ckpt_tag,
        "corpus": corpus_name,
        "easy": tier_maps.get("easy"),
        "medium": tier_maps.get("medium"),
        "hard": tier_maps.get("hard"),
        "agg": agg,
        "recall": round(rec, 4),
        "paper_in_dist_mAP50": PAPER.get(ckpt_tag),
        "metric": "pycocotools COCOeval mask mAP@0.5 (single IoU 0.5, maxDet 100); per-tier via iscrowd ignore; aggregate=mean of 3 tiers",
    }
    out_reeval = out_dir / f"{ckpt_tag}_reeval.json"
    json.dump(result, open(out_reeval, "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(CKPT.keys()),
                    help="which ckpt to eval")
    ap.add_argument("--corpus", required=True, choices=list(CORPORA.keys()))
    ap.add_argument("--out-dir", required=True, help="base out dir, e.g. ~/bench_any/runs/p5_r1_v2")
    ap.add_argument("--skip-predict", action="store_true",
                    help="reuse existing <ckpt>_pred.json; only re-eval")
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    corpus_dir = out_base / args.corpus
    corpus_dir.mkdir(parents=True, exist_ok=True)

    tiers_data = json.load(open(CORPORA[args.corpus]["tiers"]))
    tiers_manifest = tiers_data["manifest"]
    img_dir = CORPORA[args.corpus]["images"]
    print(f"=== {args.ckpt} on {args.corpus} ({len(tiers_manifest)} imgs) ===", flush=True)

    pred_path = corpus_dir / f"{args.ckpt}_pred.json"
    if args.skip_predict and pred_path.exists():
        print(f"SKIP PREDICT; reusing {pred_path}", flush=True)
        preds = json.load(open(pred_path))
    else:
        ckpt_path = CKPT[args.ckpt]
        t0 = time.time()
        if args.ckpt == "yolo11x-seg" or args.ckpt == "yolo11x-seg-combined":
            preds = predict_yolo11x(ckpt_path, img_dir)
        elif args.ckpt in ("rf-detr-seg-xlarge", "rf-detr-seg-large",
                           "rf-detr-seg-large-combined"):
            preds = predict_rfdetr_seg(args.ckpt, ckpt_path, img_dir)
        elif args.ckpt in ("eomt-large", "eomt-large-combined"):
            preds = predict_eomt(ckpt_path, img_dir)
        elif args.ckpt in ("rtdetr-4", "rtdetr-3"):
            proc, mdl = get_sam3_box()
            preds = predict_rtdetr_sam3(ckpt_path, img_dir, proc, mdl)
            del proc, mdl
            import torch; torch.cuda.empty_cache()
        elif args.ckpt == "rtdetr-x-combined":
            # Ultralytics RTDETR ckpt (combined training) -> boxes -> SAM3 box-prompt.
            # Needs the external-runtime builder? No - this is an Ultralytics .pt,
            # loaded via ultralytics.RTDETR(ckpt). Uses the SAME sam3_box_predict.
            proc, mdl = get_sam3_box()
            preds = predict_rtdetr_ultralytics_sam3(ckpt_path, img_dir, proc, mdl)
            del proc, mdl
            import torch; torch.cuda.empty_cache()
        else:
            raise ValueError(args.ckpt)
        print(f"prediction done in {time.time()-t0:.1f}s; {sum(len(v) for v in preds.values())} total preds", flush=True)
        json.dump(preds, open(pred_path, "w"))
        print(f"preds -> {pred_path}", flush=True)

    # eval
    result = run_eval_for_ckpt(args.ckpt, args.corpus, preds, tiers_manifest, corpus_dir)
    print(f"reeval -> {corpus_dir / (args.ckpt + '_reeval.json')}", flush=True)


if __name__ == "__main__":
    main()
