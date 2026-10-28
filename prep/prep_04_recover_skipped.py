# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Recover SAM3-skipped Boulders frames at downsampled long edge, write masks back to FiftyOne.

Run under ~/cvML (fiftyone 1.19.0, transformers 5.13.0, Sam3Model). The DB lives at 1.19.
Downsamples 5184x3456 frames to LONG_EDGE to avoid GPU OOM; SAM3 target_sizes rescales masks
back to the original image size so polylines land in image coordinates.
"""
import os, sys, torch, numpy as np
from pathlib import Path
os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/LinuxBackup/sam3/hub")
from transformers import Sam3Model, Sam3Processor
import fiftyone as fo

SKIPPED = ["IMG_1107.jpg", "IMG_1069.jpg", "IMG_1190.jpg", "IMG_1070.jpg"]
BOU = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin/Partner/BouldersBeachDec25/train/images")
LONG_EDGE = 2448
SCORE, MASK_THR = 0.25, 0.3

def mask_to_polyline(mask_np, label="penguin"):
    """Binary mask -> fiftyone Polyline (filled)."""
    import cv2
    mask_uint8 = (mask_np.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polys = []
    for c in contours:
        if len(c) < 3: continue
        pts = [[float(p[0][0]) / mask_np.shape[1], float(p[0][1]) / mask_np.shape[0]] for p in c]
        polys.append(pts)
    return polys

def main():
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    mdl = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda").eval()
    ds = fo.load_dataset("p5_gold200_candidate")
    added = 0
    for bn in SKIPPED:
        path = BOU / bn
        if not path.exists():
            print(f"[MISS] {bn}"); continue
        from PIL import Image
        im = Image.open(path).convert("RGB")
        W0, H0 = im.size
        scale = LONG_EDGE / max(W0, H0)
        if scale < 1.0:
            im_ds = im.resize((int(W0*scale), int(H0*scale)))
        else:
            im_ds = im
        Wds, Hds = im_ds.size
        with torch.inference_mode():
            inputs = proc(images=im_ds, text="a penguin", return_tensors="pt").to("cuda")
            out = mdl(**inputs)
            # post-process at DOWNSAMPLED size (small mask tensor, avoids OOM)
            res = proc.post_process_instance_segmentation(
                out, threshold=SCORE, mask_threshold=MASK_THR,
                target_sizes=[(Hds, Wds)])[0]
        masks_ds = res["masks"].cpu().numpy().astype(bool)  # [N, Hds, Wds] at ds size
        import cv2
        # rescale each binary mask back to ORIGINAL size (cheap uint8 resize)
        masks = np.stack([cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
                          for m in masks_ds]) if len(masks_ds) else np.zeros((0, H0, W0), bool)
        scores = res["scores"].float().cpu().numpy()
        # find the sample by filepath basename
        sample = None
        for s in ds:
            if os.path.basename(s.filepath) == bn:
                sample = s; break
        if sample is None:
            print(f"[NO-SAMPLE] {bn}"); continue
        polys = []
        for i, m in enumerate(masks):
            if m.sum() == 0: continue
            # sanity sieve: min area 64px, drop full-frame
            if m.sum() < 64: continue
            p = mask_to_polyline(m)
            if p: polys.append(fo.Polyline(points=p, label="penguin", filled=True, closed=True))
        sample["ground_truth"] = fo.Polylines(polylines=polys) if polys else None
        sample.save()
        print(f"[OK] {bn}: {len(polys)} masks (raw {len(masks)})")
        added += len(polys)
    print(f"recovery added {added} masks; dataset GT polylines now: {ds.count('ground_truth.polylines')}")

if __name__ == "__main__":
    main()
