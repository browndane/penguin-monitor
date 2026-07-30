#!/usr/bin/env python3
"""Phase 2a soft-logit SAM3 run: save mask PROBABILITIES (sigmoid) for fusion.

WHY
---
The fourth fusion strategy (soft-average) needs per-pixel mask probabilities,
not binary RLEs. prep_11 saves binary masks (sigmoid + threshold). This variant
saves the sigmoid probabilities so the CPU sweep can average them per agreed pixel
and re-threshold. Targets the strict mAP@0.5:0.95 boundary metric.

WHAT
---
For each prompt in {a_penguin, penguin, african_penguin}, each image:
  - run SAM3 text-prompt
  - take out["pred_masks"] (1, N, 288, 288) raw logits
  - sigmoid -> probabilities
  - post_process to get per-instance boxes + scores (same as prep_11)
  - upsample each instance's prob map to original HxW
  - save as {filename: [{score, box, prob_rle}]} where prob_rle is a FLOAT
    probability map encoded as a sidecar .npy (RLE doesn't fit floats).

Self-reports to ~/bench_any/runs/obj2_counting/FINDINGS.md §20a + sentinel.
GPU run, detached.
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HOME", "/mnt/LinuxBackup/sam3")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/LinuxBackup/sam3/hub")

sys.path.insert(0, "/home/vorse/bench_any/datasets/prep")
import prep_11_filebased as P11  # reuse get_sam3, Sam3Processor, constants

RUNS = Path.home() / "bench_any" / "runs"
R1 = RUNS / "p5_r1_v2"
FINDINGS = RUNS / "obj2_counting" / "FINDINGS.md"
DONE = RUNS / "obj2_counting" / "_soft_logit_v2_done"
LOG = RUNS / "obj2_counting" / "soft_logit_v2_results.txt"

CORPORA = {"boulders": ("boulders-penguin/train/images",
                        "boulders/boulders_tiers.json", "boulders"),
           "sanparks": ("sanparks-algoa/train/images",
                        "sanparks/sanparks_tiers.json", "sanparks")}
DS_ROOT = "/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin"
PROMPTS = ["a_penguin", "penguin", "african_penguin"]
# The keys above are FILE TAGS, not prompts. prep_11_filebased tags a raw file by
# args.prompt.replace(" ", "_"), so the tag "a_penguin" comes from the text
# "a penguin". The first version of this script fed the tag straight to SAM3 and
# prompted the model with "a_penguin", which is a different string and produced a
# different detection set (912 raw vs 895 soft on Boulders). Map back to the text.
PROMPT_TEXT = {"a_penguin": "a penguin",
               "penguin": "penguin",
               "african_penguin": "african penguin"}


def _raw_path(sub, prompt):
    tag = "" if prompt == "a_penguin" else "_" + prompt
    return R1 / sub / f"{sub}_sam3_soft{tag}_v2.json"


def soft_predict(proc, mdl, img_np, score, mask_thr, prompt):
    """Return (masks_bool, scores, boxes, prob_maps) where prob_maps is
    [N, H, W] float16 sigmoid probabilities upsampled to original size."""
    import torch, cv2
    from PIL import Image
    H0, W0 = img_np.shape[:2]
    long_edge = max(W0, H0)
    if long_edge > P11.LONG_EDGE_CAP:
        scale = P11.LONG_EDGE_CAP / long_edge
        Wds, Hds = int(W0 * scale), int(H0 * scale)
        img_ds = cv2.resize(img_np, (Wds, Hds))
    else:
        img_ds = img_np; Wds, Hds = W0, H0
    im = Image.fromarray(img_ds[:, :, ::-1]).convert("RGB")
    with torch.inference_mode():
        inputs = proc(images=im, text=prompt, return_tensors="pt").to("cuda")
        out = mdl(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=score, mask_threshold=mask_thr, target_sizes=[(Hds, Wds)])[0]
    masks_ds = res["masks"].cpu().numpy().astype(bool) if "masks" in res else np.zeros((0, Hds, Wds), bool)
    sc = res["scores"].float().cpu().numpy() if "scores" in res else np.zeros(0)
    boxes = res["boxes"].float().cpu().numpy() if "boxes" in res else np.zeros((0, 4))
    # raw logits -> sigmoid probs, upsample
    # Select the SAME query slots post_process kept. It filters with
    # `scores > threshold` over the query axis and preserves query order, so the
    # kept set is a scattered subset, not a prefix. Slicing the first n_keep
    # queries files every probability map under the wrong detection: on Boulders
    # that gave mean IoU 0.009 between a map and its own binary mask, which is
    # what collapsed soft-average fusion to 1.3 tier-mean mAP.
    scores_all = out["pred_logits"].sigmoid()
    presence = out.get("presence_logits")
    if presence is not None:
        scores_all = scores_all * presence.sigmoid()
    keep = scores_all[0] > score
    raw_masks = out["pred_masks"][0][keep].float()  # (n_keep, 288, 288) logits
    probs = torch.sigmoid(raw_masks).cpu().numpy().astype(np.float16)
    n_keep = len(masks_ds)
    if len(probs) != n_keep:
        raise RuntimeError(f"query select gave {len(probs)} maps for {n_keep} detections")
    # NOTE: store at native 288x288 (not upsampled to full res). A full-res float16
    # mask is ~35 MB and 2700 of them blow the disk. Native is ~170 KB, a ~250x cut.
    # The fusion sweep upsamples only the agreed-pair masks at eval time (cheap,
    # few pairs per image). Boxes still rescaled to full-res coords for matching.
    # rescale boxes
    if len(boxes) and (Wds != W0 or Hds != H0):
        sx, sy = W0 / Wds, H0 / Hds
        boxes = boxes * np.array([sx, sy, sx, sy])
    return masks_ds, sc, boxes, probs  # probs stay native-res; fusion upsamples


def main():
    import torch, cv2
    t0 = time.time()
    proc, mdl = P11.get_sam3()
    lines = ["\n## 20c. Soft-logit SAM3 re-run, both defects fixed (2026-07-29)\n\n"
             "Supersedes 20a. Two fixes: SAM3 is now prompted with the text "
             "('a penguin') rather than the file tag ('a_penguin'), and boxes are "
             "stored as the XYXY that post_process returns rather than being treated "
             "as XYWH. The alignment column below is the per-image detection-count "
             "match against the corresponding raw set, which is what fusion needs.\n\n"]
    summary = []
    for corp, (img_sub, tiers_sub, out_sub) in CORPORA.items():
        img_dir = Path(DS_ROOT) / img_sub
        tiers = json.loads((R1 / tiers_sub).read_text())
        man = tiers["manifest"]
        files = sorted(man.keys())
        for prompt in PROMPTS:
            out_path = _raw_path(out_sub, prompt)
            npy_dir = R1 / out_sub / f"{out_sub}_soft{('_' + prompt) if prompt != 'a_penguin' else ''}_v2_npys"
            npy_dir.mkdir(parents=True, exist_ok=True)
            results = {}
            for i, fn in enumerate(files):
                img_path = img_dir / fn
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                masks, sc, boxes, probs = soft_predict(proc, mdl, img, P11.SCORE, P11.MASK_THR,
                                                       PROMPT_TEXT[prompt])
                insts = []
                for k in range(len(sc)):
                    npy_name = f"{Path(fn).stem}_{k}.npy"
                    np.save(npy_dir / npy_name, probs[k])
                    bb = boxes[k]
                    # SAM3 post_process already returns XYXY. The first version
                    # added bb[0]+bb[2] as if it were XYWH, overshooting the image.
                    insts.append({"score": float(sc[k]),
                                  "box": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                  "prob_npy": npy_name})
                results[fn] = insts
                if i % 10 == 0:
                    print(f"[{corp}/{prompt}] {i}/{len(files)} ({len(sc)} masks)", file=sys.stderr, flush=True)
            out_path.write_text(json.dumps(results))
            n_anns = sum(len(v) for v in results.values())
            # alignment self-check against the raw set fusion will pair this with
            rtag = "" if prompt == "a_penguin" else "_" + prompt
            rpath = R1 / out_sub / f"{out_sub}_sam3_raw{rtag}.json"
            if rpath.exists():
                rawd = json.loads(rpath.read_text())
                n_raw = sum(len(v) for v in rawd.values())
                agree = sum(1 for f in results if f in rawd and len(rawd[f]) == len(results[f]))
                align = f"{agree}/{len(results)} imgs, {n_raw} raw anns"
            else:
                n_raw, align = -1, "raw missing"
            summary.append((corp, prompt, n_anns, align, str(out_path)))
            lines.append(f"{corp}/{prompt}: {n_anns} soft masks, align {align} -> {out_path.name}\n")
            print(f"[{corp}/{prompt}] DONE {n_anns} anns -> {out_path}", file=sys.stderr, flush=True)
    table = "\n".join(f"| {c} | {p} | {n} | {al} | {Path(f).name} |" for c, p, n, al, f in summary)
    lines.append("\n| corpus | prompt | n_anns | alignment vs raw | file |\n"
                 "|---|---|---|---|---|\n" + table + "\n")
    LOG.write_text("".join(lines))
    with open(FINDINGS, "a") as fh:
        fh.write("".join(lines))
    DONE.touch()
    print(f"[soft-logit] done {time.time()-t0:.1f}s sentinel={DONE}", file=sys.stderr)


if __name__ == "__main__":
    main()
