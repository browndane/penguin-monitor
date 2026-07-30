#!/usr/bin/env python3
"""Route 2 of P5's tab:annotation_effort: SAM 2 prompted with hand-drawn boxes.

WHY

The Mask IoU column was recomputed on 2026-07-30 under a single stated protocol,
the mean over gold instances with an unmatched gold instance scored zero. Routes
1, 3 and 4 fall out of artefacts already on disk. Route 2 does not, because no
run on record prompts SAM 2 with the gold boxes; the sam2_masks.json beside each
gold set refines SAM 3 boxes, which is a different route. Leaving the printed
0.87 in place would put one cell of the column on an unknown protocol while the
other three sit on a stated one, so it is measured here.

The hand-drawn box is the gold box, so every gold instance receives exactly one
mask and the pairing needs no matching step. Writes the recomputed column and
patches the single cell in the paper. GPU, a few minutes.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pycocotools.mask as mu

RUNS = Path.home() / "bench_any" / "runs"
OBJ2 = RUNS / "obj2_counting"
OUT_DIR = RUNS / "p5_mask_iou_routes"
IMG_ROOT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine/Penguin")
SAM2_CKPT = "/home/vorse/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "sam2.1/sam2.1_hiera_l.yaml"
CORPORA = {"boulders-penguin": "Boulders", "sanparks-algoa": "SANParks"}

# routes 3 and 4 under the adopted protocol, from mask_iou_routes.json per_gold
ROUTE34 = {"sam3_raw": {"Boulders": 0.9015, "SANParks": 0.7600},
           "sam3_sieve": {"Boulders": 0.8997, "SANParks": 0.7577}}


def build_predictor():
    sys.path.insert(0, "/home/vorse/sam2")
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir="/home/vorse/sam2/sam2/configs", version_base="1.2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=device))


def route2(key, predictor):
    import cv2
    gt = json.loads((OBJ2 / f"sam2refine_{key}" / "gt_coco.json").read_text())
    fname = {im["id"]: im["file_name"] for im in gt["images"]}
    hw = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    by_image = {}
    for a in gt["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    img_dir = IMG_ROOT / key / "train" / "images"
    ious, missing = [], 0
    for iid, anns in by_image.items():
        p = img_dir / fname[iid]
        if not p.is_file():
            missing += len(anns)
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        h, w = hw[iid]
        boxes = np.array([[a["bbox"][0], a["bbox"][1],
                           a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=boxes, multimask_output=False)
        masks = np.asarray(masks).reshape(len(anns), h, w)
        for a, m in zip(anns, masks):
            r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            ious.append(float(np.asarray(mu.iou([r], [a["segmentation"]], [0])).ravel()[0]))
    return {"mean_iou": round(float(np.mean(ious)), 4), "n_gold": len(ious),
            "n_missing_images": missing}


def patch_paper(value):
    """Replace the one cell that was still on the old protocol."""
    tex = Path.home() / "Dropbox/myPapers/CurrentlyWriting/p5_penguin_monitor_elsevier" \
                        "/penguin_monitor_elsevier.tex"
    src = tex.read_text()
    cand = "SAM~2 box-prompt + manual box draw & 3.2 & 53.3 & 0.87 \\\\"
    if src.count(cand) == 1:
        tex.write_text(src.replace(cand, cand.replace("0.87", f"{value:.2f}")))
        return f"patched, 0.87 -> {value:.2f}"
    hits = [ln.strip() for ln in src.splitlines() if "0.87" in ln and "&" in ln]
    return f"NOT patched, cell string not matched uniquely; candidates {hits[:3]}"


def patch_figure(col):
    """fig1 plots the same column, so it moves with the table or it contradicts it."""
    fig = Path(__file__).with_name("p5_figures.py")
    src = fig.read_text()
    old = "    iou = [1.00, 0.87, 0.83, 0.83]\n"
    new = (f"    iou = [{col['manual_polygon']:.2f}, {col['sam2_from_manual_boxes']:.2f}, "
           f"{col['sam3_raw']:.2f}, {col['sam3_sieve']:.2f}]\n")
    if src.count(old) != 1:
        return "NOT patched, iou list not matched uniquely"
    fig.write_text(src.replace(old, new))
    sys.path.insert(0, str(fig.parent))
    import importlib
    mod = importlib.import_module("p5_figures")
    importlib.reload(mod)
    mod.fig_cost_quality()
    return f"patched and rebuilt, {new.strip()}"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor()
    res = {k: route2(k, predictor) for k in CORPORA}

    col = {"manual_polygon": 1.0,
           "sam2_from_manual_boxes": float(np.mean([res[k]["mean_iou"] for k in CORPORA])),
           "sam3_raw": float(np.mean(list(ROUTE34["sam3_raw"].values()))),
           "sam3_sieve": float(np.mean(list(ROUTE34["sam3_sieve"].values())))}
    verdict = patch_paper(col["sam2_from_manual_boxes"])
    fig_verdict = patch_figure(col)

    L = ["\n## 27. tab:annotation_effort Mask IoU column, recomputed under one protocol "
         "(2026-07-30)\n\n",
         "Mean over gold instances, a gold instance receiving no mask scored zero, averaged "
         "over the two gold corpora. Route 2 prompts SAM 2 with the gold boxes, so every gold "
         "instance receives a mask and no matching step is needed.\n\n",
         "| route | Boulders | SANParks | column | was |\n|---|---|---|---|---|\n",
         f"| manual polygon | 1.0000 | 1.0000 | {col['manual_polygon']:.2f} | 1.00 |\n",
         f"| SAM 2 from manual boxes | {res['boulders-penguin']['mean_iou']:.4f} | "
         f"{res['sanparks-algoa']['mean_iou']:.4f} | {col['sam2_from_manual_boxes']:.2f} | 0.87 |\n",
         f"| SAM 3 raw text-prompt | {ROUTE34['sam3_raw']['Boulders']:.4f} | "
         f"{ROUTE34['sam3_raw']['SANParks']:.4f} | {col['sam3_raw']:.2f} | 0.84 |\n",
         f"| SAM 3 + sieve + review | {ROUTE34['sam3_sieve']['Boulders']:.4f} | "
         f"{ROUTE34['sam3_sieve']['SANParks']:.4f} | {col['sam3_sieve']:.2f} | 0.91 |\n",
         f"\nGold instances scored: Boulders {res['boulders-penguin']['n_gold']}, "
         f"SANParks {res['sanparks-algoa']['n_gold']}. "
         f"Images not found: {res['boulders-penguin']['n_missing_images']} and "
         f"{res['sanparks-algoa']['n_missing_images']}.\n",
         f"\nPaper cell for route 2: {verdict}\n",
         f"\nFigure 1: {fig_verdict}\n",
         "\nThe sieve does not raise mask fidelity. It removes one to three percent of "
         "predictions, which is far too few to move a mean taken over gold instances, and what "
         "it removes is false positives rather than bad masks. The printed rise from 0.84 to "
         "0.91 was not a mask-quality effect.\n"]
    (OUT_DIR / "mask_iou_column_final.json").write_text(
        json.dumps({"per_corpus": res, "column": col, "paper_patch": verdict,
                    "figure_patch": fig_verdict}, indent=2))
    (OUT_DIR / "mask_iou_column_final.txt").write_text("".join(L))
    print("".join(L), file=sys.stderr)
    print(f"[route2] {time.time()-t0:.1f}s -> {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
