# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Load Roboflow v2 user-verified masks into a FiftyOne dataset keyed by ORIGINAL filename.

User fixed FPs + added stragglers in the Roboflow GUI on projects `sanparks-algoa`
and `boulders-penguin`, then generated version 2 (auto-orient preprocessing; auto-
orient was a no-op here since all images are landscape, verified: dims unchanged).

Output dataset: p5_gold200_verified  (FiftyOne, ~174 samples: 51 SANParks + 123 Boulders)
Each sample's `ground_truth` = fo.Detections with mask (filled from polygon segmentation),
tagged by corpus so the 200-assembler can stratify.

Keyed by ORIGINAL filename:
  Boulders: image `extra.name` == manifest key == candidate filepath stem (identity)
  SANParks : image `extra.name` is Roboflow-normalized (hyphens); normalize the manifest
            key the same way to match. `1_01120002 (21).jpg` -> `1_01120002-21-.jpg`.
"""
import json, os, sys
from pathlib import Path

import fiftyone as fo

VENV = os.path.expanduser("~/cvML/bin/python")
DOWNLOAD = Path("/home/user/bench_any/datasets/p5_gold200_roboflow_download")
MANIFEST = Path("/home/user/bench_any/datasets/p5_gold200_roboflow/upload_manifest.json")
DS_NAME = "p5_gold200_verified"


def norm_sanparks_key(k):
    """'1_01120002 (21).jpg' -> '1_01120002-21-.jpg'  (Roboflow's extra.name form)."""
    stem, ext = k.rsplit(".", 1)
    s = stem.replace(" (", "-").replace(") ", "-").replace(")", "-").replace(" ", "-")
    return s + "." + ext


def coco_seg_to_detection(ann, img_w, img_h):
    """COCO polygon segmentation -> fo.Detection with mask."""
    import numpy as np

    seg = ann["segmentation"]
    # build binary mask at image resolution
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if isinstance(seg, list):
        from PIL import Image  # noqa: F401
        import cv2

        for poly in seg:
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
            pts = pts.round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    elif isinstance(seg, dict) and "counts" in seg:
        from pycocotools import mask as maskUtils

        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[:, :, 0]
        mask = m.astype(np.uint8)
    # bbox [x,y,w,h] -> fiftyone [xmin,ymin,w,h] in 0-1
    x, y, w, h = ann["bbox"]
    return fo.Detection(
        label="penguin",
        bounding_box=[x / img_w, y / img_h, w / img_w, h / img_h],
        mask=mask,
    )


def load_project(proj_dir, corpus, orig_name_index):
    """proj_dir = DOWNLOAD/<corpus>/train. Returns list of (original_filename, detections)."""
    coco = json.load(open(proj_dir / "_annotations.coco.json"))
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    out = []
    for im in coco["images"]:
        extra = im.get("extra", {})
        extra_name = extra.get("name") if isinstance(extra, dict) else None
        rf_name = im["file_name"]
        # resolve original filename
        if corpus == "boulders":
            orig = extra_name or rf_name  # extra.name == manifest key
        else:  # sanparks: extra.name normalized; reverse-map via index
            orig = orig_name_index.get(extra_name) or orig_name_index.get(rf_name)
        if orig is None:
            print(f"  WARN: {corpus} {rf_name} (extra.name={extra_name}) unkeyed, skip")
            continue
        w, h = im["width"], im["height"]
        dets = fo.Detections(
            detections=[coco_seg_to_detection(a, w, h) for a in anns_by_img.get(im["id"], [])]
        )
        out.append((orig, str(proj_dir / rf_name), dets))
    return out


def main():
    # build reverse index: roboflow normalized name -> original manifest key (SANParks)
    manifest = json.load(open(MANIFEST))
    sanparks_index = {}
    for k, v in manifest.items():
        if v.get("project") == "sanparks":
            sanparks_index[norm_sanparks_key(k)] = k
            sanparks_index[k] = k  # identity too

    all_samples = []
    for corpus, proj in [("sanparks", "sanparks"), ("boulders", "boulders")]:
        proj_dir = DOWNLOAD / corpus / "train"
        idx = sanparks_index if corpus == "sanparks" else None
        rows = load_project(proj_dir, corpus, idx)
        print(f"{corpus}: {len(rows)} verified samples loaded")
        for orig, img_path, dets in rows:
            s = fo.Sample(filepath=img_path, tags=[corpus])
            s["ground_truth"] = dets
            # stash original filename as a field so the assembler can match to candidate filepath
            s["orig_name"] = orig
            all_samples.append(s)

    # write dataset (persistent=True: fiftyone 1.19 deletes non-persistent
    # datasets when the creating process exits -- this was the vanishing bug)
    if DS_NAME in fo.list_datasets():
        fo.delete_dataset(DS_NAME)
    ds = fo.Dataset(DS_NAME, persistent=True)
    ds.add_samples(all_samples)
    ds.save()
    print(f"\n{DS_NAME}: {len(ds)} samples")
    from collections import Counter
    print("corpus tags:", dict(Counter([s.tags[0] for s in ds.iter_samples()])))
    print("total GT detections:", sum(len(s.ground_truth.detections) for s in ds.iter_samples()))


if __name__ == "__main__":
    main()
