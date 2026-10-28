# Written by the authors and refactored with the assistance of Claude (Anthropic, Claude Code).
"""Export p5_gold200_candidate (0.25 full-recall) as per-image COCO JSON for Roboflow upload.

Writes to ~/bench_any/datasets/p5_gold200_roboflow/:
  - images/<basename>          (symlinked copies of original images)
  - <basename>.json            (per-image COCO: 1 image + its penguin polygon annotations)

Roboflow single_upload(image_path, annotation_path) accepts a per-image COCO JSON whose
`images` array has exactly one entry matching the uploaded image. Polygons come from the
FiftyOne polylines (filled, closed) converted to COCO segmentation [[x,y,...]] format.
"""
import os, json, shutil, fiftyone as fo
from pathlib import Path

OUT = Path("/home/user/bench_any/datasets/p5_gold200_roboflow")
(OUT / "images").mkdir(parents=True, exist_ok=True)

ds = fo.load_dataset("p5_gold200_candidate")
n_img = 0
n_ann = 0
for s in ds:
    gt = s["ground_truth"]
    if gt is None or not gt.polylines:
        continue  # skip zero-mask frames (nothing to upload for those)
    # image: symlink original
    dst_img = OUT / "images" / os.path.basename(s.filepath)
    if not dst_img.exists():
        os.symlink(os.path.realpath(s.filepath), dst_img)
    # COCO per-image json
    w = s.metadata.width if s.metadata else None
    h = s.metadata.height if s.metadata else None
    if not (w and h):
        ds.compute_metadata()
        w, h = s.metadata.width, s.metadata.height
    anns = []
    aid = 1
    for poly in gt.polylines:
        # poly.points is list of rings; each ring is list of [x_norm, y_norm]
        seg = []
        for ring in poly.points:
            for x, y in ring:
                seg += [round(x * w, 2), round(y * h, 2)]
        if len(seg) < 6:
            continue
        # bbox from polygon extents
        xs = seg[0::2]; ys = seg[1::2]
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by
        anns.append({
            "id": aid, "image_id": 1, "category_id": 1,
            "segmentation": [seg], "area": round(bw * bh, 2),
            "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
            "iscrowd": 0,
        })
        aid += 1
    coco = {
        "images": [{"id": 1, "file_name": os.path.basename(s.filepath), "width": int(w), "height": int(h)}],
        "annotations": anns,
        "categories": [{"id": 1, "name": "penguin", "supercategory": "bird"}],
    }
    (OUT / "images" / (os.path.basename(s.filepath) + ".json")).write_text(json.dumps(coco))
    n_img += 1
    n_ann += len(anns)
print(f"exported {n_img} images + per-image COCO json, {n_ann} annotations to {OUT}/images/")
