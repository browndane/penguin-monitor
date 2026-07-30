"""Assemble the 200-image 4-corpus gold subset for P5 tab:stratified_performance.

Composition (paper L166: "200 images uniformly sampled across the four corpora", ~50 each):
  - 50 Penguin-Watch (Gentoo)         <- reuse the seed-42 50 PW from p5_r1_gold/r1_test
  - 50 Web-scraped Spheniscus (iNat)  <- reuse the seed-42 50 iNat from p5_r1_gold/r1_test
  - 50 SANParks Adversarial-AP        <- sample 50 of 51 from p5_gold200_verified (seed 42)
  - 50 Boulders Beach                 <- sample 50 of 123 from p5_gold200_verified (seed 42)

All GT becomes fo.Detections with `mask` (mask mAP@0.5 is the paper metric).
PW/iNat GT in r1_test is fo.Polylines (filled) -> .to_detections() yields Detections w/ mask.
SANParks/Boulders GT in p5_gold200_verified is already fo.Detections with mask (Roboflow v2 user-verified).

NOTE: this is a REPLICATION subset, not a reproduction of the paper's original 200
(manifest not retained, 6th limitation L373). Same corpora + uniform composition +
seed 42; per-corpus counts approximate the paper's "uniformly sampled".

Output dataset: p5_gold200 (persistent=True). Writes manifest to
~/bench_any/runs/p5_r1_gold/gold200_manifest.json.
"""
import json, os, random
from pathlib import Path

import fiftyone as fo

SEED = 42
OUT_DS = "p5_gold200"
MAN_DIR = Path(os.path.expanduser("~/bench_any/runs/p5_r1_gold"))
MAN_DIR.mkdir(parents=True, exist_ok=True)


def polylines_to_mask_detections(pl):
    """fo.Polylines (filled, closed) -> fo.Detections with mask. Each filled closed
    polyline becomes one detection whose mask is the rasterized polygon."""
    import numpy as np
    import cv2

    if pl is None or not pl.polylines:
        return fo.Detections(detections=[])
    dets = []
    for poly in pl.polylines:
        # need image dims; get from the polyline's parent sample metadata outside
        dets.append(poly)  # placeholder; handled below with dims
    return dets


def rasterize_polyline(poly, img_w, img_h):
    """One fo.Polyline (filled, closed) -> (bbox_rel, mask)."""
    import numpy as np
    import cv2

    pts = poly.points[0]  # list of [x,y] in RELATIVE coords
    abs_pts = np.array([[x * img_w, y * img_h] for x, y in pts], dtype=np.float64)
    abs_pts = abs_pts.round().astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [abs_pts], 1)
    # bbox from mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    return ([x0 / img_w, y0 / img_h, bw / img_w, bh / img_h], mask)


def main():
    rng = random.Random(SEED)

    # --- source datasets ---
    r1 = fo.load_dataset("r1_test")  # 100 samples: 50 PW + 50 iNat, GT=Polylines
    ver = fo.load_dataset("p5_gold200_verified")  # 174: 51 SANParks + 123 Boulders

    samples = []

    # --- PW + iNat from r1_test ---
    pw_n = inat_n = 0
    for s in r1.iter_samples():
        fp = s.filepath
        fn = os.path.basename(fp)
        if fn.startswith("PW_"):
            corpus = "pw"
            pw_n += 1
        else:
            corpus = "inat"
            inat_n += 1
        # rasterize Polylines -> Detections with mask
        meta = s.metadata
        if meta is None or meta.width is None:
            meta = fo.ImageMetadata.build_for(fp)  # reads dims from file
        w, h = meta.width, meta.height
        dets_list = []
        if s.ground_truth is not None:
            for poly in s.ground_truth.polylines:
                r = rasterize_polyline(poly, w, h)
                if r is None:
                    continue
                bbox, mask = r
                dets_list.append(fo.Detection(label="penguin", bounding_box=bbox, mask=mask))
        ns = fo.Sample(filepath=fp, tags=[corpus, "gold200"],
                       ground_truth=fo.Detections(detections=dets_list))
        samples.append(ns)
    print(f"from r1_test: PW={pw_n} iNat={inat_n}")

    # --- SANParks + Boulders from verified ---
    san = [s for s in ver.iter_samples() if "sanparks" in s.tags]
    bld = [s for s in ver.iter_samples() if "boulders" in s.tags]
    print(f"verified pool: SANParks={len(san)} Boulders={len(bld)}")
    rng.shuffle(san)
    rng.shuffle(bld)
    san_pick = san[:50]
    bld_pick = bld[:50]

    for s, corpus in [(s, "sanparks") for s in san_pick] + [(s, "boulders") for s in bld_pick]:
        # verified GT already fo.Detections with mask; reuse filepath (downloaded .rf. image)
        ns = fo.Sample(filepath=s.filepath, tags=[corpus, "gold200"],
                       ground_truth=s.ground_truth)
        # carry orig_name if present
        try:
            ns["orig_name"] = s["orig_name"]
        except Exception:
            pass
        samples.append(ns)
    print(f"verified picked: SANParks={len(san_pick)} Boulders={len(bld_pick)}")

    # --- write dataset ---
    if OUT_DS in fo.list_datasets():
        fo.delete_dataset(OUT_DS)
    ds = fo.Dataset(OUT_DS, persistent=True)
    ds.add_samples(samples)
    ds.save()

    from collections import Counter
    corp = Counter([s.tags[0] for s in ds.iter_samples()])
    total_dets = sum(len(s.ground_truth.detections) for s in ds.iter_samples())
    print(f"\n{OUT_DS}: {len(ds)} samples | corpora={dict(corp)} | total GT dets={total_dets}")

    # --- manifest ---
    manifest = {
        "seed": SEED,
        "n_total": len(ds),
        "composition": {"pw": pw_n, "inat": inat_n, "sanparks": len(san_pick), "boulders": len(bld_pick)},
        "corpus_counts": dict(corp),
        "total_gt_detections": total_dets,
        "replication_note": "Reconstructed 200/4-corpus gold subset (paper original 200 manifest not retained, 6th limitation L373). PW+iNat reuse seed-42 selection from p5_r1_gold manifest; SANParks+Boulders sampled seed-42 from Roboflow v2 user-verified masks (p5_gold200_verified).",
        "gt_sources": {
            "pw": "r1_test Polylines -> rasterized to mask Detections",
            "inat": "r1_test Polylines -> rasterized to mask Detections",
            "sanparks": "Roboflow sanparks-algoa v2 (user-verified masks)",
            "boulders": "Roboflow boulders-penguin v2 (user-verified masks)",
        },
        "eval_metric": "FiftyOne mask mAP@0.5; aggregate = mean of 3 tier mAPs (unweighted)",
    }
    mf = MAN_DIR / "gold200_manifest.json"
    json.dump(manifest, open(mf, "w"), indent=2)
    print(f"manifest -> {mf}")


if __name__ == "__main__":
    main()
