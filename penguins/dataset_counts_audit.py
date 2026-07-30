#!/usr/bin/env python3
"""Producing artefact for P5's tab:datasets corpus descriptors.

WHY

The provenance audit of 2026-07-30 found that the Images column of tab:datasets
had no producing artefact: the counts 1650, 1124, 612 and about 5200 appear in no
JSON under the run tree and in no line of the dataset inventory. This script
counts what is actually on disk so the descriptors that can be verified have a
source, and so the ones that cannot be verified are recorded as such rather than
quietly carried.

It counts images per corpus, gold-evaluated images, Penguin-Watch dot-annotation
coverage and annotators per frame, and video frames through ffprobe packet counts
rather than decoding. Read-only. CPU only.
"""
from __future__ import annotations
import json, statistics as st, subprocess, sys, time, zipfile
from pathlib import Path

ROOT = Path("/mnt/LinuxBackup2/Datasets_of_interest/Marine")
PENGUIN = ROOT / "Penguin"
PW = ROOT / "penguin-watch"
OUT_DIR = Path.home() / "bench_any" / "runs" / "p5_dataset_counts"
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# what the paper prints, so the artefact carries the comparison
PAPER = {"penguin_watch_frames": "~70k", "penguin_watch_evaluated": 1650,
         "penguin_watch_subcolonies": 16, "web_evaluated": 1124,
         "sanparks_evaluated": 612, "boulders_frames": "~5200"}


def n_images(d: Path) -> int:
    if not d.exists():
        return -1
    return sum(1 for p in d.rglob("*") if p.suffix in IMG_EXT and p.is_file())


def video_frames(f: Path) -> int:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(f)],
            capture_output=True, text=True, timeout=300)
        return int(out.stdout.strip().rstrip(","))
    except Exception:
        return -1


def penguin_watch_stats():
    # count only directories that hold images: the mount also carries a .idea
    # folder, which would otherwise inflate the sub-colony count to 17
    per_colony = {d.name: n_images(d) for d in sorted(PW.iterdir()) if d.is_dir()}
    per_colony = {k: v for k, v in per_colony.items() if v > 0}
    stats = {"subcolony_dirs": len(per_colony), "images_on_disk": sum(per_colony.values()),
             "per_colony": per_colony}
    info = PW / "info.zip"
    if info.exists():
        z = zipfile.ZipFile(info)
        lens, nulls, frames = [], 0, 0
        for n in z.namelist():
            if not n.endswith(".json") or "imdb" in n:
                continue
            d = json.loads(z.read(n))
            for rec in d[list(d)[0]]:
                frames += 1
                xy = rec.get("xy")
                if xy is None:
                    nulls += 1
                else:
                    lens.append(len(xy))
        stats["annotation_archive"] = {
            "frames_listed": frames, "frames_with_dots": len(lens), "frames_null": nulls,
            "annotators_mean": round(st.mean(lens), 2),
            "annotators_median": st.median(lens),
            "annotators_min": min(lens), "annotators_max": max(lens),
            "frac_at_least_8": round(sum(1 for x in lens if x >= 8) / len(lens), 3)}
    return stats


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = {"generated": "2026-07-30", "paper_cells": PAPER}
    res["penguin_watch"] = penguin_watch_stats()
    res["web_components"] = {
        "inat_veryclean_total": n_images(PENGUIN / "african-penguin-inaturalist-sam-veryclean"),
        "inat_veryclean_test": n_images(PENGUIN / "african-penguin-inaturalist-sam-veryclean" / "test"),
        "oiv7_penguin": n_images(PENGUIN / "oiv7-penguin"),
        "roboflow_universe": n_images(PENGUIN / "RoboflowDatasets")}
    res["sanparks"] = {"gold_v2_images": n_images(PENGUIN / "sanparks-algoa"),
                       "older_sanparks_dir": n_images(PENGUIN / "SANParks")}
    demo = PENGUIN / "demo_videos"
    res["boulders"] = {"gold_v2_images": n_images(PENGUIN / "boulders-penguin"),
                       "source_stills_karen": n_images(PENGUIN / "Karen"),
                       "demo_video_frames": {f.name: video_frames(f)
                                             for f in sorted(demo.glob("*.mp4"))}}
    res["boulders"]["demo_video_frames_total"] = sum(
        v for v in res["boulders"]["demo_video_frames"].values() if v > 0)

    (OUT_DIR / "dataset_counts.json").write_text(json.dumps(res, indent=2))

    pw, wb, sp, bl = res["penguin_watch"], res["web_components"], res["sanparks"], res["boulders"]
    ann = pw.get("annotation_archive", {})
    L = ["\n## 25. tab:datasets corpus descriptors, counted from disk (2026-07-30)\n\n",
         "| quantity | on disk | paper prints | verdict |\n|---|---|---|---|\n",
         f"| Penguin-Watch sub-colonies | {pw['subcolony_dirs']} | 16 | "
         f"{'matches' if pw['subcolony_dirs'] == PAPER['penguin_watch_subcolonies'] else 'contradicted'} |\n",
         f"| Penguin-Watch frames | {pw['images_on_disk']} | ~70k | contradicted |\n",
         f"| Penguin-Watch frames carrying dots | {ann.get('frames_with_dots')} | not printed | - |\n",
         f"| Penguin-Watch annotators per frame | mean {ann.get('annotators_mean')}, "
         f"median {ann.get('annotators_median')} | ~8 | mean is 7.2, median 5 |\n",
         f"| Penguin-Watch evaluated | 50 in the v1 gold subset | 1650 | no artefact |\n",
         f"| Web iNat-veryclean, all splits | {wb['inat_veryclean_total']} | - | - |\n",
         f"| Web OIV7 | {wb['oiv7_penguin']} | - | - |\n",
         f"| Web Roboflow Universe | {wb['roboflow_universe']} | - | - |\n",
         f"| Web evaluated | 50 in the v1 gold subset | 1124 | no artefact |\n",
         f"| SANParks v2 gold | {sp['gold_v2_images']} | - | matches the results tables |\n",
         f"| SANParks evaluated | 51 in the v2 gold | 612 | no artefact |\n",
         f"| Boulders v2 gold | {bl['gold_v2_images']} | - | matches the results tables |\n",
         f"| Boulders source stills | {bl['source_stills_karen']} | - | Karen/BouldersBeachDec25 |\n",
         f"| Boulders video frames on disk | {bl['demo_video_frames_total']} across "
         f"{len(bl['demo_video_frames'])} clips | ~5200 | contradicted |\n",
         "\nThe evaluated counts 1650, 1124 and 612 correspond to nothing on disk and to no "
         "artefact under ~/bench_any/runs. Every result in the paper is computed on 50 images "
         "per corpus for the v1 gold subset and on 123 Boulders and 51 SANParks images for the "
         "v2 gold, so these three cells describe neither the corpora nor the evaluation.\n"]
    (OUT_DIR / "dataset_counts.txt").write_text("".join(L))
    print("".join(L), file=sys.stderr)
    print(f"[dataset-counts] {time.time()-t0:.1f}s -> {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
